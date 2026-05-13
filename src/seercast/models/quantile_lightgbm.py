"""LightGBM quantile-forecast wrapper (one model per quantile).

Why a model per quantile?
* LightGBM's ``objective="quantile"`` only optimizes one quantile at a
  time. The standard approach is to fit independent models for each
  desired quantile (here p10, p50, p90).
* Independence trades a little statistical efficiency for simplicity and
  parallelism. Each model is just an ordinary LGBMRegressor with a
  different ``alpha``.

A consequence: independently-trained quantiles can produce ``p10 > p50``
on individual rows ("quantile crossing"). The version-1 fix is to sort
each prediction row in ascending order
(:func:`seercast.evaluation.fix_quantile_crossings`). The
:meth:`QuantileLightGBMModel.predict` method applies this fix by default.

This wrapper:

* Accepts the supervised table directly (output of
  :func:`seercast.features.build_supervised_table`).
* Learns one categorical-dtype map shared across all quantile models, so
  the three boosters see identical feature spaces.
* Persists cleanly via :mod:`joblib`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from seercast.evaluation.probabilistic_metrics import fix_quantile_crossings
from seercast.models.lightgbm_model import (
    DEFAULT_CATEGORICAL_COLUMNS,
    LightGBMPointModel,
    NON_FEATURE_COLUMNS,
    default_feature_columns,
)


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #


DEFAULT_QUANTILE_LGB_PARAMS: dict = {
    "objective": "quantile",
    "metric": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "verbose": -1,
    "force_col_wise": True,
}


def quantile_column_name(q: float) -> str:
    """Map ``0.1 -> "p10"``, ``0.5 -> "p50"``, ``0.9 -> "p90"``, etc."""
    return f"p{int(round(q * 100)):02d}"


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


@dataclass
class QuantileLightGBMModel:
    """Train one LightGBM regressor per quantile and predict all together."""

    name: str = "lightgbm_quantile"
    quantiles: tuple[float, ...] = (0.10, 0.50, 0.90)
    params: dict = field(default_factory=lambda: dict(DEFAULT_QUANTILE_LGB_PARAMS))
    n_estimators: int = 2000
    early_stopping_rounds: int = 50
    log_evaluation_period: int = 0  # 0 = silent

    # Set during fit():
    feature_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    _cat_dtypes: dict[str, pd.CategoricalDtype] = field(default_factory=dict)
    _boosters: dict[float, lgb.Booster] = field(default_factory=dict)
    _best_iterations: dict[float, int] = field(default_factory=dict)

    # ------- helpers --------------------------------------------------------
    @staticmethod
    def _coerce(df: pd.DataFrame, cat_dtypes: dict[str, pd.CategoricalDtype]) -> pd.DataFrame:
        out = df.copy()
        for c, dtype in cat_dtypes.items():
            if c in out.columns:
                out[c] = out[c].astype(dtype)
        return out

    # ------- public API -----------------------------------------------------
    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame | None = None,
        *,
        target_col: str = "target_sales",
        feature_columns: Sequence[str] | None = None,
        categorical_columns: Sequence[str] | None = None,
    ) -> "QuantileLightGBMModel":
        """Fit one booster per quantile.

        Parameters mirror :class:`LightGBMPointModel.fit`. The only
        difference is the loop over ``self.quantiles`` and the per-iteration
        ``alpha`` override.
        """
        if feature_columns is None:
            feature_columns = default_feature_columns()
        feature_columns = [c for c in feature_columns if c in train_df.columns]
        if categorical_columns is None:
            categorical_columns = list(DEFAULT_CATEGORICAL_COLUMNS)
        categorical_columns = [c for c in categorical_columns if c in feature_columns]

        self.feature_columns = list(feature_columns)
        self.categorical_columns = list(categorical_columns)

        # One categorical dtype map shared across all three quantile fits,
        # so OOV behavior at predict time is identical regardless of quantile.
        self._cat_dtypes = LightGBMPointModel._assemble_dtype_map(
            train_df, valid_df, categorical_columns
        )
        train = self._coerce(train_df, self._cat_dtypes)
        valid = self._coerce(valid_df, self._cat_dtypes) if valid_df is not None else None

        X_train = train[feature_columns]
        y_train = train[target_col].astype("float32")
        X_valid = valid[feature_columns] if valid is not None and len(valid) > 0 else None
        y_valid = valid[target_col].astype("float32") if X_valid is not None else None

        self._boosters = {}
        self._best_iterations = {}

        for q in self.quantiles:
            params = dict(self.params)
            params["alpha"] = float(q)

            train_set = lgb.Dataset(
                X_train, label=y_train,
                categorical_feature=categorical_columns,
                free_raw_data=False,
            )
            if X_valid is not None:
                valid_set = lgb.Dataset(
                    X_valid, label=y_valid,
                    categorical_feature=categorical_columns,
                    reference=train_set,
                    free_raw_data=False,
                )
                booster = lgb.train(
                    params,
                    train_set,
                    num_boost_round=self.n_estimators,
                    valid_sets=[valid_set],
                    valid_names=[f"valid_{quantile_column_name(q)}"],
                    callbacks=[
                        lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(self.log_evaluation_period),
                    ],
                )
                self._best_iterations[q] = booster.best_iteration
            else:
                booster = lgb.train(
                    params,
                    train_set,
                    num_boost_round=self.n_estimators,
                    callbacks=[lgb.log_evaluation(self.log_evaluation_period)],
                )
                self._best_iterations[q] = self.n_estimators

            self._boosters[q] = booster

        return self

    def predict(
        self,
        df: pd.DataFrame,
        *,
        fix_crossings: bool = True,
    ) -> pd.DataFrame:
        """Predict all quantiles, returning a DataFrame with one column per quantile.

        Columns are named ``p10`` / ``p50`` / ``p90`` (etc.). Index matches
        ``df.index``. With ``fix_crossings=True`` (default), values are
        sorted per row in ascending order to enforce monotonicity.
        """
        if not self._boosters:
            raise RuntimeError("QuantileLightGBMModel.predict called before fit")
        coerced = self._coerce(df, self._cat_dtypes)
        X = coerced[self.feature_columns]

        out = pd.DataFrame(index=df.index)
        for q in self.quantiles:
            preds = self._boosters[q].predict(X, num_iteration=self._best_iterations[q])
            out[quantile_column_name(q)] = np.clip(np.asarray(preds, dtype=float), 0.0, None)

        if fix_crossings:
            out = fix_quantile_crossings(out, quantile_columns=list(out.columns))
        return out

    def feature_importance(self, *, quantile: float = 0.50, kind: str = "gain") -> pd.DataFrame:
        """Feature importance from the booster for a given quantile."""
        if quantile not in self._boosters:
            raise KeyError(f"no booster for quantile {quantile}; have {sorted(self._boosters)}")
        booster = self._boosters[quantile]
        importance = booster.feature_importance(importance_type=kind)
        return (
            pd.DataFrame({"feature": self.feature_columns, kind: importance.astype(float)})
            .sort_values(kind, ascending=False)
            .reset_index(drop=True)
        )


__all__ = [
    "DEFAULT_QUANTILE_LGB_PARAMS",
    "QuantileLightGBMModel",
    "quantile_column_name",
]
