"""LightGBM point-forecast wrapper.

Why LightGBM with the Poisson objective?
* Sales are non-negative counts; Poisson is well-suited to count data and
  guarantees non-negative predictions out-of-the-box (loss is computed in
  log-space; predictions are then exponentiated).
* Tabular features with mixed numeric / categorical / NaN. LightGBM handles
  missing values natively (LightGBM's split rule sends missing to the side
  that minimizes loss).
* Fast, well-tuned defaults, cheap to retrain per backtest origin.

This wrapper exposes a small, opinionated API on top of
``lightgbm.LGBMRegressor``:

* :class:`LightGBMPointModel.fit` accepts the supervised table directly
  (output of :func:`seercast.features.build_supervised_table`) and a
  validation slice. It auto-derives feature columns, handles categorical
  encoding consistently across train / valid, and uses early stopping
  when a validation set is provided.
* :meth:`LightGBMPointModel.predict` clips to ``>= 0`` -- Poisson predictions
  are theoretically non-negative but a small numerical floor ensures we
  never emit negative units.
* :meth:`LightGBMPointModel.feature_importance` returns gain or split
  counts as a tidy DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from seercast.features.supervised import (
    IDENTITY_COLUMNS,
    META_COLUMNS,
    SUPERVISED_COLUMNS,
)


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #


DEFAULT_LGB_PARAMS: dict = {
    "objective": "poisson",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "verbose": -1,
    "force_col_wise": True,
}

# Default categoricals: identity columns. ``horizon`` is treated as numeric;
# the model can natively learn step-ahead structure across horizons that way.
DEFAULT_CATEGORICAL_COLUMNS: tuple[str, ...] = IDENTITY_COLUMNS

# Columns NEVER fed to the model (target + meta keys).
NON_FEATURE_COLUMNS: tuple[str, ...] = ("origin_date", "target_date", "target_sales")


def default_feature_columns() -> list[str]:
    """The 50 SUPERVISED_COLUMNS that aren't meta/target."""
    return [c for c in SUPERVISED_COLUMNS if c not in NON_FEATURE_COLUMNS]


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


@dataclass
class LightGBMPointModel:
    """LightGBM point regressor with Poisson objective.

    Attributes are pickle-friendly so the trained model can be persisted
    via ``joblib.dump`` to ``outputs/models/lightgbm_point_model_ca1.pkl``.
    """

    name: str = "lightgbm_point"
    params: dict = field(default_factory=lambda: dict(DEFAULT_LGB_PARAMS))
    n_estimators: int = 2000
    early_stopping_rounds: int = 50
    log_evaluation_period: int = 0  # 0 = silent

    # Set during fit():
    feature_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    _cat_dtypes: dict[str, pd.CategoricalDtype] = field(default_factory=dict)
    _booster: lgb.Booster | None = None
    _best_iteration: int | None = None

    # ------- private helpers ------------------------------------------------
    def _coerce_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the dtype map learned during fit to ``df``. Out-of-vocabulary
        values become NaN, which LightGBM's split rule handles natively.
        """
        out = df.copy()
        for c, dtype in self._cat_dtypes.items():
            if c in out.columns:
                out[c] = out[c].astype(dtype)
        return out

    @staticmethod
    def _assemble_dtype_map(
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame | None,
        cat_cols: Sequence[str],
    ) -> dict[str, pd.CategoricalDtype]:
        m: dict[str, pd.CategoricalDtype] = {}
        for c in cat_cols:
            if c not in train_df.columns:
                continue
            parts = [train_df[c].astype("object")]
            if valid_df is not None and c in valid_df.columns:
                parts.append(valid_df[c].astype("object"))
            cats = pd.unique(pd.concat(parts).dropna())
            cats = sorted(cats)
            m[c] = pd.CategoricalDtype(categories=cats, ordered=False)
        return m

    # ------- public API -----------------------------------------------------
    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame | None = None,
        *,
        target_col: str = "target_sales",
        feature_columns: Sequence[str] | None = None,
        categorical_columns: Sequence[str] | None = None,
    ) -> "LightGBMPointModel":
        """Fit on the supervised table.

        Parameters
        ----------
        train_df
            Rows allowed for training (typically ``target_date <= valid_start``).
        valid_df
            Rows used for early stopping (typically the most-recent 56 days
            of targets ``> valid_start`` and ``<= origin_date``). Pass
            ``None`` to disable early stopping; the model will train for
            exactly ``n_estimators`` rounds.
        target_col
            Label column. Defaults to ``"target_sales"``.
        feature_columns
            Override feature columns. Defaults to
            :func:`default_feature_columns` minus any missing in ``train_df``.
        categorical_columns
            Columns to treat as categorical. Defaults to identity columns.

        Returns
        -------
        self
        """
        if feature_columns is None:
            feature_columns = default_feature_columns()
        feature_columns = [c for c in feature_columns if c in train_df.columns]
        if categorical_columns is None:
            categorical_columns = list(DEFAULT_CATEGORICAL_COLUMNS)
        categorical_columns = [c for c in categorical_columns if c in feature_columns]

        self.feature_columns = list(feature_columns)
        self.categorical_columns = list(categorical_columns)

        # Build a consistent categorical encoding shared across train/valid.
        self._cat_dtypes = self._assemble_dtype_map(
            train_df, valid_df, categorical_columns
        )
        train = self._coerce_categoricals(train_df)
        valid = self._coerce_categoricals(valid_df) if valid_df is not None else None

        X_train = train[feature_columns]
        y_train = train[target_col].astype("float32")

        if valid is not None and len(valid) > 0:
            X_valid = valid[feature_columns]
            y_valid = valid[target_col].astype("float32")
            train_set = lgb.Dataset(
                X_train, label=y_train,
                categorical_feature=categorical_columns,
                free_raw_data=False,
            )
            valid_set = lgb.Dataset(
                X_valid, label=y_valid,
                categorical_feature=categorical_columns,
                reference=train_set,
                free_raw_data=False,
            )
            self._booster = lgb.train(
                self.params,
                train_set,
                num_boost_round=self.n_estimators,
                valid_sets=[valid_set],
                valid_names=["valid"],
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(self.log_evaluation_period),
                ],
            )
            self._best_iteration = self._booster.best_iteration
        else:
            train_set = lgb.Dataset(
                X_train, label=y_train,
                categorical_feature=categorical_columns,
                free_raw_data=False,
            )
            self._booster = lgb.train(
                self.params,
                train_set,
                num_boost_round=self.n_estimators,
                callbacks=[lgb.log_evaluation(self.log_evaluation_period)],
            )
            self._best_iteration = self.n_estimators

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict on the supervised table. Returns a 1-D float array.

        Predictions are clipped to ``>= 0`` (Poisson predictions should
        already be non-negative, but the floor protects against any
        numerical noise).
        """
        if self._booster is None:
            raise RuntimeError("LightGBMPointModel.predict called before fit")
        coerced = self._coerce_categoricals(df)
        X = coerced[self.feature_columns]
        preds = self._booster.predict(X, num_iteration=self._best_iteration)
        return np.clip(np.asarray(preds, dtype=float), 0.0, None)

    def feature_importance(self, kind: str = "gain") -> pd.DataFrame:
        """Return feature importance as a tidy DataFrame, sorted descending."""
        if self._booster is None:
            raise RuntimeError("feature_importance called before fit")
        importance = self._booster.feature_importance(importance_type=kind)
        return (
            pd.DataFrame(
                {"feature": self.feature_columns, kind: importance.astype(float)}
            )
            .sort_values(kind, ascending=False)
            .reset_index(drop=True)
        )


__all__ = [
    "DEFAULT_LGB_PARAMS",
    "DEFAULT_CATEGORICAL_COLUMNS",
    "NON_FEATURE_COLUMNS",
    "default_feature_columns",
    "LightGBMPointModel",
]
