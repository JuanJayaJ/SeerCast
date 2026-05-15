"""Per-horizon LightGBM wrappers.

The Phase 5/6 models use ``horizon`` as a feature -- one shared booster
across all four direct horizons. The diagnostic step found horizon-
specific behaviour (especially horizon-28 underforecasting), so this
module trains **one sub-model per horizon** instead:

    horizon 1   -> its own QuantileLightGBMModel (or LightGBMPointModel)
    horizon 7   -> its own model
    horizon 14  -> its own model
    horizon 28  -> its own model

For each horizon ``h`` the sub-model is fit only on training rows where
``horizon == h``. The Phase 5 split rule applies unchanged at the outer
level (``target_date <= origin``); per-horizon filtering happens after
that split.

The classes here are thin orchestrators on top of the existing
:class:`QuantileLightGBMModel` and :class:`LightGBMPointModel`. They
inherit all of their guarantees (categorical encoding, early stopping,
leakage-safe split, monotonic quantile crossing fix).

Implementation note: ``horizon`` is dropped from the default feature
list inside each sub-model -- the column is constant within a sub-
model's training rows and would add nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from seercast.models.lightgbm_model import (
    DEFAULT_CATEGORICAL_COLUMNS,
    DEFAULT_LGB_PARAMS,
    LightGBMPointModel,
    default_feature_columns,
)
from seercast.models.quantile_lightgbm import (
    DEFAULT_QUANTILE_LGB_PARAMS,
    QuantileLightGBMModel,
    quantile_column_name,
)


# Default horizons -- align with the project's DIRECT_HORIZONS.
_DEFAULT_HORIZONS: tuple[int, ...] = (1, 7, 14, 28)


def _features_without_horizon(feature_columns: Sequence[str] | None) -> list[str]:
    """Default feature list minus ``horizon`` (constant per sub-model)."""
    cols = default_feature_columns() if feature_columns is None else list(feature_columns)
    return [c for c in cols if c != "horizon"]


# --------------------------------------------------------------------------- #
# Quantile per-horizon
# --------------------------------------------------------------------------- #


@dataclass
class PerHorizonQuantileLightGBMModel:
    """One quantile booster per horizon in :attr:`horizons`.

    Output of :meth:`predict` is a DataFrame with one column per quantile
    (``p10`` / ``p50`` / ``p90``), indexed to match the input frame.
    Each row is filled by the booster that matches its ``horizon``.
    """

    name: str = "per_horizon_lightgbm_quantile"
    horizons: tuple[int, ...] = _DEFAULT_HORIZONS
    quantiles: tuple[float, ...] = (0.10, 0.50, 0.90)
    params: dict = field(default_factory=lambda: dict(DEFAULT_QUANTILE_LGB_PARAMS))
    n_estimators: int = 2000
    early_stopping_rounds: int = 50
    log_evaluation_period: int = 0

    _models: dict[int, QuantileLightGBMModel] = field(default_factory=dict)

    # ---------- fit / predict ----------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame | None = None,
        *,
        target_col: str = "target_sales",
        feature_columns: Sequence[str] | None = None,
        categorical_columns: Sequence[str] | None = None,
    ) -> "PerHorizonQuantileLightGBMModel":
        if "horizon" not in train_df.columns:
            raise KeyError("train_df must include a 'horizon' column")

        present = set(train_df["horizon"].unique())
        missing = [h for h in self.horizons if h not in present]
        if missing:
            raise ValueError(
                f"horizons missing from train_df: {missing}; "
                f"present horizons: {sorted(present)}"
            )

        feature_columns = _features_without_horizon(feature_columns)

        self._models = {}
        for h in self.horizons:
            train_h = train_df.loc[train_df["horizon"] == h]
            valid_h = None
            if valid_df is not None and len(valid_df) > 0:
                valid_h = valid_df.loc[valid_df["horizon"] == h]
                if len(valid_h) == 0:
                    valid_h = None
            sub_model = QuantileLightGBMModel(
                quantiles=tuple(self.quantiles),
                params=dict(self.params),
                n_estimators=self.n_estimators,
                early_stopping_rounds=self.early_stopping_rounds,
                log_evaluation_period=self.log_evaluation_period,
            )
            sub_model.fit(
                train_h,
                valid_df=valid_h,
                target_col=target_col,
                feature_columns=feature_columns,
                categorical_columns=categorical_columns,
            )
            self._models[int(h)] = sub_model
        return self

    def predict(
        self,
        df: pd.DataFrame,
        *,
        fix_crossings: bool = True,
    ) -> pd.DataFrame:
        if not self._models:
            raise RuntimeError("predict called before fit")
        if "horizon" not in df.columns:
            raise KeyError("df must include a 'horizon' column")

        col_names = [quantile_column_name(q) for q in self.quantiles]
        out = pd.DataFrame(
            0.0, index=df.index, columns=col_names, dtype="float64"
        )
        filled_mask = pd.Series(False, index=df.index)

        for h, sub_model in self._models.items():
            mask = df["horizon"] == h
            if not mask.any():
                continue
            sub = df.loc[mask]
            preds = sub_model.predict(sub, fix_crossings=fix_crossings)
            # preds.index == sub.index by construction of QuantileLightGBMModel.
            for c in col_names:
                out.loc[preds.index, c] = preds[c].values
            filled_mask.loc[preds.index] = True

        unfilled = (~filled_mask).sum()
        if unfilled > 0:
            raise ValueError(
                f"{unfilled} input rows had a horizon with no trained sub-model; "
                f"trained horizons: {sorted(self._models)}"
            )
        return out

    # ---------- introspection ----------------------------------------------

    def feature_importance(
        self,
        *,
        horizon: int,
        quantile: float = 0.5,
        kind: str = "gain",
    ) -> pd.DataFrame:
        if horizon not in self._models:
            raise KeyError(f"no model for horizon {horizon}; have {sorted(self._models)}")
        return self._models[horizon].feature_importance(quantile=quantile, kind=kind)

    @property
    def trained_horizons(self) -> list[int]:
        return sorted(self._models)


# --------------------------------------------------------------------------- #
# Point per-horizon (optional companion)
# --------------------------------------------------------------------------- #


@dataclass
class PerHorizonLightGBMPointModel:
    """One Poisson point booster per horizon. Predict returns 1-D float array."""

    name: str = "per_horizon_lightgbm_point"
    horizons: tuple[int, ...] = _DEFAULT_HORIZONS
    params: dict = field(default_factory=lambda: dict(DEFAULT_LGB_PARAMS))
    n_estimators: int = 2000
    early_stopping_rounds: int = 50
    log_evaluation_period: int = 0
    _models: dict[int, LightGBMPointModel] = field(default_factory=dict)

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame | None = None,
        *,
        target_col: str = "target_sales",
        feature_columns: Sequence[str] | None = None,
        categorical_columns: Sequence[str] | None = None,
    ) -> "PerHorizonLightGBMPointModel":
        if "horizon" not in train_df.columns:
            raise KeyError("train_df must include a 'horizon' column")
        present = set(train_df["horizon"].unique())
        missing = [h for h in self.horizons if h not in present]
        if missing:
            raise ValueError(
                f"horizons missing from train_df: {missing}; "
                f"present horizons: {sorted(present)}"
            )

        feature_columns = _features_without_horizon(feature_columns)

        self._models = {}
        for h in self.horizons:
            train_h = train_df.loc[train_df["horizon"] == h]
            valid_h = None
            if valid_df is not None and len(valid_df) > 0:
                valid_h = valid_df.loc[valid_df["horizon"] == h]
                if len(valid_h) == 0:
                    valid_h = None
            sub_model = LightGBMPointModel(
                params=dict(self.params),
                n_estimators=self.n_estimators,
                early_stopping_rounds=self.early_stopping_rounds,
                log_evaluation_period=self.log_evaluation_period,
            )
            sub_model.fit(
                train_h,
                valid_df=valid_h,
                target_col=target_col,
                feature_columns=feature_columns,
                categorical_columns=categorical_columns,
            )
            self._models[int(h)] = sub_model
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self._models:
            raise RuntimeError("predict called before fit")
        if "horizon" not in df.columns:
            raise KeyError("df must include a 'horizon' column")

        out = np.zeros(len(df), dtype=float)
        filled = np.zeros(len(df), dtype=bool)
        pos_by_index = {idx: i for i, idx in enumerate(df.index)}

        for h, sub_model in self._models.items():
            mask = df["horizon"] == h
            if not mask.any():
                continue
            sub = df.loc[mask]
            preds = sub_model.predict(sub)
            for idx, val in zip(sub.index, preds):
                pos = pos_by_index[idx]
                out[pos] = val
                filled[pos] = True

        if not filled.all():
            raise ValueError(
                f"{(~filled).sum()} rows had a horizon with no trained sub-model; "
                f"trained horizons: {sorted(self._models)}"
            )
        return out

    def feature_importance(
        self,
        *,
        horizon: int,
        kind: str = "gain",
    ) -> pd.DataFrame:
        if horizon not in self._models:
            raise KeyError(f"no model for horizon {horizon}; have {sorted(self._models)}")
        return self._models[horizon].feature_importance(kind=kind)

    @property
    def trained_horizons(self) -> list[int]:
        return sorted(self._models)


__all__ = [
    "PerHorizonQuantileLightGBMModel",
    "PerHorizonLightGBMPointModel",
]
