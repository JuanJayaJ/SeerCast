"""LightGBM objective sweep — Phase 9.

Why this module exists
----------------------
Phase 8 (credibility pass) surfaced an honest tension: the current quantile
p50 model wins on WAPE but the Poisson point model wins on RMSE/RMSSE, and
a scalar post-hoc bias correction reduces bias only at the cost of WAPE.
The right next step is to attack the loss function *inside* training rather
than after the fact.

This module wraps :class:`seercast.models.lightgbm_model.LightGBMPointModel`
into a *candidate-aware* wrapper that supports:

* Different LightGBM ``objective`` values (l2 / l1 / poisson / tweedie).
* Optional ``log1p`` target transform with ``expm1`` inverse at predict.
* Tweedie ``tweedie_variance_power`` sweep (1.1, 1.2, 1.3).
* Predictions clipped at 0 (already done by ``LightGBMPointModel.predict``,
  but the log1p variant must also apply ``expm1`` before clipping).

The goal is NOT to replace the existing model class — it's to make it easy
to run several candidates side-by-side on the same backtest splits.

Honest framing
--------------
* The current production-best lightgbm_point uses ``objective="poisson"``
  (see ``DEFAULT_LGB_PARAMS`` in lightgbm_model.py). So the "L2" candidate
  in this sweep is genuinely new, not a rebrand.
* log1p shrinks large counts and can help intermittent demand, but it
  introduces a transformation bias on the inverse: ``expm1(mean(log1p(y)))``
  is not ``mean(y)``. We document this and let the metrics speak.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from seercast.models.lightgbm_model import (
    DEFAULT_LGB_PARAMS,
    LightGBMPointModel,
)


# --------------------------------------------------------------------------- #
# Target transforms
# --------------------------------------------------------------------------- #


def _identity_transform(y: np.ndarray) -> np.ndarray:
    return y


def _identity_inverse(yhat: np.ndarray) -> np.ndarray:
    return yhat


def _log1p_transform(y: np.ndarray) -> np.ndarray:
    """log1p(y). Safe for y >= -1; we clip at 0 upstream so y >= 0 always."""
    return np.log1p(np.clip(np.asarray(y, dtype=float), a_min=0.0, a_max=None))


def _log1p_inverse(yhat: np.ndarray) -> np.ndarray:
    """expm1(yhat). Output is clipped at 0 by the model wrapper anyway."""
    return np.expm1(np.asarray(yhat, dtype=float))


TARGET_TRANSFORMS: dict[str, tuple[Callable, Callable]] = {
    "identity": (_identity_transform, _identity_inverse),
    "log1p": (_log1p_transform, _log1p_inverse),
}


# --------------------------------------------------------------------------- #
# Candidate spec
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CandidateSpec:
    """One objective-sweep candidate.

    Attributes
    ----------
    name
        Short slug used in output filenames and the ``model`` column of
        prediction frames.
    objective
        Value passed to LightGBM's ``objective`` param.
    target_transform
        ``"identity"`` (default) or ``"log1p"``. Determines the y / yhat
        transform applied around the model.
    extra_params
        Any extra LightGBM params merged on top of ``DEFAULT_LGB_PARAMS``.
        E.g. ``{"tweedie_variance_power": 1.2}``.
    description
        Free-text label for reports.
    """

    name: str
    objective: str
    target_transform: str = "identity"
    extra_params: dict = field(default_factory=dict)
    description: str = ""

    def build_params(self, base: dict | None = None) -> dict:
        params = dict(DEFAULT_LGB_PARAMS) if base is None else dict(base)
        params["objective"] = self.objective
        params.update(self.extra_params)
        return params


# --------------------------------------------------------------------------- #
# Catalog of candidates
# --------------------------------------------------------------------------- #


def default_catalog(include_tweedie: bool = True) -> list[CandidateSpec]:
    """The standard Phase 9 sweep.

    Returns a list of :class:`CandidateSpec` objects. Order matters only
    for human-readable reports.
    """
    catalog = [
        CandidateSpec(
            name="poisson",
            objective="poisson",
            description="Current lightgbm_point baseline (Poisson on raw counts).",
        ),
        CandidateSpec(
            name="regression_l2",
            objective="regression",
            description="L2 / squared error on raw counts.",
        ),
        CandidateSpec(
            name="regression_l1",
            objective="regression_l1",
            description="L1 / MAE on raw counts.",
        ),
        CandidateSpec(
            name="log1p_regression_l2",
            objective="regression",
            target_transform="log1p",
            description="L2 on log1p(target); expm1 at predict.",
        ),
        CandidateSpec(
            name="log1p_regression_l1",
            objective="regression_l1",
            target_transform="log1p",
            description="L1 on log1p(target); expm1 at predict.",
        ),
    ]
    if include_tweedie:
        for vp in (1.1, 1.2, 1.3):
            catalog.append(CandidateSpec(
                name=f"tweedie_{str(vp).replace('.', '_')}",
                objective="tweedie",
                extra_params={"tweedie_variance_power": float(vp)},
                description=f"Tweedie with variance_power={vp}.",
            ))
    return catalog


def fast_catalog() -> list[CandidateSpec]:
    """A small sweep for the --fast CLI mode: poisson, L2, log1p_L2,
    tweedie 1.2. Useful for a quick local smoke run."""
    return [
        CandidateSpec(name="poisson", objective="poisson",
                      description="Current baseline."),
        CandidateSpec(name="regression_l2", objective="regression",
                      description="L2 / squared error."),
        CandidateSpec(name="log1p_regression_l2", objective="regression",
                      target_transform="log1p",
                      description="L2 on log1p(target)."),
        CandidateSpec(name="tweedie_1_2", objective="tweedie",
                      extra_params={"tweedie_variance_power": 1.2},
                      description="Tweedie variance_power=1.2."),
    ]


# --------------------------------------------------------------------------- #
# Fit / predict wrapper
# --------------------------------------------------------------------------- #


@dataclass
class ObjectiveCandidateModel:
    """A LightGBMPointModel wrapped to apply / invert a target transform.

    Created via :func:`fit_candidate`. The transform is applied to the
    target column at training time and inverted at predict time; features
    are not transformed.
    """

    spec: CandidateSpec
    inner: LightGBMPointModel
    target_transform: str = "identity"
    transformed_target_col: str | None = None

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict with the inverse transform + non-negative clip."""
        raw = self.inner.predict(df)
        _, inverse = TARGET_TRANSFORMS[self.target_transform]
        out = inverse(raw)
        return np.clip(out, 0.0, None)


def fit_candidate(
    spec: CandidateSpec,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None,
    *,
    target_col: str = "target_sales",
    n_estimators: int = 2000,
    early_stopping_rounds: int = 50,
    log_evaluation_period: int = 0,
) -> ObjectiveCandidateModel:
    """Fit one candidate on the supervised table.

    If ``spec.target_transform == "log1p"``, we add a temporary column
    ``log1p_target_sales`` (or whatever the transform's name is) to the
    training and validation frames, and pass its name to the inner model.
    The original ``target_sales`` column is left untouched.
    """
    forward, _ = TARGET_TRANSFORMS[spec.target_transform]

    inner = LightGBMPointModel(
        name=spec.name,
        params=spec.build_params(),
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        log_evaluation_period=log_evaluation_period,
    )

    if spec.target_transform == "identity":
        inner.fit(
            train_df, valid_df=valid_df,
            target_col=target_col,
        )
        transformed_target_col = target_col
    else:
        transformed_col = f"{spec.target_transform}_{target_col}"
        train_t = train_df.copy()
        train_t[transformed_col] = forward(train_t[target_col].to_numpy())
        if valid_df is not None and len(valid_df) > 0:
            valid_t = valid_df.copy()
            valid_t[transformed_col] = forward(valid_t[target_col].to_numpy())
        else:
            valid_t = valid_df
        inner.fit(
            train_t, valid_df=valid_t,
            target_col=transformed_col,
        )
        transformed_target_col = transformed_col

    return ObjectiveCandidateModel(
        spec=spec,
        inner=inner,
        target_transform=spec.target_transform,
        transformed_target_col=transformed_target_col,
    )


__all__ = [
    "CandidateSpec",
    "TARGET_TRANSFORMS",
    "ObjectiveCandidateModel",
    "default_catalog",
    "fast_catalog",
    "fit_candidate",
]
