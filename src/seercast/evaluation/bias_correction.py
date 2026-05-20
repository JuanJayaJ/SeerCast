"""Post-hoc bias correction for SeerCast forecasts.

Phase 8 — Credibility Pass.

The shared-horizon LightGBM quantile p50 model under-forecasts by Bias ≈ −0.16
on CA_1 backtest. This module implements three post-hoc corrections that are
fit on a *calibration* slice of the backtest and then applied to the
*evaluation* slice. The correction is a scalar (or per-horizon vector) and
is NOT a new model.

Important framing:

* This is post-hoc calibration, not a new model. We do not re-fit LightGBM.
* The calibration set must not overlap with the evaluation set. The natural
  split for the lifecycle backtest is "earliest origin = calibration, the
  rest = evaluation" — three origins gives one calibration fold and two
  evaluation folds.
* Corrected predictions are clipped at zero. Negative demand is meaningless.
* The original prediction column is preserved as ``<col>_uncorrected``.

Available methods
-----------------
``"multiplicative_mean_ratio"``
    ``factor = sum(actual) / sum(pred)`` over the calibration set.
    ``new_pred = pred * factor``. Removes mean-ratio bias; scale-aware.

``"additive_mean_error"``
    ``shift = mean(actual - pred)`` over the calibration set.
    ``new_pred = pred + shift`` (clipped at 0). Removes mean-shift bias.

``"per_horizon_multiplicative"``
    A separate ``factor[h] = sum(actual_h) / sum(pred_h)`` per forecast
    horizon. Useful if bias grows with horizon (it usually does).

If we later want richer correction (per-cat, per-lifecycle-stage) the
fit/transform shape generalises without breaking callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from seercast.evaluation.metrics import all_point_metrics


SUPPORTED_METHODS: tuple[str, ...] = (
    "multiplicative_mean_ratio",
    "additive_mean_error",
    "per_horizon_multiplicative",
)


# --------------------------------------------------------------------------- #
# Corrector
# --------------------------------------------------------------------------- #


@dataclass
class BiasCorrector:
    """Fit a scalar (or per-horizon) bias correction on calibration data.

    Use :meth:`fit` to estimate the correction parameter(s) from a held-out
    calibration slice (e.g. the earliest backtest origin), then
    :meth:`transform` on any prediction frame that shares the same column
    schema. ``transform`` is pure: it never mutates the input frame.
    """

    method: str
    pred_col: str = "p50"
    actual_col: str = "actual"
    horizon_col: str = "horizon"

    # Populated by .fit():
    scalar_factor_: float | None = field(default=None, init=False)
    scalar_shift_: float | None = field(default=None, init=False)
    per_horizon_factor_: dict[int, float] | None = field(default=None, init=False)
    fitted_n_: int | None = field(default=None, init=False)
    fallback_horizon_factor_: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"unknown method '{self.method}'. Pick from {SUPPORTED_METHODS}."
            )

    # ------------------------------------------------------------------ #
    # fit
    # ------------------------------------------------------------------ #

    def fit(self, calibration_df: pd.DataFrame) -> "BiasCorrector":
        """Estimate correction parameter(s) from ``calibration_df``.

        Rows with NaN actual or prediction are dropped before fitting.
        """
        clean = calibration_df.dropna(
            subset=[self.actual_col, self.pred_col]
        )
        if clean.empty:
            raise ValueError("calibration_df has no usable rows after dropping NaNs.")

        self.fitted_n_ = int(len(clean))
        actual = clean[self.actual_col].to_numpy(dtype=float)
        pred = clean[self.pred_col].to_numpy(dtype=float)

        if self.method == "multiplicative_mean_ratio":
            sp = float(pred.sum())
            sa = float(actual.sum())
            self.scalar_factor_ = (sa / sp) if sp > 0 else 1.0

        elif self.method == "additive_mean_error":
            self.scalar_shift_ = float(np.mean(actual - pred))

        elif self.method == "per_horizon_multiplicative":
            if self.horizon_col not in clean.columns:
                raise KeyError(
                    f"per_horizon_multiplicative needs the '{self.horizon_col}' "
                    f"column in calibration_df."
                )
            factors: dict[int, float] = {}
            for h, sub in clean.groupby(self.horizon_col):
                sp = float(sub[self.pred_col].sum())
                sa = float(sub[self.actual_col].sum())
                factors[int(h)] = (sa / sp) if sp > 0 else 1.0
            self.per_horizon_factor_ = factors
            # Global fallback factor for any horizon not seen in calibration.
            sp_all = float(pred.sum())
            sa_all = float(actual.sum())
            self.fallback_horizon_factor_ = (sa_all / sp_all) if sp_all > 0 else 1.0

        return self

    # ------------------------------------------------------------------ #
    # transform
    # ------------------------------------------------------------------ #

    def transform(
        self,
        predictions_df: pd.DataFrame,
        cols: Iterable[str] | None = None,
        suffix: str = "_corrected",
        keep_uncorrected: bool = True,
        uncorrected_suffix: str = "_uncorrected",
    ) -> pd.DataFrame:
        """Apply the fitted correction to one or more prediction columns.

        Parameters
        ----------
        predictions_df
            Frame containing at least the columns in ``cols`` and (for
            per-horizon) ``self.horizon_col``.
        cols
            Which columns to correct. Defaults to ``[self.pred_col]``.
            If you also want to correct ``p10`` and ``p90`` with the same
            factor (so the interval stays valid), pass them all in.
        suffix
            Suffix for the corrected column(s).
        keep_uncorrected
            If True, copies the original column to ``<col><uncorrected_suffix>``
            before overwriting. We always write the corrected value into
            ``<col><suffix>`` rather than overwriting the original.

        Returns
        -------
        pandas.DataFrame
            A copy of ``predictions_df`` with the corrected columns added.
        """
        if self._is_unfitted():
            raise RuntimeError("BiasCorrector is not fitted. Call .fit first.")

        cols = list(cols) if cols is not None else [self.pred_col]
        out = predictions_df.copy()

        for col in cols:
            if col not in out.columns:
                raise KeyError(f"column '{col}' not in predictions_df.")
            if keep_uncorrected:
                out[f"{col}{uncorrected_suffix}"] = out[col].astype(float)

            corrected = self._apply_to_series(out, col)
            corrected = np.clip(corrected, a_min=0.0, a_max=None)
            out[f"{col}{suffix}"] = corrected

        return out

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _apply_to_series(
        self, df: pd.DataFrame, col: str,
    ) -> np.ndarray:
        pred = df[col].to_numpy(dtype=float)
        if self.method == "multiplicative_mean_ratio":
            return pred * float(self.scalar_factor_ or 1.0)
        if self.method == "additive_mean_error":
            return pred + float(self.scalar_shift_ or 0.0)
        if self.method == "per_horizon_multiplicative":
            if self.per_horizon_factor_ is None:
                raise RuntimeError("per_horizon_multiplicative not fitted.")
            if self.horizon_col not in df.columns:
                raise KeyError(
                    f"per_horizon_multiplicative needs the '{self.horizon_col}' "
                    f"column in predictions_df."
                )
            horizons = df[self.horizon_col].to_numpy()
            fallback = float(self.fallback_horizon_factor_ or 1.0)
            factors = np.array(
                [self.per_horizon_factor_.get(int(h), fallback) for h in horizons],
                dtype=float,
            )
            return pred * factors
        raise AssertionError(f"unreachable: method={self.method}")

    def _is_unfitted(self) -> bool:
        if self.method == "multiplicative_mean_ratio":
            return self.scalar_factor_ is None
        if self.method == "additive_mean_error":
            return self.scalar_shift_ is None
        if self.method == "per_horizon_multiplicative":
            return self.per_horizon_factor_ is None
        return True

    def describe(self) -> dict:
        """Human-readable summary of the fitted correction."""
        d: dict = {
            "method": self.method,
            "fitted_n": self.fitted_n_,
            "pred_col": self.pred_col,
        }
        if self.scalar_factor_ is not None:
            d["scalar_factor"] = self.scalar_factor_
        if self.scalar_shift_ is not None:
            d["scalar_shift"] = self.scalar_shift_
        if self.per_horizon_factor_ is not None:
            d["per_horizon_factor"] = dict(self.per_horizon_factor_)
            d["fallback_horizon_factor"] = self.fallback_horizon_factor_
        return d


# --------------------------------------------------------------------------- #
# Comparison helper
# --------------------------------------------------------------------------- #


def compare_corrections(
    calibration_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    *,
    pred_col: str = "p50",
    actual_col: str = "actual",
    horizon_col: str = "horizon",
    methods: Iterable[str] = SUPPORTED_METHODS,
    quantile_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Fit each correction on ``calibration_df`` and report before/after
    point metrics on ``evaluation_df``.

    The ``quantile_cols`` parameter is only used for ``"multiplicative_mean_ratio"``
    and ``"per_horizon_multiplicative"`` — when provided, the same factor is
    also applied to those columns so that the corrected interval stays
    centered on the corrected ``pred_col``. ``additive_mean_error`` does
    NOT widen-or-shift the interval (this would be wrong), so quantile_cols
    is ignored for that method.

    Returns
    -------
    pandas.DataFrame
        One row per method, columns: ``method``, ``n``, ``MAE_before``,
        ``MAE_after``, ``RMSE_before``, ``RMSE_after``, ``WAPE_before``,
        ``WAPE_after``, ``Bias_before``, ``Bias_after``, ``factor_or_shift``.
    """
    clean_eval = evaluation_df.dropna(subset=[actual_col, pred_col])
    base_metrics = all_point_metrics(
        clean_eval[actual_col], clean_eval[pred_col],
    )

    rows: list[dict] = []
    for method in methods:
        corr = BiasCorrector(
            method=method,
            pred_col=pred_col,
            actual_col=actual_col,
            horizon_col=horizon_col,
        ).fit(calibration_df)

        cols_to_correct = [pred_col]
        if (
            quantile_cols is not None
            and method in {"multiplicative_mean_ratio", "per_horizon_multiplicative"}
        ):
            cols_to_correct = list({pred_col, *quantile_cols})

        out = corr.transform(
            clean_eval, cols=cols_to_correct,
            suffix="_corrected", keep_uncorrected=False,
        )
        corrected_metrics = all_point_metrics(
            out[actual_col], out[f"{pred_col}_corrected"],
        )

        factor_or_shift: float
        if method == "additive_mean_error":
            factor_or_shift = float(corr.scalar_shift_ or 0.0)
        elif method == "multiplicative_mean_ratio":
            factor_or_shift = float(corr.scalar_factor_ or 1.0)
        else:  # per_horizon_multiplicative -> mean of factors
            factors = list((corr.per_horizon_factor_ or {}).values())
            factor_or_shift = float(np.mean(factors)) if factors else 1.0

        rows.append({
            "method": method,
            "n": int(len(clean_eval)),
            "MAE_before": base_metrics["MAE"],
            "MAE_after": corrected_metrics["MAE"],
            "RMSE_before": base_metrics["RMSE"],
            "RMSE_after": corrected_metrics["RMSE"],
            "WAPE_before": base_metrics["WAPE"],
            "WAPE_after": corrected_metrics["WAPE"],
            "Bias_before": base_metrics["Bias"],
            "Bias_after": corrected_metrics["Bias"],
            "factor_or_shift_mean": factor_or_shift,
        })

    return pd.DataFrame(rows)


__all__ = [
    "SUPPORTED_METHODS",
    "BiasCorrector",
    "compare_corrections",
]
