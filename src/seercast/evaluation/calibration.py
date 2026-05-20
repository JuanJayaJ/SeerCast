"""Quantile-interval calibration diagnostics.

Phase 8 — Credibility Pass.

The shared-horizon quantile model emits p10/p50/p90. We want to know:

1. Does the [p10, p90] interval actually cover the realised actual 80% of
   the time? If not, by how much, and is the miscalibration consistent
   across horizons / categories / demand segments?
2. Can we narrow or widen the intervals with a single scalar so the
   coverage hits the target on a calibration fold? (split-conformal,
   the simplest variant)

We deliberately keep this module pure ``pandas``-friendly: pass in a
DataFrame with at least ``p10``, ``p50``, ``p90``, ``actual`` and it
returns tidy long frames you can write straight to CSV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Coverage helpers
# --------------------------------------------------------------------------- #


def _safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else float("nan")


def overall_coverage(
    df: pd.DataFrame,
    *,
    lo_col: str = "p10",
    hi_col: str = "p90",
    actual_col: str = "actual",
    target: float = 0.80,
) -> dict:
    """Overall interval coverage and width."""
    clean = df.dropna(subset=[lo_col, hi_col, actual_col])
    lo = clean[lo_col].to_numpy(dtype=float)
    hi = clean[hi_col].to_numpy(dtype=float)
    y = clean[actual_col].to_numpy(dtype=float)
    covered = ((y >= lo) & (y <= hi)).astype(float)
    width = hi - lo
    return {
        "n": int(len(clean)),
        "target_coverage": float(target),
        "empirical_coverage": _safe_mean(covered),
        "calibration_error": (
            _safe_mean(covered) - float(target)
            if covered.size else float("nan")
        ),
        "mean_interval_width": _safe_mean(width),
        "median_interval_width": float(np.median(width)) if width.size else float("nan"),
    }


def coverage_by_group(
    df: pd.DataFrame,
    *,
    by: Sequence[str],
    lo_col: str = "p10",
    hi_col: str = "p90",
    actual_col: str = "actual",
    target: float = 0.80,
) -> pd.DataFrame:
    """Coverage table grouped by one or more columns."""
    needed = list(by) + [lo_col, hi_col, actual_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"coverage_by_group needs columns {missing}")
    clean = df.dropna(subset=[lo_col, hi_col, actual_col])

    rows: list[dict] = []
    for keys, sub in clean.groupby(list(by), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        lo = sub[lo_col].to_numpy(dtype=float)
        hi = sub[hi_col].to_numpy(dtype=float)
        y = sub[actual_col].to_numpy(dtype=float)
        covered = ((y >= lo) & (y <= hi)).astype(float)
        width = hi - lo
        row = dict(zip(by, keys))
        row.update({
            "n": int(len(sub)),
            "target_coverage": float(target),
            "empirical_coverage": _safe_mean(covered),
            "calibration_error": (
                _safe_mean(covered) - float(target)
                if covered.size else float("nan")
            ),
            "mean_interval_width": _safe_mean(width),
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(list(by)).reset_index(drop=True)
    return out


def per_tail_coverage(
    df: pd.DataFrame,
    *,
    lo_col: str = "p10",
    hi_col: str = "p90",
    actual_col: str = "actual",
    nominal_lo: float = 0.10,
    nominal_hi: float = 0.90,
) -> dict:
    """One-sided coverage (P(y <= lo) and P(y <= hi)) — useful to see
    whether miscalibration is symmetric or one-tailed.

    Returns the empirical equivalents of the nominal lo/hi quantiles.
    """
    clean = df.dropna(subset=[lo_col, hi_col, actual_col])
    y = clean[actual_col].to_numpy(dtype=float)
    lo = clean[lo_col].to_numpy(dtype=float)
    hi = clean[hi_col].to_numpy(dtype=float)
    p_below_lo = _safe_mean((y <= lo).astype(float))
    p_below_hi = _safe_mean((y <= hi).astype(float))
    return {
        "n": int(len(clean)),
        "nominal_lo_q": float(nominal_lo),
        "empirical_lo_q": p_below_lo,   # ≈ nominal_lo when calibrated
        "nominal_hi_q": float(nominal_hi),
        "empirical_hi_q": p_below_hi,   # ≈ nominal_hi when calibrated
        "lo_tail_excess": p_below_lo - nominal_lo,
        "hi_tail_excess": p_below_hi - nominal_hi,
    }


# --------------------------------------------------------------------------- #
# Split-conformal width-scaling
# --------------------------------------------------------------------------- #


@dataclass
class ConformalScaler:
    """Split-conformal calibration of [lo, hi] around a center.

    On the calibration set we compute, per row, the *non-conformity score*::

        s_i = max( (lo_i - y_i) / (center_i - lo_i),
                   (y_i - hi_i) / (hi_i - center_i) )

    The widening factor ``q`` is the empirical ``(1 - alpha)``-quantile of
    ``s_i`` with the standard (1+1/n) finite-sample correction. After
    calibration, the new interval is::

        lo_new = center - q * (center - lo)
        hi_new = center + q * (hi - center)

    ``q > 1`` widens; ``q < 1`` narrows. We bound ``q`` away from 0 to
    avoid pathological collapsed intervals when the model is wildly
    over-conservative on calibration.
    """

    target: float = 0.80
    factor_: float | None = None
    fitted_n_: int | None = None

    def fit(
        self,
        calibration_df: pd.DataFrame,
        *,
        lo_col: str = "p10",
        center_col: str = "p50",
        hi_col: str = "p90",
        actual_col: str = "actual",
        min_factor: float = 0.10,
    ) -> "ConformalScaler":
        clean = calibration_df.dropna(
            subset=[lo_col, center_col, hi_col, actual_col]
        )
        if clean.empty:
            raise ValueError("calibration_df has no usable rows.")

        lo = clean[lo_col].to_numpy(dtype=float)
        c = clean[center_col].to_numpy(dtype=float)
        hi = clean[hi_col].to_numpy(dtype=float)
        y = clean[actual_col].to_numpy(dtype=float)

        # Half-widths, guarded against zero.
        half_lo = np.maximum(c - lo, 1e-9)
        half_hi = np.maximum(hi - c, 1e-9)
        s_lo = (lo - y) / half_lo   # positive => y fell below the interval
        s_hi = (y - hi) / half_hi   # positive => y fell above
        s = np.maximum(s_lo, s_hi)

        n = s.size
        alpha = 1.0 - self.target
        # finite-sample correction: ceil((n+1)(1-alpha)) / n
        rank = np.ceil((n + 1) * (1 - alpha)) / n
        rank = float(min(max(rank, 0.0), 1.0))
        # If the model is already conservative everywhere, s is negative
        # at the chosen quantile, which would shrink the interval below
        # 1.0. We allow that, but floor at ``min_factor`` so we never
        # collapse to width 0.
        q = float(np.quantile(s, rank, method="higher"))
        # The factor that scales half-widths is 1 + q in the classic
        # locally-adaptive conformal formulation. We use ``max(0, 1+q)``
        # then clamp.
        factor = max(min_factor, 1.0 + q)
        self.factor_ = factor
        self.fitted_n_ = int(n)
        return self

    def transform(
        self,
        df: pd.DataFrame,
        *,
        lo_col: str = "p10",
        center_col: str = "p50",
        hi_col: str = "p90",
        out_lo_col: str = "p10_conformal",
        out_hi_col: str = "p90_conformal",
        clip_lo_at_zero: bool = True,
    ) -> pd.DataFrame:
        if self.factor_ is None:
            raise RuntimeError("ConformalScaler is not fitted.")
        out = df.copy()
        c = out[center_col].to_numpy(dtype=float)
        lo = out[lo_col].to_numpy(dtype=float)
        hi = out[hi_col].to_numpy(dtype=float)
        new_lo = c - self.factor_ * (c - lo)
        new_hi = c + self.factor_ * (hi - c)
        if clip_lo_at_zero:
            new_lo = np.clip(new_lo, a_min=0.0, a_max=None)
        out[out_lo_col] = new_lo
        out[out_hi_col] = new_hi
        return out


# --------------------------------------------------------------------------- #
# Convenience: full calibration report
# --------------------------------------------------------------------------- #


def build_calibration_report(
    df: pd.DataFrame,
    *,
    lo_col: str = "p10",
    hi_col: str = "p90",
    actual_col: str = "actual",
    target: float = 0.80,
    by_groups: Iterable[Sequence[str]] = (("horizon",), ("cat_id",)),
) -> pd.DataFrame:
    """Build a single long-form calibration table with one row per
    (level, key) combination, plus an "overall" row.

    Columns: ``level``, ``key``, ``n``, ``target_coverage``,
    ``empirical_coverage``, ``calibration_error``, ``mean_interval_width``.
    """
    rows: list[dict] = []

    overall = overall_coverage(
        df, lo_col=lo_col, hi_col=hi_col,
        actual_col=actual_col, target=target,
    )
    rows.append({
        "level": "overall",
        "key": "",
        **{k: overall[k] for k in (
            "n", "target_coverage", "empirical_coverage",
            "calibration_error", "mean_interval_width",
        )},
    })

    for by in by_groups:
        cols = list(by)
        # Skip levels we don't have in the frame (e.g. cat_id wasn't joined).
        missing = [c for c in cols if c not in df.columns]
        if missing:
            continue
        sub = coverage_by_group(
            df, by=cols, lo_col=lo_col, hi_col=hi_col,
            actual_col=actual_col, target=target,
        )
        for _, r in sub.iterrows():
            rows.append({
                "level": "+".join(cols),
                "key": "+".join(str(r[c]) for c in cols),
                "n": int(r["n"]),
                "target_coverage": float(r["target_coverage"]),
                "empirical_coverage": float(r["empirical_coverage"]),
                "calibration_error": float(r["calibration_error"]),
                "mean_interval_width": float(r["mean_interval_width"]),
            })

    return pd.DataFrame(rows)


__all__ = [
    "overall_coverage",
    "coverage_by_group",
    "per_tail_coverage",
    "ConformalScaler",
    "build_calibration_report",
]
