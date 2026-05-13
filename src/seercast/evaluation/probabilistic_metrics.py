"""Probabilistic forecast metrics.

Definitions (per the project charter):

* **Pinball loss** for a quantile ``q``::

      L_q(y, yhat) = max(q * (y - yhat), (q - 1) * (y - yhat))

  Lower is better. The mean pinball loss over a held-out set is the natural
  score for a quantile model -- it is consistent for the q-quantile, just
  as RMSE is for the mean.

* **Coverage** of an interval ``[lo, hi]``::

      coverage = mean(lo <= y <= hi)

  For ``[p10, p90]`` the *target* coverage is 80%. Below ~75% suggests the
  model is overconfident; above ~90% suggests it is too conservative.

* **Interval width** = mean(hi - lo). The average uncertainty band size.

* **Relative interval width** = mean(hi - lo) / mean(|y|). Width
  normalized by the typical magnitude of the actuals -- useful when
  comparing across categories with different scale.

* **Quantile crossing rate** = mean( (p10 > p50) | (p50 > p90) ). Should be
  near zero. Independently-trained quantile models can produce crossings
  on individual rows; we measure the rate before fix-up and apply a
  row-wise sort to enforce the monotonic constraint downstream.

The :func:`score_quantile_by_group` helper produces a tidy summary frame
(one row per group) with all of the above plus row counts.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Scalar metrics
# --------------------------------------------------------------------------- #


def pinball_loss(y_true: object, y_pred: object, quantile: float) -> float:
    """Mean pinball loss at the given quantile."""
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1); got {quantile}")
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    diff = yt - yp
    loss = np.maximum(quantile * diff, (quantile - 1.0) * diff)
    return float(np.mean(loss))


def coverage(y_true: object, lo: object, hi: object) -> float:
    """Empirical coverage of the interval ``[lo, hi]`` (fraction in [0, 1])."""
    yt = np.asarray(y_true, dtype=float)
    lo_a = np.asarray(lo, dtype=float)
    hi_a = np.asarray(hi, dtype=float)
    return float(np.mean((yt >= lo_a) & (yt <= hi_a)))


def interval_width(lo: object, hi: object) -> float:
    """Mean width of the interval (``hi - lo``)."""
    lo_a = np.asarray(lo, dtype=float)
    hi_a = np.asarray(hi, dtype=float)
    return float(np.mean(hi_a - lo_a))


def relative_interval_width(lo: object, hi: object, y_true: object) -> float:
    """Aggregate relative interval width: ``mean(hi - lo) / mean(|y|)``.

    This is the *aggregate* form -- mean width divided by mean magnitude --
    not a per-row ratio averaged afterwards. We use this form deliberately:
    M5 has heavy intermittent demand with many zero-sales rows, and a
    per-row ratio would explode (or be NaN) on zero-actual rows. Dividing
    aggregates first is more stable on intermittent series and still
    behaves as "interval width relative to typical magnitude".

    Returns NaN if ``mean(|y|) == 0`` (no signal to normalize against).
    """
    yt = np.asarray(y_true, dtype=float)
    s = float(np.mean(np.abs(yt)))
    if s == 0.0:
        return float("nan")
    return float(interval_width(lo, hi) / s)


def crossing_rate(p10: object, p50: object, p90: object) -> float:
    """Fraction of rows where the quantile order is violated."""
    p10_a = np.asarray(p10, dtype=float)
    p50_a = np.asarray(p50, dtype=float)
    p90_a = np.asarray(p90, dtype=float)
    crossed = (p10_a > p50_a) | (p50_a > p90_a) | (p10_a > p90_a)
    return float(np.mean(crossed))


# --------------------------------------------------------------------------- #
# Crossing fix-up
# --------------------------------------------------------------------------- #


def fix_quantile_crossings(
    df: pd.DataFrame,
    quantile_columns: Sequence[str] = ("p10", "p50", "p90"),
) -> pd.DataFrame:
    """Sort the quantile columns row-wise so they are monotonically non-decreasing.

    Independent quantile models can produce ``p10 > p50`` (or any other
    out-of-order pair) on individual rows, especially when training data is
    sparse. This is a v1 fix: we sort the values per row in ascending order
    and reassign them to the columns in the order ``quantile_columns`` was
    given. It is conservative (worst case it re-labels a few rows) and
    monotonicity-preserving.

    Returns a new DataFrame; ``df`` is not mutated.
    """
    out = df.copy()
    cols = list(quantile_columns)
    if not cols:
        return out
    missing = [c for c in cols if c not in out.columns]
    if missing:
        raise KeyError(f"missing quantile columns: {missing}")
    arr = out[cols].to_numpy(dtype=float)
    arr.sort(axis=1)  # row-wise ascending
    out[cols] = arr
    return out


# --------------------------------------------------------------------------- #
# Bundled metrics
# --------------------------------------------------------------------------- #


def all_quantile_metrics(
    df: pd.DataFrame,
    *,
    quantile_columns: Sequence[str] = ("p10", "p50", "p90"),
    quantile_levels: Sequence[float] = (0.10, 0.50, 0.90),
    actual_col: str = "actual",
) -> dict[str, float]:
    """All quantile metrics over the whole frame, returned as a flat dict.

    Keys:
    * ``pinball_<col>`` for each quantile column
    * ``coverage_<lo>_<hi>`` for the lo/hi pair (e.g. ``coverage_p10_p90``)
    * ``interval_width_<lo>_<hi>``
    * ``relative_interval_width_<lo>_<hi>``
    * ``crossing_rate``  (computed across the supplied quantile columns)
    """
    if len(quantile_columns) != len(quantile_levels):
        raise ValueError("quantile_columns and quantile_levels must align")
    out: dict[str, float] = {}
    y = df[actual_col]

    for col, q in zip(quantile_columns, quantile_levels):
        out[f"pinball_{col}"] = pinball_loss(y, df[col], q)

    if len(quantile_columns) >= 2:
        lo_col = quantile_columns[0]
        hi_col = quantile_columns[-1]
        lo = df[lo_col]
        hi = df[hi_col]
        cov_key = f"coverage_{lo_col}_{hi_col}"
        out[cov_key] = coverage(y, lo, hi)
        out[f"interval_width_{lo_col}_{hi_col}"] = interval_width(lo, hi)
        out[f"relative_interval_width_{lo_col}_{hi_col}"] = relative_interval_width(lo, hi, y)

    if len(quantile_columns) == 3:
        out["crossing_rate"] = crossing_rate(
            df[quantile_columns[0]], df[quantile_columns[1]], df[quantile_columns[2]]
        )

    return out


# --------------------------------------------------------------------------- #
# Group-wise scoring
# --------------------------------------------------------------------------- #


def score_quantile_by_group(
    df: pd.DataFrame,
    by: Sequence[str] | None = None,
    *,
    quantile_columns: Sequence[str] = ("p10", "p50", "p90"),
    quantile_levels: Sequence[float] = (0.10, 0.50, 0.90),
    actual_col: str = "actual",
) -> pd.DataFrame:
    """Compute the quantile metrics per group.

    Parameters
    ----------
    df
        Long predictions DataFrame containing ``actual_col`` and the
        ``quantile_columns``.
    by
        Columns to group by. ``None`` -> score the whole frame as one group.
    quantile_columns, quantile_levels
        Aligned sequences. ``("p10", "p50", "p90")`` and ``(0.1, 0.5, 0.9)``
        by default.
    actual_col
        Column with ground-truth values.

    Returns
    -------
    pandas.DataFrame
        One row per group with the metrics columns from
        :func:`all_quantile_metrics` plus ``n``.
    """
    needed = {actual_col, *quantile_columns}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"score_quantile_by_group needs columns {needed}; missing: {missing}")

    clean = df.dropna(subset=list(needed))
    if not by:
        m = all_quantile_metrics(
            clean,
            quantile_columns=quantile_columns,
            quantile_levels=quantile_levels,
            actual_col=actual_col,
        )
        m["n"] = int(len(clean))
        cols = ["n", *m.keys()]
        # Move 'n' to the front uniquely.
        return pd.DataFrame([m]).reindex(columns=["n"] + [k for k in m.keys() if k != "n"])

    by = list(by)
    rows: list[dict] = []
    for keys, sub in clean.groupby(by, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        m = all_quantile_metrics(
            sub,
            quantile_columns=quantile_columns,
            quantile_levels=quantile_levels,
            actual_col=actual_col,
        )
        m["n"] = int(len(sub))
        for col, val in zip(by, keys):
            m[col] = val
        rows.append(m)

    metric_cols = [
        *[f"pinball_{c}" for c in quantile_columns],
        f"coverage_{quantile_columns[0]}_{quantile_columns[-1]}",
        f"interval_width_{quantile_columns[0]}_{quantile_columns[-1]}",
        f"relative_interval_width_{quantile_columns[0]}_{quantile_columns[-1]}",
    ]
    if len(quantile_columns) == 3:
        metric_cols.append("crossing_rate")
    return pd.DataFrame(rows)[by + ["n"] + metric_cols].sort_values(by).reset_index(drop=True)


__all__ = [
    "pinball_loss",
    "coverage",
    "interval_width",
    "relative_interval_width",
    "crossing_rate",
    "fix_quantile_crossings",
    "all_quantile_metrics",
    "score_quantile_by_group",
]
