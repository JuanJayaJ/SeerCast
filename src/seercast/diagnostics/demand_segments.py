"""Demand-pattern classification (Syntetos-Boylan).

Industry-standard intermittent-demand classification uses two summary
statistics on each item's history:

* **ADI** -- Average Demand Interval -- the mean number of periods
  between consecutive non-zero demands. ADI of 1.0 means an item sells
  every day; large ADI means many zero days between sales.
* **CV2** -- squared coefficient of variation of the *non-zero* demand
  sizes: ``var(nonzero) / mean(nonzero)**2``. CV2 of 0 means every sale
  is the same size; large CV2 means highly variable lot sizes.

The 2x2 Syntetos-Boylan grid:

==============  ==========  ==========
                CV2 < 0.49  CV2 >= 0.49
==============  ==========  ==========
ADI <  1.32     smooth      erratic
ADI >= 1.32     intermittent  lumpy
==============  ==========  ==========

We add a fifth category:

* **zero_heavy** -- items with a zero rate >= 0.9 over the history
  window. Override applied before ADI/CV2 (because ADI/CV2 are noisy
  when there are barely any non-zero observations to work with).

Per-origin variant
------------------

For diagnostics we classify each *(origin_date, id)* pair using only
history rows with ``date <= origin_date``. That keeps the segment label
itself leakage-safe -- a label assigned at origin O cannot peek at sales
past O. The same id can move between segments at different origins
(common for new items that go from zero_heavy to intermittent as they
accumulate history).
"""

from __future__ import annotations

from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd


SEGMENTS: tuple[str, ...] = (
    "smooth",
    "intermittent",
    "erratic",
    "lumpy",
    "zero_heavy",
)


# --------------------------------------------------------------------------- #
# Single-series statistics
# --------------------------------------------------------------------------- #


def _adi_cv2(sales: np.ndarray) -> tuple[float, float, float]:
    """Return (adi, cv2, zero_rate) for a single id's sales history.

    Definitions:
    * adi = (number of periods) / (number of non-zero periods).
      Equivalent to the mean inter-arrival time of non-zero demand.
      Returns inf if there are no non-zero observations.
    * cv2 = var(nonzero) / mean(nonzero)**2. Returns 0 if there is
      exactly one non-zero observation (no variance to measure).
    * zero_rate = fraction of zero observations in the history.
    """
    n = len(sales)
    if n == 0:
        return float("inf"), 0.0, 1.0
    nonzero = sales[sales > 0]
    n_nz = len(nonzero)
    zero_rate = float(1.0 - n_nz / n)
    if n_nz == 0:
        return float("inf"), 0.0, zero_rate
    adi = float(n / n_nz)
    if n_nz == 1:
        cv2 = 0.0
    else:
        mu = float(nonzero.mean())
        if mu == 0.0:
            cv2 = 0.0
        else:
            cv2 = float(nonzero.var(ddof=0) / (mu * mu))
    return adi, cv2, zero_rate


def _classify(
    adi: float,
    cv2: float,
    zero_rate: float,
    n_history: int,
    *,
    adi_thresh: float,
    cv2_thresh: float,
    zero_heavy_rate: float,
    min_history_days: int,
) -> str:
    """Apply the classification rules. Order matters: zero_heavy first."""
    if n_history < min_history_days:
        # Not enough history to classify with confidence; treat as zero_heavy
        # since the pre-launch / very-young state behaves similarly.
        return "zero_heavy"
    if zero_rate >= zero_heavy_rate:
        return "zero_heavy"
    if adi < adi_thresh and cv2 < cv2_thresh:
        return "smooth"
    if adi >= adi_thresh and cv2 < cv2_thresh:
        return "intermittent"
    if adi < adi_thresh and cv2 >= cv2_thresh:
        return "erratic"
    return "lumpy"


# --------------------------------------------------------------------------- #
# Per-origin classification (leakage-safe)
# --------------------------------------------------------------------------- #


_SEGMENT_COLUMNS: tuple[str, ...] = (
    "origin_date",
    "id",
    "n_history_days",
    "adi_at_origin",
    "cv2_at_origin",
    "zero_rate_at_origin",
    "mean_demand_at_origin",
    "segment_at_origin",
)


def classify_demand_per_origin(
    base_table: pd.DataFrame,
    origin_dates: Sequence[pd.Timestamp],
    *,
    sales_col: str = "sales",
    adi_thresh: float = 1.32,
    cv2_thresh: float = 0.49,
    zero_heavy_rate: float = 0.9,
    min_history_days: int = 28,
) -> pd.DataFrame:
    """Per-(origin_date, id) demand segment labels using only history.

    For each origin in ``origin_dates`` and each id in ``base_table``, compute
    ADI / CV2 / zero_rate / mean_demand on the rows where
    ``date <= origin_date`` and ``sales`` is observed. Assign a
    segment via :data:`SEGMENTS`.

    Parameters
    ----------
    base_table
        Long base table with ``id``, ``date``, and ``sales_col``.
    origin_dates
        Origins to classify at. Typically the backtest origins.
    sales_col
        Column to read demand from. Default ``"sales"``.
    adi_thresh, cv2_thresh
        Syntetos-Boylan thresholds. Defaults 1.32 and 0.49 from the 2005 paper.
    zero_heavy_rate
        Items with zero rate >= this become ``zero_heavy`` regardless of ADI/CV2.
    min_history_days
        Items with fewer than this many history rows at the origin become
        ``zero_heavy`` (effectively "too young to classify reliably").

    Returns
    -------
    pandas.DataFrame
        Columns: ``origin_date, id, n_history_days, adi_at_origin,
        cv2_at_origin, zero_rate_at_origin, mean_demand_at_origin,
        segment_at_origin``. One row per (origin, id).
    """
    needed = {"id", "date", sales_col}
    missing = needed - set(base_table.columns)
    if missing:
        raise KeyError(f"classify_demand_per_origin needs columns {needed}; missing: {missing}")

    base_sorted = base_table.loc[base_table["date"].notna()].sort_values(["id", "date"])
    rows: list[dict] = []

    for origin in origin_dates:
        origin = pd.Timestamp(origin)
        history = base_sorted.loc[base_sorted["date"] <= origin, ["id", sales_col]]
        # Per-id reductions.
        grouped = history.groupby("id", sort=False)
        for id_, sub in grouped:
            sales = sub[sales_col].to_numpy(dtype=float)
            adi, cv2, zero_rate = _adi_cv2(sales)
            n_history = int(len(sales))
            nonzero = sales[sales > 0]
            mean_demand = float(nonzero.mean()) if nonzero.size > 0 else 0.0
            seg = _classify(
                adi, cv2, zero_rate, n_history,
                adi_thresh=adi_thresh, cv2_thresh=cv2_thresh,
                zero_heavy_rate=zero_heavy_rate, min_history_days=min_history_days,
            )
            rows.append({
                "origin_date": origin,
                "id": id_,
                "n_history_days": n_history,
                "adi_at_origin": adi,
                "cv2_at_origin": cv2,
                "zero_rate_at_origin": zero_rate,
                "mean_demand_at_origin": mean_demand,
                "segment_at_origin": seg,
            })

    out = pd.DataFrame(rows, columns=list(_SEGMENT_COLUMNS))
    return out


__all__ = [
    "SEGMENTS",
    "classify_demand_per_origin",
]
