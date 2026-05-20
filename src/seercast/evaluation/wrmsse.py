"""RMSSE / WRMSSE-style scoring for SeerCast.

Phase 8 — Credibility Pass.

Scope (v1, documented honestly)
--------------------------------

This is a **CA_1-scoped** RMSSE/WRMSSE implementation, not the full M5
competition score. The competition aggregates across 12 hierarchy levels
spanning 10 stores in 3 states; CA_1 alone covers exactly one store, so
the upper levels (state, total) collapse and the inter-store rollups
don't exist.

What we implement:

* **RMSSE per series** at the bottom (id) level. The scale denominator is
  the in-sample mean squared one-step difference over the training
  window — the standard M5 definition.
* **Hierarchical RMSSE** at three additional levels that DO exist on CA_1:
  ``store_id`` (a single number per model), ``cat_id`` (3 series),
  ``dept_id`` (7 series). For each non-bottom level we re-aggregate
  actuals and predictions to that level *before* computing the scaled
  squared error.
* **WRMSSE-style weighted average** across the four levels, using
  dollar-sales weights from the base table. Equal weights across levels
  (1/L) — we do NOT replicate M5's exact competition weighting.

What we explicitly do NOT do (yet):

* Cross-store / cross-state aggregations (no data — single store).
* The full competition's 12-level WRMSSE.
* Per-day-of-week scaling tricks.

Definition recap
----------------

For a series :math:`y_t`, the RMSSE on a forecast :math:`\\hat y_t` over
test window :math:`[T+1, T+h]` is::

    RMSSE = sqrt(
        mean_t  (y_t - yhat_t)^2
        ---------------------------------
        mean_s  (y_s - y_{s-1})^2   over training period (s = 2..T)
    )

The scale is computed *before* the test window starts and only on dates
where the series was active (we follow M5: skip leading zeros until first
non-zero sale; that's the "release date" filter).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# RMSSE primitives
# --------------------------------------------------------------------------- #


def _first_active_index(sales: np.ndarray) -> int:
    """Return the index of the first non-zero sale, or 0 if always zero.

    Matches M5's "release date" definition: leading zeros are pre-release
    and shouldn't inflate the scale denominator.
    """
    nz = np.flatnonzero(sales > 0)
    if nz.size == 0:
        return 0
    return int(nz[0])


def compute_scale_per_id(
    base_table: pd.DataFrame,
    *,
    training_end_date: pd.Timestamp | str,
    id_col: str = "id",
    date_col: str = "date",
    sales_col: str = "sales",
    min_history: int = 28,
) -> pd.DataFrame:
    """Compute the RMSSE denominator (mean squared 1-step diff over the
    active training window) for every series.

    Series that never become active in the training window get scale=NaN
    and will be skipped during scoring.

    Returns
    -------
    pandas.DataFrame
        Columns ``id``, ``scale``, ``n_active_train_days``.
    """
    training_end_date = pd.Timestamp(training_end_date)
    train = base_table.loc[base_table[date_col] <= training_end_date].copy()
    if train.empty:
        raise ValueError(
            f"no base table rows with {date_col} <= {training_end_date.date()}"
        )

    train = train.sort_values([id_col, date_col])
    rows: list[dict] = []
    for id_, sub in train.groupby(id_col, sort=False):
        s = sub[sales_col].to_numpy(dtype=float)
        first = _first_active_index(s)
        active = s[first:]
        if len(active) < min_history + 1:
            scale = float("nan")
            n_active = int(len(active))
        else:
            diffs = np.diff(active)
            scale = float(np.mean(diffs * diffs))
            n_active = int(len(active))
        rows.append({"id": id_, "scale": scale, "n_active_train_days": n_active})

    return pd.DataFrame(rows)


def rmsse_for_series(
    actual: Sequence[float] | np.ndarray,
    pred: Sequence[float] | np.ndarray,
    scale: float,
) -> float:
    """Series-level RMSSE = sqrt(MSE / scale).

    Returns NaN if scale is non-finite or zero, or if there are no
    matched non-NaN rows.
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    if mask.sum() == 0:
        return float("nan")
    mse = float(np.mean((a[mask] - p[mask]) ** 2))
    if not np.isfinite(scale) or scale <= 0.0:
        return float("nan")
    return float(np.sqrt(mse / scale))


# --------------------------------------------------------------------------- #
# Bottom-level rollup helper
# --------------------------------------------------------------------------- #


def _level_aggregates(
    predictions: pd.DataFrame,
    base_table: pd.DataFrame,
    *,
    level_keys: Sequence[str],
    actual_col: str = "actual",
    pred_col: str = "prediction",
    id_col: str = "id",
    date_col: str = "date",
    sales_col: str = "sales",
    target_date_col: str = "target_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate predictions and base sales to ``level_keys``.

    For predictions, we group by ``level_keys + [target_date_col]`` and
    sum ``actual`` and ``pred_col``.

    For base sales (used to compute the scale denominator at this level),
    we group by ``level_keys + [date_col]`` and sum ``sales``.

    Returns ``(level_predictions, level_base_sales)``.
    """
    if not all(k in predictions.columns for k in level_keys):
        # We need to join id -> level keys.
        join_cols = list({id_col, *level_keys})
        id_to_level = (
            base_table[join_cols].drop_duplicates(id_col).set_index(id_col)
        )
        predictions = predictions.merge(
            id_to_level, on=id_col, how="left"
        )

    pred_agg = (
        predictions
        .groupby(list(level_keys) + [target_date_col], as_index=False)
        [[actual_col, pred_col]]
        .sum()
    )
    sales_agg = (
        base_table
        .groupby(list(level_keys) + [date_col], as_index=False)
        [sales_col]
        .sum()
    )
    return pred_agg, sales_agg


def _scale_per_group(
    sales_agg: pd.DataFrame,
    *,
    level_keys: Sequence[str],
    training_end_date: pd.Timestamp,
    date_col: str = "date",
    sales_col: str = "sales",
    min_history: int = 28,
) -> pd.DataFrame:
    """Same as :func:`compute_scale_per_id`, but on already-aggregated
    sales at a higher hierarchy level."""
    train = sales_agg.loc[sales_agg[date_col] <= training_end_date]
    if train.empty:
        raise ValueError("no training-period sales after aggregation.")

    train = train.sort_values(list(level_keys) + [date_col])
    rows: list[dict] = []
    for keys, sub in train.groupby(list(level_keys)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        s = sub[sales_col].to_numpy(dtype=float)
        first = _first_active_index(s)
        active = s[first:]
        if len(active) < min_history + 1:
            scale = float("nan")
        else:
            diffs = np.diff(active)
            scale = float(np.mean(diffs * diffs))
        row = dict(zip(level_keys, keys))
        row["scale"] = scale
        row["n_active_train_days"] = int(len(active))
        rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Top-level scorers
# --------------------------------------------------------------------------- #


def score_rmsse_at_level(
    predictions: pd.DataFrame,
    base_table: pd.DataFrame,
    *,
    level_keys: Sequence[str],
    training_end_date: pd.Timestamp | str,
    actual_col: str = "actual",
    pred_col: str = "prediction",
    id_col: str = "id",
    date_col: str = "date",
    sales_col: str = "sales",
    target_date_col: str = "target_date",
    min_history: int = 28,
) -> pd.DataFrame:
    """Compute RMSSE per series at a given hierarchy level (e.g. ``id``,
    ``cat_id``, ``dept_id``, ``store_id``).

    Returns one row per series at that level with columns:
    ``*level_keys``, ``scale``, ``mse``, ``n``, ``rmsse``,
    ``dollar_sales_weight``.

    ``dollar_sales_weight`` is sum(sales * sell_price) over the training
    period at that level. When ``sell_price`` is missing in ``base_table``
    we fall back to sum(sales) (unit-sales weight).
    """
    training_end_date = pd.Timestamp(training_end_date)
    level_keys = list(level_keys)

    pred_agg, sales_agg = _level_aggregates(
        predictions, base_table,
        level_keys=level_keys, actual_col=actual_col, pred_col=pred_col,
        id_col=id_col, date_col=date_col, sales_col=sales_col,
        target_date_col=target_date_col,
    )

    scales = _scale_per_group(
        sales_agg, level_keys=level_keys,
        training_end_date=training_end_date,
        date_col=date_col, sales_col=sales_col, min_history=min_history,
    )

    # Dollar-sales weights at this level.
    train_window = base_table.loc[base_table[date_col] <= training_end_date].copy()
    if "sell_price" in train_window.columns:
        train_window["dollar"] = train_window[sales_col] * train_window["sell_price"]
        train_window["dollar"] = train_window["dollar"].fillna(0.0)
    else:
        train_window["dollar"] = train_window[sales_col].astype(float)
    if not all(k in train_window.columns for k in level_keys):
        join_cols = list({id_col, *level_keys})
        id_to_level = (
            base_table[join_cols].drop_duplicates(id_col).set_index(id_col)
        )
        train_window = train_window.merge(
            id_to_level, left_on=id_col, right_index=True, how="left",
        )
    dollar_weights = (
        train_window.groupby(level_keys, as_index=False)["dollar"].sum()
        .rename(columns={"dollar": "dollar_sales_weight"})
    )

    # Score per group.
    rows: list[dict] = []
    pred_grouped = pred_agg.groupby(level_keys)
    for keys, sub in pred_grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(level_keys, keys))
        a = sub[actual_col].to_numpy(dtype=float)
        p = sub[pred_col].to_numpy(dtype=float)
        mask = np.isfinite(a) & np.isfinite(p)
        if mask.sum() == 0:
            mse = float("nan"); n = 0
        else:
            mse = float(np.mean((a[mask] - p[mask]) ** 2))
            n = int(mask.sum())

        scale_row = scales
        for k, v in key_dict.items():
            scale_row = scale_row[scale_row[k] == v]
        scale = float(scale_row["scale"].iloc[0]) if len(scale_row) else float("nan")

        rmsse = (
            float(np.sqrt(mse / scale))
            if (np.isfinite(mse) and np.isfinite(scale) and scale > 0)
            else float("nan")
        )

        row = dict(key_dict)
        row.update({"scale": scale, "mse": mse, "n": n, "rmsse": rmsse})
        rows.append(row)

    scored = pd.DataFrame(rows)
    if not scored.empty:
        scored = scored.merge(dollar_weights, on=level_keys, how="left")
        scored["dollar_sales_weight"] = scored["dollar_sales_weight"].fillna(0.0)
    return scored


def weighted_rmsse(level_scores: pd.DataFrame, *, weight_col: str = "dollar_sales_weight") -> float:
    """Dollar-sales-weighted mean RMSSE across rows of a level.

    Returns NaN if every row has NaN RMSSE or total weight is zero.
    """
    df = level_scores.dropna(subset=["rmsse"])
    if df.empty:
        return float("nan")
    w = df[weight_col].to_numpy(dtype=float)
    if not np.all(np.isfinite(w)) or w.sum() == 0.0:
        # fall back to unweighted mean.
        return float(df["rmsse"].mean())
    return float(np.sum(df["rmsse"].to_numpy() * w) / w.sum())


@dataclass
class HierarchicalScore:
    model: str
    level: str
    n_series: int
    weighted_rmsse: float
    unweighted_rmsse: float


def hierarchical_rmsse(
    predictions: pd.DataFrame,
    base_table: pd.DataFrame,
    *,
    training_end_date: pd.Timestamp | str,
    model_name: str = "model",
    levels: Iterable[Sequence[str]] = (
        ("store_id",),
        ("cat_id",),
        ("dept_id",),
        ("id",),
    ),
    actual_col: str = "actual",
    pred_col: str = "prediction",
    id_col: str = "id",
    date_col: str = "date",
    sales_col: str = "sales",
    target_date_col: str = "target_date",
    min_history: int = 28,
) -> tuple[pd.DataFrame, float]:
    """Score a single model across several hierarchy levels and return:

    1. ``per_level`` DataFrame with one row per (level), columns
       ``level``, ``n_series``, ``weighted_rmsse``, ``unweighted_rmsse``.
    2. ``wrmsse`` scalar: equal-weight mean of ``weighted_rmsse`` across
       the implemented levels.

    The function makes the CA_1 scope explicit by including ``store_id``
    in the default levels even though there is only one store in scope —
    the resulting row is just the dollar-sales-weighted single number,
    which is also the natural "store total" RMSSE.
    """
    per_level_rows: list[dict] = []
    for level_keys in levels:
        scored = score_rmsse_at_level(
            predictions, base_table,
            level_keys=level_keys,
            training_end_date=training_end_date,
            actual_col=actual_col, pred_col=pred_col,
            id_col=id_col, date_col=date_col, sales_col=sales_col,
            target_date_col=target_date_col,
            min_history=min_history,
        )
        n_series = int(len(scored))
        wmean = weighted_rmsse(scored) if n_series else float("nan")
        umean = (
            float(scored["rmsse"].dropna().mean())
            if (n_series and scored["rmsse"].notna().any())
            else float("nan")
        )
        per_level_rows.append({
            "model": model_name,
            "level": "+".join(level_keys),
            "n_series": n_series,
            "weighted_rmsse": wmean,
            "unweighted_rmsse": umean,
        })

    per_level = pd.DataFrame(per_level_rows)
    valid = per_level["weighted_rmsse"].dropna()
    wrmsse = float(valid.mean()) if not valid.empty else float("nan")
    return per_level, wrmsse


__all__ = [
    "compute_scale_per_id",
    "rmsse_for_series",
    "score_rmsse_at_level",
    "weighted_rmsse",
    "hierarchical_rmsse",
    "HierarchicalScore",
]
