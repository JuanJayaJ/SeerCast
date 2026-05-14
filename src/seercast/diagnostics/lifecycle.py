"""Product lifecycle diagnostics.

M5 has many items that didn't exist for the entire calendar. Pre-launch
rows look like ``sales == 0`` with ``sell_price == NaN`` and zero-valued
rolling features. Treating them as "demand=0" rather than "doesn't
exist yet" almost certainly distorts both training and evaluation.

This module quantifies the problem before we change anything:

* :func:`lifecycle_summary` -- per-id first/last sale dates, days
  active, pre-launch days, and the share of rows that are pre-launch.
* :func:`is_active_at_origin` -- per (origin_date, id), is the item
  "active" yet (has it ever had a non-zero sale by origin_date)?

Nothing in this module modifies training data. The Step 1 diagnostics
just *measures* the lifecycle problem. Whether to filter pre-launch
rows from training is a later experimental question.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


_LIFECYCLE_COLUMNS: tuple[str, ...] = (
    "id",
    "first_sale_date",
    "last_sale_date",
    "n_calendar_days",
    "n_active_days",
    "pre_launch_days",
    "pre_launch_share",
    "any_sale",
)


def lifecycle_summary(base_table: pd.DataFrame, *, sales_col: str = "sales") -> pd.DataFrame:
    """Per-id lifecycle summary across the whole base table.

    Parameters
    ----------
    base_table
        Long base table with ``id``, ``date``, ``sales_col``.
    sales_col
        Column to read sales from.

    Returns
    -------
    pandas.DataFrame
        One row per id with columns from :data:`_LIFECYCLE_COLUMNS`.
        ``first_sale_date`` / ``last_sale_date`` are ``NaT`` for items
        that never recorded a non-zero sale.
    """
    needed = {"id", "date", sales_col}
    missing = needed - set(base_table.columns)
    if missing:
        raise KeyError(f"lifecycle_summary needs columns {needed}; missing: {missing}")

    cols = ["id", "date", sales_col]
    df = base_table.loc[:, cols].dropna(subset=["date"]).copy()
    if df.empty:
        return pd.DataFrame(columns=list(_LIFECYCLE_COLUMNS))

    # First and last sale per id (sales > 0 only).
    nonzero = df.loc[df[sales_col] > 0]
    first_sale = (
        nonzero.groupby("id", sort=False)["date"].min().rename("first_sale_date")
    )
    last_sale = (
        nonzero.groupby("id", sort=False)["date"].max().rename("last_sale_date")
    )

    # Per-id calendar coverage.
    per_id = df.groupby("id", sort=False)["date"].agg(
        n_calendar_days="count",
        cal_min="min",
        cal_max="max",
    )

    out = per_id.join(first_sale).join(last_sale).reset_index()
    out["any_sale"] = out["first_sale_date"].notna()

    # Pre-launch days = number of calendar rows BEFORE the item's first sale.
    # For items with any_sale=False, pre_launch_days = n_calendar_days (whole
    # history is pre-launch by this definition; no first sale ever).
    def _prelaunch(row) -> int:
        if not row["any_sale"]:
            return int(row["n_calendar_days"])
        # Count rows in df where id == this id and date < first_sale_date.
        # We do this via the merged frame for vectorization below; this
        # callable is only a fallback.
        return 0

    # Vectorized version: merge first_sale_date back onto df and count rows
    # where date < first_sale_date per id.
    df = df.merge(first_sale.reset_index(), on="id", how="left")
    df["is_pre_launch"] = df["first_sale_date"].notna() & (df["date"] < df["first_sale_date"])
    df.loc[df["first_sale_date"].isna(), "is_pre_launch"] = True
    prelaunch_count = df.groupby("id", sort=False)["is_pre_launch"].sum().rename("pre_launch_days")
    out = out.merge(prelaunch_count.reset_index(), on="id", how="left")
    out["pre_launch_days"] = out["pre_launch_days"].fillna(0).astype(int)
    out["n_active_days"] = (out["n_calendar_days"] - out["pre_launch_days"]).astype(int)
    out["pre_launch_share"] = out["pre_launch_days"] / out["n_calendar_days"].clip(lower=1)

    # Reorder.
    return out[list(_LIFECYCLE_COLUMNS)]


# --------------------------------------------------------------------------- #
# Per-origin "is active" flag
# --------------------------------------------------------------------------- #


def is_active_at_origin(
    base_table: pd.DataFrame,
    origin_dates: Sequence[pd.Timestamp],
    *,
    sales_col: str = "sales",
) -> pd.DataFrame:
    """Per (origin_date, id): has the item recorded a non-zero sale by origin_date?

    Returns one row per (origin, id) with columns ``origin_date``, ``id``,
    ``first_sale_date_at_origin`` (NaT if no sale yet), ``is_active`` (bool).
    """
    needed = {"id", "date", sales_col}
    missing = needed - set(base_table.columns)
    if missing:
        raise KeyError(f"is_active_at_origin needs columns {needed}; missing: {missing}")

    base_sorted = base_table.loc[base_table["date"].notna(), ["id", "date", sales_col]]
    nonzero = base_sorted.loc[base_sorted[sales_col] > 0]

    rows: list[dict] = []
    for origin in origin_dates:
        origin = pd.Timestamp(origin)
        eligible = nonzero.loc[nonzero["date"] <= origin]
        first = (
            eligible.groupby("id", sort=False)["date"].min()
            if not eligible.empty
            else pd.Series(dtype="datetime64[ns]")
        )
        # Iterate over ALL ids present in the base table so inactive items
        # produce explicit is_active=False rows (rather than missing rows).
        all_ids = base_sorted["id"].unique()
        first_by_id = first.to_dict()
        for id_ in all_ids:
            fs = first_by_id.get(id_, pd.NaT)
            rows.append({
                "origin_date": origin,
                "id": id_,
                "first_sale_date_at_origin": fs,
                "is_active": bool(pd.notna(fs)),
            })

    out = pd.DataFrame(rows, columns=["origin_date", "id", "first_sale_date_at_origin", "is_active"])
    return out


__all__ = [
    "lifecycle_summary",
    "is_active_at_origin",
]
