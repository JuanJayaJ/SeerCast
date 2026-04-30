"""Transform raw M5 frames into a single long base table.

The pipeline is:

1. Melt the wide ``sales_train_*.csv`` so each row is a single
   ``(id, d)`` observation with one ``sales`` value.
2. Join the calendar on ``d`` to attach ``date``, weekday/month/year,
   event flags, SNAP flags, and ``wm_yr_wk``.
3. Join ``sell_prices`` on ``(store_id, item_id, wm_yr_wk)`` to attach
   the weekly sell price (left join: missing prices are kept as NaN, since
   they encode "item not for sale this week" and we measure that rate as a
   data-quality signal).

The output columns are the canonical schema declared in
:data:`seercast.config.BASE_TABLE_COLUMNS`. No feature engineering happens
here -- that's :mod:`seercast.features`. Keeping load -> transform -> features
as three layers makes leakage rules easier to reason about.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from seercast.config import (
    BASE_TABLE_COLUMNS,
    DEFAULT_STORE_ID,
)
from seercast.data.load_m5 import M5Raw, load_m5_raw


# --------------------------------------------------------------------------- #
# Wide-to-long melt
# --------------------------------------------------------------------------- #


_ID_COLS: tuple[str, ...] = (
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
)


def melt_sales(
    sales_wide: pd.DataFrame,
    store_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Melt wide M5 sales into long format.

    Parameters
    ----------
    sales_wide
        Output of :func:`seercast.data.load_m5.load_sales`. Must contain the
        id columns (``id``, ``item_id``, ``dept_id``, ``cat_id``, ``store_id``,
        ``state_id``) and a series of ``d_<N>`` columns.
    store_ids
        Optional iterable of ``store_id`` values to keep. Filtering before
        the melt keeps memory in check on the full dataset (~58M rows).

    Returns
    -------
    pandas.DataFrame
        Columns: id columns + ``d`` (str like ``"d_1500"``) + ``sales`` (int32).
        Sorted by ``id, d`` for downstream lag/rolling stability.
    """
    df = sales_wide
    if store_ids is not None:
        store_ids = list(store_ids)
        df = df.loc[df["store_id"].isin(store_ids)].copy()

    d_cols = [c for c in df.columns if c.startswith("d_")]
    if not d_cols:
        raise ValueError("No d_<N> columns found in sales frame; is this the wide CSV?")

    long_df = df.melt(
        id_vars=list(_ID_COLS),
        value_vars=d_cols,
        var_name="d",
        value_name="sales",
    )
    # M5 ships sales as small non-negative ints; downcast to save ~75% memory.
    long_df["sales"] = long_df["sales"].astype(np.int32)
    # Sort once here; lag/rolling features depend on stable order.
    long_df = long_df.sort_values(["id", "d"]).reset_index(drop=True)
    return long_df


# --------------------------------------------------------------------------- #
# Join with calendar + prices
# --------------------------------------------------------------------------- #


def _attach_calendar(sales_long: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Left-join calendar on ``d``.

    LEFT, not INNER: an inner join would silently drop sales rows whose ``d``
    is missing from the calendar, which is exactly the kind of join bug we
    want to *see*. With a left join, missing-calendar rows survive with NaT
    ``date``, and :func:`seercast.data.validation.validate_base_table` flags
    them via the missing-date count.
    """
    cal_cols = [
        "d",
        "date",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    cal = calendar[cal_cols]
    merged = sales_long.merge(cal, on="d", how="left", validate="many_to_one")
    return merged


def _attach_prices(df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Left-join sell_prices on (store_id, item_id, wm_yr_wk).

    Left join (not inner) because a missing price genuinely means the item
    was not on sale that week in that store, and dropping those rows would
    bias the sales distribution. We measure the missing rate downstream.
    """
    merged = df.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )
    return merged


def build_base_table(
    raw: M5Raw | None = None,
    data_dir: str | Path | None = None,
    use_evaluation: bool = False,
    store_ids: Iterable[str] | None = (DEFAULT_STORE_ID,),
) -> pd.DataFrame:
    """Build the joined long base table.

    This is the single entry point notebooks/scripts should call. It can
    either accept a pre-loaded :class:`M5Raw` (useful for tests) or load
    everything from disk itself.

    Parameters
    ----------
    raw
        Pre-loaded M5 frames. If ``None``, loaded from ``data_dir``.
    data_dir
        Passed to :func:`seercast.data.load_m5.load_m5_raw` when ``raw`` is None.
    use_evaluation
        If ``True``, load the longer ``sales_train_evaluation`` file.
    store_ids
        Stores to keep. Defaults to ``("CA_1",)`` to match the project charter.
        Pass ``None`` to load every store (warning: large).

    Returns
    -------
    pandas.DataFrame
        Columns are :data:`seercast.config.BASE_TABLE_COLUMNS` in canonical order.
    """
    if raw is None:
        raw = load_m5_raw(data_dir=data_dir, use_evaluation=use_evaluation)

    sales_long = melt_sales(raw.sales, store_ids=store_ids)
    with_cal = _attach_calendar(sales_long, raw.calendar)
    with_price = _attach_prices(with_cal, raw.prices)

    # Reorder to canonical schema, then sort by id+date for downstream stability.
    base = with_price[BASE_TABLE_COLUMNS].sort_values(["id", "date"]).reset_index(drop=True)
    return base


__all__ = ["melt_sales", "build_base_table"]
