"""Lifecycle and price-availability features.

Captures product lifecycle state without using future information:

* ``has_price`` — was today's price recorded?
* ``days_since_first_sale`` / ``days_since_last_sale`` — number of days
  from the row's date back to the relevant sale event. Sales-based
  features use a per-id ``shift(1)`` so the row at date ``t`` never sees
  the sale on day ``t`` (same leakage convention as
  :mod:`seercast.features.demand_features`).
* ``days_since_first_price`` / ``days_since_last_price`` — analogous for
  prices, but **including** today (today's price is known at origin
  time, just like ``sell_price`` itself).
* ``pre_first_sale_flag`` / ``pre_first_price_flag`` — 1 before the
  first event ever observed, 0 afterwards.
* ``is_active_after_first_sale`` / ``is_active_after_first_price`` —
  inverse of the pre-flags. Kept as explicit columns so LightGBM splits
  can use either polarity without computing the negation.

The motivation: M5 has ~19% missing-price rate, and inspection on real
CA_1 suggests a substantial share is pre-launch / discontinued state
rather than true "demand=0" rows. Without lifecycle context, a model
trained on aggregate loss tends to underforecast active items once they
launch, because their early history was dominated by structural zeros.
This module gives the model that context as explicit features.

Nothing here filters or removes rows. We're surfacing information, not
making editorial decisions about training data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Columns this module emits. Used by the supervised-table schema and the
# validation tolerances.
LIFECYCLE_FEATURE_NAMES: tuple[str, ...] = (
    "has_price",
    "days_since_first_sale",
    "days_since_first_price",
    "days_since_last_sale",
    "days_since_last_price",
    "is_active_after_first_sale",
    "is_active_after_first_price",
    "pre_first_sale_flag",
    "pre_first_price_flag",
)


def _cummin_dates(dates: pd.Series, by: pd.Series) -> pd.Series:
    """Per-group running minimum of a datetime64 Series, skipping NaT.

    pandas' ``Series.cummin`` works on datetime64 dtypes but treats NaT as a
    valid value in older versions; we coerce to int64 (epoch ns), use the
    integer cummin, and coerce back. This is bulletproof across versions.
    """
    ints = dates.astype("int64")
    # NaT becomes int64.min; map it to a sentinel that never wins cummin.
    nat_mask = dates.isna()
    sentinel = np.iinfo(np.int64).max
    ints = ints.where(~nat_mask, sentinel)
    cum = ints.groupby(by, sort=False).cummin()
    cum = cum.where(cum != sentinel, np.iinfo(np.int64).min)
    out = pd.to_datetime(cum, unit="ns")
    out = out.where(cum != np.iinfo(np.int64).min, pd.NaT)
    return out


def _cummax_dates(dates: pd.Series, by: pd.Series) -> pd.Series:
    """Per-group running maximum of a datetime64 Series, skipping NaT."""
    ints = dates.astype("int64")
    nat_mask = dates.isna()
    sentinel = np.iinfo(np.int64).min
    ints = ints.where(~nat_mask, sentinel)
    cum = ints.groupby(by, sort=False).cummax()
    cum = cum.where(cum != sentinel, np.iinfo(np.int64).min)
    out = pd.to_datetime(cum, unit="ns")
    out = out.where(cum != np.iinfo(np.int64).min, pd.NaT)
    return out


def add_lifecycle_features(
    df: pd.DataFrame,
    *,
    sales_col: str = "sales",
    price_col: str = "sell_price",
    date_col: str = "date",
    sort: bool = False,
) -> pd.DataFrame:
    """Add lifecycle / price-availability features in place. Returns ``df``.

    Parameters
    ----------
    df
        Long table with ``id``, ``date``, ``sales``, ``sell_price``.
    sales_col, price_col, date_col
        Override column names if needed.
    sort
        Sort by ``(id, date)`` first. Default False (the supervised
        builder already sorts upstream).

    Returns
    -------
    pandas.DataFrame
        ``df`` with the columns from :data:`LIFECYCLE_FEATURE_NAMES` added.

    Notes
    -----
    Leakage rules:

    * Sales-based features use ``df.groupby('id')[date_col].where(sales>0).shift(1)``,
      so the row at date ``t`` only ever sees sales at dates ``< t``.
    * Price-based features include today's price (today's price is known
      at origin time -- the retailer chooses price, so ``sell_price[t]``
      is a known input the same way ``sell_price`` itself is).
    """
    required = {"id", date_col, sales_col, price_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"add_lifecycle_features needs columns {required}; missing: {missing}")

    if sort:
        df.sort_values(["id", date_col], kind="stable", inplace=True)

    # --------------------- has_price (today is known) --------------------- #
    df["has_price"] = df[price_col].notna().astype("int8")

    # ----- Sales-based: per-id shift(1) so today's sale is excluded ------- #
    is_sale = df[sales_col] > 0
    sale_dates = df[date_col].where(is_sale)            # NaT where no sale today
    sale_dates_shifted = sale_dates.groupby(df["id"], sort=False).shift(1)
    first_sale_date = _cummin_dates(sale_dates_shifted, df["id"])
    last_sale_date = _cummax_dates(sale_dates_shifted, df["id"])

    df["days_since_first_sale"] = (
        (df[date_col] - first_sale_date).dt.days.astype("float32")
    )
    df["days_since_last_sale"] = (
        (df[date_col] - last_sale_date).dt.days.astype("float32")
    )
    df["pre_first_sale_flag"] = first_sale_date.isna().astype("int8")
    df["is_active_after_first_sale"] = (~first_sale_date.isna()).astype("int8")

    # ----- Price-based: today INCLUDED (price is known at origin) --------- #
    price_dates = df[date_col].where(df[price_col].notna())
    first_price_date = _cummin_dates(price_dates, df["id"])
    last_price_date = _cummax_dates(price_dates, df["id"])

    df["days_since_first_price"] = (
        (df[date_col] - first_price_date).dt.days.astype("float32")
    )
    df["days_since_last_price"] = (
        (df[date_col] - last_price_date).dt.days.astype("float32")
    )
    df["pre_first_price_flag"] = first_price_date.isna().astype("int8")
    df["is_active_after_first_price"] = (~first_price_date.isna()).astype("int8")

    return df


def lifecycle_feature_columns() -> list[str]:
    """Return the canonical list of lifecycle feature column names."""
    return list(LIFECYCLE_FEATURE_NAMES)


__all__ = [
    "LIFECYCLE_FEATURE_NAMES",
    "add_lifecycle_features",
    "lifecycle_feature_columns",
]
