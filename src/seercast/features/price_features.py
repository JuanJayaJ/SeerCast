"""Price-derived features.

Unlike demand, the sell price at the origin date is genuinely known at
forecast time -- a retailer chooses price, so today's price is a known
quantity. That's why these features (and the `sell_price` column itself)
are NOT shifted by one before being read at the origin.

Lag and pct-change features are derived from the per-id price series.
The 28-day rolling mean is an inclusive window ending at ``t``; the
`price_relative_to_28d_avg` feature is then ``sell_price / rolling_mean_28``,
which gives a "is today's price unusually high?" signal that LightGBM
can split on.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


PRICE_FEATURE_NAMES: tuple[str, ...] = (
    "price_lag_1",
    "price_lag_7",
    "price_change_1",
    "price_change_7",
    "price_pct_change_1",
    "price_pct_change_7",
    "price_rolling_mean_28",
    "price_relative_to_28d_avg",
)


def _grouped_shift(df: pd.DataFrame, col: str, periods: int) -> pd.Series:
    return df.groupby("id", sort=False)[col].shift(periods)


def _grouped_rolling_mean(
    series: pd.Series,
    by: pd.Series,
    window: int,
    min_periods: int = 1,
) -> pd.Series:
    rolled = (
        series.groupby(by, sort=False)
        .rolling(window=window, min_periods=min_periods)
        .mean()
    )
    return rolled.reset_index(level=0, drop=True)


def add_price_features(
    df: pd.DataFrame,
    *,
    price_col: str = "sell_price",
    sort: bool = False,
) -> pd.DataFrame:
    """Add price-derived features in place. Returns the same df.

    Parameters
    ----------
    df
        Long base table with ``id``, ``date``, and ``sell_price``. If you've
        already sorted in :func:`add_demand_features` you can leave
        ``sort=False`` to avoid resorting.
    price_col
        Column to derive features from. Defaults to ``"sell_price"``.
    sort
        Sort by ``(id, date)`` first. Default False (caller usually sorts).

    Returns
    -------
    pandas.DataFrame
        ``df`` with the columns from :data:`PRICE_FEATURE_NAMES` appended.

    Notes
    -----
    Missing ``sell_price`` (item not stocked that week) is preserved as NaN
    throughout. ``price_relative_to_28d_avg`` divides by the rolling mean
    with a small floor to avoid division-by-zero, but if the rolling mean
    itself is NaN (no prices in window) the result will be NaN, which
    LightGBM handles natively.
    """
    required = {"id", "date", price_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"price features need columns {required}; missing: {missing}")

    if sort:
        df.sort_values(["id", "date"], kind="stable", inplace=True)

    df["price_lag_1"] = _grouped_shift(df, price_col, 1).astype("float32")
    df["price_lag_7"] = _grouped_shift(df, price_col, 7).astype("float32")

    price = df[price_col]
    df["price_change_1"] = (price - df["price_lag_1"]).astype("float32")
    df["price_change_7"] = (price - df["price_lag_7"]).astype("float32")

    # pct-change: NaN where the lag is NaN or zero (M5 prices are positive in practice
    # but guard anyway).
    with np.errstate(divide="ignore", invalid="ignore"):
        df["price_pct_change_1"] = (
            (df["price_change_1"] / df["price_lag_1"]).astype("float32")
        )
        df["price_pct_change_7"] = (
            (df["price_change_7"] / df["price_lag_7"]).astype("float32")
        )

    df["price_rolling_mean_28"] = _grouped_rolling_mean(
        price, df["id"], window=28
    ).astype("float32")

    with np.errstate(divide="ignore", invalid="ignore"):
        df["price_relative_to_28d_avg"] = (
            (price / df["price_rolling_mean_28"]).astype("float32")
        )

    return df


def price_feature_columns() -> list[str]:
    return list(PRICE_FEATURE_NAMES)


__all__ = [
    "PRICE_FEATURE_NAMES",
    "add_price_features",
    "price_feature_columns",
]
