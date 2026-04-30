"""Historical demand features (lags + rolling stats), leakage-safe.

Convention: a feature value computed AT row ``t`` (i.e. for ``date == t``)
is allowed to depend only on ``sales`` at dates ``< t``. We enforce this
by shifting sales by one *within each id group* before any rolling stat::

    shifted = df.groupby("id")["sales"].shift(1)
    df["rolling_mean_28"] = shifted.groupby(df["id"]).rolling(28, min_periods=1).mean().values

The downstream effect: when the supervised table picks features at
``origin_date == t``, those features used only sales at ``date <= t - 1``,
so today's sales never leak into the training input.

For lag-N specifically, ``shift(N)`` already gives the value N days back, so
no additional shifting is needed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# Column names emitted by this module. Used by the supervised-table builder
# and the validation report.
DEMAND_FEATURE_NAMES: tuple[str, ...] = (
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_lag_56",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_mean_56",
    "rolling_std_7",
    "rolling_std_28",
    "rolling_min_28",
    "rolling_max_28",
    "zero_sales_rate_28",
    "nonzero_sales_count_28",
)


_LAGS: tuple[int, ...] = (1, 7, 14, 28, 56)


def _grouped_shift(df: pd.DataFrame, col: str, periods: int) -> pd.Series:
    """Per-id shift that keeps the original index alignment."""
    return df.groupby("id", sort=False)[col].shift(periods)


def _grouped_rolling(
    series: pd.Series,
    by: pd.Series,
    window: int,
    func: str,
    min_periods: int = 1,
) -> pd.Series:
    """Per-group rolling aggregation that returns a series aligned with ``series``.

    ``series`` is assumed to already be lag-shifted by 1 (the leakage guard).
    """
    grouped = series.groupby(by, sort=False)
    rolled = grouped.rolling(window=window, min_periods=min_periods).agg(func)
    # rolled has a 2-level MultiIndex (group_key, original_index); drop the group level.
    return rolled.reset_index(level=0, drop=True)


def add_demand_features(
    df: pd.DataFrame,
    *,
    sales_col: str = "sales",
    sort: bool = True,
) -> pd.DataFrame:
    """Add historical demand features in place. Returns the same df.

    Parameters
    ----------
    df
        Long base table with at least ``id``, ``date``, and ``sales``. Will
        be sorted by ``(id, date)`` if ``sort=True``.
    sales_col
        Column to derive features from. Defaults to ``"sales"``.
    sort
        If True (default), sort by ``(id, date)`` before computing.

    Returns
    -------
    pandas.DataFrame
        ``df`` with the new columns appended (see :data:`DEMAND_FEATURE_NAMES`).

    Notes
    -----
    All rolling stats use the *shifted* sales series (``shift(1)`` per id) so
    the row at date ``t`` never sees ``sales(t)``. Counts and rates over the
    last 28 days similarly use shifted sales.
    """
    required = {"id", "date", sales_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"demand features need columns {required}; missing: {missing}")

    if sort:
        df.sort_values(["id", "date"], kind="stable", inplace=True)

    # Lags: shift(N) per id is exactly the lag-N value.
    for lag in _LAGS:
        df[f"sales_lag_{lag}"] = _grouped_shift(df, sales_col, lag).astype("float32")

    # Shifted sales: the leakage-safe basis for all rolling stats.
    shifted = _grouped_shift(df, sales_col, 1)

    # Rolling means.
    for w in (7, 28, 56):
        df[f"rolling_mean_{w}"] = (
            _grouped_rolling(shifted, df["id"], w, "mean").astype("float32")
        )

    # Rolling std (windows 7, 28).
    for w in (7, 28):
        df[f"rolling_std_{w}"] = (
            _grouped_rolling(shifted, df["id"], w, "std").astype("float32")
        )

    # Rolling min/max for window 28 only.
    df["rolling_min_28"] = (
        _grouped_rolling(shifted, df["id"], 28, "min").astype("float32")
    )
    df["rolling_max_28"] = (
        _grouped_rolling(shifted, df["id"], 28, "max").astype("float32")
    )

    # Zero-sales rate and nonzero-sales count over window=28, computed from shifted.
    is_zero = shifted.eq(0).astype("float32")
    is_nonzero = shifted.gt(0).astype("float32")
    df["zero_sales_rate_28"] = (
        _grouped_rolling(is_zero, df["id"], 28, "mean").astype("float32")
    )
    df["nonzero_sales_count_28"] = (
        _grouped_rolling(is_nonzero, df["id"], 28, "sum").astype("float32")
    )

    return df


def demand_feature_columns() -> list[str]:
    """Return the canonical list of demand feature column names."""
    return list(DEMAND_FEATURE_NAMES)


__all__ = [
    "DEMAND_FEATURE_NAMES",
    "add_demand_features",
    "demand_feature_columns",
]
