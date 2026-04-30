"""Row-wise calendar / event / SNAP features.

These features depend only on the date and the event/SNAP flags already
present in the base table. They have no temporal context (no lags), so
they're safe to compute at any time, including for future target dates.

The same function is used twice in the supervised pipeline:

* once at ``origin_date`` to produce features visible to the model at
  forecast time;
* once at ``target_date`` (with ``prefix="target_"``) to expose calendar /
  event / SNAP characteristics of the day being predicted, which are
  legitimately known ahead of time.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from seercast.config import SNAP_FLAG_BY_STATE


# Columns this module emits (without prefix). Useful for downstream
# schema introspection and the supervised-table column registry.
CALENDAR_FEATURE_NAMES: tuple[str, ...] = (
    "dayofweek",
    "weekofyear",
    "month",
    "year",
    "quarter",
    "dayofmonth",
    "is_weekend",
    "has_event_1",
    "has_event_2",
    "has_event",
    "is_snap_day",
)


def _has_value(series: pd.Series) -> pd.Series:
    """Return 1 where the series has a non-null, non-empty-string value."""
    if series.dtype == object:
        return series.notna() & (series.astype(str) != "")
    return series.notna()


def add_calendar_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    prefix: str = "",
    snap_state: str = "CA",
    event_name_1_col: str = "event_name_1",
    event_name_2_col: str = "event_name_2",
) -> pd.DataFrame:
    """Add calendar / event / SNAP features in place (returns the same df).

    Parameters
    ----------
    df
        Frame with at least ``date_col``, the two event-name columns, and
        the SNAP column for the chosen state. Must already be a copy if you
        don't want the input mutated; this function does not copy.
    date_col
        Column from which to derive day-of-week, month, etc.
    prefix
        Optional prefix prepended to every output column name. Use
        ``"target_"`` when applying to target-date rows so the supervised
        table can hold both origin-date and target-date variants.
    snap_state
        Two-letter state code. ``"CA"`` -> reads ``snap_CA`` -> writes
        ``is_snap_day`` (or ``f"{prefix}is_snap_day"``).

    Returns
    -------
    pandas.DataFrame
        ``df`` with the new columns appended. Columns:
        ``dayofweek``, ``weekofyear``, ``month``, ``year``, ``quarter``,
        ``dayofmonth``, ``is_weekend``, ``has_event_1``, ``has_event_2``,
        ``has_event``, ``is_snap_day``, each prefixed.
    """
    if date_col not in df.columns:
        raise KeyError(f"date column '{date_col}' missing from df")

    snap_col = SNAP_FLAG_BY_STATE.get(snap_state)
    if snap_col is None:
        raise ValueError(
            f"unknown snap_state {snap_state!r}; expected one of "
            f"{sorted(SNAP_FLAG_BY_STATE)}"
        )

    dt = df[date_col]
    if not pd.api.types.is_datetime64_any_dtype(dt):
        raise TypeError(f"{date_col} must be datetime64; got {dt.dtype}")

    p = prefix
    df[f"{p}dayofweek"] = dt.dt.weekday.astype("int8")
    # isocalendar().week returns UInt32; downcast to int16 to keep the table small.
    df[f"{p}weekofyear"] = dt.dt.isocalendar().week.astype("int16")
    df[f"{p}month"] = dt.dt.month.astype("int8")
    df[f"{p}year"] = dt.dt.year.astype("int16")
    df[f"{p}quarter"] = dt.dt.quarter.astype("int8")
    df[f"{p}dayofmonth"] = dt.dt.day.astype("int8")
    df[f"{p}is_weekend"] = (dt.dt.weekday >= 5).astype("int8")

    has_e1 = _has_value(df[event_name_1_col]) if event_name_1_col in df.columns else False
    has_e2 = _has_value(df[event_name_2_col]) if event_name_2_col in df.columns else False
    df[f"{p}has_event_1"] = (has_e1.astype("int8") if hasattr(has_e1, "astype") else 0)
    df[f"{p}has_event_2"] = (has_e2.astype("int8") if hasattr(has_e2, "astype") else 0)
    # has_event = OR of the two
    if isinstance(has_e1, pd.Series) and isinstance(has_e2, pd.Series):
        df[f"{p}has_event"] = (has_e1 | has_e2).astype("int8")
    elif isinstance(has_e1, pd.Series):
        df[f"{p}has_event"] = has_e1.astype("int8")
    elif isinstance(has_e2, pd.Series):
        df[f"{p}has_event"] = has_e2.astype("int8")
    else:
        df[f"{p}has_event"] = np.int8(0)

    if snap_col in df.columns:
        df[f"{p}is_snap_day"] = df[snap_col].fillna(0).astype("int8")
    else:
        df[f"{p}is_snap_day"] = np.int8(0)

    return df


def calendar_feature_columns(prefix: str = "") -> list[str]:
    """Return the list of calendar feature column names with optional prefix."""
    return [f"{prefix}{name}" for name in CALENDAR_FEATURE_NAMES]


__all__ = [
    "CALENDAR_FEATURE_NAMES",
    "add_calendar_features",
    "calendar_feature_columns",
]
