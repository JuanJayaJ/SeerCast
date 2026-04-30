"""Raw M5 file loaders.

The M5 competition ships five CSVs:

* ``calendar.csv`` — one row per ``d`` (day index), with weekday, month,
  events, SNAP flags, and the Walmart fiscal week ``wm_yr_wk``.
* ``sales_train_validation.csv`` — wide format. One row per item-store, then
  ``d_1`` … ``d_1913`` columns of unit sales (validation phase).
* ``sales_train_evaluation.csv`` — same shape as the above but extended to
  ``d_1941`` (evaluation phase).
* ``sell_prices.csv`` — one row per ``store_id`` × ``item_id`` × ``wm_yr_wk``
  giving the sell price that week. Missing rows mean the item was not sold
  that week in that store.
* ``sample_submission.csv`` — submission template; not used here.

This module keeps loading **dumb on purpose**: read CSV, set dtypes, parse
``date``. Joining wide-to-long, filtering to a store, and merging prices is
done by :mod:`seercast.data.transform`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from seercast.config import (
    CALENDAR_FILE,
    RAW_DIR,
    SALES_TRAIN_EVALUATION_FILE,
    SALES_TRAIN_VALIDATION_FILE,
    SELL_PRICES_FILE,
)


# --------------------------------------------------------------------------- #
# Public dataclass returned by load_m5_raw
# --------------------------------------------------------------------------- #


@dataclass
class M5Raw:
    """Raw (un-joined) M5 frames as loaded from disk.

    Attributes
    ----------
    calendar
        One row per ``d``; columns include ``date`` (datetime64), ``wm_yr_wk``,
        ``weekday``, ``wday``, ``month``, ``year``, event names/types, and
        ``snap_CA``/``snap_TX``/``snap_WI``.
    sales
        Wide-format sales. Columns: ``id``, ``item_id``, ``dept_id``,
        ``cat_id``, ``store_id``, ``state_id``, then ``d_1``…``d_N``.
    prices
        Long-format prices. Columns: ``store_id``, ``item_id``, ``wm_yr_wk``,
        ``sell_price``.
    """

    calendar: pd.DataFrame
    sales: pd.DataFrame
    prices: pd.DataFrame


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _resolve_data_dir(data_dir: str | Path | None) -> Path:
    """Return ``data_dir/raw`` if a project root is given, else ``raw_dir``.

    Accepts:
    * ``None`` → use the package default (``<repo>/data/raw``).
    * a path that *contains* a ``raw`` subdir (e.g. the project ``data/`` dir).
    * a path that *is* the raw dir (e.g. when files are stored elsewhere).
    """
    if data_dir is None:
        return RAW_DIR
    p = Path(data_dir)
    if (p / "raw").exists():
        return p / "raw"
    return p


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected M5 file not found: {path}. "
            "Place the raw competition CSVs in data/raw/."
        )
    return path


# --------------------------------------------------------------------------- #
# Individual loaders
# --------------------------------------------------------------------------- #


def load_calendar(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load ``calendar.csv`` and parse ``date`` as datetime64.

    Categorical columns (``weekday``, event names/types) are kept as ``object``
    here; promotion to ``category`` happens at feature-engineering time so
    upstream code can decide on the dtype budget.

    Parameters
    ----------
    data_dir
        Project ``data`` dir, the raw dir itself, or ``None`` for the default.

    Returns
    -------
    pandas.DataFrame
        With ``date: datetime64[ns]`` and the rest as native CSV-inferred dtypes.
    """
    raw_dir = _resolve_data_dir(data_dir)
    path = _require(raw_dir / CALENDAR_FILE)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    return df


def load_sales(
    data_dir: str | Path | None = None,
    use_evaluation: bool = False,
) -> pd.DataFrame:
    """Load wide-format sales (``sales_train_validation`` or ``_evaluation``).

    Parameters
    ----------
    data_dir
        See :func:`load_calendar`.
    use_evaluation
        If ``True``, load ``sales_train_evaluation.csv`` (extends 28 days
        further). Otherwise load ``sales_train_validation.csv``.

    Returns
    -------
    pandas.DataFrame
        Wide format: id columns + ``d_1``…``d_N``.
    """
    raw_dir = _resolve_data_dir(data_dir)
    fname = SALES_TRAIN_EVALUATION_FILE if use_evaluation else SALES_TRAIN_VALIDATION_FILE
    path = _require(raw_dir / fname)
    return pd.read_csv(path)


def load_prices(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load ``sell_prices.csv``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``store_id``, ``item_id``, ``wm_yr_wk`` (int), ``sell_price`` (float).
    """
    raw_dir = _resolve_data_dir(data_dir)
    path = _require(raw_dir / SELL_PRICES_FILE)
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Bundled loader
# --------------------------------------------------------------------------- #


def load_m5_raw(
    data_dir: str | Path | None = None,
    use_evaluation: bool = False,
) -> M5Raw:
    """Load all three M5 frames and return them in an :class:`M5Raw` bundle.

    This is a convenience wrapper. Prefer the individual loaders if you only
    need one frame.

    Examples
    --------
    >>> raw = load_m5_raw("data")  # data/raw/* must exist
    >>> raw.calendar.shape, raw.sales.shape, raw.prices.shape  # doctest: +SKIP
    """
    return M5Raw(
        calendar=load_calendar(data_dir),
        sales=load_sales(data_dir, use_evaluation=use_evaluation),
        prices=load_prices(data_dir),
    )


__all__ = [
    "M5Raw",
    "load_calendar",
    "load_sales",
    "load_prices",
    "load_m5_raw",
]
