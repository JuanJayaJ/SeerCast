"""Honest baseline forecasters.

Anything more sophisticated (LightGBM, deep learning) must beat these on
rolling-origin backtests. If a fancy model can't outperform the seasonal
naive, the fancy model has a bug.

Four baselines, all sharing a common protocol::

    model.fit(history)        # history is the long base table up to the origin
    model.forecast(ids,       # which series to forecast
                   origin_date,
                   horizons)  # list of step-ahead integers, e.g. [1,7,14,28]

returning a long DataFrame with columns
``id, origin_date, horizon, target_date, prediction``.

The contract: every (id in ``ids``, h in ``horizons``) pair gets exactly
one row. Missing history (e.g. an item that has never been sold) yields a
prediction of 0 rather than NaN, so downstream metrics behave.
"""

from __future__ import annotations

from typing import Iterable, Protocol, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Common protocol
# --------------------------------------------------------------------------- #


class Baseline(Protocol):
    """Structural type for baselines (and later, any forecaster).

    Implementations don't need to inherit from this; pandas/duck-typing is
    enough. It's here so the backtester's ``model_factory`` argument has a
    documented shape.
    """

    name: str

    def fit(self, history: pd.DataFrame) -> None: ...

    def forecast(
        self,
        ids: Sequence[str],
        origin_date: pd.Timestamp,
        horizons: Sequence[int],
    ) -> pd.DataFrame: ...


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


_PRED_COLUMNS: tuple[str, ...] = (
    "id",
    "origin_date",
    "horizon",
    "target_date",
    "prediction",
)


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in _PRED_COLUMNS})


def _expand_ids_and_horizons(
    ids: Sequence[str],
    origin_date: pd.Timestamp,
    horizons: Sequence[int],
) -> pd.DataFrame:
    """Cross product (id, horizon) with target_date precomputed."""
    ids_arr = np.asarray(list(ids))
    h_arr = np.asarray(list(horizons), dtype=int)
    n_ids, n_h = len(ids_arr), len(h_arr)
    out = pd.DataFrame(
        {
            "id": np.repeat(ids_arr, n_h),
            "origin_date": pd.Timestamp(origin_date),
            "horizon": np.tile(h_arr, n_ids),
        }
    )
    out["target_date"] = out["origin_date"] + pd.to_timedelta(out["horizon"], unit="D")
    return out


# --------------------------------------------------------------------------- #
# 1. Naive: forecast = last observed sales (constant across horizon)
# --------------------------------------------------------------------------- #


class NaiveBaseline:
    """Forecast = last observed sales per ``id`` at or before the origin.

    Constant across horizon. The simplest possible benchmark.
    """

    name: str = "naive"

    def __init__(self) -> None:
        self._last: pd.Series = pd.Series(dtype=float)
        self._origin_date: pd.Timestamp | None = None

    def fit(self, history: pd.DataFrame) -> None:
        if history.empty:
            self._last = pd.Series(dtype=float)
            self._origin_date = None
            return
        last_row = (
            history.sort_values(["id", "date"])
            .groupby("id", as_index=True)
            .tail(1)
            .set_index("id")
        )
        self._last = last_row["sales"].astype(float)
        self._origin_date = history["date"].max()

    def forecast(
        self,
        ids: Sequence[str],
        origin_date: pd.Timestamp,
        horizons: Sequence[int],
    ) -> pd.DataFrame:
        out = _expand_ids_and_horizons(ids, origin_date, horizons)
        preds = self._last.reindex(out["id"]).fillna(0.0).to_numpy()
        out["prediction"] = preds
        return out[list(_PRED_COLUMNS)]


# --------------------------------------------------------------------------- #
# 2. Seasonal naive (period=7): forecast = sales on same weekday last week
# --------------------------------------------------------------------------- #


class SeasonalNaiveBaseline:
    """Forecast = sales on the same weekday in the most recent 7-day window.

    For every horizon ``h``, the prediction for ``target_date`` is the
    history value on the most-recent occurrence of ``target_date.weekday()``
    within the training window. With a 7-day lookup we cover all 7 weekdays;
    longer horizons reuse the same weekday slot.
    """

    name: str = "seasonal_naive"

    def __init__(self, season: int = 7) -> None:
        self.season = season
        self._lookup: pd.Series = pd.Series(dtype=float)
        self._origin_date: pd.Timestamp | None = None

    def fit(self, history: pd.DataFrame) -> None:
        if history.empty:
            self._lookup = pd.Series(dtype=float)
            self._origin_date = None
            return
        origin_date = history["date"].max()
        cutoff = origin_date - pd.Timedelta(days=self.season - 1)
        window = history.loc[history["date"] >= cutoff, ["id", "date", "sales"]].copy()
        window["weekday"] = window["date"].dt.weekday
        # Within the season window each (id, weekday) appears at most once;
        # in case of any quirk, take the most recent.
        window = window.sort_values(["id", "weekday", "date"])
        self._lookup = (
            window.groupby(["id", "weekday"])["sales"].last().astype(float)
        )
        self._origin_date = origin_date

    def forecast(
        self,
        ids: Sequence[str],
        origin_date: pd.Timestamp,
        horizons: Sequence[int],
    ) -> pd.DataFrame:
        out = _expand_ids_and_horizons(ids, origin_date, horizons)
        out["weekday"] = out["target_date"].dt.weekday
        keys = list(zip(out["id"].to_numpy(), out["weekday"].to_numpy()))
        preds = self._lookup.reindex(keys).fillna(0.0).to_numpy()
        out["prediction"] = preds
        return out[list(_PRED_COLUMNS)]


# --------------------------------------------------------------------------- #
# 3. Moving average (window=28): forecast = mean of last 28 days, constant
# --------------------------------------------------------------------------- #


class MovingAverageBaseline:
    """Forecast = mean of the last ``window`` days per id (constant across horizon)."""

    def __init__(self, window: int = 28) -> None:
        self.window = window
        self.name = f"moving_average_{window}"
        self._mean: pd.Series = pd.Series(dtype=float)
        self._origin_date: pd.Timestamp | None = None

    def fit(self, history: pd.DataFrame) -> None:
        if history.empty:
            self._mean = pd.Series(dtype=float)
            self._origin_date = None
            return
        origin_date = history["date"].max()
        cutoff = origin_date - pd.Timedelta(days=self.window - 1)
        recent = history.loc[history["date"] >= cutoff, ["id", "sales"]]
        self._mean = recent.groupby("id")["sales"].mean().astype(float)
        self._origin_date = origin_date

    def forecast(
        self,
        ids: Sequence[str],
        origin_date: pd.Timestamp,
        horizons: Sequence[int],
    ) -> pd.DataFrame:
        out = _expand_ids_and_horizons(ids, origin_date, horizons)
        preds = self._mean.reindex(out["id"]).fillna(0.0).to_numpy()
        out["prediction"] = preds
        return out[list(_PRED_COLUMNS)]


# --------------------------------------------------------------------------- #
# 4. Seasonal moving average: average of same-weekday over last 4 weeks
# --------------------------------------------------------------------------- #


class SeasonalMovingAverageBaseline:
    """Forecast = mean of the last ``n_weeks`` same-weekday observations.

    With ``n_weeks=4`` and a daily series, each (id, weekday) cell averages
    the four most recent occurrences of that weekday inside the training
    window. This blends moving average smoothing with weekly seasonality.
    """

    def __init__(self, n_weeks: int = 4) -> None:
        self.n_weeks = n_weeks
        self.name = "seasonal_moving_average"
        self._lookup: pd.Series = pd.Series(dtype=float)
        self._origin_date: pd.Timestamp | None = None

    def fit(self, history: pd.DataFrame) -> None:
        if history.empty:
            self._lookup = pd.Series(dtype=float)
            self._origin_date = None
            return
        origin_date = history["date"].max()
        cutoff = origin_date - pd.Timedelta(days=self.n_weeks * 7 - 1)
        window = history.loc[history["date"] >= cutoff, ["id", "date", "sales"]].copy()
        window["weekday"] = window["date"].dt.weekday
        self._lookup = (
            window.groupby(["id", "weekday"])["sales"].mean().astype(float)
        )
        self._origin_date = origin_date

    def forecast(
        self,
        ids: Sequence[str],
        origin_date: pd.Timestamp,
        horizons: Sequence[int],
    ) -> pd.DataFrame:
        out = _expand_ids_and_horizons(ids, origin_date, horizons)
        out["weekday"] = out["target_date"].dt.weekday
        keys = list(zip(out["id"].to_numpy(), out["weekday"].to_numpy()))
        preds = self._lookup.reindex(keys).fillna(0.0).to_numpy()
        out["prediction"] = preds
        return out[list(_PRED_COLUMNS)]


# --------------------------------------------------------------------------- #
# Registry — used by the training script & notebook
# --------------------------------------------------------------------------- #


def all_baselines() -> dict[str, "type[Baseline] | callable"]:
    """Map of baseline name -> zero-arg factory.

    Using factories (not instances) so the backtester can always start from
    a fresh state per origin.
    """
    return {
        "naive": NaiveBaseline,
        "seasonal_naive": SeasonalNaiveBaseline,
        "moving_average_28": lambda: MovingAverageBaseline(window=28),
        "seasonal_moving_average": SeasonalMovingAverageBaseline,
    }


__all__ = [
    "Baseline",
    "NaiveBaseline",
    "SeasonalNaiveBaseline",
    "MovingAverageBaseline",
    "SeasonalMovingAverageBaseline",
    "all_baselines",
]
