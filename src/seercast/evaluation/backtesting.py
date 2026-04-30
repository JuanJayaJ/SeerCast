"""Rolling-origin backtesting.

The contract is strict:

* Train data for an origin = all rows where ``date <= origin_date``. No
  random splits. No future leakage. Origins march forward in time only.
* Forecast = next ``horizon`` days after the origin.
* Test actuals are looked up from the base table itself, since this is a
  retrospective evaluation. Rows whose target_date is past the end of the
  base table are dropped (they have NaN actuals).

Origins can be passed as M5 ``d`` integers (1500 -> "d_1500") or as
datetime-like values. Mixed lists work too.

The backtester takes a ``model_factory`` (a zero-argument callable) rather
than a model instance, so each origin gets a fresh model state — no
accidental persistence across folds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Origin coercion helpers
# --------------------------------------------------------------------------- #


def _origin_to_date(base: pd.DataFrame, origin: int | str | pd.Timestamp) -> pd.Timestamp:
    """Coerce an origin spec to the matching ``date`` from the base table.

    Accepts:
    * ``int`` — interpreted as M5 ``d`` integer (e.g. ``1500`` -> ``"d_1500"``).
    * ``str`` — either ``"d_1500"`` or an ISO date.
    * ``pd.Timestamp`` / datetime-like — used directly.
    """
    if isinstance(origin, (int, np.integer)):
        d_str = f"d_{int(origin)}"
        match = base.loc[base["d"] == d_str, "date"]
        if match.empty:
            raise ValueError(f"origin '{d_str}' not found in base table")
        return pd.Timestamp(match.iloc[0])

    if isinstance(origin, str):
        if origin.startswith("d_"):
            match = base.loc[base["d"] == origin, "date"]
            if match.empty:
                raise ValueError(f"origin '{origin}' not found in base table")
            return pd.Timestamp(match.iloc[0])
        return pd.Timestamp(origin)

    return pd.Timestamp(origin)


# --------------------------------------------------------------------------- #
# Single-origin backtest
# --------------------------------------------------------------------------- #


@dataclass
class _BacktestSlice:
    origin_date: pd.Timestamp
    history: pd.DataFrame
    actuals_lookup: pd.Series  # index: (id, target_date) -> actual sales
    ids: np.ndarray


def _slice_for_origin(
    base: pd.DataFrame,
    origin_date: pd.Timestamp,
    horizon: int,
) -> _BacktestSlice:
    """Build the leakage-safe history + actuals slice for a single origin.

    The returned ``history`` contains only rows with ``date <= origin_date``.
    The ``actuals_lookup`` is a Series indexed by (id, target_date) covering
    the next ``horizon`` days strictly after the origin.
    """
    history = base.loc[base["date"] <= origin_date].copy()
    test_end = origin_date + pd.Timedelta(days=horizon)
    test = base.loc[(base["date"] > origin_date) & (base["date"] <= test_end)]
    actuals_lookup = test.set_index(["id", "date"])["sales"].astype(float)
    ids = history["id"].unique()
    return _BacktestSlice(
        origin_date=origin_date,
        history=history,
        actuals_lookup=actuals_lookup,
        ids=ids,
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


ModelFactory = Callable[[], object]


def rolling_origin_backtest(
    base: pd.DataFrame,
    model_factory: ModelFactory,
    origins: Sequence[int | str | pd.Timestamp],
    horizons: Sequence[int],
    *,
    model_name: str | None = None,
    drop_missing_actuals: bool = True,
) -> pd.DataFrame:
    """Run a rolling-origin backtest.

    Parameters
    ----------
    base
        The joined base table (output of :func:`build_base_table`). Must have
        ``id``, ``date``, ``sales``. ``d`` is required if any origin is given
        as an M5 ``d`` integer or string.
    model_factory
        Zero-arg callable that returns a fresh model implementing
        ``fit(history)`` and ``forecast(ids, origin_date, horizons)``. See
        :class:`seercast.models.baselines.Baseline`.
    origins
        Sequence of origin specs (int / str / datetime). Internally
        coerced to dates via :func:`_origin_to_date`.
    horizons
        Step-ahead integers to forecast for each origin (e.g. ``[1, 7, 14, 28]``
        or ``range(1, 29)``).
    model_name
        Optional override for the ``model`` column in the output. Defaults to
        the factory's ``__name__`` if available, falling back to ``str(...)``.
    drop_missing_actuals
        If True (default), drop rows whose ``actual`` is NaN — i.e. forecast
        steps that fall past the end of the base table. Set False if you want
        to inspect those rows.

    Returns
    -------
    pandas.DataFrame
        Long predictions with columns: ``model``, ``origin_date``, ``id``,
        ``horizon``, ``target_date``, ``prediction``, ``actual``.
    """
    if not horizons:
        raise ValueError("horizons must be a non-empty sequence")
    horizon_max = int(max(horizons))

    if model_name is None:
        # Try .__name__ first (works for class objects), else __qualname__/str.
        model_name = getattr(model_factory, "__name__", None) or str(model_factory)

    all_preds: list[pd.DataFrame] = []
    for origin in origins:
        origin_date = _origin_to_date(base, origin)
        sl = _slice_for_origin(base, origin_date, horizon_max)

        model = model_factory()
        model.fit(sl.history)

        # Try to use the model's actual reported name (set on the instance);
        # fall back to the factory-derived name above.
        instance_name = getattr(model, "name", None) or model_name
        preds = model.forecast(sl.ids, origin_date, horizons)

        # Join actuals on (id, target_date).
        keys = list(zip(preds["id"].to_numpy(), preds["target_date"].to_numpy()))
        preds["actual"] = sl.actuals_lookup.reindex(keys).to_numpy()
        preds["model"] = instance_name

        all_preds.append(preds)

    if not all_preds:
        return pd.DataFrame(
            columns=[
                "model",
                "origin_date",
                "id",
                "horizon",
                "target_date",
                "prediction",
                "actual",
            ]
        )

    out = pd.concat(all_preds, ignore_index=True)
    if drop_missing_actuals:
        out = out.dropna(subset=["actual"]).reset_index(drop=True)
    # Canonical column order.
    return out[
        [
            "model",
            "origin_date",
            "id",
            "horizon",
            "target_date",
            "prediction",
            "actual",
        ]
    ]


__all__ = ["rolling_origin_backtest"]
