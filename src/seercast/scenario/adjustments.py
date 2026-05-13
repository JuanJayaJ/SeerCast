"""Scenario adjustments: pure ``df -> df`` functions.

Each function takes a slice of the supervised feature table (typically
the full-horizon table from Phase 4) and returns a new DataFrame with
the same shape, with feature columns modified per the scenario.

**Important framing.** These are *predictive* what-ifs, not causal claims.
When we ``apply_price_scenario(df, price_change_pct=10)`` we are not
modeling that "raising price by 10% causes a demand drop"; we are asking
"what does the model predict if today's input prices were 10% higher,
holding everything else fixed?" That distinction matters for
interpretation and is explicit in the module docstrings.

Adjustments do not touch the model. They are pure feature-space
transformations. The simulator orchestrates them with the model.
"""

from __future__ import annotations

from functools import reduce
from typing import Callable, Iterable

import numpy as np
import pandas as pd


# Columns that each scenario modifies. Used by the smoke tests to assert
# that exactly these columns differ from the input.
PRICE_SCENARIO_COLUMNS: tuple[str, ...] = (
    "sell_price",
    "target_sell_price",
    "price_change_1",
    "price_change_7",
    "price_pct_change_1",
    "price_pct_change_7",
    "price_relative_to_28d_avg",
)
EVENT_SCENARIO_COLUMNS: tuple[str, ...] = ("has_event", "target_has_event")
SNAP_SCENARIO_COLUMNS: tuple[str, ...] = ("is_snap_day", "target_is_snap_day")
MOMENTUM_SCENARIO_COLUMNS: tuple[str, ...] = (
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_lag_56",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_mean_56",
    "rolling_min_28",
    "rolling_max_28",
)


# --------------------------------------------------------------------------- #
# Price
# --------------------------------------------------------------------------- #


def apply_price_scenario(
    df: pd.DataFrame,
    *,
    price_change_pct: float,
) -> pd.DataFrame:
    """Multiplicative one-shot price change at the origin.

    Interpretation: today's prices are ``(1 + price_change_pct/100)`` times
    the baseline, while the past 28 days kept their original prices. We
    therefore modify ``sell_price`` and ``target_sell_price``, recompute
    every dependent same-row feature, and leave ``price_lag_1``,
    ``price_lag_7``, and ``price_rolling_mean_28`` at their unchanged
    historical values. ``price_relative_to_28d_avg`` ends up larger (or
    smaller) accordingly -- which is the intended "is today's price
    unusually high?" signal.

    Returns a new DataFrame; ``df`` is not mutated.
    """
    out = df.copy()
    factor = 1.0 + float(price_change_pct) / 100.0

    # Direct price columns.
    out["sell_price"] = (out["sell_price"] * factor).astype(out["sell_price"].dtype)
    out["target_sell_price"] = (
        out["target_sell_price"] * factor
    ).astype(out["target_sell_price"].dtype)

    # Same-row deltas. price_lag_{1,7} are historical and stay put.
    out["price_change_1"] = (out["sell_price"] - out["price_lag_1"]).astype("float32")
    out["price_change_7"] = (out["sell_price"] - out["price_lag_7"]).astype("float32")

    with np.errstate(divide="ignore", invalid="ignore"):
        out["price_pct_change_1"] = (
            out["price_change_1"] / out["price_lag_1"]
        ).astype("float32")
        out["price_pct_change_7"] = (
            out["price_change_7"] / out["price_lag_7"]
        ).astype("float32")
        out["price_relative_to_28d_avg"] = (
            out["sell_price"] / out["price_rolling_mean_28"]
        ).astype("float32")

    return out


# --------------------------------------------------------------------------- #
# Event
# --------------------------------------------------------------------------- #


def apply_event_scenario(df: pd.DataFrame, *, event_on: bool) -> pd.DataFrame:
    """Toggle the unified event flag at both origin and target.

    Sets ``has_event`` and ``target_has_event`` to ``int(event_on)``. The
    spec deliberately keeps event_name_1 / event_name_2 distinctions out
    of scenarios -- this is the simplest "is anything special happening?"
    flip.
    """
    out = df.copy()
    val = np.int8(1 if event_on else 0)
    out["has_event"] = val
    out["target_has_event"] = val
    return out


# --------------------------------------------------------------------------- #
# SNAP
# --------------------------------------------------------------------------- #


def apply_snap_scenario(df: pd.DataFrame, *, snap_on: bool) -> pd.DataFrame:
    """Toggle the SNAP flag at both origin and target.

    For CA_1, this is equivalent to forcing ``snap_CA = int(snap_on)``.
    """
    out = df.copy()
    val = np.int8(1 if snap_on else 0)
    out["is_snap_day"] = val
    out["target_is_snap_day"] = val
    return out


# --------------------------------------------------------------------------- #
# Momentum
# --------------------------------------------------------------------------- #


def apply_momentum_scenario(
    df: pd.DataFrame,
    *,
    demand_change_pct: float,
) -> pd.DataFrame:
    """Multiplicative scaling of recent demand features. Clipped at 0.

    Interpretation: "what does the model predict if recent demand had been
    ``(1 + demand_change_pct/100)`` times what we observed?" We scale all
    historical demand lags and rolling stats listed in
    :data:`MOMENTUM_SCENARIO_COLUMNS`. Negative scaling can push values
    below zero arithmetically; we clip at zero because demand is
    non-negative by construction.

    Note: ``rolling_std_*``, ``zero_sales_rate_28``, and
    ``nonzero_sales_count_28`` are *not* scaled -- the spec excludes them,
    and rescaling a std (or a count of nonzero observations) by the same
    multiplicative factor as the level isn't a clean operation in
    feature space.
    """
    out = df.copy()
    factor = 1.0 + float(demand_change_pct) / 100.0
    for c in MOMENTUM_SCENARIO_COLUMNS:
        if c not in out.columns:
            continue
        original_dtype = out[c].dtype
        scaled = (out[c].astype("float64") * factor).clip(lower=0.0)
        out[c] = scaled.astype(original_dtype)
    return out


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #


Adjustment = Callable[[pd.DataFrame], pd.DataFrame]


def chain(*adjustments: Adjustment) -> Adjustment:
    """Compose adjustments left-to-right into a single ``df -> df`` callable.

    Useful for compound scenarios::

        from functools import partial
        scenario_drop_and_event = chain(
            partial(apply_price_scenario, price_change_pct=-15.0),
            partial(apply_event_scenario, event_on=True),
        )
    """
    def composed(df: pd.DataFrame) -> pd.DataFrame:
        return reduce(lambda d, f: f(d), adjustments, df)
    return composed


__all__ = [
    "PRICE_SCENARIO_COLUMNS",
    "EVENT_SCENARIO_COLUMNS",
    "SNAP_SCENARIO_COLUMNS",
    "MOMENTUM_SCENARIO_COLUMNS",
    "apply_price_scenario",
    "apply_event_scenario",
    "apply_snap_scenario",
    "apply_momentum_scenario",
    "chain",
    "Adjustment",
]
