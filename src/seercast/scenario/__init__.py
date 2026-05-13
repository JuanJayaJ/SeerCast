"""Predictive (not causal) scenario simulation: price, event, SNAP, momentum."""

from functools import partial

from seercast.scenario.adjustments import (
    Adjustment,
    EVENT_SCENARIO_COLUMNS,
    MOMENTUM_SCENARIO_COLUMNS,
    PRICE_SCENARIO_COLUMNS,
    SNAP_SCENARIO_COLUMNS,
    apply_event_scenario,
    apply_momentum_scenario,
    apply_price_scenario,
    apply_snap_scenario,
    chain,
)
from seercast.scenario.comparison import add_delta_columns, scenario_summary
from seercast.scenario.simulator import (
    load_quantile_models_bundle,
    pick_quantile_model_for_origin,
    select_rows,
    simulate_scenarios,
)


def default_scenarios() -> dict[str, Adjustment]:
    """A reasonable starter pack of scenarios for first-look diagnostics.

    Six scenarios that exercise each adjustment type plus symmetric
    +/- variants for price and momentum. Replace with a custom dict for
    your own analyses.
    """
    return {
        "price_+10pct":     partial(apply_price_scenario, price_change_pct=10.0),
        "price_-10pct":    partial(apply_price_scenario, price_change_pct=-10.0),
        "event_on":         partial(apply_event_scenario, event_on=True),
        "snap_on":          partial(apply_snap_scenario, snap_on=True),
        "momentum_+20pct":  partial(apply_momentum_scenario, demand_change_pct=20.0),
        "momentum_-20pct": partial(apply_momentum_scenario, demand_change_pct=-20.0),
    }


__all__ = [
    "Adjustment",
    "PRICE_SCENARIO_COLUMNS",
    "EVENT_SCENARIO_COLUMNS",
    "SNAP_SCENARIO_COLUMNS",
    "MOMENTUM_SCENARIO_COLUMNS",
    "apply_price_scenario",
    "apply_event_scenario",
    "apply_snap_scenario",
    "apply_momentum_scenario",
    "chain",
    "add_delta_columns",
    "scenario_summary",
    "select_rows",
    "load_quantile_models_bundle",
    "pick_quantile_model_for_origin",
    "simulate_scenarios",
    "default_scenarios",
]
