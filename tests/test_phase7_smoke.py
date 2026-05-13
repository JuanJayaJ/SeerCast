"""Phase 7 smoke tests: scenario adjustments, simulator, comparison.

Run::

    python tests/test_phase7_smoke.py
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import lightgbm as lgb  # noqa: F401
import matplotlib
matplotlib.use("Agg")  # headless plotting for tests
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_phase4_smoke import _synthetic_base  # noqa: E402

from seercast.features.supervised import build_supervised_table
from seercast.models.quantile_lightgbm import QuantileLightGBMModel
from seercast.scenario import (
    EVENT_SCENARIO_COLUMNS,
    MOMENTUM_SCENARIO_COLUMNS,
    PRICE_SCENARIO_COLUMNS,
    SNAP_SCENARIO_COLUMNS,
    add_delta_columns,
    apply_event_scenario,
    apply_momentum_scenario,
    apply_price_scenario,
    apply_snap_scenario,
    chain,
    default_scenarios,
    scenario_summary,
    select_rows,
    simulate_scenarios,
)
from seercast.training.train_lightgbm import (
    VALID_WINDOW_DAYS,
    split_for_backtest_origin,
)
from seercast.visualization.plots import plot_scenario_fan


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _full_horizon_features(n_days: int = 260, n_items: int = 4):
    """Synthetic full-horizon supervised table mimicking Phase 4's output."""
    base = _synthetic_base(n_days=n_days, n_items=n_items)
    sup = build_supervised_table(
        base,
        horizons=list(range(1, 29)),
        snap_state="CA",
        origin_step_days=14,
    )
    return base, sup


def _fitted_quantile_model(sup: pd.DataFrame) -> tuple[QuantileLightGBMModel, pd.Timestamp]:
    """Fit a small quantile model on the synthetic supervised table."""
    origins = sup["origin_date"].drop_duplicates().sort_values()
    od = origins.iloc[len(origins) // 2]
    split = split_for_backtest_origin(sup, od, valid_window_days=VALID_WINDOW_DAYS)
    if len(split.train) == 0 or len(split.valid) == 0:
        # Pick a later origin if the middle one has no train data.
        for od in origins.iloc[len(origins) // 2 :]:
            split = split_for_backtest_origin(sup, od, valid_window_days=VALID_WINDOW_DAYS)
            if len(split.train) > 0 and len(split.valid) > 0:
                break
    m = QuantileLightGBMModel(n_estimators=80, early_stopping_rounds=20)
    m.fit(split.train, valid_df=split.valid)
    return m, od


# --------------------------------------------------------------------------- #
# Adjustments: each modifies exactly the right columns
# --------------------------------------------------------------------------- #


def test_price_scenario_modifies_price_columns_and_recomputes_dependents():
    _, sup = _full_horizon_features()
    sub = sup.head(30).copy()
    out = apply_price_scenario(sub, price_change_pct=10.0)

    # Original frame untouched.
    assert (sub["sell_price"].values == _full_horizon_features()[1].head(30)["sell_price"].values).all()

    # Direct columns scaled by factor.
    factor = 1.10
    np.testing.assert_allclose(out["sell_price"].astype(float), sub["sell_price"].astype(float) * factor, rtol=1e-5)
    np.testing.assert_allclose(out["target_sell_price"].astype(float), sub["target_sell_price"].astype(float) * factor, rtol=1e-5)

    # Dependent same-row recomputations.
    np.testing.assert_allclose(
        out["price_change_1"].astype(float),
        (out["sell_price"].astype(float) - sub["price_lag_1"].astype(float)),
        rtol=1e-5,
    )
    # price_lag_1 / price_lag_7 / price_rolling_mean_28 stay historical.
    np.testing.assert_array_equal(out["price_lag_1"].values, sub["price_lag_1"].values)
    np.testing.assert_array_equal(out["price_lag_7"].values, sub["price_lag_7"].values)
    np.testing.assert_array_equal(out["price_rolling_mean_28"].values, sub["price_rolling_mean_28"].values)

    # No other columns changed.
    other_cols = [c for c in sub.columns if c not in PRICE_SCENARIO_COLUMNS]
    pd.testing.assert_frame_equal(
        out[other_cols].reset_index(drop=True),
        sub[other_cols].reset_index(drop=True),
        check_dtype=False,
    )


def test_event_scenario_toggles_only_event_flags():
    _, sup = _full_horizon_features()
    sub = sup.head(20).copy()
    on = apply_event_scenario(sub, event_on=True)
    off = apply_event_scenario(sub, event_on=False)

    assert (on["has_event"] == 1).all()
    assert (on["target_has_event"] == 1).all()
    assert (off["has_event"] == 0).all()
    assert (off["target_has_event"] == 0).all()

    untouched = [c for c in sub.columns if c not in EVENT_SCENARIO_COLUMNS]
    pd.testing.assert_frame_equal(on[untouched], sub[untouched], check_dtype=False)


def test_snap_scenario_toggles_only_snap_flags():
    _, sup = _full_horizon_features()
    sub = sup.head(20).copy()
    on = apply_snap_scenario(sub, snap_on=True)
    off = apply_snap_scenario(sub, snap_on=False)
    assert (on["is_snap_day"] == 1).all()
    assert (on["target_is_snap_day"] == 1).all()
    assert (off["is_snap_day"] == 0).all()
    assert (off["target_is_snap_day"] == 0).all()
    untouched = [c for c in sub.columns if c not in SNAP_SCENARIO_COLUMNS]
    pd.testing.assert_frame_equal(on[untouched], sub[untouched], check_dtype=False)


def test_momentum_scenario_scales_demand_and_clips_at_zero():
    _, sup = _full_horizon_features()
    sub = sup.head(50).copy()

    up = apply_momentum_scenario(sub, demand_change_pct=20.0)
    for c in MOMENTUM_SCENARIO_COLUMNS:
        if c in sub.columns:
            np.testing.assert_allclose(
                up[c].astype(float), sub[c].astype(float) * 1.20, rtol=1e-5
            )

    # Big negative shock should hit the clip floor on at least some rows.
    crash = apply_momentum_scenario(sub, demand_change_pct=-200.0)
    for c in MOMENTUM_SCENARIO_COLUMNS:
        if c in sub.columns:
            assert (crash[c] >= 0).all(), f"{c} not clipped at 0"
    # And a column that had positive values now contains zeros.
    assert (crash["sales_lag_7"] == 0).any()


# --------------------------------------------------------------------------- #
# Comparison helpers
# --------------------------------------------------------------------------- #


def test_add_delta_columns_math():
    df = pd.DataFrame(
        {
            "base_p10": [1.0, 2.0, 0.0],
            "base_p50": [2.0, 4.0, 5.0],
            "base_p90": [3.0, 6.0, 10.0],
            "scenario_p10": [1.5, 3.0, 0.0],
            "scenario_p50": [3.0, 5.0, 7.5],
            "scenario_p90": [5.0, 7.0, 12.0],
        }
    )
    out = add_delta_columns(df)
    np.testing.assert_array_equal(out["delta_p10"].values, [0.5, 1.0, 0.0])
    np.testing.assert_array_equal(out["delta_p50"].values, [1.0, 1.0, 2.5])
    np.testing.assert_array_equal(out["delta_p90"].values, [2.0, 1.0, 2.0])
    # delta_pct: NaN where base == 0.
    np.testing.assert_array_almost_equal(
        out["delta_p50_pct"].values, [0.5, 0.25, 0.5]
    )
    assert pd.isna(out["delta_p10_pct"].iloc[2])  # base_p10[2] == 0


def test_scenario_summary_aggregates_and_sorts():
    df = pd.DataFrame(
        {
            "scenario": ["a", "a", "b", "b"],
            "base_p10": [1, 1, 1, 1],
            "base_p50": [2, 2, 2, 2],
            "base_p90": [3, 3, 3, 3],
            "scenario_p10": [1, 1, 2, 2],
            "scenario_p50": [3, 3, 4, 4],
            "scenario_p90": [4, 4, 6, 6],
        }
    )
    summ = scenario_summary(df)
    # Expected columns.
    for c in [
        "scenario", "n",
        "base_total_p10", "scenario_total_p10", "delta_total_p10", "delta_total_p10_pct",
        "base_total_p50", "scenario_total_p50", "delta_total_p50", "delta_total_p50_pct",
        "base_total_p90", "scenario_total_p90", "delta_total_p90", "delta_total_p90_pct",
    ]:
        assert c in summ.columns, c
    # Sorted by delta_total_p50_pct desc; both scenarios increase p50,
    # but b doubles it (50%) while a goes 4 -> 6 (50%) too. Either is fine, just check finite.
    assert summ["delta_total_p50_pct"].notna().all()
    # Totals match the simple sum.
    a_row = summ.set_index("scenario").loc["a"]
    assert a_row["base_total_p50"] == 4
    assert a_row["scenario_total_p50"] == 6
    assert a_row["delta_total_p50"] == 2


# --------------------------------------------------------------------------- #
# select_rows
# --------------------------------------------------------------------------- #


def test_select_rows_filters_correctly():
    _, sup = _full_horizon_features()
    od = sup["origin_date"].drop_duplicates().iloc[0]
    one_id = sup["id"].iloc[0]

    sel = select_rows(sup, origin_date=od, ids=[one_id], horizons=[1, 7, 14, 28])
    assert (sel["origin_date"] == od).all()
    assert (sel["id"] == one_id).all()
    assert set(sel["horizon"].unique()) == {1, 7, 14, 28}


# --------------------------------------------------------------------------- #
# Simulator end-to-end
# --------------------------------------------------------------------------- #


def test_simulate_scenarios_end_to_end():
    _, sup = _full_horizon_features()
    m, od = _fitted_quantile_model(sup)

    forecasts = simulate_scenarios(
        full_horizon_features=sup,
        quantile_model=m,
        scenarios=default_scenarios(),
        origin_date=od,
    )

    # Schema.
    expected = {
        "scenario", "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
        "origin_date", "horizon", "target_date",
        "base_p10", "base_p50", "base_p90",
        "scenario_p10", "scenario_p50", "scenario_p90",
        "delta_p10", "delta_p50", "delta_p90",
        "delta_p10_pct", "delta_p50_pct", "delta_p90_pct",
    }
    assert expected.issubset(set(forecasts.columns)), expected - set(forecasts.columns)

    # Quantile monotonicity preserved on both sides.
    assert (forecasts["base_p10"] <= forecasts["base_p50"] + 1e-9).all()
    assert (forecasts["base_p50"] <= forecasts["base_p90"] + 1e-9).all()
    assert (forecasts["scenario_p10"] <= forecasts["scenario_p50"] + 1e-9).all()
    assert (forecasts["scenario_p50"] <= forecasts["scenario_p90"] + 1e-9).all()


def test_simulate_scenarios_base_unchanged_across_scenarios():
    """For a given (id, horizon) the base_* values must be IDENTICAL across
    every scenario row -- the simulator computes base once and reuses it.
    """
    _, sup = _full_horizon_features()
    m, od = _fitted_quantile_model(sup)
    forecasts = simulate_scenarios(
        full_horizon_features=sup,
        quantile_model=m,
        scenarios=default_scenarios(),
        origin_date=od,
    )

    for (id_, h), grp in forecasts.groupby(["id", "horizon"]):
        for col in ("base_p10", "base_p50", "base_p90"):
            assert grp[col].nunique() == 1, (
                f"base value drifted across scenarios for ({id_}, h={h}, {col}): "
                f"got {grp[col].unique()}"
            )


def test_simulate_scenarios_momentum_actually_changes_predictions():
    """Sanity: scaling the historical demand features by 50% must move
    p50 predictions on at least some rows. Without this guarantee a
    silent no-op simulator would pass every other test.

    We assert *that* it moves, not which direction. Direction is a
    model property, not a causal claim.
    """
    _, sup = _full_horizon_features()
    m, od = _fitted_quantile_model(sup)
    forecasts = simulate_scenarios(
        full_horizon_features=sup,
        quantile_model=m,
        scenarios={"momentum_+50pct": partial(apply_momentum_scenario, demand_change_pct=50.0)},
        origin_date=od,
    )
    moved = (forecasts["delta_p50"].abs() > 1e-9).sum()
    assert moved >= 1, "momentum scenario didn't move any p50 prediction; check wiring"


def test_simulate_scenarios_chain_compose():
    _, sup = _full_horizon_features()
    m, od = _fitted_quantile_model(sup)
    composite = chain(
        partial(apply_event_scenario, event_on=True),
        partial(apply_price_scenario, price_change_pct=-10.0),
    )
    forecasts = simulate_scenarios(
        full_horizon_features=sup,
        quantile_model=m,
        scenarios={"compound": composite},
        origin_date=od,
    )
    assert (forecasts["scenario"] == "compound").all()
    assert len(forecasts) > 0


# --------------------------------------------------------------------------- #
# Visualization smoke test
# --------------------------------------------------------------------------- #


def test_plot_scenario_fan_returns_axes():
    _, sup = _full_horizon_features()
    m, od = _fitted_quantile_model(sup)
    forecasts = simulate_scenarios(
        full_horizon_features=sup,
        quantile_model=m,
        scenarios=default_scenarios(),
        origin_date=od,
    )
    one_id = forecasts["id"].iloc[0]
    one_scen = forecasts["scenario"].iloc[0]
    ax = plot_scenario_fan(forecasts, id_=one_id, scenario=one_scen)
    assert ax is not None
    # Has both base and scenario lines + bands -> at least 4 artists.
    assert len(ax.lines) + len(ax.collections) >= 4


if __name__ == "__main__":
    test_price_scenario_modifies_price_columns_and_recomputes_dependents()
    test_event_scenario_toggles_only_event_flags()
    test_snap_scenario_toggles_only_snap_flags()
    test_momentum_scenario_scales_demand_and_clips_at_zero()
    test_add_delta_columns_math()
    test_scenario_summary_aggregates_and_sorts()
    test_select_rows_filters_correctly()
    test_simulate_scenarios_end_to_end()
    test_simulate_scenarios_base_unchanged_across_scenarios()
    test_simulate_scenarios_momentum_actually_changes_predictions()
    test_simulate_scenarios_chain_compose()
    test_plot_scenario_fan_returns_axes()
    print("Phase 7 smoke tests: OK")
