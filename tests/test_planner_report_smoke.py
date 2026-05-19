"""Planner-facing risk report smoke tests (v1.1).

v1.1 adds:
* demand_floor for ratio metrics (default 10.0)
* attention scores (volume-weighted via log1p(p50))
* low_expected_high_upside subset
* fourth leaderboard CSV + new chart

Run::

    python tests/test_planner_report_smoke.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from seercast.planning import (
    DEFAULT_DEMAND_FLOOR,
    DEFAULT_SCENARIO_NAME_MAP,
    add_attention_scores,
    add_risk_labels,
    add_scenario_sensitivity,
    build_planner_risk_report,
    build_quantile_aggregates,
    low_expected_high_upside,
    top_n_by,
)
from seercast.visualization.planner_plots import (
    plot_item_planning_demand_example,
    plot_planner_dashboard,
    plot_top_high_uncertainty,
    plot_top_low_expected_high_upside,
    plot_top_scenario_sensitive,
    plot_top_stockout_attention,
)
from seercast.training import generate_planner_report as gpr


# --------------------------------------------------------------------------- #
# Synthetic builders
# --------------------------------------------------------------------------- #


def _make_quantile_predictions(
    n_ids: int = 12, horizons=(1, 7, 14, 28), include_identity: bool = True,
) -> pd.DataFrame:
    origin = pd.Timestamp("2015-05-03")
    rng = np.random.default_rng(0)
    rows = []
    for k in range(n_ids):
        id_ = f"item_{k:03d}"
        base = 1.0 + k * 0.5
        half = 0.5 + 0.3 * k
        for h in horizons:
            target = origin + pd.Timedelta(days=h)
            p50 = base + 0.05 * h
            p10 = max(0.0, p50 - half)
            p90 = p50 + half
            row = {
                "origin_date": origin, "id": id_, "horizon": h,
                "target_date": target,
                "p10": p10, "p50": p50, "p90": p90,
                "actual": p50 + rng.normal(0, 0.1),
            }
            if include_identity:
                row.update({
                    "item_id": f"FOODS_1_{k:03d}",
                    "dept_id": "FOODS_1",
                    "cat_id": "FOODS",
                    "store_id": "CA_1",
                    "state_id": "CA",
                })
            rows.append(row)
    return pd.DataFrame(rows)


def _make_scenarios(quantile_preds: pd.DataFrame) -> pd.DataFrame:
    scenarios = ["momentum_+20pct", "momentum_-20pct",
                 "price_+10pct", "price_-10pct"]
    rows = []
    for _, q in quantile_preds.iterrows():
        for s in scenarios:
            factor = {"momentum_+20pct": 1.20,
                      "momentum_-20pct": 0.80,
                      "price_+10pct":   1.05,
                      "price_-10pct":   0.95}[s]
            base_p50 = float(q["p50"])
            rows.append({
                "scenario": s, "id": q["id"],
                "item_id": q.get("item_id", q["id"]),
                "cat_id": q.get("cat_id", "FOODS"),
                "dept_id": q.get("dept_id", "FOODS_1"),
                "store_id": q.get("store_id", "CA_1"),
                "state_id": q.get("state_id", "CA"),
                "origin_date": q["origin_date"], "horizon": q["horizon"],
                "target_date": q["target_date"],
                "base_p10": q["p10"], "base_p50": base_p50, "base_p90": q["p90"],
                "scenario_p10": q["p10"] * factor,
                "scenario_p50": base_p50 * factor,
                "scenario_p90": q["p90"] * factor,
            })
    df = pd.DataFrame(rows)
    df["delta_p50"] = df["scenario_p50"] - df["base_p50"]
    df["delta_p50_pct"] = df["delta_p50"] / df["base_p50"].abs().clip(lower=1e-9)
    return df


def _frame_with_zero_p50_product() -> pd.DataFrame:
    """One product with p50=0 across all horizons but p90>0 (the case
    that produced billion-scale ratios in v1.0). Plus one normal item
    so the rest of the report has signal to rank against."""
    origin = pd.Timestamp("2015-05-03")
    rows = []
    # Zero-expected item.
    for h in (1, 7, 14, 28):
        rows.append({
            "origin_date": origin, "id": "zero_expected_item",
            "item_id": "ITEM_Z", "dept_id": "FOODS_1", "cat_id": "FOODS",
            "store_id": "CA_1", "state_id": "CA",
            "horizon": h, "target_date": origin + pd.Timedelta(days=h),
            "p10": 0.0, "p50": 0.0, "p90": 3.0,
            "actual": 0.0,
        })
    # Normal item (high volume).
    for h in (1, 7, 14, 28):
        rows.append({
            "origin_date": origin, "id": "normal_item",
            "item_id": "ITEM_N", "dept_id": "FOODS_1", "cat_id": "FOODS",
            "store_id": "CA_1", "state_id": "CA",
            "horizon": h, "target_date": origin + pd.Timedelta(days=h),
            "p10": 50.0, "p50": 80.0, "p90": 130.0,
            "actual": 80.0,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 1. demand_floor prevents near-zero ratio explosions
# --------------------------------------------------------------------------- #


def test_zero_p50_does_not_blow_up_floored_ratio():
    """The headline v1.1 fix: a zero-p50 item must NOT show a
    billion-scale relative_uncertainty in the floored column."""
    q = _frame_with_zero_p50_product()
    agg = build_quantile_aggregates(q, demand_floor=10.0)

    zero = agg.set_index("id").loc["zero_expected_item"]
    normal = agg.set_index("id").loc["normal_item"]

    # Floored ratio for zero-p50 item is bounded by uncertainty_width / floor.
    # uncertainty_width = sum(p90 - p10) = 4 * (3 - 0) = 12. floor = 10.
    # So floored ratio = 12 / 10 = 1.2.
    assert abs(float(zero["relative_uncertainty_floored"]) - 1.2) < 1e-9
    # The raw column is still huge (provenance).
    assert float(zero["relative_uncertainty"]) > 1e9, (
        "raw ratio should still be huge so the explosion is visible if you "
        "look at the raw column"
    )

    # Normal item: floored matches raw because p50 (320) is way above the floor.
    np.testing.assert_allclose(
        float(normal["relative_uncertainty_floored"]),
        float(normal["relative_uncertainty"]),
        rtol=1e-9,
    )


def test_demand_floor_parameter_is_applied():
    """Passing a different floor changes the floored column."""
    q = _frame_with_zero_p50_product()
    agg_default = build_quantile_aggregates(q, demand_floor=10.0)
    agg_big = build_quantile_aggregates(q, demand_floor=100.0)

    zero_default = agg_default.set_index("id").loc["zero_expected_item"]
    zero_big = agg_big.set_index("id").loc["zero_expected_item"]
    # Width is 12; floored ratio at floor=100 is 12/100=0.12.
    assert abs(float(zero_default["relative_uncertainty_floored"]) - 1.2) < 1e-9
    assert abs(float(zero_big["relative_uncertainty_floored"]) - 0.12) < 1e-9


def test_floored_underforecast_risk_proxy_is_stable_at_zero_p50():
    q = _frame_with_zero_p50_product()
    agg = build_quantile_aggregates(q, demand_floor=10.0)
    zero = agg.set_index("id").loc["zero_expected_item"]
    # risk_buffer = sum(p90 - p50) = 12. floor = 10. ratio = 1.2.
    assert abs(float(zero["underforecast_risk_proxy_floored"]) - 1.2) < 1e-9


# --------------------------------------------------------------------------- #
# 2. Attention scores
# --------------------------------------------------------------------------- #


def test_attention_scores_combine_buffer_with_log1p_p50():
    """``stockout_attention_score = risk_buffer * log1p(expected_demand_p50)``."""
    q = _frame_with_zero_p50_product()
    s = _make_scenarios(q)
    risk = build_quantile_aggregates(q)
    risk = add_scenario_sensitivity(risk, s)
    risk = add_attention_scores(risk)

    zero = risk.set_index("id").loc["zero_expected_item"]
    normal = risk.set_index("id").loc["normal_item"]

    # For zero p50: log1p(0) = 0, so score = 0 regardless of risk_buffer.
    assert float(zero["stockout_attention_score"]) == 0.0
    # Normal item: risk_buffer = 4 * (130 - 80) = 200. p50 = 4*80 = 320.
    # score = 200 * log1p(320).
    np.testing.assert_allclose(
        float(normal["stockout_attention_score"]),
        200.0 * np.log1p(320.0),
        rtol=1e-9,
    )

    # Scenario attention score same shape.
    np.testing.assert_allclose(
        float(normal["scenario_attention_score"]),
        float(normal["max_abs_scenario_delta_p50"]) * np.log1p(320.0),
        rtol=1e-9,
    )
    # Zero item: scenarios are real (factor 0.8/1.2 etc) but p50=0
    # -> log1p(0)=0 -> score=0, NOT NaN.
    assert float(zero["scenario_attention_score"]) == 0.0


def test_scenario_attention_score_nan_when_scenarios_missing():
    """If no scenarios were provided, the score must be NaN (not a fake 0)
    so we can still tell the difference between "scenarios ran and no
    movement" and "scenarios didn't run at all"."""
    q = _make_quantile_predictions(n_ids=2)
    risk = build_quantile_aggregates(q)
    risk = add_scenario_sensitivity(risk, None)
    risk = add_attention_scores(risk)
    assert risk["scenario_attention_score"].isna().all()


def test_scenario_sensitivity_near_zero_pct_does_not_rank_first():
    """A zero-p50 item must NOT outrank a high-volume item on the
    scenario_attention_score leaderboard, even though its raw pct can
    be billion-scale."""
    q = _frame_with_zero_p50_product()
    s = _make_scenarios(q)
    risk = build_planner_risk_report(q, s)
    top = top_n_by(risk, "scenario_attention_score", n=2)
    # The normal high-volume item must come first.
    assert top.iloc[0]["id"] == "normal_item"


# --------------------------------------------------------------------------- #
# 3. low_expected_high_upside subset
# --------------------------------------------------------------------------- #


def test_low_expected_high_upside_catches_zero_p50_item():
    q = _frame_with_zero_p50_product()
    risk = build_quantile_aggregates(q)
    low = low_expected_high_upside(risk, p50_threshold=5.0, p90_threshold=10.0)
    # zero_expected_item has p50_total=0 and p90_total=12 -> included.
    # normal_item has p50_total=320 -> excluded.
    assert set(low["id"]) == {"zero_expected_item"}


def test_low_expected_high_upside_empty_when_no_matches():
    """Build a frame where no item meets the criteria -> empty result, no crash."""
    q = _make_quantile_predictions(n_ids=3, horizons=(1,))
    risk = build_quantile_aggregates(q)
    # Push thresholds way above any synthetic value.
    low = low_expected_high_upside(risk, p50_threshold=-1.0, p90_threshold=1e9)
    assert low.empty


# --------------------------------------------------------------------------- #
# 4. Labels source from floored / score columns by default
# --------------------------------------------------------------------------- #


def test_add_risk_labels_use_floored_and_score_columns():
    """Default label mapping reads from `relative_uncertainty_floored`,
    `stockout_attention_score`, `scenario_attention_score`."""
    df = pd.DataFrame({
        "id": [f"i_{k}" for k in range(10)],
        "relative_uncertainty_floored": list(range(10)),
        "stockout_attention_score": list(range(10)),
        "scenario_attention_score": list(range(10)),
    })
    labeled = add_risk_labels(df)
    counts = labeled["uncertainty_label"].value_counts()
    assert counts.get("high", 0) == 1
    assert counts.get("medium", 0) == 2
    assert counts.get("low", 0) == 7
    # The top item is the labelled-high one across all three label cols.
    top_idx = df["relative_uncertainty_floored"].idxmax()
    for col in ("uncertainty_label", "stockout_attention_label",
                "scenario_sensitivity_label"):
        assert labeled.loc[top_idx, col] == "high"


def test_add_risk_labels_all_nan_metric_defaults_to_low():
    df = pd.DataFrame({
        "id": ["a", "b"],
        "relative_uncertainty_floored": [1.0, 2.0],
        "stockout_attention_score": [0.0, 0.0],
        "scenario_attention_score": [np.nan, np.nan],
    })
    labeled = add_risk_labels(df)
    assert (labeled["scenario_sensitivity_label"] == "low").all()


# --------------------------------------------------------------------------- #
# 5. Scenario sensitivity sanity (with floored pct)
# --------------------------------------------------------------------------- #


def test_add_scenario_sensitivity_floored_pct_is_stable():
    """The floored pct column doesn't blow up for the zero-p50 item."""
    q = _frame_with_zero_p50_product()
    s = _make_scenarios(q)
    risk = build_quantile_aggregates(q)
    out = add_scenario_sensitivity(risk, s, demand_floor=10.0)
    zero = out.set_index("id").loc["zero_expected_item"]
    # For zero base_total_p50, scenario_total_p50 is also zero -> delta is 0
    # -> floored pct = 0.
    assert float(zero["max_abs_scenario_delta_p50_pct_floored"]) == 0.0
    # And max_abs_scenario_delta_p50 in units is also 0 (the synthetic
    # builder uses multiplicative factors).
    assert float(zero["max_abs_scenario_delta_p50"]) == 0.0


def test_add_scenario_sensitivity_handles_missing_scenarios_v1_1():
    """All v1.1 scenario columns (including the new floored one) should be
    NaN when no scenarios are given."""
    q = _make_quantile_predictions(n_ids=2)
    risk = build_quantile_aggregates(q)
    out = add_scenario_sensitivity(risk, None)
    for col in (
        "max_abs_scenario_delta_p50",
        "max_abs_scenario_delta_p50_pct",
        "max_abs_scenario_delta_p50_pct_floored",
        *DEFAULT_SCENARIO_NAME_MAP.values(),
    ):
        assert col in out.columns
        assert out[col].isna().all()


# --------------------------------------------------------------------------- #
# 6. top_n_by helper
# --------------------------------------------------------------------------- #


def test_top_n_by_skips_nans():
    df = pd.DataFrame({"id": list("abcd"), "score": [1, np.nan, 3, 2]})
    out = top_n_by(df, "score", n=2)
    assert len(out) == 2
    assert out.iloc[0]["id"] == "c"


def test_top_n_by_missing_metric_returns_empty():
    df = pd.DataFrame({"id": ["a"], "score": [1.0]})
    assert top_n_by(df, "missing").empty


# --------------------------------------------------------------------------- #
# 7. Plot functions still produce PNGs
# --------------------------------------------------------------------------- #


def _build_risk_for_plots() -> tuple[pd.DataFrame, pd.DataFrame]:
    q = _make_quantile_predictions(n_ids=8, horizons=(1, 7, 14, 28))
    s = _make_scenarios(q)
    risk = build_planner_risk_report(q, s)
    return risk, q


def test_plot_top_high_uncertainty_uses_floored_metric_by_default():
    risk, _ = _build_risk_for_plots()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "u.png"
        plot_top_high_uncertainty(risk, n=5, savepath=out)
        assert out.exists() and out.stat().st_size > 0
    plt.close("all")


def test_plot_top_stockout_attention_uses_score_by_default():
    risk, _ = _build_risk_for_plots()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "s.png"
        plot_top_stockout_attention(risk, n=5, savepath=out)
        assert out.exists() and out.stat().st_size > 0
    plt.close("all")


def test_plot_top_scenario_sensitive_handles_no_scenarios():
    risk = build_planner_risk_report(_make_quantile_predictions(n_ids=4), None)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ss.png"
        plot_top_scenario_sensitive(risk, savepath=out)
        assert out.exists() and out.stat().st_size > 0
    plt.close("all")


def test_plot_item_planning_demand_example_saves_png():
    _, q = _build_risk_for_plots()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ex.png"
        plot_item_planning_demand_example(q, savepath=out)
        assert out.exists() and out.stat().st_size > 0
    plt.close("all")


def test_plot_top_low_expected_high_upside_saves_png():
    q = pd.concat([_make_quantile_predictions(n_ids=4),
                   _frame_with_zero_p50_product()], ignore_index=True)
    risk = build_planner_risk_report(q)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "low.png"
        plot_top_low_expected_high_upside(
            risk, n=5, p50_threshold=5.0, p90_threshold=10.0, savepath=out,
        )
        assert out.exists() and out.stat().st_size > 0
    plt.close("all")


def test_plot_top_low_expected_high_upside_empty_subset_does_not_crash():
    risk = build_planner_risk_report(_make_quantile_predictions(n_ids=2))
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "low_empty.png"
        # Tighten thresholds so the subset is empty.
        plot_top_low_expected_high_upside(
            risk, n=5, p50_threshold=-1.0, p90_threshold=1e9, savepath=out,
        )
        assert out.exists() and out.stat().st_size > 0
    plt.close("all")


def test_plot_planner_dashboard_saves_png():
    risk, q = _build_risk_for_plots()
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "dash.png"
        plot_planner_dashboard(risk, q, n=5, savepath=out)
        assert out.exists() and out.stat().st_size > 0
    plt.close("all")


# --------------------------------------------------------------------------- #
# 8. CLI resolution + end-to-end
# --------------------------------------------------------------------------- #


def test_resolve_quantile_predictions_explicit_path():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "explicit.parquet"
        _make_quantile_predictions(n_ids=2).to_parquet(p, index=False)
        resolved, label = gpr._resolve_quantile_predictions_path(p)
        assert resolved == p
        assert label == "explicit"


def test_resolve_quantile_predictions_missing_explicit_raises():
    try:
        gpr._resolve_quantile_predictions_path(Path("/tmp/does_not_exist.parquet"))
    except FileNotFoundError as exc:
        assert "not found" in str(exc).lower()
        return
    raise AssertionError("missing explicit path should have raised FileNotFoundError")


def test_run_end_to_end_writes_all_v1_1_csvs():
    """End-to-end CLI must write the new top_low_expected_high_upside_ca1.csv
    and the summary row must contain demand_floor_used and
    n_low_expected_high_upside."""
    q = pd.concat(
        [_make_quantile_predictions(n_ids=15), _frame_with_zero_p50_product()],
        ignore_index=True,
    )
    s = _make_scenarios(q)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        q_path = td / "quantile_preds.parquet"
        s_path = td / "scenarios.parquet"
        q.to_parquet(q_path, index=False)
        s.to_parquet(s_path, index=False)

        reports_dir = td / "reports"
        figures_dir = td / "figures"

        result = gpr.run(
            quantile_predictions_path=q_path,
            scenario_forecasts_path=s_path,
            base_table_path=td / "no_such_base.parquet",
            reports_dir=reports_dir,
            figures_dir=figures_dir,
            top_n=5,
            demand_floor=10.0,
        )

        for name in (
            "planner_risk_report_ca1.csv",
            "top_high_uncertainty_ca1.csv",
            "top_stockout_attention_ca1.csv",
            "top_scenario_sensitive_ca1.csv",
            "top_low_expected_high_upside_ca1.csv",  # new in v1.1
            "planner_summary_ca1.csv",
        ):
            assert (reports_dir / name).exists() and (reports_dir / name).stat().st_size > 0, name
        for name in (
            "top_high_uncertainty_products.png",
            "top_stockout_attention_products.png",
            "top_scenario_sensitive_products.png",
            "top_low_expected_high_upside_products.png",  # new in v1.1
            "item_planning_demand_example.png",
            "planner_dashboard_ca1.png",
        ):
            assert (figures_dir / name).exists() and (figures_dir / name).stat().st_size > 0, name

        summary = pd.read_csv(reports_dir / "planner_summary_ca1.csv")
        row = summary.iloc[0]
        assert float(row["demand_floor_used"]) == 10.0
        assert int(row["n_low_expected_high_upside"]) >= 1
        assert 0.0 <= float(row["share_low_expected_high_upside"]) <= 1.0

        # The zero-p50 item should land in the low-upside CSV.
        low_csv = pd.read_csv(reports_dir / "top_low_expected_high_upside_ca1.csv")
        assert "zero_expected_item" in low_csv["id"].values

        # And it should NOT show up in the top of either main leaderboard.
        for csv_name in ("top_stockout_attention_ca1.csv", "top_scenario_sensitive_ca1.csv"):
            df = pd.read_csv(reports_dir / csv_name)
            if not df.empty:
                assert df.iloc[0]["id"] != "zero_expected_item", (
                    f"{csv_name} should not put the zero-p50 item first under v1.1"
                )

        assert result["source_label"] == "explicit"
        assert result["demand_floor"] == 10.0


def test_run_end_to_end_handles_missing_scenarios():
    """No scenario parquet -> scenario columns NaN, low-upside still works,
    no crash, CSVs still written (scenario CSV empty but exists)."""
    q = _make_quantile_predictions(n_ids=6)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        q_path = td / "quantile_preds.parquet"
        q.to_parquet(q_path, index=False)

        reports_dir = td / "reports"
        figures_dir = td / "figures"

        gpr.run(
            quantile_predictions_path=q_path,
            scenario_forecasts_path=td / "no_such_scenarios.parquet",
            base_table_path=td / "no_such_base.parquet",
            reports_dir=reports_dir,
            figures_dir=figures_dir,
            top_n=3,
        )

        # All CSVs still on disk.
        for name in (
            "planner_risk_report_ca1.csv",
            "top_high_uncertainty_ca1.csv",
            "top_stockout_attention_ca1.csv",
            "top_scenario_sensitive_ca1.csv",
            "top_low_expected_high_upside_ca1.csv",
            "planner_summary_ca1.csv",
        ):
            assert (reports_dir / name).exists()


if __name__ == "__main__":
    test_zero_p50_does_not_blow_up_floored_ratio()
    test_demand_floor_parameter_is_applied()
    test_floored_underforecast_risk_proxy_is_stable_at_zero_p50()
    test_attention_scores_combine_buffer_with_log1p_p50()
    test_scenario_attention_score_nan_when_scenarios_missing()
    test_scenario_sensitivity_near_zero_pct_does_not_rank_first()
    test_low_expected_high_upside_catches_zero_p50_item()
    test_low_expected_high_upside_empty_when_no_matches()
    test_add_risk_labels_use_floored_and_score_columns()
    test_add_risk_labels_all_nan_metric_defaults_to_low()
    test_add_scenario_sensitivity_floored_pct_is_stable()
    test_add_scenario_sensitivity_handles_missing_scenarios_v1_1()
    test_top_n_by_skips_nans()
    test_top_n_by_missing_metric_returns_empty()
    test_plot_top_high_uncertainty_uses_floored_metric_by_default()
    test_plot_top_stockout_attention_uses_score_by_default()
    test_plot_top_scenario_sensitive_handles_no_scenarios()
    test_plot_item_planning_demand_example_saves_png()
    test_plot_top_low_expected_high_upside_saves_png()
    test_plot_top_low_expected_high_upside_empty_subset_does_not_crash()
    test_plot_planner_dashboard_saves_png()
    test_resolve_quantile_predictions_explicit_path()
    test_resolve_quantile_predictions_missing_explicit_raises()
    test_run_end_to_end_writes_all_v1_1_csvs()
    test_run_end_to_end_handles_missing_scenarios()
    print("Planner report v1.1 smoke tests: OK")
