"""Generate the planner-facing risk report (v1.1).

v1.1 changes vs v1.0:
* Demand floor applied to ratio metrics so near-zero p50 items don't
  produce billion-scale percentages.
* Volume-weighted ``stockout_attention_score`` and
  ``scenario_attention_score`` are the primary ranking signals.
* New "low-expected / high-upside" view for items with near-zero p50
  but meaningful p90.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd

from seercast.config import (
    ARTIFACTS,
    FIGURES_DIR,
    REPORTS_DIR,
    ensure_dirs,
)
from seercast.planning import (
    DEFAULT_DEMAND_FLOOR,
    add_attention_scores,
    add_risk_labels,
    add_scenario_sensitivity,
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


_LIFECYCLE_QUANTILE = (
    REPORTS_DIR / "experiments" / "lifecycle_features"
    / "quantile_backtest_predictions_ca1.parquet"
)
_PHASE6_QUANTILE = ARTIFACTS.quantile_backtest_predictions_ca1
_PLANNER_REPORTS_DIR = REPORTS_DIR / "planner"
_PLANNER_FIGURES_DIR = FIGURES_DIR / "planner"

_DEFAULT_BEST_MODEL_REFERENCE = (
    "shared-horizon LightGBM quantile p50 with lifecycle features "
    "(WAPE 0.696590)"
)

# Defaults for the low-expected / high-upside subset.
_LOW_EXPECTED_P50 = 5.0
_LOW_EXPECTED_P90 = 10.0


# --------------------------------------------------------------------------- #
# Resolution helpers
# --------------------------------------------------------------------------- #


def _resolve_quantile_predictions_path(
    explicit: Path | str | None,
) -> tuple[Path, str]:
    if explicit is not None:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"requested quantile predictions not found: {p}")
        return p, "explicit"
    if _LIFECYCLE_QUANTILE.exists():
        return _LIFECYCLE_QUANTILE, "lifecycle"
    if _PHASE6_QUANTILE.exists():
        return _PHASE6_QUANTILE, "phase6"
    raise FileNotFoundError(
        "no quantile predictions parquet found in lifecycle-experiment or "
        "phase 6 locations."
    )


def _attach_identity_from_base(
    quantile_predictions: pd.DataFrame,
    base_table_path: Path | str,
) -> pd.DataFrame:
    needed = ("item_id", "dept_id", "cat_id", "store_id", "state_id")
    missing = [c for c in needed if c not in quantile_predictions.columns]
    if not missing:
        return quantile_predictions
    base_path = Path(base_table_path)
    if not base_path.exists():
        return quantile_predictions
    base = (
        pd.read_parquet(base_path, columns=["id", *needed])
        .drop_duplicates("id")
    )
    return quantile_predictions.merge(base, on="id", how="left")


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #


def _build_summary(
    risk: pd.DataFrame,
    low_upside: pd.DataFrame,
    *,
    best_model_reference: str,
    source_label: str,
    quantile_path: Path,
    scenario_path: Path | None,
    demand_floor: float,
    p50_threshold: float,
    p90_threshold: float,
) -> pd.DataFrame:
    has_scen = (
        "scenario_attention_score" in risk.columns
        and not risk["scenario_attention_score"].isna().all()
    )
    n = max(int(len(risk)), 1)
    row = {
        "n_products": int(len(risk)),
        "total_expected_p50": float(risk["expected_demand_p50"].sum()),
        "total_conservative_p90": float(risk["conservative_demand_p90"].sum()),
        "total_risk_buffer": float(risk["risk_buffer"].sum()),
        "share_high_uncertainty": float((risk["uncertainty_label"] == "high").mean()),
        "share_high_stockout_attention": float((risk["stockout_attention_label"] == "high").mean()),
        "share_high_scenario_sensitivity": (
            float((risk["scenario_sensitivity_label"] == "high").mean()) if has_scen else None
        ),
        "demand_floor_used": float(demand_floor),
        "low_expected_p50_threshold": float(p50_threshold),
        "low_expected_p90_threshold": float(p90_threshold),
        "n_low_expected_high_upside": int(len(low_upside)),
        "share_low_expected_high_upside": float(len(low_upside)) / n,
        "best_model_reference": best_model_reference,
        "quantile_source": source_label,
        "quantile_path": str(quantile_path),
        "scenario_path": str(scenario_path) if scenario_path else "",
    }
    return pd.DataFrame([row])


# --------------------------------------------------------------------------- #
# CLI orchestrator
# --------------------------------------------------------------------------- #


def run(
    *,
    quantile_predictions_path: Path | str | None = None,
    scenario_forecasts_path: Path | str = ARTIFACTS.scenario_forecasts_ca1,
    base_table_path: Path | str = ARTIFACTS.base_table_ca1,
    reports_dir: Path | str = _PLANNER_REPORTS_DIR,
    figures_dir: Path | str = _PLANNER_FIGURES_DIR,
    best_model_reference: str = _DEFAULT_BEST_MODEL_REFERENCE,
    top_n: int = 20,
    demand_floor: float = DEFAULT_DEMAND_FLOOR,
    low_expected_p50: float = _LOW_EXPECTED_P50,
    low_expected_p90: float = _LOW_EXPECTED_P90,
) -> dict:
    """Build the planner report. Returns the in-memory frames + paths."""
    ensure_dirs()
    reports_dir = Path(reports_dir); reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(figures_dir); figures_dir.mkdir(parents=True, exist_ok=True)

    quantile_path, source_label = _resolve_quantile_predictions_path(quantile_predictions_path)
    print(f"quantile predictions: {quantile_path}  (source={source_label})")
    quantile_predictions = pd.read_parquet(quantile_path)
    quantile_predictions = _attach_identity_from_base(quantile_predictions, base_table_path)

    scenario_path = Path(scenario_forecasts_path) if scenario_forecasts_path else None
    if scenario_path is not None and scenario_path.exists():
        print(f"scenario forecasts:   {scenario_path}")
        scenario_forecasts = pd.read_parquet(scenario_path)
    else:
        print(f"scenario forecasts:   not found ({scenario_path}) -- scenario sensitivity will be NaN")
        scenario_forecasts = None
        scenario_path = None

    print(f"demand_floor: {demand_floor:g} units (used for ratio metric denominators)")
    print()

    # 1. Aggregates with floor.
    risk = build_quantile_aggregates(
        quantile_predictions, demand_floor=demand_floor,
    )
    print(f"per-id aggregates: {len(risk):,} products")

    # 2. Scenario sensitivity with floor.
    risk = add_scenario_sensitivity(
        risk, scenario_forecasts, demand_floor=demand_floor,
    )

    # 3. Volume-weighted attention scores.
    risk = add_attention_scores(risk)

    # 4. Labels source from floored / score columns by default.
    risk = add_risk_labels(risk)

    # 5. Low-expected / high-upside subset.
    low_upside = low_expected_high_upside(
        risk, p50_threshold=low_expected_p50, p90_threshold=low_expected_p90,
    )

    # 6. Persist CSVs.
    out_risk = reports_dir / "planner_risk_report_ca1.csv"
    risk.to_csv(out_risk, index=False)

    # v1.2: the high-uncertainty CSV mirrors the chart's behaviour and
    # excludes low-expected items (they show up on the low-upside CSV/chart).
    out_uncertainty = reports_dir / "top_high_uncertainty_ca1.csv"
    uncertainty_pool = risk.loc[risk["expected_demand_p50"] >= demand_floor]
    top_n_by(uncertainty_pool, "relative_uncertainty_floored", n=top_n).to_csv(
        out_uncertainty, index=False,
    )

    out_stockout = reports_dir / "top_stockout_attention_ca1.csv"
    top_n_by(risk, "stockout_attention_score", n=top_n).to_csv(out_stockout, index=False)

    out_scenario = reports_dir / "top_scenario_sensitive_ca1.csv"
    top_n_by(risk, "scenario_attention_score", n=top_n).to_csv(out_scenario, index=False)

    out_low_upside = reports_dir / "top_low_expected_high_upside_ca1.csv"
    low_upside.head(top_n).to_csv(out_low_upside, index=False)

    # 7. Summary.
    summary = _build_summary(
        risk, low_upside,
        best_model_reference=best_model_reference,
        source_label=source_label,
        quantile_path=quantile_path,
        scenario_path=scenario_path,
        demand_floor=demand_floor,
        p50_threshold=low_expected_p50,
        p90_threshold=low_expected_p90,
    )
    out_summary = reports_dir / "planner_summary_ca1.csv"
    summary.to_csv(out_summary, index=False)

    # 8. Plots.
    # v1.2: high-uncertainty excludes items with p50 < demand_floor by default
    # (those belong on the low-expected/high-upside chart instead).
    plot_top_high_uncertainty(
        risk, n=top_n,
        demand_floor=demand_floor,
        include_low_expected=False,
        savepath=figures_dir / "top_high_uncertainty_products.png",
    )
    plot_top_stockout_attention(
        risk, n=top_n,
        savepath=figures_dir / "top_stockout_attention_products.png",
    )
    plot_top_scenario_sensitive(
        risk, n=top_n,
        savepath=figures_dir / "top_scenario_sensitive_products.png",
    )
    plot_top_low_expected_high_upside(
        risk, n=top_n,
        p50_threshold=low_expected_p50, p90_threshold=low_expected_p90,
        savepath=figures_dir / "top_low_expected_high_upside_products.png",
    )
    # v1.2: example plot uses risk_df so the auto-selected id is a
    # planner-meaningful stockout-attention candidate.
    plot_item_planning_demand_example(
        quantile_predictions,
        risk_df=risk,
        savepath=figures_dir / "item_planning_demand_example.png",
    )
    plot_planner_dashboard(
        risk, quantile_predictions, n=min(12, top_n),
        demand_floor=demand_floor,
        savepath=figures_dir / "planner_dashboard_ca1.png",
    )
    plt.close("all")

    # 9. Print summary + four leaderboards.
    print("=" * 60)
    print("planner summary")
    print("=" * 60)
    print(summary.iloc[0].to_string())
    print()
    print(
        f"Ratio metrics use demand_floor={demand_floor:g} to avoid misleading "
        f"huge percentages when p50 is near zero. (v1.2)\n"
        f"High-uncertainty leaderboard EXCLUDES items with "
        f"expected_demand_p50 < {demand_floor:g} (low-expected/high-upside "
        f"products are reported separately).\n"
        f"Planning example item is selected from meaningful stockout-attention "
        f"candidates (p50 ≥ 50 AND risk_buffer ≥ 50) where possible."
    )
    # v1.2: each leaderboard pulls from the SAME pool its CSV was written
    # from. The uncertainty leaderboard pool excludes low-expected items
    # (they belong on the separate low-expected/high-upside leaderboard).
    leaderboards = [
        ("top 10 by high uncertainty (floored, p50 ≥ {0:g})".format(demand_floor),
         uncertainty_pool, "relative_uncertainty_floored",  False, "uncertainty_label"),
        ("top 10 by stockout attention score",
         risk,             "stockout_attention_score",      False, "stockout_attention_label"),
        ("top 10 by scenario attention score",
         risk,             "scenario_attention_score",      False, "scenario_sensitivity_label"),
    ]
    for title, pool, metric, ascending, label_col in leaderboards:
        print()
        print(title + ":")
        top = top_n_by(pool, metric, n=10, ascending=ascending)
        if top.empty:
            print(f"  (no rows for {metric})")
            continue
        cols_to_show = [c for c in ("id", "cat_id", "dept_id",
                                    "expected_demand_p50",
                                    "conservative_demand_p90",
                                    "risk_buffer",
                                    metric, label_col)
                        if c in top.columns]
        print(top[cols_to_show].to_string(index=False))

    print()
    print("top 10 low expected / high upside (p50 < {0:g}, p90 ≥ {1:g}):".format(
        low_expected_p50, low_expected_p90,
    ))
    if low_upside.empty:
        print("  (no rows match the low-expected/high-upside criteria)")
    else:
        cols = [c for c in ("id", "cat_id", "dept_id",
                            "expected_demand_p50", "conservative_demand_p90",
                            "risk_buffer", "stockout_attention_score")
                if c in low_upside.columns]
        print(low_upside.head(10)[cols].to_string(index=False))

    print()
    print("Reports written to:")
    for p in (out_risk, out_uncertainty, out_stockout, out_scenario,
              out_low_upside, out_summary):
        print(f"  {p}")
    print("Figures written to:")
    for f in (
        "top_high_uncertainty_products.png",
        "top_stockout_attention_products.png",
        "top_scenario_sensitive_products.png",
        "top_low_expected_high_upside_products.png",
        "item_planning_demand_example.png",
        "planner_dashboard_ca1.png",
    ):
        print(f"  {figures_dir / f}")
    print()
    print("Reminder: p90 is a planning quantile, not guaranteed demand. "
          "Scenario sensitivity is predictive, not causal. "
          "Risk labels are percentile heuristics, not service-level targets.")

    return {
        "risk_report": risk,
        "low_expected_high_upside": low_upside,
        "summary": summary,
        "quantile_path": quantile_path,
        "scenario_path": scenario_path,
        "source_label": source_label,
        "demand_floor": float(demand_floor),
    }


def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate the SeerCast planner-facing risk report (v1.1).")
    p.add_argument("--quantile-predictions", type=Path, default=None)
    p.add_argument("--scenarios", type=Path, default=ARTIFACTS.scenario_forecasts_ca1)
    p.add_argument("--base", type=Path, default=ARTIFACTS.base_table_ca1)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--demand-floor", type=float, default=DEFAULT_DEMAND_FLOOR,
                   help="Denominator floor for ratio metrics. Default 10.0 units.")
    p.add_argument("--low-expected-p50", type=float, default=_LOW_EXPECTED_P50)
    p.add_argument("--low-expected-p90", type=float, default=_LOW_EXPECTED_P90)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    run(
        quantile_predictions_path=args.quantile_predictions,
        scenario_forecasts_path=args.scenarios,
        base_table_path=args.base,
        top_n=args.top_n,
        demand_floor=args.demand_floor,
        low_expected_p50=args.low_expected_p50,
        low_expected_p90=args.low_expected_p90,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
