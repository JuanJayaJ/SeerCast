from __future__ import annotations

from pathlib import Path

import pandas as pd

from seercast.config import REPO_ROOT
from seercast.visualization.report_plots import (
    build_model_wape_table,
    choose_default_scenario_id,
    create_summary_dashboard,
    plot_coverage_by_horizon,
    plot_model_wape_comparison,
    plot_scenario_fan,
    plot_scenario_impact,
)


REPORTS_DIR = REPO_ROOT / "outputs" / "reports"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"


def run() -> dict[str, Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model_comparison_path = REPORTS_DIR / "model_comparison_ca1.csv"
    quantile_predictions_path = REPORTS_DIR / "quantile_backtest_predictions_ca1.parquet"
    diagnostics_path = REPORTS_DIR / "uncertainty_diagnostics_ca1.csv"
    scenario_forecasts_path = REPORTS_DIR / "scenario_forecasts_ca1.parquet"
    scenario_summary_path = REPORTS_DIR / "scenario_comparison_ca1.csv"

    required = [
        model_comparison_path,
        quantile_predictions_path,
        diagnostics_path,
        scenario_forecasts_path,
        scenario_summary_path,
    ]

    missing = [path for path in required if not path.exists()]
    if missing:
        missing_text = "\n".join(f" - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing required report artifact(s). Run Phases 5-7 first:\n"
            "python -m seercast.training.train_lightgbm\n"
            "python -m seercast.training.train_quantile_lightgbm\n"
            "python -m seercast.training.run_scenarios\n\n"
            f"Missing:\n{missing_text}"
        )

    print("loading report artifacts...")
    wape_table = build_model_wape_table(model_comparison_path, quantile_predictions_path)
    diagnostics = pd.read_csv(diagnostics_path)
    scenario_forecasts = pd.read_parquet(scenario_forecasts_path)
    scenario_summary = pd.read_csv(scenario_summary_path)

    print("model WAPE table:")
    print(wape_table.to_string(index=False))

    hero_scenario = "momentum_+20pct"
    hero_id = choose_default_scenario_id(scenario_forecasts, scenario=hero_scenario)
    print(f"selected hero product: {hero_id}")
    print(f"selected hero scenario: {hero_scenario}")

    outputs = {
        "model_wape": FIGURES_DIR / "model_wape_comparison.png",
        "coverage": FIGURES_DIR / "coverage_by_horizon.png",
        "scenario_impact": FIGURES_DIR / "scenario_impact_p50.png",
        "hero_fan": FIGURES_DIR / "hero_scenario_fan_momentum_plus.png",
        "dashboard": FIGURES_DIR / "seercast_summary_dashboard.png",
    }

    print("creating figures...")
    plot_model_wape_comparison(wape_table, output_path=outputs["model_wape"])
    plot_coverage_by_horizon(diagnostics, output_path=outputs["coverage"])
    plot_scenario_impact(scenario_summary, output_path=outputs["scenario_impact"])
    plot_scenario_fan(
        scenario_forecasts,
        scenario=hero_scenario,
        item_id=hero_id,
        output_path=outputs["hero_fan"],
    )
    create_summary_dashboard(
        wape_table=wape_table,
        diagnostics=diagnostics,
        scenario_summary=scenario_summary,
        scenario_forecasts=scenario_forecasts,
        scenario=hero_scenario,
        item_id=hero_id,
        output_path=outputs["dashboard"],
    )

    for name, path in outputs.items():
        print(f"wrote {name}: {path}")

    return outputs


if __name__ == "__main__":
    run()
