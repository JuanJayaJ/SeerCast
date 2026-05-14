from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _mae(actual: pd.Series, pred: pd.Series) -> float:
    return float(np.mean(np.abs(actual - pred)))


def _rmse(actual: pd.Series, pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def _wape(actual: pd.Series, pred: pd.Series) -> float:
    denom = float(np.sum(np.abs(actual)))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(actual - pred)) / denom)


def _bias(actual: pd.Series, pred: pd.Series) -> float:
    denom = float(np.sum(np.abs(actual)))
    if denom == 0:
        return np.nan
    return float(np.sum(pred - actual) / denom)


def _point_metric_row(df: pd.DataFrame, pred_col: str, model_name: str) -> dict:
    actual = df["actual"].astype(float)
    pred = df[pred_col].astype(float)

    return {
        "model": model_name,
        "n": len(df),
        "MAE": _mae(actual, pred),
        "RMSE": _rmse(actual, pred),
        "WAPE": _wape(actual, pred),
        "Bias": _bias(actual, pred),
    }


def build_model_wape_table(
    model_comparison_path: str | Path,
    quantile_predictions_path: str | Path,
) -> pd.DataFrame:
    """Build a model comparison table that includes quantile p50.

    `model_comparison_ca1.csv` already contains baselines and the LightGBM point model.
    The quantile p50 row is computed from `quantile_backtest_predictions_ca1.parquet`.
    """
    comparison = pd.read_csv(model_comparison_path)

    # Normalize column names from Phase 5 output.
    rename_map = {
        "mae": "MAE",
        "rmse": "RMSE",
        "wape": "WAPE",
        "bias": "Bias",
    }
    comparison = comparison.rename(columns=rename_map)

    quantile = pd.read_parquet(quantile_predictions_path)
    q50_row = _point_metric_row(
        quantile,
        pred_col="p50",
        model_name="lightgbm_quantile_p50",
    )

    out = pd.concat([comparison, pd.DataFrame([q50_row])], ignore_index=True)

    keep_cols = ["model", "n", "MAE", "RMSE", "WAPE", "Bias"]
    out = out[keep_cols].sort_values("WAPE", ascending=True).reset_index(drop=True)

    return out


def choose_default_scenario_id(
    scenario_forecasts: pd.DataFrame,
    scenario: str = "momentum_+20pct",
) -> str:
    """Choose a product id with non-trivial demand and visible scenario movement."""
    df = scenario_forecasts[scenario_forecasts["scenario"] == scenario].copy()
    if df.empty:
        raise ValueError(f"No rows found for scenario={scenario!r}")

    score = (
        df.groupby("id", as_index=False)
        .agg(
            base_total=("base_p50", "sum"),
            scenario_total=("scenario_p50", "sum"),
            movement=("delta_p50", lambda x: float(np.sum(np.abs(x)))),
        )
    )

    # Avoid all-flat zero demand series.
    nonzero = score[score["base_total"] > 0].copy()
    if nonzero.empty:
        nonzero = score.copy()

    nonzero["score"] = nonzero["movement"] + 0.05 * nonzero["base_total"]
    selected = nonzero.sort_values("score", ascending=False).iloc[0]["id"]

    return str(selected)


def plot_scenario_fan(
    scenario_forecasts: pd.DataFrame,
    scenario: str = "momentum_+20pct",
    item_id: str | None = None,
    output_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot base vs scenario p50 forecast with p10-p90 uncertainty bands."""
    if item_id is None:
        item_id = choose_default_scenario_id(scenario_forecasts, scenario=scenario)

    df = scenario_forecasts[
        (scenario_forecasts["scenario"] == scenario)
        & (scenario_forecasts["id"] == item_id)
    ].copy()

    if df.empty:
        raise ValueError(f"No rows found for id={item_id!r}, scenario={scenario!r}")

    df = df.sort_values("horizon")

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6))

    x = df["horizon"].astype(float).to_numpy()

    base_p10 = df["base_p10"].astype(float).to_numpy()
    base_p50 = df["base_p50"].astype(float).to_numpy()
    base_p90 = df["base_p90"].astype(float).to_numpy()

    scenario_p10 = df["scenario_p10"].astype(float).to_numpy()
    scenario_p50 = df["scenario_p50"].astype(float).to_numpy()
    scenario_p90 = df["scenario_p90"].astype(float).to_numpy()

    ax.fill_between(x, base_p10, base_p90, alpha=0.18, label="Base p10-p90")
    ax.plot(x, base_p50, linewidth=2.2, label="Base p50")

    ax.fill_between(x, scenario_p10, scenario_p90, alpha=0.18, label="Scenario p10-p90")
    ax.plot(x, scenario_p50, linewidth=2.2, linestyle="--", label="Scenario p50")

    ax.set_title(f"Base vs Scenario Forecast Fan Chart\n{id_short(item_id)} | {scenario}", fontsize=14)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Predicted daily unit sales")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax.text(
        0.0,
        -0.18,
        "p10-p90 band = forecast uncertainty range. Scenario is predictive, not causal.",
        transform=ax.transAxes,
        fontsize=9,
        alpha=0.75,
    )

    if output_path is not None:
        output_path = _ensure_parent(output_path)
        ax.figure.savefig(output_path, dpi=180, bbox_inches="tight")

    return ax


def id_short(item_id: str, max_len: int = 55) -> str:
    if len(item_id) <= max_len:
        return item_id
    return item_id[: max_len - 3] + "..."


def plot_model_wape_comparison(
    wape_table: pd.DataFrame,
    output_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot WAPE by model, sorted ascending."""
    df = wape_table.sort_values("WAPE", ascending=True).copy()

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    labels = df["model"].astype(str)
    values = df["WAPE"].astype(float)

    bars = ax.barh(labels, values)
    ax.invert_yaxis()

    ax.set_title("Model comparison by WAPE", fontsize=14)
    ax.set_xlabel("WAPE lower is better")
    ax.grid(axis="x", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:.3f}",
            va="center",
            fontsize=9,
        )

    ax.text(
        0.0,
        -0.16,
        "Comparison uses the matched evaluation grid where available. Quantile p50 is computed from Phase 6 predictions.",
        transform=ax.transAxes,
        fontsize=9,
        alpha=0.75,
    )

    if output_path is not None:
        output_path = _ensure_parent(output_path)
        ax.figure.savefig(output_path, dpi=180, bbox_inches="tight")

    return ax


def plot_coverage_by_horizon(
    diagnostics: pd.DataFrame,
    output_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot p10-p90 coverage by horizon with an 80% reference line."""
    df = diagnostics[diagnostics["dimension"] == "horizon"].copy()
    if df.empty:
        raise ValueError("No horizon rows found in uncertainty diagnostics.")

    df["value"] = df["value"].astype(int)
    df = df.sort_values("value")

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(df["value"], df["coverage_p10_p90"], marker="o", linewidth=2)
    ax.axhline(0.80, linestyle="--", linewidth=1.5, label="Target 80%")

    ax.set_title("p10-p90 Forecast Coverage by Horizon", fontsize=14)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Coverage")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, max(1.0, float(df["coverage_p10_p90"].max()) + 0.05))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax.text(
        0.0,
        -0.18,
        "Around 80% = calibrated. Above 90% = conservative. Below 75% = overconfident.",
        transform=ax.transAxes,
        fontsize=9,
        alpha=0.75,
    )

    if output_path is not None:
        output_path = _ensure_parent(output_path)
        ax.figure.savefig(output_path, dpi=180, bbox_inches="tight")

    return ax


def plot_scenario_impact(
    scenario_summary: pd.DataFrame,
    output_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot scenario impact on aggregate p50 demand forecast."""
    df = scenario_summary.sort_values("delta_total_p50_pct", ascending=True).copy()

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    values = df["delta_total_p50_pct"].astype(float)
    labels = df["scenario"].astype(str)

    bars = ax.barh(labels, values)
    ax.axvline(0, linewidth=1.2)

    ax.set_title("Scenario impact on aggregate p50 demand forecast", fontsize=14)
    ax.set_xlabel("Change in aggregate p50 forecast")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", alpha=0.25)

    for bar, value in zip(bars, values):
        x = value
        label = f"{value:+.1%}"
        ha = "left" if value >= 0 else "right"
        offset = 0.002 if value >= 0 else -0.002
        ax.text(
            x + offset,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha=ha,
            fontsize=9,
        )

    ax.text(
        0.0,
        -0.16,
        "Predictive what-if, not causal elasticity.",
        transform=ax.transAxes,
        fontsize=9,
        alpha=0.75,
    )

    if output_path is not None:
        output_path = _ensure_parent(output_path)
        ax.figure.savefig(output_path, dpi=180, bbox_inches="tight")

    return ax


def create_summary_dashboard(
    wape_table: pd.DataFrame,
    diagnostics: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    scenario_forecasts: pd.DataFrame,
    output_path: str | Path,
    scenario: str = "momentum_+20pct",
    item_id: str | None = None,
) -> Path:
    """Create a 2x2 portfolio dashboard figure."""
    output_path = _ensure_parent(output_path)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    plot_model_wape_comparison(wape_table, ax=axes[0, 0])
    plot_coverage_by_horizon(diagnostics, ax=axes[0, 1])
    plot_scenario_impact(scenario_summary, ax=axes[1, 0])
    plot_scenario_fan(scenario_forecasts, scenario=scenario, item_id=item_id, ax=axes[1, 1])

    fig.suptitle("SeerCast Forecasting Report", fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")

    return output_path
