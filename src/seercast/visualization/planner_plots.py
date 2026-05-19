"""Planner-facing plots (v1.1).

Five matplotlib helpers + one new low-upside chart. Defaults now point at
the v1.1 stable metrics (floored ratios and volume-weighted attention
scores) so leaderboards are not dominated by near-zero p50 items.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Palette.
_COLOR_BASE = "#1f77b4"
_COLOR_HIGH = "#ff7f0e"
_COLOR_LOW = "#2ca02c"
_COLOR_FILL = "#ffd6a8"
_COLOR_UPSIDE = "#9467bd"


def _maybe_save(fig: plt.Figure, savepath: Path | str | None, dpi: int = 160) -> None:
    if savepath is None:
        return
    p = Path(savepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")


def _shorten(s: str, max_len: int = 32) -> str:
    s = str(s)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _label_bars(ax: plt.Axes, values, *, fmt: str = "{:.1f}") -> None:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return
    xmax = float(np.nanmax(np.abs(arr))) if np.isfinite(np.nanmax(np.abs(arr))) else 0.0
    pad = xmax * 0.01 if xmax > 0 else 0.05
    for i, v in enumerate(arr):
        if not np.isfinite(v):
            continue
        ax.text(v + pad, i, fmt.format(v), va="center", fontsize=8)


def _empty_panel(ax: plt.Axes, message: str, savepath: Path | str | None) -> plt.Axes:
    ax.text(0.5, 0.5, message, ha="center", va="center", color="#888", fontsize=11)
    ax.set_axis_off()
    _maybe_save(ax.figure, savepath)
    return ax


# ------------------------------------------------------------------------- #
# 1. Top high-uncertainty products (uses floored metric by default)
# ------------------------------------------------------------------------- #


def plot_top_high_uncertainty(
    risk_df: pd.DataFrame,
    *,
    n: int = 20,
    metric: str = "relative_uncertainty_floored",
    ax: plt.Axes | None = None,
    savepath: Path | str | None = None,
) -> plt.Axes:
    """Top-n products by ``metric`` (default: floored relative uncertainty).

    The floored metric is stable when ``expected_demand_p50`` is small;
    it represents "uncertainty width per ``max(p50, demand_floor)`` units"
    rather than "per epsilon" so a near-zero forecast doesn't dominate.
    """
    if metric not in risk_df.columns:
        # Back-compat fallback.
        metric = "relative_uncertainty"
    sub = (
        risk_df.dropna(subset=[metric])
        .sort_values(metric, ascending=False)
        .head(n)
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(3, 0.30 * len(sub) + 1)))
    else:
        fig = ax.figure

    if sub.empty:
        return _empty_panel(ax, f"no rows with non-null {metric}", savepath)

    labels = [_shorten(s) for s in sub["id"][::-1].astype(str)]
    values = sub[metric][::-1].astype(float).values
    ax.barh(labels, values, color=_COLOR_HIGH)
    _label_bars(ax, values, fmt="{:.2f}")
    ax.set_xlabel(metric)
    ax.set_title(f"Top {len(sub)} products by {metric}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.text(0.01, -0.02,
             "Floored denominator avoids huge ratios at near-zero demand. "
             "p90 is a planning quantile, not guaranteed demand.",
             fontsize=8, color="#666")
    _maybe_save(fig, savepath)
    return ax


# ------------------------------------------------------------------------- #
# 2. Top stockout-attention (uses score by default)
# ------------------------------------------------------------------------- #


def plot_top_stockout_attention(
    risk_df: pd.DataFrame,
    *,
    n: int = 20,
    metric: str = "stockout_attention_score",
    ax: plt.Axes | None = None,
    savepath: Path | str | None = None,
) -> plt.Axes:
    """Top-n by ``stockout_attention_score = risk_buffer * log1p(p50)`` by default.

    Combines the absolute buffer size with a log-weighted demand prior, so
    high-volume items with meaningful buffer dominate over near-zero
    items where the raw buffer is small even if the ratio is huge.

    Falls back to ``risk_buffer`` if the score column isn't present.
    """
    if metric not in risk_df.columns:
        metric = "risk_buffer"
    sub = (
        risk_df.dropna(subset=[metric])
        .sort_values(metric, ascending=False)
        .head(n)
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(3, 0.30 * len(sub) + 1)))
    else:
        fig = ax.figure

    if sub.empty:
        return _empty_panel(ax, f"no rows with non-null {metric}", savepath)

    labels = [_shorten(s) for s in sub["id"][::-1].astype(str)]
    values = sub[metric][::-1].astype(float).values
    ax.barh(labels, values, color=_COLOR_BASE)
    _label_bars(ax, values, fmt="{:.1f}")
    ax.set_xlabel(metric)
    ax.set_title(f"Top {len(sub)} products by stockout-attention score")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.text(
        0.01, -0.02,
        "Score = risk_buffer × log1p(expected p50). Planning heuristic, not a service level.",
        fontsize=8, color="#666",
    )
    _maybe_save(fig, savepath)
    return ax


# ------------------------------------------------------------------------- #
# 3. Top scenario-sensitive (uses score by default)
# ------------------------------------------------------------------------- #


def plot_top_scenario_sensitive(
    risk_df: pd.DataFrame,
    *,
    n: int = 20,
    metric: str = "scenario_attention_score",
    ax: plt.Axes | None = None,
    savepath: Path | str | None = None,
) -> plt.Axes:
    """Top-n by ``scenario_attention_score = max_abs_delta_p50 * log1p(p50)``
    by default. Falls back to the floored pct if the score is unavailable."""
    if metric not in risk_df.columns or risk_df[metric].isna().all():
        # Fall back to the floored pct, which is also stable.
        metric = "max_abs_scenario_delta_p50_pct_floored"

    if metric not in risk_df.columns or risk_df[metric].isna().all():
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 3))
        else:
            fig = ax.figure
        return _empty_panel(
            ax,
            "no scenario sensitivity available\n(scenario_forecasts not provided)",
            savepath,
        )

    sub = (
        risk_df.dropna(subset=[metric])
        .sort_values(metric, ascending=False)
        .head(n)
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(3, 0.30 * len(sub) + 1)))
    else:
        fig = ax.figure

    if sub.empty:
        return _empty_panel(ax, f"no rows with non-null {metric}", savepath)

    labels = [_shorten(s) for s in sub["id"][::-1].astype(str)]
    values = sub[metric][::-1].astype(float).values
    is_pct = metric.endswith("_pct_floored") or metric.endswith("_pct")
    if is_pct:
        # Show as a percentage.
        display = values * 100.0
        fmt = "{:.1f}%"
    else:
        display = values
        fmt = "{:.1f}"
    ax.barh(labels, display, color="#9467bd")
    _label_bars(ax, display, fmt=fmt)
    ax.set_xlabel(metric + (" (%)" if is_pct else ""))
    ax.set_title(f"Top {len(sub)} scenario-sensitive products by {metric}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.text(
        0.01, -0.02,
        "Scenario sensitivity is predictive (model what-if), not causal.",
        fontsize=8, color="#666",
    )
    _maybe_save(fig, savepath)
    return ax


# ------------------------------------------------------------------------- #
# 4. Item planning demand example
# ------------------------------------------------------------------------- #


def _select_example_id(
    quantile_predictions: pd.DataFrame,
    *,
    quantile_cols: tuple[str, str, str] = ("p10", "p50", "p90"),
) -> str | None:
    p10, p50, p90 = quantile_cols
    if not {"id", p10, p50, p90}.issubset(quantile_predictions.columns):
        return None
    agg = (
        quantile_predictions
        .groupby("id", as_index=False)
        .agg(s_p50=(p50, "sum"), s_p90=(p90, "sum"), s_p10=(p10, "sum"))
    )
    agg["width"] = agg["s_p90"] - agg["s_p10"]
    qualified = agg.loc[(agg["s_p50"] >= 5) & (agg["width"] > 0)]
    if qualified.empty:
        if agg.empty:
            return None
        return str(agg.iloc[agg["s_p50"].idxmax()]["id"])
    qualified = qualified.assign(rel=lambda d: d["width"] / d["s_p50"].clip(lower=1e-9))
    return str(qualified.iloc[qualified["rel"].idxmax()]["id"])


def plot_item_planning_demand_example(
    quantile_predictions: pd.DataFrame,
    *,
    id_: str | None = None,
    quantile_cols: tuple[str, str, str] = ("p10", "p50", "p90"),
    ax: plt.Axes | None = None,
    savepath: Path | str | None = None,
) -> plt.Axes:
    """Cumulative p10/p50/p90 over horizon for one product (auto-picked)."""
    p10, p50, p90 = quantile_cols
    if id_ is None:
        id_ = _select_example_id(quantile_predictions, quantile_cols=quantile_cols)
    if id_ is None:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            fig = ax.figure
        return _empty_panel(
            ax, "no products available for the planning example", savepath,
        )

    sub = quantile_predictions.loc[quantile_predictions["id"] == id_]
    if sub.empty:
        raise ValueError(f"id {id_!r} not found in quantile_predictions")
    if "origin_date" in sub.columns and sub["origin_date"].nunique() > 1:
        latest = sub["origin_date"].max()
        sub = sub.loc[sub["origin_date"] == latest]
    sub = sub.sort_values("horizon")

    h = sub["horizon"].values
    c10 = sub[p10].astype(float).cumsum().values
    c50 = sub[p50].astype(float).cumsum().values
    c90 = sub[p90].astype(float).cumsum().values

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    ax.fill_between(h, c50, c90, color=_COLOR_FILL, alpha=0.6, label="risk_buffer (p90 - p50)")
    ax.plot(h, c10, color=_COLOR_LOW, linestyle="--", marker="o", label="cumulative p10")
    ax.plot(h, c50, color=_COLOR_BASE, marker="o", label="cumulative p50 (expected)")
    ax.plot(h, c90, color=_COLOR_HIGH, marker="o", label="cumulative p90 (conservative)")
    ax.set_xlabel("horizon (days ahead)")
    ax.set_ylabel("cumulative units")
    ax.set_title(f"Planning demand example: {_shorten(id_, 50)}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.text(
        0.01, -0.02,
        "p90 is the conservative planning quantile, not guaranteed demand.",
        fontsize=8, color="#666",
    )
    _maybe_save(fig, savepath)
    return ax


# ------------------------------------------------------------------------- #
# 5. Low-expected / high-upside chart
# ------------------------------------------------------------------------- #


def plot_top_low_expected_high_upside(
    risk_df: pd.DataFrame,
    *,
    n: int = 20,
    p50_threshold: float = 5.0,
    p90_threshold: float = 10.0,
    sort_by: str = "conservative_demand_p90",
    ax: plt.Axes | None = None,
    savepath: Path | str | None = None,
) -> plt.Axes:
    """Items with low expected p50 but meaningful conservative p90.

    These are legitimately planner-relevant (potential upside, hold a
    little buffer) but they don't belong on the uncertainty / scenario
    leaderboards where they'd dominate via near-zero denominators.
    """
    sub = risk_df.loc[
        (risk_df["expected_demand_p50"] < p50_threshold)
        & (risk_df["conservative_demand_p90"] >= p90_threshold)
    ]
    if sort_by in sub.columns:
        sub = sub.sort_values(sort_by, ascending=False)
    sub = sub.head(n)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(3, 0.30 * max(len(sub), 1) + 1)))
    else:
        fig = ax.figure

    if sub.empty:
        return _empty_panel(
            ax,
            f"no products with p50 < {p50_threshold} and p90 >= {p90_threshold}",
            savepath,
        )

    labels = [_shorten(s) for s in sub["id"][::-1].astype(str)]
    p90_vals = sub["conservative_demand_p90"][::-1].astype(float).values
    p50_vals = sub["expected_demand_p50"][::-1].astype(float).values
    ax.barh(labels, p90_vals, color=_COLOR_UPSIDE, label="conservative_demand_p90")
    ax.barh(labels, p50_vals, color=_COLOR_BASE, alpha=0.8, label="expected_demand_p50")
    _label_bars(ax, p90_vals, fmt="{:.1f}")
    ax.set_xlabel("units (cumulative across forecast window)")
    ax.set_title(
        f"Top {len(sub)} low-expected / high-upside products "
        f"(p50 < {p50_threshold:g}, p90 ≥ {p90_threshold:g})"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.text(
        0.01, -0.02,
        "p90 = conservative planning quantile, not guaranteed demand.",
        fontsize=8, color="#666",
    )
    _maybe_save(fig, savepath)
    return ax


# ------------------------------------------------------------------------- #
# 6. 2x2 dashboard
# ------------------------------------------------------------------------- #


def plot_planner_dashboard(
    risk_df: pd.DataFrame,
    quantile_predictions: pd.DataFrame,
    *,
    n: int = 12,
    savepath: Path | str | None = None,
) -> plt.Figure:
    """2x2 portfolio summary using the v1.1 stable metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    plot_top_high_uncertainty(risk_df, n=n, ax=axes[0][0])
    plot_top_stockout_attention(risk_df, n=n, ax=axes[0][1])
    plot_top_scenario_sensitive(risk_df, n=n, ax=axes[1][0])
    plot_item_planning_demand_example(quantile_predictions, ax=axes[1][1])
    fig.suptitle("SeerCast — Planner dashboard (CA_1)", fontsize=13, y=1.005)
    fig.tight_layout()
    fig.text(
        0.01, -0.005,
        "p90 = conservative planning quantile, not guaranteed demand. "
        "Scenario sensitivity is predictive, not causal. "
        "Ratio metrics use a demand floor to avoid near-zero distortions.",
        fontsize=8, color="#666",
    )
    _maybe_save(fig, savepath)
    return fig


__all__ = [
    "plot_top_high_uncertainty",
    "plot_top_stockout_attention",
    "plot_top_scenario_sensitive",
    "plot_item_planning_demand_example",
    "plot_top_low_expected_high_upside",
    "plot_planner_dashboard",
]
