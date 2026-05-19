"""Planner-facing plots (v1.2 polish).

v1.2 changes vs v1.1:

* All plot helpers accept ``show_footnote: bool = True``. The dashboard
  passes ``show_footnote=False`` to every subplot and renders a single
  global footer at the bottom of the figure -- no more overlapping
  per-subplot footnotes.
* :func:`plot_top_high_uncertainty` excludes low-expected products by
  default (``include_low_expected=False``). Set
  ``include_low_expected=True`` to restore the v1.1 behaviour.
* :func:`plot_item_planning_demand_example` accepts an optional
  ``risk_df`` so it can pick a *planner-meaningful* example (high p50
  AND meaningful buffer, ranked by ``stockout_attention_score``)
  instead of an item with near-zero p50.
* :func:`plot_top_low_expected_high_upside` updated labels and title
  so p90/p50 distinction is visually obvious; legend is in a fixed
  spot.

All copy keeps the honest framing:
- p50 = expected planning demand
- p90 = conservative planning quantile, not guaranteed demand
- scenario sensitivity is predictive, not causal
- risk labels are percentile heuristics, not service-level guarantees
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Palette.
_COLOR_BASE = "#1f77b4"     # p50 / expected (blue)
_COLOR_HIGH = "#ff7f0e"     # uncertainty (orange)
_COLOR_LOW = "#2ca02c"      # p10 (green)
_COLOR_FILL = "#ffd6a8"     # risk band fill
_COLOR_UPSIDE = "#9467bd"   # p90 / conservative (purple)


# Defaults reused for "meaningful demand" filtering / selection.
_DEFAULT_DEMAND_FLOOR = 10.0
_EXAMPLE_P50_MIN = 50.0
_EXAMPLE_BUFFER_MIN = 50.0


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #


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


def _add_footnote(fig: plt.Figure, text: str, *, show: bool) -> None:
    """Helper: only attach a per-figure footnote when not embedded in a dashboard."""
    if not show:
        return
    fig.text(0.01, -0.02, text, fontsize=8, color="#666")


# --------------------------------------------------------------------------- #
# 1. Top high-uncertainty products
# --------------------------------------------------------------------------- #


def plot_top_high_uncertainty(
    risk_df: pd.DataFrame,
    *,
    n: int = 20,
    metric: str = "relative_uncertainty_floored",
    include_low_expected: bool = False,
    demand_floor: float = _DEFAULT_DEMAND_FLOOR,
    ax: plt.Axes | None = None,
    savepath: Path | str | None = None,
    show_footnote: bool = True,
) -> plt.Axes:
    """Top-n products by ``metric`` (default: floored relative uncertainty).

    Default behaviour (v1.2) excludes products whose
    ``expected_demand_p50 < demand_floor``. Those products belong on the
    low-expected/high-upside chart instead -- they don't muddy the
    high-uncertainty leaderboard. Set ``include_low_expected=True`` to
    restore the v1.1 behaviour and show all rows.
    """
    if metric not in risk_df.columns:
        metric = "relative_uncertainty"

    candidates = risk_df.dropna(subset=[metric])
    if not include_low_expected and "expected_demand_p50" in candidates.columns:
        candidates = candidates.loc[candidates["expected_demand_p50"] >= demand_floor]

    sub = candidates.sort_values(metric, ascending=False).head(n)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(3, 0.30 * max(len(sub), 1) + 1)))
    else:
        fig = ax.figure

    if sub.empty:
        return _empty_panel(
            ax,
            "no high-uncertainty products with meaningful demand\n"
            f"(filter: expected_demand_p50 ≥ {demand_floor:g})",
            savepath,
        )

    labels = [_shorten(s) for s in sub["id"][::-1].astype(str)]
    values = sub[metric][::-1].astype(float).values
    ax.barh(labels, values, color=_COLOR_HIGH)
    _label_bars(ax, values, fmt="{:.2f}")
    ax.set_xlabel(metric)
    qualifier = "" if include_low_expected else f" (p50 ≥ {demand_floor:g})"
    ax.set_title(f"Top {len(sub)} high-uncertainty products by {metric}{qualifier}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _add_footnote(
        fig,
        "Floored denominator avoids huge ratios at near-zero demand. "
        "p90 is a planning quantile, not guaranteed demand.",
        show=show_footnote,
    )
    _maybe_save(fig, savepath)
    return ax


# --------------------------------------------------------------------------- #
# 2. Top stockout-attention
# --------------------------------------------------------------------------- #


def plot_top_stockout_attention(
    risk_df: pd.DataFrame,
    *,
    n: int = 20,
    metric: str = "stockout_attention_score",
    ax: plt.Axes | None = None,
    savepath: Path | str | None = None,
    show_footnote: bool = True,
) -> plt.Axes:
    """Top-n by ``stockout_attention_score = risk_buffer * log1p(p50)``."""
    if metric not in risk_df.columns:
        metric = "risk_buffer"
    sub = (
        risk_df.dropna(subset=[metric])
        .sort_values(metric, ascending=False)
        .head(n)
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(3, 0.30 * max(len(sub), 1) + 1)))
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
    _add_footnote(
        fig,
        "Score = risk_buffer × log1p(expected p50). Planning heuristic, not a service level.",
        show=show_footnote,
    )
    _maybe_save(fig, savepath)
    return ax


# --------------------------------------------------------------------------- #
# 3. Top scenario-sensitive
# --------------------------------------------------------------------------- #


def plot_top_scenario_sensitive(
    risk_df: pd.DataFrame,
    *,
    n: int = 20,
    metric: str = "scenario_attention_score",
    ax: plt.Axes | None = None,
    savepath: Path | str | None = None,
    show_footnote: bool = True,
) -> plt.Axes:
    """Top-n by ``scenario_attention_score = max_abs_delta_p50 * log1p(p50)``."""
    if metric not in risk_df.columns or risk_df[metric].isna().all():
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
        fig, ax = plt.subplots(figsize=(10, max(3, 0.30 * max(len(sub), 1) + 1)))
    else:
        fig = ax.figure

    if sub.empty:
        return _empty_panel(ax, f"no rows with non-null {metric}", savepath)

    labels = [_shorten(s) for s in sub["id"][::-1].astype(str)]
    values = sub[metric][::-1].astype(float).values
    is_pct = metric.endswith("_pct_floored") or metric.endswith("_pct")
    if is_pct:
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
    _add_footnote(
        fig,
        "Scenario sensitivity is predictive (model what-if), not causal.",
        show=show_footnote,
    )
    _maybe_save(fig, savepath)
    return ax


# --------------------------------------------------------------------------- #
# 4. Item planning demand example -- v1.2 smarter selection
# --------------------------------------------------------------------------- #


def _select_example_id(
    quantile_predictions: pd.DataFrame,
    risk_df: pd.DataFrame | None = None,
    *,
    p50_min: float = _EXAMPLE_P50_MIN,
    buffer_min: float = _EXAMPLE_BUFFER_MIN,
    quantile_cols: tuple[str, str, str] = ("p10", "p50", "p90"),
) -> str | None:
    """Pick an example id that's actually useful for a planner.

    Preference order (v1.2):
    1. ``risk_df`` rows where ``expected_demand_p50 >= p50_min`` AND
       ``risk_buffer >= buffer_min``, ranked by
       ``stockout_attention_score`` desc.
    2. ``risk_df`` rows with non-zero p50, ranked by
       ``stockout_attention_score`` desc.
    3. ``risk_df`` rows with non-zero p50, ranked by
       ``expected_demand_p50`` desc.
    4. Any id present in ``risk_df``.
    5. If ``risk_df`` isn't supplied: aggregate ``quantile_predictions``
       on the fly and apply the same preference.
    """
    if risk_df is not None and not risk_df.empty:
        meaningful = risk_df.loc[
            (risk_df["expected_demand_p50"] >= p50_min)
            & (risk_df["risk_buffer"] >= buffer_min)
            & risk_df["expected_demand_p50"].notna()
        ]
        if not meaningful.empty and "stockout_attention_score" in meaningful.columns:
            best = meaningful.sort_values("stockout_attention_score", ascending=False)
            if not best["stockout_attention_score"].isna().all():
                return str(best.iloc[0]["id"])

        nonzero = risk_df.loc[risk_df["expected_demand_p50"] > 0]
        if not nonzero.empty:
            if "stockout_attention_score" in nonzero.columns and not nonzero["stockout_attention_score"].isna().all():
                best = nonzero.sort_values("stockout_attention_score", ascending=False)
                return str(best.iloc[0]["id"])
            best = nonzero.sort_values("expected_demand_p50", ascending=False)
            return str(best.iloc[0]["id"])

        return str(risk_df.iloc[0]["id"])

    # Fallback: aggregate quantile_predictions on the fly.
    p10, p50, p90 = quantile_cols
    if not {"id", p10, p50, p90}.issubset(quantile_predictions.columns):
        return None
    agg = (
        quantile_predictions
        .groupby("id", as_index=False)
        .agg(s_p50=(p50, "sum"), s_p90=(p90, "sum"), s_p10=(p10, "sum"))
    )
    if agg.empty:
        return None
    agg["buffer"] = agg["s_p90"] - agg["s_p50"]
    meaningful = agg.loc[(agg["s_p50"] >= p50_min) & (agg["buffer"] >= buffer_min)]
    if not meaningful.empty:
        return str(meaningful.sort_values("s_p50", ascending=False).iloc[0]["id"])
    nonzero = agg.loc[agg["s_p50"] > 0]
    if not nonzero.empty:
        return str(nonzero.sort_values("s_p50", ascending=False).iloc[0]["id"])
    return str(agg["id"].iloc[0])


def plot_item_planning_demand_example(
    quantile_predictions: pd.DataFrame,
    *,
    id_: str | None = None,
    risk_df: pd.DataFrame | None = None,
    p50_min: float = _EXAMPLE_P50_MIN,
    buffer_min: float = _EXAMPLE_BUFFER_MIN,
    quantile_cols: tuple[str, str, str] = ("p10", "p50", "p90"),
    ax: plt.Axes | None = None,
    savepath: Path | str | None = None,
    show_footnote: bool = True,
) -> plt.Axes:
    """Cumulative p10/p50/p90 for one product. Selection prefers items
    with meaningful demand AND meaningful risk buffer."""
    p10, p50, p90 = quantile_cols
    if id_ is None:
        id_ = _select_example_id(
            quantile_predictions,
            risk_df=risk_df,
            p50_min=p50_min,
            buffer_min=buffer_min,
            quantile_cols=quantile_cols,
        )
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
    _add_footnote(
        fig,
        "p90 is the conservative planning quantile, not guaranteed demand.",
        show=show_footnote,
    )
    _maybe_save(fig, savepath)
    return ax


# --------------------------------------------------------------------------- #
# 5. Low-expected / high-upside (v1.2 clarified labels)
# --------------------------------------------------------------------------- #


def plot_top_low_expected_high_upside(
    risk_df: pd.DataFrame,
    *,
    n: int = 20,
    p50_threshold: float = 5.0,
    p90_threshold: float = 10.0,
    sort_by: str = "conservative_demand_p90",
    ax: plt.Axes | None = None,
    savepath: Path | str | None = None,
    show_footnote: bool = True,
) -> plt.Axes:
    """Items with low expected p50 but meaningful conservative p90.

    v1.2: chart labels make the p50 vs p90 distinction explicit (blue
    for expected, purple for conservative). Legend is anchored in the
    lower-right corner so the bars and labels don't fight for space.
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
            f"no products with p50 < {p50_threshold:g} and p90 ≥ {p90_threshold:g}",
            savepath,
        )

    labels = [_shorten(s) for s in sub["id"][::-1].astype(str)]
    p90_vals = sub["conservative_demand_p90"][::-1].astype(float).values
    p50_vals = sub["expected_demand_p50"][::-1].astype(float).values

    # Draw p90 (purple) first as the full bar, then overlay p50 (blue) so
    # the upside band (purple slice past p50) is visually clear.
    ax.barh(labels, p90_vals, color=_COLOR_UPSIDE,
            label="p90 (conservative planning quantile)")
    ax.barh(labels, p50_vals, color=_COLOR_BASE, alpha=0.85,
            label="p50 (expected demand)")
    _label_bars(ax, p90_vals, fmt="{:.1f}")
    ax.set_xlabel("units (cumulative across forecast window)")
    ax.set_title(
        f"Top {len(sub)} low-expected / high-upside products "
        f"(p50 < {p50_threshold:g}, p90 ≥ {p90_threshold:g})"
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _add_footnote(
        fig,
        "Purple = p90 conservative planning quantile (NOT guaranteed demand). "
        "Blue = p50 expected planning demand.",
        show=show_footnote,
    )
    _maybe_save(fig, savepath)
    return ax


# --------------------------------------------------------------------------- #
# 6. Dashboard -- single global footer
# --------------------------------------------------------------------------- #


_DASHBOARD_FOOTER = (
    "p50 = expected planning demand.   "
    "p90 = conservative planning quantile, not guaranteed demand.   "
    "Scenario sensitivity is predictive, not causal.   "
    "Risk labels are percentile heuristics.   "
    "Ratio metrics use a demand floor to avoid near-zero distortions."
)


def plot_planner_dashboard(
    risk_df: pd.DataFrame,
    quantile_predictions: pd.DataFrame,
    *,
    n: int = 12,
    demand_floor: float = _DEFAULT_DEMAND_FLOOR,
    savepath: Path | str | None = None,
) -> plt.Figure:
    """2x2 portfolio summary using the v1.1 stable metrics.

    v1.2 polish:
    * Each subplot is drawn with ``show_footnote=False``.
    * One global footer at the bottom (no overlap with axes).
    * The example panel receives ``risk_df`` so the auto-selected id is
      a meaningful stockout-attention candidate, not a near-zero item.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    plot_top_high_uncertainty(
        risk_df, n=n, ax=axes[0][0],
        demand_floor=demand_floor,
        include_low_expected=False,
        show_footnote=False,
    )
    plot_top_stockout_attention(
        risk_df, n=n, ax=axes[0][1], show_footnote=False,
    )
    plot_top_scenario_sensitive(
        risk_df, n=n, ax=axes[1][0], show_footnote=False,
    )
    plot_item_planning_demand_example(
        quantile_predictions,
        risk_df=risk_df,
        ax=axes[1][1],
        show_footnote=False,
    )

    fig.suptitle("SeerCast — Planner dashboard (CA_1)", fontsize=13, y=1.005)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    # Single global footer at the bottom. fig.text positions in figure
    # coords so it sits in the reserved bottom band, not on top of axes.
    fig.text(
        0.5, 0.012,
        _DASHBOARD_FOOTER,
        ha="center", va="bottom", fontsize=8, color="#555",
        wrap=True,
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
