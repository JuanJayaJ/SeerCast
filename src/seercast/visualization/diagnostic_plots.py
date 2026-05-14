"""Diagnostic plots for the Step 1 analysis.

Five focused plots, each consumes a breakdown DataFrame (or the long
combined-predictions frame) and saves a PNG. All functions accept
``ax=None`` to support placing them in a multi-panel layout.

Honest framing: these visuals show *what the comparison reports*. They
don't editorialize about which model is "best."
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Consistent palette: baselines in shades of grey/blue, LGBM point in
# orange, LGBM quantile p50 in green so the eye can find them quickly.
_MODEL_COLORS = {
    "naive": "#9aa0a6",
    "seasonal_naive": "#5f6368",
    "moving_average_28": "#1f77b4",
    "seasonal_moving_average": "#3a78b5",
    "lightgbm_point": "#ff7f0e",
    "lightgbm_quantile_p50": "#2ca02c",
}


def _color(model: str) -> str:
    return _MODEL_COLORS.get(model, "#777777")


def _maybe_save(fig: plt.Figure, savepath: Path | str | None, dpi: int = 160) -> None:
    if savepath is None:
        return
    p = Path(savepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")


def plot_error_by_horizon(
    breakdown_horizon: pd.DataFrame,
    metric: str = "WAPE",
    *,
    ax: plt.Axes | None = None,
    savepath: Path | str | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Line plot of ``metric`` vs horizon, one line per model."""
    df = breakdown_horizon.sort_values(["model", "horizon"])
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure
    for m, sub in df.groupby("model"):
        ax.plot(sub["horizon"], sub[metric], marker="o", color=_color(m), label=m)
    ax.set_xlabel("horizon (days ahead)")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by forecast horizon (matched grid)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _maybe_save(fig, savepath)
    return ax


def plot_metric_by_segment(
    breakdown_segment: pd.DataFrame,
    metric: str = "WAPE",
    *,
    segment_col: str = "segment_at_origin",
    ax: plt.Axes | None = None,
    savepath: Path | str | None = None,
    title: str | None = None,
    reference: float | None = None,
) -> plt.Axes:
    """Grouped bar chart: x = segment, bars per model."""
    df = breakdown_segment.copy()
    segments = sorted(df[segment_col].dropna().unique())
    models = sorted(df["model"].dropna().unique())
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4.5))
    else:
        fig = ax.figure

    x = np.arange(len(segments))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        sub = df.loc[df["model"] == m].set_index(segment_col)[metric]
        vals = [float(sub.get(s, np.nan)) for s in segments]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width, color=_color(m), label=m)

    if reference is not None:
        ax.axhline(reference, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(segments, rotation=0)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by demand segment (matched grid)")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _maybe_save(fig, savepath)
    return ax


def plot_bias_by_segment(
    breakdown_segment: pd.DataFrame,
    *,
    segment_col: str = "segment_at_origin",
    ax: plt.Axes | None = None,
    savepath: Path | str | None = None,
) -> plt.Axes:
    """Grouped bar chart of Bias by segment, with 0 reference line.

    Negative bars = under-forecasting on that segment; positive = over-forecasting.
    """
    return plot_metric_by_segment(
        breakdown_segment,
        metric="Bias",
        segment_col=segment_col,
        ax=ax,
        savepath=savepath,
        title="Bias by demand segment (negative = underforecast)",
        reference=0.0,
    )


def plot_actual_vs_pred_for_top_errors(
    combined: pd.DataFrame,
    *,
    model: str = "lightgbm_quantile_p50",
    n_items: int = 6,
    direction: str = "under",
    savepath: Path | str | None = None,
) -> plt.Figure:
    """Multi-panel actual-vs-prediction over horizon for top-N high-error ids.

    Picks the top ``n_items`` ids with the largest signed error for
    ``model`` and draws one subplot each at the latest origin.
    """
    from seercast.diagnostics.worst_cases import worst_forecasters

    worst = worst_forecasters(combined, model=model, direction=direction, top_n=n_items)
    ids = worst["id"].tolist()
    if not ids:
        raise ValueError("no ids selected -- empty combined frame for this model")

    latest = combined["origin_date"].max()

    ncols = 3
    nrows = int(np.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.0 * nrows), squeeze=False)
    for i, id_ in enumerate(ids):
        ax = axes[i // ncols][i % ncols]
        sub = combined.loc[
            (combined["model"] == model)
            & (combined["id"] == id_)
            & (combined["origin_date"] == latest)
        ].sort_values("horizon")
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.plot(sub["horizon"], sub["actual"], marker="o", color="black", label="actual")
        ax.plot(sub["horizon"], sub["prediction"], marker="o", linestyle="--",
                color=_color(model), label=model)
        ax.set_title(id_, fontsize=9)
        ax.set_xlabel("horizon")
        ax.set_ylabel("units")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=7)
    for j in range(len(ids), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(
        f"Top {len(ids)} {direction}-forecast cases for {model} (origin {pd.Timestamp(latest).date()})",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    _maybe_save(fig, savepath)
    return fig


def plot_feature_importance_top20(
    importance_df: pd.DataFrame,
    *,
    columns: Sequence[str] = ("point_gain", "quantile_p50_gain"),
    top_n: int = 20,
    savepath: Path | str | None = None,
) -> plt.Figure:
    """Side-by-side horizontal bars of the top-N features per importance kind."""
    cols = [c for c in columns if c in importance_df.columns]
    if not cols:
        raise ValueError(f"no usable importance columns; available: {list(importance_df.columns)}")

    fig, axes = plt.subplots(1, len(cols), figsize=(7 * len(cols), 6), squeeze=False)
    for i, c in enumerate(cols):
        sub = importance_df[["feature", c]].dropna().sort_values(c, ascending=False).head(top_n)
        ax = axes[0][i]
        ax.barh(sub["feature"][::-1], sub[c][::-1], color="#3a78b5" if "quantile" not in c else "#2ca02c")
        ax.set_xlabel(c)
        ax.set_title(f"Top {top_n} by {c}")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Feature importance", y=1.02, fontsize=12)
    fig.tight_layout()
    _maybe_save(fig, savepath)
    return fig


__all__ = [
    "plot_error_by_horizon",
    "plot_metric_by_segment",
    "plot_bias_by_segment",
    "plot_actual_vs_pred_for_top_errors",
    "plot_feature_importance_top20",
]
