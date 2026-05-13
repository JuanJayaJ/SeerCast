"""Reusable matplotlib plotting helpers.

Kept deliberately small: a couple of visualizations the notebooks need
repeatedly, each with sensible defaults but full ``ax``/styling overrides
for callers who want control.

All plotting functions return the ``Axes`` object so the caller can add
titles, save, etc.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------------------------------- #
# Phase 7: per-id base-vs-scenario fan
# --------------------------------------------------------------------------- #


def plot_scenario_fan(
    scenario_forecasts: pd.DataFrame,
    *,
    id_: str,
    scenario: str,
    ax: plt.Axes | None = None,
    title: str | None = None,
    base_color: str = "tab:blue",
    scenario_color: str = "tab:orange",
    actual: pd.Series | None = None,
) -> plt.Axes:
    """Plot the p10-p90 fan and p50 line for one (id, scenario) slice.

    Parameters
    ----------
    scenario_forecasts
        Long DataFrame from :func:`seercast.scenario.simulate_scenarios`.
        Must contain ``scenario``, ``id``, ``horizon`` and the six quantile
        columns (``base_p10/p50/p90``, ``scenario_p10/p50/p90``).
    id_
        The series id to plot.
    scenario
        Scenario name (e.g. ``"price_+10pct"``).
    ax
        Optional pre-built axes. If ``None``, a new figure is created.
    title
        Override the default title.
    base_color, scenario_color
        Override the default colors.
    actual
        Optional Series of actuals indexed by horizon (for the rare case
        we have ground truth at scenario time -- usually not).

    Returns
    -------
    matplotlib.axes.Axes
    """
    sub = scenario_forecasts.loc[
        (scenario_forecasts["id"] == id_) & (scenario_forecasts["scenario"] == scenario)
    ].sort_values("horizon")
    if sub.empty:
        raise ValueError(
            f"no rows for id={id_!r} and scenario={scenario!r} in scenario_forecasts"
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))

    h = sub["horizon"].to_numpy()

    ax.fill_between(
        h, sub["base_p10"], sub["base_p90"],
        color=base_color, alpha=0.15, label="base p10-p90",
    )
    ax.fill_between(
        h, sub["scenario_p10"], sub["scenario_p90"],
        color=scenario_color, alpha=0.20, label=f"{scenario} p10-p90",
    )
    ax.plot(h, sub["base_p50"], color=base_color, marker="o", label="base p50")
    ax.plot(
        h, sub["scenario_p50"],
        color=scenario_color, marker="o", linestyle="--",
        label=f"{scenario} p50",
    )

    if actual is not None:
        common = sub["horizon"].isin(actual.index)
        if common.any():
            ax.plot(
                sub.loc[common, "horizon"].values,
                actual.reindex(sub.loc[common, "horizon"].values).values,
                color="black", marker=".", linestyle=":",
                label="actual",
            )

    ax.set_xlabel("horizon (days ahead)")
    ax.set_ylabel("predicted units")
    ax.set_title(title or f"{id_} - base vs {scenario}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    return ax


def plot_multiple_scenario_fans(
    scenario_forecasts: pd.DataFrame,
    *,
    id_: str,
    scenarios: Sequence[str] | None = None,
    ncols: int = 2,
    figsize_per_subplot: tuple[float, float] = (8.5, 3.0),
) -> plt.Figure:
    """Convenience wrapper: one fan plot per scenario for a given id.

    If ``scenarios`` is ``None``, plots every scenario present in the frame.
    """
    if scenarios is None:
        scenarios = list(
            scenario_forecasts.loc[scenario_forecasts["id"] == id_, "scenario"].unique()
        )
    if not scenarios:
        raise ValueError(f"no scenarios found for id={id_!r}")

    n = len(scenarios)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows),
        squeeze=False,
    )
    for i, scen in enumerate(scenarios):
        ax = axes[i // ncols][i % ncols]
        plot_scenario_fan(scenario_forecasts, id_=id_, scenario=scen, ax=ax)
    # Hide any unused subplots.
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(f"Scenario fans for {id_}", y=1.02, fontsize=12)
    fig.tight_layout()
    return fig


__all__ = ["plot_scenario_fan", "plot_multiple_scenario_fans"]
