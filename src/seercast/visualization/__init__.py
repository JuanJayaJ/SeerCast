"""Reusable matplotlib plotting helpers."""

from seercast.visualization.diagnostic_plots import (
    plot_actual_vs_pred_for_top_errors,
    plot_bias_by_segment,
    plot_error_by_horizon,
    plot_feature_importance_top20,
    plot_metric_by_segment,
)
from seercast.visualization.planner_plots import (
    plot_item_planning_demand_example,
    plot_planner_dashboard,
    plot_top_high_uncertainty,
    plot_top_low_expected_high_upside,
    plot_top_scenario_sensitive,
    plot_top_stockout_attention,
)
from seercast.visualization.plots import (
    plot_multiple_scenario_fans,
    plot_scenario_fan,
)

__all__ = [
    "plot_scenario_fan",
    "plot_multiple_scenario_fans",
    "plot_error_by_horizon",
    "plot_metric_by_segment",
    "plot_bias_by_segment",
    "plot_actual_vs_pred_for_top_errors",
    "plot_feature_importance_top20",
    "plot_top_high_uncertainty",
    "plot_top_stockout_attention",
    "plot_top_scenario_sensitive",
    "plot_top_low_expected_high_upside",
    "plot_item_planning_demand_example",
    "plot_planner_dashboard",
]
