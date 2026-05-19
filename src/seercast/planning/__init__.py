"""Planner-facing reporting layer (v1.1).

Reads existing quantile and scenario forecast artifacts and produces a
per-product risk report aimed at planners (not ML iteration). No models
are retrained here.

v1.1 introduces:
* ``DEFAULT_DEMAND_FLOOR`` (= 10.0) used as the denominator floor for
  ratio metrics, so near-zero p50 items can't dominate via huge ratios.
* ``add_attention_scores`` -- volume-weighted scores combining the
  absolute risk signal with ``log1p(expected_demand_p50)``.
* ``low_expected_high_upside`` -- separate view for items with low p50
  but meaningful p90 (a legitimate planning case in its own right).

Public API:

* :func:`build_quantile_aggregates`
* :func:`add_scenario_sensitivity`
* :func:`add_attention_scores`
* :func:`add_risk_labels`
* :func:`low_expected_high_upside`
* :func:`build_planner_risk_report`
* :func:`top_n_by`
"""

from seercast.planning.risk_report import (
    DEFAULT_DEMAND_FLOOR,
    DEFAULT_LABEL_MAPPING,
    DEFAULT_SCENARIO_NAME_MAP,
    IDENTITY_COLS,
    QUANTILE_COLS,
    add_attention_scores,
    add_risk_labels,
    add_scenario_sensitivity,
    build_planner_risk_report,
    build_quantile_aggregates,
    filter_by_min_expected_p50,
    low_expected_high_upside,
    top_n_by,
)

__all__ = [
    "DEFAULT_DEMAND_FLOOR",
    "IDENTITY_COLS",
    "QUANTILE_COLS",
    "DEFAULT_SCENARIO_NAME_MAP",
    "DEFAULT_LABEL_MAPPING",
    "build_quantile_aggregates",
    "add_scenario_sensitivity",
    "add_attention_scores",
    "add_risk_labels",
    "low_expected_high_upside",
    "filter_by_min_expected_p50",
    "build_planner_risk_report",
    "top_n_by",
]
