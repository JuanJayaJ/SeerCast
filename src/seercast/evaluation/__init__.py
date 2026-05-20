"""Evaluation: point metrics, probabilistic metrics, rolling-origin
backtesting, and (Phase 8) credibility-pass utilities — bias correction,
clustered bootstrap CI, RMSSE/WRMSSE, and quantile calibration."""

from seercast.evaluation.backtesting import origin_to_date, rolling_origin_backtest
from seercast.evaluation.bias_correction import (
    SUPPORTED_METHODS as BIAS_CORRECTION_METHODS,
    BiasCorrector,
    compare_corrections,
)
from seercast.evaluation.bootstrap import (
    DEFAULT_KEYS as BOOTSTRAP_DEFAULT_KEYS,
    BootstrapResult,
    clustered_bootstrap_metric_diffs,
    matched_grid,
)
from seercast.evaluation.calibration import (
    ConformalScaler,
    build_calibration_report,
    coverage_by_group,
    overall_coverage,
    per_tail_coverage,
)
from seercast.evaluation.metrics import (
    all_point_metrics,
    bias,
    mae,
    rmse,
    score_by_group,
    wape,
)
from seercast.evaluation.probabilistic_metrics import (
    all_quantile_metrics,
    coverage,
    crossing_rate,
    fix_quantile_crossings,
    interval_width,
    pinball_loss,
    relative_interval_width,
    score_quantile_by_group,
)
from seercast.evaluation.wrmsse import (
    compute_scale_per_id,
    hierarchical_rmsse,
    rmsse_for_series,
    score_rmsse_at_level,
    weighted_rmsse,
)

__all__ = [
    "mae",
    "rmse",
    "wape",
    "bias",
    "all_point_metrics",
    "score_by_group",
    "rolling_origin_backtest",
    "origin_to_date",
    "pinball_loss",
    "coverage",
    "interval_width",
    "relative_interval_width",
    "crossing_rate",
    "fix_quantile_crossings",
    "all_quantile_metrics",
    "score_quantile_by_group",
    # phase 8: credibility pass
    "BIAS_CORRECTION_METHODS",
    "BiasCorrector",
    "compare_corrections",
    "BOOTSTRAP_DEFAULT_KEYS",
    "BootstrapResult",
    "clustered_bootstrap_metric_diffs",
    "matched_grid",
    "ConformalScaler",
    "build_calibration_report",
    "coverage_by_group",
    "overall_coverage",
    "per_tail_coverage",
    "compute_scale_per_id",
    "hierarchical_rmsse",
    "rmsse_for_series",
    "score_rmsse_at_level",
    "weighted_rmsse",
]
