"""Diagnostics layer.

Step 1 of the SeerCast improvement roadmap: understand where the models
fail before changing them. Everything in this subpackage reads existing
artifacts and produces breakdown tables / focused plots; no model
retraining happens here.

Public surface:

* :func:`classify_demand_per_origin` — per-(origin, id) Syntetos-Boylan
  segment labels using only history with ``date <= origin_date``.
* :func:`lifecycle_summary` and :func:`is_active_at_origin` — product
  launch dates and per-origin active flag.
* :func:`combined_predictions` — matched-grid long predictions across
  baselines, LGBM point, and ``lightgbm_quantile_p50`` (first-class).
* :func:`breakdown` and :func:`fairness_check` — group-wise WAPE / MAE
  / RMSE / Bias on the matched grid.
* :func:`worst_forecasters` — top-N under/over-forecasters per model.
* :func:`feature_importance_combined` — point + quantile importance
  side by side. Per-horizon bundles auto-detected. Emits a
  ``RuntimeWarning`` if a bundle exists but yields no importance.
* :func:`inspect_bundle` — ad-hoc debugger that prints the structure
  of a joblib model bundle. Use when ``feature_importance_combined``
  returns empty.
"""

from seercast.diagnostics.combined import GRID_COLS, PRED_COLS, combined_predictions
from seercast.diagnostics.demand_segments import SEGMENTS, classify_demand_per_origin
from seercast.diagnostics.error_breakdown import breakdown, fairness_check
from seercast.diagnostics.feature_importance import (
    feature_importance_combined,
    inspect_bundle,
)
from seercast.diagnostics.lifecycle import is_active_at_origin, lifecycle_summary
from seercast.diagnostics.worst_cases import worst_forecasters

__all__ = [
    "GRID_COLS",
    "PRED_COLS",
    "combined_predictions",
    "SEGMENTS",
    "classify_demand_per_origin",
    "breakdown",
    "fairness_check",
    "feature_importance_combined",
    "inspect_bundle",
    "is_active_at_origin",
    "lifecycle_summary",
    "worst_forecasters",
]
