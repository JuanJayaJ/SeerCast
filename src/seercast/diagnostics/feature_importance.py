"""Feature importance loaders for the LightGBM bundles.

Both the point bundle (Phase 5) and the quantile bundle (Phase 6) store
one model per backtest origin. For diagnostics we want a single tidy
table: ``feature, point_gain, point_split, quantile_p50_gain``. We
aggregate across origins by averaging — that smooths out per-origin
noise and gives a stable global ranking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd


def _agg_importance_from_bundle(
    bundle_path: Path | str,
    *,
    quantile: float | None = None,
    kind: str = "gain",
) -> pd.DataFrame:
    """Average per-feature importance across the per-origin models in a bundle.

    For the point bundle, ``quantile`` is ignored (single-output model).
    For the quantile bundle, ``quantile`` selects which booster's
    importance to read (default 0.5).
    """
    bundle = joblib.load(Path(bundle_path))
    models = bundle["models"]
    rows: list[pd.DataFrame] = []
    for origin_iso, model in models.items():
        if quantile is None:
            imp = model.feature_importance(kind=kind)
        else:
            imp = model.feature_importance(quantile=quantile, kind=kind)
        imp = imp.rename(columns={kind: f"{kind}_{origin_iso}"})
        rows.append(imp.set_index("feature"))
    if not rows:
        return pd.DataFrame(columns=["feature", kind])
    wide = pd.concat(rows, axis=1)
    avg = wide.mean(axis=1).rename(kind).reset_index()
    return avg.sort_values(kind, ascending=False).reset_index(drop=True)


def feature_importance_combined(
    point_bundle_path: Path | str | None = None,
    quantile_bundle_path: Path | str | None = None,
) -> pd.DataFrame:
    """Combined tidy importance frame.

    Returns columns ``feature, point_gain, point_split, quantile_p50_gain``.
    Any source that's missing is silently skipped (its columns are NaN).
    """
    parts: list[pd.DataFrame] = []
    if point_bundle_path is not None and Path(point_bundle_path).exists():
        gain = _agg_importance_from_bundle(point_bundle_path, kind="gain").rename(
            columns={"gain": "point_gain"}
        )
        split = _agg_importance_from_bundle(point_bundle_path, kind="split").rename(
            columns={"split": "point_split"}
        )
        parts.append(gain.set_index("feature"))
        parts.append(split.set_index("feature"))
    if quantile_bundle_path is not None and Path(quantile_bundle_path).exists():
        q50 = _agg_importance_from_bundle(
            quantile_bundle_path, quantile=0.5, kind="gain"
        ).rename(columns={"gain": "quantile_p50_gain"})
        parts.append(q50.set_index("feature"))

    if not parts:
        return pd.DataFrame(columns=["feature", "point_gain", "point_split", "quantile_p50_gain"])

    out = pd.concat(parts, axis=1).reset_index().rename(columns={"index": "feature"})
    # Sort by point_gain when available, else quantile_p50_gain.
    sort_col = "point_gain" if "point_gain" in out.columns and out["point_gain"].notna().any() \
               else ("quantile_p50_gain" if "quantile_p50_gain" in out.columns else None)
    if sort_col is not None:
        out = out.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return out


__all__ = ["feature_importance_combined"]
