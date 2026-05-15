"""Feature importance loaders for the LightGBM bundles.

Supports both flavours of model bundle:

* **Single-model-per-origin bundles** (Phase 5 LightGBM point, Phase 6
  quantile). Each per-origin entry is a :class:`LightGBMPointModel` or
  :class:`QuantileLightGBMModel` and exposes
  ``feature_importance(kind=...)`` (and ``quantile=`` for quantile).
* **Per-horizon bundles** (the per-horizon experiment). Each per-origin
  entry is a wrapper that holds one sub-model per horizon and exposes
  ``trained_horizons`` plus
  ``feature_importance(horizon=h, kind=...[, quantile=...])``.

The loader auto-detects which flavour it has and aggregates accordingly.
If a bundle exists on disk but every extraction attempt yields zero
rows, we raise a ``RuntimeWarning`` with the offending bundle path and
the structure we observed, so the failure surfaces in the script
output instead of being silently swallowed.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import pandas as pd


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def _trained_horizons(model) -> list[int] | None:
    """Return the per-horizon sub-model keys if ``model`` is a per-horizon
    wrapper, else ``None``.

    We try several attribute names so the loader is tolerant to small
    refactors of the per-horizon wrapper class:

    1. ``trained_horizons`` -- the official @property on the v1 wrappers.
    2. ``_models`` -- the raw underlying dict ``{horizon: sub_model}``.
    3. ``models_by_horizon`` -- a plausible alternate name for the same dict.

    Any value that looks like a non-empty mapping or sequence of horizons
    is accepted.
    """
    for attr in ("trained_horizons", "_models", "models_by_horizon"):
        if not hasattr(model, attr):
            continue
        try:
            val = getattr(model, attr)
        except Exception:
            continue
        if callable(val):
            continue
        if isinstance(val, dict):
            horizons = sorted(val.keys())
        elif isinstance(val, (list, tuple)):
            horizons = list(val)
        else:
            continue
        if not horizons:
            continue
        return [int(h) for h in horizons]
    return None


def _is_per_horizon(model) -> bool:
    """True if the model is a per-horizon wrapper (i.e. we have a list of
    trained horizons we can iterate over)."""
    return _trained_horizons(model) is not None


# --------------------------------------------------------------------------- #
# Per-model extraction
# --------------------------------------------------------------------------- #


def _importance_for_per_horizon_model(
    model,
    horizons: list[int],
    *,
    quantile: float | None,
    kind: str,
) -> pd.DataFrame:
    """Average per-feature importance across the supplied horizons."""
    per_horizon_frames: list[pd.DataFrame] = []
    for h in horizons:
        try:
            if quantile is None:
                imp_h = model.feature_importance(horizon=h, kind=kind)
            else:
                imp_h = model.feature_importance(horizon=h, quantile=quantile, kind=kind)
        except TypeError:
            # Wrapper that doesn't accept horizon= for this kind/quantile path.
            # Skip this horizon rather than crash; report at the end if all empty.
            continue
        except KeyError:
            # Quantile not present in this sub-model's boosters.
            continue
        if imp_h.empty or "feature" not in imp_h.columns or kind not in imp_h.columns:
            continue
        per_horizon_frames.append(
            imp_h.set_index("feature").rename(columns={kind: f"{kind}_h{h}"})
        )
    if not per_horizon_frames:
        return pd.DataFrame(columns=["feature", kind])
    wide = pd.concat(per_horizon_frames, axis=1)
    avg = wide.mean(axis=1, skipna=True).rename(kind).reset_index()
    return avg


def _importance_for_single_model(
    model,
    *,
    quantile: float | None,
    kind: str,
) -> pd.DataFrame:
    """Pull importance from a standard single-model class."""
    try:
        if quantile is None:
            return model.feature_importance(kind=kind)
        return model.feature_importance(quantile=quantile, kind=kind)
    except (TypeError, KeyError):
        return pd.DataFrame(columns=["feature", kind])


def _importance_for_model(
    model,
    *,
    quantile: float | None,
    kind: str,
) -> pd.DataFrame:
    """Dispatch to per-horizon or single-model extraction based on the model shape."""
    horizons = _trained_horizons(model)
    if horizons is not None:
        return _importance_for_per_horizon_model(
            model, horizons, quantile=quantile, kind=kind
        )
    return _importance_for_single_model(model, quantile=quantile, kind=kind)


# --------------------------------------------------------------------------- #
# Bundle-level aggregation
# --------------------------------------------------------------------------- #


def _summarize_bundle_models(models) -> str:
    """Compact one-line summary for a bundle's ``models`` dict. Used in warnings."""
    lines = []
    for k, m in models.items():
        lines.append(
            f"    origin={k!r}: cls={type(m).__name__}, "
            f"trained_horizons={_trained_horizons(m)}"
        )
    return "\n".join(lines) if lines else "    (empty)"


def _agg_importance_from_bundle(
    bundle_path: Path | str,
    *,
    quantile: float | None = None,
    kind: str = "gain",
) -> pd.DataFrame:
    """Average per-feature importance across the per-origin models in a bundle.

    Raises ``RuntimeWarning`` (via :func:`warnings.warn`) if the bundle
    exists and is non-empty but every model produced an empty importance
    frame -- this prevents silent "0 rows" outcomes downstream.
    """
    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        return pd.DataFrame(columns=["feature", kind])

    bundle = joblib.load(bundle_path)
    if "models" not in bundle:
        warnings.warn(
            f"feature_importance: bundle at {bundle_path} has no 'models' key; "
            f"top-level keys: {sorted(bundle.keys())}",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=["feature", kind])

    models = bundle["models"]
    if not models:
        warnings.warn(
            f"feature_importance: bundle at {bundle_path} has an empty 'models' dict",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=["feature", kind])

    rows: list[pd.DataFrame] = []
    for origin_iso, model in models.items():
        imp = _importance_for_model(model, quantile=quantile, kind=kind)
        if imp.empty or "feature" not in imp.columns or kind not in imp.columns:
            continue
        imp = imp.set_index("feature").rename(columns={kind: f"{kind}_{origin_iso}"})
        rows.append(imp)

    if not rows:
        warnings.warn(
            "feature_importance: every model in the bundle produced an empty "
            f"importance frame.\n  bundle: {bundle_path}\n  models:\n"
            f"{_summarize_bundle_models(models)}\n"
            f"  call args: quantile={quantile}, kind={kind!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=["feature", kind])

    wide = pd.concat(rows, axis=1)
    avg = wide.mean(axis=1, skipna=True).rename(kind).reset_index()
    return avg.sort_values(kind, ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Public combined view
# --------------------------------------------------------------------------- #


def feature_importance_combined(
    point_bundle_path: Path | str | None = None,
    quantile_bundle_path: Path | str | None = None,
) -> pd.DataFrame:
    """Combined tidy importance frame.

    Returns columns ``feature, point_gain, point_split, quantile_p50_gain``.
    Any source that's missing is silently skipped (its columns are NaN).
    Per-horizon bundles are auto-detected; their per-horizon importances
    are averaged internally before being concatenated alongside the
    single-model bundles.

    A ``RuntimeWarning`` is raised if at least one bundle path existed on
    disk but the combined result still has zero rows -- this surfaces
    structural problems (wrong class, empty ``models`` dict, etc.) in
    the script output instead of silently writing an empty CSV.
    """
    parts: list[pd.DataFrame] = []
    bundle_paths_seen: list[Path] = []

    if point_bundle_path is not None and Path(point_bundle_path).exists():
        bundle_paths_seen.append(Path(point_bundle_path))
        gain = _agg_importance_from_bundle(point_bundle_path, kind="gain").rename(
            columns={"gain": "point_gain"}
        )
        split = _agg_importance_from_bundle(point_bundle_path, kind="split").rename(
            columns={"split": "point_split"}
        )
        if not gain.empty:
            parts.append(gain.set_index("feature"))
        if not split.empty:
            parts.append(split.set_index("feature"))

    if quantile_bundle_path is not None and Path(quantile_bundle_path).exists():
        bundle_paths_seen.append(Path(quantile_bundle_path))
        q50 = _agg_importance_from_bundle(
            quantile_bundle_path, quantile=0.5, kind="gain"
        ).rename(columns={"gain": "quantile_p50_gain"})
        if not q50.empty:
            parts.append(q50.set_index("feature"))

    if not parts:
        if bundle_paths_seen:
            warnings.warn(
                "feature_importance_combined: bundle(s) existed on disk but "
                "produced zero importance rows. Paths inspected:\n"
                + "\n".join(f"  - {p}" for p in bundle_paths_seen)
                + "\n  Check the per-bundle warnings above for the root cause.",
                RuntimeWarning,
                stacklevel=2,
            )
        return pd.DataFrame(
            columns=["feature", "point_gain", "point_split", "quantile_p50_gain"]
        )

    out = pd.concat(parts, axis=1).reset_index().rename(columns={"index": "feature"})
    sort_col = (
        "point_gain"
        if "point_gain" in out.columns and out["point_gain"].notna().any()
        else ("quantile_p50_gain" if "quantile_p50_gain" in out.columns else None)
    )
    if sort_col is not None:
        out = out.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return out


__all__ = ["feature_importance_combined"]
