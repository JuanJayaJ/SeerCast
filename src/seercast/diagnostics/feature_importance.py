"""Feature importance loaders for the LightGBM bundles.

Supports both flavours of bundle:

* **Single-model-per-origin bundles** (Phase 5 LGBM point, Phase 6 quantile).
  Each per-origin entry is a :class:`LightGBMPointModel` or
  :class:`QuantileLightGBMModel`.
* **Per-horizon bundles** (the per-horizon experiment). Each per-origin
  entry is a :class:`PerHorizonLightGBMPointModel` or
  :class:`PerHorizonQuantileLightGBMModel` -- detected via the
  ``trained_horizons`` attribute. For these we loop over the trained
  horizons, pull per-horizon importance, and average across horizons
  before concatenating with single-model paths.

Empty-extraction behaviour
--------------------------

When a bundle on disk yields no importance (because every model
returned empty, or because the bundle's structure is malformed), the
loader emits an explicit ``warnings.warn(..., RuntimeWarning)`` and
returns an empty frame. The previous version silently wrote a 0-row
CSV; this one makes the failure obvious. Use
:func:`inspect_bundle` for an ad-hoc peek into the bundle's structure.

The optional ``strict`` argument is accepted for back-compatibility
with callers that previously passed it (``run_diagnostics`` used to).
It is currently treated as a hint that does not change behaviour;
warnings are always emitted.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def _is_per_horizon(model: Any) -> bool:
    """Per-horizon wrappers expose a ``trained_horizons`` attribute. Single-
    model classes do not."""
    return hasattr(model, "trained_horizons")


def _safe_trained_horizons(model: Any) -> list[int]:
    """Best-effort accessor. Returns ``[]`` if missing or if access raises."""
    try:
        th = model.trained_horizons
        return list(th) if th is not None else []
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(
            f"model.trained_horizons raised: {exc!r}",
            RuntimeWarning, stacklevel=2,
        )
        return []


# --------------------------------------------------------------------------- #
# Inspection (ad-hoc debugger)
# --------------------------------------------------------------------------- #


def inspect_bundle(bundle_path: Path | str) -> dict[str, Any]:
    """Print + return a structured summary of a joblib model bundle.

    Useful when ``feature_importance_combined`` returns empty: this dumps
    top-level keys, per-origin model types, is_per_horizon, trained
    horizons, feature_columns length, and booster state for every entry.
    """
    path = Path(bundle_path)
    if not path.exists():
        print(f"inspect_bundle: {path} does NOT exist")
        return {"exists": False, "path": str(path)}

    bundle = joblib.load(path)
    out: dict[str, Any] = {
        "exists": True,
        "path": str(path),
        "top_level_keys": list(bundle.keys()) if isinstance(bundle, dict) else [],
        "origins": {},
    }
    print(f"inspect_bundle: {path}")
    print(f"  top-level keys: {out['top_level_keys']}")
    if not isinstance(bundle, dict) or "models" not in bundle:
        print(
            f"  ERROR: bundle has no 'models' key; "
            f"type={type(bundle).__name__}"
        )
        return out

    models = bundle["models"]
    print(f"  models: {type(models).__name__} with {len(models)} entries")
    for origin_iso, model in models.items():
        is_ph = _is_per_horizon(model)
        th = _safe_trained_horizons(model) if is_ph else []
        fc = getattr(model, "feature_columns", None)
        fc_len = len(fc) if fc is not None else "n/a"
        booster_state = (
            "set" if getattr(model, "_booster", None) is not None
            else ("dict" if getattr(model, "_boosters", None)
                  else ("per_horizon-dict" if is_ph and th else "MISSING"))
        )
        out["origins"][origin_iso] = {
            "model_type": type(model).__name__,
            "is_per_horizon": is_ph,
            "trained_horizons": th,
            "feature_columns_len": fc_len,
            "booster_state": booster_state,
        }
        print(
            f"    origin={origin_iso}: type={type(model).__name__}, "
            f"is_per_horizon={is_ph}, trained_horizons={th}, "
            f"feature_columns_len={fc_len}, booster_state={booster_state}"
        )
    return out


# --------------------------------------------------------------------------- #
# Per-model importance
# --------------------------------------------------------------------------- #


def _importance_for_model(
    model: Any,
    *,
    quantile: float | None,
    kind: str,
) -> pd.DataFrame:
    """Tidy ``feature, <kind>`` frame for a single model.

    Per-horizon wrappers are auto-detected: their per-horizon importances
    are averaged across :attr:`trained_horizons` before being returned.
    Returns an empty frame if the model can't yield anything.
    """
    if _is_per_horizon(model):
        horizons = _safe_trained_horizons(model)
        if not horizons:
            return pd.DataFrame(columns=["feature", kind])
        per_horizon_frames: list[pd.DataFrame] = []
        for h in horizons:
            if quantile is None:
                imp_h = model.feature_importance(horizon=h, kind=kind)
            else:
                imp_h = model.feature_importance(horizon=h, quantile=quantile, kind=kind)
            if imp_h is None or len(imp_h) == 0:
                continue
            per_horizon_frames.append(
                imp_h.set_index("feature").rename(columns={kind: f"{kind}_h{h}"})
            )
        if not per_horizon_frames:
            return pd.DataFrame(columns=["feature", kind])
        wide = pd.concat(per_horizon_frames, axis=1)
        avg = wide.mean(axis=1).rename(kind).reset_index()
        return avg

    # Single-model path.
    if quantile is None:
        imp = model.feature_importance(kind=kind)
    else:
        imp = model.feature_importance(quantile=quantile, kind=kind)
    if imp is None or len(imp) == 0:
        return pd.DataFrame(columns=["feature", kind])
    return imp


# --------------------------------------------------------------------------- #
# Per-bundle aggregation
# --------------------------------------------------------------------------- #


def _agg_importance_from_bundle(
    bundle_path: Path | str,
    *,
    quantile: float | None = None,
    kind: str = "gain",
    verbose: bool = False,
) -> pd.DataFrame:
    """Average per-feature importance across the per-origin models in a bundle.

    Emits a ``RuntimeWarning`` (with the offending path) if the bundle is
    malformed (no ``'models'`` key) or if every model produces empty
    importance.
    """
    path = Path(bundle_path)
    if not path.exists():
        return pd.DataFrame(columns=["feature", kind])

    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "models" not in bundle:
        top = list(bundle.keys()) if isinstance(bundle, dict) else []
        warnings.warn(
            f"bundle at {path} has no 'models' key; top-level keys={top}",
            RuntimeWarning, stacklevel=2,
        )
        return pd.DataFrame(columns=["feature", kind])

    models = bundle["models"]
    if verbose:
        print(f"  loaded {path.name}: {len(models)} origin(s)")
    if not models:
        warnings.warn(
            f"bundle at {path} has an empty 'models' dict",
            RuntimeWarning, stacklevel=2,
        )
        return pd.DataFrame(columns=["feature", kind])

    rows: list[pd.DataFrame] = []
    for origin_iso, model in models.items():
        try:
            imp = _importance_for_model(model, quantile=quantile, kind=kind)
        except Exception as exc:
            warnings.warn(
                f"importance extraction failed for origin={origin_iso} "
                f"({type(model).__name__}): {exc!r}",
                RuntimeWarning, stacklevel=2,
            )
            continue
        if imp.empty:
            if verbose:
                print(f"    WARN: empty importance for origin={origin_iso}")
            continue
        imp = imp.set_index("feature").rename(columns={kind: f"{kind}_{origin_iso}"})
        rows.append(imp)

    if not rows:
        warnings.warn(
            f"every model in the bundle at {path} produced zero importance rows "
            f"(kind={kind}, quantile={quantile}). "
            "Run seercast.diagnostics.inspect_bundle(path) to debug.",
            RuntimeWarning, stacklevel=2,
        )
        return pd.DataFrame(columns=["feature", kind])

    wide = pd.concat(rows, axis=1)
    avg = wide.mean(axis=1).rename(kind).reset_index()
    return avg.sort_values(kind, ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Public combined entry point
# --------------------------------------------------------------------------- #


def feature_importance_combined(
    point_bundle_path: Path | str | None = None,
    quantile_bundle_path: Path | str | None = None,
    *,
    strict: bool = False,  # accepted for back-compat; ignored
    verbose: bool = False,
) -> pd.DataFrame:
    """Combined tidy importance frame.

    Returns columns ``feature, point_gain, point_split, quantile_p50_gain``.
    Per-horizon bundles are auto-detected via the ``trained_horizons``
    attribute; their per-horizon importances are averaged before being
    concatenated with the single-model paths.

    When a bundle exists on disk but yields no importance, a clear
    :class:`RuntimeWarning` is emitted naming the path and the cause
    (no 'models' key / every model produced zero rows). Use
    :func:`inspect_bundle` to debug.
    """
    del strict  # back-compat parameter; ignored.
    parts: list[pd.DataFrame] = []

    if point_bundle_path is not None and Path(point_bundle_path).exists():
        gain = _agg_importance_from_bundle(
            point_bundle_path, kind="gain", verbose=verbose,
        ).rename(columns={"gain": "point_gain"})
        split = _agg_importance_from_bundle(
            point_bundle_path, kind="split", verbose=verbose,
        ).rename(columns={"split": "point_split"})
        if not gain.empty:
            parts.append(gain.set_index("feature"))
        if not split.empty:
            parts.append(split.set_index("feature"))

    if quantile_bundle_path is not None and Path(quantile_bundle_path).exists():
        q50 = _agg_importance_from_bundle(
            quantile_bundle_path, quantile=0.5, kind="gain", verbose=verbose,
        ).rename(columns={"gain": "quantile_p50_gain"})
        if not q50.empty:
            parts.append(q50.set_index("feature"))

    if not parts:
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


__all__ = [
    "feature_importance_combined",
    "inspect_bundle",
]
