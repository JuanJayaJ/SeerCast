"""Per-horizon quantile + point experiment runner.

Trains one quantile LightGBM (p10/p50/p90) per horizon in
``DIRECT_HORIZONS = [1, 7, 14, 28]`` and, by default, one Poisson point
booster per horizon. Outputs go under

    outputs/models/experiments/per_horizon/
    outputs/reports/experiments/per_horizon/
    outputs/figures/experiments/per_horizon/

The pre-existing baseline and lifecycle-experiment artifacts are
NEVER overwritten. The before-vs-after CSV reads them as comparison
benchmarks alongside the per-horizon outputs.

Leakage rules are unchanged:

    train = supervised[ target_date <= O - VALID_WINDOW_DAYS ]   filtered per horizon
    valid = supervised[ O - VALID_WINDOW_DAYS < target_date <= O ]   filtered per horizon
    test  = supervised[ origin_date == O ]   (all horizons evaluated together)

Run::

    python -m seercast.training.run_per_horizon_experiment
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from seercast.config import (
    ARTIFACTS,
    BACKTEST,
    DIRECT_HORIZONS,
    FIGURES_DIR,
    MODELS_DIR,
    QUANTILES,
    REPORTS_DIR,
    ensure_dirs,
)
from seercast.evaluation.backtesting import origin_to_date
from seercast.evaluation.metrics import score_by_group
from seercast.evaluation.probabilistic_metrics import (
    crossing_rate,
    score_quantile_by_group,
)
from seercast.features.supervised import validate_supervised_table
from seercast.models.per_horizon_lightgbm import (
    PerHorizonLightGBMPointModel,
    PerHorizonQuantileLightGBMModel,
)
from seercast.models.quantile_lightgbm import quantile_column_name
from seercast.training import run_diagnostics as diag_module
from seercast.training.train_lightgbm import (
    VALID_WINDOW_DAYS,
    split_for_backtest_origin,
)


_EXPERIMENT_NAME = "per_horizon"


def _experiment_paths(name: str = _EXPERIMENT_NAME) -> dict[str, Path]:
    models = MODELS_DIR / "experiments" / name
    reports = REPORTS_DIR / "experiments" / name
    figures = FIGURES_DIR / "experiments" / name
    for d in (models, reports, figures):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "models_dir": models,
        "reports_dir": reports,
        "figures_dir": figures,
        # Quantile artifacts.
        "quantile_models": models / "per_horizon_quantile_models_ca1.pkl",
        "quantile_predictions": reports / "per_horizon_quantile_predictions_ca1.parquet",
        "quantile_scores": reports / "per_horizon_quantile_scores_ca1.csv",
        "uncertainty_diagnostics": reports / "uncertainty_diagnostics_ca1.csv",
        # Point artifacts.
        "point_models": models / "per_horizon_point_models_ca1.pkl",
        "point_predictions": reports / "per_horizon_point_predictions_ca1.parquet",
        "point_scores": reports / "per_horizon_point_scores_ca1.csv",
        # Comparison.
        "before_vs_after": reports / "before_vs_after_comparison.csv",
        "before_vs_after_by_horizon": reports / "before_vs_after_by_horizon.csv",
        "diag_reports_dir": reports / "diagnostics",
    }


# --------------------------------------------------------------------------- #
# Per-origin train + predict (quantile + point)
# --------------------------------------------------------------------------- #


def _quantile_predictions_frame(
    test_df: pd.DataFrame,
    fixed: pd.DataFrame,
    raw: pd.DataFrame,
    model_name: str,
    quantile_cols: Sequence[str],
) -> pd.DataFrame:
    out = pd.DataFrame({
        "model": model_name,
        "origin_date": test_df["origin_date"].values,
        "id": test_df["id"].values,
        "horizon": test_df["horizon"].values,
        "target_date": test_df["target_date"].values,
        "actual": test_df["target_sales"].astype(float).values,
    })
    for c in quantile_cols:
        out[c] = fixed[c].values
        out[f"{c}_raw"] = raw[c].values
    return out


def _point_predictions_frame(
    test_df: pd.DataFrame,
    preds: np.ndarray,
    model_name: str = "per_horizon_lightgbm_point",
) -> pd.DataFrame:
    return pd.DataFrame({
        "model": model_name,
        "origin_date": test_df["origin_date"].values,
        "id": test_df["id"].values,
        "horizon": test_df["horizon"].values,
        "target_date": test_df["target_date"].values,
        "prediction": preds,
        "actual": test_df["target_sales"].astype(float).values,
    })


# --------------------------------------------------------------------------- #
# Before-vs-after across three versions (baseline / lifecycle / per_horizon)
# --------------------------------------------------------------------------- #


def _read_optional(x) -> pd.DataFrame | None:
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
        return x
    path = Path(x)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _point_long(df: pd.DataFrame | None, model_label: str) -> pd.DataFrame | None:
    if df is None:
        return None
    keep = ["model", "origin_date", "id", "horizon", "target_date", "prediction", "actual"]
    out = df.copy()
    if "model" not in out.columns:
        out["model"] = model_label
    return out.loc[:, keep]


def _quantile_p50_long(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    out = df.loc[:, ["origin_date", "id", "horizon", "target_date", "p50", "actual"]].copy()
    out = out.rename(columns={"p50": "prediction"})
    out.insert(0, "model", "lightgbm_quantile_p50")
    return out


def build_three_way_comparison(
    *,
    baseline_point: pd.DataFrame | Path | str | None,
    baseline_quantile: pd.DataFrame | Path | str | None,
    lifecycle_point: pd.DataFrame | Path | str | None,
    lifecycle_quantile: pd.DataFrame | Path | str | None,
    per_horizon_quantile: pd.DataFrame | Path | str | None,
    per_horizon_point: pd.DataFrame | Path | str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (overall, per_horizon) before-vs-after tables across 3 versions.

    Each row in the returned overall table is ``(model, version)`` with the
    matched-grid MAE / RMSE / WAPE / Bias / n. The per-horizon table adds
    ``horizon`` to the group keys.
    """
    versions = (
        ("baseline_features", baseline_point, baseline_quantile),
        ("lifecycle_features", lifecycle_point, lifecycle_quantile),
        ("per_horizon", per_horizon_point, per_horizon_quantile),
    )

    rows: list[pd.DataFrame] = []
    for label, point_src, quantile_src in versions:
        pt = _point_long(_read_optional(point_src), model_label="lightgbm_point")
        if pt is not None:
            # Per-horizon point model carries its own name; baseline/lifecycle
            # use "lightgbm_point". Group under "lightgbm_point*" later.
            pt["version"] = label
            rows.append(pt.assign(model="lightgbm_point" if label != "per_horizon"
                                  else "per_horizon_lightgbm_point"))
        q = _quantile_p50_long(_read_optional(quantile_src))
        if q is not None:
            q["version"] = label
            if label == "per_horizon":
                q = q.assign(model="per_horizon_lightgbm_quantile_p50")
            rows.append(q)

    if not rows:
        return (
            pd.DataFrame(columns=["model","version","n","MAE","RMSE","WAPE","Bias"]),
            pd.DataFrame(columns=["model","version","horizon","n","MAE","RMSE","WAPE","Bias"]),
        )

    combined = pd.concat(rows, ignore_index=True)

    overall = score_by_group(combined, by=("model", "version")).sort_values(["model", "WAPE"]).reset_index(drop=True)
    by_horizon = score_by_group(combined, by=("model", "version", "horizon")).sort_values(["horizon", "WAPE"]).reset_index(drop=True)

    return overall, by_horizon


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


def run(
    *,
    supervised_path: Path | str = ARTIFACTS.train_features_ca1,
    base_path: Path | str = ARTIFACTS.base_table_ca1,
    backtest_origins: Sequence = BACKTEST.origins,
    valid_window_days: int = VALID_WINDOW_DAYS,
    horizons: Sequence[int] = DIRECT_HORIZONS,
    quantiles: Sequence[float] = QUANTILES,
    train_quantile: bool = True,
    train_point: bool = True,
    run_diagnostics_pass: bool = True,
    diagnostics_only: bool = False,
    baseline_predictions_path: Path | str = ARTIFACTS.baseline_predictions_ca1,
    # Pre-experiment benchmarks (Phase 5/6 outputs).
    baseline_point_predictions: Path | str = ARTIFACTS.lightgbm_backtest_predictions_ca1,
    baseline_quantile_predictions: Path | str = ARTIFACTS.quantile_backtest_predictions_ca1,
    # Lifecycle-experiment benchmarks.
    lifecycle_point_predictions: Path | str = (
        REPORTS_DIR / "experiments" / "lifecycle_features"
        / "lightgbm_backtest_predictions_ca1.parquet"
    ),
    lifecycle_quantile_predictions: Path | str = (
        REPORTS_DIR / "experiments" / "lifecycle_features"
        / "quantile_backtest_predictions_ca1.parquet"
    ),
) -> dict:
    """Train + backtest per-horizon models, write deliverables.

    Returns the in-memory bundles for notebook inspection.
    """
    ensure_dirs()
    paths = _experiment_paths()
    print(f"experiment: per-horizon (horizons={list(horizons)}, quantiles={list(quantiles)})")
    print(f"  models  -> {paths['models_dir']}")
    print(f"  reports -> {paths['reports_dir']}")
    print(f"  figures -> {paths['figures_dir']}")
    print()

    sup = pd.read_parquet(supervised_path)
    print(f"supervised: {sup.shape}")
    validate_supervised_table(sup, strict=True)
    base = pd.read_parquet(base_path, columns=["id", "d", "date"])
    backtest_dates = [pd.Timestamp(origin_to_date(base, o)) for o in backtest_origins]
    print(f"backtest origins: {[d.date() for d in backtest_dates]}")
    print()

    quantile_cols = [quantile_column_name(q) for q in quantiles]
    h_list = list(horizons)

    # ------- diagnostics-only: read existing experiment artifacts -------- #
    if diagnostics_only:
        train_quantile = False
        train_point = False
        print("diagnostics_only=True -- skipping training, reading existing artifacts")

    # ------- quantile training (one per-horizon set of models per origin) -- #
    quantile_preds_chunks: list[pd.DataFrame] = []
    quantile_bundles: dict[str, PerHorizonQuantileLightGBMModel] = {}
    if train_quantile:
        print("=" * 64)
        print("[1] per-horizon QUANTILE training")
        print("=" * 64)
        for origin_date in backtest_dates:
            split = split_for_backtest_origin(sup, origin_date, valid_window_days)
            print(f"origin {origin_date.date()}: "
                  f"train={len(split.train):,}, valid={len(split.valid):,}, test={len(split.test):,}")
            if len(split.train) == 0 or len(split.test) == 0:
                print("  WARN: empty split, skipping")
                continue
            m = PerHorizonQuantileLightGBMModel(
                horizons=tuple(h_list),
                quantiles=tuple(quantiles),
            )
            m.fit(split.train, valid_df=split.valid)
            fixed = m.predict(split.test, fix_crossings=True)
            raw = m.predict(split.test, fix_crossings=False)
            quantile_preds_chunks.append(
                _quantile_predictions_frame(
                    split.test, fixed, raw,
                    model_name="per_horizon_lightgbm_quantile",
                    quantile_cols=quantile_cols,
                )
            )
            quantile_bundles[origin_date.isoformat()] = m

    # ------- point training (optional) ------------------------------------- #
    point_preds_chunks: list[pd.DataFrame] = []
    point_bundles: dict[str, PerHorizonLightGBMPointModel] = {}
    if train_point:
        print()
        print("=" * 64)
        print("[2] per-horizon POINT training (optional)")
        print("=" * 64)
        for origin_date in backtest_dates:
            split = split_for_backtest_origin(sup, origin_date, valid_window_days)
            print(f"origin {origin_date.date()}: "
                  f"train={len(split.train):,}, valid={len(split.valid):,}, test={len(split.test):,}")
            if len(split.train) == 0 or len(split.test) == 0:
                continue
            m = PerHorizonLightGBMPointModel(horizons=tuple(h_list))
            m.fit(split.train, valid_df=split.valid)
            preds = m.predict(split.test)
            point_preds_chunks.append(_point_predictions_frame(split.test, preds))
            point_bundles[origin_date.isoformat()] = m

    # ------- persist intermediates ---------------------------------------- #
    quantile_predictions = (
        pd.concat(quantile_preds_chunks, ignore_index=True)
        if quantile_preds_chunks else pd.DataFrame()
    )
    point_predictions = (
        pd.concat(point_preds_chunks, ignore_index=True)
        if point_preds_chunks else pd.DataFrame()
    )

    # If we skipped training (e.g. diagnostics_only) but the parquet exists
    # on disk from a prior run, hydrate it so downstream steps still have data.
    if quantile_predictions.empty and Path(paths["quantile_predictions"]).exists():
        print(f"hydrating quantile_predictions from {paths['quantile_predictions']}")
        quantile_predictions = pd.read_parquet(paths["quantile_predictions"])
    if point_predictions.empty and Path(paths["point_predictions"]).exists():
        print(f"hydrating point_predictions from {paths['point_predictions']}")
        point_predictions = pd.read_parquet(paths["point_predictions"])

    if not quantile_predictions.empty:
        quantile_predictions.to_parquet(paths["quantile_predictions"], index=False)

        if quantile_bundles:
            joblib.dump(
                {
                    "models": quantile_bundles,
                    "quantiles": list(quantiles),
                    "horizons": h_list,
                    "valid_window_days": valid_window_days,
                    "backtest_origins": [d.isoformat() for d in backtest_dates],
                },
                paths["quantile_models"],
            )
        elif Path(paths["quantile_models"]).exists():
            print(
                "keeping existing quantile model bundle "
                f"(no trained models in memory): {paths['quantile_models']}"
            )
        else:
            print(
                "WARN: no quantile model bundle written because no trained "
                "models are in memory and no existing bundle was found."
            )
        # By (origin, horizon) probabilistic scores.
        scores = score_quantile_by_group(
            quantile_predictions, by=("origin_date", "horizon"),
            quantile_columns=quantile_cols, quantile_levels=tuple(quantiles),
        )
        scores.to_csv(paths["quantile_scores"], index=False)

        # Uncertainty diagnostics (mirror Phase 6).
        cat_lookup = (
            pd.read_parquet(base_path, columns=["id", "cat_id"]).drop_duplicates("id")
        )
        diag_input = quantile_predictions.merge(cat_lookup, on="id", how="left")
        diag_h = score_quantile_by_group(
            diag_input, by=("horizon",),
            quantile_columns=quantile_cols, quantile_levels=tuple(quantiles),
        )
        diag_h.insert(0, "dimension", "horizon")
        diag_h = diag_h.rename(columns={"horizon": "value"})
        diag_h["value"] = diag_h["value"].astype(str)
        diag_c = score_quantile_by_group(
            diag_input, by=("cat_id",),
            quantile_columns=quantile_cols, quantile_levels=tuple(quantiles),
        )
        diag_c.insert(0, "dimension", "category")
        diag_c = diag_c.rename(columns={"cat_id": "value"})
        diag_c["value"] = diag_c["value"].astype(str)
        diagnostics = pd.concat([diag_h, diag_c], ignore_index=True)
        diagnostics.to_csv(paths["uncertainty_diagnostics"], index=False)
        print(f"wrote {paths['uncertainty_diagnostics']}")

    if not point_predictions.empty:
        point_predictions.to_parquet(paths["point_predictions"], index=False)

        if point_bundles:
            joblib.dump(
                {
                    "models": point_bundles,
                    "horizons": h_list,
                    "valid_window_days": valid_window_days,
                    "backtest_origins": [d.isoformat() for d in backtest_dates],
                },
                paths["point_models"],
            )
        elif Path(paths["point_models"]).exists():
            print(
                "keeping existing point model bundle "
                f"(no trained models in memory): {paths['point_models']}"
            )
        else:
            print(
                "WARN: no point model bundle written because no trained "
                "models are in memory and no existing bundle was found."
            )
        point_scores = score_by_group(
            point_predictions, by=("origin_date", "horizon"),
        )
        point_scores.to_csv(paths["point_scores"], index=False)

    # ------- diagnostics pass on the per-horizon outputs ------------------ #
    if run_diagnostics_pass and not quantile_predictions.empty:
        print()
        print("=" * 64)
        print("[3] diagnostics on per-horizon outputs")
        print("=" * 64)
        # Diagnostics expects a single "lightgbm_point" predictions parquet;
        # we pass the per-horizon point preds here so the matched grid is
        # the same shape.
        point_path = paths["point_predictions"] if not point_predictions.empty \
                     else baseline_point_predictions
        # The diagnostics module reuses the per-origin quantile/point bundles
        # for feature importance; we pass our experiment bundles directly.
        diag_module.run(
            base_table_path=base_path,
            lightgbm_point_path=point_path,
            quantile_predictions_path=paths["quantile_predictions"],
            baseline_predictions_path=baseline_predictions_path,
            point_bundle_path=paths["point_models"] if not point_predictions.empty
                              else ARTIFACTS.lightgbm_point_model_ca1,
            quantile_bundle_path=paths["quantile_models"],
            reports_dir=paths["diag_reports_dir"],
            figures_dir=paths["figures_dir"],
        )

    # ------- three-way before-vs-after ------------------------------------ #
    print()
    print("=" * 64)
    print("[4] three-way before-vs-after (baseline / lifecycle / per_horizon)")
    print("=" * 64)
    overall, by_horizon = build_three_way_comparison(
        baseline_point=baseline_point_predictions,
        baseline_quantile=baseline_quantile_predictions,
        lifecycle_point=lifecycle_point_predictions,
        lifecycle_quantile=lifecycle_quantile_predictions,
        per_horizon_quantile=paths["quantile_predictions"] if not quantile_predictions.empty else None,
        per_horizon_point=paths["point_predictions"] if not point_predictions.empty else None,
    )
    overall.to_csv(paths["before_vs_after"], index=False)
    by_horizon.to_csv(paths["before_vs_after_by_horizon"], index=False)
    print(f"wrote {paths['before_vs_after']}")
    print(f"wrote {paths['before_vs_after_by_horizon']}")
    print()
    print("overall before-vs-after (sorted by model x WAPE):")
    print(overall.to_string(index=False))

    return {
        "paths": paths,
        "quantile_predictions": quantile_predictions,
        "point_predictions": point_predictions,
        "overall": overall,
        "by_horizon": by_horizon,
        "quantile_bundles": quantile_bundles,
        "point_bundles": point_bundles,
    }


def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-horizon LightGBM experiment runner.")
    p.add_argument("--supervised", type=Path, default=ARTIFACTS.train_features_ca1)
    p.add_argument("--base", type=Path, default=ARTIFACTS.base_table_ca1)
    p.add_argument("--skip-point", action="store_true")
    p.add_argument("--skip-quantile", action="store_true")
    p.add_argument("--skip-diagnostics", action="store_true")
    p.add_argument(
        "--diagnostics-only", action="store_true",
        help="Skip all training; read existing predictions parquet and rerun "
             "the diagnostics + before-vs-after only.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    run(
        supervised_path=args.supervised,
        base_path=args.base,
        train_quantile=not args.skip_quantile,
        train_point=not args.skip_point,
        run_diagnostics_pass=not args.skip_diagnostics,
        diagnostics_only=args.diagnostics_only,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
