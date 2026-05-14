"""Lifecycle / price-availability experiment runner.

Trains LightGBM point + quantile against the supervised table that now
includes the lifecycle features, writes outputs under

    outputs/models/experiments/<name>/
    outputs/reports/experiments/<name>/

(default ``<name> = "lifecycle_features"``), runs the diagnostics layer
on the new outputs, then builds a before-vs-after CSV comparing against
the existing baseline artifacts that are still in ``outputs/reports/``.

The existing ``outputs/reports/lightgbm_*`` artifacts are NEVER
overwritten by this script; they remain the snapshot of the
pre-lifecycle baseline so the comparison is honest.

Run::

    python -m seercast.training.run_lifecycle_experiment

Prereq: rebuild the supervised tables first so the new lifecycle columns
are present (the schema in :mod:`seercast.features.supervised` was
updated as part of this experiment)::

    python -m seercast.training.build_features
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from seercast.config import (
    ARTIFACTS,
    BACKTEST,
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    ensure_dirs,
)
from seercast.evaluation.metrics import score_by_group
from seercast.training import train_lightgbm, train_quantile_lightgbm
from seercast.training import run_diagnostics as diag_module


_DEFAULT_EXPERIMENT_NAME = "lifecycle_features"


def _experiment_paths(experiment_name: str) -> dict[str, Path]:
    models = MODELS_DIR / "experiments" / experiment_name
    reports = REPORTS_DIR / "experiments" / experiment_name
    figures = FIGURES_DIR / "experiments" / experiment_name
    models.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return {
        "models_dir": models,
        "reports_dir": reports,
        "figures_dir": figures,
        "lightgbm_point_model": models / "lightgbm_point_model_ca1.pkl",
        "lightgbm_quantile_models": models / "lightgbm_quantile_models_ca1.pkl",
        "lightgbm_backtest_predictions": reports / "lightgbm_backtest_predictions_ca1.parquet",
        "lightgbm_backtest_scores": reports / "lightgbm_backtest_scores_ca1.csv",
        "model_comparison": reports / "model_comparison_ca1.csv",
        "quantile_backtest_predictions": reports / "quantile_backtest_predictions_ca1.parquet",
        "quantile_backtest_scores": reports / "quantile_backtest_scores_ca1.csv",
        "uncertainty_diagnostics": reports / "uncertainty_diagnostics_ca1.csv",
        "diag_reports_dir": reports / "diagnostics",
        "before_vs_after": reports / "before_vs_after_comparison.csv",
    }


# --------------------------------------------------------------------------- #
# Before/after comparison helper (also useful in tests + notebook)
# --------------------------------------------------------------------------- #


def build_before_vs_after(
    baseline_lgbm_predictions: pd.DataFrame | Path | str,
    baseline_quantile_predictions: pd.DataFrame | Path | str,
    experiment_lgbm_predictions: pd.DataFrame | Path | str,
    experiment_quantile_predictions: pd.DataFrame | Path | str,
    *,
    baseline_label: str = "baseline_features",
    experiment_label: str = "lifecycle_features",
) -> pd.DataFrame:
    """Per (model, version) WAPE/MAE/RMSE/Bias on the matched grid.

    Parameters can be either an in-memory long DataFrame (`model`,
    `origin_date`, `id`, `horizon`, `target_date`, `prediction`,
    `actual` for the point frames; the quantile frames are reshaped from
    their `p50` column).

    Returns a DataFrame with columns
    ``model, version, n, MAE, RMSE, WAPE, Bias`` sorted by
    ``(model, WAPE)``. By construction every (model, version) row has
    the same `n` (the LGBM matched grid size).
    """
    def _read(x):
        if x is None:
            return None
        if isinstance(x, pd.DataFrame):
            return x
        path = Path(x)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    rows: list[pd.DataFrame] = []
    for label, point_src, quantile_src in (
        (baseline_label, baseline_lgbm_predictions, baseline_quantile_predictions),
        (experiment_label, experiment_lgbm_predictions, experiment_quantile_predictions),
    ):
        pt = _read(point_src)
        q = _read(quantile_src)
        if pt is not None:
            pt = pt[["model", "origin_date", "id", "horizon", "target_date",
                     "prediction", "actual"]].copy()
            pt["version"] = label
            rows.append(pt)
        if q is not None:
            q2 = q[["origin_date", "id", "horizon", "target_date", "p50", "actual"]].copy()
            q2 = q2.rename(columns={"p50": "prediction"})
            q2.insert(0, "model", "lightgbm_quantile_p50")
            q2["version"] = label
            rows.append(q2[["model", "origin_date", "id", "horizon", "target_date",
                            "prediction", "actual", "version"]])

    if not rows:
        return pd.DataFrame(
            columns=["model", "version", "n", "MAE", "RMSE", "WAPE", "Bias"]
        )

    combined = pd.concat(rows, ignore_index=True)
    summary = score_by_group(combined, by=("model", "version"))

    # Make sure every (model, version) row has the same n on the matched grid.
    counts = summary.groupby("model")["n"].nunique()
    uneven = counts[counts > 1]
    if not uneven.empty:
        # Don't raise here -- print the warning so the CSV still gets written.
        print(
            f"WARN: before-vs-after n differs between versions for: "
            f"{uneven.index.tolist()}\n{summary[['model','version','n']]}"
        )

    return summary.sort_values(["model", "WAPE"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


def run(
    *,
    experiment_name: str = _DEFAULT_EXPERIMENT_NAME,
    supervised_path: Path | str = ARTIFACTS.train_features_ca1,
    base_path: Path | str = ARTIFACTS.base_table_ca1,
    baseline_predictions_path: Path | str = ARTIFACTS.baseline_predictions_ca1,
    baseline_lgbm_predictions_path: Path | str = ARTIFACTS.lightgbm_backtest_predictions_ca1,
    baseline_quantile_predictions_path: Path | str = ARTIFACTS.quantile_backtest_predictions_ca1,
    backtest_origins: Sequence = BACKTEST.origins,
    skip_point: bool = False,
    skip_quantile: bool = False,
    skip_diagnostics: bool = False,
) -> dict:
    """Run the lifecycle experiment end-to-end.

    Returns a dict with the in-memory results of each step plus the path
    bundle.
    """
    ensure_dirs()
    paths = _experiment_paths(experiment_name)
    print(f"experiment: {experiment_name}")
    print(f"  models  -> {paths['models_dir']}")
    print(f"  reports -> {paths['reports_dir']}")
    print(f"  figures -> {paths['figures_dir']}")
    print()

    # 1. Train LightGBM point with experiment-scoped paths.
    point = None
    if not skip_point:
        print("=" * 64)
        print("[1/3] training LightGBM point on lifecycle-augmented features")
        print("=" * 64)
        point = train_lightgbm.run(
            supervised_path=supervised_path,
            base_path=base_path,
            backtest_origins=backtest_origins,
            baseline_predictions_path=baseline_predictions_path,
            out_model=paths["lightgbm_point_model"],
            out_predictions=paths["lightgbm_backtest_predictions"],
            out_scores=paths["lightgbm_backtest_scores"],
            out_comparison=paths["model_comparison"],
        )

    # 2. Train quantile, pointing at THIS experiment's point preds for the
    # p50-vs-point comparison inside the quantile script.
    quantile = None
    if not skip_quantile:
        print()
        print("=" * 64)
        print("[2/3] training LightGBM quantile (p10/p50/p90)")
        print("=" * 64)
        quantile = train_quantile_lightgbm.run(
            supervised_path=supervised_path,
            base_path=base_path,
            backtest_origins=backtest_origins,
            point_predictions_path=paths["lightgbm_backtest_predictions"],
            out_models=paths["lightgbm_quantile_models"],
            out_predictions=paths["quantile_backtest_predictions"],
            out_scores=paths["quantile_backtest_scores"],
            out_diagnostics=paths["uncertainty_diagnostics"],
        )

    # 3. Run diagnostics on this experiment's outputs.
    diag = None
    if not skip_diagnostics:
        print()
        print("=" * 64)
        print("[3/3] running diagnostics on experiment artifacts")
        print("=" * 64)
        diag = diag_module.run(
            base_table_path=base_path,
            lightgbm_point_path=paths["lightgbm_backtest_predictions"],
            quantile_predictions_path=paths["quantile_backtest_predictions"],
            baseline_predictions_path=baseline_predictions_path,
            point_bundle_path=paths["lightgbm_point_model"],
            quantile_bundle_path=paths["lightgbm_quantile_models"],
            reports_dir=paths["diag_reports_dir"],
            figures_dir=paths["figures_dir"],
        )

    # 4. Build before-vs-after CSV.
    print()
    print("=" * 64)
    print("[4/4] building before-vs-after comparison")
    print("=" * 64)
    before_vs_after = build_before_vs_after(
        baseline_lgbm_predictions=baseline_lgbm_predictions_path,
        baseline_quantile_predictions=baseline_quantile_predictions_path,
        experiment_lgbm_predictions=paths["lightgbm_backtest_predictions"],
        experiment_quantile_predictions=paths["quantile_backtest_predictions"],
    )
    before_vs_after.to_csv(paths["before_vs_after"], index=False)
    print(f"wrote {paths['before_vs_after']}")
    print()
    print(before_vs_after.to_string(index=False))

    return {
        "paths": paths,
        "point": point,
        "quantile": quantile,
        "diagnostics": diag,
        "before_vs_after": before_vs_after,
    }


def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lifecycle/price-availability experiment runner.")
    p.add_argument("--experiment-name", default=_DEFAULT_EXPERIMENT_NAME)
    p.add_argument("--supervised", type=Path, default=ARTIFACTS.train_features_ca1)
    p.add_argument("--base", type=Path, default=ARTIFACTS.base_table_ca1)
    p.add_argument("--skip-point", action="store_true",
                   help="Skip the LightGBM point training step.")
    p.add_argument("--skip-quantile", action="store_true",
                   help="Skip the LightGBM quantile training step.")
    p.add_argument("--skip-diagnostics", action="store_true",
                   help="Skip the diagnostics pass at the end.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    run(
        experiment_name=args.experiment_name,
        supervised_path=args.supervised,
        base_path=args.base,
        skip_point=args.skip_point,
        skip_quantile=args.skip_quantile,
        skip_diagnostics=args.skip_diagnostics,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
