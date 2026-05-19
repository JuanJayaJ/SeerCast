"""Step 1 of the SeerCast improvement roadmap: produce the diagnostics report.

Reads all existing artifacts and emits:

    outputs/reports/diagnostics/
      error_breakdown_by_horizon.csv
      error_breakdown_by_cat.csv
      error_breakdown_by_dept.csv
      error_breakdown_by_segment.csv
      error_breakdown_by_target_zero.csv
      error_breakdown_by_is_active.csv
      demand_segments_ca1.parquet
      lifecycle_summary_ca1.parquet
      is_active_at_origin_ca1.parquet
      worst_underforecast_<model>_top20.csv         (one per model)
      worst_overforecast_<model>_top20.csv          (one per model)
      feature_importance_combined.csv

    outputs/figures/diagnostics/
      error_by_horizon_wape.png
      error_by_segment_wape.png
      bias_by_segment.png
      actual_vs_pred_top_errors.png
      feature_importance_top20.png

No model training happens here. Everything reads the parquet/csv outputs
of earlier phases. Demand segments are leakage-safe (computed using
``date <= origin_date`` per origin), and the combined predictions frame
enforces equal `n` across all models.

Run::

    python -m seercast.training.run_diagnostics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from seercast.config import (
    ARTIFACTS,
    BACKTEST,
    FIGURES_DIR,
    REPORTS_DIR,
    ensure_dirs,
)
from seercast.diagnostics import (
    breakdown,
    classify_demand_per_origin,
    combined_predictions,
    feature_importance_combined,
    is_active_at_origin,
    lifecycle_summary,
    worst_forecasters,
)
from seercast.evaluation.backtesting import origin_to_date
from seercast.visualization.diagnostic_plots import (
    plot_actual_vs_pred_for_top_errors,
    plot_bias_by_segment,
    plot_error_by_horizon,
    plot_feature_importance_top20,
    plot_metric_by_segment,
)


_DIAG_REPORTS_DIR = REPORTS_DIR / "diagnostics"
_DIAG_FIGURES_DIR = FIGURES_DIR / "diagnostics"


def _resolve_backtest_dates(base_path: Path) -> list[pd.Timestamp]:
    base = pd.read_parquet(base_path, columns=["id", "d", "date"])
    return [pd.Timestamp(origin_to_date(base, o)) for o in BACKTEST.origins]


def run(
    base_table_path: Path | str = ARTIFACTS.base_table_ca1,
    lightgbm_point_path: Path | str = ARTIFACTS.lightgbm_backtest_predictions_ca1,
    quantile_predictions_path: Path | str = ARTIFACTS.quantile_backtest_predictions_ca1,
    baseline_predictions_path: Path | str = ARTIFACTS.baseline_predictions_ca1,
    point_bundle_path: Path | str = ARTIFACTS.lightgbm_point_model_ca1,
    quantile_bundle_path: Path | str = ARTIFACTS.lightgbm_quantile_models_ca1,
    reports_dir: Path | str = _DIAG_REPORTS_DIR,
    figures_dir: Path | str = _DIAG_FIGURES_DIR,
) -> dict:
    """End-to-end: build all diagnostic tables and figures.

    Returns a dict containing the intermediate frames (``combined``,
    ``segments``, ``lifecycle``, ``is_active``, plus the breakdown
    tables keyed by axis name) so notebooks can poke at them.
    """
    ensure_dirs()
    reports_dir = Path(reports_dir); reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(figures_dir); figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Backtest origin dates (used for leakage-safe per-origin segments).
    backtest_dates = _resolve_backtest_dates(Path(base_table_path))
    print(f"backtest origins: {[d.date() for d in backtest_dates]}")

    # 2. Demand segments (per origin) and lifecycle (per id + per origin).
    base = pd.read_parquet(base_table_path,
                           columns=["id", "date", "sales", "cat_id", "dept_id"])
    print("classifying demand per origin ...")
    segments = classify_demand_per_origin(base, backtest_dates)
    print(f"  segments rows: {len(segments):,}  "
          f"(unique segments: {sorted(segments['segment_at_origin'].unique())})")

    print("computing lifecycle summary ...")
    lifecycle = lifecycle_summary(base)
    print(f"  lifecycle rows: {len(lifecycle):,}  "
          f"(active items: {int(lifecycle['any_sale'].sum()):,} / {len(lifecycle):,})")

    print("computing is_active_at_origin ...")
    active = is_active_at_origin(base, backtest_dates)
    print(f"  active-at-origin rows: {len(active):,}")

    segments.to_parquet(reports_dir / "demand_segments_ca1.parquet", index=False)
    lifecycle.to_parquet(reports_dir / "lifecycle_summary_ca1.parquet", index=False)
    active.to_parquet(reports_dir / "is_active_at_origin_ca1.parquet", index=False)

    # 3. Combined predictions frame (LGBM grid; baselines + p50 injected;
    #    cat/dept/segment/is_active joined).
    print("building combined predictions ...")
    combined = combined_predictions(
        lightgbm_point_path=lightgbm_point_path,
        quantile_predictions_path=quantile_predictions_path,
        baseline_predictions_path=baseline_predictions_path,
        base_table_path=base_table_path,
        segments_at_origin=segments,
        is_active_at_origin_df=active,
        enforce_equal_n=True,
    )
    print(f"  combined rows: {len(combined):,}  "
          f"(per-model: {combined.groupby('model').size().to_dict()})")

    # 4. Breakdowns.
    print("computing error breakdowns ...")
    breakdowns: dict[str, pd.DataFrame] = {}
    for axis in [
        "horizon",
        "cat_id",
        "dept_id",
        "segment_at_origin",
        "target_zero",
        "is_active",
    ]:
        try:
            t = breakdown(combined, by=axis)
        except Exception as e:
            print(f"  WARN: breakdown by {axis} skipped ({e})")
            continue
        breakdowns[axis] = t
        out_path = reports_dir / f"error_breakdown_by_{axis}.csv"
        t.to_csv(out_path, index=False)
        print(f"  wrote {out_path}  ({len(t):,} rows)")

    # 5. Worst forecasters per model (under + over).
    print("computing worst-case lists ...")
    models = sorted(combined["model"].unique())
    for m in models:
        for direction in ("under", "over"):
            t = worst_forecasters(combined, model=m, direction=direction, top_n=20)
            out_path = reports_dir / f"worst_{direction}forecast_{m}_top20.csv"
            t.to_csv(out_path, index=False)
    print(f"  wrote {len(models) * 2} worst-case CSVs")

    # 6. Feature importance.
    print("computing feature importance ...")
    fi = feature_importance_combined(
        point_bundle_path=point_bundle_path,
        quantile_bundle_path=quantile_bundle_path,
        strict=False,   # WARN instead of raise so the rest of the diagnostics
                        # run still completes if a bundle is problematic.
        verbose=True,
    )
    fi.to_csv(reports_dir / "feature_importance_combined.csv", index=False)
    print(f"  wrote feature_importance_combined.csv  ({len(fi):,} rows)")

    # 7. Plots.
    print("rendering plots ...")
    plot_error_by_horizon(
        breakdowns["horizon"], metric="WAPE",
        savepath=figures_dir / "error_by_horizon_wape.png",
    )
    if "segment_at_origin" in breakdowns:
        plot_metric_by_segment(
            breakdowns["segment_at_origin"], metric="WAPE",
            savepath=figures_dir / "error_by_segment_wape.png",
        )
        plot_bias_by_segment(
            breakdowns["segment_at_origin"],
            savepath=figures_dir / "bias_by_segment.png",
        )
    # Pick the current best WAPE model for the actual-vs-pred panel.
    overall = (
        combined.groupby("model")
        .apply(lambda g: float(
            (g["actual"].sub(g["prediction"])).abs().sum()
            / g["actual"].abs().sum()
        ))
        .sort_values()
    )
    best_model = overall.index[0]
    plot_actual_vs_pred_for_top_errors(
        combined, model=best_model, n_items=6, direction="under",
        savepath=figures_dir / "actual_vs_pred_top_errors.png",
    )
    if not fi.empty:
        plot_feature_importance_top20(
            fi, savepath=figures_dir / "feature_importance_top20.png",
        )

    print()
    print(f"all diagnostic outputs written to:")
    print(f"  {reports_dir}")
    print(f"  {figures_dir}")
    print()
    print("Overall WAPE on matched grid (best at top):")
    print(overall.to_string())

    return {
        "combined": combined,
        "segments": segments,
        "lifecycle": lifecycle,
        "is_active": active,
        "breakdowns": breakdowns,
        "feature_importance": fi,
        "overall_wape": overall,
    }


def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 1 diagnostics report.")
    p.add_argument("--base", type=Path, default=ARTIFACTS.base_table_ca1)
    p.add_argument("--lgbm-point", type=Path,
                   default=ARTIFACTS.lightgbm_backtest_predictions_ca1)
    p.add_argument("--quantile-preds", type=Path,
                   default=ARTIFACTS.quantile_backtest_predictions_ca1)
    p.add_argument("--baseline-preds", type=Path,
                   default=ARTIFACTS.baseline_predictions_ca1)
    p.add_argument("--point-bundle", type=Path,
                   default=ARTIFACTS.lightgbm_point_model_ca1)
    p.add_argument("--quantile-bundle", type=Path,
                   default=ARTIFACTS.lightgbm_quantile_models_ca1)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    run(
        base_table_path=args.base,
        lightgbm_point_path=args.lgbm_point,
        quantile_predictions_path=args.quantile_preds,
        baseline_predictions_path=args.baseline_preds,
        point_bundle_path=args.point_bundle,
        quantile_bundle_path=args.quantile_bundle,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
