"""SeerCast Phase 9 — Objective Sweep CLI.

Compare LightGBM training objectives (l2, l1, poisson, tweedie, log1p_l2,
log1p_l1) on the same backtest origins as the existing lightgbm_point /
lifecycle-quantile pipelines. Each candidate is scored on the matched
grid against:

* the current lifecycle quantile p50 (best-by-WAPE incumbent)
* the moving_average_28 baseline

The CLI does NOT touch any planner artifacts and writes everything under
``outputs/reports/experiments/objective_sweep/`` so the headline model
reference remains the lifecycle quantile p50 until this sweep proves
otherwise.

Outputs
-------
CSVs (under ``outputs/reports/experiments/objective_sweep/``):

* ``objective_sweep_summary_ca1.csv``       overall MAE/RMSE/WAPE/Bias/n per model
* ``objective_sweep_by_horizon_ca1.csv``    same, grouped by horizon
* ``objective_sweep_by_cat_id_ca1.csv``     same, grouped by cat_id (if present)
* ``objective_sweep_predictions_ca1.parquet`` long preds for all candidates
* ``objective_sweep_bootstrap_ci_ca1.csv``  optional: clustered bootstrap CI
                                            of each candidate's WAPE/Bias diff
                                            vs MA-28 and vs quantile p50
* ``objective_sweep_wrmsse_ca1.csv``        optional: CA_1-scoped RMSSE/WRMSSE

Figures (best effort) under ``outputs/figures/experiments/objective_sweep/``.

Usage
-----
::

    # Default: all 8 candidates on all 3 origins, ~10-20 min locally
    python -m seercast.training.run_objective_sweep

    # Fast: 4 candidates on 1 origin, ~2-5 min, for smoke testing
    python -m seercast.training.run_objective_sweep --fast

    # Skip slow extras
    python -m seercast.training.run_objective_sweep --skip-bootstrap --skip-wrmsse

Honest framing
--------------
* The current ``lightgbm_point`` is Poisson, not L2. So "L2" in this sweep
  is a NEW candidate, not a rebrand of the baseline.
* log1p has an inverse-transform bias (expm1(mean) != mean(expm1)). We
  document it and let the metrics speak.
* This sweep is point-style; quantile remains the planner's primary
  uncertainty model.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from seercast.config import (
    ARTIFACTS,
    BACKTEST,
    FIGURES_DIR,
    REPORTS_DIR,
    ensure_dirs,
)
from seercast.evaluation.backtesting import origin_to_date
from seercast.evaluation.metrics import all_point_metrics, score_by_group
from seercast.features.supervised import validate_supervised_table
from seercast.models.objective_lightgbm import (
    CandidateSpec,
    ObjectiveCandidateModel,
    default_catalog,
    fast_catalog,
    fit_candidate,
)
from seercast.training.train_lightgbm import split_for_backtest_origin


_LIFECYCLE_DIR = REPORTS_DIR / "experiments" / "lifecycle_features"
_SWEEP_REPORTS_DIR = REPORTS_DIR / "experiments" / "objective_sweep"
_SWEEP_FIGURES_DIR = FIGURES_DIR / "experiments" / "objective_sweep"

_DEFAULT_BASELINE_MODEL = "moving_average_28"
_DEFAULT_HORIZONS = (1, 7, 14, 28)


# --------------------------------------------------------------------------- #
# Predictions frame builder
# --------------------------------------------------------------------------- #


def _predictions_frame(test_df, preds, model_name):
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
# Matched-grid scoring
# --------------------------------------------------------------------------- #


_GRID_COLS = ("origin_date", "id", "horizon", "target_date")


def _build_matched_predictions(
    sweep_preds: pd.DataFrame,
    incumbent_preds: pd.DataFrame | None,
    baseline_preds: pd.DataFrame | None,
    *,
    incumbent_label: str = "lifecycle_quantile_p50",
    baseline_model: str = _DEFAULT_BASELINE_MODEL,
    horizons: Sequence[int] = _DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Stack all candidate predictions plus the incumbent (quantile p50 ->
    'prediction' = p50) and the moving-average baseline, inner-joined onto
    the sweep's evaluation grid so every row is scored on identical slots.
    """
    grid = sweep_preds[list(_GRID_COLS)].drop_duplicates()
    frames = [sweep_preds]

    if incumbent_preds is not None and len(incumbent_preds):
        incumbent = incumbent_preds[
            list(_GRID_COLS) + ["actual", "p50"]
        ].rename(columns={"p50": "prediction"}).copy()
        incumbent["model"] = incumbent_label
        incumbent_matched = incumbent.merge(grid, on=list(_GRID_COLS), how="inner")
        if len(incumbent_matched) != len(grid):
            print(
                f"  WARN: incumbent coverage {len(incumbent_matched):,} != "
                f"grid size {len(grid):,} -- some sweep rows have no incumbent prediction."
            )
        frames.append(incumbent_matched)

    if baseline_preds is not None and len(baseline_preds):
        b = baseline_preds.loc[
            (baseline_preds["model"] == baseline_model)
            & (baseline_preds["horizon"].isin(horizons))
        ].copy()
        b_matched = b.merge(grid, on=list(_GRID_COLS), how="inner")
        if len(b_matched) != len(grid):
            print(
                f"  WARN: {baseline_model} matched {len(b_matched):,} of "
                f"{len(grid):,} sweep rows."
            )
        frames.append(b_matched)

    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Identity join (cat_id from base table)
# --------------------------------------------------------------------------- #


def _attach_cat_id(preds: pd.DataFrame, base_table_path: Path | None) -> pd.DataFrame:
    if "cat_id" in preds.columns:
        return preds
    if base_table_path is None or not Path(base_table_path).exists():
        return preds
    import pyarrow.parquet as pq
    try:
        schema = set(pq.read_schema(base_table_path).names)
    except Exception:
        return preds
    if "cat_id" not in schema or "id" not in schema:
        return preds
    base = pd.read_parquet(base_table_path, columns=["id", "cat_id"]).drop_duplicates("id")
    return preds.merge(base, on="id", how="left")


# --------------------------------------------------------------------------- #
# Bootstrap helper (optional)
# --------------------------------------------------------------------------- #


def _candidate_vs_reference_bootstrap(
    matched: pd.DataFrame,
    *,
    candidates: Sequence[str],
    reference_label: str,
    n_boot: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """For each candidate, return clustered (by id) bootstrap 95% CI on
    the WAPE difference (candidate - reference) and Bias difference.

    Uses :mod:`seercast.evaluation.bootstrap`. If the reference is not
    present in ``matched``, returns an empty frame with a warning.
    """
    from seercast.evaluation.bootstrap import (
        clustered_bootstrap_metric_diffs,
        matched_grid,
    )
    ref = matched.loc[matched["model"] == reference_label]
    if ref.empty:
        print(f"  WARN: reference '{reference_label}' not in matched frame; skipping.")
        return pd.DataFrame()

    rows = []
    for cand in candidates:
        cand_df = matched.loc[matched["model"] == cand]
        if cand_df.empty:
            continue
        m = matched_grid(
            cand_df, ref,
            keys=("id", "origin_date", "horizon"),
            actual_col="actual",
            pred_col_a="prediction", pred_col_b="prediction",
            label_a="candidate", label_b="reference",
        )
        if m.empty:
            continue
        out = clustered_bootstrap_metric_diffs(
            m, label_a="candidate", label_b="reference",
            cluster_col="id", n_boot=n_boot, seed=seed,
            metrics=("WAPE", "Bias"),
        )
        for _, r in out.iterrows():
            rows.append({
                "candidate": cand,
                "reference": reference_label,
                "metric": r["metric"],
                "diff_point": r["diff_point"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "n_clusters": r["n_clusters"],
                "n_rows": r["n_rows"],
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# WRMSSE helper (optional, CA_1 scope)
# --------------------------------------------------------------------------- #


def _wrmsse_table(
    matched: pd.DataFrame, base_table_path: Path, training_end_date,
) -> pd.DataFrame:
    """Run hierarchical_rmsse for every model in ``matched``."""
    from seercast.evaluation.wrmsse import hierarchical_rmsse
    base_cols = ["id", "date", "sales", "sell_price", "dept_id", "cat_id", "store_id"]
    import pyarrow.parquet as pq
    try:
        schema = set(pq.read_schema(base_table_path).names)
        keep = [c for c in base_cols if c in schema]
    except Exception:
        keep = base_cols
    base = pd.read_parquet(base_table_path, columns=keep)
    frames = []
    summary_rows = []
    for model_name in matched["model"].unique():
        sub = matched.loc[matched["model"] == model_name]
        per_level, w = hierarchical_rmsse(
            sub, base, training_end_date=training_end_date,
            model_name=str(model_name),
        )
        frames.append(per_level)
        summary_rows.append({
            "model": model_name, "level": "WRMSSE_equal_weight",
            "n_series": 0, "weighted_rmsse": w, "unweighted_rmsse": float("nan"),
        })
    out = pd.concat(frames + [pd.DataFrame(summary_rows)], ignore_index=True)
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _save_wape_bias_figure(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in summary.iterrows():
        ax.scatter(r["WAPE"], r["Bias"], s=60)
        ax.annotate(r["model"], (r["WAPE"], r["Bias"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("WAPE (lower is better)")
    ax.set_ylabel("Bias (closer to 0 is better)")
    ax.set_title("Objective sweep: WAPE vs Bias")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_rmse_wape_figure(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in summary.iterrows():
        ax.scatter(r["WAPE"], r["RMSE"], s=60)
        ax.annotate(r["model"], (r["WAPE"], r["RMSE"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("WAPE (lower is better)")
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title("Objective sweep: RMSE vs WAPE")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_by_horizon_figure(by_h: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, sub in by_h.groupby("model"):
        sub = sub.sort_values("horizon")
        ax.plot(sub["horizon"], sub["WAPE"], marker="o", label=str(model_name))
    ax.set_xlabel("forecast horizon (days)")
    ax.set_ylabel("WAPE")
    ax.set_title("WAPE by horizon, per objective")
    ax.legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main run()
# --------------------------------------------------------------------------- #


def run(
    *,
    supervised_path: Path | str = ARTIFACTS.train_features_ca1,
    base_table_path: Path | str = ARTIFACTS.base_table_ca1,
    baseline_predictions_path: Path | str = ARTIFACTS.baseline_predictions_ca1,
    incumbent_quantile_path: Path | str | None = None,
    reports_dir: Path | str = _SWEEP_REPORTS_DIR,
    figures_dir: Path | str = _SWEEP_FIGURES_DIR,
    backtest_origins: Sequence = BACKTEST.origins,
    valid_window_days: int = 56,
    catalog: Sequence[CandidateSpec] | None = None,
    n_estimators: int = 500,
    early_stopping_rounds: int = 30,
    skip_bootstrap: bool = False,
    skip_wrmsse: bool = False,
    n_boot: int = 500,
    fast: bool = False,
    horizons: Sequence[int] = _DEFAULT_HORIZONS,
) -> dict:
    """Run the full sweep. Returns a dict of in-memory frames + paths."""
    ensure_dirs()
    reports_dir = Path(reports_dir); reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(figures_dir); figures_dir.mkdir(parents=True, exist_ok=True)

    if fast:
        if catalog is None:
            catalog = fast_catalog()
        backtest_origins = (backtest_origins[1],) if len(backtest_origins) >= 2 \
            else backtest_origins[:1]
        n_estimators = min(n_estimators, 300)
        early_stopping_rounds = min(early_stopping_rounds, 20)

    if catalog is None:
        catalog = default_catalog(include_tweedie=True)

    print(f"objective sweep: {len(catalog)} candidates")
    for c in catalog:
        print(f"  - {c.name:30s} objective={c.objective:14s} transform={c.target_transform}")
    print(f"backtest origins: {list(backtest_origins)}")
    print(f"n_estimators={n_estimators}, early_stop={early_stopping_rounds}")
    print()

    sup = pd.read_parquet(supervised_path)
    print(f"loaded supervised: {sup.shape[0]:,} rows x {sup.shape[1]} cols")
    sup_report = validate_supervised_table(sup, strict=True)
    print(sup_report.summary())
    print()

    base = pd.read_parquet(base_table_path, columns=["id", "d", "date"])
    backtest_dates = [pd.Timestamp(origin_to_date(base, o)) for o in backtest_origins]

    # ---- Fit + predict each candidate at each origin -------------------
    all_preds: list[pd.DataFrame] = []
    fit_timings: list[dict] = []
    for origin_date in backtest_dates:
        split = split_for_backtest_origin(sup, origin_date, valid_window_days)
        print(f"origin {origin_date.date()}: "
              f"train={len(split.train):,}, valid={len(split.valid):,}, "
              f"test={len(split.test):,}")
        if len(split.test) == 0 or len(split.train) == 0:
            print("  skipped (empty slice)")
            continue
        for spec in catalog:
            t0 = time.perf_counter()
            try:
                model = fit_candidate(
                    spec, split.train, split.valid,
                    n_estimators=n_estimators,
                    early_stopping_rounds=early_stopping_rounds,
                )
                preds = model.predict(split.test)
            except Exception as exc:
                print(f"  {spec.name}: FAILED ({type(exc).__name__}: {exc})")
                continue
            dt = time.perf_counter() - t0
            print(f"  {spec.name:30s}  fit+predict {dt:6.1f}s  "
                  f"best_iter={model.inner._best_iteration}")
            fit_timings.append({
                "origin": str(origin_date.date()),
                "candidate": spec.name,
                "seconds": dt,
                "best_iteration": model.inner._best_iteration,
            })
            all_preds.append(_predictions_frame(split.test, preds, spec.name))

    if not all_preds:
        raise RuntimeError("no predictions produced from any candidate.")

    sweep_preds = pd.concat(all_preds, ignore_index=True)
    print(f"\nsweep predictions: {len(sweep_preds):,} rows across "
          f"{sweep_preds['model'].nunique()} candidates")

    # ---- Pull incumbent + baseline preds -------------------------------
    if incumbent_quantile_path is None:
        incumbent_quantile_path = (
            _LIFECYCLE_DIR / "quantile_backtest_predictions_ca1.parquet"
        )
    incumbent_quantile_path = Path(incumbent_quantile_path)
    incumbent_preds = None
    if incumbent_quantile_path.exists():
        incumbent_preds = pd.read_parquet(incumbent_quantile_path)
        print(f"loaded incumbent quantile preds: {incumbent_preds.shape[0]:,} rows "
              f"from {incumbent_quantile_path}")
    else:
        print(f"NOTE: incumbent quantile preds not found at {incumbent_quantile_path}; "
              "skipping incumbent comparison.")

    baseline_predictions_path = Path(baseline_predictions_path)
    baseline_preds = None
    if baseline_predictions_path.exists():
        baseline_preds = pd.read_parquet(baseline_predictions_path)
    else:
        print(f"NOTE: baseline preds not found at {baseline_predictions_path}; "
              "skipping baseline comparison.")

    # ---- Matched-grid scoring -----------------------------------------
    matched = _build_matched_predictions(
        sweep_preds, incumbent_preds, baseline_preds,
        horizons=horizons,
    )
    matched = _attach_cat_id(matched, Path(base_table_path))

    # Overall, by-horizon, by-cat_id summaries
    summary = score_by_group(matched, by=("model",)).sort_values("WAPE").reset_index(drop=True)
    by_horizon = score_by_group(matched, by=("model", "horizon")).sort_values(
        ["model", "horizon"]
    ).reset_index(drop=True)
    if "cat_id" in matched.columns:
        by_cat = score_by_group(matched, by=("model", "cat_id")).sort_values(
            ["model", "cat_id"]
        ).reset_index(drop=True)
    else:
        by_cat = pd.DataFrame()

    summary.to_csv(reports_dir / "objective_sweep_summary_ca1.csv", index=False)
    by_horizon.to_csv(reports_dir / "objective_sweep_by_horizon_ca1.csv", index=False)
    if not by_cat.empty:
        by_cat.to_csv(reports_dir / "objective_sweep_by_cat_id_ca1.csv", index=False)
    sweep_preds.to_parquet(
        reports_dir / "objective_sweep_predictions_ca1.parquet", index=False,
    )
    if fit_timings:
        pd.DataFrame(fit_timings).to_csv(
            reports_dir / "objective_sweep_fit_timings.csv", index=False,
        )

    print()
    print("=" * 60)
    print("summary (sorted by WAPE):")
    print("=" * 60)
    print(summary.to_string(index=False))

    # ---- Optional: bootstrap CI ---------------------------------------
    boot_df = pd.DataFrame()
    if not skip_bootstrap:
        print()
        print(f"bootstrap CI (n_boot={n_boot}) -- candidates vs MA-28 and vs incumbent")
        candidate_names = list(sweep_preds["model"].unique())
        parts = []
        if baseline_preds is not None:
            parts.append(_candidate_vs_reference_bootstrap(
                matched, candidates=candidate_names,
                reference_label=_DEFAULT_BASELINE_MODEL,
                n_boot=n_boot,
            ))
        if incumbent_preds is not None:
            parts.append(_candidate_vs_reference_bootstrap(
                matched, candidates=candidate_names,
                reference_label="lifecycle_quantile_p50",
                n_boot=n_boot,
            ))
        boot_df = pd.concat([p for p in parts if not p.empty], ignore_index=True) \
            if parts else pd.DataFrame()
        if not boot_df.empty:
            boot_df.to_csv(
                reports_dir / "objective_sweep_bootstrap_ci_ca1.csv", index=False,
            )
            print(boot_df.to_string(index=False))

    # ---- Optional: WRMSSE ---------------------------------------------
    wrmsse_df = pd.DataFrame()
    if not skip_wrmsse and Path(base_table_path).exists():
        print()
        print("WRMSSE (CA_1 scope: store + cat + dept + id)")
        training_end_date = pd.Timestamp(min(backtest_dates))
        try:
            wrmsse_df = _wrmsse_table(matched, Path(base_table_path), training_end_date)
            wrmsse_df.to_csv(
                reports_dir / "objective_sweep_wrmsse_ca1.csv", index=False,
            )
            print(wrmsse_df.loc[
                wrmsse_df["level"] == "WRMSSE_equal_weight"
            ].to_string(index=False))
        except Exception as exc:
            print(f"  WRMSSE failed: {type(exc).__name__}: {exc}")
            wrmsse_df = pd.DataFrame()

    # ---- Figures ------------------------------------------------------
    try:
        _save_wape_bias_figure(
            summary, figures_dir / "objective_sweep_wape_bias.png",
        )
        _save_rmse_wape_figure(
            summary, figures_dir / "objective_sweep_rmse_wape.png",
        )
        _save_by_horizon_figure(
            by_horizon, figures_dir / "objective_sweep_by_horizon.png",
        )
    except Exception as exc:
        print(f"  (figure block failed: {type(exc).__name__}: {exc})")

    print()
    print("=" * 60)
    print("objective sweep complete")
    print("=" * 60)
    print("CSVs in", reports_dir)
    print("Figures in", figures_dir)
    print()
    print("Reminder: this sweep does NOT change the planner's incumbent. "
          "Update the headline model reference only after a candidate has "
          "been independently confirmed on a held-out origin.")

    return {
        "summary": summary,
        "by_horizon": by_horizon,
        "by_cat_id": by_cat,
        "predictions": sweep_preds,
        "matched": matched,
        "bootstrap": boot_df,
        "wrmsse": wrmsse_df,
        "fit_timings": pd.DataFrame(fit_timings),
        "catalog": list(catalog),
    }


# --------------------------------------------------------------------------- #
# Argparse / main
# --------------------------------------------------------------------------- #


def _parse_args(argv):
    p = argparse.ArgumentParser(description="SeerCast Phase 9 - Objective Sweep.")
    p.add_argument("--supervised", type=Path, default=ARTIFACTS.train_features_ca1)
    p.add_argument("--base", type=Path, default=ARTIFACTS.base_table_ca1)
    p.add_argument("--baselines", type=Path, default=ARTIFACTS.baseline_predictions_ca1)
    p.add_argument("--incumbent-quantile", type=Path, default=None)
    p.add_argument("--reports-dir", type=Path, default=_SWEEP_REPORTS_DIR)
    p.add_argument("--figures-dir", type=Path, default=_SWEEP_FIGURES_DIR)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--early-stop", type=int, default=30)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--fast", action="store_true",
                   help="Smaller catalog, single origin, fewer rounds.")
    p.add_argument("--include-tweedie", action="store_true", default=True)
    p.add_argument("--no-tweedie", dest="include_tweedie", action="store_false")
    p.add_argument("--skip-bootstrap", action="store_true")
    p.add_argument("--skip-wrmsse", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    catalog = None
    if args.fast:
        catalog = fast_catalog()
    elif not args.include_tweedie:
        catalog = default_catalog(include_tweedie=False)
    run(
        supervised_path=args.supervised,
        base_table_path=args.base,
        baseline_predictions_path=args.baselines,
        incumbent_quantile_path=args.incumbent_quantile,
        reports_dir=args.reports_dir,
        figures_dir=args.figures_dir,
        catalog=catalog,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stop,
        skip_bootstrap=args.skip_bootstrap,
        skip_wrmsse=args.skip_wrmsse,
        n_boot=args.n_boot,
        fast=args.fast,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
