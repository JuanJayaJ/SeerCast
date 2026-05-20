"""SeerCast Phase 8 — Credibility Pass CLI.

Run a single end-to-end pass that answers the four questions raised in
the project review:

1. Is the model improvement over baseline statistically meaningful?
2. Can we reduce the underforecasting bias?
3. How does the model look under M5-style scoring?
4. Are the quantile intervals calibrated enough to trust p90?

Honest framing:
* Bias correction is post-hoc calibration, not a new model.
* Bootstrap CI is evidence, not a frequentist test.
* p90 is a planning quantile, not guaranteed demand.
* WRMSSE here is CA_1-scoped (1 store), not the full M5 score.

Usage::

    python -m seercast.training.run_credibility_pass
    python -m seercast.training.run_credibility_pass --n-boot 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from seercast.config import (
    ARTIFACTS,
    FIGURES_DIR,
    REPORTS_DIR,
    ensure_dirs,
)
from seercast.evaluation.bias_correction import (
    BiasCorrector,
    compare_corrections,
)
from seercast.evaluation.bootstrap import (
    clustered_bootstrap_metric_diffs,
    matched_grid,
)
from seercast.evaluation.calibration import (
    ConformalScaler,
    build_calibration_report,
    overall_coverage,
    per_tail_coverage,
)
from seercast.evaluation.metrics import all_point_metrics
from seercast.evaluation.wrmsse import hierarchical_rmsse


_LIFECYCLE_DIR = REPORTS_DIR / "experiments" / "lifecycle_features"
_CREDIBILITY_REPORTS_DIR = REPORTS_DIR / "credibility"
_CREDIBILITY_FIGURES_DIR = FIGURES_DIR / "credibility"

_DEFAULT_BASELINE_MODEL = "moving_average_28"


# ----- Path resolvers ------------------------------------------------------ #


def _resolve_quantile_path(explicit):
    if explicit is not None:
        explicit = Path(explicit)
        if not explicit.exists():
            raise FileNotFoundError(f"--quantile {explicit} not found")
        return explicit
    lifecycle = _LIFECYCLE_DIR / "quantile_backtest_predictions_ca1.parquet"
    phase6 = ARTIFACTS.quantile_backtest_predictions_ca1
    if lifecycle.exists():
        return lifecycle
    if phase6.exists():
        return phase6
    raise FileNotFoundError(
        "could not find quantile backtest predictions in lifecycle or phase-6 paths."
    )


def _resolve_lightgbm_point_path(explicit):
    """LightGBM point predictions are optional. A missing explicit path
    is treated as 'opt out' rather than an error."""
    if explicit is not None:
        explicit = Path(explicit)
        if not explicit.exists():
            print(f"  (lightgbm point preds {explicit} not found - skipped)")
            return None
        return explicit
    lifecycle = _LIFECYCLE_DIR / "lightgbm_backtest_predictions_ca1.parquet"
    phase5 = ARTIFACTS.lightgbm_backtest_predictions_ca1
    if lifecycle.exists():
        return lifecycle
    if phase5.exists():
        return phase5
    return None


def _resolve_baseline_path(explicit):
    if explicit is not None:
        explicit = Path(explicit)
        if not explicit.exists():
            raise FileNotFoundError(f"--baselines {explicit} not found")
        return explicit
    if ARTIFACTS.baseline_predictions_ca1.exists():
        return ARTIFACTS.baseline_predictions_ca1
    raise FileNotFoundError(
        f"baseline predictions not found at {ARTIFACTS.baseline_predictions_ca1}"
    )


def _resolve_base_table_path(explicit):
    """Base table optional - missing path opts out of WRMSSE."""
    if explicit is not None:
        explicit = Path(explicit)
        if not explicit.exists():
            print(f"  (base table {explicit} not found - WRMSSE skipped)")
            return None
        return explicit
    if ARTIFACTS.base_table_ca1.exists():
        return ARTIFACTS.base_table_ca1
    return None


# ----- Helpers ------------------------------------------------------------- #


def _attach_identity(predictions, base_table_path):
    """Join identity cols from the base table. Only reads columns that
    actually exist in the parquet schema (synthetic tests may have a subset).
    """
    candidates = ("cat_id", "dept_id", "store_id", "state_id", "item_id")
    missing_in_preds = [c for c in candidates if c not in predictions.columns]
    if not missing_in_preds:
        return predictions
    if base_table_path is None or not Path(base_table_path).exists():
        return predictions

    import pyarrow.parquet as pq
    try:
        schema_names = set(pq.read_schema(base_table_path).names)
    except Exception:
        schema_names = set()

    if schema_names:
        cols_to_read = [c for c in candidates if c in schema_names]
        if "id" not in schema_names:
            return predictions
        cols_to_read = ["id", *cols_to_read]
        base = pd.read_parquet(
            base_table_path, columns=cols_to_read,
        ).drop_duplicates("id")
    else:
        base = pd.read_parquet(base_table_path).drop_duplicates("id")
        keep = ["id"] + [c for c in candidates if c in base.columns]
        base = base[keep]

    return predictions.merge(base, on="id", how="left")


def _split_origins(df, n_calibration, *, origin_col="origin_date"):
    origins = sorted(df[origin_col].unique())
    if len(origins) < n_calibration + 1:
        raise ValueError(
            f"need at least {n_calibration + 1} distinct origins, got {len(origins)}"
        )
    cal_origins = origins[:n_calibration]
    eval_origins = origins[n_calibration:]
    cal = df.loc[df[origin_col].isin(cal_origins)].copy()
    ev = df.loc[df[origin_col].isin(eval_origins)].copy()
    return cal, ev, cal_origins, eval_origins


# ----- Figures (best-effort) ---------------------------------------------- #


def _save_bootstrap_diff_figure(boot, path):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = np.arange(len(boot))
    ax.errorbar(
        boot["diff_point"], y,
        xerr=[boot["diff_point"] - boot["ci_low"],
              boot["ci_high"] - boot["diff_point"]],
        fmt="o", color="#1f77b4", capsize=4,
    )
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(y, boot["metric"])
    ax.invert_yaxis()
    ax.set_xlabel("model - baseline (negative = model better for WAPE/MAE/RMSE)")
    ax.set_title("Clustered bootstrap 95% CI for metric difference")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_bias_before_after_figure(corr, path):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(corr))
    width = 0.35
    ax.bar(x - width / 2, corr["Bias_before"], width=width, label="before", color="#aab5be")
    ax.bar(x + width / 2, corr["Bias_after"], width=width, label="after", color="#1f77b4")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xticks(x, corr["method"], rotation=20, ha="right")
    ax.set_ylabel("Bias = sum(yhat-y)/sum(y)")
    ax.set_title("Bias before/after post-hoc correction (eval fold)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_coverage_by_horizon_figure(cal, path):
    horizon_rows = cal.loc[cal["level"] == "horizon"].copy()
    if horizon_rows.empty:
        return
    horizon_rows["horizon"] = horizon_rows["key"].astype(float)
    horizon_rows = horizon_rows.sort_values("horizon")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(horizon_rows["horizon"], horizon_rows["empirical_coverage"],
            marker="o", color="#1f77b4", label="empirical")
    target = float(horizon_rows["target_coverage"].iloc[0])
    ax.axhline(target, color="grey", linestyle="--", linewidth=1,
               label=f"target = {target:.2f}")
    ax.set_xlabel("forecast horizon (days)")
    ax.set_ylabel("[p10, p90] coverage")
    ax.set_ylim(0, 1)
    ax.set_title("Quantile interval coverage by horizon")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ----- Main orchestrator -------------------------------------------------- #


def run(
    *,
    quantile_path=None,
    lightgbm_point_path=None,
    baseline_path=None,
    base_table_path=None,
    reports_dir=_CREDIBILITY_REPORTS_DIR,
    figures_dir=_CREDIBILITY_FIGURES_DIR,
    baseline_model=_DEFAULT_BASELINE_MODEL,
    horizons=(1, 7, 14, 28),
    n_calibration_origins=1,
    n_boot=1000,
    seed=0,
    target_coverage=0.80,
):
    ensure_dirs()
    reports_dir = Path(reports_dir); reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(figures_dir); figures_dir.mkdir(parents=True, exist_ok=True)

    # If the test/caller wants to suppress auto-discovery of optional paths,
    # they should pass an explicit non-existent Path (we now treat that as
    # "opt out" rather than error).
    q_path = _resolve_quantile_path(quantile_path)
    p_path = _resolve_lightgbm_point_path(lightgbm_point_path)
    b_path = _resolve_baseline_path(baseline_path)
    bt_path = _resolve_base_table_path(base_table_path)

    print(f"quantile predictions : {q_path}")
    print(f"lightgbm point preds : {p_path if p_path else '(skipped)'}")
    print(f"baseline predictions : {b_path}")
    print(f"base table for WRMSSE: {bt_path if bt_path else '(skipped)'}")
    print()

    quantile = pd.read_parquet(q_path)
    baselines = pd.read_parquet(b_path)
    baseline_only = baselines.loc[
        (baselines["model"] == baseline_model)
        & (baselines["horizon"].isin(horizons))
    ].copy()
    if baseline_only.empty:
        raise ValueError(
            f"no rows for baseline model='{baseline_model}' in {b_path}"
        )
    quantile = _attach_identity(quantile, bt_path)
    baseline_only = _attach_identity(baseline_only, bt_path)
    if p_path is not None:
        lgb_point = pd.read_parquet(p_path)
        lgb_point = _attach_identity(lgb_point, bt_path)
    else:
        lgb_point = None

    q_cal, q_eval, cal_origins, eval_origins = _split_origins(
        quantile, n_calibration_origins,
    )
    print(f"calibration origins : {[str(o)[:10] for o in cal_origins]}")
    print(f"evaluation origins  : {[str(o)[:10] for o in eval_origins]}")
    print()

    # 1. Headline comparison
    print("[1/5] headline model comparison")
    headline_rows = []

    def _score(df, pred_col, label):
        clean = df.dropna(subset=["actual", pred_col])
        m = all_point_metrics(clean["actual"], clean[pred_col])
        m.update({"model": label, "n": int(len(clean))})
        return m

    headline_rows.append(_score(q_eval, "p50", "lightgbm_quantile_p50"))
    lgb_eval = None
    if lgb_point is not None:
        _, lgb_eval, _, _ = _split_origins(lgb_point, n_calibration_origins)
        headline_rows.append(_score(lgb_eval, "prediction", "lightgbm_point"))
    base_eval = baseline_only.loc[baseline_only["origin_date"].isin(eval_origins)]
    headline_rows.append(_score(base_eval, "prediction", baseline_model))
    headline = pd.DataFrame(headline_rows)[
        ["model", "n", "MAE", "RMSE", "WAPE", "Bias"]
    ]
    headline.to_csv(reports_dir / "headline_model_comparison_ca1.csv", index=False)
    print(headline.to_string(index=False))
    print()

    # 2. Bootstrap
    print(f"[2/5] clustered bootstrap CI (n_boot={n_boot})")
    matched = matched_grid(
        q_eval.assign(prediction=q_eval["p50"]),
        base_eval,
        keys=("id", "origin_date", "horizon"),
        actual_col="actual", pred_col_a="prediction", pred_col_b="prediction",
        label_a="model", label_b="baseline",
    )
    boot = clustered_bootstrap_metric_diffs(
        matched, label_a="model", label_b="baseline",
        cluster_col="id", n_boot=n_boot, seed=seed,
        metrics=("WAPE", "MAE", "RMSE", "Bias"),
    )
    boot.to_csv(reports_dir / "bootstrap_wape_ci_ca1.csv", index=False)
    print(boot.to_string(index=False))
    try:
        _save_bootstrap_diff_figure(boot, figures_dir / "bootstrap_wape_diff.png")
    except Exception as exc:
        print(f"  (figure skipped: {exc})")
    print()

    # 3. Bias correction
    print("[3/5] bias correction methods (fit cal -> eval)")
    corr = compare_corrections(
        calibration_df=q_cal,
        evaluation_df=q_eval,
        pred_col="p50",
        actual_col="actual",
        horizon_col="horizon",
        quantile_cols=("p10", "p90"),
    )
    corr.to_csv(reports_dir / "bias_correction_comparison_ca1.csv", index=False)
    print(corr.to_string(index=False))
    try:
        _save_bias_before_after_figure(corr, figures_dir / "bias_before_after.png")
    except Exception as exc:
        print(f"  (figure skipped: {exc})")
    print()

    # 4. WRMSSE
    wrmsse_summary = {}
    wrmsse_df = pd.DataFrame(
        columns=["model", "level", "n_series", "weighted_rmsse", "unweighted_rmsse"]
    )
    if bt_path is None:
        print("[4/5] WRMSSE skipped (no base table)")
    else:
        print("[4/5] WRMSSE (CA_1 scope: store + cat + dept + id)")
        base_table = pd.read_parquet(bt_path)
        # Keep only the columns we need for memory.
        keep_cols = [c for c in ("id", "date", "sales", "sell_price",
                                 "dept_id", "cat_id", "store_id")
                     if c in base_table.columns]
        base_table = base_table[keep_cols]
        training_end_date = pd.Timestamp(max(cal_origins))

        q_eval_for_w = q_eval.rename(columns={"p50": "prediction"})
        q_per_level, q_wrmsse = hierarchical_rmsse(
            q_eval_for_w, base_table,
            training_end_date=training_end_date,
            model_name="lightgbm_quantile_p50",
        )
        b_per_level, b_wrmsse = hierarchical_rmsse(
            base_eval, base_table,
            training_end_date=training_end_date,
            model_name=baseline_model,
        )
        frames = [q_per_level, b_per_level]
        wrmsse_summary = {
            "lightgbm_quantile_p50": q_wrmsse,
            baseline_model: b_wrmsse,
        }
        if lgb_eval is not None:
            lgb_per_level, lgb_w = hierarchical_rmsse(
                lgb_eval, base_table,
                training_end_date=training_end_date,
                model_name="lightgbm_point",
            )
            frames.append(lgb_per_level)
            wrmsse_summary["lightgbm_point"] = lgb_w
        wrmsse_df = pd.concat(frames, ignore_index=True)
        summary_rows = [
            {"model": m, "level": "WRMSSE_equal_weight",
             "n_series": 0,
             "weighted_rmsse": w, "unweighted_rmsse": float("nan")}
            for m, w in wrmsse_summary.items()
        ]
        wrmsse_df = pd.concat([wrmsse_df, pd.DataFrame(summary_rows)], ignore_index=True)
        wrmsse_df.to_csv(reports_dir / "wrmsse_or_rmsse_ca1.csv", index=False)
        print(wrmsse_df.to_string(index=False))
    print()

    # 5. Calibration
    print("[5/5] calibration report (split-conformal scaler fit on cal)")
    raw_overall = overall_coverage(q_eval, target=target_coverage)
    raw_tails = per_tail_coverage(q_eval)
    print(f"  raw [p10,p90] coverage on eval: "
          f"{raw_overall['empirical_coverage']:.3f} (target {target_coverage:.2f})")

    by_groups = [("horizon",)]
    if "cat_id" in q_eval.columns:
        by_groups.append(("cat_id",))
    cal_table = build_calibration_report(
        q_eval, by_groups=by_groups, target=target_coverage,
    )

    conformal = ConformalScaler(target=target_coverage).fit(q_cal)
    q_eval_conformal = conformal.transform(
        q_eval, out_lo_col="p10_conformal", out_hi_col="p90_conformal",
    )
    conformal_overall = overall_coverage(
        q_eval_conformal,
        lo_col="p10_conformal", hi_col="p90_conformal",
        target=target_coverage,
    )

    cal_table["intervals"] = "raw"
    conformal_table = build_calibration_report(
        q_eval_conformal,
        lo_col="p10_conformal", hi_col="p90_conformal",
        by_groups=by_groups, target=target_coverage,
    )
    conformal_table["intervals"] = "conformal"
    cal_table = pd.concat([cal_table, conformal_table], ignore_index=True)
    cal_table["conformal_width_factor"] = float(conformal.factor_)
    cal_table.to_csv(reports_dir / "calibration_report_ca1.csv", index=False)

    print(f"  conformal width factor: {conformal.factor_:.3f} (n_cal={conformal.fitted_n_})")
    print(f"  conformal [p10,p90] coverage on eval: "
          f"{conformal_overall['empirical_coverage']:.3f}")
    try:
        _save_coverage_by_horizon_figure(
            cal_table.loc[cal_table["intervals"] == "raw"],
            figures_dir / "coverage_by_horizon.png",
        )
    except Exception as exc:
        print(f"  (figure skipped: {exc})")
    print()

    print("=" * 60)
    print("credibility pass complete")
    print("=" * 60)
    for p in (
        reports_dir / "headline_model_comparison_ca1.csv",
        reports_dir / "bootstrap_wape_ci_ca1.csv",
        reports_dir / "bias_correction_comparison_ca1.csv",
        reports_dir / "calibration_report_ca1.csv",
        reports_dir / "wrmsse_or_rmsse_ca1.csv",
    ):
        if p.exists():
            print(f"  csv:    {p}")
    for f in (
        figures_dir / "bootstrap_wape_diff.png",
        figures_dir / "bias_before_after.png",
        figures_dir / "coverage_by_horizon.png",
    ):
        if f.exists():
            print(f"  figure: {f}")
    print()
    print("Reminder: bias correction is post-hoc calibration, not a new model. "
          "Bootstrap CI is evidence, not proof. p90 is a planning quantile, "
          "not guaranteed demand. WRMSSE is CA_1-scoped here.")

    return {
        "headline": headline,
        "bootstrap": boot,
        "bias_correction": corr,
        "wrmsse_per_level": wrmsse_df,
        "wrmsse_summary": wrmsse_summary,
        "calibration_table": cal_table,
        "conformal_factor": float(conformal.factor_),
        "raw_overall_coverage": raw_overall,
        "conformal_overall_coverage": conformal_overall,
        "calibration_origins": cal_origins,
        "evaluation_origins": eval_origins,
    }


# ----- Argparse / main ---------------------------------------------------- #


def _parse_args(argv):
    p = argparse.ArgumentParser(description="SeerCast Phase 8 - Credibility Pass.")
    p.add_argument("--quantile", type=Path, default=None)
    p.add_argument("--lightgbm", type=Path, default=None)
    p.add_argument("--baselines", type=Path, default=None)
    p.add_argument("--base", type=Path, default=None)
    p.add_argument("--baseline-model", type=str, default=_DEFAULT_BASELINE_MODEL)
    p.add_argument("--n-calibration-origins", type=int, default=1)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-coverage", type=float, default=0.80)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    run(
        quantile_path=args.quantile,
        lightgbm_point_path=args.lightgbm,
        baseline_path=args.baselines,
        base_table_path=args.base,
        baseline_model=args.baseline_model,
        n_calibration_origins=args.n_calibration_origins,
        n_boot=args.n_boot,
        seed=args.seed,
        target_coverage=args.target_coverage,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
