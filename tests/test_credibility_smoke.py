"""Smoke tests for the Phase 8 credibility pass.

Covers:

* bias correction maths (multiplicative, additive, per-horizon)
* corrected predictions are non-negative
* clustered bootstrap returns finite, well-shaped CIs
* matched-grid preserves equal n on both prediction sides
* RMSSE handles a known synthetic case
* calibration coverage maths agrees with hand-computed values
* split-conformal scaler narrows or widens the right way
* end-to-end CLI runs on synthetic inputs without M5 files

Run::

    python tests/test_credibility_smoke.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from seercast.evaluation.bias_correction import (
    SUPPORTED_METHODS,
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
    coverage_by_group,
    overall_coverage,
    per_tail_coverage,
)
from seercast.evaluation.wrmsse import (
    compute_scale_per_id,
    hierarchical_rmsse,
    rmsse_for_series,
    score_rmsse_at_level,
    weighted_rmsse,
)
from seercast.training import run_credibility_pass as rcp


# --------------------------------------------------------------------------- #
# Synthetic builders
# --------------------------------------------------------------------------- #


def _make_quantile_preds(
    n_ids: int = 10,
    origins=("2015-03-08", "2015-05-03", "2015-06-28"),
    horizons=(1, 7, 14, 28),
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Build a small synthetic quantile predictions frame mimicking the
    lifecycle parquet shape. Predictions UNDER-forecast by a known scale
    so bias correction has something to do."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    underforecast = 0.8  # p50 is 0.8x the true mean -> bias correction factor ~1.25
    for k in range(n_ids):
        id_ = f"FOODS_1_{k:03d}_CA_1_validation"
        true_mean = 5.0 + 2.0 * k
        for origin in origins:
            for h in horizons:
                actual = max(0.0, rng.normal(true_mean, 1.5))
                p50 = max(0.0, true_mean * underforecast)
                rows.append({
                    "model": "lightgbm_quantile",
                    "origin_date": pd.Timestamp(origin),
                    "id": id_,
                    "horizon": h,
                    "target_date": pd.Timestamp(origin) + pd.Timedelta(days=int(h)),
                    "actual": actual,
                    "p10": max(0.0, p50 - 1.2),
                    "p50": p50,
                    "p90": p50 + 1.2,
                    "cat_id": "FOODS",
                    "dept_id": "FOODS_1",
                    "store_id": "CA_1",
                    "state_id": "CA",
                    "item_id": f"FOODS_1_{k:03d}",
                })
    return pd.DataFrame(rows)


def _make_baseline_preds(
    quantile_preds: pd.DataFrame,
    baseline_model: str = "moving_average_28",
    noise: float = 1.8,
    rng_seed: int = 1,
) -> pd.DataFrame:
    """A worse-than-quantile baseline so the comparison goes the
    'expected' direction (model better than baseline)."""
    rng = np.random.default_rng(rng_seed)
    out = quantile_preds[[
        "origin_date", "id", "horizon", "target_date", "actual",
    ]].copy()
    # baseline = actual + bigger noise than quantile p50 - actual.
    out["prediction"] = np.clip(
        out["actual"] + rng.normal(0.0, noise, size=len(out)),
        a_min=0.0, a_max=None,
    )
    out["model"] = baseline_model
    return out


# --------------------------------------------------------------------------- #
# 1. Bias correction maths
# --------------------------------------------------------------------------- #


def test_multiplicative_mean_ratio_recovers_true_scale():
    """If actual = pred * k everywhere, fitted factor must equal k."""
    df = pd.DataFrame({
        "id": list("aaaabbbb"),
        "horizon": [1, 7, 14, 28] * 2,
        "p50": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "actual": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
    })
    corr = BiasCorrector("multiplicative_mean_ratio").fit(df)
    assert abs(corr.scalar_factor_ - 2.0) < 1e-9


def test_additive_mean_error_recovers_mean_shift():
    df = pd.DataFrame({
        "id": ["a"] * 4,
        "horizon": [1, 7, 14, 28],
        "p50": [1.0, 1.0, 1.0, 1.0],
        "actual": [3.0, 3.0, 3.0, 3.0],
    })
    corr = BiasCorrector("additive_mean_error").fit(df)
    assert abs(corr.scalar_shift_ - 2.0) < 1e-9


def test_per_horizon_multiplicative_is_per_horizon():
    df = pd.DataFrame({
        "id": ["a"] * 4,
        "horizon": [1, 7, 14, 28],
        "p50":   [1.0, 1.0, 1.0, 1.0],
        "actual":[2.0, 3.0, 4.0, 5.0],
    })
    corr = BiasCorrector("per_horizon_multiplicative").fit(df)
    assert corr.per_horizon_factor_ == {1: 2.0, 7: 3.0, 14: 4.0, 28: 5.0}


def test_corrected_predictions_are_non_negative():
    """Even if additive shift would push some rows below zero, output
    must clip at 0."""
    df = pd.DataFrame({
        "id": ["a"] * 4,
        "horizon": [1, 7, 14, 28],
        "p50": [10.0, 10.0, 10.0, 10.0],
        "actual": [1.0, 1.0, 1.0, 1.0],
    })
    corr = BiasCorrector("additive_mean_error").fit(df)
    # shift is -9; raw -> 10 + -9 = 1; but if we pass smaller preds it could go negative.
    test = pd.DataFrame({
        "horizon": [1, 7, 14, 28],
        "p50": [5.0, 1.0, 0.5, 0.1],
    })
    out = corr.transform(test, cols=["p50"], keep_uncorrected=False)
    assert (out["p50_corrected"] >= 0.0).all()


def test_compare_corrections_returns_one_row_per_method():
    q = _make_quantile_preds(n_ids=4)
    cal = q.loc[q["origin_date"] == q["origin_date"].min()]
    ev = q.loc[q["origin_date"] != q["origin_date"].min()]
    out = compare_corrections(
        cal, ev, pred_col="p50",
        quantile_cols=("p10", "p90"),
    )
    assert set(out["method"]) == set(SUPPORTED_METHODS)
    # WAPE_before is identical across rows (it's the uncorrected eval WAPE).
    assert out["WAPE_before"].nunique() == 1


# --------------------------------------------------------------------------- #
# 2. Bootstrap
# --------------------------------------------------------------------------- #


def test_matched_grid_inner_joins_and_preserves_actual():
    q = _make_quantile_preds(n_ids=5)
    b = _make_baseline_preds(q)
    m = matched_grid(
        q.assign(prediction=q["p50"]), b,
        keys=("id", "origin_date", "horizon"),
        label_a="model", label_b="baseline",
    )
    # n_a = n_b = n_matched, since we built b from q.
    assert len(m) == len(q)
    # Both predictions present and finite.
    assert m["pred_model"].notna().all()
    assert m["pred_baseline"].notna().all()
    assert m["actual"].notna().all()


def test_bootstrap_returns_finite_ci():
    q = _make_quantile_preds(n_ids=8, rng_seed=2)
    b = _make_baseline_preds(q, rng_seed=3)
    m = matched_grid(
        q.assign(prediction=q["p50"]), b,
        keys=("id", "origin_date", "horizon"),
        label_a="model", label_b="baseline",
    )
    out = clustered_bootstrap_metric_diffs(
        m, label_a="model", label_b="baseline",
        cluster_col="id", n_boot=200, seed=42,
    )
    assert set(out["metric"]) == {"WAPE", "MAE", "RMSE", "Bias"}
    for _, r in out.iterrows():
        assert np.isfinite(r["diff_point"])
        assert np.isfinite(r["ci_low"])
        assert np.isfinite(r["ci_high"])
        assert r["ci_low"] <= r["ci_high"]


def test_bootstrap_clusters_correctly():
    """Sampling clusters means: if every id has correlated predictions,
    bootstrap CI on Bias should be wider than naive row-bootstrap would
    suggest. We check the structural property that n_clusters in the
    output equals the number of unique ids in input."""
    q = _make_quantile_preds(n_ids=12)
    b = _make_baseline_preds(q)
    m = matched_grid(
        q.assign(prediction=q["p50"]), b,
        keys=("id", "origin_date", "horizon"),
        label_a="model", label_b="baseline",
    )
    out = clustered_bootstrap_metric_diffs(
        m, label_a="model", label_b="baseline",
        cluster_col="id", n_boot=50, seed=0,
    )
    assert (out["n_clusters"] == 12).all()
    assert (out["n_rows"] == len(m)).all()


# --------------------------------------------------------------------------- #
# 3. RMSSE / WRMSSE
# --------------------------------------------------------------------------- #


def _build_synthetic_base_table(
    n_ids: int = 4, n_days: int = 200, rng_seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
    # Cycle through a small palette so any n_ids works.
    cat_palette = ["FOODS", "FOODS", "HOBBIES", "HOUSEHOLD"]
    dept_palette = ["FOODS_1", "FOODS_2", "HOBBIES_1", "HOUSEHOLD_1"]
    rows = []
    for k in range(n_ids):
        id_ = f"FOODS_1_{k:03d}_CA_1_validation"
        cat = cat_palette[k % len(cat_palette)]
        dept = dept_palette[k % len(dept_palette)]
        for d in dates:
            rows.append({
                "id": id_,
                "date": d,
                "sales": float(max(0, int(rng.normal(5.0 + k, 2.0)))),
                "sell_price": 1.5 + 0.5 * k,
                "dept_id": dept,
                "cat_id": cat,
                "store_id": "CA_1",
                "item_id": f"FOODS_1_{k:03d}",
            })
    return pd.DataFrame(rows)


def test_rmsse_for_series_zero_for_perfect_predictions():
    a = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 3.0])
    assert rmsse_for_series(a, p, scale=4.0) == 0.0


def test_rmsse_for_series_handles_zero_scale():
    a = np.array([1.0, 2.0])
    p = np.array([2.0, 3.0])
    assert np.isnan(rmsse_for_series(a, p, scale=0.0))
    assert np.isnan(rmsse_for_series(a, p, scale=float("nan")))


def test_compute_scale_per_id_returns_one_row_per_id():
    base = _build_synthetic_base_table(n_ids=3, n_days=120)
    scales = compute_scale_per_id(
        base, training_end_date="2015-04-01",
    )
    assert set(scales["id"]) == set(base["id"].unique())
    assert (scales["scale"].dropna() > 0).all()


def test_weighted_rmsse_falls_back_to_unweighted_when_no_weight():
    df = pd.DataFrame({
        "rmsse": [1.0, 2.0, 3.0],
        "dollar_sales_weight": [0.0, 0.0, 0.0],
    })
    out = weighted_rmsse(df)
    assert abs(out - 2.0) < 1e-9


def test_weighted_rmsse_dollar_weight():
    df = pd.DataFrame({
        "rmsse": [1.0, 3.0],
        "dollar_sales_weight": [1.0, 3.0],
    })
    # weighted mean = (1*1 + 3*3) / 4 = 10/4 = 2.5
    assert abs(weighted_rmsse(df) - 2.5) < 1e-9


def test_hierarchical_rmsse_runs_on_synthetic_data():
    base = _build_synthetic_base_table(n_ids=4, n_days=200, rng_seed=0)
    # Build a tiny predictions frame for the last 28 days of base.
    origin = pd.Timestamp("2015-06-01")
    horizons = [1, 7, 14, 28]
    target_dates = [origin + pd.Timedelta(days=h) for h in horizons]
    rows = []
    for id_ in base["id"].unique():
        actuals = base.loc[
            (base["id"] == id_) & (base["date"].isin(target_dates)), "sales"
        ].tolist()
        # If a date isn't present, pad with 0 (synthetic; this is fine).
        while len(actuals) < len(horizons):
            actuals.append(0.0)
        for h, td, a in zip(horizons, target_dates, actuals):
            rows.append({
                "id": id_, "origin_date": origin,
                "horizon": h, "target_date": td,
                "actual": a,
                "prediction": a * 0.9,   # mild under-forecast
            })
    preds = pd.DataFrame(rows)
    per_level, wrmsse = hierarchical_rmsse(
        preds, base, training_end_date="2015-05-01",
        model_name="test",
        levels=(("store_id",), ("cat_id",), ("dept_id",), ("id",)),
    )
    assert {"store_id", "cat_id", "dept_id", "id"} == set(per_level["level"])
    assert per_level["weighted_rmsse"].notna().any()
    assert np.isfinite(wrmsse) or wrmsse != wrmsse  # NaN allowed for tiny data


# --------------------------------------------------------------------------- #
# 4. Calibration
# --------------------------------------------------------------------------- #


def test_overall_coverage_full_coverage_gives_1():
    df = pd.DataFrame({
        "p10": [0.0, 0.0, 0.0],
        "p50": [1.0, 2.0, 3.0],
        "p90": [10.0, 10.0, 10.0],
        "actual": [1.0, 2.0, 3.0],
    })
    out = overall_coverage(df, target=0.80)
    assert out["empirical_coverage"] == 1.0
    # Use float tolerance rather than equality (1.0 - 0.8 != exactly 0.2).
    assert abs(out["calibration_error"] - 0.2) < 1e-9, out


def test_overall_coverage_zero_when_actual_outside():
    df = pd.DataFrame({
        "p10": [0.0, 0.0],
        "p50": [1.0, 2.0],
        "p90": [1.0, 2.0],
        "actual": [10.0, 20.0],
    })
    out = overall_coverage(df, target=0.80)
    assert out["empirical_coverage"] == 0.0


def test_coverage_by_group_per_horizon():
    df = pd.DataFrame({
        "horizon": [1, 1, 7, 7],
        "p10": [0, 0, 0, 0],
        "p50": [1, 2, 1, 2],
        "p90": [10, 10, 10, 10],
        "actual": [5, 5, 5, 15],
    })
    out = coverage_by_group(df, by=("horizon",), target=0.80)
    # horizon=1: both rows covered (5 in [0,10])
    # horizon=7: 1/2 covered (5 yes, 15 no)
    assert out.loc[out["horizon"] == 1, "empirical_coverage"].iloc[0] == 1.0
    assert out.loc[out["horizon"] == 7, "empirical_coverage"].iloc[0] == 0.5


def test_per_tail_coverage_signs():
    df = pd.DataFrame({
        "p10": [0.0, 0.0, 0.0, 0.0],
        "p50": [5.0, 5.0, 5.0, 5.0],
        "p90": [10.0, 10.0, 10.0, 10.0],
        "actual": [1.0, 2.0, 12.0, 15.0],  # 2/4 inside, 0/4 below lo, 2/4 above hi
    })
    out = per_tail_coverage(df, nominal_lo=0.10, nominal_hi=0.90)
    assert out["empirical_lo_q"] == 0.0   # none <= 0
    assert out["empirical_hi_q"] == 0.5   # 2 of 4 <= 10


def test_conformal_scaler_narrows_when_over_conservative():
    """If actual is always near p50 and intervals are huge, the conformal
    factor should be < 1 (narrow them)."""
    df = pd.DataFrame({
        "p10": [0.0] * 100,
        "p50": [5.0] * 100,
        "p90": [10.0] * 100,
        "actual": [5.0] * 100,   # always inside
    })
    sc = ConformalScaler(target=0.80).fit(df)
    assert sc.factor_ < 1.0


def test_conformal_scaler_widens_when_underconfident():
    """If actual is far outside on most rows, factor should be > 1."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "p10": [4.0] * n,
        "p50": [5.0] * n,
        "p90": [6.0] * n,
        "actual": rng.normal(5.0, 3.0, size=n),  # most fall outside [4,6]
    })
    sc = ConformalScaler(target=0.80).fit(df)
    assert sc.factor_ > 1.0
    # After widening, eval coverage should be closer to 0.80.
    out = sc.transform(df)
    cov_before = float(
        np.mean((df["actual"] >= df["p10"]) & (df["actual"] <= df["p90"]))
    )
    cov_after = float(
        np.mean((out["actual"] >= out["p10_conformal"])
                & (out["actual"] <= out["p90_conformal"]))
    )
    assert cov_after > cov_before


def test_build_calibration_report_has_overall_row():
    df = pd.DataFrame({
        "horizon": [1, 1, 7, 7],
        "p10": [0, 0, 0, 0], "p50": [1, 2, 1, 2], "p90": [10, 10, 10, 10],
        "actual": [5, 5, 5, 15],
        "cat_id": ["FOODS", "FOODS", "HOBBIES", "HOBBIES"],
    })
    out = build_calibration_report(df, by_groups=[("horizon",), ("cat_id",)])
    assert (out["level"] == "overall").sum() == 1
    assert (out["level"] == "horizon").sum() == 2
    assert (out["level"] == "cat_id").sum() == 2


# --------------------------------------------------------------------------- #
# 5. CLI end-to-end on synthetic inputs
# --------------------------------------------------------------------------- #


def test_credibility_cli_runs_on_synthetic_inputs():
    q = _make_quantile_preds(n_ids=6)
    b = _make_baseline_preds(q)
    base = _build_synthetic_base_table(n_ids=6, n_days=240, rng_seed=0)

    # Make sure ids match between preds and base table; the synthetic
    # quantile builder uses k=0..n-1; the synthetic base does too.
    pred_ids = set(q["id"].unique())
    base_ids = set(base["id"].unique())
    common = pred_ids & base_ids
    q = q.loc[q["id"].isin(common)]
    b = b.loc[b["id"].isin(common)]
    base = base.loc[base["id"].isin(common)]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        q_path = td / "quantile.parquet"
        b_path = td / "baselines.parquet"
        bt_path = td / "base.parquet"
        q.to_parquet(q_path, index=False)
        b.to_parquet(b_path, index=False)
        base.to_parquet(bt_path, index=False)

        reports_dir = td / "reports"
        figures_dir = td / "figures"

        result = rcp.run(
            quantile_path=q_path,
            lightgbm_point_path=None,
            baseline_path=b_path,
            base_table_path=bt_path,
            reports_dir=reports_dir,
            figures_dir=figures_dir,
            baseline_model="moving_average_28",
            n_calibration_origins=1,
            n_boot=100,
            seed=0,
        )

        for name in (
            "headline_model_comparison_ca1.csv",
            "bootstrap_wape_ci_ca1.csv",
            "bias_correction_comparison_ca1.csv",
            "calibration_report_ca1.csv",
            "wrmsse_or_rmsse_ca1.csv",
        ):
            assert (reports_dir / name).exists(), f"missing {name}"

        # Bootstrap CI on WAPE must be finite and ordered.
        boot = pd.read_csv(reports_dir / "bootstrap_wape_ci_ca1.csv")
        wape_row = boot.loc[boot["metric"] == "WAPE"].iloc[0]
        assert np.isfinite(wape_row["ci_low"])
        assert np.isfinite(wape_row["ci_high"])
        assert wape_row["ci_low"] <= wape_row["ci_high"]

        # Bias correction frame has one row per method.
        corr = pd.read_csv(reports_dir / "bias_correction_comparison_ca1.csv")
        assert set(corr["method"]) == set(SUPPORTED_METHODS)

        # The result dict shape we promised.
        assert "headline" in result
        assert "bootstrap" in result
        assert "bias_correction" in result
        assert "wrmsse_per_level" in result
        assert "calibration_table" in result
        assert "conformal_factor" in result


def test_credibility_cli_runs_without_base_table():
    """No base table -> WRMSSE skipped but the rest still produces CSVs."""
    q = _make_quantile_preds(n_ids=5)
    b = _make_baseline_preds(q)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        q_path = td / "quantile.parquet"
        b_path = td / "baselines.parquet"
        q.to_parquet(q_path, index=False)
        b.to_parquet(b_path, index=False)

        reports_dir = td / "reports"
        figures_dir = td / "figures"

        # Passing base_table_path=None opts out of the WRMSSE section.
        # (Passing an explicit non-existent path is treated as a user error
        # and raises, by design.)
        rcp.run(
            quantile_path=q_path,
            baseline_path=b_path,
            base_table_path=None,
            reports_dir=reports_dir,
            figures_dir=figures_dir,
            n_boot=50,
        )

        # Four required CSVs even without WRMSSE.
        for name in (
            "headline_model_comparison_ca1.csv",
            "bootstrap_wape_ci_ca1.csv",
            "bias_correction_comparison_ca1.csv",
            "calibration_report_ca1.csv",
        ):
            assert (reports_dir / name).exists(), f"missing {name}"


if __name__ == "__main__":
    test_multiplicative_mean_ratio_recovers_true_scale()
    test_additive_mean_error_recovers_mean_shift()
    test_per_horizon_multiplicative_is_per_horizon()
    test_corrected_predictions_are_non_negative()
    test_compare_corrections_returns_one_row_per_method()
    test_matched_grid_inner_joins_and_preserves_actual()
    test_bootstrap_returns_finite_ci()
    test_bootstrap_clusters_correctly()
    test_rmsse_for_series_zero_for_perfect_predictions()
    test_rmsse_for_series_handles_zero_scale()
    test_compute_scale_per_id_returns_one_row_per_id()
    test_weighted_rmsse_falls_back_to_unweighted_when_no_weight()
    test_weighted_rmsse_dollar_weight()
    test_hierarchical_rmsse_runs_on_synthetic_data()
    test_overall_coverage_full_coverage_gives_1()
    test_overall_coverage_zero_when_actual_outside()
    test_coverage_by_group_per_horizon()
    test_per_tail_coverage_signs()
    test_conformal_scaler_narrows_when_over_conservative()
    test_conformal_scaler_widens_when_underconfident()
    test_build_calibration_report_has_overall_row()
    test_credibility_cli_runs_on_synthetic_inputs()
    test_credibility_cli_runs_without_base_table()
    print("Credibility pass smoke tests: OK")
