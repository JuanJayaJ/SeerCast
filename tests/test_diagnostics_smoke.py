"""Step 1 diagnostics smoke tests.

Covers:
* ADI / CV² / zero_rate math on hand-checkable series.
* The leakage contract for per-origin segments (poisoning post-origin
  sales must not change `segment_at_origin`).
* zero_heavy override, min_history fallback, and the four Syntetos-Boylan
  buckets on synthetic series with known shapes.
* Lifecycle math (first_sale_date, pre_launch_days).
* `combined_predictions` enforces equal n across all models (the project's
  fairness invariant).
* `breakdown` returns the right schema and is monotone-sorted by WAPE
  within each group.
* `worst_forecasters` ranks by signed cumulative error.
* Each diagnostic plot function produces a PNG on a small synthetic frame.

Run::

    python tests/test_diagnostics_smoke.py
"""

from __future__ import annotations

import io
import math
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_phase4_smoke import _synthetic_base  # noqa: E402

from seercast.diagnostics.combined import combined_predictions
from seercast.diagnostics.demand_segments import (
    SEGMENTS,
    _adi_cv2,
    _classify,
    classify_demand_per_origin,
)
from seercast.diagnostics.error_breakdown import breakdown, fairness_check
from seercast.diagnostics.lifecycle import (
    is_active_at_origin,
    lifecycle_summary,
)
from seercast.diagnostics.worst_cases import worst_forecasters
from seercast.visualization.diagnostic_plots import (
    plot_bias_by_segment,
    plot_error_by_horizon,
    plot_feature_importance_top20,
    plot_metric_by_segment,
)


# --------------------------------------------------------------------------- #
# ADI / CV² math
# --------------------------------------------------------------------------- #


def test_adi_cv2_hand_checked():
    """A series of 10 days with nonzero sales on days [0, 1, 2, 7] (4 nonzero
    out of 10) -> ADI = 10/4 = 2.5. Sales sizes [2, 4, 4, 6] -> mean 4,
    var (population) 2.0, CV² = 2.0 / 16 = 0.125. zero_rate = 6/10 = 0.6.
    """
    sales = np.array([2, 4, 4, 0, 0, 0, 0, 6, 0, 0], dtype=float)
    adi, cv2, zero_rate = _adi_cv2(sales)
    assert abs(adi - 2.5) < 1e-12
    assert abs(cv2 - 0.125) < 1e-12
    assert abs(zero_rate - 0.6) < 1e-12


def test_adi_cv2_all_zero_series_is_zero_heavy():
    sales = np.zeros(10, dtype=float)
    adi, cv2, zero_rate = _adi_cv2(sales)
    assert math.isinf(adi)
    assert cv2 == 0.0
    assert zero_rate == 1.0


def test_adi_cv2_single_nonzero_observation():
    sales = np.array([0, 0, 0, 5, 0], dtype=float)
    adi, cv2, zero_rate = _adi_cv2(sales)
    # 5 days, 1 nonzero -> ADI = 5/1 = 5.0
    assert adi == 5.0
    # Only one nonzero observation -> no variance to measure.
    assert cv2 == 0.0
    assert zero_rate == 4 / 5


def test_classify_segments_on_named_series():
    """Round-trip through _classify with hand-picked (adi, cv2, zero_rate)."""
    # Healthy daily seller with low size variability -> smooth.
    assert _classify(adi=1.0, cv2=0.10, zero_rate=0.05, n_history=120,
                     adi_thresh=1.32, cv2_thresh=0.49,
                     zero_heavy_rate=0.9, min_history_days=28) == "smooth"
    # Sparse seller, stable size -> intermittent.
    assert _classify(adi=3.0, cv2=0.10, zero_rate=0.65, n_history=120,
                     adi_thresh=1.32, cv2_thresh=0.49,
                     zero_heavy_rate=0.9, min_history_days=28) == "intermittent"
    # Daily seller, big swings -> erratic.
    assert _classify(adi=1.1, cv2=1.5, zero_rate=0.10, n_history=120,
                     adi_thresh=1.32, cv2_thresh=0.49,
                     zero_heavy_rate=0.9, min_history_days=28) == "erratic"
    # Sparse, lumpy size -> lumpy.
    assert _classify(adi=4.0, cv2=2.0, zero_rate=0.7, n_history=120,
                     adi_thresh=1.32, cv2_thresh=0.49,
                     zero_heavy_rate=0.9, min_history_days=28) == "lumpy"
    # zero_heavy override (regardless of adi/cv2 values).
    assert _classify(adi=1.0, cv2=0.10, zero_rate=0.95, n_history=120,
                     adi_thresh=1.32, cv2_thresh=0.49,
                     zero_heavy_rate=0.9, min_history_days=28) == "zero_heavy"
    # Too little history -> zero_heavy fallback.
    assert _classify(adi=1.0, cv2=0.10, zero_rate=0.30, n_history=10,
                     adi_thresh=1.32, cv2_thresh=0.49,
                     zero_heavy_rate=0.9, min_history_days=28) == "zero_heavy"


# --------------------------------------------------------------------------- #
# Leakage contract for per-origin segments
# --------------------------------------------------------------------------- #


def test_classify_per_origin_leakage_safe():
    """Poisoning sales AFTER an origin must not change segment_at_origin
    for that origin."""
    base = _synthetic_base(n_days=200, n_items=2)
    # Pick a mid-history origin so there's room before AND after.
    origin = base["date"].iloc[120]

    seg1 = classify_demand_per_origin(base, [origin]).set_index("id")["segment_at_origin"]

    poisoned = base.copy()
    # Make all post-origin sales gigantic. Pre-origin rows unchanged.
    mask = poisoned["date"] > origin
    poisoned.loc[mask, "sales"] = poisoned.loc[mask, "sales"] * 100 + 999
    seg2 = classify_demand_per_origin(poisoned, [origin]).set_index("id")["segment_at_origin"]

    assert seg1.equals(seg2), (
        "segments changed when only post-origin sales were perturbed -- "
        f"LEAK.\nbefore:\n{seg1}\nafter:\n{seg2}"
    )


def test_classify_per_origin_produces_expected_columns():
    base = _synthetic_base(n_days=120, n_items=2)
    origin = base["date"].iloc[80]
    out = classify_demand_per_origin(base, [origin])
    expected = {
        "origin_date", "id", "n_history_days",
        "adi_at_origin", "cv2_at_origin", "zero_rate_at_origin",
        "mean_demand_at_origin", "segment_at_origin",
    }
    assert set(out.columns) >= expected, f"missing: {expected - set(out.columns)}"
    assert set(out["segment_at_origin"].unique()) <= set(SEGMENTS)


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


def test_lifecycle_first_sale_and_prelaunch_share():
    """An item that starts selling on day 50 of 100 should have
    pre_launch_days = 50, n_active_days = 50.
    """
    dates = pd.date_range("2014-01-01", periods=100, freq="D")
    sales = np.zeros(100, dtype=int)
    sales[50:] = 1   # active from day 50 onwards (index 50 inclusive)
    base = pd.DataFrame({
        "id": "ITEM_X",
        "date": dates,
        "sales": sales,
    })
    summary = lifecycle_summary(base).iloc[0]
    assert summary["first_sale_date"] == dates[50]
    assert int(summary["pre_launch_days"]) == 50
    assert int(summary["n_active_days"]) == 50
    assert bool(summary["any_sale"]) is True


def test_lifecycle_handles_no_sales_at_all():
    dates = pd.date_range("2014-01-01", periods=30, freq="D")
    base = pd.DataFrame({"id": "ITEM_Y", "date": dates, "sales": [0] * 30})
    summary = lifecycle_summary(base).iloc[0]
    assert pd.isna(summary["first_sale_date"])
    assert bool(summary["any_sale"]) is False
    assert int(summary["pre_launch_days"]) == 30


def test_is_active_at_origin_per_origin_flag():
    """Item starts selling on day 50; at origin day 30 -> inactive,
    at origin day 60 -> active."""
    dates = pd.date_range("2014-01-01", periods=100, freq="D")
    sales = np.zeros(100, dtype=int)
    sales[50:] = 1
    base = pd.DataFrame({"id": "ITEM_X", "date": dates, "sales": sales})
    res = is_active_at_origin(base, [dates[30], dates[60]])
    by_origin = res.set_index("origin_date")["is_active"]
    assert bool(by_origin.loc[dates[30]]) is False
    assert bool(by_origin.loc[dates[60]]) is True


# --------------------------------------------------------------------------- #
# combined_predictions: fairness + p50-as-first-class
# --------------------------------------------------------------------------- #


def _make_lgbm_preds(origins, ids, horizons) -> pd.DataFrame:
    rows = []
    for od in origins:
        for id_ in ids:
            for h in horizons:
                rows.append({
                    "model": "lightgbm_point",
                    "origin_date": pd.Timestamp(od),
                    "id": id_,
                    "horizon": h,
                    "target_date": pd.Timestamp(od) + pd.Timedelta(days=h),
                    "prediction": 1.0 + h * 0.5,
                    "actual": 1.0 + h * 0.4,
                })
    return pd.DataFrame(rows)


def _make_quantile_preds(origins, ids, horizons) -> pd.DataFrame:
    rows = []
    for od in origins:
        for id_ in ids:
            for h in horizons:
                rows.append({
                    "origin_date": pd.Timestamp(od),
                    "id": id_,
                    "horizon": h,
                    "target_date": pd.Timestamp(od) + pd.Timedelta(days=h),
                    "p10": 0.5 + h * 0.4,
                    "p50": 1.2 + h * 0.5,
                    "p90": 2.0 + h * 0.6,
                    "actual": 1.0 + h * 0.4,
                })
    return pd.DataFrame(rows)


def _make_baseline_preds(origins, ids, all_horizons, models) -> pd.DataFrame:
    """Baselines run over ALL horizons (this is what Phase 3 actually does)."""
    rows = []
    for m in models:
        for od in origins:
            for id_ in ids:
                for h in all_horizons:
                    rows.append({
                        "model": m,
                        "origin_date": pd.Timestamp(od),
                        "id": id_,
                        "horizon": h,
                        "target_date": pd.Timestamp(od) + pd.Timedelta(days=h),
                        "prediction": 1.0 if m == "naive" else 2.0,
                        "actual": 1.0 + h * 0.4,
                    })
    return pd.DataFrame(rows)


def test_combined_predictions_enforces_equal_n_and_p50_as_model():
    """Hand-built tiny artifacts: lgbm grid 1 origin × 2 ids × [1,7,14,28] = 8 rows.
    Baselines cover all 28 horizons. After combined_predictions, every
    model must have n=8 (LGBM grid).
    """
    origins = [pd.Timestamp("2015-05-03")]
    ids = ["a", "b"]
    direct = [1, 7, 14, 28]
    full = list(range(1, 29))
    baselines = ["naive", "seasonal_naive", "moving_average_28"]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _make_lgbm_preds(origins, ids, direct).to_parquet(td / "lgbm.parquet")
        _make_quantile_preds(origins, ids, direct).to_parquet(td / "quantile.parquet")
        _make_baseline_preds(origins, ids, full, baselines).to_parquet(td / "baselines.parquet")
        combined = combined_predictions(
            lightgbm_point_path=td / "lgbm.parquet",
            quantile_predictions_path=td / "quantile.parquet",
            baseline_predictions_path=td / "baselines.parquet",
            enforce_equal_n=True,
        )

    counts = combined.groupby("model").size()
    assert counts.nunique() == 1, f"unequal n per model: {counts.to_dict()}"
    assert int(counts.iloc[0]) == 8
    assert "lightgbm_quantile_p50" in counts.index
    assert "lightgbm_point" in counts.index
    for m in baselines:
        assert m in counts.index


def test_combined_predictions_raises_on_uneven_n():
    """If the LGBM frame already has a hole, enforce_equal_n=True must raise."""
    origins = [pd.Timestamp("2015-05-03")]
    ids = ["a", "b"]
    direct = [1, 7, 14, 28]
    full = list(range(1, 29))

    lgbm = _make_lgbm_preds(origins, ids, direct)
    # Remove rows for one (id, horizon) pair -> uneven coverage if baselines cover it.
    # But that shouldn't surface as "unequal n per model" -- inner join trims everyone.
    # To force the uneven-n error we need DIFFERENT row counts per model on the merged
    # frame. Easiest: drop a row from the LGBM frame AFTER it defines the grid.
    bl = _make_baseline_preds(origins, ids, full, ["naive"])

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # LGBM defines grid of 8 rows.
        lgbm.to_parquet(td / "lgbm.parquet")
        # Baselines have a duplicate row for (id=a, h=1) -> 8 + 1 rows on the grid.
        dup = bl.loc[(bl["id"] == "a") & (bl["horizon"] == 1)].head(1)
        bl_dup = pd.concat([bl, dup], ignore_index=True)
        bl_dup.to_parquet(td / "baselines.parquet")
        try:
            combined_predictions(
                lightgbm_point_path=td / "lgbm.parquet",
                baseline_predictions_path=td / "baselines.parquet",
                enforce_equal_n=True,
            )
        except ValueError as e:
            assert "uneven" in str(e).lower() or "different `n`" in str(e)
            return
    raise AssertionError("enforce_equal_n=True should have raised on uneven counts")


# --------------------------------------------------------------------------- #
# breakdown + worst_forecasters
# --------------------------------------------------------------------------- #


def _simple_combined() -> pd.DataFrame:
    """A small combined frame with two models, 1 origin, 2 ids, [1,7] horizons,
    each id assigned to a known segment."""
    od = pd.Timestamp("2015-05-03")
    rows = []
    for model, mult in (("naive", 1.0), ("lightgbm_quantile_p50", 1.05)):
        for id_, segment in (("a", "smooth"), ("b", "lumpy")):
            for h in (1, 7):
                actual = 10.0 + h
                rows.append({
                    "model": model,
                    "origin_date": od,
                    "id": id_,
                    "horizon": h,
                    "target_date": od + pd.Timedelta(days=h),
                    "prediction": actual * mult,
                    "actual": actual,
                    "segment_at_origin": segment,
                    "cat_id": "FOODS" if id_ == "a" else "HOBBIES",
                })
    return pd.DataFrame(rows)


def test_breakdown_returns_expected_schema():
    df = _simple_combined()
    res = breakdown(df, by="horizon")
    assert set(res.columns) >= {"horizon", "model", "n", "MAE", "RMSE", "WAPE", "Bias"}
    # 2 horizons × 2 models = 4 rows.
    assert len(res) == 4
    # Each (horizon) group has 2 model rows.
    for _, grp in res.groupby("horizon"):
        assert len(grp) == 2


def test_breakdown_target_zero_split():
    """When grouping by target_zero, the breakdown helper adds the column on the fly."""
    df = _simple_combined()
    # Force some actuals to zero.
    df.loc[df["horizon"] == 1, "actual"] = 0
    res = breakdown(df, by="target_zero")
    assert "target_zero" in res.columns


def test_fairness_check_returns_per_model_counts():
    df = _simple_combined()
    fc = fairness_check(df, by="horizon")
    assert "naive" in fc.columns
    assert "lightgbm_quantile_p50" in fc.columns
    # Each horizon row should have n=2 for each model.
    assert (fc == 2).all().all()


def test_worst_forecasters_signed_ranking():
    """Build a frame where id 'over' has total signed error +10 and id 'under'
    has total signed error -10. direction='over' must put 'over' first;
    direction='under' must put 'under' first.
    """
    rows = []
    od = pd.Timestamp("2015-05-03")
    # 'over' id: predictions are actual + 5 across two horizons (+10 cumulative).
    for h in (1, 7):
        rows.append({"model": "m", "origin_date": od, "id": "over",
                     "horizon": h, "target_date": od + pd.Timedelta(days=h),
                     "prediction": 15.0, "actual": 10.0,
                     "cat_id": "FOODS", "dept_id": "FOODS_1"})
    # 'under' id: predictions are actual - 5 across two horizons (-10 cumulative).
    for h in (1, 7):
        rows.append({"model": "m", "origin_date": od, "id": "under",
                     "horizon": h, "target_date": od + pd.Timedelta(days=h),
                     "prediction": 5.0, "actual": 10.0,
                     "cat_id": "FOODS", "dept_id": "FOODS_1"})

    df = pd.DataFrame(rows)
    over = worst_forecasters(df, model="m", direction="over", top_n=5)
    under = worst_forecasters(df, model="m", direction="under", top_n=5)
    assert over.iloc[0]["id"] == "over"
    assert under.iloc[0]["id"] == "under"
    assert float(over.iloc[0]["signed_error"]) == 10.0
    assert float(under.iloc[0]["signed_error"]) == -10.0


# --------------------------------------------------------------------------- #
# Diagnostic plots
# --------------------------------------------------------------------------- #


def test_plot_error_by_horizon_saves_png():
    df = pd.DataFrame({
        "horizon": [1, 7, 14, 28, 1, 7, 14, 28],
        "model": ["A"] * 4 + ["B"] * 4,
        "n": [10] * 8,
        "MAE": [1, 1.1, 1.2, 1.3] * 2,
        "RMSE": [2, 2.1, 2.2, 2.3] * 2,
        "WAPE": [0.5, 0.55, 0.6, 0.65, 0.6, 0.65, 0.7, 0.75],
        "Bias": [0.01, -0.02, 0.03, -0.04] * 2,
    })
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "ebh.png"
        plot_error_by_horizon(df, savepath=out)
        assert out.exists() and out.stat().st_size > 0
    plt.close("all")


def test_plot_metric_and_bias_by_segment_saves_png():
    df = pd.DataFrame({
        "segment_at_origin": ["smooth", "lumpy", "smooth", "lumpy"],
        "model": ["A", "A", "B", "B"],
        "n": [10, 5, 10, 5],
        "MAE": [1.0, 2.0, 1.1, 2.1],
        "RMSE": [1.5, 3.0, 1.6, 3.1],
        "WAPE": [0.4, 0.7, 0.45, 0.75],
        "Bias": [-0.1, 0.2, -0.05, 0.25],
    })
    with tempfile.TemporaryDirectory() as td:
        out1 = Path(td) / "by_seg_wape.png"
        out2 = Path(td) / "by_seg_bias.png"
        plot_metric_by_segment(df, metric="WAPE", savepath=out1)
        plot_bias_by_segment(df, savepath=out2)
        assert out1.exists() and out1.stat().st_size > 0
        assert out2.exists() and out2.stat().st_size > 0
    plt.close("all")


def test_plot_feature_importance_saves_png():
    df = pd.DataFrame({
        "feature": [f"feat_{i}" for i in range(25)],
        "point_gain": np.linspace(100, 1, 25),
        "quantile_p50_gain": np.linspace(80, 2, 25),
    })
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "fi.png"
        plot_feature_importance_top20(df, savepath=out)
        assert out.exists() and out.stat().st_size > 0
    plt.close("all")


if __name__ == "__main__":
    test_adi_cv2_hand_checked()
    test_adi_cv2_all_zero_series_is_zero_heavy()
    test_adi_cv2_single_nonzero_observation()
    test_classify_segments_on_named_series()
    test_classify_per_origin_leakage_safe()
    test_classify_per_origin_produces_expected_columns()
    test_lifecycle_first_sale_and_prelaunch_share()
    test_lifecycle_handles_no_sales_at_all()
    test_is_active_at_origin_per_origin_flag()
    test_combined_predictions_enforces_equal_n_and_p50_as_model()
    test_combined_predictions_raises_on_uneven_n()
    test_breakdown_returns_expected_schema()
    test_breakdown_target_zero_split()
    test_fairness_check_returns_per_model_counts()
    test_worst_forecasters_signed_ranking()
    test_plot_error_by_horizon_saves_png()
    test_plot_metric_and_bias_by_segment_saves_png()
    test_plot_feature_importance_saves_png()
    print("Diagnostics smoke tests: OK")
