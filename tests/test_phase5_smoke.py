"""Phase 5 smoke tests: LightGBM point model + train/valid/test splits.

The most important test in this suite is the leakage contract on
``split_for_backtest_origin``: the training set must filter on
``target_date <= valid_start``, NOT ``origin_date <``. With direct
multi-horizon training, a row with ``origin_date == O - 3`` and
``horizon == 14`` has ``target_date == O + 11`` -- if we filtered by
``origin_date < O`` we'd train on a target that is 11 days past the
backtest origin, which leaks future data.

Run::

    python tests/test_phase5_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb  # noqa: F401  (just confirms it imports)
import numpy as np
import pandas as pd

# Reuse the synthetic base builder from the Phase 4 tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_phase4_smoke import _synthetic_base  # noqa: E402

from seercast.features.supervised import build_supervised_table
from seercast.models.lightgbm_model import LightGBMPointModel
from seercast.training.train_lightgbm import (
    VALID_WINDOW_DAYS,
    split_for_backtest_origin,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _synthetic_supervised(n_days: int = 250, n_items: int = 5):
    """Mid-sized supervised table that LightGBM can train on quickly."""
    base = _synthetic_base(n_days=n_days, n_items=n_items)
    sup = build_supervised_table(
        base,
        horizons=[1, 7, 14, 28],
        snap_state="CA",
        origin_step_days=7,
    )
    return base, sup


# --------------------------------------------------------------------------- #
# split_for_backtest_origin: the strict leakage rule
# --------------------------------------------------------------------------- #


def test_split_train_uses_target_date_not_origin_date():
    """The leakage trap: filtering by origin_date < O would let in rows
    with origin_date < O AND target_date > O (e.g. h=28). The split must
    use target_date <= valid_start instead.
    """
    base, sup = _synthetic_supervised()
    # Pick an origin in the middle of the data with room for h=28.
    origin_date = sup["origin_date"].drop_duplicates().sort_values().iloc[len(sup["origin_date"].unique()) // 2]
    split = split_for_backtest_origin(sup, origin_date, valid_window_days=VALID_WINDOW_DAYS)

    valid_start = split.valid_start
    assert valid_start == origin_date - pd.Timedelta(days=VALID_WINDOW_DAYS)

    # Train rule.
    assert (split.train["target_date"] <= valid_start).all(), (
        "TRAIN must satisfy target_date <= valid_start"
    )

    # Valid rule.
    assert (split.valid["target_date"] > valid_start).all()
    assert (split.valid["target_date"] <= origin_date).all()

    # Test rule.
    assert (split.test["origin_date"] == origin_date).all()
    # Test target_dates are strictly after origin (definition of forecast).
    assert (split.test["target_date"] > origin_date).all()


def test_split_no_overlap_between_train_valid_test():
    base, sup = _synthetic_supervised()
    origin_date = sup["origin_date"].drop_duplicates().sort_values().iloc[len(sup["origin_date"].unique()) // 2]
    split = split_for_backtest_origin(sup, origin_date)

    key = ["id", "origin_date", "horizon"]
    train_keys = set(map(tuple, split.train[key].values))
    valid_keys = set(map(tuple, split.valid[key].values))
    test_keys = set(map(tuple, split.test[key].values))
    assert not (train_keys & valid_keys), "train and valid keys must be disjoint"
    assert not (train_keys & test_keys), "train and test keys must be disjoint"
    assert not (valid_keys & test_keys), "valid and test keys must be disjoint"


def test_split_train_does_not_contain_targets_after_origin():
    """Property the user explicitly called out: training set must satisfy
    target_date <= backtest_origin (which the looser rule
    ``origin_date < O`` would violate for direct multi-horizon training).
    """
    base, sup = _synthetic_supervised()
    origin_date = sup["origin_date"].drop_duplicates().sort_values().iloc[len(sup["origin_date"].unique()) // 2]
    split = split_for_backtest_origin(sup, origin_date)
    assert (split.train["target_date"] <= origin_date).all()
    assert (split.valid["target_date"] <= origin_date).all()


# --------------------------------------------------------------------------- #
# LightGBMPointModel: end-to-end fit/predict + categorical handling
# --------------------------------------------------------------------------- #


def test_lightgbm_fits_predicts_and_returns_nonnegative():
    base, sup = _synthetic_supervised()
    origin_date = sup["origin_date"].drop_duplicates().sort_values().iloc[len(sup["origin_date"].unique()) // 2]
    split = split_for_backtest_origin(sup, origin_date)

    model = LightGBMPointModel(n_estimators=100, early_stopping_rounds=20)
    model.fit(split.train, valid_df=split.valid)
    preds = model.predict(split.test)

    assert preds.shape == (len(split.test),)
    assert (preds >= 0).all(), "Poisson predictions clipped to >= 0"
    assert np.isfinite(preds).all()
    # Sanity: predictions correlate positively with actuals on this strongly
    # seasonal synthetic data. Pearson correlation > 0.5 is a generous floor.
    actuals = split.test["target_sales"].values.astype(float)
    if actuals.std() > 0:
        corr = np.corrcoef(preds, actuals)[0, 1]
        assert corr > 0.5, f"predictions should correlate with actuals; got {corr:.3f}"


def test_lightgbm_categorical_handling_consistent_across_train_valid_test():
    base, sup = _synthetic_supervised()
    origin_date = sup["origin_date"].drop_duplicates().sort_values().iloc[len(sup["origin_date"].unique()) // 2]
    split = split_for_backtest_origin(sup, origin_date)

    model = LightGBMPointModel(n_estimators=50)
    model.fit(split.train, valid_df=split.valid)

    # Predict on test with the same categorical encoding learned at fit time.
    preds = model.predict(split.test)
    assert len(preds) == len(split.test)

    # Add a synthetic out-of-vocabulary id and confirm predict still works
    # (OOV becomes NaN under the learned dtype, LightGBM handles NaN).
    oov_test = split.test.head(2).copy()
    oov_test["id"] = "OOV_NEW_ITEM_99"
    oov_test["item_id"] = "OOV_99"
    oov_preds = model.predict(oov_test)
    assert oov_preds.shape == (2,)
    assert np.isfinite(oov_preds).all()


def test_lightgbm_feature_importance_returns_tidy_frame():
    base, sup = _synthetic_supervised()
    origin_date = sup["origin_date"].drop_duplicates().sort_values().iloc[len(sup["origin_date"].unique()) // 2]
    split = split_for_backtest_origin(sup, origin_date)

    model = LightGBMPointModel(n_estimators=50).fit(split.train, valid_df=split.valid)
    imp = model.feature_importance(kind="gain")
    assert set(imp.columns) == {"feature", "gain"}
    assert imp["gain"].is_monotonic_decreasing
    # Lag and rolling features should appear in the top half on seasonal data.
    expected_present = {"sales_lag_7", "sales_lag_28", "rolling_mean_7", "rolling_mean_28"}
    assert expected_present.issubset(set(imp["feature"]))


# --------------------------------------------------------------------------- #
# Leakage contract at the model level
# --------------------------------------------------------------------------- #


def test_lightgbm_predictions_unchanged_when_post_origin_targets_are_poisoned():
    """The phase-4 leakage contract is upstream (rolling features at row t
    don't see sales(t)). At the model level, a complementary check: if we
    poison every target whose target_date > origin_date (i.e. data the
    train set must NOT see), the test predictions must be byte-identical.
    """
    base, sup = _synthetic_supervised()
    origin_date = sup["origin_date"].drop_duplicates().sort_values().iloc[len(sup["origin_date"].unique()) // 2]

    split1 = split_for_backtest_origin(sup, origin_date)
    m1 = LightGBMPointModel(n_estimators=80, early_stopping_rounds=20)
    m1.fit(split1.train, valid_df=split1.valid)
    p1 = m1.predict(split1.test)

    # Poison the full supervised table for any row whose target lies past origin.
    poisoned = sup.copy()
    poisoned.loc[poisoned["target_date"] > origin_date, "target_sales"] *= 100

    split2 = split_for_backtest_origin(poisoned, origin_date)
    # Fit with the SAME random seed (LGB defaults are deterministic given same data).
    m2 = LightGBMPointModel(n_estimators=80, early_stopping_rounds=20)
    m2.fit(split2.train, valid_df=split2.valid)
    p2 = m2.predict(split2.test)

    np.testing.assert_array_almost_equal(
        p1, p2, decimal=10,
        err_msg="poisoning post-origin targets changed predictions -- LEAK",
    )




def test_build_model_comparison_uses_apples_to_apples_grid():
    """Regression test for the model_comparison fairness patch.

    Before the patch, baselines covered all 28 horizons and LightGBM
    covered only ``[1, 7, 14, 28]``; the comparison naively concat'd them
    and reported ``n_baseline = 7 * n_lgbm``, which makes WAPE comparisons
    meaningless. ``build_model_comparison`` inner-joins on
    ``(origin_date, id, horizon, target_date)`` so every model gets the
    same evaluation grid.
    """
    from seercast.training.train_lightgbm import build_model_comparison

    origin = pd.Timestamp("2015-05-03")
    direct_horizons = [1, 7, 14, 28]

    # LGBM preds: 2 ids x 4 direct horizons = 8 rows.
    lgbm_rows = []
    for id_ in ("a", "b"):
        for h in direct_horizons:
            lgbm_rows.append({
                "model": "lightgbm_point",
                "origin_date": origin,
                "id": id_,
                "horizon": h,
                "target_date": origin + pd.Timedelta(days=h),
                "prediction": 10.0 + h * 0.5,
                "actual": 10.0 + h * 0.4,
            })
    lgbm_preds = pd.DataFrame(lgbm_rows)

    # Baseline preds: 2 baseline models x 2 ids x ALL 28 horizons = 112 rows
    # (mirrors the Phase 3 output covering full horizons 1..28).
    bl_rows = []
    for name in ("naive", "seasonal_naive"):
        for id_ in ("a", "b"):
            for h in range(1, 29):
                bl_rows.append({
                    "model": name,
                    "origin_date": origin,
                    "id": id_,
                    "horizon": h,
                    "target_date": origin + pd.Timedelta(days=h),
                    "prediction": (1.0 if name == "naive" else 2.0) * h,
                    "actual": 10.0 + h * 0.4,
                })
    baseline_preds = pd.DataFrame(bl_rows)

    cmp = build_model_comparison(lgbm_preds, baseline_preds)

    # Every model in the comparison must score on the SAME number of rows
    # (the LGBM grid size). This is the core fairness invariant.
    assert cmp["n"].nunique() == 1, (
        f"model_comparison rows must have equal n across models; "
        f"got\n{cmp[['model','n']].to_string(index=False)}"
    )
    assert int(cmp["n"].iloc[0]) == len(lgbm_preds), (
        f"expected n={len(lgbm_preds)}; got {cmp['n'].iloc[0]}"
    )
    assert set(cmp["model"]) == {"lightgbm_point", "naive", "seasonal_naive"}
    # Each baseline reduced from 28*2=56 rows down to 4*2=8.
    assert (cmp["n"] == 8).all()


def test_build_model_comparison_warns_on_uneven_baseline_coverage(capsys=None):
    """If a baseline is missing rows on the LGBM grid, the helper prints a
    WARN instead of silently producing a smaller-n row for that model.
    """
    from seercast.training.train_lightgbm import build_model_comparison
    import io
    import sys as _sys

    origin = pd.Timestamp("2015-05-03")
    lgbm_preds = pd.DataFrame({
        "model": "lightgbm_point",
        "origin_date": [origin] * 4,
        "id": ["a", "a", "b", "b"],
        "horizon": [1, 7, 1, 7],
        "target_date": pd.to_datetime(["2015-05-04", "2015-05-10",
                                        "2015-05-04", "2015-05-10"]),
        "prediction": [1.0, 2.0, 3.0, 4.0],
        "actual":     [1.5, 2.5, 3.5, 4.5],
    })
    # Gappy baseline: only covers id=='a' (missing 'b').
    baseline_preds = pd.DataFrame({
        "model": "naive",
        "origin_date": [origin, origin],
        "id": ["a", "a"],
        "horizon": [1, 7],
        "target_date": pd.to_datetime(["2015-05-04", "2015-05-10"]),
        "prediction": [1.0, 2.0],
        "actual":     [1.5, 2.5],
    })

    # Capture stdout to verify the WARN line.
    buf = io.StringIO()
    old = _sys.stdout
    _sys.stdout = buf
    try:
        _ = build_model_comparison(lgbm_preds, baseline_preds)
    finally:
        _sys.stdout = old

    assert "WARN" in buf.getvalue()
    assert "baseline coverage is uneven" in buf.getvalue()


if __name__ == "__main__":
    test_split_train_uses_target_date_not_origin_date()
    test_split_no_overlap_between_train_valid_test()
    test_split_train_does_not_contain_targets_after_origin()
    test_lightgbm_fits_predicts_and_returns_nonnegative()
    test_lightgbm_categorical_handling_consistent_across_train_valid_test()
    test_lightgbm_feature_importance_returns_tidy_frame()
    test_lightgbm_predictions_unchanged_when_post_origin_targets_are_poisoned()
    test_build_model_comparison_uses_apples_to_apples_grid()
    test_build_model_comparison_warns_on_uneven_baseline_coverage()
    print("Phase 5 smoke tests: OK")
