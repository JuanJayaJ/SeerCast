"""Phase 6 smoke tests: probabilistic metrics + QuantileLightGBMModel.

Goals:

1. Pinball / coverage / interval-width / crossing-rate match their
   definitions on hand-checkable inputs.
2. ``fix_quantile_crossings`` enforces ``p10 <= p50 <= p90`` row-wise
   without changing already-monotonic rows.
3. ``QuantileLightGBMModel`` fits, predicts, and the predict() output
   contains the right column names with non-negative values.
4. Pinball-at-0.5 reduces to 0.5 * MAE.
5. End-to-end on synthetic data: coverage is positive (model intervals
   capture some actuals) and predictions carry over the leakage contract
   from the underlying split.

Run::

    python tests/test_phase6_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb  # noqa: F401
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_phase4_smoke import _synthetic_base  # noqa: E402

from seercast.evaluation.metrics import mae
from seercast.evaluation.probabilistic_metrics import (
    all_quantile_metrics,
    coverage,
    crossing_rate,
    fix_quantile_crossings,
    interval_width,
    pinball_loss,
    relative_interval_width,
    score_quantile_by_group,
)
from seercast.features.supervised import build_supervised_table
from seercast.models.quantile_lightgbm import QuantileLightGBMModel
from seercast.training.train_lightgbm import (
    VALID_WINDOW_DAYS,
    split_for_backtest_origin,
)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_pinball_loss_definition():
    """Hand-checked example.

    y    = [10, 0, 5, 5]
    yhat = [12, 1, 4, 6]
    diff = y - yhat = [-2, -1, 1, -1]
    For q=0.9: max(0.9*d, -0.1*d) = [0.2, 0.1, 0.9, 0.1] -> mean 0.325
    For q=0.5: max(0.5*d, -0.5*d) = [1.0, 0.5, 0.5, 0.5] -> mean 0.625
    """
    y = np.array([10.0, 0.0, 5.0, 5.0])
    yhat = np.array([12.0, 1.0, 4.0, 6.0])

    assert abs(pinball_loss(y, yhat, 0.9) - 0.325) < 1e-12
    assert abs(pinball_loss(y, yhat, 0.5) - 0.625) < 1e-12


def test_pinball_at_half_equals_half_mae():
    """A useful sanity check: pinball loss at q=0.5 = 0.5 * MAE."""
    rng = np.random.default_rng(0)
    y = rng.normal(10, 3, size=200)
    yhat = rng.normal(10, 3, size=200)
    assert abs(pinball_loss(y, yhat, 0.5) - 0.5 * mae(y, yhat)) < 1e-10


def test_coverage_and_interval_width():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lo = np.array([0.0, 1.0, 5.0, 3.0, 4.0])
    hi = np.array([2.0, 3.0, 6.0, 5.0, 6.0])
    # in_interval = [T, T, F, T, T] -> coverage 0.8
    assert coverage(y, lo, hi) == 0.8
    # widths = [2, 2, 1, 2, 2] -> mean 1.8
    assert interval_width(lo, hi) == 1.8
    # mean(|y|) = 3 -> relative width = 0.6
    assert abs(relative_interval_width(lo, hi, y) - 0.6) < 1e-12


def test_crossing_rate_and_fix():
    df = pd.DataFrame(
        {
            "p10": [1.0, 2.0, 3.0, 0.5],
            "p50": [2.0, 1.0, 4.0, 1.5],   # row 1 violates p10 <= p50
            "p90": [3.0, 0.0, 2.0, 2.5],   # row 1 and row 2 also violate
        }
    )
    # Two of four rows have crossings.
    assert crossing_rate(df["p10"], df["p50"], df["p90"]) == 0.5

    fixed = fix_quantile_crossings(df, ("p10", "p50", "p90"))
    # After row-wise sort every row should satisfy p10 <= p50 <= p90.
    assert (fixed["p10"] <= fixed["p50"]).all()
    assert (fixed["p50"] <= fixed["p90"]).all()

    # Already-monotonic rows unchanged. Rows 0 and 3 were monotonic.
    assert fixed.loc[0, "p10"] == 1.0 and fixed.loc[0, "p50"] == 2.0 and fixed.loc[0, "p90"] == 3.0
    assert fixed.loc[3, "p10"] == 0.5 and fixed.loc[3, "p50"] == 1.5 and fixed.loc[3, "p90"] == 2.5


def test_score_quantile_by_group_shape():
    df = pd.DataFrame(
        {
            "horizon": [1, 1, 7, 7],
            "p10": [0, 1, 0, 1],
            "p50": [1, 2, 1, 2],
            "p90": [2, 3, 2, 3],
            "actual": [1, 2, 1, 2],
        }
    )
    out = score_quantile_by_group(df, by=["horizon"])
    expected_metric_cols = {
        "n",
        "pinball_p10",
        "pinball_p50",
        "pinball_p90",
        "coverage_p10_p90",
        "interval_width_p10_p90",
        "relative_interval_width_p10_p90",
        "crossing_rate",
    }
    assert set(out.columns) == expected_metric_cols | {"horizon"}
    assert len(out) == 2


# --------------------------------------------------------------------------- #
# QuantileLightGBMModel
# --------------------------------------------------------------------------- #


def _supervised(n_days: int = 250, n_items: int = 5):
    base = _synthetic_base(n_days=n_days, n_items=n_items)
    sup = build_supervised_table(
        base, horizons=[1, 7, 14, 28],
        snap_state="CA", origin_step_days=7,
    )
    return base, sup


def test_quantile_model_fits_predicts_and_columns():
    base, sup = _supervised()
    origin_date = sup["origin_date"].drop_duplicates().sort_values().iloc[len(sup["origin_date"].unique()) // 2]
    split = split_for_backtest_origin(sup, origin_date, valid_window_days=VALID_WINDOW_DAYS)

    m = QuantileLightGBMModel(
        quantiles=(0.1, 0.5, 0.9),
        n_estimators=100, early_stopping_rounds=20,
    )
    m.fit(split.train, valid_df=split.valid)
    preds = m.predict(split.test, fix_crossings=True)

    assert list(preds.columns) == ["p10", "p50", "p90"]
    assert len(preds) == len(split.test)
    assert (preds.values >= 0).all(), "quantile preds clipped to >= 0"
    # After row-wise fix, monotonic.
    assert (preds["p10"] <= preds["p50"]).all()
    assert (preds["p50"] <= preds["p90"]).all()


def test_quantile_model_p50_correlates_with_actuals():
    """Plumbing check on synthetic seasonal data: p50 should correlate
    positively with target_sales. Don't assume any specific WAPE level.
    """
    base, sup = _supervised()
    origin_date = sup["origin_date"].drop_duplicates().sort_values().iloc[len(sup["origin_date"].unique()) // 2]
    split = split_for_backtest_origin(sup, origin_date)

    m = QuantileLightGBMModel(n_estimators=100, early_stopping_rounds=20).fit(
        split.train, valid_df=split.valid
    )
    preds = m.predict(split.test, fix_crossings=True)
    actual = split.test["target_sales"].astype(float).values
    if actual.std() > 0:
        corr = np.corrcoef(preds["p50"].values, actual)[0, 1]
        assert corr > 0.5, f"p50 should correlate with actuals; got {corr:.3f}"


def test_quantile_model_predictions_unchanged_when_post_origin_targets_poisoned():
    """Phase 5 leakage contract carries over to the quantile fitter.

    Poisoning every target whose target_date > origin_date must leave
    the test predictions byte-identical (the train/valid split must not
    have included those rows).
    """
    base, sup = _supervised()
    origin_date = sup["origin_date"].drop_duplicates().sort_values().iloc[len(sup["origin_date"].unique()) // 2]

    split1 = split_for_backtest_origin(sup, origin_date)
    m1 = QuantileLightGBMModel(n_estimators=80, early_stopping_rounds=20)
    m1.fit(split1.train, valid_df=split1.valid)
    preds1 = m1.predict(split1.test, fix_crossings=True)

    poisoned = sup.copy()
    poisoned.loc[poisoned["target_date"] > origin_date, "target_sales"] *= 100
    split2 = split_for_backtest_origin(poisoned, origin_date)
    m2 = QuantileLightGBMModel(n_estimators=80, early_stopping_rounds=20)
    m2.fit(split2.train, valid_df=split2.valid)
    preds2 = m2.predict(split2.test, fix_crossings=True)

    np.testing.assert_array_almost_equal(
        preds1.values, preds2.values, decimal=10,
        err_msg="poisoning post-origin targets changed quantile predictions -- LEAK",
    )


def test_quantile_feature_importance_per_quantile():
    base, sup = _supervised()
    origin_date = sup["origin_date"].drop_duplicates().sort_values().iloc[len(sup["origin_date"].unique()) // 2]
    split = split_for_backtest_origin(sup, origin_date)

    m = QuantileLightGBMModel(n_estimators=50).fit(split.train, valid_df=split.valid)
    imp_50 = m.feature_importance(quantile=0.5)
    imp_90 = m.feature_importance(quantile=0.9)
    assert set(imp_50.columns) == {"feature", "gain"}
    assert imp_50["gain"].is_monotonic_decreasing
    # Different quantile -> potentially different ranking, but both non-empty.
    assert len(imp_50) == len(imp_90) == len(m.feature_columns)


if __name__ == "__main__":
    test_pinball_loss_definition()
    test_pinball_at_half_equals_half_mae()
    test_coverage_and_interval_width()
    test_crossing_rate_and_fix()
    test_score_quantile_by_group_shape()
    test_quantile_model_fits_predicts_and_columns()
    test_quantile_model_p50_correlates_with_actuals()
    test_quantile_model_predictions_unchanged_when_post_origin_targets_poisoned()
    test_quantile_feature_importance_per_quantile()
    print("Phase 6 smoke tests: OK")
