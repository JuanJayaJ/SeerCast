"""Phase 3 smoke tests: metrics, baselines, rolling-origin backtester.

Goals:

1. Metrics match their definitions on hand-checkable inputs.
2. Each baseline produces a complete (id x horizon) prediction grid with no
   NaNs, and uses only history at or before the origin.
3. The rolling-origin backtester:
   - Slices history correctly (no leakage).
   - Joins predictions with actuals.
   - Runs all four baselines end-to-end on a small synthetic series.
4. On a synthetic series with clean weekly seasonality, seasonal baselines
   beat non-seasonal ones on WAPE.

Run::

    python tests/test_phase3_smoke.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seercast.evaluation.backtesting import _origin_to_date, rolling_origin_backtest
from seercast.evaluation.metrics import (
    all_point_metrics,
    bias,
    mae,
    rmse,
    score_by_group,
    wape,
)
from seercast.models.baselines import (
    MovingAverageBaseline,
    NaiveBaseline,
    SeasonalMovingAverageBaseline,
    SeasonalNaiveBaseline,
    all_baselines,
)


# --------------------------------------------------------------------------- #
# Synthetic base table: 3 ids x 200 days with a strong weekly cycle
# --------------------------------------------------------------------------- #


def _synthetic_base(n_days: int = 200, n_ids: int = 3, seed: int = 0) -> pd.DataFrame:
    """Long base table mimicking what build_base_table produces.

    Each id has the same weekly profile [1, 2, 3, 4, 5, 8, 6] (Mon..Sun)
    multiplied by a per-id scale, plus a tiny bit of noise. Strong enough
    weekly seasonality that seasonal_naive should beat naive.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2014-01-06", periods=n_days, freq="D")  # 2014-01-06 is a Monday
    rows = []
    weekly = np.array([1, 2, 3, 4, 5, 8, 6], dtype=float)
    for i in range(n_ids):
        scale = 1.0 + i  # 1, 2, 3
        for j, dt in enumerate(dates):
            wd = dt.weekday()
            base_val = weekly[wd] * scale
            noise = rng.normal(0, 0.3)
            sales = max(0, int(round(base_val + noise)))
            rows.append(
                {
                    "id": f"item_{i}",
                    "date": dt,
                    "d": f"d_{j + 1}",
                    "sales": sales,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_metrics_match_definitions():
    y = np.array([10.0, 0.0, 5.0, 5.0])
    yhat = np.array([12.0, 1.0, 4.0, 6.0])

    # Hand-computed:
    #   |y - yhat| = [2, 1, 1, 1] -> sum 5, mean 1.25
    #   (y - yhat)^2 = [4, 1, 1, 1] -> mean 1.75 -> sqrt ~ 1.3229
    #   sum |y| = 20 -> WAPE = 5 / 20 = 0.25
    #   sum (yhat - y) = 3 -> Bias = 3 / 20 = 0.15
    assert abs(mae(y, yhat) - 1.25) < 1e-12
    assert abs(rmse(y, yhat) - np.sqrt(1.75)) < 1e-12
    assert abs(wape(y, yhat) - 0.25) < 1e-12
    assert abs(bias(y, yhat) - 0.15) < 1e-12

    bag = all_point_metrics(y, yhat)
    assert set(bag) == {"MAE", "RMSE", "WAPE", "Bias"}


def test_wape_returns_nan_when_actual_is_all_zero():
    import math
    assert math.isnan(wape([0, 0, 0], [1, 2, 3]))
    assert math.isnan(bias([0, 0, 0], [1, 2, 3]))


def test_score_by_group_shapes():
    df = pd.DataFrame(
        {
            "model": ["a", "a", "b", "b"],
            "horizon": [1, 2, 1, 2],
            "actual": [10, 10, 10, 10],
            "prediction": [11, 9, 12, 8],
        }
    )
    by_model = score_by_group(df, by=["model"])
    assert set(by_model.columns) == {"model", "n", "MAE", "RMSE", "WAPE", "Bias"}
    assert len(by_model) == 2

    by_pair = score_by_group(df, by=["model", "horizon"])
    assert len(by_pair) == 4


# --------------------------------------------------------------------------- #
# Baselines: shape and leakage contract
# --------------------------------------------------------------------------- #


def test_each_baseline_emits_complete_grid_and_no_nans():
    base = _synthetic_base()
    origin_date = base["date"].iloc[100]
    history = base.loc[base["date"] <= origin_date]
    ids = history["id"].unique()
    horizons = [1, 7, 14, 28]

    for cls in (
        NaiveBaseline,
        SeasonalNaiveBaseline,
        MovingAverageBaseline,
        SeasonalMovingAverageBaseline,
    ):
        m = cls()
        m.fit(history)
        preds = m.forecast(ids, origin_date, horizons)
        # Complete grid: |ids| * |horizons| rows, exactly one row per pair.
        assert len(preds) == len(ids) * len(horizons)
        assert preds["prediction"].isna().sum() == 0
        assert set(preds.columns) >= {"id", "origin_date", "horizon", "target_date", "prediction"}
        # target_date == origin_date + horizon days
        assert (
            (preds["target_date"] - preds["origin_date"]).dt.days
            == preds["horizon"]
        ).all()


def test_baseline_fit_does_not_see_future_data():
    """Leakage contract: nothing past origin_date may influence predictions.

    We fit on history truncated at origin, then mutate base sales AFTER
    origin and refit -- predictions must NOT change.
    """
    base = _synthetic_base()
    origin_date = base["date"].iloc[100]
    history = base.loc[base["date"] <= origin_date]
    ids = history["id"].unique()

    m1 = SeasonalNaiveBaseline()
    m1.fit(history)
    preds1 = m1.forecast(ids, origin_date, [1, 7, 14, 28])

    # Mutate the future portion of base only; refit using same history slice.
    polluted = base.copy()
    polluted.loc[polluted["date"] > origin_date, "sales"] *= 100  # noise bomb
    history2 = polluted.loc[polluted["date"] <= origin_date]  # same slice as before
    m2 = SeasonalNaiveBaseline()
    m2.fit(history2)
    preds2 = m2.forecast(ids, origin_date, [1, 7, 14, 28])

    pd.testing.assert_frame_equal(preds1, preds2)


# --------------------------------------------------------------------------- #
# Rolling-origin backtester
# --------------------------------------------------------------------------- #


def test_origin_to_date_accepts_int_str_and_datetime():
    base = _synthetic_base()
    d = _origin_to_date(base, 50)
    assert d == base.loc[base["d"] == "d_50", "date"].iloc[0]
    assert _origin_to_date(base, "d_50") == d
    assert _origin_to_date(base, d) == d


def test_rolling_origin_backtest_runs_and_joins_actuals():
    base = _synthetic_base()
    # Pick origins in the middle so 28-day horizons fit inside the data.
    origins = [100, 130, 160]

    preds = rolling_origin_backtest(
        base=base,
        model_factory=NaiveBaseline,
        origins=origins,
        horizons=[1, 7, 14, 28],
        model_name="naive",
    )

    assert set(preds.columns) == {
        "model",
        "origin_date",
        "id",
        "horizon",
        "target_date",
        "prediction",
        "actual",
    }
    # No NaN actuals (we picked origins so all 28d windows fit in base).
    assert preds["actual"].isna().sum() == 0
    # Origin x ids x horizons with all 3 ids in the synthetic frame.
    assert len(preds) == len(origins) * 3 * 4

    # Actuals come from base on (id, target_date).
    sample = preds.iloc[0]
    truth = base.loc[
        (base["id"] == sample["id"]) & (base["date"] == sample["target_date"]), "sales"
    ].iloc[0]
    assert float(sample["actual"]) == float(truth)


def test_seasonal_baseline_beats_naive_on_seasonal_data():
    """Sanity check: with a strong weekly cycle, seasonal_naive should have
    lower WAPE than plain naive. If it doesn't, something is wrong with
    either the baseline or the backtester wiring.
    """
    base = _synthetic_base()
    origins = [100, 130, 160]

    out = []
    for name, factory in all_baselines().items():
        out.append(
            rolling_origin_backtest(
                base=base,
                model_factory=factory,
                origins=origins,
                horizons=list(range(1, 29)),
                model_name=name,
            )
        )
    all_preds = pd.concat(out, ignore_index=True)
    summary = score_by_group(all_preds, by=["model"]).set_index("model")

    assert summary.loc["seasonal_naive", "WAPE"] < summary.loc["naive", "WAPE"], (
        f"seasonal_naive should beat naive on seasonal data, got\n{summary}"
    )
    assert (
        summary.loc["seasonal_moving_average", "WAPE"]
        < summary.loc["moving_average_28", "WAPE"]
    ), (
        f"seasonal_moving_average should beat plain moving_average_28, got\n{summary}"
    )


if __name__ == "__main__":
    test_metrics_match_definitions()
    test_wape_returns_nan_when_actual_is_all_zero()
    test_score_by_group_shapes()
    test_each_baseline_emits_complete_grid_and_no_nans()
    test_baseline_fit_does_not_see_future_data()
    test_origin_to_date_accepts_int_str_and_datetime()
    test_rolling_origin_backtest_runs_and_joins_actuals()
    test_seasonal_baseline_beats_naive_on_seasonal_data()
    print("Phase 3 smoke tests: OK")
