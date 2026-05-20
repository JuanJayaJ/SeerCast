"""Smoke tests for the Phase 9 objective sweep.

Covers:

* log1p transform / expm1 inverse round-trip
* corrected predictions are non-negative
* CandidateSpec.build_params merges correctly
* default / fast catalog contents
* fit_candidate runs for every supported objective on tiny synthetic data
* predictions from log1p candidates are non-negative
* CLI run() on a tiny synthetic supervised frame writes the expected files
* matched grid keeps an equal n across candidates in the summary frame

Tests use very small synthetic supervised tables and tiny n_estimators so
the suite stays fast (<30 s locally).

Run::

    python -m pytest tests/test_objective_sweep_smoke.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from seercast.features.supervised import SUPERVISED_COLUMNS, IDENTITY_COLUMNS
from seercast.models.objective_lightgbm import (
    CandidateSpec,
    ObjectiveCandidateModel,
    TARGET_TRANSFORMS,
    default_catalog,
    fast_catalog,
    fit_candidate,
)
from seercast.training import run_objective_sweep as ros


# --------------------------------------------------------------------------- #
# Synthetic supervised table
# --------------------------------------------------------------------------- #


_ORIGINS = (
    pd.Timestamp("2015-03-08"),
    pd.Timestamp("2015-05-03"),
    pd.Timestamp("2015-06-28"),
)
_HORIZONS = (1, 7, 14, 28)


def _make_synthetic_supervised(
    n_ids: int = 12,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Build a tiny supervised frame with all 64 SUPERVISED_COLUMNS so
    LightGBMPointModel.fit will run end-to-end.

    Most numeric features are mild noise; target_sales is a small count
    drawn from a Poisson-ish distribution. Origins span the project's
    three canonical backtest dates so :func:`split_for_backtest_origin`
    has meaningful train/valid/test slices.
    """
    rng = np.random.default_rng(rng_seed)
    cats = ["FOODS", "HOBBIES", "HOUSEHOLD"]
    depts = ["FOODS_1", "FOODS_2", "FOODS_3",
            "HOBBIES_1", "HOBBIES_2",
            "HOUSEHOLD_1", "HOUSEHOLD_2"]

    # Pre-build a few origins ago to give the train slice content.
    # Each origin gets every (id, horizon) combination. Plus we add
    # an "older training" origin whose target_date sits earlier
    # so train slice for the first real origin isn't empty.
    older_origin = _ORIGINS[0] - pd.Timedelta(days=120)
    all_origins = [older_origin] + list(_ORIGINS)

    rows: list[dict] = []
    for k in range(n_ids):
        id_ = f"FOODS_1_{k:03d}_CA_1_validation"
        cat = cats[k % len(cats)]
        dept = depts[k % len(depts)]
        for origin in all_origins:
            for h in _HORIZONS:
                target_date = origin + pd.Timedelta(days=int(h))
                base_mean = 2.0 + 0.5 * k
                # Identity columns first (so they line up with IDENTITY_COLUMNS).
                row = {
                    "id": id_,
                    "item_id": f"FOODS_1_{k:03d}",
                    "dept_id": dept,
                    "cat_id": cat,
                    "store_id": "CA_1",
                    "state_id": "CA",
                    "origin_date": origin,
                    "horizon": int(h),
                    "target_date": target_date,
                    "target_sales": float(rng.poisson(base_mean)),
                }
                # Fill the remaining SUPERVISED_COLUMNS with noise.
                for col in SUPERVISED_COLUMNS:
                    if col in row:
                        continue
                    if col.startswith("days_") or col.endswith("_count_28"):
                        row[col] = float(rng.integers(0, 365))
                    elif col.startswith("is_") or col.startswith("has_") or col.endswith("_flag"):
                        row[col] = int(rng.integers(0, 2))
                    elif col.startswith("target_") and col != "target_sell_price":
                        row[col] = float(rng.integers(0, 12))
                    else:
                        row[col] = float(rng.normal(base_mean, 0.5))
                rows.append(row)
    df = pd.DataFrame(rows)
    # Ensure column order matches SUPERVISED_COLUMNS.
    return df[list(SUPERVISED_COLUMNS)]


def _make_synthetic_base_table(supervised: pd.DataFrame) -> pd.DataFrame:
    """A tiny base table so the CLI's WRMSSE / origin-resolution code
    doesn't error.

    Requires the M5 ``d`` column (sequential ``"d_N"``) so the CLI's
    unconditional ``read_parquet(..., columns=['id','d','date'])`` works
    even when our test origins are already Timestamps.
    """
    ids = supervised[["id", "item_id", "dept_id", "cat_id", "store_id"]].drop_duplicates("id")
    end_date = max(supervised["origin_date"].max(), supervised["target_date"].max())
    start_date = supervised["origin_date"].min() - pd.Timedelta(days=180)
    dates = pd.date_range(start_date, end_date, freq="D")
    d_by_date = {dt: f"d_{i+1}" for i, dt in enumerate(dates)}
    rng = np.random.default_rng(0)
    rows = []
    for _, r in ids.iterrows():
        for d in dates:
            rows.append({
                "id": r["id"],
                "d": d_by_date[d],
                "date": d,
                "sales": float(max(0, int(rng.normal(3.0, 1.5)))),
                "sell_price": 1.5,
                "dept_id": r["dept_id"],
                "cat_id": r["cat_id"],
                "store_id": r["store_id"],
            })
    return pd.DataFrame(rows)


def _make_synthetic_baseline_preds(supervised: pd.DataFrame) -> pd.DataFrame:
    """A naive baseline predictions frame matching the supervised test grid.

    The CLI's matched-grid scorer filters by model='moving_average_28' and
    horizon in (1,7,14,28). We just use target_sales as the prediction
    (perfect oracle baseline) for simplicity; the test only checks that
    the matched-grid plumbing works, not numerical baseline quality.
    """
    test_grid = supervised.loc[supervised["origin_date"].isin(list(_ORIGINS))][
        ["origin_date", "id", "horizon", "target_date", "target_sales"]
    ].copy()
    test_grid["prediction"] = test_grid["target_sales"].astype(float)
    test_grid["actual"] = test_grid["target_sales"].astype(float)
    test_grid["model"] = "moving_average_28"
    return test_grid.drop(columns=["target_sales"])[
        ["model", "origin_date", "id", "horizon", "target_date", "prediction", "actual"]
    ]


# --------------------------------------------------------------------------- #
# 1. Pure-function transforms
# --------------------------------------------------------------------------- #


def test_log1p_roundtrip():
    """expm1(log1p(y)) == y for y >= 0."""
    y = np.array([0.0, 1.0, 5.0, 100.0, 1234.5])
    forward, inverse = TARGET_TRANSFORMS["log1p"]
    out = inverse(forward(y))
    assert np.allclose(out, y, atol=1e-9)


def test_log1p_transform_clips_negatives_at_zero():
    """log1p input is clipped at 0 so we never produce NaN."""
    forward, _ = TARGET_TRANSFORMS["log1p"]
    out = forward(np.array([-1.0, -0.5, 0.0, 5.0]))
    assert np.all(np.isfinite(out))
    assert out[0] == 0.0
    assert out[1] == 0.0


def test_identity_transform_is_identity():
    forward, inverse = TARGET_TRANSFORMS["identity"]
    y = np.array([0.0, 1.0, 5.0, -2.0])
    assert np.allclose(forward(y), y)
    assert np.allclose(inverse(y), y)


# --------------------------------------------------------------------------- #
# 2. CandidateSpec config
# --------------------------------------------------------------------------- #


def test_candidate_spec_build_params_merges_extras():
    spec = CandidateSpec(
        name="tweedie_1_2",
        objective="tweedie",
        extra_params={"tweedie_variance_power": 1.2, "learning_rate": 0.01},
    )
    params = spec.build_params()
    assert params["objective"] == "tweedie"
    assert params["tweedie_variance_power"] == 1.2
    # Extras override defaults:
    assert params["learning_rate"] == 0.01


def test_default_catalog_with_and_without_tweedie():
    with_t = default_catalog(include_tweedie=True)
    no_t = default_catalog(include_tweedie=False)
    with_names = {c.name for c in with_t}
    no_names = {c.name for c in no_t}
    # Tweedie variants only appear in the full catalog:
    assert "tweedie_1_2" in with_names
    assert "tweedie_1_2" not in no_names
    # Non-tweedie candidates appear in both:
    for name in ("poisson", "regression_l2", "regression_l1",
                 "log1p_regression_l2", "log1p_regression_l1"):
        assert name in with_names
        assert name in no_names


def test_fast_catalog_is_subset_and_small():
    fast = fast_catalog()
    assert len(fast) <= 5
    names = {c.name for c in fast}
    assert {"poisson", "regression_l2"}.issubset(names)


# --------------------------------------------------------------------------- #
# 3. fit_candidate on tiny synthetic data
# --------------------------------------------------------------------------- #


def _build_train_valid_test(n_ids: int = 8):
    """Use the project's own split_for_backtest_origin so the test exercises
    the same code path the CLI uses."""
    from seercast.training.train_lightgbm import split_for_backtest_origin
    sup = _make_synthetic_supervised(n_ids=n_ids)
    split = split_for_backtest_origin(
        sup, origin_date=_ORIGINS[1], valid_window_days=56,
    )
    return split.train, split.valid, split.test


def test_fit_candidate_l2_runs_and_predicts_non_negative():
    train, valid, test = _build_train_valid_test()
    spec = CandidateSpec(name="regression_l2", objective="regression")
    model = fit_candidate(
        spec, train, valid,
        n_estimators=10, early_stopping_rounds=5,
    )
    preds = model.predict(test)
    assert preds.shape == (len(test),)
    assert np.all(preds >= 0.0)
    assert np.all(np.isfinite(preds))


def test_fit_candidate_log1p_l2_runs_and_predicts_non_negative():
    """The log1p branch: even if expm1(yhat) is theoretically negative for
    yhat < 0, the wrapper clips at 0."""
    train, valid, test = _build_train_valid_test()
    spec = CandidateSpec(
        name="log1p_regression_l2",
        objective="regression",
        target_transform="log1p",
    )
    model = fit_candidate(
        spec, train, valid,
        n_estimators=10, early_stopping_rounds=5,
    )
    preds = model.predict(test)
    assert np.all(preds >= 0.0)
    assert np.all(np.isfinite(preds))


def test_fit_candidate_poisson_runs():
    train, valid, test = _build_train_valid_test()
    spec = CandidateSpec(name="poisson", objective="poisson")
    model = fit_candidate(spec, train, valid,
                          n_estimators=10, early_stopping_rounds=5)
    preds = model.predict(test)
    assert np.all(preds >= 0.0)


def test_fit_candidate_tweedie_runs():
    train, valid, test = _build_train_valid_test()
    spec = CandidateSpec(
        name="tweedie_1_2", objective="tweedie",
        extra_params={"tweedie_variance_power": 1.2},
    )
    model = fit_candidate(spec, train, valid,
                          n_estimators=10, early_stopping_rounds=5)
    preds = model.predict(test)
    assert np.all(preds >= 0.0)


# --------------------------------------------------------------------------- #
# 4. End-to-end CLI on synthetic inputs
# --------------------------------------------------------------------------- #


def test_run_sweep_on_synthetic_writes_expected_files():
    """Drive ros.run() on a tiny synthetic supervised + base + baseline
    set. Verifies the expected CSVs land on disk, the summary frame has
    one row per candidate, and matched-grid n is equal across rows."""
    sup = _make_synthetic_supervised(n_ids=8)
    base_table = _make_synthetic_base_table(sup)
    baseline_preds = _make_synthetic_baseline_preds(sup)

    # Use a custom 2-candidate catalog so the sweep is fast and we can
    # check the "one row per candidate" invariant.
    catalog = [
        CandidateSpec(name="regression_l2", objective="regression",
                      description="L2 candidate."),
        CandidateSpec(name="log1p_regression_l2", objective="regression",
                      target_transform="log1p",
                      description="log1p L2 candidate."),
    ]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sup_path = td / "supervised.parquet"
        base_path = td / "base.parquet"
        baselines_path = td / "baselines.parquet"
        sup.to_parquet(sup_path, index=False)
        base_table.to_parquet(base_path, index=False)
        baseline_preds.to_parquet(baselines_path, index=False)

        reports_dir = td / "reports"
        figures_dir = td / "figures"

        result = ros.run(
            supervised_path=sup_path,
            base_table_path=base_path,
            baseline_predictions_path=baselines_path,
            # Don't auto-discover incumbent quantile preds from the real workspace.
            incumbent_quantile_path=td / "no_such_incumbent.parquet",
            reports_dir=reports_dir,
            figures_dir=figures_dir,
            # Pass Timestamp origins so the CLI uses them directly
            # without consulting M5's d-column mapping.
            backtest_origins=(_ORIGINS[1],),
            catalog=catalog,
            n_estimators=15,
            early_stopping_rounds=5,
            skip_bootstrap=True,        # bootstrap requires more rows
            skip_wrmsse=True,           # WRMSSE requires longer history
        )

        # Required CSVs:
        for name in (
            "objective_sweep_summary_ca1.csv",
            "objective_sweep_by_horizon_ca1.csv",
        ):
            assert (reports_dir / name).exists(), f"missing {name}"
        assert (reports_dir / "objective_sweep_predictions_ca1.parquet").exists()

        # Summary has one row per (candidate + baseline) — we passed
        # incumbent=None and the baseline IS present.
        summary = pd.read_csv(reports_dir / "objective_sweep_summary_ca1.csv")
        candidate_rows = summary.loc[summary["model"].isin(
            ["regression_l2", "log1p_regression_l2"]
        )]
        assert len(candidate_rows) == 2

        # Matched-grid invariant: equal n across candidates in the summary.
        assert candidate_rows["n"].nunique() == 1, (
            f"candidates have unequal n in summary:\n{candidate_rows}"
        )

        # Metric values are finite.
        for col in ("MAE", "RMSE", "WAPE", "Bias"):
            assert np.all(np.isfinite(candidate_rows[col].to_numpy())), col

        # Result dict shape we promised:
        assert "summary" in result
        assert "by_horizon" in result
        assert "predictions" in result
        assert "matched" in result


def test_run_sweep_handles_missing_optional_inputs():
    """Pass non-existent incumbent + baseline paths -> CLI should still
    produce the candidate summary frame (just without comparison rows)."""
    sup = _make_synthetic_supervised(n_ids=6)
    base_table = _make_synthetic_base_table(sup)

    catalog = [
        CandidateSpec(name="regression_l2", objective="regression",
                      description="L2 candidate."),
    ]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sup_path = td / "supervised.parquet"
        base_path = td / "base.parquet"
        sup.to_parquet(sup_path, index=False)
        base_table.to_parquet(base_path, index=False)

        reports_dir = td / "reports"
        figures_dir = td / "figures"

        result = ros.run(
            supervised_path=sup_path,
            base_table_path=base_path,
            baseline_predictions_path=td / "no_baselines.parquet",
            incumbent_quantile_path=td / "no_incumbent.parquet",
            reports_dir=reports_dir,
            figures_dir=figures_dir,
            backtest_origins=(_ORIGINS[1],),
            catalog=catalog,
            n_estimators=10,
            early_stopping_rounds=3,
            skip_bootstrap=True,
            skip_wrmsse=True,
        )

        summary = pd.read_csv(reports_dir / "objective_sweep_summary_ca1.csv")
        # Only the single candidate should be in the summary.
        assert len(summary) == 1
        assert summary.iloc[0]["model"] == "regression_l2"


if __name__ == "__main__":
    test_log1p_roundtrip()
    test_log1p_transform_clips_negatives_at_zero()
    test_identity_transform_is_identity()
    test_candidate_spec_build_params_merges_extras()
    test_default_catalog_with_and_without_tweedie()
    test_fast_catalog_is_subset_and_small()
    test_fit_candidate_l2_runs_and_predicts_non_negative()
    test_fit_candidate_log1p_l2_runs_and_predicts_non_negative()
    test_fit_candidate_poisson_runs()
    test_fit_candidate_tweedie_runs()
    test_run_sweep_on_synthetic_writes_expected_files()
    test_run_sweep_handles_missing_optional_inputs()
    print("Objective sweep smoke tests: OK")
