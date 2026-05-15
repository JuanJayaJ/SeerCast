"""Per-horizon LightGBM experiment smoke tests.

Covers:

1. Per-horizon training filters train_df to one horizon per sub-model.
2. Predictions cover every input row and route to the correct sub-model.
3. p10 <= p50 <= p90 after fix_crossings.
4. Leakage rule from Phase 5 is preserved: train rows satisfy
   ``target_date <= O - 56``.
5. Output frame contains all four direct horizons.
6. ``build_three_way_comparison`` returns equal `n` per (model, version),
   covers baseline / lifecycle / per_horizon versions, and the
   per-horizon table breaks out by horizon.
7. ``horizon`` is intentionally excluded from per-horizon sub-model
   feature lists (constant column -> useless feature).

Run::

    python tests/test_per_horizon_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb  # noqa: F401
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_phase4_smoke import _synthetic_base  # noqa: E402

from seercast.features.supervised import build_supervised_table
from seercast.models.per_horizon_lightgbm import (
    PerHorizonLightGBMPointModel,
    PerHorizonQuantileLightGBMModel,
    _features_without_horizon,
)
from seercast.training.run_lifecycle_experiment import build_before_vs_after  # noqa: F401
from seercast.training.run_per_horizon_experiment import (
    build_three_way_comparison,
)
from seercast.training.train_lightgbm import (
    VALID_WINDOW_DAYS,
    split_for_backtest_origin,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _supervised(n_days: int = 260, n_items: int = 5):
    base = _synthetic_base(n_days=n_days, n_items=n_items)
    sup = build_supervised_table(
        base, horizons=[1, 7, 14, 28],
        snap_state="CA", origin_step_days=7,
    )
    return base, sup


def _split_at_middle(sup: pd.DataFrame):
    origins = sup["origin_date"].drop_duplicates().sort_values()
    od = origins.iloc[len(origins) // 2]
    split = split_for_backtest_origin(sup, od, valid_window_days=VALID_WINDOW_DAYS)
    if len(split.train) == 0 or len(split.valid) == 0:
        for od in origins.iloc[len(origins) // 2 :]:
            split = split_for_backtest_origin(sup, od, valid_window_days=VALID_WINDOW_DAYS)
            if len(split.train) > 0 and len(split.valid) > 0:
                break
    return od, split




class _ShamEmptyPerHorizon:
    """Pickle-safe fake per-horizon model used by warning-path tests.

    This class must live at module scope. Joblib/pickle cannot serialize
    classes defined inside test functions on Windows/Python 3.14.
    """

    @property
    def trained_horizons(self):
        return [1, 7, 14, 28]

    def feature_importance(self, *, horizon, quantile=None, kind="gain"):
        return pd.DataFrame(columns=["feature", kind])


# --------------------------------------------------------------------------- #
# Per-horizon QUANTILE
# --------------------------------------------------------------------------- #


def test_per_horizon_quantile_fits_per_horizon_sub_models():
    """After fit, the wrapper holds one sub-model per requested horizon."""
    _, sup = _supervised()
    _, split = _split_at_middle(sup)

    m = PerHorizonQuantileLightGBMModel(
        horizons=(1, 7, 14, 28),
        quantiles=(0.1, 0.5, 0.9),
        n_estimators=40, early_stopping_rounds=10,
    )
    m.fit(split.train, valid_df=split.valid)
    assert sorted(m._models) == [1, 7, 14, 28]
    # Every sub-model produced a booster (i.e. fit completed).
    for h, sub in m._models.items():
        assert sub._boosters, f"sub-model for horizon {h} has no boosters"


def test_per_horizon_quantile_predictions_cover_full_grid_and_are_monotonic():
    _, sup = _supervised()
    _, split = _split_at_middle(sup)

    m = PerHorizonQuantileLightGBMModel(
        n_estimators=40, early_stopping_rounds=10,
    )
    m.fit(split.train, valid_df=split.valid)
    preds = m.predict(split.test, fix_crossings=True)

    # Same number of rows as test.
    assert len(preds) == len(split.test)
    # All four horizons covered.
    assert set(split.test["horizon"].unique()) == {1, 7, 14, 28}
    # Monotonic after fix-up.
    assert (preds["p10"] <= preds["p50"] + 1e-9).all()
    assert (preds["p50"] <= preds["p90"] + 1e-9).all()
    # Non-negative (clip floor).
    assert (preds.values >= 0).all()


def test_per_horizon_quantile_no_leakage_in_sub_model_train_rows():
    """Sales poisoned at target_date > origin should leave per-horizon
    predictions on the test set byte-identical -- the same leakage check
    we have for the single-model variant."""
    _, sup = _supervised()
    od, _ = _split_at_middle(sup)

    split1 = split_for_backtest_origin(sup, od, VALID_WINDOW_DAYS)
    m1 = PerHorizonQuantileLightGBMModel(n_estimators=30, early_stopping_rounds=10)
    m1.fit(split1.train, valid_df=split1.valid)
    preds1 = m1.predict(split1.test, fix_crossings=True)

    poisoned = sup.copy()
    poisoned.loc[poisoned["target_date"] > od, "target_sales"] *= 100
    split2 = split_for_backtest_origin(poisoned, od, VALID_WINDOW_DAYS)
    m2 = PerHorizonQuantileLightGBMModel(n_estimators=30, early_stopping_rounds=10)
    m2.fit(split2.train, valid_df=split2.valid)
    preds2 = m2.predict(split2.test, fix_crossings=True)

    np.testing.assert_array_almost_equal(
        preds1.values, preds2.values, decimal=10,
        err_msg="per-horizon quantile predictions changed under post-origin poisoning -- LEAK",
    )


def test_per_horizon_quantile_each_sub_model_saw_only_its_horizon():
    """Sanity: the sub-model for horizon h must only have learned from
    rows where horizon == h. We verify this by injecting an artificial
    feature that's constant per horizon, then checking that predictions
    on a fixed feature row depend ONLY on its declared horizon (not on
    poisoning rows of other horizons in the training frame).
    """
    _, sup = _supervised()
    od, _ = _split_at_middle(sup)
    split = split_for_backtest_origin(sup, od, VALID_WINDOW_DAYS)

    # Fit baseline per-horizon model.
    m_baseline = PerHorizonQuantileLightGBMModel(n_estimators=30, early_stopping_rounds=10)
    m_baseline.fit(split.train, valid_df=split.valid)
    preds_baseline = m_baseline.predict(split.test, fix_crossings=True)

    # Poison ONLY horizon-1 training labels heavily. If the per-horizon
    # model is correctly isolated, predictions at horizons 7/14/28 are
    # unchanged.
    poisoned_train = split.train.copy()
    poisoned_train.loc[poisoned_train["horizon"] == 1, "target_sales"] *= 100
    m_poisoned = PerHorizonQuantileLightGBMModel(n_estimators=30, early_stopping_rounds=10)
    m_poisoned.fit(poisoned_train, valid_df=split.valid)
    preds_poisoned = m_poisoned.predict(split.test, fix_crossings=True)

    # Predictions for horizons 7/14/28 must match the baseline (other
    # sub-models never saw horizon-1 rows).
    mask_other = split.test["horizon"].isin([7, 14, 28]).values
    np.testing.assert_array_almost_equal(
        preds_baseline.loc[mask_other].values,
        preds_poisoned.loc[mask_other].values, decimal=10,
        err_msg="poisoning horizon-1 train rows changed predictions for other horizons -- ISOLATION BROKEN",
    )
    # Horizon-1 predictions SHOULD change.
    mask_h1 = (split.test["horizon"] == 1).values
    if mask_h1.any():
        diff = np.abs(preds_baseline.loc[mask_h1].values - preds_poisoned.loc[mask_h1].values).sum()
        assert diff > 0, "horizon-1 predictions identical despite poisoning -- something is wrong"


def test_per_horizon_excludes_horizon_from_feature_columns():
    base_features = _features_without_horizon(None)
    assert "horizon" not in base_features
    # Sanity: at least the lag/rolling/identity columns survive.
    for required in ("sales_lag_1", "sales_lag_7", "rolling_mean_28",
                     "id", "item_id", "dept_id"):
        assert required in base_features


# --------------------------------------------------------------------------- #
# Per-horizon POINT
# --------------------------------------------------------------------------- #


def test_per_horizon_point_fits_predicts_nonnegative():
    _, sup = _supervised()
    _, split = _split_at_middle(sup)

    m = PerHorizonLightGBMPointModel(n_estimators=40, early_stopping_rounds=10)
    m.fit(split.train, valid_df=split.valid)
    preds = m.predict(split.test)
    assert preds.shape == (len(split.test),)
    assert (preds >= 0).all()
    assert np.isfinite(preds).all()
    assert m.trained_horizons == [1, 7, 14, 28]


# --------------------------------------------------------------------------- #
# build_three_way_comparison
# --------------------------------------------------------------------------- #


def _fake_predictions(version: str, n_ids=2, horizons=(1, 7, 14, 28), with_quantile=True):
    origin = pd.Timestamp("2015-05-03")
    rows_pt = []
    rows_q = []
    for id_ in [f"id_{i}" for i in range(n_ids)]:
        for h in horizons:
            target = origin + pd.Timedelta(days=h)
            actual = 5.0 + h * 0.4
            rows_pt.append({
                "model": "lightgbm_point",
                "origin_date": origin, "id": id_, "horizon": h,
                "target_date": target,
                "prediction": actual * (0.95 if version == "per_horizon"
                                        else (1.0 if version == "baseline_features" else 1.02)),
                "actual": actual,
            })
            rows_q.append({
                "origin_date": origin, "id": id_, "horizon": h,
                "target_date": target,
                "p10": actual - 1, "p50": actual + (0.1 if version == "per_horizon" else 0.4),
                "p90": actual + 2, "actual": actual,
            })
    pt = pd.DataFrame(rows_pt)
    q = pd.DataFrame(rows_q) if with_quantile else None
    return pt, q


def test_build_three_way_comparison_covers_all_versions_equal_n():
    base_pt, base_q = _fake_predictions("baseline_features")
    life_pt, life_q = _fake_predictions("lifecycle_features")
    ph_pt, ph_q = _fake_predictions("per_horizon")

    overall, by_h = build_three_way_comparison(
        baseline_point=base_pt, baseline_quantile=base_q,
        lifecycle_point=life_pt, lifecycle_quantile=life_q,
        per_horizon_quantile=ph_q, per_horizon_point=ph_pt,
    )

    # Six combinations expected:
    #   lightgbm_point baseline + lifecycle
    #   per_horizon_lightgbm_point per_horizon
    #   lightgbm_quantile_p50 baseline + lifecycle
    #   per_horizon_lightgbm_quantile_p50 per_horizon
    combos = set(zip(overall["model"], overall["version"]))
    assert ("lightgbm_point", "baseline_features") in combos
    assert ("lightgbm_point", "lifecycle_features") in combos
    assert ("per_horizon_lightgbm_point", "per_horizon") in combos
    assert ("lightgbm_quantile_p50", "baseline_features") in combos
    assert ("lightgbm_quantile_p50", "lifecycle_features") in combos
    assert ("per_horizon_lightgbm_quantile_p50", "per_horizon") in combos

    # Every (model, version) row scored on the same n.
    assert overall["n"].nunique() == 1

    # By-horizon table contains all four horizons.
    assert set(by_h["horizon"].unique()) == {1, 7, 14, 28}


def test_build_three_way_comparison_handles_missing_inputs():
    """Missing benchmark predictions just skip those rows; the helper
    doesn't crash."""
    ph_pt, ph_q = _fake_predictions("per_horizon")
    overall, by_h = build_three_way_comparison(
        baseline_point=None,
        baseline_quantile=None,
        lifecycle_point=None,
        lifecycle_quantile=None,
        per_horizon_quantile=ph_q,
        per_horizon_point=ph_pt,
    )
    versions = set(overall["version"])
    assert versions == {"per_horizon"}
    assert not overall.empty
    # By-horizon still has all four horizons present.
    assert set(by_h["horizon"].unique()) == {1, 7, 14, 28}




# --------------------------------------------------------------------------- #
# feature_importance_combined supports per-horizon bundles
# --------------------------------------------------------------------------- #


def test_feature_importance_combined_handles_per_horizon_bundle(tmp_path=None):
    """End-to-end: a per-horizon bundle on disk must produce a clean
    importance frame without raising the old
    ``TypeError: feature_importance() missing 1 required keyword-only
    argument: 'horizon'``.
    """
    import joblib
    import tempfile
    from pathlib import Path

    from seercast.diagnostics.feature_importance import feature_importance_combined

    _, sup = _supervised()
    _, split = _split_at_middle(sup)

    # Fit one per-horizon quantile model and one per-horizon point model.
    q_model = PerHorizonQuantileLightGBMModel(
        n_estimators=20, early_stopping_rounds=5,
    ).fit(split.train, valid_df=split.valid)
    p_model = PerHorizonLightGBMPointModel(
        n_estimators=20, early_stopping_rounds=5,
    ).fit(split.train, valid_df=split.valid)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        q_bundle = {"models": {"2015-05-03": q_model}, "horizons": [1, 7, 14, 28]}
        p_bundle = {"models": {"2015-05-03": p_model}, "horizons": [1, 7, 14, 28]}
        q_path = td / "q_bundle.pkl"
        p_path = td / "p_bundle.pkl"
        joblib.dump(q_bundle, q_path)
        joblib.dump(p_bundle, p_path)

        fi = feature_importance_combined(
            point_bundle_path=p_path,
            quantile_bundle_path=q_path,
        )

    assert not fi.empty
    assert "feature" in fi.columns
    assert "point_gain" in fi.columns
    assert "point_split" in fi.columns
    assert "quantile_p50_gain" in fi.columns
    # Importance values must be finite floats.
    for col in ("point_gain", "point_split", "quantile_p50_gain"):
        finite = fi[col].dropna()
        assert finite.notna().sum() > 0, f"no finite values for {col}"
        assert (finite >= 0).all(), f"{col} should be non-negative"


def test_feature_importance_combined_normal_bundle_still_works(tmp_path=None):
    """Regression: single-model bundles must still work after the patch."""
    import joblib
    import tempfile
    from pathlib import Path

    from seercast.diagnostics.feature_importance import feature_importance_combined
    from seercast.models.lightgbm_model import LightGBMPointModel
    from seercast.models.quantile_lightgbm import QuantileLightGBMModel

    _, sup = _supervised()
    _, split = _split_at_middle(sup)

    p_model = LightGBMPointModel(n_estimators=20, early_stopping_rounds=5)
    p_model.fit(split.train, valid_df=split.valid)
    q_model = QuantileLightGBMModel(n_estimators=20, early_stopping_rounds=5)
    q_model.fit(split.train, valid_df=split.valid)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p_path = td / "p.pkl"
        q_path = td / "q.pkl"
        joblib.dump({"models": {"2015-05-03": p_model}}, p_path)
        joblib.dump({"models": {"2015-05-03": q_model}}, q_path)

        fi = feature_importance_combined(
            point_bundle_path=p_path,
            quantile_bundle_path=q_path,
        )

    assert not fi.empty
    assert set(fi.columns) >= {"feature", "point_gain", "point_split", "quantile_p50_gain"}


def test_per_horizon_importance_aggregates_across_horizons():
    """The patched _agg_importance_from_bundle should average importance
    across all trained horizons inside a per-horizon model. We verify by
    comparing the aggregate result against manual per-horizon means."""
    import joblib
    import tempfile
    from pathlib import Path

    from seercast.diagnostics.feature_importance import (
        _agg_importance_from_bundle,
        _is_per_horizon,
    )

    _, sup = _supervised()
    _, split = _split_at_middle(sup)
    m = PerHorizonQuantileLightGBMModel(
        n_estimators=20, early_stopping_rounds=5,
    ).fit(split.train, valid_df=split.valid)

    # Sanity: detection works.
    assert _is_per_horizon(m)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "bundle.pkl"
        joblib.dump({"models": {"o1": m}}, path)
        agg = _agg_importance_from_bundle(path, quantile=0.5, kind="gain")

    # Manually compute the expected average across horizons.
    horizons = m.trained_horizons
    per_h = []
    for h in horizons:
        imp_h = m.feature_importance(horizon=h, quantile=0.5, kind="gain")
        per_h.append(imp_h.set_index("feature")["gain"])
    expected = pd.concat(per_h, axis=1).mean(axis=1)

    # Compare on the intersection of feature names.
    feature_index = expected.index
    got = agg.set_index("feature")["gain"].reindex(feature_index)
    pd.testing.assert_series_equal(got.rename(None), expected.rename(None), check_exact=False, rtol=1e-9)




def test_feature_importance_matches_real_runner_bundle_structure():
    """End-to-end with the EXACT bundle dict shape that
    run_per_horizon_experiment.py saves to disk. This is the regression
    test for "feature_importance_combined.csv = 0 rows on real artifacts".
    """
    import joblib
    import tempfile
    from pathlib import Path

    from seercast.diagnostics.feature_importance import feature_importance_combined

    _, sup = _supervised()

    # Build TWO origins worth of trained per-horizon wrappers, mirroring
    # the multi-origin bundle the runner writes.
    quantile_bundles = {}
    point_bundles = {}
    origins = sup["origin_date"].drop_duplicates().sort_values().iloc[::-1]
    for od in origins:
        split = split_for_backtest_origin(sup, od, valid_window_days=VALID_WINDOW_DAYS)
        if len(split.train) == 0 or len(split.valid) == 0:
            continue
        qm = PerHorizonQuantileLightGBMModel(
            n_estimators=20, early_stopping_rounds=5,
        ).fit(split.train, valid_df=split.valid)
        pm = PerHorizonLightGBMPointModel(
            n_estimators=20, early_stopping_rounds=5,
        ).fit(split.train, valid_df=split.valid)
        quantile_bundles[od.isoformat()] = qm
        point_bundles[od.isoformat()] = pm
        if len(quantile_bundles) >= 2:
            break

    assert len(quantile_bundles) >= 1, "need at least one trained origin for this test"

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Match run_per_horizon_experiment.run()'s exact bundle schemas.
        joblib.dump(
            {"models": quantile_bundles, "quantiles": [0.1, 0.5, 0.9],
             "horizons": [1, 7, 14, 28], "valid_window_days": 56,
             "backtest_origins": list(quantile_bundles.keys())},
            td / "per_horizon_quantile_models_ca1.pkl",
        )
        joblib.dump(
            {"models": point_bundles, "horizons": [1, 7, 14, 28],
             "valid_window_days": 56,
             "backtest_origins": list(point_bundles.keys())},
            td / "per_horizon_point_models_ca1.pkl",
        )

        fi = feature_importance_combined(
            point_bundle_path=td / "per_horizon_point_models_ca1.pkl",
            quantile_bundle_path=td / "per_horizon_quantile_models_ca1.pkl",
        )

    # The critical assertion: the loader must produce non-zero rows for
    # a real-shape per-horizon bundle. This is what was failing on
    # outputs/models/experiments/per_horizon/*.pkl.
    assert len(fi) > 0, "feature_importance_combined returned 0 rows on a real-shape per-horizon bundle"
    assert set(fi.columns) >= {"feature", "point_gain", "point_split", "quantile_p50_gain"}
    # All three importance columns should have some finite values.
    for col in ("point_gain", "point_split", "quantile_p50_gain"):
        assert fi[col].notna().sum() > 0, f"{col} is all-NaN despite a populated bundle"


def test_feature_importance_warns_on_empty_extraction():
    """Synthetic bundle with no working models: loader must emit a
    RuntimeWarning instead of silently returning empty.
    """
    import joblib
    import tempfile
    import warnings
    from pathlib import Path

    from seercast.diagnostics.feature_importance import feature_importance_combined

    # Build a sham wrapper that LOOKS per-horizon (has trained_horizons)
    # but whose feature_importance always returns an empty frame.
    sham = _ShamEmptyPerHorizon()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        joblib.dump({"models": {"o1": sham}}, td / "p.pkl")
        joblib.dump({"models": {"o1": sham}}, td / "q.pkl")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fi = feature_importance_combined(
                point_bundle_path=td / "p.pkl",
                quantile_bundle_path=td / "q.pkl",
            )

    # No rows produced, AND the user got a clear warning explaining why.
    assert len(fi) == 0
    messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("produced zero importance rows" in m or "every model in the bundle" in m
               for m in messages), \
        f"expected a RuntimeWarning explaining the empty extraction; got: {messages}"


def test_feature_importance_warns_when_bundle_has_no_models_key():
    """If a bundle is malformed (missing 'models' key), the loader names
    the path and the keys it actually saw."""
    import joblib
    import tempfile
    import warnings
    from pathlib import Path

    from seercast.diagnostics.feature_importance import feature_importance_combined

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bad = td / "broken.pkl"
        joblib.dump({"horizons": [1, 7, 14, 28], "info": "no models here"}, bad)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fi = feature_importance_combined(quantile_bundle_path=bad)

    assert len(fi) == 0
    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("no 'models' key" in m for m in msgs), msgs


if __name__ == "__main__":
    test_per_horizon_quantile_fits_per_horizon_sub_models()
    test_per_horizon_quantile_predictions_cover_full_grid_and_are_monotonic()
    test_per_horizon_quantile_no_leakage_in_sub_model_train_rows()
    test_per_horizon_quantile_each_sub_model_saw_only_its_horizon()
    test_per_horizon_excludes_horizon_from_feature_columns()
    test_per_horizon_point_fits_predicts_nonnegative()
    test_build_three_way_comparison_covers_all_versions_equal_n()
    test_build_three_way_comparison_handles_missing_inputs()
    test_feature_importance_combined_handles_per_horizon_bundle()
    test_feature_importance_combined_normal_bundle_still_works()
    test_per_horizon_importance_aggregates_across_horizons()
    test_feature_importance_matches_real_runner_bundle_structure()
    test_feature_importance_warns_on_empty_extraction()
    test_feature_importance_warns_when_bundle_has_no_models_key()
    print("Per-horizon smoke tests: OK")
