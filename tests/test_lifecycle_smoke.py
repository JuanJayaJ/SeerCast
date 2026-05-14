"""Lifecycle / price-availability features smoke tests.

Covers:

1. The 9 new lifecycle columns appear in the supervised table (+ ``target_has_price``).
2. Origin-side sales-based lifecycle features are LEAKAGE-SAFE: mutating
   sales strictly after the row's date must not change the row's
   ``days_since_first_sale``, ``days_since_last_sale``, or
   ``pre_first_sale_flag``. This is the same shift(1) contract as the
   demand features.
3. Price-based lifecycle features use today's price (it's known at
   origin time -- same convention as ``sell_price`` itself).
4. Delayed-launch math on a hand-built series: item that starts selling
   on day 50 of 100 should have, at row t=60:
       pre_first_sale_flag = 0
       is_active_after_first_sale = 1
       days_since_first_sale = 60 - 50 = 10
       days_since_last_sale = small positive (depends on most recent sale)
5. ``has_price`` correctly tracks the row's own price availability;
   ``target_has_price`` reflects the target date.
6. The patched supervised table passes ``validate_supervised_table(strict=True)``.
7. The before-vs-after comparison helper enforces equal n per (model, version).

Run::

    python tests/test_lifecycle_smoke.py
"""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_phase4_smoke import _synthetic_base  # noqa: E402

from seercast.features.lifecycle_features import (
    LIFECYCLE_FEATURE_NAMES,
    add_lifecycle_features,
)
from seercast.features.supervised import (
    SUPERVISED_COLUMNS,
    build_supervised_table,
    validate_supervised_table,
)
from seercast.training.run_lifecycle_experiment import build_before_vs_after


# --------------------------------------------------------------------------- #
# 1. Schema: columns present in the supervised table
# --------------------------------------------------------------------------- #


def test_lifecycle_columns_present_in_supervised_schema():
    """All 9 lifecycle columns + target_has_price must be in SUPERVISED_COLUMNS."""
    for col in LIFECYCLE_FEATURE_NAMES:
        assert col in SUPERVISED_COLUMNS, f"missing origin col {col}"
    assert "target_has_price" in SUPERVISED_COLUMNS


def test_lifecycle_columns_populated_in_built_supervised_table():
    """Build a real (synthetic) supervised table and confirm the new cols
    appear with non-NaN values for at least some rows."""
    base = _synthetic_base(n_days=120, n_items=2)
    sup = build_supervised_table(
        base, horizons=[1, 7, 14, 28],
        snap_state="CA", origin_step_days=14,
    )
    for col in LIFECYCLE_FEATURE_NAMES:
        assert col in sup.columns, col
    assert "target_has_price" in sup.columns
    # has_price flags shouldn't all be 0 (synthetic data has prices everywhere).
    assert int(sup["has_price"].sum()) > 0
    assert int(sup["target_has_price"].sum()) > 0


# --------------------------------------------------------------------------- #
# 2. Leakage contract: sales features at row t must not see sales >= t
# --------------------------------------------------------------------------- #


def test_lifecycle_sales_features_are_leakage_safe():
    """Poison every sales value AFTER a chosen anchor date for one id and
    refit lifecycle features. The lifecycle columns at rows BEFORE the
    anchor must be byte-identical.

    Specifically tests: days_since_first_sale, days_since_last_sale,
    pre_first_sale_flag, is_active_after_first_sale.
    """
    base = _synthetic_base(n_days=150, n_items=2)

    enriched1 = base.copy()
    add_lifecycle_features(enriched1)

    poisoned = base.copy()
    target_id = poisoned["id"].iloc[0]
    anchor_idx = 80
    anchor_date = enriched1.loc[
        (enriched1["id"] == target_id) & (
            enriched1["date"] == enriched1[enriched1["id"] == target_id]["date"].iloc[anchor_idx]
        )
    ]["date"].iloc[0]
    mask = (poisoned["id"] == target_id) & (poisoned["date"] > anchor_date)
    poisoned.loc[mask, "sales"] = 9999

    enriched2 = poisoned.copy()
    add_lifecycle_features(enriched2)

    before_mask = (enriched1["id"] == target_id) & (enriched1["date"] <= anchor_date)
    for col in ("days_since_first_sale", "days_since_last_sale",
                "pre_first_sale_flag", "is_active_after_first_sale"):
        a = enriched1.loc[before_mask, col].to_numpy()
        b = enriched2.loc[before_mask, col].to_numpy()
        # Allow NaN equality.
        same = pd.Series(a).equals(pd.Series(b))
        assert same, (
            f"{col} changed at rows BEFORE the anchor when post-anchor sales were poisoned"
        )


def test_lifecycle_price_features_include_today():
    """If today is the FIRST day with sell_price, has_price=1 and
    days_since_first_price=0 (today is the first price day)."""
    dates = pd.date_range("2014-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "id": "ITEM_X",
        "date": dates,
        "sales": [0] * 10,
        "sell_price": [np.nan, np.nan, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
    })
    add_lifecycle_features(df)
    row2 = df.iloc[2]
    assert row2["has_price"] == 1
    assert float(row2["days_since_first_price"]) == 0.0
    assert float(row2["days_since_last_price"]) == 0.0
    assert row2["pre_first_price_flag"] == 0


# --------------------------------------------------------------------------- #
# 3. Delayed-launch hand-checked math
# --------------------------------------------------------------------------- #


def test_lifecycle_delayed_launch_math():
    """Item starts selling on day 50 of 100. Verify per-row state."""
    dates = pd.date_range("2014-01-01", periods=100, freq="D")
    sales = [0] * 50 + [1] * 50
    df = pd.DataFrame({
        "id": "ITEM_X",
        "date": dates,
        "sales": sales,
        "sell_price": [np.nan] * 50 + [2.5] * 50,
    })
    add_lifecycle_features(df)

    # Row 30 (pre-launch): no first sale yet.
    row30 = df.iloc[30]
    assert row30["pre_first_sale_flag"] == 1
    assert row30["is_active_after_first_sale"] == 0
    assert pd.isna(row30["days_since_first_sale"])

    # Row 50 (first sale day): shift(1) means first_sale at THIS row is still NaT
    # because we haven't *recorded* yesterday's sale. days_since_first_sale = NaN.
    row50 = df.iloc[50]
    assert row50["pre_first_sale_flag"] == 1  # shift(1): yesterday had no sale
    assert pd.isna(row50["days_since_first_sale"])

    # Row 51 (one day after first sale): shift(1) puts first_sale at day 50.
    # days_since_first_sale = day 51 - day 50 = 1.
    row51 = df.iloc[51]
    assert row51["pre_first_sale_flag"] == 0
    assert row51["is_active_after_first_sale"] == 1
    assert float(row51["days_since_first_sale"]) == 1.0
    assert float(row51["days_since_last_sale"]) == 1.0   # day 50 is the last sale before 51

    # Row 60: 10 days after first sale.
    row60 = df.iloc[60]
    assert float(row60["days_since_first_sale"]) == 10.0
    assert float(row60["days_since_last_sale"]) == 1.0  # latest sale was yesterday


def test_lifecycle_never_selling_item_stays_pre_launch():
    dates = pd.date_range("2014-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "id": "ITEM_Y",
        "date": dates,
        "sales": [0] * 30,
        "sell_price": [np.nan] * 30,
    })
    add_lifecycle_features(df)
    assert (df["pre_first_sale_flag"] == 1).all()
    assert (df["is_active_after_first_sale"] == 0).all()
    assert df["days_since_first_sale"].isna().all()
    assert df["days_since_last_sale"].isna().all()


# --------------------------------------------------------------------------- #
# 4. Supervised table passes strict validation
# --------------------------------------------------------------------------- #


def test_supervised_table_with_lifecycle_passes_strict_validation():
    base = _synthetic_base(n_days=200, n_items=2)
    sup = build_supervised_table(
        base, horizons=[1, 7, 14, 28],
        snap_state="CA", origin_step_days=14,
    )
    report = validate_supervised_table(sup, strict=True)
    assert report.ok, report.summary()
    # And the schema order matches SUPERVISED_COLUMNS exactly.
    assert list(sup.columns) == list(SUPERVISED_COLUMNS)


# --------------------------------------------------------------------------- #
# 5. before_vs_after helper enforces equal n per (model, version)
# --------------------------------------------------------------------------- #


def _fake_predictions(version_label, n_ids=2, horizons=(1, 7), include_quantile=True):
    """Build the minimal LGBM + quantile prediction frames for one version."""
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
                "prediction": actual + (0.5 if version_label == "lifecycle_features" else 1.0),
                "actual": actual,
            })
            rows_q.append({
                "origin_date": origin, "id": id_, "horizon": h,
                "target_date": target,
                "p10": actual - 1,
                "p50": actual + (0.2 if version_label == "lifecycle_features" else 0.6),
                "p90": actual + 2,
                "actual": actual,
            })
    pt_df = pd.DataFrame(rows_pt)
    q_df = pd.DataFrame(rows_q) if include_quantile else None
    return pt_df, q_df


def test_before_vs_after_equal_n_across_versions_and_models():
    base_pt, base_q = _fake_predictions("baseline_features")
    exp_pt, exp_q = _fake_predictions("lifecycle_features")
    out = build_before_vs_after(
        baseline_lgbm_predictions=base_pt,
        baseline_quantile_predictions=base_q,
        experiment_lgbm_predictions=exp_pt,
        experiment_quantile_predictions=exp_q,
    )
    # Two models x two versions = 4 rows.
    assert set(zip(out["model"], out["version"])) == {
        ("lightgbm_point", "baseline_features"),
        ("lightgbm_point", "lifecycle_features"),
        ("lightgbm_quantile_p50", "baseline_features"),
        ("lightgbm_quantile_p50", "lifecycle_features"),
    }
    # All rows must have the same n (matched grid size).
    assert out["n"].nunique() == 1, out
    # Sort order is (model, WAPE asc).
    for model, group in out.groupby("model"):
        wapes = group["WAPE"].tolist()
        assert wapes == sorted(wapes), f"WAPE not sorted asc for model {model}: {wapes}"


def test_before_vs_after_handles_missing_quantile_predictions():
    """If quantile preds are missing for the baseline, the comparison still
    works -- it just omits the quantile p50 row for that version."""
    base_pt, _ = _fake_predictions("baseline_features", include_quantile=False)
    exp_pt, exp_q = _fake_predictions("lifecycle_features")
    out = build_before_vs_after(
        baseline_lgbm_predictions=base_pt,
        baseline_quantile_predictions=None,    # missing
        experiment_lgbm_predictions=exp_pt,
        experiment_quantile_predictions=exp_q,
    )
    versions_per_model = out.groupby("model")["version"].apply(set).to_dict()
    # Quantile p50 only present for the experiment version.
    assert versions_per_model["lightgbm_quantile_p50"] == {"lifecycle_features"}
    # Point model has both versions.
    assert versions_per_model["lightgbm_point"] == {"baseline_features", "lifecycle_features"}


if __name__ == "__main__":
    test_lifecycle_columns_present_in_supervised_schema()
    test_lifecycle_columns_populated_in_built_supervised_table()
    test_lifecycle_sales_features_are_leakage_safe()
    test_lifecycle_price_features_include_today()
    test_lifecycle_delayed_launch_math()
    test_lifecycle_never_selling_item_stays_pre_launch()
    test_supervised_table_with_lifecycle_passes_strict_validation()
    test_before_vs_after_equal_n_across_versions_and_models()
    test_before_vs_after_handles_missing_quantile_predictions()
    print("Lifecycle smoke tests: OK")
