"""Phase 4 smoke tests: features + supervised-table assembly + validator.

Goals:

1. Calendar / demand / price feature builders emit the right columns and
   correct values on a small synthetic frame.
2. Demand rolling features are LEAKAGE-SAFE: the value at row ``t`` only
   uses sales at ``date <= t-1``. (The most important test in Phase 4.)
3. ``build_supervised_table`` produces the canonical schema in canonical
   order, with target_date == origin_date + horizon and unique (id,
   origin_date, horizon).
4. ``validate_supervised_table`` catches corrupted tables (bad horizon,
   duplicates, missing target_sales).

Run::

    python tests/test_phase4_smoke.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seercast.features.calendar_features import (
    CALENDAR_FEATURE_NAMES,
    add_calendar_features,
)
from seercast.features.demand_features import (
    DEMAND_FEATURE_NAMES,
    add_demand_features,
)
from seercast.features.price_features import (
    PRICE_FEATURE_NAMES,
    add_price_features,
)
from seercast.features.supervised import (
    ORIGIN_FEATURE_COLUMNS,
    SUPERVISED_COLUMNS,
    TARGET_FEATURE_COLUMNS,
    build_supervised_table,
    default_training_origins,
    validate_supervised_table,
)


# --------------------------------------------------------------------------- #
# Synthetic base table
# --------------------------------------------------------------------------- #


def _synthetic_base(n_days: int = 120, n_items: int = 3, seed: int = 0) -> pd.DataFrame:
    """Build a structurally-faithful CA_1 base table.

    Columns match what build_base_table emits: id, item_id, dept_id, cat_id,
    store_id, state_id, d, date, sales, sell_price, event_name_1/2,
    event_type_1/2, snap_CA/TX/WI, weekday, wday, month, year, wm_yr_wk.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2014-01-06", periods=n_days, freq="D")  # Monday start
    weekly = np.array([1, 2, 3, 4, 5, 8, 6], dtype=float)

    rows = []
    for k in range(n_items):
        item_id = f"FOODS_1_{k + 1:03d}"
        scale = 1.0 + k
        for j, dt in enumerate(dates):
            wd = dt.weekday()
            sales = max(0, int(round(weekly[wd] * scale + rng.normal(0, 0.3))))
            rows.append(
                {
                    "id": f"{item_id}_CA_1_validation",
                    "item_id": item_id,
                    "dept_id": "FOODS_1",
                    "cat_id": "FOODS",
                    "store_id": "CA_1",
                    "state_id": "CA",
                    "d": f"d_{j + 1}",
                    "date": dt,
                    "sales": sales,
                    "sell_price": float(round(2.0 + 0.5 * k + rng.normal(0, 0.05), 2)),
                    "weekday": dt.day_name(),
                    "wday": wd + 1,
                    "month": dt.month,
                    "year": dt.year,
                    "wm_yr_wk": 11500 + (j // 7),
                    "event_name_1": "Easter" if j == 50 else None,
                    "event_type_1": "Cultural" if j == 50 else None,
                    "event_name_2": None,
                    "event_type_2": None,
                    "snap_CA": 1 if j % 4 == 0 else 0,
                    "snap_TX": 0,
                    "snap_WI": 0,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Calendar features
# --------------------------------------------------------------------------- #


def test_calendar_features_unprefixed_and_prefixed():
    base = _synthetic_base(n_days=14, n_items=1)

    unprefixed = add_calendar_features(base.copy(), prefix="", snap_state="CA")
    for col in CALENDAR_FEATURE_NAMES:
        assert col in unprefixed.columns, col

    # Spot-check: 2014-01-06 is a Monday => dayofweek=0, is_weekend=0.
    monday = unprefixed.iloc[0]
    assert monday["dayofweek"] == 0
    assert monday["is_weekend"] == 0
    # 2014-01-12 is a Sunday => dayofweek=6, is_weekend=1.
    sunday = unprefixed.iloc[6]
    assert sunday["dayofweek"] == 6
    assert sunday["is_weekend"] == 1

    # is_snap_day mirrors snap_CA.
    assert (unprefixed["is_snap_day"] == unprefixed["snap_CA"]).all()

    # Prefix mode.
    prefixed = add_calendar_features(base.copy(), prefix="target_", snap_state="CA")
    for col in CALENDAR_FEATURE_NAMES:
        assert f"target_{col}" in prefixed.columns


# --------------------------------------------------------------------------- #
# Demand features: shape + leakage contract
# --------------------------------------------------------------------------- #


def test_demand_features_shape_and_lag_values():
    base = _synthetic_base(n_days=120, n_items=2)
    out = add_demand_features(base.copy())
    for col in DEMAND_FEATURE_NAMES:
        assert col in out.columns, col

    # Pick a single id and verify lag_1 / lag_7 against the underlying sales series.
    one_id = out["id"].iloc[0]
    series = (
        out.loc[out["id"] == one_id, ["date", "sales", "sales_lag_1", "sales_lag_7"]]
        .sort_values("date")
        .reset_index(drop=True)
    )
    # lag_1[t] == sales[t-1]
    assert pd.isna(series["sales_lag_1"].iloc[0])
    assert series["sales_lag_1"].iloc[1] == float(series["sales"].iloc[0])
    # lag_7[t] == sales[t-7]
    assert series["sales_lag_7"].iloc[7] == float(series["sales"].iloc[0])
    # The first 7 lag_7 values are NaN.
    assert series["sales_lag_7"].iloc[:7].isna().all()


def test_demand_rolling_uses_shifted_sales_no_leakage():
    """The leakage contract: rolling_mean_7 at row t must equal mean of
    sales[t-7..t-1], NOT sales[t-6..t]. The test mutates sales(t) and
    confirms rolling_mean_7 at row t doesn't change.
    """
    base = _synthetic_base(n_days=120, n_items=1)
    out1 = add_demand_features(base.copy())

    # Pick a row well past the warmup so rolling features are populated.
    one_id = out1["id"].iloc[0]
    by_id = out1.loc[out1["id"] == one_id].sort_values("date").reset_index(drop=True)
    t_idx = 50
    rolling_mean_7_at_t = by_id["rolling_mean_7"].iloc[t_idx]
    rolling_mean_28_at_t = by_id["rolling_mean_28"].iloc[t_idx]

    # Mutate sales at row t and t+1, t+2 (poison the future).
    poisoned = base.copy().sort_values(["id", "date"]).reset_index(drop=True)
    mask = (poisoned["id"] == one_id)
    indices = poisoned.loc[mask].index
    poisoned.loc[indices[t_idx], "sales"] *= 100
    poisoned.loc[indices[t_idx + 1], "sales"] *= 100
    poisoned.loc[indices[t_idx + 2], "sales"] *= 100

    out2 = add_demand_features(poisoned)
    by_id2 = out2.loc[out2["id"] == one_id].sort_values("date").reset_index(drop=True)
    assert by_id2["rolling_mean_7"].iloc[t_idx] == rolling_mean_7_at_t, (
        "rolling_mean_7 at row t LEAKED future sales"
    )
    assert by_id2["rolling_mean_28"].iloc[t_idx] == rolling_mean_28_at_t, (
        "rolling_mean_28 at row t LEAKED future sales"
    )

    # But rolling_mean at row t+1 should change because sales(t) is now in window.
    assert by_id2["rolling_mean_7"].iloc[t_idx + 1] != by_id["rolling_mean_7"].iloc[t_idx + 1], (
        "rolling_mean_7 at t+1 should reflect mutated sales(t)"
    )


# --------------------------------------------------------------------------- #
# Price features
# --------------------------------------------------------------------------- #


def test_price_features_columns_and_relative():
    base = _synthetic_base(n_days=60, n_items=1)
    out = add_price_features(base.copy().sort_values(["id", "date"]).reset_index(drop=True))
    for col in PRICE_FEATURE_NAMES:
        assert col in out.columns, col
    # price_relative_to_28d_avg should be ~1 for stable prices.
    rel = out["price_relative_to_28d_avg"].dropna()
    assert (abs(rel - 1.0) < 0.5).mean() > 0.9, (
        "with near-stable prices, price_relative_to_28d_avg should hover near 1"
    )


# --------------------------------------------------------------------------- #
# Supervised assembly
# --------------------------------------------------------------------------- #


def test_default_training_origins_respects_history_and_future():
    base = _synthetic_base(n_days=120, n_items=1)
    origins = default_training_origins(
        base, min_history_days=56, max_horizon=28, step_days=7
    )
    assert len(origins) > 0
    earliest_required = base["date"].min() + pd.Timedelta(days=56)
    latest_allowed = base["date"].max() - pd.Timedelta(days=28)
    for o in origins:
        assert o >= earliest_required
        assert o <= latest_allowed


def test_supervised_table_full_schema_and_horizon_invariant():
    base = _synthetic_base(n_days=120, n_items=2)
    sup = build_supervised_table(
        base,
        horizons=[1, 7, 14, 28],
        snap_state="CA",
        origin_step_days=7,
    )

    # Schema in canonical order.
    assert list(sup.columns) == list(SUPERVISED_COLUMNS), (
        f"unexpected schema:\n  got: {list(sup.columns)}\n  expected: {list(SUPERVISED_COLUMNS)}"
    )

    # target_date == origin_date + horizon (per row, vectorized).
    delta = (sup["target_date"] - sup["origin_date"]).dt.days
    assert (delta == sup["horizon"]).all()

    # target_date strictly after origin_date.
    assert (sup["target_date"] > sup["origin_date"]).all()

    # No duplicates.
    assert not sup.duplicated(["id", "origin_date", "horizon"]).any()

    # target_sales never NaN (inner merge dropped past-end rows).
    assert sup["target_sales"].notna().all()

    # Each (id, origin) yields exactly len(horizons) rows.
    counts = sup.groupby(["id", "origin_date"]).size().unique()
    assert (counts == 4).all(), f"expected 4 rows per (id, origin); got {counts}"


def test_supervised_table_full_horizon_28_rows_per_origin():
    base = _synthetic_base(n_days=200, n_items=1)
    sup = build_supervised_table(
        base,
        horizons=list(range(1, 29)),
        snap_state="CA",
        origin_step_days=14,
    )
    counts = sup.groupby(["id", "origin_date"]).size().unique()
    assert (counts == 28).all(), f"expected 28 rows per (id, origin); got {counts}"


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #


def test_validator_clean_table_ok():
    base = _synthetic_base(n_days=120, n_items=2)
    sup = build_supervised_table(base, horizons=[1, 7, 14, 28], origin_step_days=7)
    report = validate_supervised_table(sup)
    assert report.ok, report.summary()
    assert report.duplicate_keys_count == 0
    assert report.horizon_mismatch_count == 0
    assert report.missing_target_sales_count == 0


def test_validator_catches_bad_horizon_and_duplicates():
    base = _synthetic_base(n_days=120, n_items=1)
    sup = build_supervised_table(base, horizons=[1, 7], origin_step_days=14).copy()

    # Corrupt one row's target_date so the horizon invariant breaks.
    sup.loc[0, "target_date"] = sup.loc[0, "target_date"] + pd.Timedelta(days=10)

    # Duplicate one row.
    sup = pd.concat([sup, sup.head(1)], ignore_index=True)

    report = validate_supervised_table(sup)
    assert not report.ok
    assert report.horizon_mismatch_count >= 1
    assert report.duplicate_keys_count >= 2  # both originals counted


def test_validator_strict_raises():
    base = _synthetic_base(n_days=120, n_items=1)
    sup = build_supervised_table(base, horizons=[1], origin_step_days=14).copy()
    sup.loc[0, "target_sales"] = np.nan
    try:
        validate_supervised_table(sup, strict=True)
    except AssertionError as e:
        assert "missing target_sales" in str(e)
        return
    raise AssertionError("strict=True should have raised on missing target_sales")




def test_backtest_origins_are_guaranteed_in_supervised_table():
    """Patch contract: must_include_origins guarantees the listed backtest
    origins land in the supervised table even when the default weekly
    training origins skip past them.

    We pass an int (M5 d-style) and a date, and confirm both end up as
    origin_date values in the output.
    """
    base = _synthetic_base(n_days=200, n_items=2)

    # Pick two backtest origins:
    # (a) a d-integer that probably falls between weekly training origins,
    # (b) an explicit date.
    d_int = 95
    explicit_date = base["date"].iloc[140]

    sup = build_supervised_table(
        base,
        horizons=[1, 7, 14, 28],
        snap_state="CA",
        origin_step_days=14,  # sparse weekly origins so we can prove the union worked
        must_include_origins=[d_int, explicit_date],
    )

    # Convert d_95 to its corresponding date via the same helper used internally.
    from seercast.evaluation.backtesting import origin_to_date

    d_int_date = pd.Timestamp(origin_to_date(base, d_int))

    present = set(sup["origin_date"].unique())
    assert d_int_date in present, (
        f"d_{d_int} ({d_int_date.date()}) should appear in supervised origins "
        f"but did not; present sample: {sorted(present)[:5]}"
    )
    assert pd.Timestamp(explicit_date) in present, (
        f"explicit date {explicit_date} should appear in supervised origins"
    )

    # Sanity: each backtest origin still has all 4 horizons.
    for o in (d_int_date, pd.Timestamp(explicit_date)):
        n = (sup["origin_date"] == o).sum()
        # 2 ids * 4 horizons = 8 rows per origin
        assert n == 2 * 4, f"origin {o.date()} produced {n} rows, expected 8"


def test_must_include_origins_string_form():
    """origin_to_date should accept "d_N" strings as well."""
    base = _synthetic_base(n_days=200, n_items=1)
    sup = build_supervised_table(
        base,
        horizons=[1, 7],
        snap_state="CA",
        origin_step_days=21,
        must_include_origins=["d_120"],
    )
    from seercast.evaluation.backtesting import origin_to_date
    expected = pd.Timestamp(origin_to_date(base, "d_120"))
    assert expected in set(sup["origin_date"].unique())

if __name__ == "__main__":
    test_calendar_features_unprefixed_and_prefixed()
    test_demand_features_shape_and_lag_values()
    test_demand_rolling_uses_shifted_sales_no_leakage()
    test_price_features_columns_and_relative()
    test_default_training_origins_respects_history_and_future()
    test_supervised_table_full_schema_and_horizon_invariant()
    test_supervised_table_full_horizon_28_rows_per_origin()
    test_validator_clean_table_ok()
    test_validator_catches_bad_horizon_and_duplicates()
    test_validator_strict_raises()
    test_backtest_origins_are_guaranteed_in_supervised_table()
    test_must_include_origins_string_form()
    print("Phase 4 smoke tests: OK")
