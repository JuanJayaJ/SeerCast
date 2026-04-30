"""Phase 2 smoke test using a synthetic mini-M5 frame.

Goal: prove that melt_sales then calendar join then price join then
validation all wire up correctly, without needing the actual ~58M-row M5
CSVs.

Run with::

    python -m pytest tests/test_phase2_smoke.py -v

or just::

    python tests/test_phase2_smoke.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from seercast.config import BASE_TABLE_COLUMNS
from seercast.data.load_m5 import M5Raw
from seercast.data.transform import build_base_table, melt_sales
from seercast.data.validation import validate_base_table


def _synthetic_raw(n_days: int = 14, n_items: int = 3) -> M5Raw:
    """Build a tiny but structurally faithful M5Raw bundle.

    Two stores (CA_1, CA_2), three items, 14 days. Enough to exercise the
    melt + calendar join + price join + validation pipeline.
    """
    rng = np.random.default_rng(0)

    # ----- calendar ------------------------------------------------------ #
    dates = pd.date_range("2015-01-01", periods=n_days, freq="D")
    cal = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": [11501 + (i // 7) for i in range(n_days)],
            "weekday": dates.day_name(),
            "wday": dates.weekday + 1,
            "month": dates.month,
            "year": dates.year,
            "d": [f"d_{i + 1}" for i in range(n_days)],
            "event_name_1": [None] * n_days,
            "event_type_1": [None] * n_days,
            "event_name_2": [None] * n_days,
            "event_type_2": [None] * n_days,
            "snap_CA": [1 if i % 4 == 0 else 0 for i in range(n_days)],
            "snap_TX": [0] * n_days,
            "snap_WI": [0] * n_days,
        }
    )

    # ----- sales (wide) -------------------------------------------------- #
    rows = []
    for store_id in ["CA_1", "CA_2"]:
        for k in range(n_items):
            row = {
                "id": f"FOODS_1_{k + 1:03d}_{store_id}_validation",
                "item_id": f"FOODS_1_{k + 1:03d}",
                "dept_id": "FOODS_1",
                "cat_id": "FOODS",
                "store_id": store_id,
                "state_id": store_id.split("_")[0],
            }
            for i in range(n_days):
                row[f"d_{i + 1}"] = int(rng.integers(0, 5))
            rows.append(row)
    sales = pd.DataFrame(rows)

    # ----- prices: cover all (store, item, week) combos ----------------- #
    prices = []
    weeks = sorted(set(cal["wm_yr_wk"]))
    for store_id in ["CA_1", "CA_2"]:
        for k in range(n_items):
            for w in weeks:
                prices.append(
                    {
                        "store_id": store_id,
                        "item_id": f"FOODS_1_{k + 1:03d}",
                        "wm_yr_wk": w,
                        "sell_price": float(round(1.0 + rng.random() * 4.0, 2)),
                    }
                )
    prices_df = pd.DataFrame(prices)

    # Match the real loader: parse 'date' as datetime64.
    cal["date"] = pd.to_datetime(cal["date"])
    return M5Raw(calendar=cal, sales=sales, prices=prices_df)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_melt_sales_shape_and_dtypes():
    raw = _synthetic_raw()
    long = melt_sales(raw.sales, store_ids=["CA_1"])

    # 3 items by 14 days = 42 rows for CA_1.
    assert len(long) == 3 * 14
    assert set(long.columns) == {
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "d",
        "sales",
    }
    assert long["sales"].dtype == np.int32
    assert long["store_id"].unique().tolist() == ["CA_1"]


def test_build_base_table_full_schema_and_join():
    raw = _synthetic_raw()
    base = build_base_table(raw=raw, store_ids=("CA_1",))
    assert list(base.columns) == BASE_TABLE_COLUMNS
    assert base["date"].isna().sum() == 0
    assert base["sell_price"].isna().sum() == 0
    assert not base.duplicated(["id", "date"]).any()
    assert (base["store_id"] == "CA_1").all()


def test_validate_base_table_ok_path():
    raw = _synthetic_raw()
    base = build_base_table(raw=raw, store_ids=("CA_1",))
    report = validate_base_table(base)
    assert report.ok, report.summary()
    assert report.duplicate_id_date_count == 0
    assert report.negative_sales_count == 0
    assert report.missing_date_count == 0


def test_validate_base_table_catches_negative_sales():
    raw = _synthetic_raw()
    base = build_base_table(raw=raw, store_ids=("CA_1",)).copy()
    base.loc[0, "sales"] = -1
    report = validate_base_table(base)
    assert not report.ok
    assert any("negative" in e for e in report.errors)


def test_validate_base_table_catches_duplicates():
    raw = _synthetic_raw()
    base = build_base_table(raw=raw, store_ids=("CA_1",))
    dup = pd.concat([base, base.head(1)], ignore_index=True)
    report = validate_base_table(dup)
    assert not report.ok
    assert any("duplicate" in e for e in report.errors)


def test_left_join_keeps_sales_when_calendar_day_missing():
    """If a d_x is in sales but missing from calendar, the LEFT join must
    preserve the sales rows with NaT date, and validate_base_table must
    flag them. An INNER join would silently drop them, which is the bug
    we are guarding against.
    """
    raw = _synthetic_raw(n_days=14, n_items=3)

    # Drop d_5 from calendar.
    raw.calendar.drop(
        raw.calendar.index[raw.calendar["d"] == "d_5"], inplace=True
    )

    base = build_base_table(raw=raw, store_ids=("CA_1",))

    assert len(base) == 3 * 14, (
        f"left join should keep sales rows even when calendar is missing d_5; "
        f"got {len(base)} rows, expected 42"
    )

    report = validate_base_table(base)
    assert not report.ok, "validation should fail when calendar is missing a day"
    assert report.missing_date_count == 3, (
        f"expected 3 NaT-date rows (one per item on d_5); "
        f"got {report.missing_date_count}"
    )
    assert any("NaT" in e or "missing" in e.lower() for e in report.errors), (
        f"expected an error mentioning missing/NaT date; got {report.errors}"
    )


if __name__ == "__main__":
    test_melt_sales_shape_and_dtypes()
    test_build_base_table_full_schema_and_join()
    test_validate_base_table_ok_path()
    test_validate_base_table_catches_negative_sales()
    test_validate_base_table_catches_duplicates()
    test_left_join_keeps_sales_when_calendar_day_missing()
    print("Phase 2 smoke tests: OK")
