"""Supervised-table assembly + validation.

Given a base table and a set of (origin_date, horizon) pairs, produce one
row per ``(id, origin_date, horizon)`` containing:

* identity columns
* historical demand features at origin (lag/rolling, ``shift(1)``-safe)
* price features at origin
* calendar / event / SNAP features at origin
* target-date known features (sell_price + calendar/event/SNAP at target)
* ``target_sales`` label

The leakage rule is enforced *upstream*: the demand and price feature
modules use a per-id ``shift(1)`` before any rolling stat, so a feature
read at ``origin_date == t`` only ever sees sales at ``date <= t-1``.
The supervised builder simply selects rows at ``date == origin_date``
and joins target-date columns from rows at ``date == target_date``.

The validator checks the ML contract:

* ``target_date == origin_date + horizon``
* ``target_date > origin_date``
* unique ``(id, origin_date, horizon)``
* no missing ``target_sales``
* per-feature missing-value rates (warning only, since LightGBM handles NaN)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from seercast.config import DEFAULT_STATE_ID, DIRECT_HORIZONS
from seercast.evaluation.backtesting import origin_to_date
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


# --------------------------------------------------------------------------- #
# Schema constants
# --------------------------------------------------------------------------- #

IDENTITY_COLUMNS: tuple[str, ...] = (
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
)

# Calendar columns we copy at the target (the spec uses a subset -- no
# target_has_event_1/has_event_2, just the unified target_has_event).
_TARGET_CALENDAR_SUBSET: tuple[str, ...] = (
    "dayofweek",
    "weekofyear",
    "month",
    "year",
    "quarter",
    "dayofmonth",
    "is_weekend",
    "has_event",
    "is_snap_day",
)

ORIGIN_FEATURE_COLUMNS: tuple[str, ...] = (
    *DEMAND_FEATURE_NAMES,
    "sell_price",
    *PRICE_FEATURE_NAMES,
    *CALENDAR_FEATURE_NAMES,
)

TARGET_FEATURE_COLUMNS: tuple[str, ...] = (
    "target_sell_price",
    *(f"target_{c}" for c in _TARGET_CALENDAR_SUBSET),
)

META_COLUMNS: tuple[str, ...] = (
    "origin_date",
    "horizon",
    "target_date",
    "target_sales",
)

# Final canonical column order.
SUPERVISED_COLUMNS: tuple[str, ...] = (
    *IDENTITY_COLUMNS,
    *META_COLUMNS,
    *ORIGIN_FEATURE_COLUMNS,
    *TARGET_FEATURE_COLUMNS,
)


# --------------------------------------------------------------------------- #
# Origin-date helpers
# --------------------------------------------------------------------------- #


def default_training_origins(
    base: pd.DataFrame,
    *,
    min_history_days: int = 56,
    max_horizon: int = 28,
    step_days: int = 7,
) -> list[pd.Timestamp]:
    """Pick a sensible default set of origin dates from the base table.

    Returns dates that have:

    * at least ``min_history_days`` of data before them (so the longest lag
      / rolling window has something to chew on), and
    * at least ``max_horizon`` days of data after them (so the longest
      horizon has a target value).

    ``step_days`` controls density: 7 (default) gives weekly origins,
    1 gives daily. The full M5 ``CA_1`` slice with 1913 days, 3049 ids,
    and weekly origins generates ~3.3M training rows for the 4-horizon
    table -- enough signal for LightGBM, manageable in memory.
    """
    if "date" not in base.columns:
        raise KeyError("base must have a 'date' column")
    dates = pd.Series(base["date"].dropna().unique()).sort_values().reset_index(drop=True)
    if dates.empty:
        return []
    first = dates.iloc[0] + pd.Timedelta(days=min_history_days)
    last = dates.iloc[-1] - pd.Timedelta(days=max_horizon)
    candidates = dates[(dates >= first) & (dates <= last)]
    if candidates.empty:
        return []
    if step_days <= 1:
        return [pd.Timestamp(d) for d in candidates]
    return [pd.Timestamp(d) for d in candidates.iloc[::step_days]]


# --------------------------------------------------------------------------- #
# Build the supervised table
# --------------------------------------------------------------------------- #


def _enrich_base(base: pd.DataFrame, snap_state: str) -> pd.DataFrame:
    """Apply demand + price + calendar feature builders in place, return the enriched frame.

    Operates on a copy so the caller's frame is untouched.
    """
    enriched = base.copy()
    add_demand_features(enriched, sort=True)
    add_price_features(enriched, sort=False)
    add_calendar_features(enriched, prefix="", snap_state=snap_state)
    return enriched


def _build_target_features(enriched: pd.DataFrame) -> pd.DataFrame:
    """Project the enriched base to a frame keyed by (id, target_date).

    The columns are renamed with the ``target_`` prefix (and ``sales`` ->
    ``target_sales``, ``sell_price`` -> ``target_sell_price``). This is
    the table we'll ``merge(..., how="inner")`` against to attach
    target-date known features and the label.
    """
    keep = ["id", "date", "sales", "sell_price", *_TARGET_CALENDAR_SUBSET]
    target = enriched.loc[:, keep].copy()
    rename = {
        "date": "target_date",
        "sales": "target_sales",
        "sell_price": "target_sell_price",
    }
    rename.update({c: f"target_{c}" for c in _TARGET_CALENDAR_SUBSET})
    target = target.rename(columns=rename)
    return target


def build_supervised_table(
    base: pd.DataFrame,
    *,
    horizons: Sequence[int] = DIRECT_HORIZONS,
    origin_dates: Sequence[pd.Timestamp] | None = None,
    must_include_origins: Sequence[int | str | pd.Timestamp] | None = None,
    snap_state: str = DEFAULT_STATE_ID,
    min_history_days: int = 56,
    origin_step_days: int = 7,
) -> pd.DataFrame:
    """Build the leakage-safe supervised training table.

    Parameters
    ----------
    base
        Long base table (output of ``build_base_table``).
    horizons
        Step-ahead integers. ``DIRECT_HORIZONS = [1, 7, 14, 28]`` for the
        initial table; ``range(1, 29)`` for the full-horizon scenario table.
    origin_dates
        Explicit list of origin dates. If ``None``, calls
        :func:`default_training_origins` with ``min_history_days``,
        ``max_horizon=max(horizons)``, ``step_days=origin_step_days``.
    must_include_origins
        Backtest origins that MUST be present in the supervised table even
        when ``origin_dates`` was generated by ``default_training_origins``
        (which only emits weekly samples by default). Each entry can be an
        M5 ``d`` integer (``1500``), a ``"d_N"`` string, or a date-like.
        Coerced via :func:`seercast.evaluation.backtesting.origin_to_date`,
        unioned with ``origin_dates``, deduped, and sorted. This is what
        guarantees the Phase 5 LightGBM backtest evaluates on the same
        dates as the Phase 3 baselines.
    snap_state
        Two-letter state for the SNAP flag, default ``"CA"`` to match CA_1.
    min_history_days, origin_step_days
        Forwarded to :func:`default_training_origins` when
        ``origin_dates`` is None.

    Returns
    -------
    pandas.DataFrame
        Columns are :data:`SUPERVISED_COLUMNS` in canonical order. One row
        per ``(id, origin_date, horizon)``. ``target_sales`` is never NaN
        because the inner merge against target-date features drops rows
        whose target is past the end of the base table.
    """
    if not horizons:
        raise ValueError("horizons must be a non-empty sequence")
    horizons = sorted(set(int(h) for h in horizons))
    max_h = horizons[-1]

    enriched = _enrich_base(base, snap_state=snap_state)
    target_features = _build_target_features(enriched)

    if origin_dates is None:
        origin_dates = default_training_origins(
            enriched,
            min_history_days=min_history_days,
            max_horizon=max_h,
            step_days=origin_step_days,
        )
    origin_dates = [pd.Timestamp(d) for d in origin_dates]

    # Guarantee that any backtest origins are in the table -- otherwise the
    # Phase 5 ML backtest can't evaluate on the same days as the Phase 3
    # baselines. Backtest origins may be M5 d-integers / "d_N" strings;
    # origin_to_date does the conversion using the base table itself.
    if must_include_origins:
        coerced = [
            pd.Timestamp(origin_to_date(base, o)) for o in must_include_origins
        ]
        origin_dates = sorted(set(origin_dates) | set(coerced))
    else:
        origin_dates = sorted(set(origin_dates))

    if not origin_dates:
        return pd.DataFrame(columns=list(SUPERVISED_COLUMNS))

    # Slice origin rows once; reuse for every horizon.
    origin_rows = enriched.loc[enriched["date"].isin(origin_dates)].copy()
    origin_rows = origin_rows.rename(columns={"date": "origin_date"})

    keep_origin_cols = [
        *IDENTITY_COLUMNS,
        "origin_date",
        *ORIGIN_FEATURE_COLUMNS,
    ]
    origin_rows = origin_rows.loc[:, keep_origin_cols]

    chunks: list[pd.DataFrame] = []
    for h in horizons:
        chunk = origin_rows.copy()
        chunk["horizon"] = np.int16(h)
        chunk["target_date"] = chunk["origin_date"] + pd.Timedelta(days=h)
        merged = chunk.merge(
            target_features,
            on=["id", "target_date"],
            how="inner",
            validate="one_to_one",
        )
        chunks.append(merged)

    out = pd.concat(chunks, ignore_index=True)

    # Reorder to canonical schema.
    out = out.reindex(columns=list(SUPERVISED_COLUMNS))

    # Stable sort for deterministic downstream behavior.
    out = out.sort_values(["origin_date", "horizon", "id"]).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #


@dataclass
class SupervisedValidationReport:
    """Structured result of validating a supervised training table."""

    n_rows: int
    n_unique_ids: int
    n_origins: int
    n_horizons: int
    duplicate_keys_count: int
    horizon_mismatch_count: int
    nonpositive_horizon_count: int
    missing_target_sales_count: int
    missing_feature_rates: dict[str, float] = field(default_factory=dict)
    schema_missing_columns: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        worst = sorted(
            self.missing_feature_rates.items(), key=lambda kv: -kv[1]
        )[:5]
        worst_str = ", ".join(f"{k}={v:.2%}" for k, v in worst) or "none"
        lines = [
            f"rows                              : {self.n_rows:,}",
            f"unique ids                        : {self.n_unique_ids:,}",
            f"unique origin_date                : {self.n_origins:,}",
            f"unique horizon                    : {self.n_horizons:,}",
            f"duplicate (id,origin,horizon)     : {self.duplicate_keys_count:,}",
            f"target_date != origin + horizon   : {self.horizon_mismatch_count:,}",
            f"horizon <= 0                      : {self.nonpositive_horizon_count:,}",
            f"missing target_sales              : {self.missing_target_sales_count:,}",
            f"top-5 missing feature rates       : {worst_str}",
            f"schema missing                    : {self.schema_missing_columns or 'none'}",
            f"errors                            : {len(self.errors)}",
            f"warnings                          : {len(self.warnings)}",
        ]
        if self.errors:
            lines.append("")
            lines.append("ERRORS:")
            lines.extend(f"  - {e}" for e in self.errors)
        if self.warnings:
            lines.append("")
            lines.append("WARNINGS:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_rows": int(self.n_rows),
            "n_unique_ids": int(self.n_unique_ids),
            "n_origins": int(self.n_origins),
            "n_horizons": int(self.n_horizons),
            "duplicate_keys_count": int(self.duplicate_keys_count),
            "horizon_mismatch_count": int(self.horizon_mismatch_count),
            "nonpositive_horizon_count": int(self.nonpositive_horizon_count),
            "missing_target_sales_count": int(self.missing_target_sales_count),
            "missing_feature_rates": {k: float(v) for k, v in self.missing_feature_rates.items()},
            "schema_missing_columns": list(self.schema_missing_columns),
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


# Features whose missingness is expected and acceptable (warmup / sparse prices).
_HIGH_MISS_TOLERANCE: dict[str, float] = {
    "sales_lag_56": 0.20,
    "rolling_mean_56": 0.10,
    "rolling_std_28": 0.05,
    "rolling_std_7": 0.05,
    "price_lag_7": 0.10,
    "price_pct_change_1": 0.10,
    "price_pct_change_7": 0.10,
    "price_rolling_mean_28": 0.10,
    "price_relative_to_28d_avg": 0.10,
    "sell_price": 0.10,
    "target_sell_price": 0.10,
}
_DEFAULT_MISS_TOLERANCE: float = 0.02


def validate_supervised_table(
    df: pd.DataFrame,
    *,
    strict: bool = False,
    expected_columns: Sequence[str] | None = None,
) -> SupervisedValidationReport:
    """Validate the contract of a supervised training table.

    Checks
    ------
    * Schema: all canonical columns present.
    * ``target_date == origin_date + horizon`` (per-row).
    * ``horizon > 0``.
    * No duplicate ``(id, origin_date, horizon)``.
    * No missing ``target_sales``.
    * Per-feature missing rates (warning if a non-tolerable column exceeds
      ``_HIGH_MISS_TOLERANCE`` or the default 2%).

    Set ``strict=True`` to raise :class:`AssertionError` on any error.
    """
    expected = list(expected_columns) if expected_columns else list(SUPERVISED_COLUMNS)
    errors: list[str] = []
    warnings: list[str] = []

    schema_missing = [c for c in expected if c not in df.columns]
    if schema_missing:
        errors.append(f"missing supervised columns: {schema_missing}")

    n_rows = int(len(df))

    if {"origin_date", "horizon", "target_date"}.issubset(df.columns):
        # Vectorized check: target_date - origin_date == horizon days.
        delta_days = (
            (df["target_date"] - df["origin_date"]).dt.days
            if pd.api.types.is_datetime64_any_dtype(df["origin_date"])
            and pd.api.types.is_datetime64_any_dtype(df["target_date"])
            else None
        )
        if delta_days is None:
            errors.append("origin_date / target_date must be datetime64")
            horizon_mismatch = 0
            nonpositive = 0
        else:
            horizon_mismatch = int(((delta_days != df["horizon"]).fillna(True)).sum())
            if horizon_mismatch:
                errors.append(
                    f"{horizon_mismatch} rows where target_date != origin_date + horizon days"
                )
            nonpositive = int((df["horizon"] <= 0).sum())
            if nonpositive:
                errors.append(f"{nonpositive} rows with horizon <= 0")
    else:
        horizon_mismatch = 0
        nonpositive = 0

    duplicate_count = 0
    if {"id", "origin_date", "horizon"}.issubset(df.columns):
        dup_mask = df.duplicated(subset=["id", "origin_date", "horizon"], keep=False)
        duplicate_count = int(dup_mask.sum())
        if duplicate_count:
            errors.append(
                f"{duplicate_count} duplicate (id, origin_date, horizon) rows"
            )

    missing_target = 0
    if "target_sales" in df.columns:
        missing_target = int(df["target_sales"].isna().sum())
        if missing_target:
            errors.append(f"{missing_target} rows with missing target_sales")

    # Per-feature missing rates (warnings only).
    missing_rates: dict[str, float] = {}
    feature_cols: Iterable[str] = (
        *ORIGIN_FEATURE_COLUMNS,
        *TARGET_FEATURE_COLUMNS,
    )
    if n_rows > 0:
        for col in feature_cols:
            if col not in df.columns:
                continue
            rate = float(df[col].isna().mean())
            missing_rates[col] = rate
            tol = _HIGH_MISS_TOLERANCE.get(col, _DEFAULT_MISS_TOLERANCE)
            if rate > tol:
                warnings.append(f"{col}: missing rate {rate:.2%} > tolerance {tol:.0%}")

    n_unique_ids = int(df["id"].nunique()) if "id" in df.columns else 0
    n_origins = (
        int(df["origin_date"].nunique()) if "origin_date" in df.columns else 0
    )
    n_horizons = int(df["horizon"].nunique()) if "horizon" in df.columns else 0

    report = SupervisedValidationReport(
        n_rows=n_rows,
        n_unique_ids=n_unique_ids,
        n_origins=n_origins,
        n_horizons=n_horizons,
        duplicate_keys_count=duplicate_count,
        horizon_mismatch_count=horizon_mismatch,
        nonpositive_horizon_count=nonpositive,
        missing_target_sales_count=missing_target,
        missing_feature_rates=missing_rates,
        schema_missing_columns=schema_missing,
        errors=errors,
        warnings=warnings,
    )
    if strict and not report.ok:
        raise AssertionError("Supervised table validation failed:\n" + report.summary())
    return report


__all__ = [
    "IDENTITY_COLUMNS",
    "ORIGIN_FEATURE_COLUMNS",
    "TARGET_FEATURE_COLUMNS",
    "META_COLUMNS",
    "SUPERVISED_COLUMNS",
    "default_training_origins",
    "build_supervised_table",
    "SupervisedValidationReport",
    "validate_supervised_table",
]
