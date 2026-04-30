"""Validation checks for the joined base table.

These are *fail-loud* data-quality assertions plus a few measured
diagnostics (e.g. missing-price rate). The intent is to catch silent join
or schema bugs before they leak into feature engineering.

The core check function returns a :class:`BaseTableValidationReport`. It
does not raise unless ``strict=True`` is passed; otherwise it lets callers
decide how to react (e.g. log + continue in a notebook vs. abort in CI).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from seercast.config import BASE_TABLE_COLUMNS


# --------------------------------------------------------------------------- #
# Report dataclass
# --------------------------------------------------------------------------- #


@dataclass
class BaseTableValidationReport:
    """Structured result of running validation on the base table.

    ``errors`` are conditions that must be fixed before continuing.
    ``warnings`` are observations that may be acceptable depending on
    context (e.g. some missing prices are expected in M5).
    """

    n_rows: int
    n_unique_ids: int
    date_min: pd.Timestamp | None
    date_max: pd.Timestamp | None
    missing_price_rate: float
    duplicate_id_date_count: int
    negative_sales_count: int
    missing_date_count: int
    schema_missing_columns: list[str]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        """Human-readable one-screen summary."""
        lines = [
            f"rows                     : {self.n_rows:,}",
            f"unique ids               : {self.n_unique_ids:,}",
            f"date range               : {self.date_min} → {self.date_max}",
            f"missing-price rate       : {self.missing_price_rate:.4%}",
            f"duplicate (id, date)     : {self.duplicate_id_date_count:,}",
            f"negative-sales rows      : {self.negative_sales_count:,}",
            f"missing-date rows        : {self.missing_date_count:,}",
            f"schema missing cols      : {self.schema_missing_columns or 'none'}",
            f"errors                   : {len(self.errors)}",
            f"warnings                 : {len(self.warnings)}",
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

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view (useful for writing to a report)."""
        return {
            "n_rows": int(self.n_rows),
            "n_unique_ids": int(self.n_unique_ids),
            "date_min": None if self.date_min is None else str(self.date_min.date()),
            "date_max": None if self.date_max is None else str(self.date_max.date()),
            "missing_price_rate": float(self.missing_price_rate),
            "duplicate_id_date_count": int(self.duplicate_id_date_count),
            "negative_sales_count": int(self.negative_sales_count),
            "missing_date_count": int(self.missing_date_count),
            "schema_missing_columns": list(self.schema_missing_columns),
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #


# Threshold for "way too many missing prices" — empirically M5 is ~7-8% on CA_1
# because items are listed before they start selling. 30% is a generous
# upper bound that still flags broken joins.
_MAX_TOLERATED_MISSING_PRICE_RATE: float = 0.30


def validate_base_table(
    df: pd.DataFrame,
    *,
    strict: bool = False,
    expected_columns: list[str] | None = None,
) -> BaseTableValidationReport:
    """Run data-quality checks against the long base table.

    Checks performed:

    * **Schema** — all columns in :data:`seercast.config.BASE_TABLE_COLUMNS`
      (or ``expected_columns``) are present.
    * **Date parses** — ``date`` is datetime64 with no NaT after the calendar join.
    * **No duplicate observations** — ``(id, date)`` is unique.
    * **Sales non-negative** — sales must be ≥ 0.
    * **Missing prices measured** — left-join may legitimately leave NaN
      ``sell_price`` rows, so we measure the rate rather than reject them.
      A missing rate above 30% is treated as a warning that something
      structural is off.

    Parameters
    ----------
    df
        Output of :func:`seercast.data.transform.build_base_table`.
    strict
        If ``True``, raise :class:`AssertionError` when any error is found.
        Default is ``False`` so callers can render the report and decide.
    expected_columns
        Override schema. Defaults to ``BASE_TABLE_COLUMNS``.

    Returns
    -------
    BaseTableValidationReport
    """
    expected = expected_columns or BASE_TABLE_COLUMNS
    errors: list[str] = []
    warnings: list[str] = []

    # ----- Schema -------------------------------------------------------- #
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        errors.append(f"missing required columns: {missing_cols}")

    # ----- Date dtype + NaT --------------------------------------------- #
    missing_date_count = 0
    date_min = date_max = None
    if "date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            errors.append("'date' is not datetime64 dtype after calendar join")
        else:
            missing_date_count = int(df["date"].isna().sum())
            if missing_date_count > 0:
                errors.append(
                    f"{missing_date_count} rows have NaT 'date' (calendar join "
                    "missed some 'd' values)"
                )
            if len(df) > 0:
                date_min = df["date"].min()
                date_max = df["date"].max()

    # ----- Duplicate (id, date) ----------------------------------------- #
    duplicate_count = 0
    if {"id", "date"}.issubset(df.columns):
        dup_mask = df.duplicated(subset=["id", "date"], keep=False)
        duplicate_count = int(dup_mask.sum())
        if duplicate_count > 0:
            errors.append(
                f"{duplicate_count} duplicate (id, date) rows — joins are not "
                "many-to-one as expected"
            )

    # ----- Sales non-negative ------------------------------------------- #
    negative_sales_count = 0
    if "sales" in df.columns:
        neg = (df["sales"] < 0)
        negative_sales_count = int(neg.sum())
        if negative_sales_count > 0:
            errors.append(f"{negative_sales_count} rows have negative 'sales'")

    # ----- Missing-price rate (warning only) ---------------------------- #
    missing_price_rate = float("nan")
    if "sell_price" in df.columns and len(df) > 0:
        missing_price_rate = float(df["sell_price"].isna().mean())
        if missing_price_rate > _MAX_TOLERATED_MISSING_PRICE_RATE:
            warnings.append(
                f"missing-price rate is {missing_price_rate:.2%}, above the "
                f"{_MAX_TOLERATED_MISSING_PRICE_RATE:.0%} sanity threshold; "
                "double-check the price join keys (store_id, item_id, wm_yr_wk)"
            )

    # ----- Sales dtype: should be a numeric integer-ish ----------------- #
    if "sales" in df.columns and not pd.api.types.is_integer_dtype(df["sales"]):
        warnings.append(
            f"'sales' dtype is {df['sales'].dtype}, expected an integer dtype "
            "(int32 from melt_sales)"
        )

    report = BaseTableValidationReport(
        n_rows=int(len(df)),
        n_unique_ids=int(df["id"].nunique()) if "id" in df.columns else 0,
        date_min=date_min,
        date_max=date_max,
        missing_price_rate=missing_price_rate,
        duplicate_id_date_count=duplicate_count,
        negative_sales_count=negative_sales_count,
        missing_date_count=missing_date_count,
        schema_missing_columns=missing_cols,
        errors=errors,
        warnings=warnings,
    )

    if strict and not report.ok:
        raise AssertionError("Base table validation failed:\n" + report.summary())

    return report


__all__ = ["BaseTableValidationReport", "validate_base_table"]
