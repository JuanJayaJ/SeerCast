"""Scenario comparison helpers.

Two responsibilities:

* :func:`add_delta_columns` -- given a long frame with ``base_p10/p50/p90``
  and ``scenario_p10/p50/p90``, fill in ``delta_*`` and ``delta_*_pct``.
  Pure df -> df.
* :func:`scenario_summary` -- aggregate across rows per scenario into the
  summary table the project charter requires (``base_total_p50``,
  ``scenario_total_p50``, ``delta_total_p50``, ``delta_total_p50_pct``,
  with p10/p90 totals included for uncertainty range).

Pct columns are NaN where the corresponding base is zero (avoid
divide-by-zero noise).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


_DEFAULT_QUANTILES: tuple[str, ...] = ("p10", "p50", "p90")


def add_delta_columns(
    df: pd.DataFrame,
    *,
    quantile_columns: Sequence[str] = _DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Add ``delta_<col>`` and ``delta_<col>_pct`` for every quantile.

    Expects ``base_<col>`` and ``scenario_<col>`` to already exist.

    Returns a new DataFrame; ``df`` is not mutated.
    """
    out = df.copy()
    for col in quantile_columns:
        base_col = f"base_{col}"
        scen_col = f"scenario_{col}"
        if base_col not in out.columns or scen_col not in out.columns:
            raise KeyError(
                f"add_delta_columns: missing {base_col} or {scen_col}; "
                f"have {list(out.columns)}"
            )
        out[f"delta_{col}"] = out[scen_col] - out[base_col]
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = out[f"delta_{col}"] / out[base_col]
        out[f"delta_{col}_pct"] = pct.where(out[base_col] != 0, np.nan)
    return out


def scenario_summary(
    forecasts: pd.DataFrame,
    *,
    quantile_columns: Sequence[str] = _DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Per-scenario summary across all rows.

    Parameters
    ----------
    forecasts
        Long DataFrame with at least ``scenario`` and the 6 quantile
        columns (``base_p10/p50/p90``, ``scenario_p10/p50/p90``).
    quantile_columns
        Quantile suffixes to include in the totals, default ``("p10","p50","p90")``.

    Returns
    -------
    pandas.DataFrame
        One row per scenario with columns:
        ``scenario``, ``n``,
        ``base_total_<q>`` and ``scenario_total_<q>`` for each quantile,
        ``delta_total_<q>`` and ``delta_total_<q>_pct`` for each quantile.

        Sorted by ``delta_total_p50_pct`` descending so the largest
        positive uplift sits at the top -- handy for screen-reading
        "which scenario predicted the biggest demand bump?"
    """
    if "scenario" not in forecasts.columns:
        raise KeyError("scenario_summary needs a 'scenario' column")

    base_cols = [f"base_{q}" for q in quantile_columns]
    scen_cols = [f"scenario_{q}" for q in quantile_columns]
    missing = [c for c in base_cols + scen_cols if c not in forecasts.columns]
    if missing:
        raise KeyError(f"scenario_summary missing columns: {missing}")

    grouped = forecasts.groupby("scenario", sort=False)
    rows: list[dict] = []
    for scenario, sub in grouped:
        row: dict = {"scenario": scenario, "n": int(len(sub))}
        for q in quantile_columns:
            b = float(sub[f"base_{q}"].sum())
            s = float(sub[f"scenario_{q}"].sum())
            row[f"base_total_{q}"] = b
            row[f"scenario_total_{q}"] = s
            row[f"delta_total_{q}"] = s - b
            row[f"delta_total_{q}_pct"] = (
                (s - b) / b if b != 0 else float("nan")
            )
        rows.append(row)

    cols = ["scenario", "n"]
    for q in quantile_columns:
        cols.extend(
            [
                f"base_total_{q}",
                f"scenario_total_{q}",
                f"delta_total_{q}",
                f"delta_total_{q}_pct",
            ]
        )
    out = pd.DataFrame(rows)[cols]
    sort_key = f"delta_total_p50_pct" if "p50" in quantile_columns else f"delta_total_{quantile_columns[0]}_pct"
    return out.sort_values(sort_key, ascending=False).reset_index(drop=True)


__all__ = ["add_delta_columns", "scenario_summary"]
