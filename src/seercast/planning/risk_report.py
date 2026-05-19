"""Planner-facing risk report (v1.1).

Aggregates the existing quantile forecasts (and scenario forecasts, if
present) into a per-product report aimed at human planners rather than
ML iteration.

v1.1 metric refinement
----------------------

v1.0 produced extreme ratio values when ``expected_demand_p50`` was
near zero, which made the headline leaderboards misleading. v1.1
introduces two changes:

1. **Demand floor for ratio metrics.** All ratios use a configurable
   denominator floor (default 10 units). The original raw ratios are
   kept under their original names for provenance; the *floored*
   versions are what we display and rank by.
2. **Attention scores.** Volume-weighted scores combine the absolute
   risk signal with ``log1p(expected_demand_p50)``, so a tiny p50 with
   a large percentage shift no longer dominates the leaderboards.

There's also a new "low-expected / high-upside" view that surfaces
items with near-zero p50 but meaningful p90, since those are a
legitimate planning use-case -- they just don't belong on the
"uncertainty" or "scenario-sensitive" leaderboard.

Honest framing carried from v1.0:
- p90 is the conservative *planning* quantile, NOT a guarantee.
- Scenario sensitivity is predictive (model what-if), NOT causal.
- Risk labels are heuristic percentile bands, NOT service-level targets.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd


_EPS = 1e-9

# Default denominator floor for ratio metrics. Below this, the displayed
# ratio is calibrated to "uncertainty per N units of typical demand" rather
# than "uncertainty per epsilon of demand". 10 units is a reasonable
# floor for retail item-day forecasts (covers fast and medium sellers;
# only the rarest intermittent items hit the floor).
DEFAULT_DEMAND_FLOOR: float = 10.0

IDENTITY_COLS: tuple[str, ...] = (
    "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
)
QUANTILE_COLS: tuple[str, str, str] = ("p10", "p50", "p90")

DEFAULT_SCENARIO_NAME_MAP: Mapping[str, str] = {
    "momentum_+20pct": "momentum_plus_delta_pct",
    "momentum_-20pct": "momentum_minus_delta_pct",
    "price_+10pct":   "price_plus_delta_pct",
    "price_-10pct":   "price_minus_delta_pct",
}


# --------------------------------------------------------------------------- #
# 1. Per-id quantile aggregates
# --------------------------------------------------------------------------- #


def build_quantile_aggregates(
    quantile_predictions: pd.DataFrame,
    *,
    group_by: str = "id",
    identity_cols: Iterable[str] = IDENTITY_COLS,
    quantile_cols: tuple[str, str, str] = QUANTILE_COLS,
    demand_floor: float = DEFAULT_DEMAND_FLOOR,
) -> pd.DataFrame:
    """Per-id sums of p10/p50/p90 plus raw and floored risk metrics.

    Parameters
    ----------
    demand_floor
        Minimum value used as the denominator in ratio metrics. Items
        with expected p50 below the floor get a *deflated* ratio that
        reflects "uncertainty per floor units" rather than "uncertainty
        per epsilon", which avoids billion-scale percentages.
    """
    p10, p50, p90 = quantile_cols
    needed = {group_by, p10, p50, p90}
    missing = needed - set(quantile_predictions.columns)
    if missing:
        raise KeyError(
            f"build_quantile_aggregates needs columns {sorted(needed)}; "
            f"missing: {sorted(missing)}"
        )

    g = quantile_predictions.groupby(group_by, as_index=False)
    agg = g.agg(
        expected_demand_p50=(p50, "sum"),
        conservative_demand_p90=(p90, "sum"),
        low_demand_p10=(p10, "sum"),
        n_forecast_rows=(p50, "size"),
    )

    agg["risk_buffer"] = agg["conservative_demand_p90"] - agg["expected_demand_p50"]
    agg["uncertainty_width"] = agg["conservative_demand_p90"] - agg["low_demand_p10"]

    # Raw ratios (kept for provenance; can blow up near zero by design).
    raw_denom = agg["expected_demand_p50"].abs().clip(lower=_EPS)
    agg["relative_uncertainty"] = agg["uncertainty_width"] / raw_denom
    agg["underforecast_risk_proxy"] = agg["risk_buffer"] / raw_denom

    # Floored ratios used for display + ranking + labels.
    floored_denom = agg["expected_demand_p50"].abs().clip(lower=float(demand_floor))
    agg["relative_uncertainty_floored"] = agg["uncertainty_width"] / floored_denom
    agg["underforecast_risk_proxy_floored"] = agg["risk_buffer"] / floored_denom

    # Attach identity context.
    other_id_cols = [
        c for c in identity_cols
        if c != group_by and c in quantile_predictions.columns
    ]
    if other_id_cols:
        id_lookup = (
            quantile_predictions[[group_by, *other_id_cols]]
            .drop_duplicates(group_by)
        )
        agg = agg.merge(id_lookup, on=group_by, how="left")

    # Carry the floor used so downstream summary can record it.
    agg.attrs["demand_floor"] = float(demand_floor)
    return agg


# --------------------------------------------------------------------------- #
# 2. Scenario sensitivity (with floored pct)
# --------------------------------------------------------------------------- #


def add_scenario_sensitivity(
    risk_df: pd.DataFrame,
    scenario_forecasts: pd.DataFrame | None,
    *,
    group_by: str = "id",
    scenario_name_map: Mapping[str, str] = DEFAULT_SCENARIO_NAME_MAP,
    demand_floor: float = DEFAULT_DEMAND_FLOOR,
) -> pd.DataFrame:
    """Add per-id scenario sensitivity columns. Uses ``demand_floor`` so a
    near-zero base p50 doesn't produce an exploding pct.

    Output columns:
    * ``max_abs_scenario_delta_p50`` -- max |delta| in units (always stable).
    * ``max_abs_scenario_delta_p50_pct`` -- max |delta_pct| using raw denom
      (can blow up; kept for provenance).
    * ``max_abs_scenario_delta_p50_pct_floored`` -- max |delta_pct| using
      the floored denominator (this is the stable metric for ranking).
    * ``most_sensitive_scenario`` -- scenario name with the max *floored*
      |delta_pct|.
    * fixed-name per-scenario columns from ``scenario_name_map`` carry
      raw delta_pct (per scenario) -- they're per-scenario context, not
      a ranking metric.
    """
    out = risk_df.copy()
    fixed_cols = list(scenario_name_map.values())

    if scenario_forecasts is None or len(scenario_forecasts) == 0:
        out["max_abs_scenario_delta_p50"] = np.nan
        out["max_abs_scenario_delta_p50_pct"] = np.nan
        out["max_abs_scenario_delta_p50_pct_floored"] = np.nan
        out["most_sensitive_scenario"] = None
        for col in fixed_cols:
            out[col] = np.nan
        return out

    needed = {group_by, "scenario", "base_p50", "scenario_p50"}
    missing = needed - set(scenario_forecasts.columns)
    if missing:
        raise KeyError(
            f"add_scenario_sensitivity needs columns {sorted(needed)}; "
            f"missing: {sorted(missing)}"
        )

    # Per (id, scenario) totals.
    per_id_scen = (
        scenario_forecasts
        .groupby([group_by, "scenario"], as_index=False)
        .agg(
            base_total_p50=("base_p50", "sum"),
            scenario_total_p50=("scenario_p50", "sum"),
        )
    )
    per_id_scen["delta_total_p50"] = (
        per_id_scen["scenario_total_p50"] - per_id_scen["base_total_p50"]
    )
    # Raw and floored pct. Raw can blow up; floored is the ranking-friendly one.
    raw_denom_scen = per_id_scen["base_total_p50"].abs().clip(lower=_EPS)
    floored_denom_scen = per_id_scen["base_total_p50"].abs().clip(lower=float(demand_floor))
    per_id_scen["delta_total_p50_pct"] = per_id_scen["delta_total_p50"] / raw_denom_scen
    per_id_scen["delta_total_p50_pct_floored"] = per_id_scen["delta_total_p50"] / floored_denom_scen

    rows: list[dict] = []
    for id_, sub in per_id_scen.groupby(group_by):
        abs_units = sub["delta_total_p50"].abs()
        abs_pct_raw = sub["delta_total_p50_pct"].abs()
        abs_pct_floor = sub["delta_total_p50_pct_floored"].abs()
        if abs_pct_floor.empty:
            continue
        # Most-sensitive scenario picked using the FLOORED pct so a tiny
        # base doesn't crown an irrelevant item.
        idx_floor = abs_pct_floor.idxmax()
        row: dict = {
            group_by: id_,
            "max_abs_scenario_delta_p50": float(abs_units.max()),
            "max_abs_scenario_delta_p50_pct": float(abs_pct_raw.max()),
            "max_abs_scenario_delta_p50_pct_floored": float(abs_pct_floor.max()),
            "most_sensitive_scenario": str(sub.loc[idx_floor, "scenario"]),
        }
        s_to_pct = dict(zip(sub["scenario"], sub["delta_total_p50_pct"]))
        for raw_name, fixed_name in scenario_name_map.items():
            row[fixed_name] = float(s_to_pct.get(raw_name, np.nan))
        rows.append(row)

    if not rows:
        out["max_abs_scenario_delta_p50"] = np.nan
        out["max_abs_scenario_delta_p50_pct"] = np.nan
        out["max_abs_scenario_delta_p50_pct_floored"] = np.nan
        out["most_sensitive_scenario"] = None
        for col in fixed_cols:
            out[col] = np.nan
        return out

    scen_df = pd.DataFrame(rows)
    return out.merge(scen_df, on=group_by, how="left")


# --------------------------------------------------------------------------- #
# 3. Volume-weighted attention scores
# --------------------------------------------------------------------------- #


def add_attention_scores(
    risk_df: pd.DataFrame,
    *,
    expected_col: str = "expected_demand_p50",
) -> pd.DataFrame:
    """Add volume-weighted scores that combine an absolute risk signal
    with ``log1p(expected_demand)`` so near-zero p50 items don't dominate
    rankings:

    * ``stockout_attention_score = risk_buffer * log1p(expected_demand_p50)``
    * ``scenario_attention_score = max_abs_scenario_delta_p50 * log1p(expected_demand_p50)``
      (only added if the scenario column is present)
    """
    out = risk_df.copy()
    base = out[expected_col].clip(lower=0.0)
    log_weight = np.log1p(base)
    out["stockout_attention_score"] = out["risk_buffer"] * log_weight
    if "max_abs_scenario_delta_p50" in out.columns:
        out["scenario_attention_score"] = (
            out["max_abs_scenario_delta_p50"].fillna(0.0) * log_weight
        )
        # Where the underlying scenario data was missing, the score should be NaN
        # (not a fake zero from fillna).
        scen_missing = out["max_abs_scenario_delta_p50"].isna()
        out.loc[scen_missing, "scenario_attention_score"] = np.nan
    else:
        out["scenario_attention_score"] = np.nan
    return out


# --------------------------------------------------------------------------- #
# 4. Heuristic risk labels (now sourcing from floored / score columns)
# --------------------------------------------------------------------------- #


# Labels source from the stable columns. The raw versions still exist on
# the report; they're just not used for ranking/labelling.
DEFAULT_LABEL_MAPPING: dict[str, str] = {
    "uncertainty_label":          "relative_uncertainty_floored",
    "stockout_attention_label":   "stockout_attention_score",
    "scenario_sensitivity_label": "scenario_attention_score",
}

_HIGH_QUANTILE = 0.90
_MEDIUM_QUANTILE = 0.70


def add_risk_labels(
    df: pd.DataFrame,
    *,
    label_mapping: Mapping[str, str] = DEFAULT_LABEL_MAPPING,
    high_quantile: float = _HIGH_QUANTILE,
    medium_quantile: float = _MEDIUM_QUANTILE,
) -> pd.DataFrame:
    """Percentile-band labels: top 10% high, next 20% medium, rest low.
    Strict ``>`` boundaries so the bands are exactly 10% / 20% / 70%."""
    out = df.copy()
    for label_col, source_col in label_mapping.items():
        if source_col not in out.columns or out[source_col].isna().all():
            out[label_col] = "low"
            continue
        rank = out[source_col].rank(method="average", pct=True, na_option="bottom")
        out[label_col] = np.where(
            rank > high_quantile, "high",
            np.where(rank > medium_quantile, "medium", "low"),
        )
    return out


# --------------------------------------------------------------------------- #
# 5. Low-expected / high-upside view
# --------------------------------------------------------------------------- #


def low_expected_high_upside(
    risk_df: pd.DataFrame,
    *,
    p50_threshold: float = 5.0,
    p90_threshold: float = 10.0,
    sort_by: str = "conservative_demand_p90",
    ascending: bool = False,
) -> pd.DataFrame:
    """Subset of products with near-zero expected demand AND meaningful
    upside under the conservative quantile.

    These are legitimately worth a planner's attention but don't belong
    on the "uncertainty" or "scenario-sensitive" leaderboards (where
    they'd otherwise dominate via near-zero denominators).
    """
    if not {"expected_demand_p50", "conservative_demand_p90"}.issubset(risk_df.columns):
        raise KeyError(
            "low_expected_high_upside needs expected_demand_p50 and "
            "conservative_demand_p90 columns"
        )
    mask = (
        (risk_df["expected_demand_p50"] < p50_threshold)
        & (risk_df["conservative_demand_p90"] >= p90_threshold)
    )
    out = risk_df.loc[mask].copy()
    if sort_by in out.columns:
        out = out.sort_values(sort_by, ascending=ascending)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 6. Convenience: filter + top-N slices
# --------------------------------------------------------------------------- #


def filter_by_min_expected_p50(
    risk_df: pd.DataFrame,
    *,
    min_expected_p50: float,
) -> pd.DataFrame:
    """Return rows whose ``expected_demand_p50`` is at least ``min_expected_p50``.

    Used to keep the high-uncertainty leaderboard from overlapping with
    the low-expected / high-upside leaderboard: items whose forecasts
    are dominated by structural zeros belong on the low-upside report,
    not on the "noisy forecast" report. NaN-p50 rows are excluded.
    """
    if "expected_demand_p50" not in risk_df.columns:
        return risk_df.iloc[:0].copy()
    return (
        risk_df.loc[risk_df["expected_demand_p50"] >= float(min_expected_p50)]
        .reset_index(drop=True)
    )


def top_n_by(
    risk_df: pd.DataFrame,
    metric: str,
    *,
    n: int = 20,
    ascending: bool = False,
) -> pd.DataFrame:
    """Return the top-N rows by ``metric``. Drops NaN values in that metric."""
    if metric not in risk_df.columns:
        return risk_df.iloc[:0].copy()
    return (
        risk_df.dropna(subset=[metric])
        .sort_values(metric, ascending=ascending)
        .head(n)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# 7. End-to-end builder
# --------------------------------------------------------------------------- #


def build_planner_risk_report(
    quantile_predictions: pd.DataFrame,
    scenario_forecasts: pd.DataFrame | None = None,
    *,
    group_by: str = "id",
    demand_floor: float = DEFAULT_DEMAND_FLOOR,
) -> pd.DataFrame:
    """One-shot helper: aggregates + scenario sensitivity + attention scores + labels."""
    risk = build_quantile_aggregates(
        quantile_predictions, group_by=group_by, demand_floor=demand_floor,
    )
    risk = add_scenario_sensitivity(
        risk, scenario_forecasts, group_by=group_by, demand_floor=demand_floor,
    )
    risk = add_attention_scores(risk)
    risk = add_risk_labels(risk)
    risk.attrs["demand_floor"] = float(demand_floor)
    return risk


__all__ = [
    "DEFAULT_DEMAND_FLOOR",
    "IDENTITY_COLS",
    "QUANTILE_COLS",
    "DEFAULT_SCENARIO_NAME_MAP",
    "DEFAULT_LABEL_MAPPING",
    "build_quantile_aggregates",
    "add_scenario_sensitivity",
    "add_attention_scores",
    "add_risk_labels",
    "low_expected_high_upside",
    "filter_by_min_expected_p50",
    "top_n_by",
    "build_planner_risk_report",
]
