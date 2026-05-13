"""Phase 5 entry point: LightGBM point-forecast backtest.

Strict ML backtesting leakage rule (per the project charter):

* For each backtest origin O:
    train  = supervised[ target_date <= O - VALID_WINDOW_DAYS ]
    valid  = supervised[ O - VALID_WINDOW_DAYS < target_date <= O ]
    test   = supervised[ origin_date == O ]

* The training set deliberately uses ``target_date <=``, NOT
  ``origin_date <``. With direct multi-horizon training a future target
  can have an *earlier* origin (e.g. origin_date=O-3, horizon=14 -> target
  at O+11), so filtering by origin_date alone leaks future targets.

Reads:
    data/processed/train_features_ca1.parquet
    data/interim/m5_base_ca1.parquet  (only for d-int -> date coercion)
    outputs/reports/baseline_predictions_ca1.parquet  (for the comparison)

Writes:
    outputs/models/lightgbm_point_model_ca1.pkl                 (joblib bundle)
    outputs/reports/lightgbm_backtest_predictions_ca1.parquet   (long preds + actuals)
    outputs/reports/lightgbm_backtest_scores_ca1.csv            (per origin x horizon)
    outputs/reports/model_comparison_ca1.csv                    (LightGBM vs baselines)

Run::

    python -m seercast.training.train_lightgbm
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from seercast.config import (
    ARTIFACTS,
    BACKTEST,
    ensure_dirs,
)
from seercast.evaluation.backtesting import origin_to_date
from seercast.evaluation.metrics import score_by_group
from seercast.features.supervised import validate_supervised_table
from seercast.models.lightgbm_model import LightGBMPointModel


# 56 days = ~8 weeks of recent targets reserved for early-stopping validation.
VALID_WINDOW_DAYS: int = 56


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


@dataclass
class BacktestSplit:
    """Materialized train / valid / test slices for a single backtest origin."""

    origin_date: pd.Timestamp
    valid_start: pd.Timestamp
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def split_for_backtest_origin(
    supervised: pd.DataFrame,
    origin_date: pd.Timestamp,
    valid_window_days: int = VALID_WINDOW_DAYS,
) -> BacktestSplit:
    """Slice the supervised table into train / valid / test for one origin.

    The function enforces the ML leakage rule:

    * train: ``target_date <= valid_start`` (and target_date <= origin_date by extension)
    * valid: ``valid_start < target_date <= origin_date``
    * test:  ``origin_date == origin_date``

    Together, train + valid is exactly the set ``target_date <= origin_date``,
    matching the project charter rule.
    """
    origin_date = pd.Timestamp(origin_date)
    valid_start = origin_date - pd.Timedelta(days=valid_window_days)

    target = supervised["target_date"]
    train_mask = target <= valid_start
    valid_mask = (target > valid_start) & (target <= origin_date)
    test_mask = supervised["origin_date"] == origin_date

    return BacktestSplit(
        origin_date=origin_date,
        valid_start=valid_start,
        train=supervised.loc[train_mask].reset_index(drop=True),
        valid=supervised.loc[valid_mask].reset_index(drop=True),
        test=supervised.loc[test_mask].reset_index(drop=True),
    )


# --------------------------------------------------------------------------- #
# Predictions wiring
# --------------------------------------------------------------------------- #


def _predictions_frame(
    test_df: pd.DataFrame,
    preds: np.ndarray,
    model_name: str = "lightgbm_point",
) -> pd.DataFrame:
    """Build the canonical long predictions frame for one origin."""
    out = pd.DataFrame(
        {
            "model": model_name,
            "origin_date": test_df["origin_date"].values,
            "id": test_df["id"].values,
            "horizon": test_df["horizon"].values,
            "target_date": test_df["target_date"].values,
            "prediction": preds,
            "actual": test_df["target_sales"].astype(float).values,
        }
    )
    return out



# --------------------------------------------------------------------------- #
# Apples-to-apples model-comparison grid
# --------------------------------------------------------------------------- #


_GRID_COLS: tuple[str, ...] = ("origin_date", "id", "horizon", "target_date")


def build_model_comparison(
    lgbm_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    *,
    grid_cols: Sequence[str] = _GRID_COLS,
) -> pd.DataFrame:
    """Build the per-model MAE/RMSE/WAPE/Bias frame on a fair evaluation grid.

    Phase 3 baselines run on *all* 28 horizons of every backtest origin,
    while Phase 5 LightGBM runs only on the four direct horizons
    ``[1, 7, 14, 28]``. Naively concatenating their long prediction frames
    would score baselines on 7x more rows than LightGBM and make WAPE
    comparisons meaningless.

    Fix: the LightGBM predictions define the canonical evaluation grid.
    Baselines are inner-joined onto that grid by ``grid_cols``
    (default ``(origin_date, id, horizon, target_date)``) so every model is
    scored on exactly the same `(id, horizon, target_date)` slots.

    A WARN is printed if the matched baseline row count per model differs
    from the LGBM grid size (i.e. a baseline has gaps on the matched grid).

    Parameters
    ----------
    lgbm_predictions
        Long DataFrame from :func:`run` for the LightGBM point model. Must
        contain ``model``, ``prediction``, ``actual``, plus ``grid_cols``.
    baseline_predictions
        Long DataFrame from
        :func:`seercast.training.train_baselines.run`. Same column set.
    grid_cols
        Join keys. Default ``(origin_date, id, horizon, target_date)``.

    Returns
    -------
    pandas.DataFrame
        One row per model, columns
        ``model, n, MAE, RMSE, WAPE, Bias`` -- sorted by WAPE asc, with
        equal ``n`` across rows.
    """
    grid_cols = list(grid_cols)
    grid = lgbm_predictions[grid_cols].drop_duplicates()

    bl_matched = baseline_predictions.merge(grid, on=grid_cols, how="inner")

    expected_per_model = len(grid)
    counts = bl_matched.groupby("model").size()
    uneven = counts[counts != expected_per_model]
    if not uneven.empty:
        print(
            f"WARN: baseline coverage is uneven on the matched grid. "
            f"Expected {expected_per_model:,} rows per model; got:\n"
            f"{uneven.to_string()}"
        )

    combined = pd.concat([bl_matched, lgbm_predictions], ignore_index=True)
    comparison = (
        score_by_group(combined, by=("model",))
        .sort_values("WAPE")
        .reset_index(drop=True)
    )
    return comparison


# --------------------------------------------------------------------------- #
# Top-level orchestration
# --------------------------------------------------------------------------- #


def run(
    supervised_path: Path | str = ARTIFACTS.train_features_ca1,
    base_path: Path | str = ARTIFACTS.base_table_ca1,
    *,
    backtest_origins: Sequence = BACKTEST.origins,
    valid_window_days: int = VALID_WINDOW_DAYS,
    baseline_predictions_path: Path | str = ARTIFACTS.baseline_predictions_ca1,
    out_model: Path | str = ARTIFACTS.lightgbm_point_model_ca1,
    out_predictions: Path | str = ARTIFACTS.lightgbm_backtest_predictions_ca1,
    out_scores: Path | str = ARTIFACTS.lightgbm_backtest_scores_ca1,
    out_comparison: Path | str = ARTIFACTS.model_comparison_ca1,
) -> dict:
    """Train + backtest LightGBM on each origin, then write deliverables.

    Returns a dict with keys ``predictions``, ``scores``, ``summary``,
    ``comparison``, and ``models`` (a dict ``{origin_date_iso: trained_model}``).
    """
    ensure_dirs()

    sup = pd.read_parquet(supervised_path)
    print(f"loaded supervised table: {sup.shape[0]:,} rows x {sup.shape[1]} cols")

    # Hard-fail on a corrupted feature table before training touches it.
    sup_report = validate_supervised_table(sup, strict=True)
    print(sup_report.summary())
    print()

    # Need the base table to coerce d-integer / d-string origin specs to dates.
    base = pd.read_parquet(base_path, columns=["id", "d", "date"])

    backtest_dates = [pd.Timestamp(origin_to_date(base, o)) for o in backtest_origins]

    print(f"backtesting on origins: {[d.date() for d in backtest_dates]}")
    print(f"valid window: last {valid_window_days} days of targets per origin")
    print()

    all_predictions: list[pd.DataFrame] = []
    models: dict[str, LightGBMPointModel] = {}

    for origin_date in backtest_dates:
        split = split_for_backtest_origin(sup, origin_date, valid_window_days)
        print(
            f"origin {origin_date.date()}: "
            f"train={len(split.train):,}, valid={len(split.valid):,}, test={len(split.test):,}"
        )
        if len(split.test) == 0:
            print(f"  WARN: no test rows at origin {origin_date.date()}; skipping")
            continue
        if len(split.train) == 0:
            print(f"  WARN: no training rows for origin {origin_date.date()}; skipping")
            continue

        model = LightGBMPointModel()
        model.fit(split.train, valid_df=split.valid)
        preds = model.predict(split.test)
        models[origin_date.isoformat()] = model

        all_predictions.append(_predictions_frame(split.test, preds))

    if not all_predictions:
        raise RuntimeError("no predictions produced -- check backtest_origins / supervised table")

    predictions = pd.concat(all_predictions, ignore_index=True)

    # Per-origin x horizon scores.
    scores = score_by_group(predictions, by=("origin_date", "horizon"))
    # Overall summary per model (just lightgbm here).
    summary = score_by_group(predictions, by=("model",))

    # ------- comparison vs baselines (apples-to-apples grid) -----------
    # The model_comparison file is OPTIONAL: it requires Phase 3's
    # baseline_predictions parquet. If you haven't run train_baselines yet,
    # the LGBM-only deliverables still get written.
    baseline_predictions_path_p = Path(baseline_predictions_path)
    if baseline_predictions_path_p.exists():
        baseline_predictions = pd.read_parquet(baseline_predictions_path_p)
        comparison = build_model_comparison(
            lgbm_predictions=predictions,
            baseline_predictions=baseline_predictions,
        )
    else:
        print(
            f"NOTE: Phase 3 baseline predictions not found at "
            f"{baseline_predictions_path_p};\n"
            f"      skipping {ARTIFACTS.model_comparison_ca1.name}. "
            f"Run train_baselines first to enable the comparison."
        )
        comparison = pd.DataFrame()

    # ------- persist ---------------------------------------------------
    out_model = Path(out_model)
    out_predictions = Path(out_predictions)
    out_scores = Path(out_scores)
    out_comparison = Path(out_comparison)
    for p in (out_model, out_predictions, out_scores, out_comparison):
        p.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "models": models,
        "valid_window_days": valid_window_days,
        "backtest_origins": [d.isoformat() for d in backtest_dates],
    }
    joblib.dump(bundle, out_model)
    predictions.to_parquet(out_predictions, index=False)
    scores.to_csv(out_scores, index=False)
    if not comparison.empty:
        comparison.to_csv(out_comparison, index=False)

    print()
    print(f"wrote {out_model}")
    print(f"wrote {out_predictions}  ({len(predictions):,} rows)")
    print(f"wrote {out_scores}        ({len(scores):,} rows)")
    if not comparison.empty:
        print(f"wrote {out_comparison}    ({len(comparison):,} rows)")
    print()
    if not comparison.empty:
        print("model comparison (sorted by WAPE):")
        print(comparison.to_string(index=False))
    else:
        print("model comparison skipped (no Phase 3 baseline predictions found).")

    return {
        "predictions": predictions,
        "scores": scores,
        "summary": summary,
        "comparison": comparison,
        "models": models,
    }


def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5 LightGBM point-model trainer.")
    p.add_argument("--supervised", type=Path, default=ARTIFACTS.train_features_ca1)
    p.add_argument("--base", type=Path, default=ARTIFACTS.base_table_ca1)
    p.add_argument("--valid-window", type=int, default=VALID_WINDOW_DAYS)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    run(
        supervised_path=args.supervised,
        base_path=args.base,
        valid_window_days=args.valid_window,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
