"""Phase 6 entry point: quantile LightGBM backtest (p10 / p50 / p90).

Reuses the same leakage-safe split rule as Phase 5 -- training data for
each backtest origin O satisfies ``target_date <= O``, with the most
recent 56 days of targets reserved for early-stopping validation.

Reads:
    data/processed/train_features_ca1.parquet
    data/interim/m5_base_ca1.parquet  (columns id/d/date/cat_id used for diagnostics)
    outputs/reports/lightgbm_backtest_predictions_ca1.parquet  (Phase 5 point preds)

Writes:
    outputs/models/lightgbm_quantile_models_ca1.pkl
    outputs/reports/quantile_backtest_predictions_ca1.parquet
    outputs/reports/quantile_backtest_scores_ca1.csv
    outputs/reports/uncertainty_diagnostics_ca1.csv

Run::

    python -m seercast.training.train_quantile_lightgbm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from seercast.config import (
    ARTIFACTS,
    BACKTEST,
    QUANTILES,
    ensure_dirs,
)
from seercast.evaluation.backtesting import origin_to_date
from seercast.evaluation.metrics import score_by_group
from seercast.evaluation.probabilistic_metrics import (
    crossing_rate,
    score_quantile_by_group,
)
from seercast.features.supervised import validate_supervised_table
from seercast.models.quantile_lightgbm import (
    QuantileLightGBMModel,
    quantile_column_name,
)
from seercast.training.train_lightgbm import (
    VALID_WINDOW_DAYS,
    split_for_backtest_origin,
)


# --------------------------------------------------------------------------- #
# Per-origin training
# --------------------------------------------------------------------------- #


def _quantile_predictions_frame(
    test_df: pd.DataFrame,
    fixed: pd.DataFrame,
    raw: pd.DataFrame,
    model_name: str,
    quantile_cols: Sequence[str],
) -> pd.DataFrame:
    """Long predictions frame: fixed (sorted) quantiles plus raw counterparts.

    The ``_raw`` columns let us measure crossing rate before fix-up; the
    fixed columns are what downstream consumers should use.
    """
    out = pd.DataFrame(
        {
            "model": model_name,
            "origin_date": test_df["origin_date"].values,
            "id": test_df["id"].values,
            "horizon": test_df["horizon"].values,
            "target_date": test_df["target_date"].values,
            "actual": test_df["target_sales"].astype(float).values,
        }
    )
    for c in quantile_cols:
        out[c] = fixed[c].values
        out[f"{c}_raw"] = raw[c].values
    return out


# --------------------------------------------------------------------------- #
# Top-level orchestration
# --------------------------------------------------------------------------- #


def run(
    supervised_path: Path | str = ARTIFACTS.train_features_ca1,
    base_path: Path | str = ARTIFACTS.base_table_ca1,
    *,
    quantiles: Sequence[float] = QUANTILES,
    backtest_origins: Sequence = BACKTEST.origins,
    valid_window_days: int = VALID_WINDOW_DAYS,
    point_predictions_path: Path | str = ARTIFACTS.lightgbm_backtest_predictions_ca1,
    out_models: Path | str = ARTIFACTS.lightgbm_quantile_models_ca1,
    out_predictions: Path | str = ARTIFACTS.quantile_backtest_predictions_ca1,
    out_scores: Path | str = ARTIFACTS.quantile_backtest_scores_ca1,
    out_diagnostics: Path | str = ARTIFACTS.uncertainty_diagnostics_ca1,
) -> dict:
    """Train + backtest the quantile LightGBM and write deliverables.

    Returns a dict with keys ``predictions``, ``scores``, ``diagnostics``,
    ``p50_vs_point``, ``models``.
    """
    ensure_dirs()

    sup = pd.read_parquet(supervised_path)
    print(f"loaded supervised table: {sup.shape[0]:,} rows x {sup.shape[1]} cols")
    sup_report = validate_supervised_table(sup, strict=True)
    print(sup_report.summary())
    print()

    base = pd.read_parquet(base_path, columns=["id", "d", "date"])
    backtest_dates = [pd.Timestamp(origin_to_date(base, o)) for o in backtest_origins]
    quantile_cols = [quantile_column_name(q) for q in quantiles]

    print(f"backtesting on origins: {[d.date() for d in backtest_dates]}")
    print(f"quantiles: {quantile_cols} (alpha = {list(quantiles)})")
    print()

    all_predictions: list[pd.DataFrame] = []
    models: dict[str, QuantileLightGBMModel] = {}

    for origin_date in backtest_dates:
        split = split_for_backtest_origin(sup, origin_date, valid_window_days)
        print(
            f"origin {origin_date.date()}: "
            f"train={len(split.train):,}, valid={len(split.valid):,}, test={len(split.test):,}"
        )
        if len(split.train) == 0 or len(split.test) == 0:
            print(f"  WARN: empty split at origin {origin_date.date()}; skipping")
            continue

        m = QuantileLightGBMModel(quantiles=tuple(quantiles))
        m.fit(split.train, valid_df=split.valid)

        fixed = m.predict(split.test, fix_crossings=True)
        raw = m.predict(split.test, fix_crossings=False)
        all_predictions.append(
            _quantile_predictions_frame(split.test, fixed, raw, m.name, quantile_cols)
        )
        models[origin_date.isoformat()] = m

    if not all_predictions:
        raise RuntimeError("no predictions produced -- check inputs")

    predictions = pd.concat(all_predictions, ignore_index=True)

    # ------- per-(origin, horizon) scores ------------------------------ #
    scores = score_quantile_by_group(
        predictions, by=("origin_date", "horizon"),
        quantile_columns=quantile_cols, quantile_levels=tuple(quantiles),
    )

    # ------- uncertainty diagnostics: by horizon and by category -------- #
    cat_lookup = (
        pd.read_parquet(base_path, columns=["id", "cat_id"])
        .drop_duplicates("id")
    )
    diag_input = predictions.merge(cat_lookup, on="id", how="left")

    diag_h = score_quantile_by_group(
        diag_input, by=("horizon",),
        quantile_columns=quantile_cols, quantile_levels=tuple(quantiles),
    )
    diag_h.insert(0, "dimension", "horizon")
    diag_h = diag_h.rename(columns={"horizon": "value"})
    diag_h["value"] = diag_h["value"].astype(str)

    diag_c = score_quantile_by_group(
        diag_input, by=("cat_id",),
        quantile_columns=quantile_cols, quantile_levels=tuple(quantiles),
    )
    diag_c.insert(0, "dimension", "category")
    diag_c = diag_c.rename(columns={"cat_id": "value"})
    diag_c["value"] = diag_c["value"].astype(str)

    # Crossing rate on RAW predictions (pre-fix), measured per horizon and
    # per category, so the diagnostic table tells us how often fix-up kicks in.
    raw_cols = [f"{c}_raw" for c in quantile_cols]
    cross_h_rows = []
    for h, sub in diag_input.groupby("horizon"):
        cross_h_rows.append(
            {"dimension": "horizon", "value": str(h),
             "crossing_rate_raw": crossing_rate(sub[raw_cols[0]], sub[raw_cols[1]], sub[raw_cols[2]])}
        )
    cross_c_rows = []
    for c, sub in diag_input.groupby("cat_id"):
        cross_c_rows.append(
            {"dimension": "category", "value": str(c),
             "crossing_rate_raw": crossing_rate(sub[raw_cols[0]], sub[raw_cols[1]], sub[raw_cols[2]])}
        )
    crossing_rates = pd.DataFrame(cross_h_rows + cross_c_rows)

    diagnostics = pd.concat([diag_h, diag_c], ignore_index=True)
    diagnostics = diagnostics.merge(crossing_rates, on=["dimension", "value"], how="left")

    # ------- p50 vs Phase 5 point predictions --------------------------- #
    p50_vs_point = pd.DataFrame()
    point_path = Path(point_predictions_path)
    if point_path.exists():
        point_preds = pd.read_parquet(point_path)
        # Only keep matching rows; inner-join enforces apples-to-apples.
        merged = predictions[
            ["origin_date", "id", "horizon", "target_date", "p50", "actual"]
        ].merge(
            point_preds[["origin_date", "id", "horizon", "target_date", "prediction"]]
                .rename(columns={"prediction": "lgbm_point"}),
            on=["origin_date", "id", "horizon", "target_date"],
            how="inner",
        )
        if not merged.empty:
            p50_long = merged[["origin_date", "id", "horizon", "target_date", "actual"]].copy()
            p50_long["model"] = "lightgbm_quantile_p50"
            p50_long["prediction"] = merged["p50"]
            point_long = merged[["origin_date", "id", "horizon", "target_date", "actual"]].copy()
            point_long["model"] = "lightgbm_point"
            point_long["prediction"] = merged["lgbm_point"]
            p50_vs_point = (
                score_by_group(
                    pd.concat([p50_long, point_long], ignore_index=True),
                    by=("model",),
                ).sort_values("WAPE").reset_index(drop=True)
            )
    else:
        print(f"NOTE: Phase 5 predictions not found at {point_path}; "
              f"skipping p50 vs lightgbm_point comparison.")

    # ------- persist ---------------------------------------------------- #
    out_models = Path(out_models)
    out_predictions = Path(out_predictions)
    out_scores = Path(out_scores)
    out_diagnostics = Path(out_diagnostics)
    for p in (out_models, out_predictions, out_scores, out_diagnostics):
        p.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "models": models,
        "quantiles": list(quantiles),
        "valid_window_days": valid_window_days,
        "backtest_origins": [d.isoformat() for d in backtest_dates],
    }
    joblib.dump(bundle, out_models)
    predictions.to_parquet(out_predictions, index=False)
    scores.to_csv(out_scores, index=False)
    diagnostics.to_csv(out_diagnostics, index=False)

    print()
    print(f"wrote {out_models}")
    print(f"wrote {out_predictions}  ({len(predictions):,} rows)")
    print(f"wrote {out_scores}        ({len(scores):,} rows)")
    print(f"wrote {out_diagnostics}   ({len(diagnostics):,} rows)")
    print()

    if not p50_vs_point.empty:
        print("p50 vs lightgbm_point (sorted by WAPE):")
        print(p50_vs_point.to_string(index=False))
    print()
    print("uncertainty diagnostics by horizon:")
    print(diagnostics[diagnostics["dimension"] == "horizon"].to_string(index=False))

    return {
        "predictions": predictions,
        "scores": scores,
        "diagnostics": diagnostics,
        "p50_vs_point": p50_vs_point,
        "models": models,
    }


def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6 quantile LightGBM trainer.")
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
