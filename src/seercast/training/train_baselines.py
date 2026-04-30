"""Phase 3 entry point: run all baselines through the rolling-origin backtester.

Reads:
    data/interim/m5_base_ca1.parquet

Writes:
    outputs/reports/baseline_predictions_ca1.parquet  (long predictions + actuals)
    outputs/reports/baseline_scores_ca1.csv            (MAE/RMSE/WAPE/Bias by model x horizon)
    outputs/reports/baseline_summary_ca1.csv           (overall MAE/RMSE/WAPE/Bias by model, ranked by WAPE)

Run::

    python -m seercast.training.train_baselines

Or import :func:`run` from a notebook/test if you want predictions in memory.

Note on validation: this script calls ``validate_base_table(strict=True)``;
a bad join will hard-fail before any model touches the data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from seercast.config import (
    ARTIFACTS,
    BACKTEST,
    FULL_HORIZONS,
    REPORTS_DIR,
    ensure_dirs,
)
from seercast.data.validation import validate_base_table
from seercast.evaluation.backtesting import rolling_origin_backtest
from seercast.evaluation.metrics import score_by_group
from seercast.models.baselines import all_baselines


def run(
    base_path: Path | str = ARTIFACTS.base_table_ca1,
    origins: Sequence[int] = BACKTEST.origins,
    horizons: Sequence[int] = FULL_HORIZONS,
    out_predictions: Path | str = ARTIFACTS.baseline_predictions_ca1,
    out_scores: Path | str = ARTIFACTS.baseline_scores_ca1,
    out_summary: Path | str = ARTIFACTS.baseline_summary_ca1,
) -> dict[str, pd.DataFrame]:
    """Backtest all baselines and persist predictions + scores.

    Returns a dict with keys ``predictions``, ``scores`` (per model x horizon),
    and ``summary`` (per model, sorted by WAPE).
    """
    ensure_dirs()

    base = pd.read_parquet(base_path)

    # Hard-fail on bad joins before any model touches the data.
    report = validate_base_table(base, strict=True)
    print(report.summary())
    print()

    factories = all_baselines()
    print(f"running {len(factories)} baselines x {len(origins)} origins "
          f"x {len(horizons)} horizons ...")

    all_preds: list[pd.DataFrame] = []
    for name, factory in factories.items():
        print(f"  - {name}")
        preds = rolling_origin_backtest(
            base=base,
            model_factory=factory,
            origins=origins,
            horizons=horizons,
            model_name=name,
        )
        all_preds.append(preds)

    predictions = pd.concat(all_preds, ignore_index=True)

    # Per (model, horizon) — useful for "where does each baseline break down?"
    scores = score_by_group(predictions, by=("model", "horizon"))

    # Overall per model, sorted by WAPE (the primary ranking metric).
    summary = score_by_group(predictions, by=("model",)).sort_values("WAPE").reset_index(drop=True)

    # Persist.
    out_predictions = Path(out_predictions)
    out_scores = Path(out_scores)
    out_summary = Path(out_summary)
    out_predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(out_predictions, index=False)
    scores.to_csv(out_scores, index=False)
    summary.to_csv(out_summary, index=False)

    print()
    print(f"wrote {out_predictions}  ({len(predictions):,} rows)")
    print(f"wrote {out_scores}        ({len(scores):,} rows)")
    print(f"wrote {out_summary}       ({len(summary):,} rows)")
    print()
    print("baseline ranking by WAPE:")
    print(summary.to_string(index=False))

    return {"predictions": predictions, "scores": scores, "summary": summary}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 baseline backtest runner.")
    p.add_argument("--base", type=Path, default=ARTIFACTS.base_table_ca1,
                   help="Path to the base table parquet.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run(base_path=args.base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
