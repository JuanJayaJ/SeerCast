"""Phase 2 entry point: build the joined base table from raw M5 CSVs.

Reads (from ``dataset/`` by default):
    calendar.csv
    sales_train_validation.csv      (or sales_train_evaluation.csv if --use-evaluation)
    sell_prices.csv

Writes:
    data/interim/m5_base_ca1.parquet

Pipeline: load M5 raw -> melt + calendar (LEFT) join + sell-price (LEFT)
join -> ``validate_base_table(strict=True)``. Strict validation aborts on
any join / schema / duplicate / missing-date error before any downstream
phase reads the file.

Run::

    python -m seercast.training.build_base_table
    python -m seercast.training.build_base_table --use-evaluation
    python -m seercast.training.build_base_table --store-id CA_2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from seercast.config import (
    ARTIFACTS,
    CALENDAR_FILE,
    DEFAULT_STORE_ID,
    RAW_DIR,
    SALES_TRAIN_EVALUATION_FILE,
    SALES_TRAIN_VALIDATION_FILE,
    SELL_PRICES_FILE,
    ensure_dirs,
)
from seercast.data.load_m5 import load_m5_raw
from seercast.data.transform import build_base_table
from seercast.data.validation import validate_base_table


# Files we always require. ``sales_train_evaluation.csv`` is added to the
# check when ``use_evaluation=True``.
_REQUIRED_RAW_FILES: tuple[str, ...] = (
    CALENDAR_FILE,
    SALES_TRAIN_VALIDATION_FILE,
    SELL_PRICES_FILE,
)


def _resolve_raw_dir(data_dir: Path | str | None) -> Path:
    """Coerce ``data_dir`` to the directory that *actually* contains the M5 CSVs.

    Accepts:
    * ``None`` -- use the package default (``<repo>/dataset``).
    * a path containing a ``dataset/`` subdir (e.g. the repo root) -- descended.
    * a path that *is* the dataset dir.

    Also handles the Kaggle-CLI layout where the competition zip extracts
    into a nested ``m5-forecasting-accuracy/`` subdir: if ``calendar.csv``
    isn't directly inside the chosen dir, descend one level into the first
    subdir that does contain it.
    """
    if data_dir is None:
        chosen = RAW_DIR
    else:
        p = Path(data_dir)
        chosen = p / "dataset" if (p / "dataset").exists() else p

    if (chosen / CALENDAR_FILE).exists():
        return chosen
    if chosen.is_dir():
        for child in sorted(chosen.iterdir()):
            if child.is_dir() and (child / CALENDAR_FILE).exists():
                return child
    return chosen


def _check_raw_files(raw_dir: Path, use_evaluation: bool) -> None:
    """Hard-fail with a friendly error if any required CSV is missing."""
    required = list(_REQUIRED_RAW_FILES)
    if use_evaluation:
        required.append(SALES_TRAIN_EVALUATION_FILE)
    missing = [f for f in required if not (raw_dir / f).exists()]
    if missing:
        msg = (
            f"Missing M5 file(s) in {raw_dir}:\n"
            + "\n".join(f"  - {f}" for f in missing)
            + "\n\nDownload them from the M5 Forecasting - Accuracy "
            "competition on Kaggle and place them in the dataset/ folder "
            "at the repo root. The files are not bundled with this repo."
        )
        raise FileNotFoundError(msg)


def run(
    data_dir: Path | str | None = None,
    *,
    store_id: str = DEFAULT_STORE_ID,
    use_evaluation: bool = False,
    out_path: Path | str = ARTIFACTS.base_table_ca1,
) -> pd.DataFrame:
    """Build and persist the base table for ``store_id``.

    Returns the in-memory DataFrame.

    Side effects: writes ``out_path`` (default ``data/interim/m5_base_ca1.parquet``).
    """
    ensure_dirs()

    raw_dir = _resolve_raw_dir(data_dir)
    _check_raw_files(raw_dir, use_evaluation)

    print(f"loading M5 raw frames from: {raw_dir}")
    raw = load_m5_raw(data_dir=raw_dir, use_evaluation=use_evaluation)
    print(f"  calendar : {raw.calendar.shape}")
    print(f"  sales    : {raw.sales.shape}")
    print(f"  prices   : {raw.prices.shape}")
    print()

    print(f"building base table for store_id={store_id!r} ...")
    base = build_base_table(raw=raw, store_ids=(store_id,))
    print(f"base shape: {base.shape}")
    print()

    print("validating (strict=True) ...")
    report = validate_base_table(base, strict=True)
    print(report.summary())
    print()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path}  ({size_mb:.1f} MB)")

    return base


def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 base-table builder.")
    p.add_argument(
        "--data-dir", type=Path, default=None,
        help="Repo root (containing dataset/) or the dataset/ dir directly. "
             "Defaults to <repo>/dataset.",
    )
    p.add_argument(
        "--store-id", default=DEFAULT_STORE_ID,
        help=f"Store filter (default: {DEFAULT_STORE_ID}).",
    )
    p.add_argument(
        "--use-evaluation", action="store_true",
        help="Load sales_train_evaluation.csv (extends 28 days further) "
             "instead of sales_train_validation.csv.",
    )
    p.add_argument(
        "--out", type=Path, default=ARTIFACTS.base_table_ca1,
        help=f"Output parquet path (default: {ARTIFACTS.base_table_ca1}).",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    run(
        data_dir=args.data_dir,
        store_id=args.store_id,
        use_evaluation=args.use_evaluation,
        out_path=args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
