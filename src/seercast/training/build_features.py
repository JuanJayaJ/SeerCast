"""Phase 4 entry point: build the supervised feature tables from the base table.

Reads:
    data/interim/m5_base_ca1.parquet

Writes:
    data/processed/train_features_ca1.parquet              (horizons [1,7,14,28])
    data/processed/train_features_ca1_full_horizon.parquet (horizons 1..28)

Run::

    python -m seercast.training.build_features

Or import :func:`run` from a notebook.

This script calls ``validate_base_table(strict=True)`` on input and
``validate_supervised_table(strict=True)`` on output, so any join /
schema / leakage-contract issue aborts the run instead of silently
producing a broken training table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from seercast.config import (
    ARTIFACTS,
    BACKTEST,
    DEFAULT_STATE_ID,
    DIRECT_HORIZONS,
    FULL_HORIZONS,
    PROCESSED_DIR,
    ensure_dirs,
)
from seercast.data.validation import validate_base_table
from seercast.features.supervised import (
    build_supervised_table,
    validate_supervised_table,
)


def _build_one(
    base: pd.DataFrame,
    horizons,
    out_path: Path,
    *,
    snap_state: str,
    origin_step_days: int,
    label: str,
    must_include_origins=None,
) -> pd.DataFrame:
    print(f"[{label}] building supervised table for horizons {list(horizons)} ...")
    sup = build_supervised_table(
        base=base,
        horizons=horizons,
        snap_state=snap_state,
        origin_step_days=origin_step_days,
        must_include_origins=must_include_origins,
    )
    print(f"[{label}] built {len(sup):,} rows; validating ...")
    report = validate_supervised_table(sup, strict=True)
    print(report.summary())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sup.to_parquet(out_path, index=False)
    print(f"[{label}] wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    return sup


def run(
    base_path: Path | str = ARTIFACTS.base_table_ca1,
    *,
    snap_state: str = DEFAULT_STATE_ID,
    origin_step_days: int = 7,
    must_include_origins=BACKTEST.origins,
    out_direct: Path | str = ARTIFACTS.train_features_ca1,
    out_full: Path | str = ARTIFACTS.train_features_ca1_full_horizon,
) -> dict[str, pd.DataFrame]:
    """Build both supervised tables (direct horizons + full horizon).

    Returns a dict ``{"direct": <df>, "full": <df>}``.
    """
    ensure_dirs()
    base = pd.read_parquet(base_path)
    print("validating base table ...")
    base_report = validate_base_table(base, strict=True)
    print(base_report.summary())
    print()

    direct = _build_one(
        base, DIRECT_HORIZONS, Path(out_direct),
        snap_state=snap_state, origin_step_days=origin_step_days, label="direct",
        must_include_origins=must_include_origins,
    )
    print()
    full = _build_one(
        base, FULL_HORIZONS, Path(out_full),
        snap_state=snap_state, origin_step_days=origin_step_days, label="full",
        must_include_origins=must_include_origins,
    )
    return {"direct": direct, "full": full}


def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4 supervised feature builder.")
    p.add_argument("--base", type=Path, default=ARTIFACTS.base_table_ca1,
                   help="Path to the base table parquet.")
    p.add_argument("--snap-state", default=DEFAULT_STATE_ID,
                   help="Two-letter state for the SNAP flag (default: CA).")
    p.add_argument("--origin-step", type=int, default=7,
                   help="Days between consecutive origin dates (default: 7 = weekly).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    run(
        base_path=args.base,
        snap_state=args.snap_state,
        origin_step_days=args.origin_step,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
