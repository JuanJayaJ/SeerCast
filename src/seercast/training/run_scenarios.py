"""Phase 7 entry point: predictive scenario simulation.

Reads:
    data/processed/train_features_ca1_full_horizon.parquet
    outputs/models/lightgbm_quantile_models_ca1.pkl

Writes:
    outputs/reports/scenario_forecasts_ca1.parquet
    outputs/reports/scenario_comparison_ca1.csv

Run::

    python -m seercast.training.run_scenarios

The script picks the most-recent backtest origin from the bundled
quantile model and runs the default scenario set
(:func:`seercast.scenario.default_scenarios`) over every id at that
origin. Override via ``--origin`` and ``--ids``.

Reminder: scenario simulation is *predictive*, not causal. Outputs answer
"what does the model predict if this input changes?" -- not "this input
caused the demand change."
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
    ensure_dirs,
)
from seercast.evaluation.backtesting import origin_to_date
from seercast.scenario import (
    default_scenarios,
    load_quantile_models_bundle,
    pick_quantile_model_for_origin,
    scenario_summary,
    simulate_scenarios,
)


def _resolve_origin(
    base_path: Path,
    full_horizon_features: pd.DataFrame,
    bundle: dict,
    origin: int | str | pd.Timestamp | None,
) -> pd.Timestamp:
    """Coerce the ``origin`` arg (int / str / Timestamp / None) to a date."""
    if origin is None:
        # Default to the most-recent backtest origin we have a model for.
        if bundle.get("backtest_origins"):
            return pd.Timestamp(bundle["backtest_origins"][-1])
        # Fallback: latest configured backtest origin.
        if base_path.exists():
            base = pd.read_parquet(base_path, columns=["id", "d", "date"])
            return pd.Timestamp(origin_to_date(base, BACKTEST.origins[-1]))
        return pd.Timestamp(full_horizon_features["origin_date"].max())

    if isinstance(origin, (int, str)):
        if base_path.exists():
            base = pd.read_parquet(base_path, columns=["id", "d", "date"])
            return pd.Timestamp(origin_to_date(base, origin))
        # Fallback: try direct date parse.
        return pd.Timestamp(origin)
    return pd.Timestamp(origin)


def run(
    full_horizon_path: Path | str = ARTIFACTS.train_features_ca1_full_horizon,
    quantile_bundle_path: Path | str = ARTIFACTS.lightgbm_quantile_models_ca1,
    *,
    base_path: Path | str = ARTIFACTS.base_table_ca1,
    origin: int | str | pd.Timestamp | None = None,
    ids: Sequence[str] | None = None,
    cat_id: str | Sequence[str] | None = None,
    store_id: str | Sequence[str] | None = None,
    horizons: Sequence[int] | None = None,
    scenarios: dict | None = None,
    out_forecasts: Path | str = ARTIFACTS.scenario_forecasts_ca1,
    out_summary: Path | str = ARTIFACTS.scenario_comparison_ca1,
) -> dict:
    """Run the simulator and persist the two deliverables.

    Returns a dict ``{"forecasts": <df>, "summary": <df>, "origin_date": <Timestamp>}``.
    """
    ensure_dirs()

    full = pd.read_parquet(full_horizon_path)
    print(f"loaded full-horizon table: {full.shape[0]:,} rows x {full.shape[1]} cols")

    bundle = load_quantile_models_bundle(quantile_bundle_path)
    origin_date = _resolve_origin(Path(base_path), full, bundle, origin)
    print(f"simulating at origin: {origin_date.date()}")

    quantile_model = pick_quantile_model_for_origin(bundle, origin_date)
    if scenarios is None:
        scenarios = default_scenarios()
    print(f"running {len(scenarios)} scenarios: {list(scenarios)}")

    forecasts = simulate_scenarios(
        full_horizon_features=full,
        quantile_model=quantile_model,
        scenarios=scenarios,
        origin_date=origin_date,
        ids=ids,
        cat_id=cat_id,
        store_id=store_id,
        horizons=horizons,
    )
    print(f"produced {len(forecasts):,} prediction rows "
          f"({forecasts['scenario'].nunique()} scenarios x "
          f"{forecasts['id'].nunique()} ids x "
          f"{forecasts['horizon'].nunique()} horizons)")

    summary = scenario_summary(forecasts)

    out_forecasts = Path(out_forecasts)
    out_summary = Path(out_summary)
    out_forecasts.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(out_forecasts, index=False)
    summary.to_csv(out_summary, index=False)

    print()
    print(f"wrote {out_forecasts}  ({len(forecasts):,} rows)")
    print(f"wrote {out_summary}    ({len(summary):,} rows)")
    print()
    print("scenario summary (sorted by delta_total_p50_pct desc):")
    print(summary.to_string(index=False))

    return {"forecasts": forecasts, "summary": summary, "origin_date": origin_date}


def _parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 7 predictive scenario simulator (NOT causal)."
    )
    p.add_argument("--full-horizon", type=Path,
                   default=ARTIFACTS.train_features_ca1_full_horizon)
    p.add_argument("--bundle", type=Path,
                   default=ARTIFACTS.lightgbm_quantile_models_ca1)
    p.add_argument("--base", type=Path, default=ARTIFACTS.base_table_ca1)
    p.add_argument(
        "--origin", default=None,
        help="Origin spec (M5 d-int, 'd_N', or YYYY-MM-DD). "
             "Defaults to the latest backtest origin in the model bundle."
    )
    p.add_argument(
        "--ids", default=None,
        help="Comma-separated list of ids to restrict to (default: all).",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    ids = args.ids.split(",") if args.ids else None
    origin = args.origin
    if origin and origin.isdigit():
        origin = int(origin)
    run(
        full_horizon_path=args.full_horizon,
        quantile_bundle_path=args.bundle,
        base_path=args.base,
        origin=origin,
        ids=ids,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
