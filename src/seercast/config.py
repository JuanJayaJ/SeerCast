"""Project-wide configuration.

This module centralizes the small number of magic strings and constants that
the rest of the codebase depends on, so they can be changed in exactly one
place. Keep this file dependency-free (only stdlib) so it can be imported
from anywhere in the package, including tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# Repo root: this file lives at <repo>/src/seercast/config.py, so go up three.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DIR: Path = REPO_ROOT / "dataset"   # raw M5 CSVs live here; interim/processed stay under data/
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
MODELS_DIR: Path = OUTPUTS_DIR / "models"


# --------------------------------------------------------------------------- #
# M5 file names (raw)
# --------------------------------------------------------------------------- #

CALENDAR_FILE: str = "calendar.csv"
SALES_TRAIN_VALIDATION_FILE: str = "sales_train_validation.csv"
SALES_TRAIN_EVALUATION_FILE: str = "sales_train_evaluation.csv"
SELL_PRICES_FILE: str = "sell_prices.csv"
SAMPLE_SUBMISSION_FILE: str = "sample_submission.csv"


# --------------------------------------------------------------------------- #
# Modelling subset & horizons
# --------------------------------------------------------------------------- #

DEFAULT_STORE_ID: str = "CA_1"
DEFAULT_STATE_ID: str = "CA"

# In M5, snap_<state> is the "is SNAP day" flag. For CA_1 we map snap_CA -> is_snap_day.
SNAP_FLAG_BY_STATE: dict[str, str] = {
    "CA": "snap_CA",
    "TX": "snap_TX",
    "WI": "snap_WI",
}

FORECAST_HORIZON: int = 28
DIRECT_HORIZONS: list[int] = [1, 7, 14, 28]
FULL_HORIZONS: list[int] = list(range(1, FORECAST_HORIZON + 1))


# --------------------------------------------------------------------------- #
# Schema: the canonical columns of the joined "base table"
# --------------------------------------------------------------------------- #

BASE_TABLE_COLUMNS: list[str] = [
    "date",
    "d",
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "sales",
    "sell_price",
    "weekday",
    "wday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "wm_yr_wk",
]


# --------------------------------------------------------------------------- #
# Output filenames (kept in one place so notebooks/scripts agree)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Artifacts:
    """Canonical artifact paths used across phases."""

    base_table_ca1: Path = INTERIM_DIR / "m5_base_ca1.parquet"

    # Phase 4
    train_features_ca1: Path = PROCESSED_DIR / "train_features_ca1.parquet"
    train_features_ca1_full_horizon: Path = (
        PROCESSED_DIR / "train_features_ca1_full_horizon.parquet"
    )

    # Phase 3
    baseline_predictions_ca1: Path = REPORTS_DIR / "baseline_predictions_ca1.parquet"
    baseline_scores_ca1: Path = REPORTS_DIR / "baseline_scores_ca1.csv"
    baseline_summary_ca1: Path = REPORTS_DIR / "baseline_summary_ca1.csv"

    # Phase 5
    lightgbm_point_model_ca1: Path = MODELS_DIR / "lightgbm_point_model_ca1.pkl"
    lightgbm_backtest_predictions_ca1: Path = (
        REPORTS_DIR / "lightgbm_backtest_predictions_ca1.parquet"
    )
    lightgbm_backtest_scores_ca1: Path = REPORTS_DIR / "lightgbm_backtest_scores_ca1.csv"
    model_comparison_ca1: Path = REPORTS_DIR / "model_comparison_ca1.csv"

    # Phase 6
    lightgbm_quantile_models_ca1: Path = MODELS_DIR / "lightgbm_quantile_models_ca1.pkl"
    quantile_backtest_predictions_ca1: Path = (
        REPORTS_DIR / "quantile_backtest_predictions_ca1.parquet"
    )
    quantile_backtest_scores_ca1: Path = REPORTS_DIR / "quantile_backtest_scores_ca1.csv"
    uncertainty_diagnostics_ca1: Path = REPORTS_DIR / "uncertainty_diagnostics_ca1.csv"

    # Phase 7
    scenario_forecasts_ca1: Path = REPORTS_DIR / "scenario_forecasts_ca1.parquet"
    scenario_comparison_ca1: Path = REPORTS_DIR / "scenario_comparison_ca1.csv"


ARTIFACTS = Artifacts()


# --------------------------------------------------------------------------- #
# Backtesting defaults (used by Phase 3 onward)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class BacktestConfig:
    """Default rolling-origin backtest configuration.

    Origins are expressed as M5 ``d`` integers (e.g. 1500 means ``d_1500``).
    The training data must end at ``origin``; the forecast covers the next
    ``horizon`` days. Move the origin forward by ``step_days`` and repeat.
    """

    origins: tuple[int, ...] = (1500, 1556, 1612)
    horizon: int = FORECAST_HORIZON
    step_days: int = 56  # gap between consecutive origins (8 weeks)


BACKTEST = BacktestConfig()


# --------------------------------------------------------------------------- #
# Quantile levels for Phase 6
# --------------------------------------------------------------------------- #

QUANTILES: tuple[float, ...] = (0.10, 0.50, 0.90)


def ensure_dirs() -> None:
    """Create all output directories if they don't exist. Safe to call twice."""
    for p in (
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
        MODELS_DIR,
    ):
        p.mkdir(parents=True, exist_ok=True)


__all__ = [
    "REPO_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "INTERIM_DIR",
    "PROCESSED_DIR",
    "OUTPUTS_DIR",
    "FIGURES_DIR",
    "REPORTS_DIR",
    "MODELS_DIR",
    "CALENDAR_FILE",
    "SALES_TRAIN_VALIDATION_FILE",
    "SALES_TRAIN_EVALUATION_FILE",
    "SELL_PRICES_FILE",
    "SAMPLE_SUBMISSION_FILE",
    "DEFAULT_STORE_ID",
    "DEFAULT_STATE_ID",
    "SNAP_FLAG_BY_STATE",
    "FORECAST_HORIZON",
    "DIRECT_HORIZONS",
    "FULL_HORIZONS",
    "BASE_TABLE_COLUMNS",
    "Artifacts",
    "ARTIFACTS",
    "BacktestConfig",
    "BACKTEST",
    "QUANTILES",
    "ensure_dirs",
]
