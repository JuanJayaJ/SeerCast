"""Forecasting models: baselines (Phase 3), point LightGBM (Phase 5), quantile LightGBM (Phase 6)."""

from seercast.models.baselines import (
    Baseline,
    MovingAverageBaseline,
    NaiveBaseline,
    SeasonalMovingAverageBaseline,
    SeasonalNaiveBaseline,
    all_baselines,
)

__all__ = [
    "Baseline",
    "NaiveBaseline",
    "SeasonalNaiveBaseline",
    "MovingAverageBaseline",
    "SeasonalMovingAverageBaseline",
    "all_baselines",
]
