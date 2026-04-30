"""SeerCast: drift-aware multi-horizon retail demand forecasting.

This package implements a portfolio-grade forecasting system over the M5
dataset. The top-level package re-exports the most commonly used data
loaders so notebooks and scripts can do::

    from seercast import load_m5_raw, build_base_table

Anything beyond loading lives in submodules (``seercast.features``,
``seercast.models``, ``seercast.evaluation``, ``seercast.scenario``).
"""

from seercast.data.load_m5 import (
    load_calendar,
    load_m5_raw,
    load_prices,
    load_sales,
)
from seercast.data.transform import build_base_table, melt_sales

__all__ = [
    "load_calendar",
    "load_sales",
    "load_prices",
    "load_m5_raw",
    "melt_sales",
    "build_base_table",
]

__version__ = "0.1.0"
