"""Feature engineering: calendar, demand, price, lifecycle, and supervised-table builders."""

from seercast.features.calendar_features import (
    CALENDAR_FEATURE_NAMES,
    add_calendar_features,
    calendar_feature_columns,
)
from seercast.features.demand_features import (
    DEMAND_FEATURE_NAMES,
    add_demand_features,
    demand_feature_columns,
)
from seercast.features.lifecycle_features import (
    LIFECYCLE_FEATURE_NAMES,
    add_lifecycle_features,
    lifecycle_feature_columns,
)
from seercast.features.price_features import (
    PRICE_FEATURE_NAMES,
    add_price_features,
    price_feature_columns,
)
from seercast.features.supervised import (
    IDENTITY_COLUMNS,
    META_COLUMNS,
    ORIGIN_FEATURE_COLUMNS,
    SUPERVISED_COLUMNS,
    SupervisedValidationReport,
    TARGET_FEATURE_COLUMNS,
    build_supervised_table,
    default_training_origins,
    validate_supervised_table,
)

__all__ = [
    "CALENDAR_FEATURE_NAMES",
    "add_calendar_features",
    "calendar_feature_columns",
    "DEMAND_FEATURE_NAMES",
    "add_demand_features",
    "demand_feature_columns",
    "PRICE_FEATURE_NAMES",
    "add_price_features",
    "price_feature_columns",
    "LIFECYCLE_FEATURE_NAMES",
    "add_lifecycle_features",
    "lifecycle_feature_columns",
    "IDENTITY_COLUMNS",
    "ORIGIN_FEATURE_COLUMNS",
    "TARGET_FEATURE_COLUMNS",
    "META_COLUMNS",
    "SUPERVISED_COLUMNS",
    "build_supervised_table",
    "default_training_origins",
    "SupervisedValidationReport",
    "validate_supervised_table",
]
