"""Predictive scenario simulator.

Runs the trained quantile LightGBM model twice for every scenario:

1. once on the unmodified feature slice -> ``base_p10/p50/p90``;
2. once on a feature slice with the scenario's adjustments applied ->
   ``scenario_p10/p50/p90``.

Then attaches deltas via :func:`seercast.scenario.comparison.add_delta_columns`.

Important framing again, in module form:

    Scenario simulation answers "what does the model predict if this input
    changes?" -- not "this input caused the demand change." The simulator
    is mechanical: it perturbs features and asks the same model again.
    Causal inference would require a different design (e.g. randomized
    interventions, instrumental variables, do-calculus). We don't claim
    that here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd

from seercast.models.quantile_lightgbm import (
    QuantileLightGBMModel,
    quantile_column_name,
)
from seercast.scenario.adjustments import Adjustment
from seercast.scenario.comparison import add_delta_columns


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def _coerce_iterable(x) -> list | None:
    """Normalize ``None`` / scalar / iterable to ``None`` or a list."""
    if x is None:
        return None
    if isinstance(x, (str, bytes)) or not isinstance(x, Iterable):
        return [x]
    return list(x)


def select_rows(
    full_horizon_features: pd.DataFrame,
    *,
    origin_date: pd.Timestamp,
    ids: Sequence[str] | None = None,
    cat_id: str | Sequence[str] | None = None,
    store_id: str | Sequence[str] | None = None,
    horizons: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Slice the full-horizon feature table for a single ``origin_date``.

    Filters by any combination of ``ids``, ``cat_id``, ``store_id``, and
    ``horizons``. Always pins to the requested ``origin_date`` (so the
    simulator's "what-if at this origin" framing is unambiguous).

    Returns a copy with a fresh integer index.
    """
    df = full_horizon_features
    sel = df.loc[df["origin_date"] == pd.Timestamp(origin_date)]
    if sel.empty:
        raise ValueError(
            f"no rows for origin_date={origin_date!r} in the feature table; "
            f"available dates: {sorted(df['origin_date'].unique())[:5]} ..."
        )

    if (ids := _coerce_iterable(ids)) is not None:
        sel = sel.loc[sel["id"].isin(ids)]
    if (cat_id := _coerce_iterable(cat_id)) is not None:
        sel = sel.loc[sel["cat_id"].isin(cat_id)]
    if (store_id := _coerce_iterable(store_id)) is not None:
        sel = sel.loc[sel["store_id"].isin(store_id)]
    if (horizons := _coerce_iterable(horizons)) is not None:
        sel = sel.loc[sel["horizon"].isin([int(h) for h in horizons])]

    return sel.reset_index(drop=True).copy()


# --------------------------------------------------------------------------- #
# Model bundle helpers
# --------------------------------------------------------------------------- #


def load_quantile_models_bundle(path: Path | str) -> dict:
    """Load the joblib bundle produced by Phase 6 train_quantile_lightgbm.run().

    Bundle structure: ``{"models": {origin_iso: QuantileLightGBMModel}, "quantiles": [...], ...}``.
    """
    return joblib.load(Path(path))


def pick_quantile_model_for_origin(
    bundle: dict,
    origin_date: pd.Timestamp,
) -> QuantileLightGBMModel:
    """Return the bundled quantile model whose backtest origin best matches ``origin_date``.

    Strategy:
    1. exact match by iso-string;
    2. otherwise the most-recent backtest origin <= ``origin_date``;
    3. otherwise the earliest available (with a warning printed once).
    """
    if "models" not in bundle:
        raise KeyError("bundle has no 'models' key (was it produced by Phase 6?)")
    models: dict[str, QuantileLightGBMModel] = bundle["models"]
    if not models:
        raise ValueError("bundle is empty -- no quantile models to pick from")

    target = pd.Timestamp(origin_date)
    if target.isoformat() in models:
        return models[target.isoformat()]

    parsed = sorted((pd.Timestamp(k), m) for k, m in models.items())
    earlier_or_eq = [(d, m) for d, m in parsed if d <= target]
    if earlier_or_eq:
        return earlier_or_eq[-1][1]

    print(
        f"NOTE: no bundled model with origin <= {target.date()}; "
        f"falling back to the earliest available ({parsed[0][0].date()})."
    )
    return parsed[0][1]


# --------------------------------------------------------------------------- #
# Core simulator
# --------------------------------------------------------------------------- #


_PRED_KEY_COLS: tuple[str, ...] = (
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "origin_date",
    "horizon",
    "target_date",
)


def _build_output_frame(
    selected: pd.DataFrame,
    base_preds: pd.DataFrame,
    scenario_preds: pd.DataFrame,
    scenario_name: str,
    quantile_columns: Sequence[str],
) -> pd.DataFrame:
    """Stitch base + scenario predictions onto the identity / meta columns."""
    out = pd.DataFrame(
        {col: selected[col].values for col in _PRED_KEY_COLS if col in selected.columns}
    )
    out.insert(0, "scenario", scenario_name)
    for q in quantile_columns:
        out[f"base_{q}"] = base_preds[q].values
    for q in quantile_columns:
        out[f"scenario_{q}"] = scenario_preds[q].values
    return out


def simulate_scenarios(
    full_horizon_features: pd.DataFrame,
    quantile_model: QuantileLightGBMModel,
    scenarios: dict[str, Adjustment],
    *,
    origin_date: pd.Timestamp,
    ids: Sequence[str] | None = None,
    cat_id: str | Sequence[str] | None = None,
    store_id: str | Sequence[str] | None = None,
    horizons: Sequence[int] | None = None,
    fix_crossings: bool = True,
) -> pd.DataFrame:
    """Run the simulator for every scenario in ``scenarios``.

    Parameters
    ----------
    full_horizon_features
        The Phase 4 full-horizon table (``train_features_ca1_full_horizon.parquet``).
    quantile_model
        A fitted :class:`QuantileLightGBMModel` (from the Phase 6 bundle).
    scenarios
        Mapping of scenario name -> ``df -> df`` callable. Use the
        adjustment functions in :mod:`seercast.scenario.adjustments` and
        ``functools.partial`` to bind their parameters.
    origin_date
        The origin to simulate from. Must exist in ``full_horizon_features``.
    ids, cat_id, store_id, horizons
        Optional row filters; all default to "everything".
    fix_crossings
        Whether to row-wise sort each prediction (recommended).

    Returns
    -------
    pandas.DataFrame
        Long: one row per (scenario, id, horizon) with columns
        ``scenario``, identity, ``origin_date``, ``horizon``, ``target_date``,
        ``base_p10/p50/p90``, ``scenario_p10/p50/p90``,
        ``delta_p10/p50/p90`` and ``delta_p10/p50/p90_pct``.

    The base predictions are computed *once* per call (not per scenario)
    and the same values are repeated for every scenario row -- so by
    construction "base remains unchanged when scenario is applied".
    """
    if not scenarios:
        raise ValueError("scenarios must be a non-empty dict")

    selected = select_rows(
        full_horizon_features,
        origin_date=origin_date,
        ids=ids,
        cat_id=cat_id,
        store_id=store_id,
        horizons=horizons,
    )
    if selected.empty:
        raise ValueError("selection produced 0 rows; check filters")

    quantile_cols = [quantile_column_name(q) for q in quantile_model.quantiles]

    base_preds = quantile_model.predict(selected, fix_crossings=fix_crossings)

    chunks: list[pd.DataFrame] = []
    for name, adjust in scenarios.items():
        adjusted = adjust(selected)
        if not isinstance(adjusted, pd.DataFrame):
            raise TypeError(
                f"scenario {name!r} adjustment must return a DataFrame; "
                f"got {type(adjusted).__name__}"
            )
        if len(adjusted) != len(selected):
            raise ValueError(
                f"scenario {name!r} adjustment changed row count "
                f"({len(selected)} -> {len(adjusted)})"
            )
        scenario_preds = quantile_model.predict(adjusted, fix_crossings=fix_crossings)
        chunks.append(
            _build_output_frame(selected, base_preds, scenario_preds, name, quantile_cols)
        )

    out = pd.concat(chunks, ignore_index=True)
    out = add_delta_columns(out, quantile_columns=quantile_cols)
    return out


__all__ = [
    "select_rows",
    "load_quantile_models_bundle",
    "pick_quantile_model_for_origin",
    "simulate_scenarios",
]
