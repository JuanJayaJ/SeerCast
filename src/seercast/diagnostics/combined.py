"""Combined predictions across all models on the matched evaluation grid.

This is the input frame for every accuracy breakdown. Construction:

1. Load LightGBM point predictions -- their ``(origin_date, id, horizon,
   target_date)`` keys define the canonical grid.
2. Inner-join baseline predictions onto that grid (same fairness fix as
   :func:`seercast.training.train_lightgbm.build_model_comparison`).
3. Inner-join the quantile predictions onto the grid and emit them as a
   synthetic ``model="lightgbm_quantile_p50"`` slice -- using ``p50`` as
   ``prediction``. This is the explicit "p50 as first-class model"
   design choice for diagnostics.
4. Attach ``cat_id`` and ``dept_id`` from the base table.
5. Attach ``segment_at_origin`` and ``is_active`` from the diagnostics
   modules.
6. Verify every model has the same ``n`` on the matched grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


GRID_COLS: tuple[str, ...] = ("origin_date", "id", "horizon", "target_date")
PRED_COLS: tuple[str, ...] = ("model", *GRID_COLS, "prediction", "actual")


def _to_long_predictions(quantile_preds: pd.DataFrame) -> pd.DataFrame:
    """Reshape quantile predictions into the canonical long format.

    Quantile preds have a ``p50`` column (and p10/p90, and raw variants).
    We emit one row per (origin_date, id, horizon, target_date) with
    ``model="lightgbm_quantile_p50"`` and ``prediction = p50``.
    """
    needed = {"p50", "actual", *GRID_COLS}
    missing = needed - set(quantile_preds.columns)
    if missing:
        raise KeyError(f"quantile preds missing columns: {missing}")

    out = quantile_preds.loc[:, list(GRID_COLS) + ["p50", "actual"]].copy()
    out = out.rename(columns={"p50": "prediction"})
    out.insert(0, "model", "lightgbm_quantile_p50")
    return out[list(PRED_COLS)]


def _filter_to_grid(preds: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Inner-join ``preds`` onto ``grid`` by the canonical grid columns."""
    return preds.merge(grid, on=list(GRID_COLS), how="inner")


def combined_predictions(
    lightgbm_point_path: Path | str,
    *,
    quantile_predictions_path: Path | str | None = None,
    baseline_predictions_path: Path | str | None = None,
    base_table_path: Path | str | None = None,
    segments_at_origin: pd.DataFrame | None = None,
    is_active_at_origin_df: pd.DataFrame | None = None,
    enforce_equal_n: bool = True,
) -> pd.DataFrame:
    """Build the long predictions frame used by every diagnostic breakdown.

    Parameters
    ----------
    lightgbm_point_path
        Path to ``lightgbm_backtest_predictions_ca1.parquet``. Defines the
        canonical evaluation grid.
    quantile_predictions_path
        Path to ``quantile_backtest_predictions_ca1.parquet``. Adds
        ``lightgbm_quantile_p50`` as a first-class model row. Optional.
    baseline_predictions_path
        Path to ``baseline_predictions_ca1.parquet``. Inner-joined onto
        the LGBM grid (matches the Phase 5 fairness fix). Optional.
    base_table_path
        Path to ``m5_base_ca1.parquet``. If supplied, ``cat_id`` and
        ``dept_id`` are joined on ``id``.
    segments_at_origin
        Output of
        :func:`seercast.diagnostics.demand_segments.classify_demand_per_origin`.
        If supplied, ``segment_at_origin`` and the per-origin demand
        statistics are joined on ``(origin_date, id)``.
    is_active_at_origin_df
        Output of :func:`seercast.diagnostics.lifecycle.is_active_at_origin`.
        If supplied, ``is_active`` is joined on ``(origin_date, id)``.
    enforce_equal_n
        If True (default), assert that every model in the result has the
        same row count.

    Returns
    -------
    pandas.DataFrame
        Long format with columns ``model, origin_date, id, horizon,
        target_date, prediction, actual`` plus optional joins for
        ``cat_id``, ``dept_id``, ``segment_at_origin``,
        ``adi_at_origin``, ``cv2_at_origin``, ``zero_rate_at_origin``,
        ``mean_demand_at_origin``, ``is_active``,
        ``first_sale_date_at_origin``.
    """
    lgbm = pd.read_parquet(lightgbm_point_path)
    required = {"model", *GRID_COLS, "prediction", "actual"}
    missing = required - set(lgbm.columns)
    if missing:
        raise KeyError(f"lightgbm preds missing columns: {missing}")

    grid = lgbm.loc[:, list(GRID_COLS)].drop_duplicates()
    chunks: list[pd.DataFrame] = [lgbm.loc[:, list(PRED_COLS)]]

    if baseline_predictions_path is not None and Path(baseline_predictions_path).exists():
        bl = pd.read_parquet(baseline_predictions_path)
        bl_matched = _filter_to_grid(bl.loc[:, list(PRED_COLS)], grid)
        chunks.append(bl_matched)

    if quantile_predictions_path is not None and Path(quantile_predictions_path).exists():
        q = pd.read_parquet(quantile_predictions_path)
        q_long = _to_long_predictions(q)
        q_matched = _filter_to_grid(q_long, grid)
        chunks.append(q_matched)

    combined = pd.concat(chunks, ignore_index=True)

    # Fairness invariant.
    if enforce_equal_n:
        counts = combined.groupby("model").size()
        if counts.nunique() != 1:
            raise ValueError(
                "matched grid is uneven across models -- "
                "different `n` per model breaks like-for-like comparison:\n"
                f"{counts.to_string()}"
            )

    # Optional joins.
    if base_table_path is not None and Path(base_table_path).exists():
        cat_dept = (
            pd.read_parquet(base_table_path, columns=["id", "cat_id", "dept_id"])
            .drop_duplicates("id")
        )
        combined = combined.merge(cat_dept, on="id", how="left")

    if segments_at_origin is not None and not segments_at_origin.empty:
        seg_keep = [
            "origin_date", "id", "segment_at_origin",
            "adi_at_origin", "cv2_at_origin",
            "zero_rate_at_origin", "mean_demand_at_origin",
        ]
        keep = [c for c in seg_keep if c in segments_at_origin.columns]
        combined = combined.merge(
            segments_at_origin.loc[:, keep], on=["origin_date", "id"], how="left"
        )

    if is_active_at_origin_df is not None and not is_active_at_origin_df.empty:
        keep = [c for c in ("origin_date", "id", "is_active", "first_sale_date_at_origin")
                if c in is_active_at_origin_df.columns]
        combined = combined.merge(
            is_active_at_origin_df.loc[:, keep], on=["origin_date", "id"], how="left"
        )

    return combined


__all__ = ["GRID_COLS", "PRED_COLS", "combined_predictions"]
