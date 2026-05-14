"""Worst under/over-forecasters for a chosen model.

Ranks ids by signed cumulative error::

    signed_error = sum(prediction - actual)

* Most-negative signed error -> the model consistently under-forecasts that id.
* Most-positive signed error -> the model consistently over-forecasts that id.

These lists let us look at concrete items (with their cat/dept/segment
context) where the model is failing, rather than reasoning only about
aggregates.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def worst_forecasters(
    preds: pd.DataFrame,
    model: str,
    *,
    direction: Literal["under", "over"] = "under",
    top_n: int = 20,
    extra_cols: tuple[str, ...] = ("cat_id", "dept_id"),
) -> pd.DataFrame:
    """Top-N worst-forecast ids for one model.

    Parameters
    ----------
    preds
        Long DataFrame from
        :func:`seercast.diagnostics.combined.combined_predictions`. Must
        include ``model``, ``id``, ``prediction``, ``actual``.
    model
        Which model to rank.
    direction
        ``"under"`` -> ids with the most-negative signed error (model
        too low). ``"over"`` -> ids with the most-positive signed error
        (model too high).
    top_n
        How many ids to return.
    extra_cols
        Optional metadata columns (joined per id from the first row in
        ``preds`` for that id). ``segment_at_origin`` is included if
        present in ``preds``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``id``, ``n_rows``, ``total_actual``, ``total_pred``,
        ``signed_error``, ``abs_error``, ``wape_id`` (per-id WAPE), plus
        any ``extra_cols`` and ``segment_at_origin`` if present.
        Sorted by ``signed_error`` ascending (under) or descending (over).
    """
    if direction not in ("under", "over"):
        raise ValueError(f"direction must be 'under' or 'over', got {direction!r}")

    sub = preds.loc[preds["model"] == model]
    if sub.empty:
        return pd.DataFrame(
            columns=["id", "n_rows", "total_actual", "total_pred",
                     "signed_error", "abs_error", "wape_id"]
        )

    agg = (
        sub.groupby("id", sort=False)
        .agg(
            n_rows=("prediction", "size"),
            total_actual=("actual", "sum"),
            total_pred=("prediction", "sum"),
            abs_error=("prediction", lambda p: float(np.sum(np.abs(p.values - sub.loc[p.index, "actual"].values)))),
        )
        .reset_index()
    )
    agg["signed_error"] = agg["total_pred"] - agg["total_actual"]
    agg["wape_id"] = np.where(
        agg["total_actual"].abs() > 0,
        agg["abs_error"] / agg["total_actual"].abs(),
        np.nan,
    )

    # Attach extra metadata columns and segment_at_origin.
    meta_cols: list[str] = []
    for c in (*extra_cols, "segment_at_origin"):
        if c in sub.columns and c not in meta_cols:
            meta_cols.append(c)
    if meta_cols:
        # Take the first observed value per id for each meta column. For
        # segment_at_origin this is OK because it doesn't usually change
        # across origins for the same id; if it does, the breakdown table
        # already shows the per-origin truth.
        meta = sub.drop_duplicates("id")[["id", *meta_cols]]
        agg = agg.merge(meta, on="id", how="left")

    asc = direction == "under"
    agg = agg.sort_values("signed_error", ascending=asc).head(top_n).reset_index(drop=True)
    return agg


__all__ = ["worst_forecasters"]
