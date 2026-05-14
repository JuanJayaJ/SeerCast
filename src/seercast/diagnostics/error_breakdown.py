"""Per-group accuracy breakdowns on the matched grid.

Thin wrappers over :func:`seercast.evaluation.metrics.score_by_group` that
add a couple of project-specific niceties:

* Derive a ``target_zero`` flag (``actual == 0``) on the fly so we can
  split errors across "rare-spike" vs "routine" rows.
* :func:`fairness_check` reports per-(group, model) row counts so the
  notebook can confirm every model is scored on the same slots within
  every breakdown bucket.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from seercast.evaluation.metrics import score_by_group


def _ensure_target_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``target_zero`` if missing. Returns ``df`` unchanged otherwise."""
    if "target_zero" in df.columns:
        return df
    out = df.copy()
    out["target_zero"] = (out["actual"] == 0).astype("int8")
    return out


def breakdown(
    preds: pd.DataFrame,
    by: str | Sequence[str],
) -> pd.DataFrame:
    """MAE / RMSE / WAPE / Bias per (by) x model.

    Parameters
    ----------
    preds
        Long DataFrame from
        :func:`seercast.diagnostics.combined.combined_predictions`.
    by
        Column(s) to group by, *excluding* model. ``model`` is always added.

    Returns
    -------
    pandas.DataFrame
        Columns: by-columns, ``model``, ``n``, ``MAE``, ``RMSE``, ``WAPE``, ``Bias``.
        Sorted by the by-columns then by WAPE ascending so the per-group winner
        sits at the top of each group.
    """
    if isinstance(by, str):
        by_cols = [by]
    else:
        by_cols = list(by)
    if "model" in by_cols:
        raise ValueError("`model` is always included automatically; remove it from `by`")

    df = _ensure_target_zero(preds) if "target_zero" in by_cols else preds
    out = score_by_group(df, by=tuple(by_cols + ["model"]))
    # score_by_group sorts by the group keys; re-sort so WAPE asc within group.
    out = (
        out.sort_values(by_cols + ["WAPE"], ascending=[True] * len(by_cols) + [True])
        .reset_index(drop=True)
    )
    return out


def fairness_check(preds: pd.DataFrame, by: str | Sequence[str]) -> pd.DataFrame:
    """Per-(group, model) row counts. Useful to verify equal `n` within each group.

    Output is a wide table: rows = group keys, columns = model names, values = ``n``.
    Any cell where the counts differ across columns flags an uneven slice.
    """
    if isinstance(by, str):
        by_cols = [by]
    else:
        by_cols = list(by)
    df = _ensure_target_zero(preds) if "target_zero" in by_cols else preds
    counts = (
        df.groupby(by_cols + ["model"], dropna=False)
        .size()
        .unstack("model", fill_value=0)
    )
    return counts


__all__ = ["breakdown", "fairness_check"]
