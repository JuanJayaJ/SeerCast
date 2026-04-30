"""Point-forecast metrics: MAE, RMSE, WAPE, Bias.

Definitions (from the project charter):

* WAPE = sum(|y - yhat|) / sum(|y|)
* Bias = sum(yhat - y) / sum(y)

WAPE is the primary ranking metric. Bias is signed and tells you whether
the model systematically over- or under-forecasts.

All scoring is array-friendly. ``score_by_group`` is a small convenience
helper that takes a long predictions DataFrame and emits one row of
metrics per slice (e.g. by horizon, by model).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Scalar metric primitives
# --------------------------------------------------------------------------- #


def _to_float_array(x: object) -> np.ndarray:
    return np.asarray(x, dtype=float)


def mae(y_true: object, y_pred: object) -> float:
    """Mean absolute error."""
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: object, y_pred: object) -> float:
    """Root mean squared error."""
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def wape(y_true: object, y_pred: object) -> float:
    """Weighted absolute percentage error: sum(|y - yhat|) / sum(|y|).

    Returns NaN if sum(|y|) is zero (no signal to normalize against).
    """
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    denom = float(np.sum(np.abs(yt)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(np.abs(yt - yp)) / denom)


def bias(y_true: object, y_pred: object) -> float:
    """Signed bias: sum(yhat - y) / sum(y).

    Positive => over-forecasting on average. Returns NaN if sum(y) is 0.
    """
    yt = _to_float_array(y_true)
    yp = _to_float_array(y_pred)
    denom = float(np.sum(yt))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(yp - yt) / denom)


# --------------------------------------------------------------------------- #
# Convenience: all four at once
# --------------------------------------------------------------------------- #


def all_point_metrics(y_true: object, y_pred: object) -> dict[str, float]:
    """Return the four metrics in a dict keyed ``MAE`` / ``RMSE`` / ``WAPE`` / ``Bias``."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
        "Bias": bias(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# Group-wise scoring
# --------------------------------------------------------------------------- #


def score_by_group(
    df: pd.DataFrame,
    by: Sequence[str] | None = None,
    y_col: str = "actual",
    yhat_col: str = "prediction",
) -> pd.DataFrame:
    """Compute MAE/RMSE/WAPE/Bias per group.

    Parameters
    ----------
    df
        Long DataFrame with at least ``y_col`` and ``yhat_col``. Rows where
        ``actual`` is NaN are dropped before scoring (they correspond to
        forecast horizons past the end of the test window).
    by
        Columns to group by. ``None`` or ``[]`` means score the whole frame
        as one group.
    y_col, yhat_col
        Column names in ``df``.

    Returns
    -------
    pandas.DataFrame
        One row per group, columns: ``*by``, ``n``, ``MAE``, ``RMSE``,
        ``WAPE``, ``Bias``.
    """
    if y_col not in df.columns or yhat_col not in df.columns:
        raise KeyError(
            f"score_by_group needs columns '{y_col}' and '{yhat_col}'; "
            f"got {list(df.columns)}"
        )

    clean = df.dropna(subset=[y_col, yhat_col])

    if not by:
        m = all_point_metrics(clean[y_col], clean[yhat_col])
        m["n"] = int(len(clean))
        return pd.DataFrame([m])[["n", "MAE", "RMSE", "WAPE", "Bias"]]

    by = list(by)
    rows: list[dict] = []
    for keys, sub in clean.groupby(by, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        m = all_point_metrics(sub[y_col], sub[yhat_col])
        m["n"] = int(len(sub))
        for col, val in zip(by, keys):
            m[col] = val
        rows.append(m)
    out = pd.DataFrame(rows)
    return out[by + ["n", "MAE", "RMSE", "WAPE", "Bias"]].sort_values(by).reset_index(drop=True)


__all__ = [
    "mae",
    "rmse",
    "wape",
    "bias",
    "all_point_metrics",
    "score_by_group",
]
