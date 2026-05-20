"""Bootstrap confidence intervals on metric differences between two models.

Phase 8 — Credibility Pass.

Goal: turn "WAPE 0.696590 vs 0.727918" into a statement we can defend, e.g.
"WAPE difference −0.0313, 95% bootstrap CI [−0.0398, −0.0224]". A non-zero
CI that doesn't cross zero is evidence that the lift is robust to the
sampling of individual products; it is *not* a frequentist p-value and we
don't pretend it is.

Bootstrap strategy
------------------
We use a *clustered* bootstrap with the product ``id`` as the cluster: at
each iteration we draw ``n_clusters`` ids with replacement (where
``n_clusters`` is the number of unique ids in the matched grid), then
pull every (origin, horizon) row for those ids. This respects within-id
error correlation — a single product's three-origin × four-horizon block
shares much more error structure than rows across different products, so
treating rows as independent would understate the variance.

Matching is by ``(id, origin_date, horizon)``: only rows present in BOTH
prediction frames contribute. This guarantees apples-to-apples comparison
even if one model has more horizons (the baseline parquet covers 1..28,
the quantile parquet only covers 1/7/14/28).

The 95% CI is the percentile method (2.5/97.5 of the bootstrap
distribution). For thousands of clusters this is fine; we are not in BCa
territory here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from seercast.evaluation.metrics import all_point_metrics


DEFAULT_KEYS: tuple[str, ...] = ("id", "origin_date", "horizon")


# --------------------------------------------------------------------------- #
# Matched-grid join
# --------------------------------------------------------------------------- #


def matched_grid(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    keys: Sequence[str] = DEFAULT_KEYS,
    actual_col: str = "actual",
    pred_col_a: str = "prediction",
    pred_col_b: str = "prediction",
    label_a: str = "a",
    label_b: str = "b",
) -> pd.DataFrame:
    """Inner-join two prediction frames on ``keys`` and return a long frame
    with one row per matched cell and three value columns:
    ``actual``, ``pred_<label_a>``, ``pred_<label_b>``.

    The actual is taken from ``df_a``; if it differs from ``df_b``'s actual
    on any row the function raises (something is wrong upstream).
    """
    keys = list(keys)
    needed_a = [*keys, actual_col, pred_col_a]
    needed_b = [*keys, actual_col, pred_col_b]
    for col in needed_a:
        if col not in df_a.columns:
            raise KeyError(f"df_a missing column '{col}'")
    for col in needed_b:
        if col not in df_b.columns:
            raise KeyError(f"df_b missing column '{col}'")

    left = df_a[needed_a].rename(columns={
        actual_col: f"actual_{label_a}",
        pred_col_a: f"pred_{label_a}",
    })
    right = df_b[needed_b].rename(columns={
        actual_col: f"actual_{label_b}",
        pred_col_b: f"pred_{label_b}",
    })
    merged = left.merge(right, on=keys, how="inner")

    # Verify the two "actual" columns agree before we collapse them.
    a = merged[f"actual_{label_a}"].to_numpy(dtype=float)
    b = merged[f"actual_{label_b}"].to_numpy(dtype=float)
    if not np.allclose(a, b, equal_nan=True):
        n_disagree = int(np.sum(~np.isclose(a, b, equal_nan=True)))
        raise ValueError(
            f"actual columns disagree on {n_disagree} of {len(merged)} matched rows; "
            f"check upstream prediction pipelines."
        )

    merged["actual"] = merged[f"actual_{label_a}"]
    merged = merged.drop(columns=[f"actual_{label_a}", f"actual_{label_b}"])

    # Drop rows with NaN actual or either prediction.
    merged = merged.dropna(
        subset=["actual", f"pred_{label_a}", f"pred_{label_b}"]
    ).reset_index(drop=True)
    return merged


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #


@dataclass
class BootstrapResult:
    metric: str
    point_a: float
    point_b: float
    diff_point: float            # = point_a - point_b (a minus b)
    diff_mean_boot: float
    ci_low: float
    ci_high: float
    n_clusters: int
    n_rows: int
    n_boot: int

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "point_a": self.point_a,
            "point_b": self.point_b,
            "diff_point": self.diff_point,
            "diff_mean_boot": self.diff_mean_boot,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_clusters": self.n_clusters,
            "n_rows": self.n_rows,
            "n_boot": self.n_boot,
        }


def _metric_values(
    actual: np.ndarray, pred: np.ndarray,
) -> dict[str, float]:
    return all_point_metrics(actual, pred)


def clustered_bootstrap_metric_diffs(
    matched: pd.DataFrame,
    *,
    label_a: str = "a",
    label_b: str = "b",
    cluster_col: str = "id",
    n_boot: int = 1000,
    metrics: Iterable[str] = ("WAPE", "MAE", "RMSE", "Bias"),
    seed: int = 0,
    ci: float = 0.95,
    progress: bool = False,
) -> pd.DataFrame:
    """Clustered bootstrap on metric differences ``a - b``.

    Negative ``diff_point`` for an error-style metric (WAPE/MAE/RMSE) means
    ``a`` is better than ``b``. If you want "improvement of model M over
    baseline B", pass ``label_a="model"`` and ``label_b="baseline"``: a
    diff like ``WAPE −0.031, 95% CI [−0.040, −0.022]`` then reads
    naturally as "model lowers WAPE by 3.1 points, robust to resampling".

    Parameters
    ----------
    matched
        Output of :func:`matched_grid`. Must have columns ``actual``,
        ``pred_<label_a>``, ``pred_<label_b>``, and ``cluster_col``.
    n_boot
        Number of bootstrap iterations. 1,000 is enough for stable 95%
        percentile CIs at thousands of clusters.
    cluster_col
        Column used to cluster rows. ``id`` is the right choice for retail
        backtests: it captures within-product error correlation across
        origins and horizons.
    metrics
        Subset of ``{"WAPE","MAE","RMSE","Bias"}``.
    seed
        For reproducibility.
    ci
        Two-sided coverage (default 0.95).

    Returns
    -------
    pandas.DataFrame
        One row per metric, columns from :class:`BootstrapResult`.
    """
    needed = {f"pred_{label_a}", f"pred_{label_b}", "actual", cluster_col}
    missing = sorted(needed - set(matched.columns))
    if missing:
        raise KeyError(f"matched frame missing columns: {missing}")
    if len(matched) == 0:
        raise ValueError("matched frame is empty.")

    metrics = list(metrics)
    clusters = matched[cluster_col].unique()
    n_clusters = len(clusters)
    cluster_to_idx = {c: np.where(matched[cluster_col].to_numpy() == c)[0]
                      for c in clusters}

    actual_arr = matched["actual"].to_numpy(dtype=float)
    pred_a_arr = matched[f"pred_{label_a}"].to_numpy(dtype=float)
    pred_b_arr = matched[f"pred_{label_b}"].to_numpy(dtype=float)

    # Point estimates on the full sample.
    point_a = _metric_values(actual_arr, pred_a_arr)
    point_b = _metric_values(actual_arr, pred_b_arr)

    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {m: [] for m in metrics}

    for b_iter in range(n_boot):
        sampled = rng.choice(clusters, size=n_clusters, replace=True)
        idx_parts = [cluster_to_idx[c] for c in sampled]
        idx = np.concatenate(idx_parts)
        a = actual_arr[idx]
        p_a = pred_a_arr[idx]
        p_b = pred_b_arr[idx]
        m_a = _metric_values(a, p_a)
        m_b = _metric_values(a, p_b)
        for metric in metrics:
            val_a = m_a[metric]
            val_b = m_b[metric]
            if np.isfinite(val_a) and np.isfinite(val_b):
                draws[metric].append(val_a - val_b)
        if progress and (b_iter % max(1, n_boot // 10) == 0):
            print(f"  bootstrap iter {b_iter}/{n_boot}")

    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q

    rows: list[BootstrapResult] = []
    for metric in metrics:
        arr = np.asarray(draws[metric], dtype=float)
        if arr.size == 0:
            ci_low = ci_high = mean_boot = float("nan")
        else:
            ci_low = float(np.quantile(arr, lo_q))
            ci_high = float(np.quantile(arr, hi_q))
            mean_boot = float(np.mean(arr))

        rows.append(BootstrapResult(
            metric=metric,
            point_a=point_a[metric],
            point_b=point_b[metric],
            diff_point=point_a[metric] - point_b[metric],
            diff_mean_boot=mean_boot,
            ci_low=ci_low,
            ci_high=ci_high,
            n_clusters=int(n_clusters),
            n_rows=int(len(matched)),
            n_boot=int(n_boot),
        ))

    return pd.DataFrame([r.to_dict() for r in rows])


__all__ = [
    "matched_grid",
    "clustered_bootstrap_metric_diffs",
    "BootstrapResult",
    "DEFAULT_KEYS",
]
