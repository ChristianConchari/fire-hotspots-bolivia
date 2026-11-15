"""Forecast accuracy metrics on the count scale.

A forecast table is long: one row per (origin, target_week, h) with columns
``origin, target_week, h, y_true, y_pred`` (counts, not log1p). Targets whose
``y_true`` is NaN (sensor outages) are left out of every metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EXCLUDED_YEAR = 2024  # ISO year left out of the "excl_2024" evaluation set
EVAL_SETS = ("all", "excl_2024")


def _valid_pairs(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    keep = ~np.isnan(t)
    return t[keep], p[keep]


def mae(y_true, y_pred) -> float:
    """Mean absolute error, ignoring pairs whose actual value is NaN."""
    t, p = _valid_pairs(y_true, y_pred)
    return float(np.mean(np.abs(t - p))) if len(t) else float("nan")


def rmse(y_true, y_pred) -> float:
    """Root mean squared error, ignoring pairs whose actual value is NaN."""
    t, p = _valid_pairs(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2))) if len(t) else float("nan")


def bias(y_true, y_pred) -> float:
    """Mean of (y_pred - y_true), ignoring NaN actuals. Positive = over-forecast."""
    t, p = _valid_pairs(y_true, y_pred)
    return float(np.mean(p - t)) if len(t) else float("nan")


def eval_subset(fc: pd.DataFrame, eval_set: str) -> pd.DataFrame:
    """Rows of the forecast table that belong to an evaluation set."""
    if eval_set == "all":
        return fc
    if eval_set == "excl_2024":
        year = pd.DatetimeIndex(fc["target_week"]).isocalendar().year.to_numpy()
        return fc[year != EXCLUDED_YEAR]
    raise ValueError(f"unknown eval_set {eval_set!r}")


def metrics_by_horizon(fc: pd.DataFrame, model: str = "") -> pd.DataFrame:
    """MAE and RMSE per horizon for each evaluation set.

    Returns columns ``model, eval_set, h, mae, rmse, bias, n`` (n = targets actually scored).
    """
    rows = []
    for eval_set in EVAL_SETS:
        sub = eval_subset(fc, eval_set)
        for h, g in sub.groupby("h"):
            t, _ = _valid_pairs(g["y_true"], g["y_pred"])
            rows.append(
                {
                    "model": model,
                    "eval_set": eval_set,
                    "h": int(h),
                    "mae": mae(g["y_true"], g["y_pred"]),
                    "rmse": rmse(g["y_true"], g["y_pred"]),
                    "bias": bias(g["y_true"], g["y_pred"]),
                    "n": len(t),
                }
            )
    return pd.DataFrame(rows, columns=["model", "eval_set", "h", "mae", "rmse", "bias", "n"])


def relative_mae(model_metrics: pd.DataFrame, naive_metrics: pd.DataFrame) -> pd.DataFrame:
    """MAE_model / MAE_seasonal_naive per horizon and evaluation set (< 1 beats naive)."""
    keys = ["eval_set", "h"]
    m = model_metrics[[*keys, "mae"]].merge(
        naive_metrics[[*keys, "mae"]], on=keys, suffixes=("", "_naive")
    )
    m["rel_mae"] = m["mae"] / m["mae_naive"]
    return m[[*keys, "rel_mae"]]


def common_targets(forecasts: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restrict every model to the (origin, target_week, h) rows where all models have a
    finite prediction and the actual is observed, so metrics compare identical targets."""
    key = ["origin", "target_week", "h"]
    keep = None
    for fc in forecasts.values():
        ok = fc.loc[fc["y_true"].notna() & fc["y_pred"].notna(), key]
        idx = pd.MultiIndex.from_frame(ok)
        keep = idx if keep is None else keep.intersection(idx)
    out = {}
    for name, fc in forecasts.items():
        mask = pd.MultiIndex.from_frame(fc[key]).isin(keep)
        out[name] = fc[mask]
    return out


def comparison_table(forecasts: dict[str, pd.DataFrame], baseline: str = "seasonal_naive"):
    """Long table eval_set, h, model, mae, rmse, bias, n, mae_ratio_vs_naive on common targets."""
    common = common_targets(forecasts)
    parts = [metrics_by_horizon(fc, model=name) for name, fc in common.items()]
    m = pd.concat(parts, ignore_index=True)
    base = m.loc[m["model"] == baseline, ["eval_set", "h", "mae"]].rename(
        columns={"mae": "mae_base"})
    m = m.merge(base, on=["eval_set", "h"])
    m["mae_ratio_vs_naive"] = m["mae"] / m["mae_base"]
    cols = ["eval_set", "h", "model", "mae", "rmse", "bias", "n", "mae_ratio_vs_naive"]
    return m[cols].sort_values(["eval_set", "h", "model"]).reset_index(drop=True)
