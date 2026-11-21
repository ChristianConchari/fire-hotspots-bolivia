"""Rolling-origin backtest shared by every model.

A model is any callable ``fit_forecast(train: pd.Series, h: int) -> pd.Series``:
``train`` is y = log1p(hotspots) (NaN kept) up to and including the origin week, and
the result has length h, on the log1p scale, indexed by the h following weeks. It is either
a Series (the log1p forecast) or a DataFrame with columns ``y_log`` and optionally ``duan``
(Duan smearing factor) and ``sigma2`` (residual variance), both estimated at the origin.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from fires.download import ROOT
from fires.weekly import PROCESSED

SAMPLE_START = pd.Timestamp("2012-01-23")  # first full week
SAMPLE_END = pd.Timestamp("2025-12-29")  # last week of the sample
FIRST_ORIGIN = pd.Timestamp("2021-01-04")  # first Monday of 2021
ORIGIN_STEP = 4  # weeks between origins
MAX_H = 12
RESULTS = ROOT / "results"

FitForecast = Callable[[pd.Series, int], pd.Series]


def load_series(path=PROCESSED) -> pd.DataFrame:
    """Weekly table restricted to the modelling sample, with y = log1p(hotspots)."""
    df = pd.read_parquet(path).loc[SAMPLE_START:SAMPLE_END].copy()
    df["y"] = np.log1p(df["hotspots"])
    return df


def make_origins(
    index: pd.DatetimeIndex, first=FIRST_ORIGIN, step: int = ORIGIN_STEP, max_h: int = MAX_H
) -> pd.DatetimeIndex:
    """Origins every ``step`` weeks from ``first`` while origin + max_h weeks is in the sample."""
    last = index[-1] - pd.Timedelta(weeks=max_h)
    return pd.date_range(first, last, freq=pd.Timedelta(weeks=step))


def rolling_origin(
    fit_forecast: FitForecast,
    df: pd.DataFrame,
    *,
    origins: pd.DatetimeIndex | None = None,
    max_h: int = MAX_H,
    refit_every: int = 1,
) -> pd.DataFrame:
    """Expanding-window backtest. Returns origin, target_week, h, y_true and count forecasts.

    ``y_pred`` = expm1(f), the median back-transform (as in Gate 2); ``y_pred_duan`` =
    exp(f) * mean(exp(resid)) - 1 (Duan smearing); ``y_pred_sigma2`` = expm1(f + sigma2/2);
    the last two are NaN for models that give no residual stats. ``y_log`` is f itself.

    ``refit_every``: refit at every k-th origin (default 1 = every origin).
    TODO(phase 4): for k > 1, reuse the fitted model between refits and only update its
    state (e.g. SARIMAX ``results.append(..., refit=False)``). That needs a model object
    with fit/update instead of a plain function, so it is not implemented yet.
    """
    if refit_every != 1:
        raise NotImplementedError("refit_every > 1 is planned for phase 4 (SARIMA state update)")
    origins = make_origins(df.index, max_h=max_h) if origins is None else origins
    rows = []
    for origin in origins:
        train = df["y"].loc[:origin]  # includes the origin, nothing later
        fc = fit_forecast(train, max_h)
        if isinstance(fc, pd.Series):
            fc = fc.rename("y_log").to_frame()
        if len(fc) != max_h:
            raise ValueError(f"fit_forecast returned {len(fc)} values, expected {max_h}")
        for h, (target, r) in enumerate(fc.iterrows(), start=1):
            f = float(r["y_log"])
            duan = float(r.get("duan", np.nan))
            s2 = float(r.get("sigma2", np.nan))
            rows.append(
                {
                    "origin": origin,
                    "target_week": target,
                    "h": h,
                    "y_true": df["hotspots"].get(target, np.nan),
                    "y_pred": float(np.expm1(f)),
                    "y_pred_duan": float(np.exp(f) * duan - 1),
                    "y_pred_sigma2": float(np.expm1(f + s2 / 2)),
                    "y_log": f,
                }
            )
    return pd.DataFrame(rows)
