"""Forecasting models. Each is a ``fit_forecast(train, h) -> Series`` on the log1p scale."""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEK = pd.Timedelta(days=7)


def future_index(train: pd.Series, h: int) -> pd.DatetimeIndex:
    """The h weekly Mondays that follow the last week of ``train``."""
    return pd.DatetimeIndex([train.index[-1] + WEEK * k for k in range(1, h + 1)])


def seasonal_naive(train: pd.Series, h: int) -> pd.Series:
    """y_hat(t+h) = y(t+h-52): the same ISO week one year earlier (364 days).

    If that week is NaN (sensor outage), fall back to lag 104. Needs h <= 52 so every
    source week is inside ``train``. Stays NaN if both lags are missing.
    """
    if h > 52:
        raise ValueError("seasonal_naive supports h <= 52")
    idx = future_index(train, h)
    out = []
    for t in idx:
        v = train.get(t - 52 * WEEK, np.nan)
        if pd.isna(v):
            v = train.get(t - 104 * WEEK, np.nan)
        out.append(v)
    return pd.Series(out, index=idx, name="y_pred", dtype=float)
