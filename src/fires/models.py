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


# ---- Fourier-ARIMA (dynamic harmonic regression) ---------------------------------------
FOURIER_ORIGIN = pd.Timestamp("2012-01-02")  # a Monday; time zero of the harmonics
FOURIER_PERIOD = 365.25 / 7  # weeks per year (52.1775)


def fourier_terms(index: pd.DatetimeIndex, K: int) -> pd.DataFrame:
    """Annual harmonics sin_k, cos_k (k = 1..K) with period 52.1775 weeks.

    t = (week - 2012-01-02).days / 7, so the phase stays right across 53-week years.
    Deterministic, hence generating them for future weeks leaks nothing.
    """
    index = pd.DatetimeIndex(index)
    t = np.asarray((index - FOURIER_ORIGIN).days, dtype=float) / 7.0
    cols = {}
    for k in range(1, K + 1):
        angle = 2 * np.pi * k * t / FOURIER_PERIOD
        cols[f"sin_{k}"] = np.sin(angle)
        cols[f"cos_{k}"] = np.cos(angle)
    return pd.DataFrame(cols, index=index)


def regular_series(y: pd.Series) -> pd.Series:
    """Same values on an index with an explicit 7-day frequency (SARIMAX likes that)."""
    idx = pd.date_range(y.index[0], periods=len(y), freq="7D")
    return pd.Series(y.to_numpy(dtype=float), index=idx, name=y.name)


def fit_sarimax(y: pd.Series, K: int, order, trend: str, seasonal_order=(0, 0, 0, 0)):
    """SARIMAX on y (NaN allowed) with K Fourier harmonics as exog. Returns the results."""
    import warnings

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = regular_series(y)
    exog = fourier_terms(y.index, K) if K > 0 else None
    mod = SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal_order, trend=trend)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mod.fit(disp=False, maxiter=200)


def make_fourier_arima(K: int, order, trend: str):
    """Factory: fit_forecast(train, h) for Fourier terms + ARIMA(order) errors (log1p scale)."""

    def fit_forecast(train: pd.Series, h: int) -> pd.Series:
        res = fit_sarimax(train, K, order, trend)
        idx = future_index(train, h)
        exog = fourier_terms(idx, K) if K > 0 else None
        fc = res.forecast(h, exog=exog)
        return pd.Series(fc.to_numpy(), index=idx, name="y_pred")

    return fit_forecast
