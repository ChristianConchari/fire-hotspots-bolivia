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


PUBLICATION_DELAY_WEEKS = 9  # ONI of a month is published ~1 month after it ends (~8-9 weeks)
MAX_H = 12
ONI_LAG_CANDIDATES = (21, 26, 34, 43)  # weeks (~5, 6, 8, 10 months); all >= MAX_H + delay


def lagged_oni_exog(
    oni: pd.Series, index: pd.DatetimeIndex, lag_weeks: int, origin: pd.Timestamp | None = None
) -> pd.Series:
    """ONI used for each week in ``index``: oni(week - lag_weeks).

    With ``origin`` given, only values published by then count: oni(w) is usable at origin t
    iff w <= t - PUBLICATION_DELAY_WEEKS. Anything later is treated as missing (NaN), so
    the caller cannot see it. Weeks whose lagged ONI is unavailable come back NaN.
    """
    src = pd.DatetimeIndex(index) - WEEK * lag_weeks
    vals = oni.reindex(src).to_numpy(dtype=float)
    if origin is not None:
        vals = np.where(src <= origin - WEEK * PUBLICATION_DELAY_WEEKS, vals, np.nan)
    return pd.Series(vals, index=pd.DatetimeIndex(index), name="oni_lag")


def fit_sarimax(y: pd.Series, K: int, order, trend: str, seasonal_order=(0, 0, 0, 0),
                extra_exog: pd.DataFrame | None = None):
    """SARIMAX on y (NaN allowed) with K Fourier harmonics (+ optional extra regressors).

    Rows where an extra regressor is NaN are dropped from the likelihood: y is set to NaN
    there and the regressor filled with 0 (a missing y adds no information).
    Returns the results.
    """
    import warnings

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = regular_series(y)
    parts = [fourier_terms(y.index, K)] if K > 0 else []
    if extra_exog is not None:
        ex = pd.DataFrame(extra_exog).copy()
        ex.index = y.index
        bad = ex.isna().any(axis=1).to_numpy()
        y = y.copy()
        y[bad] = np.nan
        parts.append(ex.fillna(0.0))
    exog = pd.concat(parts, axis=1) if parts else None
    mod = SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal_order, trend=trend)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mod.fit(disp=False, maxiter=200)


def resid_stats(res) -> tuple[float, float]:
    """(Duan smearing factor, residual variance) of the in-sample one-step residuals.

    Residuals come from the training sample only (burn-in and missing weeks dropped), so
    they are available at the origin. Both are on the log1p scale.
    """
    e = np.asarray(res.resid, dtype=float)[res.loglikelihood_burn:]
    e = e[np.isfinite(e)]
    return float(np.mean(np.exp(e))), float(np.var(e))


def _make_factory(K: int, order, trend: str, oni: pd.Series | None, lag_weeks: int | None):
    fitlog: list[dict] = []
    state = {"oni": oni}

    def fit_forecast(train: pd.Series, h: int) -> pd.DataFrame:
        """Columns: y_log (log1p-scale mean), duan (smearing factor), sigma2 (resid variance)."""
        idx = future_index(train, h)
        origin = train.index[-1]
        entry = {"origin": origin, "converged": False, "error": ""}
        fitlog.append(entry)
        nan = np.full(h, np.nan)
        out = {"y_log": nan, "duan": nan, "sigma2": nan}
        try:
            extra = fexog = None
            if lag_weeks is not None:
                if h + PUBLICATION_DELAY_WEEKS > lag_weeks:
                    raise ValueError(f"lag_weeks={lag_weeks} too short for h={h}: needs "
                                     f">= {h + PUBLICATION_DELAY_WEEKS}")
                if state["oni"] is None:
                    from fires.weekly import PROCESSED

                    state["oni"] = pd.read_parquet(PROCESSED)["oni"]
                extra = lagged_oni_exog(state["oni"], train.index, lag_weeks, origin).to_frame()
                fexog = lagged_oni_exog(state["oni"], idx, lag_weeks, origin).to_frame()
                if fexog.isna().any().any():
                    raise ValueError("ONI not yet published for a forecast week")
            res = fit_sarimax(train, K, order, trend, extra_exog=extra)
            entry["converged"] = bool(res.mle_retvals.get("converged", False))
            parts = [fourier_terms(idx, K)] if K > 0 else []
            if fexog is not None:
                parts.append(fexog)
            exog = pd.concat(parts, axis=1) if parts else None
            fc = res.forecast(h, exog=exog).to_numpy()
            duan, sigma2 = resid_stats(res)
            out = {"y_log": fc, "duan": np.full(h, duan), "sigma2": np.full(h, sigma2)}
        except Exception as e:  # noqa: BLE001
            entry["error"] = f"{type(e).__name__}: {e}"
        return pd.DataFrame(out, index=idx)

    fit_forecast.fitlog = fitlog
    return fit_forecast


def make_fourier_arima(K: int, order, trend: str):
    """Factory: fit_forecast(train, h) for Fourier terms + ARIMA(order) errors (log1p scale).

    Returns a DataFrame (y_log, duan, sigma2), see ``rolling_origin`` for the back-transforms.
    The returned function keeps ``fit_forecast.fitlog``: one dict per call with the origin,
    convergence flag and any exception. A non-converged fit is still used; a failed fit
    returns NaN forecasts.
    """
    return _make_factory(K, order, trend, None, None)


def make_fourier_arima_oni(K: int, order, trend: str, lag_weeks: int, oni: pd.Series | None = None):
    """Like ``make_fourier_arima`` plus ONI lagged ``lag_weeks`` as an extra regressor.

    The ONI for target week s is oni(s - lag_weeks). It must have been published at the
    origin t: s - lag <= t - PUBLICATION_DELAY_WEEKS for every h <= 12, i.e. lag >= 21.
    Values after t - 9 weeks are masked and never reach the model. ``oni`` is a weekly
    series (default: the ``oni`` column of the processed weekly table).
    """
    if lag_weeks < MAX_H + PUBLICATION_DELAY_WEEKS:
        raise ValueError(f"lag_weeks must be >= {MAX_H + PUBLICATION_DELAY_WEEKS} (no leakage)")
    return _make_factory(K, order, trend, oni, lag_weeks)
