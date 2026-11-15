"""Stationarity and ACF/PACF helpers for the identification step (phase 2).

Diagnostics need gap-free series, so NaN weeks are linearly interpolated in log1p
space *for these tests only* (never for the backtest).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from fires.backtest import load_series

IDENT_END = pd.Timestamp("2020-12-28")  # identification sample ends before the first origin
ALPHA = 0.05
SEASON = 52
NLAGS = 156
KEY_LAGS = (1, 2, 52, 104, 156)
SERIES = ("y", "d1", "d52", "d1d52")
SERIES_LABEL = {"y": "y", "d1": "Δ1 y", "d52": "Δ52 y", "d1d52": "Δ1Δ52 y"}


def identification_samples() -> tuple[dict[str, pd.Series], dict[str, int]]:
    """Gap-free log1p samples and the NaN count each had before interpolation.

    Samples: 'ident' = 2012-01-23..2020-12-28, 'full' = 2012-01-23..last observed week.

    'full' drops the trailing partial week (2025-12-29, NaN) and interpolates the interior NaN.
    The identification sample must have no NaN; interpolation is applied and reported if it does.
    """
    y = load_series()["y"]
    ident = y.loc[:IDENT_END]
    full = y.loc[: y.last_valid_index()]
    return {"ident": ident.interpolate("linear"), "full": full.interpolate("linear")}, {
        "ident": int(ident.isna().sum()),
        "full": int(full.isna().sum()),
    }


def transforms(y: pd.Series) -> dict[str, pd.Series]:
    """y, Δ1 y, Δ52 y, Δ1Δ52 y (NaN heads dropped)."""
    d1 = y.diff(1)
    d52 = y.diff(SEASON)
    d1d52 = d52.diff(1)
    return {"y": y, "d1": d1.dropna(), "d52": d52.dropna(), "d1d52": d1d52.dropna()}


def _adf(x: pd.Series, reg: str) -> dict:
    stat, p, lags, *_ = adfuller(x, regression=reg, autolag="AIC", result_object=False)
    # MacKinnon p-values are clipped to [0, 1]; ADF has no truncation flag worth reporting
    return {"stat": stat, "pvalue": p, "lags": lags, "pvalue_truncated": False,
            "stationary": p < ALPHA}


def _kpss(x: pd.Series, reg: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # InterpolationWarning: p-value outside table
        stat, p, lags, _ = kpss(x, regression=reg, nlags="auto")
    return {"stat": stat, "pvalue": p, "lags": lags,
            "pvalue_truncated": bool(p <= 0.01 or p >= 0.1),
            "stationary": p >= ALPHA}  # KPSS null = stationary


def stationarity_table(samples: dict[str, pd.Series]) -> pd.DataFrame:
    """ADF and KPSS on y, Δ1 y, Δ52 y, Δ1Δ52 y for each sample."""
    rows = []
    for sample, y in samples.items():
        for name, x in transforms(y).items():
            regs = ("c", "ct") if name == "y" else ("c",)
            for test, fn in (("ADF", _adf), ("KPSS", _kpss)):
                for reg in regs:
                    r = fn(x, reg)
                    rows.append({
                        "sample": sample, "series": SERIES_LABEL[name], "test": test,
                        "regression": reg, "stat": r["stat"], "pvalue": r["pvalue"],
                        "lags": r["lags"], "pvalue_truncated": r["pvalue_truncated"],
                        "conclusion": "estacionaria" if r["stationary"] else "no estacionaria",
                    })
    return pd.DataFrame(rows)


def seasonal_diffs(y: pd.Series, m: int = SEASON) -> dict[str, object]:
    """pmdarima nsdiffs with OCSB and CH; an error is reported as a string, not raised."""
    from pmdarima.arima import nsdiffs

    out = {}
    for test in ("ocsb", "ch"):
        try:
            out[test] = int(nsdiffs(y.to_numpy(), m=m, test=test, max_D=2))
        except Exception as e:  # noqa: BLE001
            out[test] = f"error: {type(e).__name__}: {e}"
    return out


def acf_pacf(x: pd.Series, nlags: int = NLAGS) -> tuple[np.ndarray, np.ndarray]:
    return acf(x, nlags=nlags, fft=True), pacf(x, nlags=nlags, method="ywm")


def acf_key_lags(series: dict[str, pd.Series], lags=KEY_LAGS) -> pd.DataFrame:
    rows = {SERIES_LABEL[k]: acf(v, nlags=max(lags), fft=True)[list(lags)]
            for k, v in series.items()}
    return pd.DataFrame(rows, index=[f"lag {k}" for k in lags]).T


def plot_acf_pacf(x: pd.Series, title: str, path=None, nlags: int = NLAGS):
    """ACF (top) and PACF (bottom) with 95 % bands; marks multiples of 52."""
    import matplotlib.pyplot as plt

    a, p = acf_pacf(x, nlags)
    band = 1.96 / np.sqrt(len(x))
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for ax, vals, name in ((axes[0], a, "ACF"), (axes[1], p, "PACF")):
        ax.vlines(range(len(vals)), 0, vals, lw=1.2)
        ax.plot(range(len(vals)), vals, "o", ms=2.5)
        ax.axhline(0, color="k", lw=0.6)
        ax.axhspan(-band, band, alpha=0.15, color="tab:blue", lw=0)
        for k in (52, 104, 156):
            ax.axvline(k, color="tab:red", ls=":", lw=0.8)
        ax.set_ylabel(name)
    axes[0].set_title(f"{title} (n={len(x)}, bandas 95 %)")
    axes[1].set_xlabel("lag (semanas)")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=130)
    return fig
