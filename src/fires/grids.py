"""Model-selection grids for the Fourier-ARIMA / SARIMA baselines (phase 3)."""

from __future__ import annotations

import ast
import itertools
import time

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

from fires.models import fit_sarimax

TIME_LIMIT = 60.0  # seconds per fit; slower fits are recorded as failed
COLS = ["grid", "K", "trend", "order", "seasonal_order", "aic", "bic", "llf", "converged",
        "fit_seconds", "max_pvalue_arma", "lb_p52", "lb_p104", "n_params"]


def grid_specs(grid: str, best_pq: tuple[int, int] | None = None):
    """Yield (K, trend, order, seasonal_order) for grid 'A' (d=0), 'B' (d=1) or 'C' (SARIMA)."""
    if grid == "A":
        for K, tr, p, q in itertools.product((2, 3, 4), ("c", "ct"), (1, 2, 3), (0, 1)):
            yield K, tr, (p, 0, q), (0, 0, 0, 0)
    elif grid == "B":
        for K, tr, p, q in itertools.product((2, 3, 4), ("n", "t"), (0, 1, 2), (0, 1, 2)):
            yield K, tr, (p, 1, q), (0, 0, 0, 0)
    elif grid == "C":
        p, q = best_pq
        for so in ((1, 0, 0, 52), (0, 0, 1, 52), (1, 0, 1, 52)):
            yield 0, "c", (p, 0, q), so  # K = 0: seasonality is in the seasonal ARMA part
    else:
        raise ValueError(grid)


def arma_pvalue_max(res) -> float:
    names = [n for n in res.pvalues.index if n.startswith(("ar.", "ma."))]
    return float(res.pvalues[names].max()) if names else 0.0


def ljung_box(res, order, seasonal_order) -> tuple[float, float]:
    """p-values of Ljung-Box at lags 52 and 104 on standardized residuals.

    Drops the first max(p, 52) residuals with a seasonal part, the first 10 otherwise.
    model_df = number of ARMA coefficients (p+q+P+Q).
    """
    seasonal = seasonal_order[3] > 0
    skip = max(order[0], 52) if seasonal else 10
    r = pd.Series(res.resid).iloc[skip:].dropna()
    df = order[0] + order[2] + seasonal_order[0] + seasonal_order[2]
    lb = acorr_ljungbox(r, lags=[52, 104], model_df=df)
    return float(lb["lb_pvalue"].iloc[0]), float(lb["lb_pvalue"].iloc[1])


def run_grid(grid: str, y: pd.Series, best_pq=None, verbose: bool = True) -> pd.DataFrame:
    rows = []
    for K, tr, order, so in grid_specs(grid, best_pq):
        t0 = time.perf_counter()
        row = {"grid": grid, "K": K, "trend": tr, "order": str(order), "seasonal_order": str(so)}
        try:
            res = fit_sarimax(y, K, order, tr, so)
            sec = time.perf_counter() - t0
            lb52, lb104 = ljung_box(res, order, so)
            ok = bool(res.mle_retvals.get("converged", False)) and sec <= TIME_LIMIT
            row.update(aic=res.aic, bic=res.bic, llf=res.llf, converged=ok, fit_seconds=sec,
                       max_pvalue_arma=arma_pvalue_max(res), lb_p52=lb52, lb_p104=lb104,
                       n_params=len(res.params))
        except Exception as e:  # noqa: BLE001
            row.update(converged=False, fit_seconds=time.perf_counter() - t0,
                       error=f"{type(e).__name__}: {e}")
        rows.append(row)
        if verbose:
            print(grid, K, tr, order, so, f"{row['fit_seconds']:.1f}s", row.get("bic"))
    return pd.DataFrame(rows)


def select(df: pd.DataFrame, bic_tol: float = 2.0) -> pd.Series | None:
    """Within one grid: converged and max ARMA p < 0.05; lowest BIC, but take the simplest
    (fewest parameters) among models within ``bic_tol`` of the best. None if nobody qualifies."""
    ok = df[df["converged"].astype(bool) & (df["max_pvalue_arma"] < 0.05)]
    if ok.empty:
        return None
    near = ok[ok["bic"] <= ok["bic"].min() + bic_tol]
    return near.sort_values(["n_params", "bic"]).iloc[0]


def parse_spec(row: pd.Series):
    return int(row["K"]), ast.literal_eval(row["order"]), row["trend"], eval(row["seasonal_order"])


def best_pq_from(row: pd.Series) -> tuple[int, int]:
    _, order, _, _ = parse_spec(row)
    return order[0], order[2]





def plot_resid(res, title: str, path=None):
    """Residuals over time with annual mean, residual ACF (lag 104) and QQ plot."""
    import matplotlib.pyplot as plt
    from statsmodels.graphics.gofplots import qqplot
    from statsmodels.graphics.tsaplots import plot_acf

    r = pd.Series(res.resid).iloc[10:]  # drop the diffuse start-up residuals
    fig = plt.figure(figsize=(12, 7.5))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(r.index, r.values, lw=0.7)
    annual = r.groupby(r.index.year).mean()
    for yr, m in annual.items():
        ax1.hlines(m, pd.Timestamp(f"{yr}-01-01"), pd.Timestamp(f"{yr}-12-31"),
                   colors="tab:red", lw=2)
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_title(f"{title}: residuos (rojo = media anual)")
    plot_acf(r, lags=104, ax=fig.add_subplot(2, 2, 3), title="ACF de residuos")
    qqplot(r, line="s", ax=fig.add_subplot(2, 2, 4))
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=130)
    return fig, annual


def plot_fc2021(y: pd.Series, hotspots: pd.Series, spec, title: str, path=None):
    """52-week forecast from the end of y against the observed 2021, on the count scale."""
    import matplotlib.pyplot as plt

    from fires.models import fit_sarimax, fourier_terms, future_index

    K, order, trend, so = spec
    res = fit_sarimax(y, K, order, trend, so)
    idx = future_index(y, 52)
    exog = fourier_terms(idx, K) if K > 0 else None
    fc = res.get_forecast(52, exog=exog)
    mean = fc.predicted_mean.to_numpy()
    ci = fc.conf_int(alpha=0.2).to_numpy()
    obs = hotspots.reindex(idx)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(idx, obs, "k.-", label="observado 2021")
    ax.plot(idx, np.expm1(mean), color="tab:red", label="pronóstico (mediana)")
    ax.fill_between(idx, np.expm1(ci[:, 0]), np.expm1(ci[:, 1]), color="tab:red", alpha=0.15,
                    label="intervalo 80 %")
    ax.set_yscale("log")
    ax.set_title(f"{title}: 52 semanas desde {y.index[-1]:%Y-%m-%d} (escala log del eje)")
    ax.legend()
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=130)
    return fig, pd.DataFrame({"obs": obs, "pred": np.expm1(mean)}, index=idx)
