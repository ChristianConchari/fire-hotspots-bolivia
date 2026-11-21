"""Run a model through the backtest and write its forecasts and metrics.

Usage:
    uv run python -m fires.run_baseline seasonal_naive
"""

from __future__ import annotations

import argparse

import pandas as pd

from fires.backtest import RESULTS, load_series, rolling_origin
from fires.metrics import metrics_by_horizon
from fires.models import make_fourier_arima, make_fourier_arima_oni, seasonal_naive

ONI_LAG_WEEKS = 34

MODELS = {
    "seasonal_naive": seasonal_naive,
    "fourier_arma101": make_fourier_arima(2, (1, 0, 1), "c"),
    "fourier_arima012": make_fourier_arima(2, (0, 1, 2), "n"),
    # lag chosen on 2012-2020 only (results/oni_lag_selection.csv, lowest AIC)
    "fourier_arima012_oni": make_fourier_arima_oni(2, (0, 1, 2), "n", lag_weeks=ONI_LAG_WEEKS),
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", choices=sorted(MODELS))
    name = ap.parse_args(argv).model

    df = load_series()
    model = MODELS[name]
    fc = rolling_origin(model, df)
    fitlog = getattr(model, "fitlog", None)
    if fitlog is not None:
        log = pd.DataFrame(fitlog).assign(model=name)[["origin", "model", "converged", "error"]]
        (RESULTS / "forecasts").mkdir(parents=True, exist_ok=True)
        log.to_csv(RESULTS / "forecasts" / f"{name}_fitlog.csv", index=False)
        print(f"not converged: {(~log['converged']).sum()} of {len(log)}; "
              f"errors: {(log['error'] != '').sum()}")
    (RESULTS / "forecasts").mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics").mkdir(parents=True, exist_ok=True)
    fc.to_parquet(RESULTS / "forecasts" / f"{name}.parquet", index=False)
    m = metrics_by_horizon(fc, model=name)
    m.to_csv(RESULTS / "metrics" / f"{name}.csv", index=False)
    print(f"{fc['origin'].nunique()} origins, {fc['origin'].min():%Y-%m-%d} -> "
          f"{fc['origin'].max():%Y-%m-%d}")
    print(m.to_string(index=False))


if __name__ == "__main__":
    main()
