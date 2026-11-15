"""Run a model through the backtest and write its forecasts and metrics.

Usage:
    uv run python -m fires.run_baseline seasonal_naive
"""

from __future__ import annotations

import argparse

from fires.backtest import RESULTS, load_series, rolling_origin
from fires.metrics import metrics_by_horizon
from fires.models import seasonal_naive

MODELS = {"seasonal_naive": seasonal_naive}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", choices=sorted(MODELS))
    name = ap.parse_args(argv).model

    df = load_series()
    fc = rolling_origin(MODELS[name], df)
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
