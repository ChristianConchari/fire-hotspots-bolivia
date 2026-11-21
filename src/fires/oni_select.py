"""Choose the ONI lag using only the identification sample (2012-2020). No backtest data.

Every model (B and B+ONI at each candidate lag) is fitted on the SAME rows: weeks up to
2020-12-28 whose lagged ONI exists for ALL candidate lags (the first ``max(lag)`` weeks of the
sample are dropped for every model, B included), so AIC/BIC and the likelihood ratio are
comparable. Rule fixed before looking at results: lowest AIC wins (ties -> BIC); the LR test
and the z-value of the ONI coefficient are reported, not used to select.

Usage:
    uv run python -m fires.oni_select
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from fires.backtest import FIRST_ORIGIN, RESULTS, load_series
from fires.models import ONI_LAG_CANDIDATES, fit_sarimax, lagged_oni_exog

K, ORDER, TREND = 2, (0, 1, 2), "n"
IDENT_END = FIRST_ORIGIN - pd.Timedelta(weeks=1)  # 2020-12-28, last identification week


def run() -> pd.DataFrame:
    df = load_series()
    oni = pd.read_parquet("data/processed/weekly.parquet")["oni"]
    y = df["y"].loc[:IDENT_END]
    ex = {lag: lagged_oni_exog(oni, y.index, lag) for lag in ONI_LAG_CANDIDATES}
    valid = pd.concat(ex.values(), axis=1).notna().all(axis=1)
    y = y.where(valid)  # identical rows for every model
    first = y.index[valid.to_numpy()][0]

    base = fit_sarimax(y, K, ORDER, TREND)
    n_rows = int(y.notna().sum())
    rows = [{"model": "B", "lag_weeks": np.nan, "n_rows": n_rows, "first_week": first,
             "llf": base.llf, "aic": base.aic, "bic": base.bic, "beta_oni": np.nan,
             "z_oni": np.nan, "lr_stat": np.nan, "lr_pvalue": np.nan,
             "converged": bool(base.mle_retvals.get("converged", False))}]
    for lag in ONI_LAG_CANDIDATES:
        r = fit_sarimax(y, K, ORDER, TREND, extra_exog=ex[lag].to_frame())
        lr = 2 * (r.llf - base.llf)
        rows.append({"model": "B+ONI", "lag_weeks": lag, "n_rows": n_rows, "first_week": first,
                     "llf": r.llf, "aic": r.aic, "bic": r.bic,
                     "beta_oni": float(r.params["oni_lag"]), "z_oni": float(r.tvalues["oni_lag"]),
                     "lr_stat": lr, "lr_pvalue": float(stats.chi2.sf(lr, 1)),
                     "converged": bool(r.mle_retvals.get("converged", False))})
    out = pd.DataFrame(rows)
    out["d_aic_vs_B"] = out["aic"] - out.loc[0, "aic"]
    out["d_bic_vs_B"] = out["bic"] - out.loc[0, "bic"]
    return out


def choose(table: pd.DataFrame) -> int:
    c = table[table["model"] == "B+ONI"].sort_values(["aic", "bic"])
    return int(c["lag_weeks"].iloc[0])


def main() -> None:
    t = run()
    t.to_csv(RESULTS / "oni_lag_selection.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.to_string(index=False, float_format="%.4f"))
    print("chosen lag (lowest AIC):", choose(t))


if __name__ == "__main__":
    main()
