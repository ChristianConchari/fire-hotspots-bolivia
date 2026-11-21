"""Gate 3 (part 1): seasonal naive vs model B vs B+ONI, median vs bias-corrected back-transform.

Usage:
    uv run python -m fires.compare_oni   # gate3_oni_comparison.csv, gate3_dm.csv, figure
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from fires.backtest import ORIGIN_STEP, RESULTS
from fires.metrics import EVAL_SETS, comparison_table, eval_subset

NAIVE, B, BONI = "seasonal_naive", "fourier_arima012", "fourier_arima012_oni"
BACKTRANSFORMS = {"median": "y_pred", "duan": "y_pred_duan", "sigma2": "y_pred_sigma2"}
PRIMARY = "duan"  # chosen a priori: nonparametric, no normality assumption
COLORS = {B: "tab:orange", BONI: "tab:green", NAIVE: "tab:gray"}


def load() -> dict[str, pd.DataFrame]:
    return {m: pd.read_parquet(RESULTS / "forecasts" / f"{m}.parquet") for m in (NAIVE, B, BONI)}


def comparison(fcs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Long table: backtransform, eval_set, h, model, mae, rmse, bias, n, ratios.

    ``mae_ratio_vs_naive`` is against the seasonal naive count forecast; ``mae_ratio_vs_B``
    is against B with the SAME back-transform. Targets are the common set of all three
    models (same for every back-transform).
    """
    parts = []
    for bt, col in BACKTRANSFORMS.items():
        sub = {NAIVE: fcs[NAIVE]}
        for m in (B, BONI):
            sub[m] = fcs[m].assign(y_pred=fcs[m][col])
        t = comparison_table(sub, baseline=NAIVE)
        ref = t.loc[t["model"] == B, ["eval_set", "h", "mae"]].rename(columns={"mae": "mae_B"})
        t = t.merge(ref, on=["eval_set", "h"])
        t["mae_ratio_vs_B"] = t["mae"] / t["mae_B"]
        t.insert(0, "backtransform", bt)
        parts.append(t.drop(columns="mae_B"))
    out = pd.concat(parts, ignore_index=True)
    return out[~((out["model"] == NAIVE) & (out["backtransform"] != "median"))].reset_index(
        drop=True
    )


def dm_test(d: np.ndarray, h: int, step: int = ORIGIN_STEP) -> dict:
    """Diebold-Mariano on a loss differential series d (one value per origin, spaced ``step``
    weeks). Newey-West bandwidth q = floor((h-1)/step) origins, Harvey-Leybourne-Newbold
    small-sample correction, Student t with n-1 df. d > 0 means the second model is better."""
    d = d[np.isfinite(d)]
    n = len(d)
    q = (h - 1) // step
    dbar = d.mean()
    dc = d - dbar
    lrv = float(dc @ dc) / n
    for k in range(1, q + 1):
        lrv += 2 * (1 - k / (q + 1)) * float(dc[k:] @ dc[:-k]) / n
    k = q + 1
    stat = dbar / np.sqrt(lrv / n) * np.sqrt((n + 1 - 2 * k + k * (k - 1) / n) / n)
    return {"n": n, "mean_diff": dbar, "dm_stat": stat, "p_value": 2 * stats.t.sf(abs(stat), n - 1),
            "share_oni_better": float(np.mean(d > 0))}


def dm_table(fcs: dict[str, pd.DataFrame], hs=(4, 12)) -> pd.DataFrame:
    rows = []
    for bt, col in BACKTRANSFORMS.items():
        for es in EVAL_SETS:
            for h in hs:
                a = eval_subset(fcs[B], es).query("h == @h").set_index("origin")
                b = eval_subset(fcs[BONI], es).query("h == @h").set_index("origin")
                ea = (a["y_true"] - a[col]).abs()
                eb = (b["y_true"] - b[col]).abs()
                d = (ea - eb.reindex(ea.index)).dropna().to_numpy()  # > 0: B+ONI smaller error
                rows.append({"backtransform": bt, "eval_set": es, "h": h, **dm_test(d, h)})
    return pd.DataFrame(rows)


def plot_mae(table: pd.DataFrame, path=None):
    path = path or RESULTS / "figures" / "gate3_oni_mae_by_horizon.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, es in zip(axes, EVAL_SETS, strict=True):
        for m, ls in ((NAIVE, ":"), (B, "-"), (BONI, "-")):
            for bt, mk in (("median", "o"), ("duan", "s")):
                if m == NAIVE and bt != "median":
                    continue
                d = table[(table.eval_set == es) & (table.model == m) & (table.backtransform == bt)]
                lab = "seasonal naive" if m == NAIVE else f"{'B' if m == B else 'B+ONI'} ({bt})"
                ax.plot(d.h, d.mae, ls, marker=mk, ms=4, color=COLORS[m],
                        alpha=1 if bt == "duan" else 0.55, label=lab)
        ax.set_title(f"eval_set = {es}")
        ax.set_xlabel("horizonte h (semanas)")
        ax.set_xticks(range(1, 13))
    axes[0].set_ylabel("MAE (hotspots/semana)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return fig


def main() -> None:
    fcs = load()
    t = comparison(fcs)
    t.to_csv(RESULTS / "metrics" / "gate3_oni_comparison.csv", index=False)
    dm = dm_table(fcs)
    dm.to_csv(RESULTS / "metrics" / "gate3_oni_dm.csv", index=False)
    plot_mae(t)
    pd.set_option("display.width", 200)
    s = t[(t.h.isin([1, 4, 8, 12])) & (t.model != NAIVE)]
    print(s.pivot_table(index=["eval_set", "h"], columns=["model", "backtransform"],
                        values="mae_ratio_vs_B").round(3).to_string())
    print(dm.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
