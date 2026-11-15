"""Gate 2 comparison: seasonal naive vs the two Fourier-ARIMA models.

Usage:
    uv run python -m fires.compare     # writes results/metrics/gate2_comparison.csv + figures
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from fires.backtest import RESULTS
from fires.metrics import EVAL_SETS, common_targets, comparison_table

MODELS = ["seasonal_naive", "fourier_arma101", "fourier_arima012"]
COLORS = dict(zip(MODELS, ["tab:gray", "tab:blue", "tab:orange"], strict=True))


def load_forecasts() -> dict[str, pd.DataFrame]:
    return {m: pd.read_parquet(RESULTS / "forecasts" / f"{m}.parquet") for m in MODELS}


def _panel_plot(table: pd.DataFrame, col: str, ylabel: str, title: str, path, hline=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, es in zip(axes, EVAL_SETS, strict=True):
        for m in MODELS:
            if col == "mae_ratio_vs_naive" and m == "seasonal_naive":
                continue
            d = table[(table["eval_set"] == es) & (table["model"] == m)]
            ax.plot(d["h"], d[col], "o-", color=COLORS[m], label=m)
        if hline is not None:
            ax.axhline(hline, color="k", lw=0.8, ls="--")
        ax.set_title(f"eval_set = {es}")
        ax.set_xlabel("horizonte h (semanas)")
        ax.set_xticks(range(1, 13))
    axes[0].set_ylabel(ylabel)
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return fig


def plot_mae(table: pd.DataFrame, path=None):
    path = path or RESULTS / "figures" / "gate2_mae_by_h.png"
    return _panel_plot(table, "mae", "MAE (hotspots/semana)", "MAE por horizonte", path)


def plot_ratio(table: pd.DataFrame, path=None):
    path = path or RESULTS / "figures" / "gate2_ratio_by_h.png"
    return _panel_plot(table, "mae_ratio_vs_naive", "MAE / MAE seasonal naive",
                       "Error relativo a seasonal naive (<1 mejora)", path, hline=1.0)


def plot_forecasts(forecasts: dict[str, pd.DataFrame], hotspots: pd.Series, path=None):
    """Observed 2021-2025 vs h=1 and h=4 forecasts (count scale, log y axis)."""
    path = path or RESULTS / "figures" / "gate2_forecasts_h1_h4.png"
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, h in zip(axes, (1, 4), strict=True):
        obs = hotspots.loc["2021-01-04":]
        ax.plot(obs.index, obs.values, "k-", lw=1.4, label="observado")
        for m, fc in forecasts.items():
            d = fc[fc["h"] == h].set_index("target_week")["y_pred"]
            ax.plot(d.index, d.values, ".-", lw=0.8, ms=3, color=COLORS[m], label=m)
        ax.set_yscale("log")
        ax.set_ylabel(f"hotspots/semana (eje log), h={h}")
        ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return fig


def main() -> None:
    fcs = load_forecasts()
    table = comparison_table(fcs)
    table.to_csv(RESULTS / "metrics" / "gate2_comparison.csv", index=False)
    from fires.backtest import load_series

    hotspots = load_series()["hotspots"]
    plot_mae(table)
    plot_ratio(table)
    plot_forecasts(fcs, hotspots)
    dropped = {m: len(fcs[m]) - len(c) for m, c in common_targets(fcs).items()}
    print("rows dropped by target intersection:", dropped)
    print(table[table["h"].isin([1, 4, 8, 12])].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
