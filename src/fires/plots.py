"""First exploratory figures from the weekly table.

Usage:
    uv run python -m fires.plots        # writes results/figures/*.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fires.download import ROOT
from fires.weekly import PROCESSED

FIGURES = ROOT / "results" / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"  # series 1
ORANGE = "#eb6834"  # series 2
CONTEXT = "#c9c8c2"  # other years
MISSING = "#d9d8d2"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 10,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def _thousands(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:,.0f}k" if v >= 1000 else f"{v:,.0f}")


def weekly_series(w: pd.DataFrame) -> plt.Figure:
    """Hotspots per week with missing weeks shaded, and ONI on its own panel below."""
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.25})
    for week in w.index[w["hotspots"].isna()]:
        ax.axvspan(week, week + pd.Timedelta(days=7), color=MISSING, lw=0)
    ax.plot(w.index, w["hotspots"], color=BLUE, lw=1.4)
    ax.set_title("Fire hotspots per week, Bolivia (VIIRS S-NPP, 2012–2025)")
    ax.set_ylabel("hotspots / week")
    _thousands(ax)
    ax.annotate("shaded: weeks missing\nfrom sensor outages", xy=(0.01, 0.97), xycoords="axes fraction",
                va="top", fontsize=9, color=INK_2)

    ax2.axhline(0, color=INK_2, lw=0.8)
    for level in (-0.5, 0.5):
        ax2.axhline(level, color=GRID, lw=0.8, ls="--")
    ax2.plot(w.index, w["oni"], color=ORANGE, lw=1.4)
    ax2.set_title("Oceanic Niño Index (ONI, °C anomaly)", fontsize=11)
    ax2.set_ylabel("°C")
    ax2.annotate("El Niño > 0.5 · La Niña < −0.5", xy=(0.01, 0.95), xycoords="axes fraction",
                 va="top", fontsize=9, color=INK_2)
    return fig


def seasonal_profile(w: pd.DataFrame) -> plt.Figure:
    """Each year's weekly series against week of year (log scale), highlighting the extreme years."""
    iso = w.index.isocalendar()
    df = w.assign(year=iso["year"].to_numpy(), woy=iso["week"].to_numpy())
    df = df[(df["year"] >= 2012) & (df["year"] <= 2025)]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    highlight = {2019: ORANGE, 2024: BLUE}
    for year, g in df.groupby("year"):
        if year not in highlight:
            ax.plot(g["woy"], g["hotspots"], color=CONTEXT, lw=1)
    median = df.groupby("woy")["hotspots"].median()
    ax.plot(median.index, median.values, color=INK, lw=2, label="median 2012–2025")
    for year, color in highlight.items():
        g = df[df["year"] == year]
        ax.plot(g["woy"], g["hotspots"], color=color, lw=2, label=str(year))
    ax.plot([], [], color=CONTEXT, lw=1, label="other years")

    ax.set_yscale("log")
    _thousands(ax)
    ax.set_xlim(1, 53)
    months = pd.date_range("2023-01-01", periods=12, freq="MS")
    ax.set_xticks([d.isocalendar().week if d.isocalendar().week < 50 else 1 for d in months],
                  [d.strftime("%b") for d in months])
    ax.set_title("Seasonal profile: hotspots per week by week of year (log scale)")
    ax.set_ylabel("hotspots / week")
    ax.legend(loc="upper left")
    return fig


def main() -> None:
    w = pd.read_parquet(PROCESSED)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, make in [("weekly_series", weekly_series), ("seasonal_profile", seasonal_profile)]:
        path = FIGURES / f"{name}.png"
        make(w).savefig(path)
        print(f"saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
