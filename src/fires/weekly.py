"""Build the weekly modelling table from the raw downloads.

Usage:
    uv run python -m fires.weekly        # writes data/processed/weekly.parquet

Rules (see the execution plan in the vault):
- Hotspots: keep type 0 (presumed vegetation fire) and drop low confidence.
- A day with no detection of any kind (any type, any confidence) is treated as
  *not observed* (sensor outage), not as a day with zero fires.
- Weeks are ISO weeks, Monday start, indexed by their Monday.
- A week with up to MAX_MISSING_DAYS unobserved days is rescaled by 7 / days observed;
  with more, its count is missing (NaN), never 0.
- ONI is monthly; each week takes the value of the month that contains its Thursday
  (the ISO-week convention). Lags are applied at modelling time, not here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fires.download import RAW, ROOT, read_oni

PROCESSED = ROOT / "data" / "processed" / "weekly.parquet"
MAX_MISSING_DAYS = 2
KEEP_TYPES = {0}
DROP_CONFIDENCE = {"l"}


def load_firms(paths: list[Path] | None = None) -> pd.DataFrame:
    paths = paths or sorted((RAW / "firms").glob("viirs-snpp_*_Bolivia.csv"))
    if not paths:
        raise FileNotFoundError("no FIRMS files in data/raw/firms; run `python -m fires.download`")
    cols = ["acq_date", "confidence", "type"]
    df = pd.concat((pd.read_csv(p, usecols=cols) for p in paths), ignore_index=True)
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    return df


def daily_counts(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per day over whole ISO weeks: filtered hotspot count and whether the day was observed."""
    keep = raw["type"].isin(KEEP_TYPES) & ~raw["confidence"].isin(DROP_CONFIDENCE)
    counts = raw.loc[keep].groupby("acq_date").size()
    observed_days = raw["acq_date"].unique()

    start = raw["acq_date"].min().to_period("W-SUN").start_time
    end = raw["acq_date"].max().to_period("W-SUN").end_time.normalize()
    days = pd.date_range(start, end, freq="D", name="date")
    return pd.DataFrame(
        {
            "hotspots": counts.reindex(days, fill_value=0).astype(int),
            "observed": days.isin(observed_days),
        }
    )


def to_weekly(daily: pd.DataFrame, max_missing_days: int = MAX_MISSING_DAYS) -> pd.DataFrame:
    week = daily.index.to_period("W-SUN").start_time
    g = daily.groupby(week)
    out = pd.DataFrame(
        {
            "hotspots_raw": g["hotspots"].sum(),
            "days_observed": g["observed"].sum().astype(int),
        }
    )
    out.index.name = "week"
    ok = out["days_observed"] >= 7 - max_missing_days
    out["hotspots"] = (out["hotspots_raw"] * 7 / out["days_observed"]).where(ok).round()
    return out


def add_oni(weekly: pd.DataFrame, oni: pd.DataFrame) -> pd.DataFrame:
    thursday_month = (weekly.index + pd.Timedelta(days=3)).to_period("M").to_timestamp()
    by_month = oni.set_index("month")["oni"]
    return weekly.assign(oni=by_month.reindex(thursday_month).to_numpy())


def build() -> pd.DataFrame:
    weekly = to_weekly(daily_counts(load_firms()))
    weekly = add_oni(weekly, read_oni())
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(PROCESSED)
    return weekly


def main() -> None:
    w = build()
    missing = w[w["hotspots"].isna()]
    print(f"saved {PROCESSED.relative_to(ROOT)}: {len(w)} weeks, "
          f"{w.index.min().date()} → {w.index.max().date()}, {len(missing)} missing")
    if len(missing):
        print(missing[["hotspots_raw", "days_observed"]].to_string())


if __name__ == "__main__":
    main()
