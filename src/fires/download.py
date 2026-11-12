"""Download the raw data: FIRMS VIIRS S-NPP hotspots for Bolivia and the NOAA ONI index.

Usage:
    uv run python -m fires.download                 # everything, skips files already present
    uv run python -m fires.download --only firms --years 2012 2025
    uv run python -m fires.download --force         # download again

Files land in data/raw/ (git-ignored). Every download is recorded in
data/manifest.json (committed) with its URL, size, SHA-256 and date, so the
exact inputs of each result can be traced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.json"

# FIRMS publishes one CSV per country and year (standard processing, no MAP_KEY needed).
FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/{year}/viirs-snpp_{year}_Bolivia.csv"
FIRMS_FIRST_YEAR = 2012  # VIIRS S-NPP starts 2012-01-20
FIRMS_LAST_YEAR = 2025  # last complete year; 2026 is not in the yearly archive yet

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

USER_AGENT = "fire-hotspots-bolivia/0.1 (academic time-series project)"

# ONI seasons are 3-month running means; each is assigned to its centre month.
ONI_SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _record(entry: dict) -> None:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest[entry["file"]] = entry
    MANIFEST.write_text(json.dumps(dict(sorted(manifest.items())), indent=2) + "\n")


def fetch(url: str, dest: Path, force: bool = False) -> Path:
    """Download url to dest atomically (via a .part file) and record it in the manifest."""
    if dest.exists() and not force:
        print(f"skip  {dest.relative_to(ROOT)} (exists)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as resp, part.open("wb") as out:
        shutil.copyfileobj(resp, out)
    part.replace(dest)
    _record(
        {
            "file": str(dest.relative_to(ROOT)),
            "url": url,
            "bytes": dest.stat().st_size,
            "sha256": _sha256(dest),
            "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )
    print(f"saved {dest.relative_to(ROOT)} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def download_firms(first: int, last: int, force: bool = False) -> list[Path]:
    return [
        fetch(FIRMS_URL.format(year=y), RAW / "firms" / f"viirs-snpp_{y}_Bolivia.csv", force)
        for y in range(first, last + 1)
    ]


def download_oni(force: bool = False) -> Path:
    return fetch(ONI_URL, RAW / "oni" / "oni.ascii.txt", force)


def read_oni(path: Path | None = None) -> pd.DataFrame:
    """Parse oni.ascii.txt into one row per month: month (period start), oni (anomaly, °C)."""
    path = path or RAW / "oni" / "oni.ascii.txt"
    df = pd.read_csv(path, sep=r"\s+")
    month = df["SEAS"].map({s: i + 1 for i, s in enumerate(ONI_SEASONS)})
    if month.isna().any():
        raise ValueError(f"unknown ONI seasons: {sorted(df.loc[month.isna(), 'SEAS'].unique())}")
    out = pd.DataFrame(
        {
            "month": pd.to_datetime({"year": df["YR"], "month": month, "day": 1}),
            "oni": df["ANOM"].astype(float),
        }
    )
    return out.sort_values("month").reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", choices=["firms", "oni"], help="download one source only")
    p.add_argument("--years", nargs=2, type=int, metavar=("FIRST", "LAST"),
                   default=[FIRMS_FIRST_YEAR, FIRMS_LAST_YEAR], help="FIRMS year range, inclusive")
    p.add_argument("--force", action="store_true", help="download again even if the file exists")
    args = p.parse_args()

    if args.only in (None, "firms"):
        download_firms(*args.years, force=args.force)
    if args.only in (None, "oni"):
        download_oni(force=args.force)


if __name__ == "__main__":
    main()
