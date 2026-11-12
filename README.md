# Fire hotspots in Bolivia — weekly forecasting

Final project for *Time Series Analysis I* (Master of Artificial Intelligence, FIUBA–UBA).

**Question:** how much does adding precipitation and ENSO (ONI) to a seasonal model improve the weekly forecast of fire-hotspot counts in Bolivia, compared with a seasonal naive baseline?

## Data

| Series | Source | File |
|---|---|---|
| Hotspots | NASA FIRMS, VIIRS S-NPP 375 m, yearly country archive for Bolivia, 2012–2025 (standard processing) | `data/raw/firms/viirs-snpp_<year>_Bolivia.csv` |
| ENSO | NOAA CPC Oceanic Niño Index (3-month running mean, assigned to the centre month) | `data/raw/oni/oni.ascii.txt` |

Raw files are not committed. `data/manifest.json` records the URL, size and SHA-256 of every downloaded file.

## Setup

```bash
uv sync
uv run python -m fires.download      # FIRMS 2012–2025 + ONI into data/raw/
uv run pytest
```

## Layout

```
src/fires/      pipeline: download, weekly aggregation, backtest, metrics
notebooks/      01_eda, 02_baselines, 03_exogenous, 04_robustness
data/raw/       downloads (git-ignored)
data/processed/ weekly table
results/        metrics tables and figures for the report
report/         report and slides
```
