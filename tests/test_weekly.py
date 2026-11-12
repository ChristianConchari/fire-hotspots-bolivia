import pandas as pd

from fires.weekly import add_oni, daily_counts, to_weekly


def _raw(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["acq_date", "confidence", "type"])
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    return df


def test_filters_and_zero_vs_unobserved_days() -> None:
    # Week of Mon 2024-07-01. Only a low-confidence detection on Tue: observed, zero kept fires.
    raw = _raw(
        [("2024-07-01", "n", 0), ("2024-07-01", "h", 0), ("2024-07-01", "n", 2),  # static source dropped
         ("2024-07-02", "l", 0)]
        + [(f"2024-07-0{d}", "n", 0) for d in range(3, 8)]
    )
    daily = daily_counts(raw)
    assert daily.loc["2024-07-01", "hotspots"] == 2
    assert daily.loc["2024-07-02", "hotspots"] == 0
    assert daily["observed"].all()
    w = to_weekly(daily)
    assert w.loc["2024-07-01", "hotspots"] == 7


def test_partial_week_rescaled_and_outage_week_missing() -> None:
    # Week 1 (Mon 2024-07-01): observed 5 days, 1 fire each -> 5 * 7/5 = 7.
    # Week 2 (Mon 2024-07-08): observed 4 days -> missing, not zero.
    days = ["2024-07-01", "2024-07-02", "2024-07-03", "2024-07-04", "2024-07-05",
            "2024-07-08", "2024-07-09", "2024-07-10", "2024-07-14"]
    w = to_weekly(daily_counts(_raw([(d, "n", 0) for d in days])))
    assert list(w["days_observed"]) == [5, 4]
    assert w.loc["2024-07-01", "hotspots"] == 7
    assert pd.isna(w.loc["2024-07-08", "hotspots"])
    assert w.loc["2024-07-08", "hotspots_raw"] == 4


def test_oni_uses_month_of_thursday() -> None:
    # Week of Mon 2024-01-29: Thursday is 2024-02-01 -> February's ONI.
    weekly = pd.DataFrame({"hotspots": [1.0]}, index=pd.DatetimeIndex(["2024-01-29"], name="week"))
    oni = pd.DataFrame({"month": pd.to_datetime(["2024-01-01", "2024-02-01"]), "oni": [1.8, 1.5]})
    assert add_oni(weekly, oni)["oni"].tolist() == [1.5]
