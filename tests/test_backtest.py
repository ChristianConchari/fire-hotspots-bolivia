import numpy as np
import pandas as pd
import pytest

from fires.backtest import SAMPLE_END, SAMPLE_START, make_origins, rolling_origin
from fires.metrics import mae, metrics_by_horizon, relative_mae, rmse
from fires.models import future_index, seasonal_naive


def _frame(n_weeks: int = 14 * 52 + 10, seed: int = 0) -> pd.DataFrame:
    idx = pd.date_range(SAMPLE_START, periods=n_weeks, freq="7D")
    rng = np.random.default_rng(seed)
    hot = pd.Series(rng.integers(0, 1000, n_weeks).astype(float), index=idx)
    return pd.DataFrame({"hotspots": hot, "y": np.log1p(hot)})


def test_no_leakage() -> None:
    df = _frame()
    seen = []

    def spy(train: pd.Series, h: int) -> pd.Series:
        seen.append(train.index[-1])
        return pd.Series(np.zeros(h), index=future_index(train, h))

    origins = make_origins(df.index)
    out = rolling_origin(spy, df, origins=origins)
    assert seen == list(origins)  # last training week is exactly the origin
    assert (out["target_week"] > out["origin"]).all()
    assert (out["target_week"] - out["origin"] == pd.to_timedelta(out["h"] * 7, unit="D")).all()


def test_seasonal_naive_exact_on_period_52() -> None:
    idx = pd.date_range(SAMPLE_START, periods=14 * 52 + 12, freq="7D")
    pattern = np.random.default_rng(1).uniform(1, 9, 52)
    hot = pd.Series(np.expm1(np.tile(pattern, 15)[: len(idx)]), index=idx)
    df = pd.DataFrame({"hotspots": hot, "y": np.log1p(hot)})
    out = rolling_origin(seasonal_naive, df)
    assert len(out) > 0
    assert np.allclose(out["y_pred"], out["y_true"])
    m = metrics_by_horizon(out)
    assert (m["mae"] < 1e-8).all() and (m["rmse"] < 1e-8).all()


def test_seasonal_naive_falls_back_to_lag_104() -> None:
    idx = pd.date_range(SAMPLE_START, periods=120, freq="7D")
    y = pd.Series(np.arange(120, dtype=float), index=idx)
    train = y.iloc[:110].copy()
    t1 = train.index[-1] + pd.Timedelta(weeks=1)  # target index 110
    train.loc[t1 - pd.Timedelta(weeks=52)] = np.nan
    fc = seasonal_naive(train, 3)
    assert fc.iloc[0] == y.loc[t1 - pd.Timedelta(weeks=104)]
    assert fc.iloc[1] == y.loc[fc.index[1] - pd.Timedelta(weeks=52)]


def test_metrics_ignore_nan() -> None:
    t = [1.0, np.nan, 5.0]
    p = [2.0, 100.0, 3.0]
    assert mae(t, p) == pytest.approx(1.5)
    assert rmse(t, p) == pytest.approx(np.sqrt((1 + 4) / 2))


def test_eval_sets_and_relative_error() -> None:
    fc = pd.DataFrame(
        {
            "origin": pd.to_datetime(["2024-01-01"] * 3),
            "target_week": pd.to_datetime(["2024-01-08", "2024-06-03", "2025-01-06"]),
            "h": 1,
            "y_true": [10.0, 20.0, np.nan],
            "y_pred": [12.0, 10.0, 7.0],
        }
    )
    m = metrics_by_horizon(fc, "x").set_index("eval_set")
    assert m.loc["all", "mae"] == pytest.approx(6.0) and m.loc["all", "n"] == 2
    assert m.loc["excl_2024", "n"] == 0 and np.isnan(m.loc["excl_2024", "mae"])
    rel = relative_mae(metrics_by_horizon(fc), metrics_by_horizon(fc))
    assert rel.loc[rel["eval_set"] == "all", "rel_mae"].iloc[0] == pytest.approx(1.0)


def test_origins_count_and_range() -> None:
    idx = pd.date_range(SAMPLE_START, SAMPLE_END, freq="7D")
    assert idx[-1] == SAMPLE_END
    o = make_origins(idx)
    assert o[0] == pd.Timestamp("2021-01-04")
    assert o[-1] == pd.Timestamp("2025-10-06")
    assert len(o) == 63
    assert (o.to_series().diff().dropna() == pd.Timedelta(weeks=4)).all()
    assert o[-1] + pd.Timedelta(weeks=12) <= SAMPLE_END < o[-1] + pd.Timedelta(weeks=16)


def test_comparison_uses_common_targets_and_ratio() -> None:
    from fires.metrics import comparison_table

    base = pd.DataFrame({
        "origin": pd.to_datetime(["2023-01-02"] * 3),
        "target_week": pd.to_datetime(["2023-01-09", "2023-01-16", "2023-01-23"]),
        "h": [1, 1, 1],
        "y_true": [10.0, 20.0, np.nan],
        "y_pred": [14.0, 20.0, 5.0],
    })
    other = base.copy()
    other["y_pred"] = [12.0, np.nan, 5.0]  # NaN where naive has a value: target dropped for both
    t = comparison_table({"seasonal_naive": base, "m": other})
    a = t[t["eval_set"] == "all"].set_index("model")
    assert (a["n"] == 1).all()
    assert a.loc["seasonal_naive", "bias"] == 4.0 and a.loc["m", "bias"] == 2.0
    assert a.loc["m", "mae_ratio_vs_naive"] == 0.5
