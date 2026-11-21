import numpy as np
import pandas as pd
import pytest

from fires.backtest import load_series, make_origins
from fires.models import (
    ONI_LAG_CANDIDATES,
    PUBLICATION_DELAY_WEEKS,
    WEEK,
    lagged_oni_exog,
    make_fourier_arima_oni,
)

ORIGIN_POS = 420  # weeks into the sample


def _fake_oni(index: pd.DatetimeIndex, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(np.cumsum(rng.normal(0, 0.3, len(index))), index=index, name="oni")


def test_lagged_exog_values_and_masking() -> None:
    idx = pd.date_range("2012-01-16", periods=200, freq="7D")
    oni = pd.Series(np.arange(200, dtype=float), index=idx)
    target = idx[100:103]
    ex = lagged_oni_exog(oni, target, 30)
    assert list(ex) == [70.0, 71.0, 72.0]
    origin = idx[100]  # published by origin: week <= origin - 9
    ex = lagged_oni_exog(oni, idx[100:130], 21, origin=origin)
    src = idx[100:130] - 21 * WEEK
    assert ex.notna().to_numpy().tolist() == (src <= origin - 9 * WEEK).tolist()


def test_too_short_lag_rejected() -> None:
    with pytest.raises(ValueError):
        make_fourier_arima_oni(2, (0, 1, 2), "n", lag_weeks=20)


@pytest.mark.parametrize("lag", ONI_LAG_CANDIDATES)
def test_forecast_ignores_oni_published_after_origin(lag: int) -> None:
    df = load_series()
    origin = make_origins(df.index)[0]
    train = df["y"].loc[:origin]
    oni = _fake_oni(df.index)
    f = make_fourier_arima_oni(2, (0, 1, 2), "n", lag, oni=oni)(train, 12)

    # Altering ONI after origin - 9 weeks must not change anything.
    late = oni.index > origin - PUBLICATION_DELAY_WEEKS * WEEK
    oni2 = oni.copy()
    oni2[late] = oni2[late] + 100.0
    g = make_fourier_arima_oni(2, (0, 1, 2), "n", lag, oni=oni2)(train, 12)
    pd.testing.assert_frame_equal(f, g)
    assert np.isfinite(f["y_log"]).all()

    # Sanity: altering ONI that WAS published does change it (the regressor is used).
    oni3 = oni.copy()
    early = oni3.index <= origin - PUBLICATION_DELAY_WEEKS * WEEK
    oni3[early] = oni3[early] + np.linspace(0, 5, early.sum())
    h = make_fourier_arima_oni(2, (0, 1, 2), "n", lag, oni=oni3)(train, 12)
    assert not np.allclose(f["y_log"], h["y_log"])


def test_forecast_weeks_use_only_published_oni() -> None:
    """For every candidate lag and h <= 12, the ONI week needed is published at the origin."""
    origin = pd.Timestamp("2021-01-04")
    idx = pd.date_range("2012-01-16", "2022-01-01", freq="7D")
    oni = pd.Series(1.0, index=idx)
    fut = pd.DatetimeIndex([origin + WEEK * h for h in range(1, 13)])
    for lag in ONI_LAG_CANDIDATES:
        assert lagged_oni_exog(oni, fut, lag, origin=origin).notna().all()
    assert lagged_oni_exog(oni, fut, 20, origin=origin).isna().any()
