import numpy as np
import pandas as pd

from fires.backtest import load_series, make_origins, rolling_origin
from fires.models import fourier_terms, make_fourier_arima


def test_fourier_terms_shape_and_period() -> None:
    idx = pd.date_range("2012-01-02", periods=300, freq="7D")
    f = fourier_terms(idx, 3)
    assert list(f.columns) == ["sin_1", "cos_1", "sin_2", "cos_2", "sin_3", "cos_3"]
    assert f.loc[idx[0], "sin_1"] == 0 and f.loc[idx[0], "cos_1"] == 1
    # 365.25 days later the phase repeats: 1 harmonic at 52.1775 weeks
    t = pd.DatetimeIndex(["2013-01-02", "2014-01-01"])  # 366 and 365 days after origin
    g = fourier_terms(t, 1)
    assert abs(g["sin_1"].iloc[0] - g["sin_1"].iloc[1]) < 0.05


def test_fourier_arima_interface_on_few_origins() -> None:
    df = load_series()
    origins = make_origins(df.index)[[0, 20, 40]]
    for order, trend in (((1, 0, 1), "c"), ((0, 1, 1), "n")):
        out = rolling_origin(make_fourier_arima(2, order, trend), df, origins=origins)
        assert len(out) == 3 * 12
        assert not out["y_pred"].isna().any() and np.isfinite(out["y_pred"]).all()
        for o, g in out.groupby("origin"):
            assert list(g["h"]) == list(range(1, 13))
            assert g["target_week"].iloc[0] == o + pd.Timedelta(weeks=1)
