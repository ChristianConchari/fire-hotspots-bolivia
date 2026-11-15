import numpy as np
import pandas as pd

from fires import eda


def test_transforms_lengths_and_diffs() -> None:
    y = pd.Series(np.arange(200.0), index=pd.date_range("2012-01-23", periods=200, freq="7D"))
    t = eda.transforms(y)
    assert len(t["d1"]) == 199 and len(t["d52"]) == 148 and len(t["d1d52"]) == 147
    assert (t["d1"] == 1).all() and (t["d52"] == 52).all() and (t["d1d52"] == 0).all()


def test_identification_sample_has_no_gaps() -> None:
    samples, n_nan = eda.identification_samples()
    assert n_nan["ident"] == 0
    assert samples["ident"].index[-1] == eda.IDENT_END
    assert not samples["full"].isna().any()
