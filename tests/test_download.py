from pathlib import Path

from fires.download import read_oni

SAMPLE = """ SEAS  YR   TOTAL   ANOM
  NDJ 1999  24.90  -1.66
  DJF 2000  24.71  -1.65
  JFM 2000  25.05  -1.41
"""


def test_read_oni_assigns_centre_month(tmp_path: Path) -> None:
    f = tmp_path / "oni.ascii.txt"
    f.write_text(SAMPLE)
    df = read_oni(f)
    assert df["month"].dt.strftime("%Y-%m").tolist() == ["1999-12", "2000-01", "2000-02"]
    assert df["oni"].tolist() == [-1.66, -1.65, -1.41]
