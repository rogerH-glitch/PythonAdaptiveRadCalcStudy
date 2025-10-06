from src.util.filenames import ts_prefix, join_with_ts
from datetime import datetime
from pathlib import Path


def test_ts_prefix_format():
    fake = datetime(2025, 10, 4, 22, 26, 28)
    assert ts_prefix(fake) == "20251004_222628"


def test_join_with_ts(tmp_path):
    fake = datetime(2025, 10, 4, 22, 26, 28)
    out = join_with_ts(tmp_path, "heatmap.png", now=fake)
    assert out.name == "20251004_222628_heatmap.png"
    assert out.parent == tmp_path
