from types import SimpleNamespace
from pathlib import Path


def test_csv_path_not_double_prefixed(tmp_path):
    from src.cli_results import _csv_path
    args = SimpleNamespace(outdir=str(tmp_path / "results"), method="adaptive")
    p = _csv_path(args, "adaptive")
    assert p == tmp_path / "results" / "adaptive.csv"   # not tmp/results/results/...
