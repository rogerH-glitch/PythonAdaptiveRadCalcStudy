from types import SimpleNamespace
from pathlib import Path
from src.cli_results import _csv_path

def test_outdir_same_as_user_when_building_csv_path(tmp_path):
    # Simulate args as they appear right before writing/printing
    args = SimpleNamespace(outdir=str(tmp_path/"results"), _outdir_user=str(tmp_path/"results"), method="adaptive")
    p = _csv_path(args, "adaptive")
    # Path should be exactly under the user path (no double 'results')
    expected_path = tmp_path / "results" / "adaptive.csv"
    assert p == expected_path
    # Ensure no double 'results' in the path
    path_str = str(p).replace("\\", "/")
    assert path_str.count("results") == 1
