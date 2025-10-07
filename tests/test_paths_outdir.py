from pathlib import Path
from src.util.paths import get_outdir


def test_get_outdir_uses_given_path(tmp_path):
    out = get_outdir(tmp_path / "results")
    assert out == tmp_path / "results"
    assert out.exists()
    # Passing a nested path stays nested (no extra prefixing)
    out2 = get_outdir(tmp_path / "custom" / "nested")
    assert out2 == tmp_path / "custom" / "nested"
    assert out2.exists()
