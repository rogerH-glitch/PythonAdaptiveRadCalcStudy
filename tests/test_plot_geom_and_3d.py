import os
import sys
from pathlib import Path
import pytest


def test_parser_has_plot_geom_flag():
    from src.cli_parser import create_parser, normalize_args
    parser = create_parser()
    args = parser.parse_args(["--method","analytical","--emitter","5","2","--setback","1","--plot-both"])
    args = normalize_args(args)
    assert getattr(args, "_plot_mode", None) == "both"


@pytest.mark.skipif(pytest.importorskip("plotly", reason="plotly not installed") is None, reason="plotly not installed")
def test_plot_geometry_3d_writes_file(tmp_path):
    from src.viz.plot3d import plot_geometry_3d
    out_html = tmp_path / "geom.html"
    
    # Create a result dictionary with the required fields
    result = {
        "We": 5.0, "He": 2.0, "Wr": 5.0, "Hr": 2.0,
        "emitter_center": (1.0, 0.0, 0.0),
        "receiver_center": (0.0, 0.5, 0.0),
        "setback": 3.0,
        "rotate_axis": "z",
        "angle": 0.0,
        "angle_pivot": "toe",
        "rotate_target": "emitter",
        "dy": 0.0, "dz": 0.0
    }
    
    plot_geometry_3d(result, str(out_html))
    assert out_html.exists()
    assert out_html.read_text(encoding="utf-8").lower().count("<html") >= 1


