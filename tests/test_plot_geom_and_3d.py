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
def test_geometry_3d_html_writes_file(tmp_path):
    from src.viz.plot3d import geometry_3d_html
    out_html = tmp_path / "geom.html"
    path = geometry_3d_html(
        emitter_center=(1.0,0.0,0.0),
        receiver_center=(0.0,0.5,0.0),
        We=5.0, He=2.0, Wr=5.0, Hr=2.0,
        out_html=str(out_html),
        include_plotlyjs="cdn",
    )
    assert Path(path).exists()
    assert out_html.read_text(encoding="utf-8").lower().count("<html") >= 1


