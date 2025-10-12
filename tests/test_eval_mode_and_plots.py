import io
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.cli_parser import create_parser, map_eval_mode_args
from src.cli_results import print_single_line_summary
from src.rc_eval.grid_eval import sample_receiver_grid, evaluate_grid, peak_from_field
from src.viz.plots import _rect_wire_points, _heatmap


def _make_min_args(extra=None):
    if extra is None:
        extra = []
    base = ["--method","analytical","--emitter","5","2","--setback","1"]
    return base + extra


def test_eval_mode_alias_mapping_rc_to_eval(capsys):
    parser = create_parser()
    args = parser.parse_args(_make_min_args(["--rc-mode","grid"]))
    map_eval_mode_args(args)
    assert args.eval_mode == "grid"
    assert args.rc_mode == "grid"
    # deprecation note goes to stderr
    captured = capsys.readouterr()
    assert "deprecated" in captured.err.lower()


def test_eval_mode_alias_mapping_eval_to_rc():
    parser = create_parser()
    args = parser.parse_args(_make_min_args(["--eval-mode","search"]))
    map_eval_mode_args(args)
    assert args.rc_mode == "search"


def test_eval_grid_applies_offset():
    """
    Synthetic peak f(y,z) = -( (y)^2 + 4(z)^2 ). With dy=+0.5,dz=-0.2,
    the apparent maximum (in receiver frame) should occur at (y,z)=(-0.5,+0.2).
    """
    Y, Z = sample_receiver_grid(width=4.0, height=2.0, ny=81, nz=41)
    def fake_point_vf(y, z):  # vectorised
        return -(y**2 + 4.0*(z**2))
    F = evaluate_grid(fake_point_vf, Y, Z, dy=+0.5, dz=-0.2)
    Fpk, ypk, zpk = peak_from_field(F, Y, Z)
    assert abs(ypk - (-0.5)) < 0.06
    assert abs(zpk - (+0.2)) < 0.06
    assert np.isfinite(Fpk)


def test_peak_prints_even_when_not_verbose(capsys):
    # Minimal result dict for grid/search mode
    result = {
        "method":"adaptive",
        "vf":0.42,
        "calc_time":0.01,
        "rc_mode":"grid",
        "search_metadata":{},
        "x_peak":-0.5,  # y*
        "y_peak":0.0,   # z*
    }
    class A: pass
    args = A(); args.verbose=False
    print_single_line_summary(result, args)
    out = capsys.readouterr().out
    assert "Peak VF" in out and "(-0.500, 0.000)" in out


def test_rect_wire_points_closed_loop_and_sizes():
    x,y,z = _rect_wire_points(center_xyz=(1.0,2.0,3.0), w=2.0, h=1.0)
    assert np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1]) and np.isclose(z[0], z[-1])
    assert np.isclose(y.max()-y.min(), 2.0)
    assert np.isclose(z.max()-z.min(), 1.0)


def test_heatmap_labels_and_marker(tmp_path):
    Y, Z = np.mgrid[-1:1:51j, -0.5:0.5:41j]
    Y = Y.T; Z = Z.T
    F = np.exp(-((Y-0.3)**2 + (Z+0.1)**2))
    fig, ax = plt.subplots(1,1, figsize=(4,3))
    _heatmap(ax, Y, Z, F, 0.3, -0.1)
    assert ax.get_xlabel() == "Y (m)"
    assert ax.get_ylabel() == "Z (m)"
    fig.savefig(tmp_path/"hm.png")
    plt.close(fig)


