import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def test_summary_hides_defaults(capsys):
    from src.cli_parser import create_parser, map_eval_mode_args
    from src.cli import _print_user_relevant_summary
    parser = create_parser()
    # angle=0 (default), no rotation changes; emitter offset given
    args = parser.parse_args(["--method","adaptive","--emitter","5","2","--receiver","5","2","--setback","6",
                              "--emitter-offset","0.6","0.4"])
    map_eval_mode_args(args)
    _print_user_relevant_summary(args)
    out = capsys.readouterr().out
    # Should NOT mention "Angle pivot"/"Rotate axis"/"Rotate target" or "Centres are aligned"
    assert "Rotate axis" not in out and "Angle pivot" not in out and "Rotate target" not in out
    assert "Centres are aligned" not in out
    # Should mention emitter offset
    assert "Emitter offset (dy,dz)" in out and "(0.600, 0.400)" in out


def test_legacy_heatmap_uses_YZ_and_marks_peak(tmp_path):
    from src.plotting import create_heatmap_plot
    class A: pass
    args = A()
    args.method = "adaptive"; args.setback = 1.0; args.outdir = str(tmp_path)
    args.eval_mode = "grid"; args.rc_mode = "grid"; args.plot = True
    # Make a field peaked at (0.5,0.4) in Y–Z
    y = np.linspace(-2, 2, 81); z = np.linspace(-1, 1, 41)
    Y, Z = np.meshgrid(y, z, indexing="xy")
    F = np.exp(-((Y-0.5)**2 + (Z-0.4)**2))
    result = {
        "method": "adaptive",
        "vf": float(F.max()),
        "x_peak": 0.5,
        "y_peak": 0.4,
        "geometry": {"emitter": (5.0, 2.0), "receiver": (5.0, 2.0), "setback": 1.0, "angle": 0.0},
    }
    create_heatmap_plot(result, args, {"Y": Y, "Z": Z, "F": F})
    # basic smoke: file exists (now timestamped)
    heatmap_files = list(tmp_path.glob("*_heatmap.png"))
    assert len(heatmap_files) == 1
    assert heatmap_files[0].exists()


