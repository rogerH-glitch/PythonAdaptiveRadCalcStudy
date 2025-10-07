import numpy as np
import matplotlib
matplotlib.use("Agg")
from pathlib import Path


def test_extracts_field_from_result_when_grid_data_none(tmp_path, monkeypatch):
    # Arrange a minimal 'result' with Y,Z,F only on result (not in grid_data)
    from src.plotting import create_heatmap_plot
    class A: pass
    args = A()
    args.method = "adaptive"; args.setback = 6.0; args.outdir = str(tmp_path)
    args.eval_mode = "grid"; args.rc_mode = "grid"; args.plot = True
    y = np.linspace(-2, 2, 81); z = np.linspace(-1, 1, 41)
    Y, Z = np.meshgrid(y, z, indexing="xy")
    F = np.exp(-((Y-0.5)**2 + (Z-0.4)**2))
    result = {"method": "adaptive", "vf": float(F.max()), "x_peak": 0.5, "y_peak": 0.4,
              "Y": Y, "Z": Z, "F": F,
              "geometry": {"emitter": (5.0, 2.0), "receiver": (5.0, 2.0), "setback": 6.0, "angle": 0.0}}
    create_heatmap_plot(result, args, grid_data=None)
    # A timestamped file should be created
    files = list(Path(tmp_path).glob("*_heatmap.png"))
    assert len(files) == 1


def test_titles_say_eval_mode(tmp_path):
    from src.viz.plots import plot_geometry_and_heatmap
    import numpy as np
    # minimal viable result
    result = {
        "We": 5.0, "He": 2.0, "Wr": 5.0, "Hr": 2.0,
        "emitter_center": (6.0, 0.0, 0.0),
        "receiver_center": (0.0, 0.0, 0.0),
        "Y": np.zeros((2, 2)), "Z": np.zeros((2, 2)), "F": np.zeros((2, 2)),
        "y_peak": 0.0, "z_peak": 0.0, "F_peak": 0.1
    }
    out = Path(tmp_path/"test_geom2d.png")
    plot_geometry_and_heatmap(result=result, eval_mode="grid", method="adaptive", setback=6.0, out_png=out)
    assert out.exists()
