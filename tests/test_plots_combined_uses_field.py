import numpy as np
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

def test_combined_plot_writes_when_field_present(tmp_path):
    from src.viz.plots import plot_geometry_and_heatmap
    # minimal synthetic field
    y = np.linspace(-2, 2, 41)
    z = np.linspace(-1, 1, 21)
    Y, Z = np.meshgrid(y, z, indexing="xy")
    F = np.exp(-((Y-0.5)**2 + (Z-0.4)**2))
    result = {
        "Y": Y, "Z": Z, "F": F,
        "y_peak": 0.5, "z_peak": 0.4, "F_peak": float(F.max()),
        "We": 5.0, "He": 2.0, "Wr": 5.0, "Hr": 2.0,
    }
    out = tmp_path / "combined.png"
    plot_geometry_and_heatmap(result=result, eval_mode="grid", method="adaptive", setback=3.0, out_png=out)
    assert out.exists()


