import numpy as np
import matplotlib
matplotlib.use("Agg")
from src.viz.plots import plot_geometry_and_heatmap


def test_both_markers_when_requested():
    geom = {
        "xy": {"emitter": np.zeros((2,2)), "receiver": np.zeros((2,2))},
        "corners3d": {"emitter": np.zeros((4,3)), "receiver": np.zeros((4,3))},
    }
    gy = np.linspace(-2.55, 2.55, 21)
    gz = np.linspace(-1.05, 1.05, 21)
    Y, Z = np.meshgrid(gy, gz, indexing="ij")
    F = np.exp(-((Y + 0.50) ** 2 + (Z - 0.10) ** 2) / (0.3 ** 2))
    fig, _ = plot_geometry_and_heatmap(
        result={"We": 2.0, "He": 2.0, "Wr": 2.0, "Hr": 2.0, "rotate_axis": "z", "rotate_target": "emitter", "angle_pivot": "toe", "angle": 0.0},
        eval_mode="grid", method="adaptive", setback=3.0, out_png="ignore.png", return_fig=True,
        vf_field=F, vf_grid={"y": gy, "z": gz}, adaptive_peak_yz=(-0.52, 0.08), prefer_eval_field=True,
        marker_mode="both", title="t",
    )
    meta = getattr(fig, "_vf_plot_meta", {})
    placed = meta.get("markers", {})
    assert placed.get("adaptive") is not None
    assert placed.get("grid") is not None


