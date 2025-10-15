import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.viz.plots import plot_geometry_and_heatmap


def _fake_geom():
    zeros2 = np.zeros((4, 2))
    zeros3 = np.zeros((4, 3))
    return {
        "xy": {"emitter": zeros2[:2], "receiver": zeros2[:2]},
        "corners3d": {"emitter": zeros3, "receiver": zeros3},
    }


def test_legends_anchor_below_axes():
    geom = _fake_geom()
    gy = np.linspace(-2.55, 2.55, 21)
    gz = np.linspace(-1.05, 1.05, 21)
    Y, Z = np.meshgrid(gy, gz, indexing="ij")
    F = np.exp(-((Y + 0.25) ** 2 + (Z - 0.10) ** 2) / (0.35 ** 2))

    fig, (ax_plan, ax_elev, ax_hm) = plot_geometry_and_heatmap(
        result={"We": 2.0, "He": 2.0, "Wr": 2.0, "Hr": 2.0, "rotate_axis": "z", "rotate_target": "emitter", "angle_pivot": "toe", "angle": 0.0},
        eval_mode="grid", method="adaptive", setback=3.0, out_png="ignore.png", return_fig=True,
        vf_field=F, vf_grid={"y": gy, "z": gz}, adaptive_peak_yz=(-0.25, 0.0), marker_mode="adaptive", title="legend placement test",
    )

    # Check legends exist and are below the axes
    for ax in (ax_plan, ax_elev):
        lg = ax.get_legend()
        assert lg is not None, "Legend not found on XY/XZ axis"
        # Map legend anchor bbox to axes coordinates
        bbox = lg.get_bbox_to_anchor().transformed(ax.transAxes.inverted())
        assert bbox.y0 < 0.0, f"Legend anchor not below axis (y0={bbox.y0:.3f})"

    plt.close(fig)


