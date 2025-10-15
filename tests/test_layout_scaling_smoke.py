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


def test_layout_axes_area_fraction_is_reasonable():
    geom = _fake_geom()
    gy = np.linspace(-2.55, 2.55, 21)
    gz = np.linspace(-1.05, 1.05, 21)
    Y, Z = np.meshgrid(gy, gz, indexing="ij")
    F = np.exp(-((Y + 0.25) ** 2 + (Z - 0.10) ** 2) / (0.35 ** 2))

    fig, _ = plot_geometry_and_heatmap(
        result={"We": 2.0, "He": 2.0, "Wr": 2.0, "Hr": 2.0, "rotate_axis": "z", "rotate_target": "emitter", "angle_pivot": "toe", "angle": 0.0},
        eval_mode="grid", method="adaptive", setback=3.0, out_png="ignore.png", return_fig=True,
        vf_field=F, vf_grid={"y": gy, "z": gz}, prefer_eval_field=True,
        adaptive_peak_yz=(-0.25, 0.0), marker_mode="adaptive", title="layout smoke",
    )

    ax_positions = [ax.get_position() for ax in fig.axes]
    areas = [p.width * p.height for p in ax_positions]
    total_area = sum(areas)
    assert total_area >= 0.35, f"Axes too small; total area={total_area:.3f}"
    largest_three = sorted(areas, reverse=True)[:3]
    assert largest_three[0] >= largest_three[1] >= largest_three[2]
    assert (largest_three[0] / max(largest_three[2], 1e-6)) <= 3.0

    plt.close(fig)


