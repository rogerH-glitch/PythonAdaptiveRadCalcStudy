import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.viz.plots import plot_geometry_and_heatmap


def _geom_rects(setback=3.0, w=5.1, h=2.1):
    half_w = w * 0.5
    half_h = h * 0.5
    em = np.array([
        [0.0, -half_w, -half_h],
        [0.0,  half_w, -half_h],
        [0.0,  half_w,  half_h],
        [0.0, -half_w,  half_h],
    ], float)
    rc = em.copy(); rc[:, 0] = setback
    def xy_from(c): return c[:, [0, 1]]
    def xz_from(c): return c[:, [0, 2]]
    return {
        "xy": {"emitter": xy_from(em), "receiver": xy_from(rc)},
        "corners3d": {"emitter": em, "receiver": rc},
        "emitter": {"xy": xy_from(em), "xz": xz_from(em)},
        "receiver": {"xy": xy_from(rc), "xz": xz_from(rc)},
    }


def test_xy_xz_have_padding_beyond_geometry():
    geom = _geom_rects()
    gy = np.linspace(-2.55, 2.55, 9)
    gz = np.linspace(-1.05, 1.05, 9)
    Y, Z = np.meshgrid(gy, gz, indexing="ij")
    F = np.zeros_like(Y) + 1.0

    fig, (ax_xy, ax_xz, ax_hm) = plot_geometry_and_heatmap(
        result={"We": 5.1, "He": 2.1, "Wr": 5.1, "Hr": 2.1, "rotate_axis": "z", "rotate_target": "emitter", "angle_pivot": "toe", "angle": 0.0},
        eval_mode="grid", method="adaptive", setback=3.0, out_png="ignore.png", return_fig=True,
        vf_field=F, vf_grid={"y": gy, "z": gz}, adaptive_peak_yz=(0.0, 0.0), marker_mode="adaptive", title="padding test",
        prefer_eval_field=True,
    )

    xs = np.concatenate([geom["emitter"]["xy"][:, 0], geom["receiver"]["xy"][:, 0]])
    ys = np.concatenate([geom["emitter"]["xy"][:, 1], geom["receiver"]["xy"][:, 1]])
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    span_x = x_max - x_min
    span_y = y_max - y_min

    x0, x1 = ax_xy.get_xlim(); y0, y1 = ax_xy.get_ylim()
    assert (x0 < x_min - 0.03 * span_x) and (x1 > x_max + 0.03 * span_x)
    assert (y0 < y_min - 0.03 * span_y) and (y1 > y_max + 0.03 * span_y)

    xz_xs = np.concatenate([geom["emitter"]["xz"][:, 0], geom["receiver"]["xz"][:, 0]])
    xz_zs = np.concatenate([geom["emitter"]["xz"][:, 1], geom["receiver"]["xz"][:, 1]])
    xx_min, xx_max = float(xz_xs.min()), float(xz_xs.max())
    zz_min, zz_max = float(xz_zs.min()), float(xz_zs.max())
    span_xx = xx_max - xx_min
    span_zz = zz_max - zz_min

    x0e, x1e = ax_xz.get_xlim(); z0e, z1e = ax_xz.get_ylim()
    assert (x0e < xx_min - 0.03 * span_xx) and (x1e > xx_max + 0.03 * span_xx)
    assert (z0e < zz_min - 0.03 * span_zz) and (z1e > zz_max + 0.03 * span_zz)

    plt.close(fig)


