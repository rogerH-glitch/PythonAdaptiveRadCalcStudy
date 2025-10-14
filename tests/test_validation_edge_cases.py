import numpy as np
from src.viz.display_geom import build_display_geom
from src.viz.plots import plot_geometry_and_heatmap


class _R:
    # Defaults; individual tests override as needed.
    emitter_w = 1.0
    emitter_h = 1.0
    receiver_w = 1.0
    receiver_h = 1.0
    setback = 3.0
    angle = 0.0
    rotate_axis = "z"
    rotate_target = "emitter"
    angle_pivot = "toe"
    receiver_offset = (0.0, 0.0)


def _mk(overrides=None):
    a = _R()
    if overrides:
        for k, v in overrides.items():
            setattr(a, k, v)
    r = _R()
    for k in vars(a).keys():
        setattr(r, k, getattr(a, k))
    return a, r


def test_extreme_aspect_ratio_panels_still_plot_and_bounds_make_sense():
    # Tall, skinny emitter; wide, flat receiver
    args, res = _mk({
        "emitter_w": 0.2, "emitter_h": 5.0,
        "receiver_w": 8.0, "receiver_h": 0.3,
        "setback": 4.0, "angle": 15.0, "rotate_axis": "y", "rotate_target": "emitter",
    })
    geom = build_display_geom(args, res)
    xy = geom["xy_limits"]  # xmin,xmax,ymin,ymax
    xz = geom["xz_limits"]  # xmin,xmax,zmin,zmax
    # Nonzero spans and finite bounds
    assert np.isfinite(xy).all() and np.isfinite(xz).all()
    assert xy[1] > xy[0] and xy[3] > xy[2]
    assert xz[1] > xz[0] and xz[3] > xz[2]


def test_large_offsets_keep_receiver_within_plot_limits():
    args, res = _mk({
        "receiver_offset": (3.5, 2.25),
        "receiver_w": 2.0, "receiver_h": 1.0,
        "setback": 5.0,
    })
    geom = build_display_geom(args, res)
    r_xy = geom["xy"]["receiver"]
    # Both receiver XY endpoints should lie inside XY limits
    (x0, y0), (x1, y1) = r_xy
    xmin, xmax, ymin, ymax = geom["xy_limits"]
    assert xmin <= x0 <= xmax and ymin <= y0 <= ymax
    assert xmin <= x1 <= xmax and ymin <= y1 <= ymax
