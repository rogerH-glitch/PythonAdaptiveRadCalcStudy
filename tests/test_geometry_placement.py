import numpy as np
from src.geometry import PanelSpec, PlacementOpts, place_panels


def _approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_yaw_edge_pivot_preserves_min_setback():
    em = PanelSpec(3.0, 2.4)
    rc = PanelSpec(3.0, 2.4)
    g = place_panels(
        em,
        rc,
        setback=1.5,
        opts=PlacementOpts(angle_deg=25, rotate_axis="z", pivot="toe", align_centres=True),
    )
    P0, P1 = g["emitter_segment_xy"]
    assert _approx(g["receiver_plane_x"], 1.5, 1e-12)
    assert _approx(P0[0], 0.0, 1e-12)


def test_pitch_edge_pivot_preserves_min_setback():
    em = PanelSpec(3.0, 2.4)
    rc = PanelSpec(3.0, 2.4)
    g = place_panels(
        em,
        rc,
        setback=1.5,
        opts=PlacementOpts(angle_deg=15, rotate_axis="y", pivot="toe", align_centres=True),
    )
    Q0, Q1 = g["emitter_segment_xz"]
    assert _approx(g["receiver_plane_x"], 1.5, 1e-12)
    assert _approx(Q0[0], 0.0, 1e-12)


def test_offsets_between_centres_honoured():
    em = PanelSpec(5.0, 2.0)
    rc = PanelSpec(5.0, 2.0)
    g = place_panels(
        em,
        rc,
        setback=1.0,
        opts=PlacementOpts(
            angle_deg=20,
            rotate_axis="z",
            pivot="toe",
            align_centres=False,
            offset_dy=0.6,
            offset_dz=0.4,
        ),
    )
    (y_e, z_e) = g["centres"]["emitter"]
    (y_r, z_r) = g["centres"]["receiver"]
    dy = y_r - y_e
    dz = z_r - z_e
    assert abs(dy - 0.6) < 1e-9
    assert abs(dz - 0.4) < 1e-9

