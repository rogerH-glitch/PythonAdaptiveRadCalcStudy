import math
import numpy as np

from src.analytical import vf_point_rect_to_point_parallel

# Geometry used across tests
EM_W, EM_H = 5.0, 2.0
RC_W, RC_H = 5.0, 2.0
S = 1.0  # setback

def test_offset_dx_inside_receiver_peak_at_emitter_center():
    """
    Receiver center shifted +0.5 in x (still overlapping the emitter center).
    Prediction: the local peak across the receiver occurs at global (0,0).
    Therefore F at (rx,ry)=(0,0) > F at the receiver center (rx,ry)=(+0.5,0).
    """
    from src.analytical import vf_point_rect_to_point_parallel
    EM_W, EM_H, S = 5.0, 2.0, 1.0
    dx = +0.5

    F_at_center = vf_point_rect_to_point_parallel(EM_W, EM_H, S, rx=dx, ry=0.0, nx=200, ny=200)
    F_at_em_center = vf_point_rect_to_point_parallel(EM_W, EM_H, S, rx=0.0, ry=0.0, nx=200, ny=200)

    assert 0.0 <= F_at_center <= 1.0
    assert 0.0 <= F_at_em_center <= 1.0

    # Center of emitter should be higher; require at least a 0.5% relative gap
    rel_gap = (F_at_em_center - F_at_center) / max(F_at_em_center, 1e-12)
    assert rel_gap > 0.005, f"rel_gap too small: {rel_gap:.5f}"

def test_offset_dx_outside_receiver_peak_on_near_boundary():
    """
    Receiver center shifted +3.0 in x; emitter center (0,0) is now outside the receiver.
    Prediction: peak occurs at the receiver boundary closest to (0,0), i.e. x = dx - RC_W/2.
    """
    dx = +3.0
    x_left  = dx - RC_W/2   # nearest boundary to x=0
    x_mid   = dx            # receiver center
    x_right = dx + RC_W/2   # far boundary

    F_left  = vf_point_rect_to_point_parallel(EM_W, EM_H, S, rx=x_left,  ry=0.0, nx=140, ny=140)
    F_mid   = vf_point_rect_to_point_parallel(EM_W, EM_H, S, rx=x_mid,   ry=0.0, nx=140, ny=140)
    F_right = vf_point_rect_to_point_parallel(EM_W, EM_H, S, rx=x_right, ry=0.0, nx=140, ny=140)

    # Expect the nearest boundary to dominate
    assert F_left >= F_mid and F_left >= F_right
    # And still within physical bounds
    for F in (F_left, F_mid, F_right):
        assert 0.0 <= F <= 1.0

def test_offset_dy_inside_receiver_vertical_behavior():
    """
    Vertical offset case: center shifted +0.7 in y; (0,0) still inside receiver.
    Prediction: F(0,0) > F(center at y=+0.7).
    """
    dy = +0.7
    F_at_center_y = vf_point_rect_to_point_parallel(EM_W, EM_H, S, rx=0.0, ry=dy, nx=140, ny=140)
    F_at_em_center = vf_point_rect_to_point_parallel(EM_W, EM_H, S, rx=0.0, ry=0.0, nx=140, ny=140)

    assert F_at_em_center > F_at_center_y
    assert 0.0 <= F_at_center_y <= 1.0 and 0.0 <= F_at_em_center <= 1.0
