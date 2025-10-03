import numpy as np
from src.analytical import vf_point_rect_to_point_parallel

def test_local_gradient_around_center_monotone_decrease():
    """
    In concentric parallel geometry (peak at (0,0)), values should decrease as we move away a small delta.
    """
    EM_W, EM_H, S = 5.0, 2.0, 1.0
    deltas = [0.1, 0.2, 0.3]
    F0 = vf_point_rect_to_point_parallel(EM_W, EM_H, S, rx=0.0, ry=0.0, nx=200, ny=200)
    for d in deltas:
        Fx = vf_point_rect_to_point_parallel(EM_W, EM_H, S, rx=d, ry=0.0, nx=200, ny=200)
        Fy = vf_point_rect_to_point_parallel(EM_W, EM_H, S, rx=0.0, ry=d, nx=200, ny=200)
        assert Fx < F0 and Fy < F0
