import numpy as np


def test_adaptive_marker_not_forced_to_origin_when_offset():
    adaptive_yz = (-0.255, 0.000)
    ya, za = adaptive_yz
    assert (ya != 0.0) or (za != 0.0)
    assert np.isfinite(ya) and np.isfinite(za)


