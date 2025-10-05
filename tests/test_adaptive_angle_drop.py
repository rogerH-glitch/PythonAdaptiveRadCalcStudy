import math
import pytest

# This test documents the expected behaviour once the adaptive integrand
# uses the oriented emitter frame. Skipped until that wiring is complete.
@pytest.mark.skip(reason="Oriented adaptive integrand not yet wired into src/adaptive.py")
def test_vf_drops_with_yaw_angle():
    from src.adaptive import local_vf_adaptive  # expected API when wired

    geom = {
        "emitter_width": 5.0,
        "emitter_height": 2.0,
        "setback": 1.0,
        "rotate_axis": "z",
        "angle_pivot": "toe",
        "dy": 0.0, "dz": 0.0,
    }
    params = {"rel_tol": 3e-3, "abs_tol": 1e-6, "max_depth": 10}
    yR, zR = 0.0, 0.0

    for ang in (0.0, 20.0, 45.0, 80.0):
        geom["angle"] = ang
        F, _ = local_vf_adaptive((yR, zR), geom, params)
        if ang == 0.0:
            F0 = F
        if ang == 80.0:
            F80 = F

    assert F80 < 0.7 * F0, f"Expected significant drop with angle: F80={F80}, F0={F0}"

