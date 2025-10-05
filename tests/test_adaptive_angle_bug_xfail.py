import pytest
from src.peak_locator import local_vf


@pytest.mark.xfail(reason="Known issue: angle invariance in adaptive integrand; fix pending")
def test_vf_should_drop_with_large_yaw_angle():
    geom = {
        "emitter_width": 5.0,
        "emitter_height": 2.0,
        "setback": 1.0,
        "rotate_axis": "z",
        "angle_pivot": "toe",
        "dy": 0.0,
        "dz": 0.0,
        # NOTE: 'angle' is set inside the loop below; this test documents the bug.
    }
    params = {"rel_tol": 3e-3, "abs_tol": 1e-6, "max_depth": 10}
    yR, zR = 0.0, 0.0
    geom["angle"] = 0.0
    F0, _ = local_vf("adaptive", (yR, zR), geom, params)
    geom["angle"] = 80.0
    F80, _ = local_vf("adaptive", (yR, zR), geom, params)
    assert F80 < 0.7 * F0  # expected drop (currently fails)

