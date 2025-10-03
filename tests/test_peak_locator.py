import math
import pytest

xfail = pytest.mark.xfail(reason="rc_mode=search + offsets/rotation not implemented yet", strict=False)

@xfail
def test_peak_locator_center_case(cli_runner=None):
    """
    Concentric, parallel case: peak should be ~center (0,0).
    When rc_mode=search is implemented, ensure the search returns near-center and F_peak ≈ center eval.
    """
    # Placeholder: once implemented, call a function like:
    # res = find_local_peak(em_w=5, em_h=2, rc_w=5, rc_h=2, setback=1, angle=0, rc_mode="search")
    # assert abs(res["x_peak"]) < 1e-3 and abs(res["y_peak"]) < 1e-3
    assert True

@xfail
def test_peak_locator_offset_shifts_peak():
    """
    With receiver offset dx=+0.5 m, the peak should shift toward +x.
    """
    # res = find_local_peak(..., receiver_offset=(+0.5, 0.0), rc_mode="search")
    # assert res["x_peak"] > 0.0
    assert True

@xfail
def test_peak_locator_rotation_shifts_peak():
    """
    With angle=5°, the peak should move toward the nearer projected region.
    """
    # res = find_local_peak(..., angle=5.0, rc_mode="search")
    # assert not (abs(res["x_peak"]) < 1e-3 and abs(res["y_peak"]) < 1e-3)
    assert True