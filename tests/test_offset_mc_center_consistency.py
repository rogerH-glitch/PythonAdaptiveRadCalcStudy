import math
from src.analytical import vf_point_rect_to_point_parallel
from src.montecarlo import vf_montecarlo

def test_mc_matches_analytical_at_offset_center_loose():
    """
    Compare MC (receiver center) with analytical at the same center point.
    Use a loose tolerance because MC variance is nonzero.
    """
    em_w, em_h = 5.1, 2.1
    rc_w, rc_h = 5.1, 2.1
    s = 1.0
    dx = +0.5

    F_center_anal = vf_point_rect_to_point_parallel(em_w, em_h, s, rx=dx, ry=0.0, nx=140, ny=140)

    res = vf_montecarlo(em_w, em_h, rc_w, rc_h, s,
                        samples=150000, target_rel_ci=0.03, max_iters=40,
                        seed=7, time_limit_s=10)
    F_center_mc = res["vf_mean"]

    # Loose agreement: within ~8–10% is fine for a quick test
    rel_err = abs(F_center_mc - F_center_anal) / max(F_center_anal, 1e-12)
    assert rel_err < 0.10
