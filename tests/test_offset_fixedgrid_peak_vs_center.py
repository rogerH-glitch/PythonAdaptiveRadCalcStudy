from src.analytical import vf_point_rect_to_point_parallel
from src.fixed_grid import vf_fixed_grid

def test_fixedgrid_peak_exceeds_offset_center_value():
    """
    For an inside-overlap offset, fixed-grid (which samples across receiver)
    should find a value >= the value at the receiver center.
    """
    em_w, em_h = 5.0, 2.0
    rc_w, rc_h = 5.0, 2.0
    setback = 1.0
    dx = +0.5  # offset in x (receiver center at +0.5)

    # Analytical value at receiver center (global coords)
    F_center = vf_point_rect_to_point_parallel(em_w, em_h, setback, rx=dx, ry=0.0, nx=120, ny=120)

    # Run fixed-grid with a modest receiver sampling (internally)
    res = vf_fixed_grid(em_w, em_h, rc_w, rc_h, setback,
                        grid_nx=120, grid_ny=120, quadrature="centroid", time_limit_s=10)

    assert res["status"] in ("converged", "reached_limits")
    # Fixed grid max across receiver should be >= value at the receiver center
    assert res["vf"] >= F_center - 1e-6
