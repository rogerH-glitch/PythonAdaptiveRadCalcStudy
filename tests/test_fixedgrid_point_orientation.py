from src.fixed_grid import vf_point_fixed_grid


def test_fixedgrid_point_center_runs_and_returns_float():
    geom = {
        "emitter_width": 5.0,
        "emitter_height": 2.0,
        "setback": 1.0,
        "rotate_axis": "z",
        "angle": 0.0,
        "angle_pivot": "toe",
        "dy": 0.0,
        "dz": 0.0,
    }
    F, meta = vf_point_fixed_grid((0.0, 0.0), geom, grid_nx=24, grid_ny=12, quadrature="centroid")
    assert isinstance(F, float)
    assert "grid_nx" in meta and "grid_ny" in meta

