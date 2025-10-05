from src.montecarlo import vf_point_montecarlo


def test_montecarlo_point_center_runs_and_returns_float():
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
    vf, meta = vf_point_montecarlo((0.0, 0.0), geom, samples=20000, seed=123, target_rel_ci=0.10, max_iters=3)
    assert isinstance(vf, float)
    assert "iters" in meta and meta["iters"] >= 1

