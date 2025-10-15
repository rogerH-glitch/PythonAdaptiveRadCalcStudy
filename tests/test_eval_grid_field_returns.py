import numpy as np

from src.viz.display_geom import build_display_geom
from src.cli import run_calculation


def eval_case(method: str, eval_mode: str, geom, plot: bool = False):
    class Args:
        pass
    args = Args()
    args.method = method
    args.emitter = (geom["W_emitter"], geom["H_emitter"]) if isinstance(geom, dict) and "W_emitter" in geom else (5.1, 2.1)
    args.receiver = (geom["W_receiver"], geom["H_receiver"]) if isinstance(geom, dict) and "W_receiver" in geom else (5.1, 2.1)
    args.setback = geom.get("setback", 3.0) if isinstance(geom, dict) else 3.0
    args.angle = geom.get("angle", 0.0) if isinstance(geom, dict) else 0.0
    args.rotate_axis = geom.get("rotate_axis", "z") if isinstance(geom, dict) else "z"
    args.rotate_target = geom.get("rotate_target", "emitter") if isinstance(geom, dict) else "emitter"
    args.angle_pivot = geom.get("angle_pivot", "toe") if isinstance(geom, dict) else "toe"
    args.align_centres = False
    args.receiver_offset = (0.0, 0.0)
    args.emitter_offset = (0.0, 0.0)
    args.plot = bool(plot)
    args.plot_3d = False
    args.plot_both = False
    args.outdir = "results"
    args.version = False
    # rc/eval mode mirrors
    args.eval_mode = eval_mode
    args.rc_mode = eval_mode
    # grid/search controls
    args.rc_grid_n = 41
    args.rc_search_rel_tol = 3e-3
    args.rc_search_max_iters = 50
    args.rc_search_multistart = 4
    args.rc_search_time_limit_s = 2.0
    args.rc_bounds = "auto"
    # Run calculation using CLI orchestration
    result = run_calculation(args)
    return result


def test_grid_returns_dense_field():
    g = build_display_geom(width=5.1, height=2.1, setback=3.0,
                           angle=0.0, pitch=0.0,
                           angle_pivot="toe", target="emitter")
    out = eval_case(method="adaptive", eval_mode="grid", geom=g, plot=False)
    vf_field = out.get("F", out.get("vf_field", None))
    gy = out.get("grid_y", out.get("Yy", None))
    gz = out.get("grid_z", out.get("Zz", None))
    assert vf_field is not None and isinstance(vf_field, np.ndarray)
    assert gy is not None and gz is not None
    assert getattr(vf_field, "ndim", 0) == 2
    assert getattr(gy, "ndim", 0) == 1 and getattr(gz, "ndim", 0) == 1
    v = float(np.nanmax(vf_field))
    assert 0.1 < v < 0.4, f"peak {v} out of expected rough range"



