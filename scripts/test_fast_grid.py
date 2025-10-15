# scripts/test_fast_grid.py
import json
import numpy as np
from src.cli import run_calculation

class A:
    pass

args = A()
args.method = 'adaptive'
args.emitter = (5.1, 2.1)
args.receiver = (5.1, 2.1)
args.setback = 3.0
args.angle = 0.0
args.rotate_axis = 'z'
args.rotate_target = 'emitter'
args.angle_pivot = 'toe'
args.align_centres = False
args.receiver_offset = (0.0, 0.0)
args.emitter_offset = (0.0, 0.0)
args.plot = False
args.plot_3d = False
args.plot_both = False
args.outdir = 'results'
args.version = False
args.eval_mode = 'grid'
args.rc_mode = 'grid'
args.rc_grid_n = 15
args.rc_search_rel_tol = 3e-3
args.rc_search_max_iters = 50
args.rc_search_multistart = 1
args.rc_search_time_limit_s = 1.5
args.rc_bounds = 'auto'

res = run_calculation(args)
vf = res.get('vf_field') or res.get('F')
out = {
    "ok": isinstance(res, dict),
    "has_vf": vf is not None,
    "shape": None if vf is None else getattr(vf, "shape", None),
    "peak": None if vf is None else float(np.nanmax(vf)),
}
print(json.dumps(out))
