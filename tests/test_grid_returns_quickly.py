import time
from src.cli import run_calculation

def test_grid_fastpath_returns_under_budget():
    class A:
        pass
    
    args = A()
    args.method='adaptive'; args.emitter=(5.1,2.1); args.receiver=(5.1,2.1)
    args.setback=3.0; args.angle=0.0
    args.rotate_axis='z'; args.rotate_target='emitter'; args.angle_pivot='toe'
    args.align_centres=False
    args.receiver_offset=(0.0,0.0); args.emitter_offset=(0.0,0.0)
    args.plot=False; args.plot_3d=False; args.plot_both=False
    args.outdir='results'; args.version=False
    args.eval_mode='grid'; args.rc_mode='grid'
    args.rc_grid_n=11; args.rc_search_time_limit_s=1.0
    args.rc_search_rel_tol=3e-3; args.rc_search_max_iters=50
    args.rc_search_multistart=1; args.rc_bounds='auto'
    
    t0=time.time()
    res = run_calculation(args)
    assert isinstance(res, dict)
    assert (time.time()-t0) < 3.0, "grid fast-path exceeded budget"
    vf = res.get('vf_field') or res.get('F')
    assert vf is not None
