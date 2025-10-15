import time
from src.cli import run_calculation
class A: pass

def test_default_grid_finishes_fast_and_returns_field():
    a=A()
    a.method='adaptive'; a.emitter=(5.1,2.1); a.receiver=(5.1,2.1)
    a.setback=3.0; a.angle=0.0; a.rotate_axis='z'; a.rotate_target='emitter'
    a.angle_pivot='toe'; a.align_centres=False
    a.receiver_offset=(0.0,0.0); a.emitter_offset=(0.0,0.0)
    a.plot=False; a.plot_3d=False; a.plot_both=False; a.outdir='results'; a.version=False
    a.eval_mode='grid'; a.rc_mode='grid'; a.rc_grid_n=15; a.rc_search_time_limit_s=1.5
    a.rc_search_rel_tol=3e-3; a.rc_search_max_iters=50; a.rc_search_multistart=1
    a.rc_bounds='auto'; a.heatmap_n=None; a.heatmap_interp='bilinear'
    t0=time.time(); res=run_calculation(a); dt=time.time()-t0
    assert isinstance(res, dict) and dt < 4.0
    vf = res.get('vf_field') or res.get('F'); assert vf is not None
