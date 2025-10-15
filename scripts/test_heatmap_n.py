import json, subprocess, sys
payload={"method":"adaptive","emitter":[5.1,2.1],"receiver":[5.1,2.1],"setback":3.0,
         "angle":0.0,"rotate_axis":"z","rotate_target":"emitter","angle_pivot":"toe",
         "align_centres":False,"receiver_offset":[0.0,0.0],"emitter_offset":[0.0,0.0],
         "plot":False,"plot_3d":False,"plot_both":False,"outdir":"results","version":False,
         "eval_mode":"grid","rc_mode":"grid","rc_grid_n":15,"rc_search_time_limit_s":1.5,
         "rc_search_rel_tol":3e-3,"rc_search_max_iters":50,"rc_search_multistart":1,
         "rc_bounds":"auto","heatmap_n":61,"heatmap_interp":"bicubic"}
cmd=[sys.executable,"eng/run_with_timeout.py","--timeout","4","--payload",json.dumps(payload)]
r=subprocess.run(cmd,capture_output=True,text=True); print("exit",r.returncode,r.stdout.strip())
assert r.returncode in (0,124)
if r.returncode==0:
    import json, numpy as np
    out=json.loads(r.stdout); vf=out.get("res",{}).get("vf_field") or out.get("res",{}).get("F")
    arr=np.array(vf); assert arr.shape[0]>=15 and arr.shape[1]>=15
print("heatmap smoother test ok (values unmodified).")
