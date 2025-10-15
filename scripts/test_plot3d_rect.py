from src.viz.display_geom import build_display_geom
import numpy as np

def is_const(a,t=1e-9): return (a.max()-a.min())<t
g=build_display_geom(width=5.1,height=2.1,setback=3.0,angle=0,pitch=0,angle_pivot="toe",target="emitter")
em=g["emitter"]["corners"]; rc=g["receiver"]["corners"]
assert is_const(em[:,0]); assert is_const(rc[:,0]-3.0)
wy=np.ptp(em[:,1]); hz=np.ptp(em[:,2])
assert abs(wy-5.1)<1e-6 and abs(hz-2.1)<1e-6
print("3D rectangles reflect computed geometry (no value changes).")
