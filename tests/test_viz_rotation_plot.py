from src.viz.plots import plot_geometry_and_heatmap
import numpy as np

def _segment_from_axis_lines(ax, label):
    # find the first line with that label and return its (x,y) data
    for ln in ax.lines:
        if label in ln.get_label():
            x,y = ln.get_xdata(), ln.get_ydata()
            return np.array(x), np.array(y)
    return None,None

def test_yaw_20_rotates_emitter_in_plan(tmp_path):
    out = tmp_path/"g.png"
    result = {
        "We": 5.0, "He": 2.0, "Wr": 5.0, "Hr": 2.0,
        "emitter_center": (0.0,0.0,0.0),
        "receiver_center": (0.0,0.0,0.0),
        "vf": 0.2, "y_peak":0.5, "z_peak":0.4,
        "rotate_axis":"z","rotate_target":"emitter","angle_pivot":"center","angle":20.0,
    }
    fig,(ax_xy,ax_xz,ax_hm)=plot_geometry_and_heatmap(result=result, eval_mode="grid", method="adaptive",
                                                      setback=3.0, out_png=str(out), return_fig=True)
    x,y = _segment_from_axis_lines(ax_xy, "Emitter")
    assert x is not None
    # if rotated, the x coordinates should not both be ~0
    assert not (abs(x[0])<1e-9 and abs(x[1])<1e-9)
