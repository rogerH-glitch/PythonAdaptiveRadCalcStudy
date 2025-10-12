from src.viz.plots import plot_geometry_and_heatmap
import numpy as np
from matplotlib.patches import Rectangle

def test_emitter_in_xy_bounds_and_xz_rectangle(tmp_path):
    out = tmp_path / "g.png"
    result = {
        "We": 5.0, "He": 2.0, "Wr": 5.0, "Hr": 2.0,
        "emitter_center": (0.0, 0.0, 0.0),
        "receiver_center": (0.0, 0.6, 0.4),
        "rotate_axis": "z", "rotate_target": "emitter", "angle": 20.0, "angle_pivot": "center",
        "vf": 0.2, "y_peak": 0.5, "z_peak": 0.4
    }
    fig,(ax_xy,ax_xz,ax_hm) = plot_geometry_and_heatmap(result=result, eval_mode="grid", method="adaptive",
                                                        setback=3.0, out_png=str(out), return_fig=True)
    # XY limits include emitter x coords
    em_line = [ln for ln in ax_xy.lines if ln.get_label().startswith("Emitter")][0]
    xs = em_line.get_xdata()
    xmin, xmax = ax_xy.get_xlim()
    assert xs.min() >= xmin - 1e-9 and xs.max() <= xmax + 1e-9
    # XZ has rectangles
    assert any(isinstance(p, Rectangle) for p in ax_xz.patches)
