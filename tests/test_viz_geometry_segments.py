import numpy as np
from src.viz.plots import plot_geometry_and_heatmap

def test_geometry_panels_show_both_emitter_and_receiver(tmp_path):
    out = tmp_path / "geom.png"
    # Create synthetic field data
    y = np.linspace(-2, 2, 41)
    z = np.linspace(-1, 1, 21)
    Y, Z = np.meshgrid(y, z, indexing="xy")
    F = np.exp(-((Y-0.5)**2 + (Z-0.4)**2))
    
    result = {
        "We": 5.0, "He": 2.0, "Wr": 5.0, "Hr": 2.0,
        "emitter_center": (1.0, 0.0, 0.0),
        "receiver_center": (0.0, 0.6, 0.4),
        "vf": 0.12, "y_peak": 0.5, "z_peak": 0.4,
        "Y": Y, "Z": Z, "F": F,  # Add field data
    }
    fig, (ax_xy, ax_xz, ax_hm) = plot_geometry_and_heatmap(
        result=result, eval_mode="grid", method="adaptive", setback=3.0, out_png=str(out), return_fig=True
    )
    xy_labels = {l.get_label() for l in ax_xy.lines}
    xz_labels = {p.get_label() for p in ax_xz.patches}
    assert any("Emitter" in label for label in xy_labels) and any("Receiver" in label for label in xy_labels)
    assert any("Emitter" in label for label in xz_labels) and any("Receiver" in label for label in xz_labels)
    # peak star exists on heatmap axis
    assert any(getattr(l, "get_marker", lambda: None)() == "*" for l in ax_hm.lines)
