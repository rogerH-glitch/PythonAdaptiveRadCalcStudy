from src.viz.display_geom import build_display_geom
from src.viz.plots import _compute_bounds_from_panels
import numpy as np

def test_elevation_bounds_cover_setback_panel():
    g = build_display_geom(width=5.1, height=2.1, setback=3.0,
                           angle=0.0, pitch=0.0, angle_pivot="toe", target="emitter")
    em_xy, rec_xy = g["emitter"]["xy"], g["receiver"]["xy"]
    em_xz, rec_xz = g["emitter"]["xz"], g["receiver"]["xz"]
    b = _compute_bounds_from_panels(em_xy, rec_xy, em_xz, rec_xz, pad=0.01)
    (xlim_xz, zlim_xz) = b["xz"]
    # Should include x≈0 (emitter) and x≈3.0 (receiver)
    assert xlim_xz[0] <= 0.0 and xlim_xz[1] >= 3.0
    # Heatmap Y/Z span should equal receiver physical half-sizes (±2.55, ±1.05)
    y_span = np.max(rec_xy[:,1]) - np.min(rec_xy[:,1])
    z_span = np.max(rec_xz[:,1]) - np.min(rec_xz[:,1])
    assert 5.05 < y_span < 5.15   # ~5.1
    assert 2.05 < z_span < 2.15   # ~2.1
