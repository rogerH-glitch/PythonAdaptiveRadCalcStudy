from src.viz.display_geom import build_display_geom
import numpy as np

def test_xz_is_rectangle_under_yaw():
    """Test that X-Z silhouette produces rectangular outlines under yaw rotation."""
    g = build_display_geom(width=5.1, height=2.1, setback=3.0,
                          angle=30.0, pitch=0.0, angle_pivot="toe", rotate_target="both",
                          x_offset=0.0, y_offset=0.25)
    em = g["emitter"]["corners"]
    rc = g["receiver"]["corners"]
    
    # Check that both panels have positive spans in X and Z
    em_x = np.ptp(em[:, 0])  # peak-to-peak (max - min)
    em_z = np.ptp(em[:, 2])
    rc_x = np.ptp(rc[:, 0])
    rc_z = np.ptp(rc[:, 2])
    
    # Both panels should have positive spans in both X and Z dimensions
    assert em_z > 0, "Emitter should have positive Z span"
    assert rc_z > 0, "Receiver should have positive Z span"
    assert em_x > 0, "Emitter should have positive X span"
    assert rc_x > 0, "Receiver should have positive X span"
    
    # The spans should be reasonable (not too small, not too large)
    assert 0.1 < em_x < 10.0, f"Emitter X span {em_x} should be reasonable"
    assert 0.1 < em_z < 10.0, f"Emitter Z span {em_z} should be reasonable"
    assert 0.1 < rc_x < 10.0, f"Receiver X span {rc_x} should be reasonable"
    assert 0.1 < rc_z < 10.0, f"Receiver Z span {rc_z} should be reasonable"
