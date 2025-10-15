import numpy as np
from src.viz.display_geom import build_display_geom

def support_gap(a, b, n):
    """Compute the minimum signed distance between polygons along normal n."""
    n = n/np.linalg.norm(n)
    return np.min(b @ n) - np.min(a @ n)

def test_yaw_toe_target_emitter_setback():
    """Test that setback is preserved under yaw rotation with toe pivot and emitter target."""
    g = build_display_geom(width=5.1, height=2.1, setback=3.0,
                          angle=30.0, pitch=0.0, angle_pivot="toe", target="emitter",
                          x_offset=0.0, y_offset=0.25)
    em = g["emitter"]["corners"]
    rc = g["receiver"]["corners"]
    
    # Emitter normal (post-yaw): rotate +x by yaw about z
    yaw = np.deg2rad(30.0)
    n_em = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    
    gap = support_gap(em, rc, n_em)
    assert abs(gap - 3.0) < 1e-6, f"Expected setback 3.0, got {gap}"

def test_yaw_toe_target_receiver_setback():
    """Test that setback is preserved under yaw rotation with toe pivot and receiver target."""
    g = build_display_geom(width=5.1, height=2.1, setback=3.0,
                          angle=30.0, pitch=0.0, angle_pivot="toe", rotate_target="receiver",
                          x_offset=0.0, y_offset=0.25)
    em = g["emitter"]["corners"]
    rc = g["receiver"]["corners"]
    
    # Receiver normal (post-yaw): rotate -x by yaw about z
    yaw = np.deg2rad(30.0)
    n_rc = np.array([-np.cos(yaw), -np.sin(yaw), 0.0])
    
    gap = support_gap(rc, em, n_rc)
    assert abs(gap - 3.0) < 1e-6, f"Expected setback 3.0, got {gap}"

def test_center_pivot_unchanged():
    """Test that center pivot behavior is unchanged (no setback preservation needed)."""
    g = build_display_geom(width=5.1, height=2.1, setback=3.0,
                          angle=30.0, pitch=0.0, angle_pivot="center", rotate_target="both",
                          x_offset=0.0, y_offset=0.25)
    em = g["emitter"]["corners"]
    rc = g["receiver"]["corners"]
    
    # For center pivot, setback should be preserved as distance between centers
    em_center = np.mean(em, axis=0)
    rc_center = np.mean(rc, axis=0)
    distance = np.linalg.norm(rc_center - em_center)
    assert abs(distance - 3.0) < 1e-6, f"Expected center distance 3.0, got {distance}"
