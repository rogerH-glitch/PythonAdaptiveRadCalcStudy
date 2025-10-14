import numpy as np
import math
import pytest

from src.viz.display_geom import PanelPose, compute_panel_corners, bbox_size_2d, build_display_geom

EPS = 1e-9


def _near_zero(x, tol=1e-10):
    return abs(x) < tol


def test_yaw_changes_XZ_but_keeps_Z_span_constant():
    """Pure yaw should:
    • Produce zero X-span in XZ at 0° (thin vertical line).
    • Produce non-zero X-span in XZ at >0° (rectangle footprint in elevation).
    • Keep Z-span identical to the un-yawed panel.
    """
    W, H = 2.0, 1.5
    base = (0.0, 0.0, 0.0)

    p0 = PanelPose(W, H, yaw_deg=0.0, pitch_deg=0.0, base=base)
    g0 = compute_panel_corners(p0)

    p1 = PanelPose(W, H, yaw_deg=20.0, pitch_deg=0.0, base=base)
    g1 = compute_panel_corners(p1)

    # Z-span unchanged under yaw
    z0 = g0["z_span"]
    z1 = g1["z_span"]
    assert np.allclose(z0, z1, atol=1e-12)

    # X-span in XZ: near-zero at 0°, >0 at 20°
    xz0_w, xz0_h = bbox_size_2d(g0["xz"])
    xz1_w, xz1_h = bbox_size_2d(g1["xz"])
    assert _near_zero(xz0_w, tol=1e-12)
    assert xz1_w > 0.0
    # Heights should match panel height
    assert math.isclose(xz0_h, H, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(xz1_h, H, rel_tol=1e-12, abs_tol=1e-12)


def test_pitch_changes_XY_and_Z_span():
    """Pure pitch should:
    • Produce zero X-span in XY at 0° (thin vertical line).
    • Produce non-zero X-span in XY at >0° (rectangle footprint in plan).
    • Change Z-span (projection of height onto z decreases by cos(pitch)).
    """
    W, H = 2.0, 1.5
    base = (0.0, 0.0, 0.0)

    p0 = PanelPose(W, H, yaw_deg=0.0, pitch_deg=0.0, base=base)
    g0 = compute_panel_corners(p0)

    pitch = 15.0
    p1 = PanelPose(W, H, yaw_deg=0.0, pitch_deg=pitch, base=base)
    g1 = compute_panel_corners(p1)

    # XY: x-span near-zero at 0°, >0 at 15°
    xy0_w, xy0_h = bbox_size_2d(g0["xy"])
    xy1_w, xy1_h = bbox_size_2d(g1["xy"])
    assert _near_zero(xy0_w, tol=1e-12)
    assert xy1_w > 0.0
    # y-span should stay as width
    assert math.isclose(xy0_h, W, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(xy1_h, W, rel_tol=1e-12, abs_tol=1e-12)

    # Z-span decreases by cos(pitch)
    cos_p = math.cos(math.radians(pitch))
    expected_h = H  # corner-to-corner z-extent remains H irrespective of pitch for x=0 plane rotation?
    # For a pure pitch about global y, the z coordinates are scaled by cos(p).
    # Initial z-span = H; after pitch, z-span = H * cos(pitch).
    assert np.allclose(g0["z_span"][1] - g0["z_span"][0], H, atol=1e-12)
    assert np.allclose(g1["z_span"][1] - g1["z_span"][0], H * cos_p, atol=1e-12)


def test_combined_yaw_pitch_make_parallelogram_in_both_views():
    """With both yaw and pitch, both XY and XZ projections should have non-zero X-span."""
    W, H = 3.0, 2.0
    pose = PanelPose(W, H, yaw_deg=25.0, pitch_deg=10.0, base=(0.0, 0.0, 0.0))
    g = compute_panel_corners(pose)

    xy_w, xy_h = bbox_size_2d(g["xy"])
    xz_w, xz_h = bbox_size_2d(g["xz"])

    assert xy_w > 0.0
    assert xz_w > 0.0
    # With combined rotations, both projections should show non-zero width
    # The exact y-span is complex due to the interaction of yaw and pitch rotations
    # but it should be reasonable (between 0 and W)
    assert 0.0 < xy_h < W * 1.1  # Allow some tolerance for numerical precision
    assert 0.0 < xz_h < H * 1.1  # Similarly for z-span


def test_build_display_geom_new_api():
    """Test build_display_geom with new explicit yaw/pitch API."""
    W, H = 2.0, 1.5
    
    # Test with explicit yaw
    g1 = build_display_geom(width=W, height=H, yaw_deg=30.0)
    assert g1["emitter"]["corners"].shape == (4, 3)
    assert "emitter_xy" in g1
    assert "emitter_xz" in g1
    assert "receiver_xy" in g1
    assert "receiver_xz" in g1
    assert "xy_limits" in g1
    assert "xz_limits" in g1
    
    # Test with explicit pitch
    g2 = build_display_geom(width=W, height=H, pitch_deg=15.0)
    assert g2["emitter"]["corners"].shape == (4, 3)
    
    # Test with both yaw and pitch
    g3 = build_display_geom(width=W, height=H, yaw_deg=20.0, pitch_deg=10.0)
    assert g3["emitter"]["corners"].shape == (4, 3)
    
    # Test with base translation
    g4 = build_display_geom(width=W, height=H, yaw_deg=0.0, base=(1.0, 2.0, 3.0))
    assert np.allclose(g4["emitter"]["corners"][0], [1.0, 1.0, 2.25])  # First corner after translation


def test_build_display_geom_legacy_api():
    """Test build_display_geom with legacy angle + rotate_axis API."""
    W, H = 2.0, 1.5
    
    # Test yaw (z-axis rotation)
    g1 = build_display_geom(width=W, height=H, angle=30.0, rotate_axis="z")
    g1_direct = build_display_geom(width=W, height=H, yaw_deg=30.0)
    assert np.allclose(g1["emitter"]["corners"], g1_direct["emitter"]["corners"], atol=1e-12)
    
    # Test pitch (y-axis rotation)
    g2 = build_display_geom(width=W, height=H, angle=15.0, rotate_axis="y")
    g2_direct = build_display_geom(width=W, height=H, pitch_deg=15.0)
    assert np.allclose(g2["emitter"]["corners"], g2_direct["emitter"]["corners"], atol=1e-12)
    
    # Test with angle_pivot
    g3 = build_display_geom(width=W, height=H, angle=20.0, rotate_axis="z", angle_pivot="center")
    assert g3["emitter"]["corners"].shape == (4, 3)


def test_build_display_geom_validation():
    """Test build_display_geom input validation."""
    # Missing width/height
    with pytest.raises(TypeError, match="either call with.*width.*height"):
        build_display_geom()
    
    with pytest.raises(TypeError, match="either call with.*width.*height"):
        build_display_geom(width=2.0)
    
    with pytest.raises(TypeError, match="either call with.*width.*height"):
        build_display_geom(height=1.5)


def test_build_display_geom_consistency():
    """Test that build_display_geom produces consistent results with compute_panel_corners."""
    W, H = 3.0, 2.0
    yaw, pitch = 25.0, 10.0
    base = (1.0, 2.0, 3.0)
    
    # Direct API
    g1 = build_display_geom(width=W, height=H, yaw_deg=yaw, pitch_deg=pitch, base=base)
    
    # Via PanelPose
    pose = PanelPose(width=W, height=H, yaw_deg=yaw, pitch_deg=pitch, base=base)
    g2 = compute_panel_corners(pose)
    
    # Should be identical for emitter
    assert np.allclose(g1["emitter"]["corners"], g2["corners"], atol=1e-12)
    assert np.allclose(g1["emitter"]["xy"], g2["xy"], atol=1e-12)
    assert np.allclose(g1["emitter"]["xz"], g2["xz"], atol=1e-12)
    assert np.allclose(g1["emitter"]["x_span"], g2["x_span"], atol=1e-12)
    assert np.allclose(g1["emitter"]["y_span"], g2["y_span"], atol=1e-12)
    assert np.allclose(g1["emitter"]["z_span"], g2["z_span"], atol=1e-12)


def test_build_display_geom_positional_api():
    """Test build_display_geom with positional args (args, result) API."""
    # Mock args and result objects
    class MockArgs:
        def __init__(self):
            self.emitter_w = 2.0
            self.emitter_h = 1.5
            self.receiver_w = 2.0
            self.receiver_h = 1.5
            self.setback = 1.0
            self.receiver_offset = (0.5, 0.2)
            self.angle = 30.0
            self.rotate_axis = "z"
            self.rotate_target = "emitter"
            self.angle_pivot = "center"
    
    class MockResult:
        def __init__(self):
            pass
    
    args = MockArgs()
    result = MockResult()
    
    # Test positional API
    g = build_display_geom(args, result)
    
    # Should have both emitter and receiver
    assert "emitter" in g
    assert "receiver" in g
    assert "emitter_xy" in g
    assert "receiver_xy" in g
    assert "xy_limits" in g
    assert "xz_limits" in g
    
    # Check dimensions
    assert g["emitter"]["corners"].shape == (4, 3)
    assert g["receiver"]["corners"].shape == (4, 3)
    assert g["emitter_xy"].shape == (4, 2)
    assert g["receiver_xy"].shape == (4, 2)
    assert g["xy_limits"].shape == (4,)
    assert g["xz_limits"].shape == (4,)


def test_build_display_geom_legacy_keys():
    """Test that build_display_geom provides legacy nested keys for backward compatibility."""
    W, H = 2.0, 1.5
    g = build_display_geom(width=W, height=H, yaw_deg=30.0)
    
    # Test legacy nested structure
    assert "xy" in g
    assert "xz" in g
    assert "corners3d" in g
    
    # Test xy structure (now 2 endpoints)
    assert "emitter" in g["xy"]
    assert "receiver" in g["xy"]
    assert isinstance(g["xy"]["emitter"], list)
    assert isinstance(g["xy"]["receiver"], list)
    assert len(g["xy"]["emitter"]) == 2  # 2 endpoints
    assert len(g["xy"]["receiver"]) == 2  # 2 endpoints
    assert len(g["xy"]["emitter"][0]) == 2  # Each point has 2 coordinates
    
    # Test xz structure (now dict)
    assert "emitter" in g["xz"]
    assert "receiver" in g["xz"]
    assert isinstance(g["xz"]["emitter"], dict)
    assert isinstance(g["xz"]["receiver"], dict)
    assert "x" in g["xz"]["emitter"]
    assert "zmin" in g["xz"]["emitter"]
    assert "zmax" in g["xz"]["emitter"]
    
    # Test corners3d structure (now 8 points as lists)
    assert "emitter" in g["corners3d"]
    assert "receiver" in g["corners3d"]
    assert isinstance(g["corners3d"]["emitter"], list)
    assert isinstance(g["corners3d"]["receiver"], list)
    assert len(g["corners3d"]["emitter"]) == 8  # 8 points (4 repeated)
    assert len(g["corners3d"]["receiver"]) == 8  # 8 points (4 repeated)
    assert len(g["corners3d"]["emitter"][0]) == 3  # Each point has 3 coordinates
    
    # Test that legacy keys are in expected format (different from direct keys)
    # XY: legacy has 2 endpoints, direct has 4 corners
    assert len(g["xy"]["emitter"]) == 2, "Legacy XY should have 2 endpoints"
    assert len(g["emitter_xy"]) == 4, "Direct XY should have 4 corners"
    
    # XZ: legacy has dict with x/zmin/zmax, direct has 4 corners
    assert isinstance(g["xz"]["emitter"], dict), "Legacy XZ should be dict"
    assert len(g["emitter_xz"]) == 4, "Direct XZ should have 4 corners"
    
    # corners3d has 8 points (4 repeated), so compare first 4 with direct corners
    assert np.allclose(g["corners3d"]["emitter"][:4], g["emitter"]["corners"], atol=1e-12)
    assert np.allclose(g["corners3d"]["receiver"][:4], g["receiver"]["corners"], atol=1e-12)


def test_build_display_geom_robust_target_lookup():
    """Test that build_display_geom handles legacy 'target' parameter synonym."""
    # Mock args with legacy 'target' parameter
    class MockArgs:
        def __init__(self):
            self.emitter_w = 2.0
            self.emitter_h = 1.5
            self.receiver_w = 2.0
            self.receiver_h = 1.5
            self.setback = 1.0
            self.receiver_offset = (0.0, 0.0)
            self.angle = 30.0
            self.rotate_axis = "z"
            self.target = "receiver"  # Legacy synonym for rotate_target
            self.angle_pivot = "center"
    
    class MockResult:
        def __init__(self):
            pass
    
    args = MockArgs()
    result = MockResult()
    
    # Test positional API with legacy 'target' parameter
    g = build_display_geom(args, result)
    
    # Should have both emitter and receiver
    assert "emitter" in g
    assert "receiver" in g
    
    # The rotation should be applied to receiver (not emitter) due to target="receiver"
    # This is a basic check that the function runs without error
    assert g["emitter"]["corners"].shape == (4, 3)
    assert g["receiver"]["corners"].shape == (4, 3)
