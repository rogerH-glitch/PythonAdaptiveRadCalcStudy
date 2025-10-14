"""Test that 3D plot shows rotated geometry correctly."""
import numpy as np
from types import SimpleNamespace
from src.viz.display_geom import build_display_geom


def test_plot3d_rotates_emitter():
    """Test that 3D corners show rotation for yaw (z-axis) rotation."""
    # Create test args for angle=0
    args_0 = SimpleNamespace(
        emitter=(5.0, 2.0),
        receiver=(5.0, 2.0),
        setback=3.0,
        rotate_axis="z",
        angle=0.0,
        angle_pivot="toe",
        rotate_target="emitter"
    )
    
    result = {
        "We": 5.0, "He": 2.0, "Wr": 5.0, "Hr": 2.0,
        "dy": 0.0, "dz": 0.0
    }
    
    # Get corners for angle=0
    geom_0 = build_display_geom(args_0, result)
    corners_0 = geom_0["corners3d"]["emitter"]
    
    # Create test args for angle=20
    args_20 = SimpleNamespace(
        emitter=(5.0, 2.0),
        receiver=(5.0, 2.0),
        setback=3.0,
        rotate_axis="z",
        angle=20.0,
        angle_pivot="toe",
        rotate_target="emitter"
    )
    
    # Get corners for angle=20
    geom_20 = build_display_geom(args_20, result)
    corners_20 = geom_20["corners3d"]["emitter"]
    
    # Assert that at least one emitter XY coordinate changes with the angle
    xy_changed = False
    for i, (c0, c20) in enumerate(zip(corners_0, corners_20)):
        x0, y0, z0 = c0
        x20, y20, z20 = c20
        
        # Check if XY coordinates changed (yaw should affect X and Y)
        if abs(x0 - x20) > 1e-6 or abs(y0 - y20) > 1e-6:
            xy_changed = True
            break
    
    assert xy_changed, "Yaw rotation should change at least one emitter XY coordinate"
    
    # Assert that Z coordinates remain the same for pure yaw
    z_same = True
    for c0, c20 in zip(corners_0, corners_20):
        z0 = c0[2]
        z20 = c20[2]
        if abs(z0 - z20) > 1e-6:
            z_same = False
            break
    
    assert z_same, "Yaw rotation should not change Z coordinates"


def test_plot3d_rotates_receiver_unchanged():
    """Test that receiver corners don't change when only emitter is rotated."""
    # Create test args for angle=0
    args_0 = SimpleNamespace(
        emitter=(5.0, 2.0),
        receiver=(5.0, 2.0),
        setback=3.0,
        rotate_axis="z",
        angle=0.0,
        angle_pivot="toe",
        rotate_target="emitter"  # Only emitter should rotate
    )
    
    result = {
        "We": 5.0, "He": 2.0, "Wr": 5.0, "Hr": 2.0,
        "dy": 0.0, "dz": 0.0
    }
    
    # Get receiver corners for angle=0
    geom_0 = build_display_geom(args_0, result)
    receiver_corners_0 = geom_0["corners3d"]["receiver"]
    
    # Create test args for angle=20
    args_20 = SimpleNamespace(
        emitter=(5.0, 2.0),
        receiver=(5.0, 2.0),
        setback=3.0,
        rotate_axis="z",
        angle=20.0,
        angle_pivot="toe",
        rotate_target="emitter"  # Only emitter should rotate
    )
    
    # Get receiver corners for angle=20
    geom_20 = build_display_geom(args_20, result)
    receiver_corners_20 = geom_20["corners3d"]["receiver"]
    
    # Assert that receiver corners don't change when only emitter rotates
    for c0, c20 in zip(receiver_corners_0, receiver_corners_20):
        x0, y0, z0 = c0
        x20, y20, z20 = c20
        assert abs(x0 - x20) < 1e-6, "Receiver X should not change when only emitter rotates"
        assert abs(y0 - y20) < 1e-6, "Receiver Y should not change when only emitter rotates"
        assert abs(z0 - z20) < 1e-6, "Receiver Z should not change when only emitter rotates"


def test_plot3d_corners_form_rectangle():
    """Test that corners form a proper rectangle (8 corners for 3D box)."""
    args = SimpleNamespace(
        emitter=(5.0, 2.0),
        receiver=(5.0, 2.0),
        setback=3.0,
        rotate_axis="z",
        angle=0.0,
        angle_pivot="toe",
        rotate_target="emitter"
    )
    
    result = {
        "We": 5.0, "He": 2.0, "Wr": 5.0, "Hr": 2.0,
        "dy": 0.0, "dz": 0.0
    }
    
    geom = build_display_geom(args, result)
    emitter_corners = geom["corners3d"]["emitter"]
    receiver_corners = geom["corners3d"]["receiver"]
    
    # Should have 8 corners for a 3D box
    assert len(emitter_corners) == 8, f"Emitter should have 8 corners, got {len(emitter_corners)}"
    assert len(receiver_corners) == 8, f"Receiver should have 8 corners, got {len(receiver_corners)}"
    
    # Check that corners are 3D (x, y, z)
    for corner in emitter_corners:
        assert len(corner) == 3, f"Each corner should have 3 coordinates, got {len(corner)}"
    
    for corner in receiver_corners:
        assert len(corner) == 3, f"Each corner should have 3 coordinates, got {len(corner)}"
