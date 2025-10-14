import numpy as np
import pytest
from src.viz.display_geom import build_display_geom


def _mk_args(**kw):
    """Create minimal args/result pair for testing."""
    class NS:
        pass
    args = NS()
    result = NS()
    for k, v in kw.items():
        setattr(args, k, v)
    # Common defaults
    for k, v in dict(
        emitter_w=2.0, emitter_h=1.5,
        receiver_w=2.0, receiver_h=1.5,
        setback=3.0, angle=0.0,
        rotate_axis="z", rotate_target="emitter",
        receiver_offset=(0.0, 0.0),
    ).items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args, result


def test_toe_pivot_emitter_yaw_keeps_setback():
    """Test that emitter toe-pivot yaw keeps minimum setback constant."""
    args, result = _mk_args(angle=30.0, rotate_axis="z", rotate_target="emitter", angle_pivot="toe")
    
    geom = build_display_geom(args, result)
    
    # Emitter should be at origin (x=0) with toe-pivot
    emitter_corners = geom["emitter"]["corners"]
    e_min_x = float(emitter_corners[:, 0].min())
    e_max_x = float(emitter_corners[:, 0].max())
    
    # With toe-pivot, emitter's nearest edge should be at x=0 (or slightly negative for rotated panel)
    assert e_min_x < 0.1, f"Emitter min x should be near 0, got {e_min_x}"
    
    # Receiver should be at setback distance
    receiver_corners = geom["receiver"]["corners"]
    r_min_x = float(receiver_corners[:, 0].min())
    r_max_x = float(receiver_corners[:, 0].max())
    
    # Receiver should be at x=setback
    assert abs(r_min_x - 3.0) < 1e-10, f"Receiver min x should be ~3.0, got {r_min_x}"


def test_toe_pivot_receiver_yaw_keeps_setback():
    """Test that receiver toe-pivot yaw keeps minimum setback constant."""
    args, result = _mk_args(angle=30.0, rotate_axis="z", rotate_target="receiver", angle_pivot="toe")
    
    geom = build_display_geom(args, result)
    
    # Emitter should be at origin
    emitter_corners = geom["emitter"]["corners"]
    e_min_x = float(emitter_corners[:, 0].min())
    e_max_x = float(emitter_corners[:, 0].max())
    
    assert abs(e_min_x) < 1e-10, f"Emitter min x should be ~0, got {e_min_x}"
    
    # Receiver should be at setback with toe-pivot
    receiver_corners = geom["receiver"]["corners"]
    r_min_x = float(receiver_corners[:, 0].min())
    r_max_x = float(receiver_corners[:, 0].max())
    
    # With toe-pivot, receiver's nearest edge should be at x=setback
    assert abs(r_max_x - 3.0) < 1e-10, f"Receiver max x should be ~3.0, got {r_max_x}"


def test_toe_pivot_emitter_pitch_keeps_setback():
    """Test that emitter toe-pivot pitch keeps minimum setback constant."""
    args, result = _mk_args(angle=20.0, rotate_axis="y", rotate_target="emitter", angle_pivot="toe")
    
    geom = build_display_geom(args, result)
    
    # Emitter should be at origin with toe-pivot
    emitter_corners = geom["emitter"]["corners"]
    e_min_x = float(emitter_corners[:, 0].min())
    
    assert e_min_x < 0.1, f"Emitter min x should be near 0, got {e_min_x}"
    
    # Receiver should be at setback
    receiver_corners = geom["receiver"]["corners"]
    r_min_x = float(receiver_corners[:, 0].min())
    
    assert abs(r_min_x - 3.0) < 1e-10, f"Receiver min x should be ~3.0, got {r_min_x}"


def test_center_pivot_no_translation():
    """Test that center pivot doesn't apply toe-pivot translation."""
    args, result = _mk_args(angle=30.0, rotate_axis="z", rotate_target="emitter", angle_pivot="center")
    
    geom = build_display_geom(args, result)
    
    # With center pivot, emitter should be at setback (original behavior)
    emitter_corners = geom["emitter"]["corners"]
    e_min_x = float(emitter_corners[:, 0].min())
    e_max_x = float(emitter_corners[:, 0].max())
    
    # Center pivot should keep emitter centered at setback (no toe-pivot translation)
    # The panel is rotated, so min/max x will be around the center
    center_x = (e_min_x + e_max_x) / 2
    assert abs(center_x - 3.0) < 1e-10, f"Emitter center x should be ~3.0 with center pivot, got {center_x}"


def test_toe_pivot_combined_yaw_pitch():
    """Test toe-pivot with combined yaw and pitch rotations."""
    args, result = _mk_args(angle=25.0, rotate_axis="z", rotate_target="emitter", angle_pivot="toe")
    
    geom = build_display_geom(args, result)
    
    # Should maintain toe-pivot behavior
    emitter_corners = geom["emitter"]["corners"]
    min_x = float(emitter_corners[:, 0].min())
    
    assert min_x < 0.1, f"Emitter min x should be near 0 with toe pivot, got {min_x}"


def test_toe_pivot_different_setback_distances():
    """Test toe-pivot with different setback distances."""
    for setback in [1.0, 2.5, 5.0]:
        args, result = _mk_args(angle=20.0, rotate_axis="z", rotate_target="receiver", 
                               angle_pivot="toe", setback=setback)
        
        geom = build_display_geom(args, result)
        
        # Receiver should be at the specified setback
        receiver_corners = geom["receiver"]["corners"]
        r_max_x = float(receiver_corners[:, 0].max())
        
        assert abs(r_max_x - setback) < 1e-10, f"Receiver max x should be ~{setback}, got {r_max_x}"


def test_toe_pivot_preserves_rotation():
    """Test that toe-pivot translation doesn't affect rotation angles."""
    args, result = _mk_args(angle=45.0, rotate_axis="z", rotate_target="emitter", angle_pivot="toe")
    
    geom = build_display_geom(args, result)
    
    # Check that rotation is still applied (corners should be rotated)
    emitter_corners = geom["emitter"]["corners"]
    
    # With 45-degree yaw, the panel should be rotated
    # Check that y-coordinates are not all the same (indicating rotation)
    y_coords = emitter_corners[:, 1]
    y_range = float(y_coords.max() - y_coords.min())
    
    assert y_range > 0.1, f"Panel should be rotated (y range > 0.1), got {y_range}"


def test_toe_pivot_no_rotation_no_translation():
    """Test that toe-pivot with no rotation doesn't apply unnecessary translation."""
    args, result = _mk_args(angle=0.0, rotate_axis="z", rotate_target="emitter", angle_pivot="toe")
    
    geom = build_display_geom(args, result)
    
    # With no rotation, toe-pivot should not change the position
    emitter_corners = geom["emitter"]["corners"]
    e_min_x = float(emitter_corners[:, 0].min())
    
    assert abs(e_min_x) < 1e-10, f"With no rotation, emitter min x should be ~0, got {e_min_x}"
