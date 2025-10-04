"""
Tests for offset and rotation functionality in CLI and geometry.

This module tests the new offset and rotation features including:
- Receiver and emitter offset transformations
- Rotation about z-axis with different pivot points
- Integration with analytical method
- CLI argument parsing and validation
"""

import math
import numpy as np
import pytest
from src.analytical import vf_point_rect_to_point_parallel
from src.geometry import to_emitter_frame, Rz


def test_receiver_offset_center_matches_direct_shift():
    """Test that receiver offset is equivalent to direct coordinate shift."""
    em_w, em_h, s = 5.0, 2.0, 1.0
    
    # Direct: evaluate at rx=+0.5 in emitter frame
    F_direct = vf_point_rect_to_point_parallel(em_w, em_h, s, rx_local=0.5, ry_local=0.0, nx=180, ny=180)
    
    # Via transform: receiver offset +0.5, receiver local point = (0,0)
    rx, ry = to_emitter_frame(
        (0.0, 0.0), 
        em_offset=(0.0, 0.0), 
        rc_offset=(0.5, 0.0), 
        angle_deg=0.0, 
        rotate_target="emitter"
    )
    F_xform = vf_point_rect_to_point_parallel(em_w, em_h, s, rx_local=rx, ry_local=ry, nx=180, ny=180)
    
    assert abs(F_direct - F_xform) / max(F_direct, 1e-12) < 5e-3


def test_emitter_offset_is_equivalent_negative_receiver_offset():
    """Test that emitter offset is equivalent to negative receiver offset."""
    em_w, em_h, s = 5.0, 2.0, 1.0
    
    # Direct: receiver at +0.3 in emitter frame
    F1 = vf_point_rect_to_point_parallel(em_w, em_h, s, rx_local=+0.3, ry_local=0.0, nx=160, ny=160)
    
    # Emitter shifted +0.3 → equivalent to receiver shifted -0.3
    rx, ry = to_emitter_frame(
        (0.0, 0.0), 
        em_offset=(+0.3, 0.0), 
        rc_offset=(0.0, 0.0), 
        angle_deg=0.0, 
        rotate_target="emitter"
    )
    F2 = vf_point_rect_to_point_parallel(em_w, em_h, s, rx_local=rx, ry_local=ry, nx=160, ny=160)
    
    assert abs(F1 - F2) / max(F1, 1e-12) < 5e-3


def test_rotation_matrix_properties():
    """Test that rotation matrix has correct mathematical properties."""
    # Test 90 degree rotation
    R90 = Rz(90.0)
    expected_90 = np.array([[0, -1], [1, 0]], dtype=float)
    np.testing.assert_allclose(R90, expected_90, atol=1e-15)
    
    # Test 180 degree rotation
    R180 = Rz(180.0)
    expected_180 = np.array([[-1, 0], [0, -1]], dtype=float)
    np.testing.assert_allclose(R180, expected_180, atol=1e-15)
    
    # Test 360 degree rotation (identity)
    R360 = Rz(360.0)
    expected_360 = np.eye(2, dtype=float)
    np.testing.assert_allclose(R360, expected_360, atol=1e-15)
    
    # Test orthogonality: R @ R.T should be identity
    R45 = Rz(45.0)
    should_be_identity = R45 @ R45.T
    np.testing.assert_allclose(should_be_identity, np.eye(2), atol=1e-15)


def test_emitter_rotation_transform():
    """Test rotation when emitter is the target (default behavior)."""
    # Test 90 degree emitter rotation
    rx, ry = to_emitter_frame(
        (1.0, 0.0),  # receiver point at (1, 0) in receiver frame
        em_offset=(0.0, 0.0),
        rc_offset=(0.0, 0.0),
        angle_deg=90.0,
        rotate_target="emitter"
    )
    
    # With emitter rotated 90 degrees, (1,0) in receiver should become (0,1) in emitter frame
    expected_rx, expected_ry = 0.0, 1.0
    assert abs(rx - expected_rx) < 1e-15
    assert abs(ry - expected_ry) < 1e-15


def test_receiver_rotation_transform():
    """Test rotation when receiver is the target."""
    # Test 90 degree receiver rotation
    rx, ry = to_emitter_frame(
        (1.0, 0.0),  # receiver point at (1, 0) in receiver frame
        em_offset=(0.0, 0.0),
        rc_offset=(0.0, 0.0),
        angle_deg=90.0,
        rotate_target="receiver"
    )
    
    # With receiver rotated 90 degrees, (1,0) becomes (0,1) before transformation
    expected_rx, expected_ry = 0.0, 1.0
    assert abs(rx - expected_rx) < 1e-15
    assert abs(ry - expected_ry) < 1e-15


def test_combined_offset_and_rotation():
    """Test combined offset and rotation transformations."""
    # Test with both offsets and rotation
    rx, ry = to_emitter_frame(
        (0.5, 0.2),  # receiver point
        em_offset=(0.1, 0.3),  # emitter offset
        rc_offset=(0.4, 0.1),  # receiver offset
        angle_deg=45.0,
        rotate_target="emitter"
    )
    
    # This should be a valid transformation (no exceptions)
    assert isinstance(rx, float)
    assert isinstance(ry, float)
    assert not (np.isnan(rx) or np.isnan(ry))
    assert not (np.isinf(rx) or np.isinf(ry))


def test_zero_angle_shortcut():
    """Test that zero angle uses the fast path."""
    # Test that zero angle doesn't trigger rotation calculations
    rx, ry = to_emitter_frame(
        (1.0, 2.0),
        em_offset=(0.5, 0.3),
        rc_offset=(0.2, 0.1),
        angle_deg=0.0,
        rotate_target="emitter"
    )
    
    # Should be simple translation: (1.0 + 0.2 - 0.5, 2.0 + 0.1 - 0.3)
    expected_rx, expected_ry = 0.7, 1.8
    assert abs(rx - expected_rx) < 1e-15
    assert abs(ry - expected_ry) < 1e-15


def test_very_small_angle_shortcut():
    """Test that very small angles use the fast path."""
    # Test with angle smaller than threshold
    rx, ry = to_emitter_frame(
        (1.0, 0.0),
        em_offset=(0.0, 0.0),
        rc_offset=(0.0, 0.0),
        angle_deg=1e-15,  # Very small angle
        rotate_target="emitter"
    )
    
    # Should use fast path (no rotation)
    expected_rx, expected_ry = 1.0, 0.0
    assert abs(rx - expected_rx) < 1e-15
    assert abs(ry - expected_ry) < 1e-15


def test_cli_offset_arguments():
    """Test that CLI correctly parses offset arguments."""
    from src.cli import create_parser
    
    parser = create_parser()
    
    # Test receiver offset parsing
    args = parser.parse_args([
        '--method', 'analytical',
        '--emitter', '5.0', '2.0',
        '--setback', '1.0',
        '--receiver-offset', '0.5', '0.3'
    ])
    
    assert args.receiver_offset == (0.5, 0.3)
    assert args.emitter_offset == (0.0, 0.0)  # default
    assert args.rotate_target == 'emitter'  # default
    assert args.angle_pivot == 'toe'  # default
    
    # Test emitter offset parsing
    args = parser.parse_args([
        '--method', 'analytical',
        '--emitter', '5.0', '2.0',
        '--setback', '1.0',
        '--emitter-offset', '0.2', '0.4'
    ])
    
    assert args.emitter_offset == (0.2, 0.4)
    assert args.receiver_offset == (0.0, 0.0)  # default
    
    # Test rotation arguments
    args = parser.parse_args([
        '--method', 'analytical',
        '--emitter', '5.0', '2.0',
        '--setback', '1.0',
        '--rotate-target', 'receiver',
        '--angle-pivot', 'center',
        '--angle', '45.0'
    ])
    
    assert args.rotate_target == 'receiver'
    assert args.angle_pivot == 'center'
    assert args.angle == 45.0


def test_geometry_config_creation():
    """Test that geometry configuration is created correctly from CLI args."""
    from src.cli import create_parser
    
    parser = create_parser()
    args = parser.parse_args([
        '--method', 'analytical',
        '--emitter', '5.0', '2.0',
        '--setback', '1.0',
        '--receiver-offset', '0.5', '0.3',
        '--emitter-offset', '0.2', '0.1',
        '--rotate-target', 'receiver',
        '--angle-pivot', 'center',
        '--angle', '30.0'
    ])
    
    # Create geometry config as done in CLI
    geom_cfg = {
        "emitter_offset": tuple(args.emitter_offset),
        "receiver_offset": tuple(args.receiver_offset),
        "angle_deg": float(args.angle),
        "angle_pivot": args.angle_pivot,
        "rotate_target": args.rotate_target,
    }
    
    assert geom_cfg["emitter_offset"] == (0.2, 0.1)
    assert geom_cfg["receiver_offset"] == (0.5, 0.3)
    assert geom_cfg["angle_deg"] == 30.0
    assert geom_cfg["angle_pivot"] == "center"
    assert geom_cfg["rotate_target"] == "receiver"


def test_toe_pivot_adjustment():
    """Test toe pivot adjustment function."""
    from src.geometry import toe_pivot_adjust_emitter_center
    
    # For now, toe pivot should return unchanged offset
    em_offset = (0.5, 0.3)
    adjusted = toe_pivot_adjust_emitter_center(45.0, em_offset)
    assert adjusted == em_offset


if __name__ == "__main__":
    pytest.main([__file__])
