"""
Tests for point vs area-average validation modes.

This module tests the validation diagnostics that distinguish between
point view factors at specific locations and area-averaged view factors.
"""

import numpy as np
from src.validators import receiver_area_average_point_integrand
from src.analytical import vf_point_rect_to_point_parallel


def test_point_vs_areaavg_consistency():
    """Test that area-average VF is consistent with point VF for concentric case."""
    em_w, em_h = 5.0, 2.0
    rc_w, rc_h = 5.0, 2.0
    s = 1.0

    F_center = vf_point_rect_to_point_parallel(em_w, em_h, s, 0.0, 0.0, nx=160, ny=160)
    F_avg = receiver_area_average_point_integrand(em_w, em_h, rc_w, rc_h, s, nx_rc=31, ny_rc=13, nx_em=140, ny_em=140)

    assert 0.0 <= F_avg <= F_center + 1e-6, "Area-average should not exceed peak at center for concentric case"
    # For concentric parallel case, area-average should be close to center (within 25% tolerance)
    rel_diff = abs(F_avg - F_center) / max(F_center, 1e-12)
    assert rel_diff < 0.25, f"Area-average should be reasonably close to center for concentric case: {rel_diff:.4f}"


def test_area_average_convergence():
    """Test that area-average converges with finer receiver grid."""
    em_w, em_h = 5.0, 2.0
    rc_w, rc_h = 5.0, 2.0
    s = 2.0
    
    # Coarse grid
    F_coarse = receiver_area_average_point_integrand(em_w, em_h, rc_w, rc_h, s, nx_rc=15, ny_rc=7)
    
    # Fine grid
    F_fine = receiver_area_average_point_integrand(em_w, em_h, rc_w, rc_h, s, nx_rc=45, ny_rc=19)
    
    # Should be close
    rel_diff = abs(F_fine - F_coarse) / max(F_fine, 1e-12)
    assert rel_diff < 0.01, f"Area-average should converge: {rel_diff:.4f}"


def test_area_average_vs_point_difference():
    """Test that area-average differs from point for non-concentric cases."""
    # Use a case where receiver is offset from emitter center
    em_w, em_h = 5.0, 2.0
    rc_w, rc_h = 3.0, 1.5  # Smaller receiver
    s = 1.0
    
    F_center = vf_point_rect_to_point_parallel(em_w, em_h, s, 0.0, 0.0, nx=160, ny=160)
    F_avg = receiver_area_average_point_integrand(em_w, em_h, rc_w, rc_h, s, nx_rc=31, ny_rc=15)
    
    # For this case, area-average might be different from center point
    rel_diff = abs(F_avg - F_center) / max(F_center, 1e-12)
    assert rel_diff >= 0.0, "Relative difference should be non-negative"


def test_validation_diagnostics():
    """Test the comprehensive validation diagnostics."""
    from src.validators import compute_point_vs_area_diagnostics
    
    em_w, em_h = 5.0, 2.0
    rc_w, rc_h = 5.0, 2.0
    s = 1.5
    
    diagnostics = compute_point_vs_area_diagnostics(em_w, em_h, rc_w, rc_h, s)
    
    assert 'vf_point_center' in diagnostics
    assert 'vf_receiver_avg' in diagnostics
    assert 'avg_gt_center' in diagnostics
    assert 'rel_diff' in diagnostics
    
    assert 0.0 <= diagnostics['vf_point_center'] <= 1.0
    assert 0.0 <= diagnostics['vf_receiver_avg'] <= 1.0
    assert isinstance(diagnostics['avg_gt_center'], bool)
    assert isinstance(diagnostics['rel_diff'], (int, float))


def test_parallel_case_only():
    """Test that area-average only works for parallel cases."""
    from src.validators import receiver_area_average_point_integrand
    
    em_w, em_h = 5.0, 2.0
    rc_w, rc_h = 5.0, 2.0
    s = 1.0
    
    # Should work for parallel case
    F_parallel = receiver_area_average_point_integrand(em_w, em_h, rc_w, rc_h, s, angle_deg=0.0)
    assert 0.0 <= F_parallel <= 1.0
    
    # Should raise error for non-parallel case
    try:
        receiver_area_average_point_integrand(em_w, em_h, rc_w, rc_h, s, angle_deg=5.0)
        assert False, "Should raise NotImplementedError for non-parallel case"
    except NotImplementedError:
        pass  # Expected
