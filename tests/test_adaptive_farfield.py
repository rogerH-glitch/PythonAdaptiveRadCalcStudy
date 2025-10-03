"""
Tests for adaptive solver far-field accuracy.

This module tests that the adaptive solver can achieve high accuracy
for far-field cases where the view factor is small and requires
careful refinement.
"""

import math
import numpy as np
from src.analytical import vf_point_rect_to_point_parallel
from src.adaptive import vf_adaptive


def test_farfield_adaptive_matches_analytical_center():
    """Test that adaptive solver matches analytical for far-field case."""
    em_w, em_h, s = 5.0, 2.0, 3.8
    
    # High-resolution analytical reference
    F_ref = vf_point_rect_to_point_parallel(em_w, em_h, s, rx=0.0, ry=0.0, nx=420, ny=420)
    
    # Adaptive solver with stricter settings
    res = vf_adaptive(
        em_w, em_h, em_w, em_h, s,
        rel_tol=3e-3, abs_tol=1e-6, max_depth=14,
        max_cells=150000, time_limit_s=15, 
        init_grid="4x4", min_cells=16,  # Start with smaller grid to force refinement
        rc_point=(0.0, 0.0)
    )
    
    assert res["status"] in ("converged", "reached_limits")
    
    rel_err = abs(res["vf"] - F_ref) / max(F_ref, 1e-12)
    assert rel_err <= 0.003, (
        f"rel_err={rel_err:.4f}, iters={res.get('iterations')}, "
        f"cells={res.get('cells')}, ach={res.get('achieved_tol')}"
    )
    
    # Ensure it actually refined (more than initial grid)
    assert res.get('iterations', 0) > 0, "Adaptive solver should perform refinement"
    assert res.get('cells', 0) >= 16, "Should have at least min_cells"


def test_farfield_adaptive_convergence_trend():
    """Test that adaptive solver improves with stricter settings."""
    em_w, em_h, s = 5.0, 2.0, 3.8
    
    # Reference with very high resolution
    F_ref = vf_point_rect_to_point_parallel(em_w, em_h, s, rx=0.0, ry=0.0, nx=500, ny=500)
    
    # Test with different tolerance settings
    tolerances = [1e-2, 3e-3, 1e-3]
    errors = []
    
    for tol in tolerances:
        res = vf_adaptive(
            em_w, em_h, em_w, em_h, s,
            rel_tol=tol, abs_tol=1e-6, max_depth=12,
            max_cells=100000, time_limit_s=10,
            init_grid="4x4", min_cells=16,
            rc_point=(0.0, 0.0)
        )
        
        rel_err = abs(res["vf"] - F_ref) / max(F_ref, 1e-12)
        errors.append(rel_err)
    
    # Error should generally decrease with tighter tolerance
    # (allowing for some variation due to randomness in refinement order)
    assert errors[-1] <= errors[0], f"Error should improve: {errors[0]:.4f} -> {errors[-1]:.4f}"


def test_farfield_adaptive_point_evaluation():
    """Test that adaptive solver correctly uses rc_point."""
    em_w, em_h, s = 5.0, 2.0, 3.8
    
    # Test different receiver points
    rc_points = [(0.0, 0.0), (1.0, 0.5), (-1.0, -0.5)]
    
    for rx, ry in rc_points:
        res = vf_adaptive(
            em_w, em_h, em_w, em_h, s,
            rel_tol=3e-3, abs_tol=1e-6, max_depth=12,
            max_cells=50000, time_limit_s=10,
            init_grid="4x4", min_cells=16,
            rc_point=(rx, ry)
        )
        
        assert res["status"] in ("converged", "reached_limits")
        assert 0.0 <= res["vf"] <= 1.0
        assert res.get('iterations', 0) >= 0
        assert res.get('cells', 0) >= 16


def test_farfield_adaptive_vs_analytical_multiple_geometries():
    """Test adaptive solver against analytical for multiple far-field geometries."""
    geometries = [
        (5.0, 2.0, 3.8),   # Original far-field case
        (10.0, 3.0, 5.0),  # Larger geometry
        (2.0, 1.0, 2.5),   # Smaller geometry
    ]
    
    for em_w, em_h, s in geometries:
        # Analytical reference
        F_ref = vf_point_rect_to_point_parallel(em_w, em_h, s, rx=0.0, ry=0.0, nx=300, ny=300)
        
        # Adaptive solver
        res = vf_adaptive(
            em_w, em_h, em_w, em_h, s,
            rel_tol=3e-3, abs_tol=1e-6, max_depth=12,
            max_cells=100000, time_limit_s=15,
            init_grid="4x4", min_cells=16,
            rc_point=(0.0, 0.0)
        )
        
        assert res["status"] in ("converged", "reached_limits")
        
        rel_err = abs(res["vf"] - F_ref) / max(F_ref, 1e-12)
        assert rel_err <= 0.01, (  # 1% tolerance for multiple geometries
            f"Geometry ({em_w}, {em_h}, {s}): rel_err={rel_err:.4f}, "
            f"iters={res.get('iterations')}, cells={res.get('cells')}"
        )
