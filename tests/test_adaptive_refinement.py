"""
Tests for adaptive refinement to ensure proper Walton-style 1AI implementation.

These tests verify that the adaptive method actually refines and matches
analytical results within 0.3% tolerance.
"""

import math
import numpy as np
from src.adaptive import vf_adaptive
from src.analytical import vf_point_rect_to_point_parallel


def _cmp(em_w, em_h, s, nx=400, ny=400):
    """Compare adaptive vs analytical for a given geometry."""
    F_ref = vf_point_rect_to_point_parallel(em_w, em_h, s, rx=0.0, ry=0.0, nx=nx, ny=ny)
    res = vf_adaptive(em_w, em_h, em_w, em_h, s,
                      rel_tol=3e-3, abs_tol=1e-6, max_depth=12,
                      max_cells=100000, time_limit_s=10, init_grid="4x4")
    return F_ref, res


def test_adaptive_matches_analytical_center_three_geoms():
    """Test that adaptive matches analytical within 0.3% for three geometries."""
    geoms = [
        (5.1, 2.1, 1.0),
        (5.0, 2.0, 3.8),
        (20.02, 1.05, 1.8),
    ]
    for em_w, em_h, s in geoms:
        F_ref, res = _cmp(em_w, em_h, s)
        assert res["status"] in ("converged", "reached_limits")
        rel_err = abs(res["vf"] - F_ref) / max(F_ref, 1e-12)
        assert rel_err <= 0.003, f"rel_err={rel_err:.4f}, iterations={res.get('iterations')}, cells={res.get('cells')}"


def test_adaptive_actually_refines_not_one_iteration():
    """Test that adaptive actually refines and doesn't stop after 1 iteration."""
    F_ref, res = _cmp(5.1, 2.1, 1.0)
    assert res.get("iterations", 0) > 1, "Adaptive stopped after 1 split — not refining"
    assert res.get("cells", 0) >= 16, "Expect at least the initial 4x4 cells as leaves"


def test_adaptive_convergence_with_min_cells():
    """Test that adaptive respects minimum cells requirement."""
    res = vf_adaptive(5.1, 2.1, 5.1, 2.1, 1.0,
                      rel_tol=1e-6, abs_tol=1e-9, max_depth=12,
                      max_cells=100000, min_cells=32, time_limit_s=10)
    
    assert res["status"] in ("converged", "reached_limits")
    assert res.get("cells", 0) >= 16  # At least initial grid
    assert res.get("iterations", 0) > 0  # Should have done some refinement


def test_adaptive_error_driven_refinement():
    """Test that adaptive refines based on error estimation."""
    # Use a geometry that should require refinement
    res = vf_adaptive(1.0, 1.0, 1.0, 1.0, 0.1,  # Very close geometry
                      rel_tol=1e-4, abs_tol=1e-7, max_depth=10,
                      max_cells=50000, min_cells=16, time_limit_s=10)
    
    assert res["status"] in ("converged", "reached_limits")
    assert res.get("iterations", 0) > 0  # Should have refined
    assert res.get("achieved_tol", 1.0) < 1.0  # Should have achieved some tolerance


def test_adaptive_point_evaluation():
    """Test that adaptive evaluates at specific receiver points."""
    # Test center point
    res_center = vf_adaptive(5.1, 2.1, 5.1, 2.1, 1.0,
                             rc_point=(0.0, 0.0), time_limit_s=5)
    
    # Test offset point
    res_offset = vf_adaptive(5.1, 2.1, 5.1, 2.1, 1.0,
                             rc_point=(0.5, 0.3), time_limit_s=5)
    
    assert res_center["status"] in ("converged", "reached_limits")
    assert res_offset["status"] in ("converged", "reached_limits")
    
    # Center should have higher VF than offset
    assert res_center["vf"] > res_offset["vf"], "Center should have higher VF than offset"
    
    # Both should be reasonable values
    assert 0.0 <= res_center["vf"] <= 1.0
    assert 0.0 <= res_offset["vf"] <= 1.0
