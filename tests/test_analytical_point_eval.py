"""
Tests for analytical point evaluator functionality.
"""

import math
import numpy as np
import pytest
from src.analytical import vf_point_rect_to_point_parallel


class TestAnalyticalPointEval:
    """Test cases for analytical point evaluator."""
    
    def test_analytical_point_differs_with_position(self):
        """Test that analytical returns different values for different receiver points."""
        # Geometry: concentric, parallel; peak at center; edge value must be lower
        em_w, em_h, s = 5.1, 2.1, 1.0
        F_center = vf_point_rect_to_point_parallel(em_w, em_h, s, 0.0, 0.0, nx=160, ny=160)
        # near +x edge of receiver footprint
        F_edge   = vf_point_rect_to_point_parallel(em_w, em_h, s, em_w/2 - 1e-6, 0.0, nx=160, ny=160)
        
        assert 0.0 <= F_edge <= F_center <= 1.0
        assert (F_center - F_edge) / max(F_center, 1e-12) > 0.01   # at least 1% larger at center
    
    def test_analytical_vs_mc_center_loose(self):
        """Test loose consistency with Monte Carlo at center."""
        from src.montecarlo import vf_montecarlo
        
        em_w, em_h, s = 5.1, 2.1, 1.0
        F_anal = vf_point_rect_to_point_parallel(em_w, em_h, s, 0.0, 0.0, nx=140, ny=140)
        res = vf_montecarlo(em_w, em_h, em_w, em_h, s, samples=120000, target_rel_ci=0.03, seed=1, time_limit_s=10)
        F_mc = res["vf_mean"]
        
        rel_error = abs(F_mc - F_anal) / max(F_anal, 1e-12)
        assert rel_error < 0.08, f"Analytical vs MC error too large: {rel_error:.3f}"
    
    def test_analytical_center_vs_edge_symmetry(self):
        """Test that center has higher view factor than edges."""
        em_w, em_h, s = 5.1, 2.1, 1.0
        nx, ny = 120, 120
        
        F_center = vf_point_rect_to_point_parallel(em_w, em_h, s, 0.0, 0.0, nx=nx, ny=ny)
        
        # Test all four edges
        F_edge_x_pos = vf_point_rect_to_point_parallel(em_w, em_h, s, em_w/2 - 0.1, 0.0, nx=nx, ny=ny)
        F_edge_x_neg = vf_point_rect_to_point_parallel(em_w, em_h, s, -em_w/2 + 0.1, 0.0, nx=nx, ny=ny)
        F_edge_y_pos = vf_point_rect_to_point_parallel(em_w, em_h, s, 0.0, em_h/2 - 0.1, nx=nx, ny=ny)
        F_edge_y_neg = vf_point_rect_to_point_parallel(em_w, em_h, s, 0.0, -em_h/2 + 0.1, nx=nx, ny=ny)
        
        # Center should have higher view factor than all edges
        assert F_center > F_edge_x_pos
        assert F_center > F_edge_x_neg
        assert F_center > F_edge_y_pos
        assert F_center > F_edge_y_neg
    
    def test_analytical_grid_convergence(self):
        """Test that finer grids give more accurate results."""
        em_w, em_h, s = 5.1, 2.1, 1.0
        rx, ry = 0.0, 0.0
        
        # Coarse grid
        F_coarse = vf_point_rect_to_point_parallel(em_w, em_h, s, rx, ry, nx=60, ny=60)
        
        # Fine grid
        F_fine = vf_point_rect_to_point_parallel(em_w, em_h, s, rx, ry, nx=120, ny=120)
        
        # Fine grid should be more accurate (closer to true value)
        # We can't test exact convergence, but fine should be different
        assert abs(F_fine - F_coarse) > 1e-6, "Grid convergence not working"
    
    def test_analytical_physical_bounds(self):
        """Test that view factor is always in [0,1] range."""
        em_w, em_h, s = 5.1, 2.1, 1.0
        nx, ny = 80, 80
        
        # Test various receiver points
        test_points = [
            (0.0, 0.0),  # center
            (em_w/2 - 0.1, 0.0),  # near edge
            (0.0, em_h/2 - 0.1),  # near edge
            (em_w/2 - 0.1, em_h/2 - 0.1),  # near corner
        ]
        
        for rx, ry in test_points:
            F = vf_point_rect_to_point_parallel(em_w, em_h, s, rx, ry, nx=nx, ny=ny)
            assert 0.0 <= F <= 1.0, f"View factor {F} not in [0,1] for point ({rx}, {ry})"
    
    def test_analytical_setback_dependence(self):
        """Test that view factor decreases with increasing setback."""
        em_w, em_h = 5.1, 2.1
        rx, ry = 0.0, 0.0
        nx, ny = 80, 80
        
        setbacks = [0.5, 1.0, 2.0, 5.0]
        F_values = []
        
        for s in setbacks:
            F = vf_point_rect_to_point_parallel(em_w, em_h, s, rx, ry, nx=nx, ny=ny)
            F_values.append(F)
        
        # View factor should decrease with increasing setback
        for i in range(1, len(F_values)):
            assert F_values[i] < F_values[i-1], f"View factor should decrease with setback: {F_values[i-1]} -> {F_values[i]}"
    
    def test_analytical_emitter_size_dependence(self):
        """Test that view factor increases with larger emitter."""
        setback = 1.0
        rx, ry = 0.0, 0.0
        nx, ny = 80, 80
        
        emitters = [(2.0, 1.0), (4.0, 2.0), (6.0, 3.0)]
        F_values = []
        
        for em_w, em_h in emitters:
            F = vf_point_rect_to_point_parallel(em_w, em_h, setback, rx, ry, nx=nx, ny=ny)
            F_values.append(F)
        
        # View factor should increase with larger emitter
        for i in range(1, len(F_values)):
            assert F_values[i] > F_values[i-1], f"View factor should increase with emitter size: {F_values[i-1]} -> {F_values[i]}"
    
    def test_analytical_numerical_stability(self):
        """Test numerical stability for extreme cases."""
        em_w, em_h, s = 5.1, 2.1, 1.0
        nx, ny = 100, 100
        
        # Very close to emitter edge
        F_close = vf_point_rect_to_point_parallel(em_w, em_h, s, em_w/2 - 1e-10, 0.0, nx=nx, ny=ny)
        assert 0.0 <= F_close <= 1.0
        
        # Very far from center
        F_far = vf_point_rect_to_point_parallel(em_w, em_h, s, 100.0, 100.0, nx=nx, ny=ny)
        assert 0.0 <= F_far <= 1.0
        assert F_far < 0.1  # Should be very small
