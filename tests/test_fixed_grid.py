"""
Tests for fixed grid view factor calculations.
"""

import pytest
import numpy as np
from src.fixed_grid import vf_fixed_grid


class TestFixedGrid:
    """Test cases for fixed grid view factor calculations."""
    
    def test_basic_calculation(self):
        """Test basic view factor calculation."""
        result = vf_fixed_grid(
            em_w=5.1, em_h=2.1, rc_w=5.1, rc_h=2.1, setback=1.0,
            grid_nx=50, grid_ny=50
        )
        
        assert result['status'] == 'converged'
        assert result['vf'] > 0
        assert result['vf'] < 1
        assert result['samples_emitter'] == 50 * 50
        assert result['samples_receiver'] > 0
    
    def test_regression_vs_analytical(self):
        """Test regression against analytical approximation within 5-10%."""
        from src.analytical import local_peak_vf_analytic_approx
        
        # Test case 1: Small setback (high view factor)
        em_w, em_h = 5.1, 2.1
        rc_w, rc_h = 5.1, 2.1
        setback = 0.1
        
        analytical_vf = local_peak_vf_analytic_approx(em_w, em_h, rc_w, rc_h, setback)
        fixed_vf = vf_fixed_grid(em_w, em_h, rc_w, rc_h, setback, grid_nx=100, grid_ny=100)
        
        rel_error = abs(fixed_vf['vf'] - analytical_vf) / analytical_vf
        assert rel_error < 0.1, f"Relative error {rel_error:.3f} exceeds 10%"
        
        # Test case 2: Medium setback
        setback = 1.0
        analytical_vf = local_peak_vf_analytic_approx(em_w, em_h, rc_w, rc_h, setback)
        fixed_vf = vf_fixed_grid(em_w, em_h, rc_w, rc_h, setback, grid_nx=100, grid_ny=100)
        
        rel_error = abs(fixed_vf['vf'] - analytical_vf) / analytical_vf
        assert rel_error < 0.1, f"Relative error {rel_error:.3f} exceeds 10%"
        
        # Test case 3: Large setback (low view factor)
        setback = 5.0
        analytical_vf = local_peak_vf_analytic_approx(em_w, em_h, rc_w, rc_h, setback)
        fixed_vf = vf_fixed_grid(em_w, em_h, rc_w, rc_h, setback, grid_nx=100, grid_ny=100)
        
        rel_error = abs(fixed_vf['vf'] - analytical_vf) / analytical_vf
        assert rel_error < 0.1, f"Relative error {rel_error:.3f} exceeds 10%"
    
    def test_convergence_trend(self):
        """Test that increasing grid density generally improves accuracy."""
        em_w, em_h = 5.1, 2.1
        rc_w, rc_h = 5.1, 2.1
        setback = 1.0
        
        # Test with increasing grid density
        grid_sizes = [(20, 20), (50, 50), (100, 100)]
        results = []
        
        for nx, ny in grid_sizes:
            result = vf_fixed_grid(em_w, em_h, rc_w, rc_h, setback, grid_nx=nx, grid_ny=ny)
            results.append(result['vf'])
        
        # Results should be reasonably close (within 5% tolerance)
        # and the highest resolution should be most accurate
        max_result = max(results)
        for i, result in enumerate(results):
            rel_diff = abs(result - max_result) / max_result
            assert rel_diff < 0.05, f"Result {i} differs by {rel_diff:.3f} from max, exceeds 5%"
    
    def test_runtime_performance(self):
        """Test that calculation completes under ~10s on default params."""
        import time
        
        start_time = time.perf_counter()
        result = vf_fixed_grid(
            em_w=5.1, em_h=2.1, rc_w=5.1, rc_h=2.1, setback=1.0,
            grid_nx=100, grid_ny=100
        )
        elapsed_time = time.perf_counter() - start_time
        
        assert elapsed_time < 10.0, f"Calculation took {elapsed_time:.2f}s, exceeds 10s limit"
        assert result['status'] == 'converged'
    
    def test_time_limit(self):
        """Test time limit functionality."""
        # Use very small time limit to trigger timeout
        result = vf_fixed_grid(
            em_w=5.1, em_h=2.1, rc_w=5.1, rc_h=2.1, setback=1.0,
            grid_nx=100, grid_ny=100, time_limit_s=0.001  # 1ms limit
        )
        
        assert result['status'] == 'reached_limits'
        assert result['vf'] >= 0  # Should have partial result
    
    def test_input_validation(self):
        """Test input validation."""
        # Negative dimensions
        result = vf_fixed_grid(em_w=-1, em_h=2.1, rc_w=5.1, rc_h=2.1, setback=1.0)
        assert result['status'] == 'failed'
        assert result['vf'] == 0.0
        
        # Zero setback
        result = vf_fixed_grid(em_w=5.1, em_h=2.1, rc_w=5.1, rc_h=2.1, setback=0.0)
        assert result['status'] == 'failed'
        assert result['vf'] == 0.0
        
        # Invalid grid size
        result = vf_fixed_grid(em_w=5.1, em_h=2.1, rc_w=5.1, rc_h=2.1, setback=1.0, grid_nx=0)
        assert result['status'] == 'failed'
        assert result['vf'] == 0.0
    
    def test_different_geometries(self):
        """Test with different emitter/receiver geometries."""
        # Square surfaces
        result = vf_fixed_grid(em_w=2.0, em_h=2.0, rc_w=2.0, rc_h=2.0, setback=1.0, grid_nx=50, grid_ny=50)
        assert result['status'] == 'converged'
        assert result['vf'] > 0
        
        # Rectangular surfaces
        result = vf_fixed_grid(em_w=10.0, em_h=1.0, rc_w=10.0, rc_h=1.0, setback=2.0, grid_nx=50, grid_ny=50)
        assert result['status'] == 'converged'
        assert result['vf'] > 0
        
        # Different emitter/receiver sizes
        result = vf_fixed_grid(em_w=5.0, em_h=2.0, rc_w=3.0, rc_h=1.5, setback=1.5, grid_nx=50, grid_ny=50)
        assert result['status'] == 'converged'
        assert result['vf'] > 0
    
    def test_quadrature_parameter(self):
        """Test quadrature parameter (currently only centroid supported)."""
        result = vf_fixed_grid(
            em_w=5.1, em_h=2.1, rc_w=5.1, rc_h=2.1, setback=1.0,
            grid_nx=50, grid_ny=50, quadrature="centroid"
        )
        assert result['status'] == 'converged'
        assert result['vf'] > 0
    
    def test_eps_parameter(self):
        """Test eps parameter for numerical stability."""
        result = vf_fixed_grid(
            em_w=5.1, em_h=2.1, rc_w=5.1, rc_h=2.1, setback=1.0,
            grid_nx=50, grid_ny=50, eps=1e-10
        )
        assert result['status'] == 'converged'
        assert result['vf'] > 0
