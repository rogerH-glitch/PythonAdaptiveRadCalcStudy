"""
Tests for peak locator functionality.
"""

import pytest
import numpy as np
from src.peak_locator import find_local_peak, create_vf_evaluator


class TestPeakLocator:
    """Test cases for peak locator functionality."""
    
    def test_center_mode_concentric_parallel(self):
        """Test that center mode finds peak at center for concentric parallel case."""
        evaluator = create_vf_evaluator('analytical', 5.1, 2.1, 5.1, 2.1, 1.0, 0.0)
        result = find_local_peak(
            5.1, 2.1, 5.1, 2.1, 1.0, 0.0, evaluator, rc_mode='center'
        )
        
        assert result['status'] == 'converged'
        assert result['x_peak'] == 0.0
        assert result['y_peak'] == 0.0
        assert result['vf_peak'] > 0
        assert result['search_metadata']['method'] == 'center'
    
    def test_grid_mode_concentric_parallel(self):
        """Test that grid mode finds peak near center for concentric parallel case."""
        evaluator = create_vf_evaluator('analytical', 5.1, 2.1, 5.1, 2.1, 1.0, 0.0)
        result = find_local_peak(
            5.1, 2.1, 5.1, 2.1, 1.0, 0.0, evaluator, rc_mode='grid', rc_grid_n=11
        )
        
        assert result['status'] == 'converged'
        # Peak should be near center (within 10% of receiver size)
        assert abs(result['x_peak']) < 0.1 * 5.1
        assert abs(result['y_peak']) < 0.1 * 2.1
        assert result['vf_peak'] > 0
        assert result['search_metadata']['method'] == 'grid'
        assert result['search_metadata']['evaluations'] == 121  # 11x11 grid
    
    def test_search_mode_concentric_parallel(self):
        """Test that search mode finds peak near center for concentric parallel case."""
        evaluator = create_vf_evaluator('analytical', 5.1, 2.1, 5.1, 2.1, 1.0, 0.0)
        result = find_local_peak(
            5.1, 2.1, 5.1, 2.1, 1.0, 0.0, evaluator, rc_mode='search', 
            rc_grid_n=11, rc_search_multistart=4
        )
        
        assert result['status'] in ['converged', 'reached_limits']
        # Peak should be near center (within 5% of receiver size)
        assert abs(result['x_peak']) < 0.05 * 5.1
        assert abs(result['y_peak']) < 0.05 * 2.1
        assert result['vf_peak'] > 0
        assert result['search_metadata']['method'] == 'search'
        assert result['search_metadata']['evaluations'] >= 121  # At least grid evaluations
    
    def test_offset_case_peak_shift(self):
        """Test that peak shifts appropriately for offset receiver."""
        # This test would require implementing offset geometry support
        # For now, just test that the system handles different geometries
        evaluator = create_vf_evaluator('analytical', 5.1, 2.1, 3.0, 1.5, 1.0, 0.0)
        result = find_local_peak(
            5.1, 2.1, 3.0, 1.5, 1.0, 0.0, evaluator, rc_mode='grid', rc_grid_n=11
        )
        
        assert result['status'] == 'converged'
        assert result['vf_peak'] > 0
        # Peak should be within receiver bounds
        assert -1.5 <= result['x_peak'] <= 1.5  # rc_w/2
        assert -0.75 <= result['y_peak'] <= 0.75  # rc_h/2
    
    def test_different_methods(self):
        """Test peak locator with different calculation methods."""
        methods = ['analytical', 'fixedgrid', 'montecarlo']
        
        for method in methods:
            evaluator = create_vf_evaluator(method, 5.1, 2.1, 5.1, 2.1, 1.0, 0.0)
            result = find_local_peak(
                5.1, 2.1, 5.1, 2.1, 1.0, 0.0, evaluator, rc_mode='center'
            )
            
            assert result['status'] == 'converged'
            assert result['vf_peak'] > 0
            assert result['x_peak'] == 0.0
            assert result['y_peak'] == 0.0
    
    def test_search_parameters(self):
        """Test different search parameters."""
        evaluator = create_vf_evaluator('analytical', 5.1, 2.1, 5.1, 2.1, 1.0, 0.0)
        
        # Test with different grid sizes
        for grid_n in [5, 11, 21]:
            result = find_local_peak(
                5.1, 2.1, 5.1, 2.1, 1.0, 0.0, evaluator, rc_mode='grid', rc_grid_n=grid_n
            )
            assert result['status'] == 'converged'
            assert result['search_metadata']['evaluations'] == grid_n * grid_n
    
    def test_time_limit(self):
        """Test time limit functionality."""
        evaluator = create_vf_evaluator('analytical', 5.1, 2.1, 5.1, 2.1, 1.0, 0.0)
        result = find_local_peak(
            5.1, 2.1, 5.1, 2.1, 1.0, 0.0, evaluator, rc_mode='search',
            rc_search_time_limit_s=0.001  # Very short time limit
        )
        
        assert result['status'] in ['converged', 'reached_limits']
        assert result['vf_peak'] > 0
    
    def test_input_validation(self):
        """Test input validation."""
        evaluator = create_vf_evaluator('analytical', 5.1, 2.1, 5.1, 2.1, 1.0, 0.0)
        
        # Invalid geometry
        result = find_local_peak(
            -1.0, 2.1, 5.1, 2.1, 1.0, 0.0, evaluator, rc_mode='center'
        )
        assert result['status'] == 'failed'
        assert result['vf_peak'] == 0.0
    
    def test_search_metadata(self):
        """Test that search metadata is properly populated."""
        evaluator = create_vf_evaluator('analytical', 5.1, 2.1, 5.1, 2.1, 1.0, 0.0)
        result = find_local_peak(
            5.1, 2.1, 5.1, 2.1, 1.0, 0.0, evaluator, rc_mode='search',
            rc_grid_n=11, rc_search_multistart=4
        )
        
        metadata = result['search_metadata']
        assert 'time_s' in metadata
        assert 'evaluations' in metadata
        assert 'method' in metadata
        assert metadata['method'] == 'search'
        assert metadata['evaluations'] >= 121  # At least grid evaluations
    
    def test_peak_value_consistency(self):
        """Test that peak values are consistent across modes for same geometry."""
        evaluator = create_vf_evaluator('analytical', 5.1, 2.1, 5.1, 2.1, 1.0, 0.0)
        
        # Center mode
        center_result = find_local_peak(
            5.1, 2.1, 5.1, 2.1, 1.0, 0.0, evaluator, rc_mode='center'
        )
        
        # Grid mode
        grid_result = find_local_peak(
            5.1, 2.1, 5.1, 2.1, 1.0, 0.0, evaluator, rc_mode='grid', rc_grid_n=21
        )
        
        # Values should be close (within 1% for analytical method)
        rel_error = abs(center_result['vf_peak'] - grid_result['vf_peak']) / center_result['vf_peak']
        assert rel_error < 0.01, f"Peak values differ by {rel_error:.3f}"
    
    def test_receiver_bounds(self):
        """Test that peak locations are within receiver bounds."""
        evaluator = create_vf_evaluator('analytical', 5.1, 2.1, 3.0, 1.5, 1.0, 0.0)
        result = find_local_peak(
            5.1, 2.1, 3.0, 1.5, 1.0, 0.0, evaluator, rc_mode='grid', rc_grid_n=11
        )
        
        # Peak should be within receiver bounds
        assert -1.5 <= result['x_peak'] <= 1.5  # rc_w/2
        assert -0.75 <= result['y_peak'] <= 0.75  # rc_h/2
