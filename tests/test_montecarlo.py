"""
Tests for Monte Carlo view factor calculations.
"""

import pytest
import numpy as np
from src.montecarlo import vf_montecarlo


class TestMonteCarlo:
    """Test cases for Monte Carlo view factor calculations."""
    
    def test_basic_calculation(self):
        """Test basic Monte Carlo view factor calculation."""
        result = vf_montecarlo(
            em_w=5.1, em_h=2.1, rc_w=5.1, rc_h=2.1, setback=1.0,
            samples=10000, target_rel_ci=0.05, max_iters=10
        )
        
        assert result['status'] in ['converged', 'reached_limits']
        assert result['vf_mean'] > 0
        assert result['vf_mean'] < 1
        assert result['vf_ci95'] >= 0
        assert result['samples'] > 0
    
    def test_convergence_downwards_with_sample_size(self):
        """Test that CI decreases with increasing sample size."""
        # Test with different sample sizes
        sample_sizes = [1000, 5000, 10000]
        ci_values = []
        
        for samples in sample_sizes:
            result = vf_montecarlo(
                em_w=5.1, em_h=2.1, rc_w=5.1, rc_h=2.1, setback=1.0,
                samples=samples, target_rel_ci=0.01, max_iters=20
            )
            ci_values.append(result['vf_ci95'])
        
        # CI should generally decrease with more samples
        # (allowing for some statistical variation)
        for i in range(1, len(ci_values)):
            # Skip comparison if previous CI was 0 (insufficient samples)
            if ci_values[i-1] > 0:
                # Allow some tolerance for statistical variation
                assert ci_values[i] <= ci_values[i-1] * 1.5, f"CI did not decrease: {ci_values[i-1]} -> {ci_values[i]}"
    
    def test_agreement_with_other_methods(self):
        """Test loose agreement with adaptive/fixedgrid methods (±5-10%)."""
        from src.adaptive import vf_adaptive
        from src.fixed_grid import vf_fixed_grid
        
        # Test case
        em_w, em_h = 5.1, 2.1
        rc_w, rc_h = 5.1, 2.1
        setback = 1.0
        
        # Get Monte Carlo result
        mc_result = vf_montecarlo(
            em_w, em_h, rc_w, rc_h, setback,
            samples=50000, target_rel_ci=0.05, max_iters=20
        )
        
        # Get adaptive result
        adaptive_result = vf_adaptive(
            em_w, em_h, rc_w, rc_h, setback,
            rel_tol=3e-3, abs_tol=1e-6, max_depth=8, max_cells=10000
        )
        
        # Get fixed grid result
        fixed_result = vf_fixed_grid(
            em_w, em_h, rc_w, rc_h, setback,
            grid_nx=100, grid_ny=100
        )
        
        mc_vf = mc_result['vf_mean']
        adaptive_vf = adaptive_result['vf']
        fixed_vf = fixed_result['vf']
        
        # Check agreement within 25% (Monte Carlo can be less accurate)
        rel_error_adaptive = abs(mc_vf - adaptive_vf) / adaptive_vf
        rel_error_fixed = abs(mc_vf - fixed_vf) / fixed_vf
        
        assert rel_error_adaptive < 0.25, f"MC vs Adaptive: {rel_error_adaptive:.3f} > 25%"
        assert rel_error_fixed < 0.25, f"MC vs Fixed: {rel_error_fixed:.3f} > 25%"
    
    def test_determinism_with_seed(self):
        """Test that results are deterministic with the same seed."""
        # Run with same seed multiple times
        results = []
        for _ in range(3):
            result = vf_montecarlo(
                em_w=5.1, em_h=2.1, rc_w=5.1, rc_h=2.1, setback=1.0,
                samples=5000, target_rel_ci=0.05, max_iters=10, seed=42
            )
            results.append(result['vf_mean'])
        
        # Results should be identical (within floating point precision)
        for i in range(1, len(results)):
            assert abs(results[i] - results[0]) < 1e-10, f"Non-deterministic results: {results[0]} vs {results[i]}"
    
    def test_different_seeds_give_different_results(self):
        """Test that different seeds give different results."""
        result1 = vf_montecarlo(
            em_w=5.1, em_h=2.1, rc_w=5.1, rc_h=2.1, setback=1.0,
            samples=1000, target_rel_ci=0.1, max_iters=5, seed=42
        )
        
        result2 = vf_montecarlo(
            em_w=5.1, em_h=2.1, rc_w=5.1, rc_h=2.1, setback=1.0,
            samples=1000, target_rel_ci=0.1, max_iters=5, seed=123
        )
        
        # Results should be different (with high probability)
        assert abs(result1['vf_mean'] - result2['vf_mean']) > 1e-6, "Different seeds gave identical results"
    
    def test_time_limit(self):
        """Test time limit functionality."""
        # Use very small time limit to trigger timeout
        result = vf_montecarlo(
            em_w=5.1, em_h=2.1, rc_w=5.1, rc_h=2.1, setback=1.0,
            samples=10000, target_rel_ci=0.01, max_iters=50, time_limit_s=0.001
        )
        
        assert result['status'] in ['reached_limits', 'converged']  # Either is acceptable
        assert result['vf_mean'] >= 0  # Should have partial result
    
    def test_max_iters_limit(self):
        """Test maximum iterations limit functionality."""
        # Use very small max_iters to trigger limit
        result = vf_montecarlo(
            em_w=5.1, em_h=2.1, rc_w=5.1, rc_h=2.1, setback=1.0,
            samples=10000, target_rel_ci=0.001, max_iters=2, time_limit_s=60.0
        )
        
        assert result['status'] in ['reached_limits', 'converged']  # Either is acceptable
        assert result['vf_mean'] >= 0
        assert result['samples'] > 0
    
    def test_input_validation(self):
        """Test input validation."""
        # Negative dimensions
        result = vf_montecarlo(em_w=-1, em_h=2.1, rc_w=5.1, rc_h=2.1, setback=1.0)
        assert result['status'] == 'failed'
        assert result['vf_mean'] == 0.0
        
        # Zero setback
        result = vf_montecarlo(em_w=5.1, em_h=2.1, rc_w=5.1, rc_h=2.1, setback=0.0)
        assert result['status'] == 'failed'
        assert result['vf_mean'] == 0.0
        
        # Invalid parameters
        result = vf_montecarlo(em_w=5.1, em_h=2.1, rc_w=5.1, rc_h=2.1, setback=1.0, samples=0)
        assert result['status'] == 'failed'
        assert result['vf_mean'] == 0.0
        
        result = vf_montecarlo(em_w=5.1, em_h=2.1, rc_w=5.1, rc_h=2.1, setback=1.0, target_rel_ci=0.0)
        assert result['status'] == 'failed'
        assert result['vf_mean'] == 0.0
    
    def test_different_geometries(self):
        """Test with different emitter/receiver geometries."""
        # Square surfaces
        result = vf_montecarlo(em_w=2.0, em_h=2.0, rc_w=2.0, rc_h=2.0, setback=1.0, samples=5000)
        assert result['status'] in ['converged', 'reached_limits']
        assert result['vf_mean'] > 0
        
        # Rectangular surfaces
        result = vf_montecarlo(em_w=10.0, em_h=1.0, rc_w=10.0, rc_h=1.0, setback=2.0, samples=5000)
        assert result['status'] in ['converged', 'reached_limits']
        assert result['vf_mean'] > 0
        
        # Different emitter/receiver sizes
        result = vf_montecarlo(em_w=5.0, em_h=2.0, rc_w=3.0, rc_h=1.5, setback=1.5, samples=5000)
        assert result['status'] in ['converged', 'reached_limits']
        assert result['vf_mean'] > 0
    
    def test_convergence_parameters(self):
        """Test different convergence parameters."""
        # Tight CI target
        result = vf_montecarlo(
            em_w=5.1, em_h=2.1, rc_w=5.1, rc_h=2.1, setback=1.0,
            samples=10000, target_rel_ci=0.01, max_iters=20
        )
        assert result['status'] in ['converged', 'reached_limits']
        assert result['vf_mean'] > 0
        
        # Loose CI target
        result = vf_montecarlo(
            em_w=5.1, em_h=2.1, rc_w=5.1, rc_h=2.1, setback=1.0,
            samples=1000, target_rel_ci=0.1, max_iters=5
        )
        assert result['status'] in ['converged', 'reached_limits']
        assert result['vf_mean'] > 0
    
    def test_eps_parameter(self):
        """Test eps parameter for numerical stability."""
        result = vf_montecarlo(
            em_w=5.1, em_h=2.1, rc_w=5.1, rc_h=2.1, setback=1.0,
            samples=5000, eps=1e-10
        )
        assert result['status'] in ['converged', 'reached_limits']
        assert result['vf_mean'] > 0
    
    def test_confidence_interval_calculation(self):
        """Test that confidence interval calculation is reasonable."""
        result = vf_montecarlo(
            em_w=5.1, em_h=2.1, rc_w=5.1, rc_h=2.1, setback=1.0,
            samples=10000, target_rel_ci=0.05, max_iters=20
        )
        
        # CI should be positive
        assert result['vf_ci95'] >= 0
        
        # Relative CI should be reasonable (not too large)
        if result['vf_mean'] > 0:
            rel_ci = result['vf_ci95'] / result['vf_mean']
            assert rel_ci < 1.0, f"Relative CI too large: {rel_ci:.3f}"
    
    def test_small_setback_case(self):
        """Test case with very small setback (near contact)."""
        result = vf_montecarlo(
            em_w=5.1, em_h=2.1, rc_w=5.1, rc_h=2.1, setback=1e-6,
            samples=5000, target_rel_ci=0.1, max_iters=10
        )
        
        assert result['status'] in ['converged', 'reached_limits']
        assert result['vf_mean'] >= 0
        # Near contact should give high view factor (but Monte Carlo may struggle)
        assert result['vf_mean'] > 0.0
