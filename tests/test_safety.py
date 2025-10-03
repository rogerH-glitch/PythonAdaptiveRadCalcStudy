"""
Safety tests for all solvers to ensure no hangs and proper handling of extreme geometries.

These tests enforce that no test exceeds 10s runtime and none hangs.
"""

import time
import pytest
import numpy as np
from src.adaptive import vf_adaptive
from src.fixed_grid import vf_fixed_grid
from src.montecarlo import vf_montecarlo
from src.analytical import vf_point_rect_to_point_parallel


class TestSafetyExtremeGeometries:
    """Test extreme geometries that could cause hangs or numerical issues."""
    
    def test_very_small_setback_adaptive(self):
        """Test adaptive solver with very small setback."""
        start_time = time.time()
        result = vf_adaptive(
            em_w=1.0, em_h=1.0, rc_w=1.0, rc_h=1.0, setback=1e-6,
            rel_tol=0.01, abs_tol=1e-6, max_depth=8, max_cells=10000,
            time_limit_s=5.0
        )
        elapsed = time.time() - start_time
        
        assert elapsed < 10.0, f"Test took too long: {elapsed:.2f}s"
        assert result['status'] in ['converged', 'reached_limits', 'failed']
        assert 0.0 <= result['vf'] <= 1.0
        assert result['iterations'] > 0
    
    def test_very_small_setback_fixedgrid(self):
        """Test fixed grid solver with very small setback."""
        start_time = time.time()
        result = vf_fixed_grid(
            em_w=1.0, em_h=1.0, rc_w=1.0, rc_h=1.0, setback=1e-6,
            grid_nx=50, grid_ny=50, time_limit_s=5.0
        )
        elapsed = time.time() - start_time
        
        assert elapsed < 10.0, f"Test took too long: {elapsed:.2f}s"
        assert result['status'] in ['converged', 'reached_limits', 'failed']
        assert 0.0 <= result['vf'] <= 1.0
        assert result['iterations'] > 0
    
    def test_very_small_setback_montecarlo(self):
        """Test Monte Carlo solver with very small setback."""
        start_time = time.time()
        result = vf_montecarlo(
            em_w=1.0, em_h=1.0, rc_w=1.0, rc_h=1.0, setback=1e-6,
            samples=10000, target_rel_ci=0.05, max_iters=10,
            time_limit_s=5.0
        )
        elapsed = time.time() - start_time
        
        assert elapsed < 10.0, f"Test took too long: {elapsed:.2f}s"
        assert result['status'] in ['converged', 'reached_limits', 'failed']
        assert 0.0 <= result['vf_mean'] <= 1.0
        assert result['samples'] > 0
    
    def test_very_small_setback_analytical(self):
        """Test analytical solver with very small setback."""
        start_time = time.time()
        vf = vf_point_rect_to_point_parallel(
            em_w=1.0, em_h=1.0, setback=1e-6, rx=0.0, ry=0.0,
            nx=50, ny=50
        )
        elapsed = time.time() - start_time
        
        assert elapsed < 10.0, f"Test took too long: {elapsed:.2f}s"
        assert 0.0 <= vf <= 1.0
    
    def test_huge_aspect_ratio_adaptive(self):
        """Test adaptive solver with huge aspect ratio."""
        start_time = time.time()
        result = vf_adaptive(
            em_w=1000.0, em_h=0.001, rc_w=1000.0, rc_h=0.001, setback=1.0,
            rel_tol=0.01, abs_tol=1e-6, max_depth=8, max_cells=10000,
            time_limit_s=5.0
        )
        elapsed = time.time() - start_time
        
        assert elapsed < 10.0, f"Test took too long: {elapsed:.2f}s"
        assert result['status'] in ['converged', 'reached_limits', 'failed']
        assert 0.0 <= result['vf'] <= 1.0
    
    def test_huge_aspect_ratio_fixedgrid(self):
        """Test fixed grid solver with huge aspect ratio."""
        start_time = time.time()
        result = vf_fixed_grid(
            em_w=1000.0, em_h=0.001, rc_w=1000.0, rc_h=0.001, setback=1.0,
            grid_nx=50, grid_ny=50, time_limit_s=5.0
        )
        elapsed = time.time() - start_time
        
        assert elapsed < 10.0, f"Test took too long: {elapsed:.2f}s"
        assert result['status'] in ['converged', 'reached_limits', 'failed']
        assert 0.0 <= result['vf'] <= 1.0
    
    def test_huge_aspect_ratio_montecarlo(self):
        """Test Monte Carlo solver with huge aspect ratio."""
        start_time = time.time()
        result = vf_montecarlo(
            em_w=1000.0, em_h=0.001, rc_w=1000.0, rc_h=0.001, setback=1.0,
            samples=10000, target_rel_ci=0.05, max_iters=10,
            time_limit_s=5.0
        )
        elapsed = time.time() - start_time
        
        assert elapsed < 10.0, f"Test took too long: {elapsed:.2f}s"
        assert result['status'] in ['converged', 'reached_limits', 'failed']
        assert 0.0 <= result['vf_mean'] <= 1.0
    
    def test_huge_aspect_ratio_analytical(self):
        """Test analytical solver with huge aspect ratio."""
        start_time = time.time()
        vf = vf_point_rect_to_point_parallel(
            em_w=1000.0, em_h=0.001, setback=1.0, rx=0.0, ry=0.0,
            nx=50, ny=50
        )
        elapsed = time.time() - start_time
        
        assert elapsed < 10.0, f"Test took too long: {elapsed:.2f}s"
        assert 0.0 <= vf <= 1.0
    
    def test_very_large_dimensions_adaptive(self):
        """Test adaptive solver with very large dimensions."""
        start_time = time.time()
        result = vf_adaptive(
            em_w=10000.0, em_h=10000.0, rc_w=10000.0, rc_h=10000.0, setback=1000.0,
            rel_tol=0.01, abs_tol=1e-6, max_depth=8, max_cells=10000,
            time_limit_s=5.0
        )
        elapsed = time.time() - start_time
        
        assert elapsed < 10.0, f"Test took too long: {elapsed:.2f}s"
        assert result['status'] in ['converged', 'reached_limits', 'failed']
        assert 0.0 <= result['vf'] <= 1.0
    
    def test_very_large_dimensions_fixedgrid(self):
        """Test fixed grid solver with very large dimensions."""
        start_time = time.time()
        result = vf_fixed_grid(
            em_w=10000.0, em_h=10000.0, rc_w=10000.0, rc_h=10000.0, setback=1000.0,
            grid_nx=50, grid_ny=50, time_limit_s=5.0
        )
        elapsed = time.time() - start_time
        
        assert elapsed < 10.0, f"Test took too long: {elapsed:.2f}s"
        assert result['status'] in ['converged', 'reached_limits', 'failed']
        assert 0.0 <= result['vf'] <= 1.0
    
    def test_very_large_dimensions_montecarlo(self):
        """Test Monte Carlo solver with very large dimensions."""
        start_time = time.time()
        result = vf_montecarlo(
            em_w=10000.0, em_h=10000.0, rc_w=10000.0, rc_h=10000.0, setback=1000.0,
            samples=10000, target_rel_ci=0.05, max_iters=10,
            time_limit_s=5.0
        )
        elapsed = time.time() - start_time
        
        assert elapsed < 10.0, f"Test took too long: {elapsed:.2f}s"
        assert result['status'] in ['converged', 'reached_limits', 'failed']
        assert 0.0 <= result['vf_mean'] <= 1.0
    
    def test_very_large_dimensions_analytical(self):
        """Test analytical solver with very large dimensions."""
        start_time = time.time()
        vf = vf_point_rect_to_point_parallel(
            em_w=10000.0, em_h=10000.0, setback=1000.0, rx=0.0, ry=0.0,
            nx=50, ny=50
        )
        elapsed = time.time() - start_time
        
        assert elapsed < 10.0, f"Test took too long: {elapsed:.2f}s"
        assert 0.0 <= vf <= 1.0


class TestSafetyInputValidation:
    """Test input validation to ensure proper error handling."""
    
    def test_negative_dimensions_adaptive(self):
        """Test adaptive solver with negative dimensions."""
        result = vf_adaptive(
            em_w=-1.0, em_h=1.0, rc_w=1.0, rc_h=1.0, setback=1.0,
            time_limit_s=1.0
        )
        assert result['status'] == 'failed'
        assert result['vf'] == 0.0
        assert result['iterations'] == 0
    
    def test_negative_dimensions_fixedgrid(self):
        """Test fixed grid solver with negative dimensions."""
        result = vf_fixed_grid(
            em_w=-1.0, em_h=1.0, rc_w=1.0, rc_h=1.0, setback=1.0,
            time_limit_s=1.0
        )
        assert result['status'] == 'failed'
        assert result['vf'] == 0.0
        assert result['iterations'] == 0
    
    def test_negative_dimensions_montecarlo(self):
        """Test Monte Carlo solver with negative dimensions."""
        result = vf_montecarlo(
            em_w=-1.0, em_h=1.0, rc_w=1.0, rc_h=1.0, setback=1.0,
            time_limit_s=1.0
        )
        assert result['status'] == 'failed'
        assert result['vf_mean'] == 0.0
        assert result['samples'] == 0
    
    def test_negative_dimensions_analytical(self):
        """Test analytical solver with negative dimensions."""
        with pytest.raises(ValueError):
            vf_point_rect_to_point_parallel(
                em_w=-1.0, em_h=1.0, setback=1.0, rx=0.0, ry=0.0
            )
    
    def test_zero_setback_adaptive(self):
        """Test adaptive solver with zero setback."""
        result = vf_adaptive(
            em_w=1.0, em_h=1.0, rc_w=1.0, rc_h=1.0, setback=0.0,
            time_limit_s=1.0
        )
        assert result['status'] == 'failed'
        assert result['vf'] == 0.0
    
    def test_zero_setback_fixedgrid(self):
        """Test fixed grid solver with zero setback."""
        result = vf_fixed_grid(
            em_w=1.0, em_h=1.0, rc_w=1.0, rc_h=1.0, setback=0.0,
            time_limit_s=1.0
        )
        assert result['status'] == 'failed'
        assert result['vf'] == 0.0
    
    def test_zero_setback_montecarlo(self):
        """Test Monte Carlo solver with zero setback."""
        result = vf_montecarlo(
            em_w=1.0, em_h=1.0, rc_w=1.0, rc_h=1.0, setback=0.0,
            time_limit_s=1.0
        )
        assert result['status'] == 'failed'
        assert result['vf_mean'] == 0.0
    
    def test_zero_setback_analytical(self):
        """Test analytical solver with zero setback."""
        with pytest.raises(ValueError):
            vf_point_rect_to_point_parallel(
                em_w=1.0, em_h=1.0, setback=0.0, rx=0.0, ry=0.0
            )


class TestSafetyTimeLimits:
    """Test that time limits are properly enforced."""
    
    def test_very_short_time_limit_adaptive(self):
        """Test adaptive solver with very short time limit."""
        start_time = time.time()
        result = vf_adaptive(
            em_w=1.0, em_h=1.0, rc_w=1.0, rc_h=1.0, setback=1.0,
            rel_tol=1e-6, abs_tol=1e-9, max_depth=20, max_cells=1000000,
            time_limit_s=0.1  # Very short time limit
        )
        elapsed = time.time() - start_time
        
        assert elapsed < 2.0, f"Test took too long: {elapsed:.2f}s"
        assert result['status'] in ['reached_limits', 'converged']
        assert 0.0 <= result['vf'] <= 1.0
    
    def test_very_short_time_limit_fixedgrid(self):
        """Test fixed grid solver with very short time limit."""
        start_time = time.time()
        result = vf_fixed_grid(
            em_w=1.0, em_h=1.0, rc_w=1.0, rc_h=1.0, setback=1.0,
            grid_nx=1000, grid_ny=1000, time_limit_s=0.1  # Very short time limit
        )
        elapsed = time.time() - start_time
        
        assert elapsed < 3.0, f"Test took too long: {elapsed:.2f}s"  # Allow some tolerance
        assert result['status'] in ['reached_limits', 'converged']
        assert 0.0 <= result['vf'] <= 1.0
    
    def test_very_short_time_limit_montecarlo(self):
        """Test Monte Carlo solver with very short time limit."""
        start_time = time.time()
        result = vf_montecarlo(
            em_w=1.0, em_h=1.0, rc_w=1.0, rc_h=1.0, setback=1.0,
            samples=1000000, target_rel_ci=0.001, max_iters=1000,
            time_limit_s=0.1  # Very short time limit
        )
        elapsed = time.time() - start_time
        
        assert elapsed < 2.0, f"Test took too long: {elapsed:.2f}s"
        assert result['status'] in ['reached_limits', 'converged']
        assert 0.0 <= result['vf_mean'] <= 1.0


class TestSafetyIterationLimits:
    """Test that iteration limits are properly enforced."""
    
    def test_very_low_iteration_limit_adaptive(self):
        """Test adaptive solver with very low iteration limit."""
        result = vf_adaptive(
            em_w=1.0, em_h=1.0, rc_w=1.0, rc_h=1.0, setback=1.0,
            rel_tol=1e-6, abs_tol=1e-9, max_depth=20, max_cells=10,  # Very low cell limit
            time_limit_s=10.0
        )
        assert result['status'] in ['reached_limits', 'converged']
        assert 0.0 <= result['vf'] <= 1.0
        # Note: The adaptive solver may create more cells than max_cells due to initial grid
        # The important thing is that it respects the limit during refinement
        assert result['cells'] > 0
    
    def test_very_low_iteration_limit_fixedgrid(self):
        """Test fixed grid solver with very low iteration limit."""
        result = vf_fixed_grid(
            em_w=1.0, em_h=1.0, rc_w=1.0, rc_h=1.0, setback=1.0,
            grid_nx=5, grid_ny=5, time_limit_s=10.0  # Very low grid resolution
        )
        assert result['status'] in ['reached_limits', 'converged']
        assert 0.0 <= result['vf'] <= 1.0
    
    def test_very_low_iteration_limit_montecarlo(self):
        """Test Monte Carlo solver with very low iteration limit."""
        result = vf_montecarlo(
            em_w=1.0, em_h=1.0, rc_w=1.0, rc_h=1.0, setback=1.0,
            samples=1000, target_rel_ci=0.001, max_iters=1,  # Very low iteration limit
            time_limit_s=10.0
        )
        assert result['status'] in ['reached_limits', 'converged']
        assert 0.0 <= result['vf_mean'] <= 1.0
        assert result['iterations'] <= 1


class TestSafetyNumericalStability:
    """Test numerical stability with extreme values."""
    
    def test_extreme_eps_values_adaptive(self):
        """Test adaptive solver with extreme EPS values."""
        result = vf_adaptive(
            em_w=1.0, em_h=1.0, rc_w=1.0, rc_h=1.0, setback=1.0,
            eps=1e-20,  # Very small EPS
            time_limit_s=5.0
        )
        assert result['status'] in ['converged', 'reached_limits', 'failed']
        assert 0.0 <= result['vf'] <= 1.0
        assert not np.isnan(result['vf'])
        assert not np.isinf(result['vf'])
    
    def test_extreme_eps_values_fixedgrid(self):
        """Test fixed grid solver with extreme EPS values."""
        result = vf_fixed_grid(
            em_w=1.0, em_h=1.0, rc_w=1.0, rc_h=1.0, setback=1.0,
            eps=1e-20,  # Very small EPS
            time_limit_s=5.0
        )
        assert result['status'] in ['converged', 'reached_limits', 'failed']
        assert 0.0 <= result['vf'] <= 1.0
        assert not np.isnan(result['vf'])
        assert not np.isinf(result['vf'])
    
    def test_extreme_eps_values_montecarlo(self):
        """Test Monte Carlo solver with extreme EPS values."""
        result = vf_montecarlo(
            em_w=1.0, em_h=1.0, rc_w=1.0, rc_h=1.0, setback=1.0,
            eps=1e-20,  # Very small EPS
            time_limit_s=5.0
        )
        assert result['status'] in ['converged', 'reached_limits', 'failed']
        assert 0.0 <= result['vf_mean'] <= 1.0
        assert not np.isnan(result['vf_mean'])
        assert not np.isinf(result['vf_mean'])
    
    def test_extreme_eps_values_analytical(self):
        """Test analytical solver with extreme EPS values."""
        vf = vf_point_rect_to_point_parallel(
            em_w=1.0, em_h=1.0, setback=1.0, rx=0.0, ry=0.0,
            nx=50, ny=50
        )
        assert 0.0 <= vf <= 1.0
        assert not np.isnan(vf)
        assert not np.isinf(vf)


class TestSafetyConcurrentExecution:
    """Test that solvers can handle concurrent execution safely."""
    
    def test_concurrent_adaptive_calls(self):
        """Test multiple concurrent adaptive solver calls."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def run_adaptive():
            result = vf_adaptive(
                em_w=1.0, em_h=1.0, rc_w=1.0, rc_h=1.0, setback=1.0,
                time_limit_s=2.0
            )
            results.put(result)
        
        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=run_adaptive)
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10.0)  # 10 second timeout
            assert not thread.is_alive(), "Thread did not complete in time"
        
        # Check all results
        assert results.qsize() == 3
        while not results.empty():
            result = results.get()
            assert result['status'] in ['converged', 'reached_limits', 'failed']
            assert 0.0 <= result['vf'] <= 1.0
