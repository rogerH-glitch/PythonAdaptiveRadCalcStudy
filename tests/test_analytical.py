"""
Tests for the analytical view factor baseline module.

This module tests the analytical baseline approximations for radiation
view factors between rectangular surfaces.
"""

import pytest
import time
from src.analytical import (
    local_peak_vf_analytic_approx,
    validate_geometry,
    get_analytical_info
)


class TestLocalPeakVFAnalyticApprox:
    """Test the local peak view factor analytical approximation."""
    
    def test_valid_range(self):
        """Test that view factor is in valid range [0, 1]."""
        # Test various geometries
        test_cases = [
            (5.1, 2.1, 5.1, 2.1, 1.0),    # Equal squares, moderate setback
            (1.0, 1.0, 1.0, 1.0, 0.1),    # Close surfaces
            (10.0, 5.0, 2.0, 1.0, 5.0),   # Different sizes, large setback
            (0.5, 0.5, 20.0, 10.0, 2.0),  # Small emitter, large receiver
        ]
        
        for em_w, em_h, rc_w, rc_h, setback in test_cases:
            vf = local_peak_vf_analytic_approx(em_w, em_h, rc_w, rc_h, setback)
            assert 0.0 <= vf <= 1.0, f"VF {vf} out of range for geometry ({em_w}, {em_h}, {rc_w}, {rc_h}, {setback})"
    
    def test_monotonic_setback_decrease(self):
        """Test that VF decreases monotonically as setback increases."""
        em_w, em_h = 5.0, 2.0
        rc_w, rc_h = 5.0, 2.0
        
        setbacks = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        vf_values = []
        
        for setback in setbacks:
            vf = local_peak_vf_analytic_approx(em_w, em_h, rc_w, rc_h, setback)
            vf_values.append(vf)
        
        # Check monotonic decrease
        for i in range(1, len(vf_values)):
            assert vf_values[i] < vf_values[i-1], (
                f"VF not decreasing: setback {setbacks[i-1]} -> {setbacks[i]}, "
                f"VF {vf_values[i-1]} -> {vf_values[i]}"
            )
    
    def test_symmetry_equal_surfaces(self):
        """Test symmetry: swapping dimensions of equal surfaces doesn't change VF."""
        setback = 1.5
        
        # Test cases with equal emitter and receiver
        test_cases = [
            (3.0, 2.0),  # Rectangle
            (4.0, 4.0),  # Square
            (10.0, 1.0), # Long thin rectangle
        ]
        
        for w, h in test_cases:
            # Original orientation
            vf1 = local_peak_vf_analytic_approx(w, h, w, h, setback)
            
            # Swapped dimensions
            vf2 = local_peak_vf_analytic_approx(h, w, h, w, setback)
            
            # Should be equal (within numerical tolerance)
            assert abs(vf1 - vf2) < 1e-10, (
                f"Symmetry broken: ({w}x{h}) -> VF={vf1}, ({h}x{w}) -> VF={vf2}"
            )
    
    def test_runtime_performance(self):
        """Test that calculation completes within reasonable time (~5s)."""
        # Test with default geometry
        em_w, em_h = 5.1, 2.1
        rc_w, rc_h = 5.1, 2.1
        setback = 1.0
        
        start_time = time.time()
        vf = local_peak_vf_analytic_approx(em_w, em_h, rc_w, rc_h, setback)
        calc_time = time.time() - start_time
        
        assert calc_time < 5.0, f"Calculation took {calc_time:.3f}s, expected < 5s"
        assert isinstance(vf, float), "Result should be a float"
    
    def test_extreme_geometries(self):
        """Test edge cases and extreme geometries."""
        # Very small surfaces
        vf1 = local_peak_vf_analytic_approx(0.01, 0.01, 0.01, 0.01, 0.01)
        assert 0.0 <= vf1 <= 1.0
        
        # Very large surfaces
        vf2 = local_peak_vf_analytic_approx(100.0, 50.0, 100.0, 50.0, 10.0)
        assert 0.0 <= vf2 <= 1.0
        
        # High aspect ratio
        vf3 = local_peak_vf_analytic_approx(0.1, 10.0, 0.1, 10.0, 1.0)
        assert 0.0 <= vf3 <= 1.0
        
        # Very close surfaces (should be higher than far surfaces)
        vf4 = local_peak_vf_analytic_approx(1.0, 1.0, 1.0, 1.0, 0.01)
        assert vf4 > 0.5, f"Close surfaces should have reasonably high VF, got {vf4}"
        
        # Very far surfaces (should approach 0.0)
        vf5 = local_peak_vf_analytic_approx(1.0, 1.0, 1.0, 1.0, 100.0)
        assert vf5 < 0.01, f"Far surfaces should have low VF, got {vf5}"
    
    def test_input_validation(self):
        """Test input validation and error handling."""
        # Test negative dimensions
        with pytest.raises(ValueError, match="All dimensions and setback must be positive"):
            local_peak_vf_analytic_approx(-1.0, 2.0, 3.0, 4.0, 1.0)
        
        with pytest.raises(ValueError, match="All dimensions and setback must be positive"):
            local_peak_vf_analytic_approx(1.0, -2.0, 3.0, 4.0, 1.0)
        
        with pytest.raises(ValueError, match="All dimensions and setback must be positive"):
            local_peak_vf_analytic_approx(1.0, 2.0, -3.0, 4.0, 1.0)
        
        with pytest.raises(ValueError, match="All dimensions and setback must be positive"):
            local_peak_vf_analytic_approx(1.0, 2.0, 3.0, -4.0, 1.0)
        
        with pytest.raises(ValueError, match="All dimensions and setback must be positive"):
            local_peak_vf_analytic_approx(1.0, 2.0, 3.0, 4.0, -1.0)
        
        # Test zero dimensions
        with pytest.raises(ValueError, match="All dimensions and setback must be positive"):
            local_peak_vf_analytic_approx(0.0, 2.0, 3.0, 4.0, 1.0)
        
        with pytest.raises(ValueError, match="All dimensions and setback must be positive"):
            local_peak_vf_analytic_approx(1.0, 2.0, 3.0, 4.0, 0.0)
    
    def test_known_reference_values(self):
        """Test against some expected reference values for sanity check."""
        # These are approximate values for validation - not exact analytical solutions
        
        # Case 1: Unit squares, unit setback
        vf1 = local_peak_vf_analytic_approx(1.0, 1.0, 1.0, 1.0, 1.0)
        assert 0.15 < vf1 < 0.25, f"Unit squares at unit setback: expected ~0.2, got {vf1}"
        
        # Case 2: Large emitter, small receiver, close
        vf2 = local_peak_vf_analytic_approx(10.0, 10.0, 1.0, 1.0, 0.1)
        assert vf2 > 0.8, f"Large close emitter should have high VF, got {vf2}"
        
        # Case 3: Equal surfaces, very far
        vf3 = local_peak_vf_analytic_approx(2.0, 2.0, 2.0, 2.0, 20.0)
        assert vf3 < 0.05, f"Far surfaces should have low VF, got {vf3}"


class TestValidateGeometry:
    """Test the geometry validation function."""
    
    def test_valid_geometry(self):
        """Test validation of valid geometries."""
        valid_cases = [
            (1.0, 1.0, 1.0, 1.0, 1.0),
            (5.1, 2.1, 3.0, 1.5, 0.5),
            (0.001, 0.001, 100.0, 50.0, 10.0),
        ]
        
        for em_w, em_h, rc_w, rc_h, setback in valid_cases:
            is_valid, msg = validate_geometry(em_w, em_h, rc_w, rc_h, setback)
            assert is_valid, f"Valid geometry rejected: {msg}"
            assert msg == "", f"Valid geometry should have empty error message, got: {msg}"
    
    def test_invalid_geometry(self):
        """Test validation of invalid geometries."""
        invalid_cases = [
            (-1.0, 1.0, 1.0, 1.0, 1.0, "Emitter width must be positive"),
            (1.0, -1.0, 1.0, 1.0, 1.0, "Emitter height must be positive"),
            (1.0, 1.0, -1.0, 1.0, 1.0, "Receiver width must be positive"),
            (1.0, 1.0, 1.0, -1.0, 1.0, "Receiver height must be positive"),
            (1.0, 1.0, 1.0, 1.0, -1.0, "Setback must be positive"),
            (0.0, 1.0, 1.0, 1.0, 1.0, "Emitter width must be positive"),
            (1.0, 1.0, 1.0, 1.0, 0.0, "Setback must be positive"),
        ]
        
        for *params, expected_msg_part in invalid_cases:
            is_valid, msg = validate_geometry(*params)
            assert not is_valid, f"Invalid geometry accepted: {params}"
            assert expected_msg_part in msg, f"Expected '{expected_msg_part}' in error message, got: {msg}"


class TestGetAnalyticalInfo:
    """Test the analytical method information function."""
    
    def test_info_string(self):
        """Test that info string is returned and contains expected content."""
        info = get_analytical_info()
        
        assert isinstance(info, str), "Info should be a string"
        assert len(info) > 0, "Info should not be empty"
        assert "baseline" in info.lower(), "Info should mention baseline"
        assert "200" in info, "Info should mention grid size"
        assert "walton" in info.lower(), "Info should mention Walton"


class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_very_small_numbers(self):
        """Test with very small dimensions."""
        vf = local_peak_vf_analytic_approx(1e-6, 1e-6, 1e-6, 1e-6, 1e-6)
        assert 0.0 <= vf <= 1.0
        assert not (vf != vf), "Result should not be NaN"  # NaN check
    
    def test_very_large_numbers(self):
        """Test with very large dimensions."""
        vf = local_peak_vf_analytic_approx(1e6, 1e6, 1e6, 1e6, 1e3)
        assert 0.0 <= vf <= 1.0
        assert not (vf != vf), "Result should not be NaN"  # NaN check
    
    def test_mixed_scales(self):
        """Test with mixed scales (small and large dimensions)."""
        # Small emitter, large receiver
        vf1 = local_peak_vf_analytic_approx(1e-3, 1e-3, 1e3, 1e3, 1.0)
        assert 0.0 <= vf1 <= 1.0
        
        # Large emitter, small receiver  
        vf2 = local_peak_vf_analytic_approx(1e3, 1e3, 1e-3, 1e-3, 1.0)
        assert 0.0 <= vf2 <= 1.0
