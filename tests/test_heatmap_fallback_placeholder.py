import numpy as np
import pytest
from src.viz.plots import _build_placeholder_field, PLACEHOLDER_NY, PLACEHOLDER_NZ


def test_placeholder_field_dimensions():
    """Test that placeholder field has correct dimensions and shape."""
    Y, Z, F = _build_placeholder_field(2.0, 1.5)
    
    assert Y.shape == (PLACEHOLDER_NZ, PLACEHOLDER_NY)
    assert Z.shape == (PLACEHOLDER_NZ, PLACEHOLDER_NY)
    assert F.shape == (PLACEHOLDER_NZ, PLACEHOLDER_NY)
    
    # Check that Y spans [-W/2, +W/2]
    assert np.isclose(Y.min(), -1.0, atol=1e-10)  # -2.0/2
    assert np.isclose(Y.max(), 1.0, atol=1e-10)   # +2.0/2
    
    # Check that Z spans [-H/2, +H/2]
    assert np.isclose(Z.min(), -0.75, atol=1e-10)  # -1.5/2
    assert np.isclose(Z.max(), 0.75, atol=1e-10)   # +1.5/2


def test_placeholder_field_peak_location():
    """Test that placeholder field has a peak at the center."""
    Y, Z, F = _build_placeholder_field(2.0, 1.5)
    
    # Find the peak location
    peak_idx = np.unravel_index(np.argmax(F), F.shape)
    
    # Should be at the center
    expected_row = PLACEHOLDER_NZ // 2
    expected_col = PLACEHOLDER_NY // 2
    assert peak_idx == (expected_row, expected_col)
    
    # Peak value should be the tiny nonzero value
    assert F[peak_idx] == 1e-12


def test_placeholder_field_zero_background():
    """Test that placeholder field is mostly zeros except for the peak."""
    Y, Z, F = _build_placeholder_field(2.0, 1.5)
    
    # All values should be zero except the peak
    non_zero_mask = F != 0
    non_zero_count = np.sum(non_zero_mask)
    
    # Should have exactly one non-zero value
    assert non_zero_count == 1
    
    # The non-zero value should be at the center
    center_row, center_col = PLACEHOLDER_NZ // 2, PLACEHOLDER_NY // 2
    assert F[center_row, center_col] == 1e-12


def test_placeholder_field_different_sizes():
    """Test placeholder field with different receiver dimensions."""
    # Test with different aspect ratios
    Y1, Z1, F1 = _build_placeholder_field(4.0, 2.0)  # 2:1 aspect ratio
    Y2, Z2, F2 = _build_placeholder_field(1.0, 3.0)  # 1:3 aspect ratio
    
    # Check Y ranges
    assert np.isclose(Y1.min(), -2.0, atol=1e-10)  # -4.0/2
    assert np.isclose(Y1.max(), 2.0, atol=1e-10)   # +4.0/2
    assert np.isclose(Y2.min(), -0.5, atol=1e-10)  # -1.0/2
    assert np.isclose(Y2.max(), 0.5, atol=1e-10)   # +1.0/2
    
    # Check Z ranges
    assert np.isclose(Z1.min(), -1.0, atol=1e-10)  # -2.0/2
    assert np.isclose(Z1.max(), 1.0, atol=1e-10)   # +2.0/2
    assert np.isclose(Z2.min(), -1.5, atol=1e-10)  # -3.0/2
    assert np.isclose(Z2.max(), 1.5, atol=1e-10)   # +3.0/2
    
    # Both should have the same shape
    assert Y1.shape == Y2.shape == (PLACEHOLDER_NZ, PLACEHOLDER_NY)
    assert Z1.shape == Z2.shape == (PLACEHOLDER_NZ, PLACEHOLDER_NY)
    assert F1.shape == F2.shape == (PLACEHOLDER_NZ, PLACEHOLDER_NY)


def test_placeholder_field_meshgrid_properties():
    """Test that Y and Z form proper meshgrids."""
    Y, Z, F = _build_placeholder_field(2.0, 1.5)
    
    # Y should be constant along rows (Z direction) - all rows have same Y pattern
    for i in range(1, PLACEHOLDER_NZ):
        assert np.allclose(Y[0, :], Y[i, :], atol=1e-10), f"Y pattern not same in row {i}"
    
    # Z should be constant along columns (Y direction) - all columns have same Z pattern
    for j in range(1, PLACEHOLDER_NY):
        assert np.allclose(Z[:, 0], Z[:, j], atol=1e-10), f"Z pattern not same in column {j}"
    
    # Y should vary along columns (different Y values across columns)
    assert not np.allclose(Y[0, 0], Y[0, -1], atol=1e-10), "Y should vary along columns"
    
    # Z should vary along rows (different Z values across rows)
    assert not np.allclose(Z[0, 0], Z[-1, 0], atol=1e-10), "Z should vary along rows"
