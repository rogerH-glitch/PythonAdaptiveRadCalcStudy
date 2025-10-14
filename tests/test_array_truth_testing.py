"""Test that array truth-testing is handled safely."""
import numpy as np
from src.util.plot_payload import has_field


def test_has_field_with_arrays():
    """Test that has_field works correctly with numpy arrays."""
    # Test with valid arrays
    Y = np.array([[1, 2], [3, 4]])
    Z = np.array([[1, 2], [3, 4]])
    F = np.array([[0.1, 0.2], [0.3, 0.4]])
    
    result = {"Y": Y, "Z": Z, "F": F}
    assert has_field(result) is True, "Should return True for valid arrays"
    
    # Test with mismatched shapes
    Y_bad = np.array([[1, 2], [3, 4]])
    Z_bad = np.array([[1, 2, 3], [4, 5, 6]])  # Different shape
    F_bad = np.array([[0.1, 0.2], [0.3, 0.4]])
    
    result_bad = {"Y": Y_bad, "Z": Z_bad, "F": F_bad}
    assert has_field(result_bad) is False, "Should return False for mismatched shapes"
    
    # Test with None values
    result_none = {"Y": None, "Z": Z, "F": F}
    assert has_field(result_none) is False, "Should return False for None values"
    
    # Test with missing keys
    result_missing = {"Y": Y, "Z": Z}  # Missing F
    assert has_field(result_missing) is False, "Should return False for missing keys"
    
    # Test with empty arrays
    Y_empty = np.array([])
    Z_empty = np.array([])
    F_empty = np.array([])
    
    result_empty = {"Y": Y_empty, "Z": Z_empty, "F": F_empty}
    assert has_field(result_empty) is False, "Should return False for empty arrays"


def test_has_field_avoids_truth_testing():
    """Test that has_field doesn't trigger numpy truth-value errors."""
    # Create arrays that would cause truth-value errors if used in boolean context
    Y = np.array([True, False])  # This would cause ValueError if used in boolean context
    Z = np.array([True, False])
    F = np.array([0.1, 0.2])
    
    result = {"Y": Y, "Z": Z, "F": F}
    
    # This should not raise a ValueError
    try:
        result_bool = has_field(result)
        assert result_bool is True, "Should handle boolean arrays correctly"
    except ValueError as e:
        if "truth value" in str(e).lower():
            pytest.fail(f"has_field triggered numpy truth-value error: {e}")
        else:
            raise  # Re-raise if it's a different ValueError
