"""Test that has_field safely handles numpy arrays without truth-testing errors."""
import numpy as np
from src.util.plot_payload import has_field


def test_has_field_true_for_numpy_arrays_same_shape():
    """Test that has_field returns True for valid numpy arrays with same shape."""
    Y = np.zeros((5, 5))
    Z = np.ones((5, 5))
    F = np.full((5, 5), 2.0)
    result = {"Y": Y, "Z": Z, "F": F}
    assert has_field(result) is True


def test_has_field_false_for_missing_or_mismatched():
    """Test that has_field returns False for missing or mismatched arrays."""
    Y = np.zeros((5, 5))
    Z = np.ones((4, 5))  # Different shape
    F = np.zeros((5, 5))
    
    # Test mismatched shapes
    assert has_field({"Y": Y, "Z": Z, "F": F}) is False
    
    # Test missing keys
    assert has_field({"Y": Y}) is False
    assert has_field({"Y": Y, "Z": Z}) is False  # Missing F
    
    # Test None values
    assert has_field({"Y": None, "Z": Z, "F": F}) is False
    assert has_field({"Y": Y, "Z": None, "F": F}) is False
    assert has_field({"Y": Y, "Z": Z, "F": None}) is False


def test_has_field_handles_boolean_arrays_safely():
    """Test that has_field doesn't trigger numpy truth-value errors with boolean arrays."""
    # Create boolean arrays that would cause ValueError if used in boolean context
    Y = np.array([True, False, True])
    Z = np.array([False, True, False])
    F = np.array([0.1, 0.2, 0.3])
    
    result = {"Y": Y, "Z": Z, "F": F}
    
    # This should not raise a ValueError about truth values
    try:
        result_bool = has_field(result)
        assert result_bool is True, "Should handle boolean arrays correctly"
    except ValueError as e:
        if "truth value" in str(e).lower():
            raise AssertionError(f"has_field triggered numpy truth-value error: {e}")
        else:
            raise  # Re-raise if it's a different ValueError


def test_has_field_handles_empty_arrays():
    """Test that has_field returns False for empty arrays."""
    Y = np.array([])
    Z = np.array([])
    F = np.array([])
    
    result = {"Y": Y, "Z": Z, "F": F}
    assert has_field(result) is False


def test_has_field_handles_scalar_values():
    """Test that has_field returns False for scalar values (not arrays)."""
    result = {"Y": 1.0, "Z": 2.0, "F": 3.0}
    assert has_field(result) is False
