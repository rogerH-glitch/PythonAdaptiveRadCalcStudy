"""Test heatmap fallback functionality."""
import numpy as np
from types import SimpleNamespace
from src.viz.field_sampler import sample_receiver_field
from src.util.plot_payload import has_field


def test_sample_receiver_field_without_data():
    """Test that sample_receiver_field populates Y,Z,F when not present."""
    # Create a minimal fake result without Y/Z/F
    result = {
        "Wr": 5.0,
        "Hr": 2.0,
        "dy": 0.6,
        "dz": 0.4
    }
    
    # Create mock args
    args = SimpleNamespace(
        receiver=(5.0, 2.0),
        emitter=(5.1, 2.1)
    )
    
    # Call sample_receiver_field
    updated_result = sample_receiver_field(args, result)
    
    # Assert Y,Z,F shapes > 0
    assert "Y" in updated_result
    assert "Z" in updated_result
    assert "F" in updated_result
    
    Y = updated_result["Y"]
    Z = updated_result["Z"]
    F = updated_result["F"]
    
    assert Y.shape[0] > 0, f"Y shape should be > 0, got {Y.shape}"
    assert Y.shape[1] > 0, f"Y shape should be > 0, got {Y.shape}"
    assert Z.shape[0] > 0, f"Z shape should be > 0, got {Z.shape}"
    assert Z.shape[1] > 0, f"Z shape should be > 0, got {Z.shape}"
    assert F.shape[0] > 0, f"F shape should be > 0, got {F.shape}"
    assert F.shape[1] > 0, f"F shape should be > 0, got {F.shape}"
    
    # Assert field_is_sampled is True
    assert updated_result.get("field_is_sampled") is True, "field_is_sampled should be True"
    
    # Assert has_field returns True
    assert has_field(updated_result), "has_field should return True"


def test_sample_receiver_field_with_existing_data():
    """Test that sample_receiver_field does not overwrite existing Y/Z/F."""
    # Create result with existing field data
    result = {
        "Wr": 5.0,
        "Hr": 2.0,
        "dy": 0.6,
        "dz": 0.4,
        "Y": np.array([[1, 2], [3, 4]]),
        "Z": np.array([[1, 2], [3, 4]]),
        "F": np.array([[0.1, 0.2], [0.3, 0.4]])
    }
    
    # Create mock args
    args = SimpleNamespace(
        receiver=(5.0, 2.0),
        emitter=(5.1, 2.1)
    )
    
    # Call sample_receiver_field
    updated_result = sample_receiver_field(args, result)
    
    # Assert original data is preserved
    assert np.array_equal(updated_result["Y"], result["Y"]), "Y should not be overwritten"
    assert np.array_equal(updated_result["Z"], result["Z"]), "Z should not be overwritten"
    assert np.array_equal(updated_result["F"], result["F"]), "F should not be overwritten"
    
    # Assert field_is_sampled is not set
    assert "field_is_sampled" not in updated_result, "field_is_sampled should not be set when data exists"


def test_attach_field_from_tap_returns_result_with_keys():
    """Test that attach_field_from_tap returns result with Y,Z,F keys if captured."""
    from src.util.plot_payload import attach_field_from_tap
    from src.util.grid_tap import capture as _capture
    
    # Create a test result
    result = {"method": "adaptive"}
    
    # Create some test data
    Y = np.array([[1, 2], [3, 4]])
    Z = np.array([[1, 2], [3, 4]])
    F = np.array([[0.1, 0.2], [0.3, 0.4]])
    
    # Capture the data in the grid tap
    _capture(Y, Z, F)
    
    # Call attach_field_from_tap
    updated_result = attach_field_from_tap(result)
    
    # Assert Y,Z,F are present
    assert "Y" in updated_result
    assert "Z" in updated_result
    assert "F" in updated_result
    
    # Assert the data matches
    assert np.array_equal(updated_result["Y"], Y)
    assert np.array_equal(updated_result["Z"], Z)
    assert np.array_equal(updated_result["F"], F)
