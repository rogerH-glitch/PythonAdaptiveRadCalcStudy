import numpy as np
from src.util.plot_payload import attach_grid_field

def test_attach_grid_field_schema():
    Y, Z = np.zeros((3,4)), np.zeros((3,4))
    F = np.ones((3,4))
    result = {}
    attach_grid_field(result, Y, Z, F)
    assert "grid_data" in result and set(result["grid_data"].keys()) == {"Y","Z","F"}
    assert result["Y"] is Y and result["Z"] is Z and result["F"] is F

def test_attach_grid_field_handles_none():
    result = {}
    attach_grid_field(result, None, None, None)
    assert "grid_data" not in result
    assert "Y" not in result

def test_attach_grid_field_handles_different_shapes():
    Y, Z = np.zeros((3,4)), np.zeros((3,4))
    F = np.ones((2,4))  # Different shape
    result = {}
    # The simplified function doesn't validate shapes, so this should work
    attach_grid_field(result, Y, Z, F)
    assert "grid_data" in result and set(result["grid_data"].keys()) == {"Y","Z","F"}
    assert result["Y"] is Y and result["Z"] is Z and result["F"] is F
