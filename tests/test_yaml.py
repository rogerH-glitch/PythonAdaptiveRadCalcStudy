"""
Tests for YAML case loading functionality.

This module tests the YAML case loading, validation, and conversion
functions for the view factor validation tool.
"""

import pytest
from src.io_yaml import load_cases, validate_case_schema, coerce_case_to_cli_kwargs, YamlError


def test_load_and_validate_cases():
    """Test loading and validating cases from the validation YAML file."""
    cases = load_cases("docs/validation_cases.yaml")
    assert isinstance(cases, list) and len(cases) >= 2
    
    # Validate all cases have proper schema
    for c in cases:
        validate_case_schema(c)


def test_coerce_kwargs_shape():
    """Test that case conversion produces correct kwargs structure."""
    cases = load_cases("docs/validation_cases.yaml")
    c_enabled = next(c for c in cases if c.get("enabled"))
    kw = coerce_case_to_cli_kwargs(c_enabled)
    
    # Check required keys are present
    assert {"method","emitter","receiver","setback","angle","id"}.issubset(kw.keys())
    
    # Check types are correct
    assert isinstance(kw["emitter"], tuple) and len(kw["emitter"]) == 2
    assert isinstance(kw["receiver"], tuple) and len(kw["receiver"]) == 2
    assert isinstance(kw["setback"], float)
    assert isinstance(kw["angle"], float)
    assert isinstance(kw["id"], str)
    assert isinstance(kw["method"], str)


def test_case_schema_validation():
    """Test case schema validation with various inputs."""
    # Valid case
    valid_case = {
        "id": "test_case",
        "enabled": True,
        "geometry": {
            "emitter": {"w": 5.1, "h": 2.1},
            "receiver": {"w": 5.1, "h": 2.1},
            "setback": 1.0,
            "angle": 0
        }
    }
    
    # Should not raise exception
    validate_case_schema(valid_case)
    
    # Test missing id
    invalid_case = valid_case.copy()
    del invalid_case["id"]
    with pytest.raises(YamlError, match="Case missing 'id'"):
        validate_case_schema(invalid_case)
    
    # Test missing enabled
    invalid_case = valid_case.copy()
    del invalid_case["enabled"]
    with pytest.raises(YamlError, match="missing 'enabled'"):
        validate_case_schema(invalid_case)
    
    # Test missing geometry
    invalid_case = valid_case.copy()
    del invalid_case["geometry"]
    with pytest.raises(YamlError, match="'geometry' dict required"):
        validate_case_schema(invalid_case)
    
    # Test missing geometry keys
    invalid_case = valid_case.copy()
    del invalid_case["geometry"]["setback"]
    with pytest.raises(YamlError, match="geometry missing key 'setback'"):
        validate_case_schema(invalid_case)
    
    # Test missing surface dimensions - create a fresh valid case first
    valid_case2 = {
        "id": "test_case2",
        "enabled": True,
        "geometry": {
            "emitter": {"w": 5.1, "h": 2.1},
            "receiver": {"w": 5.1, "h": 2.1},
            "setback": 1.0,
            "angle": 0
        }
    }
    del valid_case2["geometry"]["emitter"]["w"]
    with pytest.raises(YamlError, match="geometry.emitter.w missing"):
        validate_case_schema(valid_case2)


def test_coerce_case_conversion():
    """Test detailed case conversion functionality."""
    test_case = {
        "id": "test_conversion",
        "enabled": True,
        "method": "fixedgrid",
        "geometry": {
            "emitter": {"w": 10.5, "h": 3.2},
            "receiver": {"w": 8.0, "h": 2.5},
            "setback": 2.5,
            "angle": 15.0
        },
        "expected": {
            "F12": 0.456789,
            "tolerance": {"type": "relative", "value": 0.01}
        },
        "method_overrides": {
            "grid_nx": 200,
            "grid_ny": 150
        }
    }
    
    kw = coerce_case_to_cli_kwargs(test_case)
    
    # Check all values are converted correctly
    assert kw["id"] == "test_conversion"
    assert kw["method"] == "fixedgrid"
    assert kw["emitter"] == (10.5, 3.2)
    assert kw["receiver"] == (8.0, 2.5)
    assert kw["setback"] == 2.5
    assert kw["angle"] == 15.0
    assert kw["expected"] == 0.456789
    assert kw["overrides"] == {"grid_nx": 200, "grid_ny": 150}


def test_load_cases_error_handling():
    """Test error handling for YAML loading."""
    # Test non-existent file
    with pytest.raises(YamlError, match="YAML not found"):
        load_cases("nonexistent_file.yaml")
    
    # Test invalid YAML structure (would need a temporary file)
    # For now, we'll test the validation logic with the real file


def test_enabled_disabled_cases():
    """Test handling of enabled and disabled cases."""
    cases = load_cases("docs/validation_cases.yaml")
    
    # Should have both enabled and disabled cases
    enabled_cases = [c for c in cases if c.get("enabled", False)]
    disabled_cases = [c for c in cases if not c.get("enabled", False)]
    
    assert len(enabled_cases) > 0, "Should have at least one enabled case"
    assert len(disabled_cases) > 0, "Should have at least one disabled case"
    
    # All enabled cases should validate and convert properly
    for case in enabled_cases:
        validate_case_schema(case)
        kw = coerce_case_to_cli_kwargs(case)
        assert kw["id"] == case["id"]


def test_expected_values_handling():
    """Test handling of expected values and tolerances."""
    cases = load_cases("docs/validation_cases.yaml")
    
    # Find cases with expected values
    cases_with_expected = [c for c in cases if c.get("expected")]
    assert len(cases_with_expected) > 0, "Should have cases with expected values"
    
    for case in cases_with_expected:
        kw = coerce_case_to_cli_kwargs(case)
        
        # Should have expected F12 value
        assert kw["expected"] is not None
        assert isinstance(kw["expected"], (int, float))
        
        # Should have tolerance info
        assert "expected_tol" in kw
        
        # Expected value should be reasonable (between 0 and 1)
        assert 0 <= kw["expected"] <= 1


def test_method_overrides():
    """Test handling of method-specific overrides."""
    cases = load_cases("docs/validation_cases.yaml")
    
    # Find case with method overrides
    case_with_overrides = next((c for c in cases if "method_overrides" in c), None)
    
    if case_with_overrides:
        kw = coerce_case_to_cli_kwargs(case_with_overrides)
        assert "overrides" in kw
        assert isinstance(kw["overrides"], dict)
    
    # Test case without overrides
    case_without_overrides = next((c for c in cases if "method_overrides" not in c), None)
    if case_without_overrides:
        kw = coerce_case_to_cli_kwargs(case_without_overrides)
        assert kw["overrides"] == {}


if __name__ == "__main__":
    pytest.main([__file__])
