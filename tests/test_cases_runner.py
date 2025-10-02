"""
Tests for the cases runner functionality.

This module tests the case runner that processes YAML validation cases
and generates summary CSV files.
"""

import os
import pytest
from pathlib import Path
from src.cli import run_cases


def test_run_cases_creates_summary(tmp_path):
    """Test that run_cases creates a summary CSV file."""
    outdir = tmp_path / "results"
    rc = run_cases("docs/validation_cases.yaml", str(outdir))
    
    # Should return success
    assert rc == 0
    
    # Should create output directory and summary file
    summary = outdir / "cases_summary.csv"
    assert summary.exists()
    
    # Check CSV content
    text = summary.read_text(encoding="utf-8")
    assert "id,method,vf,expected,rel_err,status,notes" in text
    
    # Should have header plus at least one data row
    lines = text.strip().split('\n')
    assert len(lines) >= 2  # header + at least one case
    
    # Check that enabled cases are processed
    enabled_lines = [line for line in lines[1:] if not line.endswith(',skipped,disabled')]
    assert len(enabled_lines) > 0, "Should have at least one enabled case processed"


def test_run_cases_handles_enabled_disabled():
    """Test that run_cases properly handles enabled and disabled cases."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        rc = run_cases("docs/validation_cases.yaml", tmpdir)
        assert rc == 0
        
        summary_path = os.path.join(tmpdir, "cases_summary.csv")
        assert os.path.exists(summary_path)
        
        with open(summary_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should have skipped entries for disabled cases
        assert "skipped,disabled" in content
        
        # Should have placeholder entries for enabled cases
        assert "placeholder" in content
        
        # Check specific case handling
        lines = content.strip().split('\n')
        data_lines = lines[1:]  # Skip header
        
        # Should have entries for both enabled and disabled cases
        skipped_count = sum(1 for line in data_lines if "skipped,disabled" in line)
        processed_count = sum(1 for line in data_lines if "placeholder" in line)
        
        assert skipped_count > 0, "Should have at least one disabled case"
        assert processed_count > 0, "Should have at least one enabled case"


def test_run_cases_csv_format():
    """Test the format and content of the generated CSV."""
    import tempfile
    import csv
    
    with tempfile.TemporaryDirectory() as tmpdir:
        rc = run_cases("docs/validation_cases.yaml", tmpdir)
        assert rc == 0
        
        summary_path = os.path.join(tmpdir, "cases_summary.csv")
        
        # Read CSV and check structure
        with open(summary_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            # Check header format
            expected_header = ["id","method","vf","expected","rel_err","status","notes"]
            assert header == expected_header
            
            # Check data rows
            rows = list(reader)
            assert len(rows) > 0, "Should have at least one data row"
            
            for row in rows:
                assert len(row) == len(header), f"Row should have {len(header)} columns: {row}"
                
                # Check that id is not empty
                assert row[0], "ID should not be empty"
                
                # Check method is valid
                assert row[1] in ["adaptive", "fixedgrid", "montecarlo", "analytical", ""], f"Invalid method: {row[1]}"
                
                # Check status is valid
                assert row[5] in ["pending", "skipped", "invalid"], f"Invalid status: {row[5]}"


def test_run_cases_placeholder_values():
    """Test that placeholder values are generated correctly."""
    import tempfile
    import csv
    
    with tempfile.TemporaryDirectory() as tmpdir:
        rc = run_cases("docs/validation_cases.yaml", tmpdir)
        assert rc == 0
        
        summary_path = os.path.join(tmpdir, "cases_summary.csv")
        
        with open(summary_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        
        # Find enabled cases (not skipped)
        enabled_rows = [row for row in rows if row[5] != "skipped"]
        
        for row in enabled_rows:
            if row[5] == "pending":  # Successfully processed case
                # Should have placeholder VF value
                assert row[2] == "0.12345600", f"Expected placeholder VF, got: {row[2]}"
                
                # Should have expected value if present in YAML
                if row[3]:  # Has expected value
                    expected_val = float(row[3])
                    assert 0 <= expected_val <= 1, f"Expected value out of range: {expected_val}"
                
                # Should have relative error calculation if expected value present
                if row[3] and row[4]:  # Has both expected and rel_err
                    rel_err = float(row[4])
                    assert rel_err >= 0, f"Relative error should be non-negative: {rel_err}"


def test_run_cases_error_handling():
    """Test error handling for invalid cases file."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with non-existent file
        rc = run_cases("nonexistent_file.yaml", tmpdir)
        assert rc == 1  # Should return error code
        
        # Output directory should still be created
        assert os.path.exists(tmpdir)


def test_run_cases_output_directory_creation():
    """Test that output directory is created if it doesn't exist."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use a nested directory that doesn't exist
        outdir = os.path.join(tmpdir, "nested", "output")
        
        rc = run_cases("docs/validation_cases.yaml", outdir)
        assert rc == 0
        
        # Directory should be created
        assert os.path.exists(outdir)
        
        # Summary file should be created
        summary_path = os.path.join(outdir, "cases_summary.csv")
        assert os.path.exists(summary_path)


def test_run_cases_specific_case_ids():
    """Test that specific case IDs are processed correctly."""
    import tempfile
    import csv
    
    with tempfile.TemporaryDirectory() as tmpdir:
        rc = run_cases("docs/validation_cases.yaml", tmpdir)
        assert rc == 0
        
        summary_path = os.path.join(tmpdir, "cases_summary.csv")
        
        with open(summary_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        
        # Extract case IDs
        case_ids = [row[0] for row in rows]
        
        # Should have the expected case IDs from the YAML file
        expected_ids = [
            "nist_analytic_obstructed_unit_squares",
            "hc_5p1x2p1_s0p05",
            "hc_5p1x2p1_s1p0",
            "hc_5x2_s3p8",
            "hc_20p02x1p05_s0p81",
            "hc_20p02x1p05_s1p8",
            "hc_21x1_s3p67"
        ]
        
        for expected_id in expected_ids:
            assert expected_id in case_ids, f"Missing case ID: {expected_id}"


def test_run_cases_expected_values():
    """Test that expected values are correctly extracted and processed."""
    import tempfile
    import csv
    
    with tempfile.TemporaryDirectory() as tmpdir:
        rc = run_cases("docs/validation_cases.yaml", tmpdir)
        assert rc == 0
        
        summary_path = os.path.join(tmpdir, "cases_summary.csv")
        
        with open(summary_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        
        # Find rows with expected values
        rows_with_expected = [row for row in rows if row[3]]  # Has expected value
        
        assert len(rows_with_expected) > 0, "Should have cases with expected values"
        
        # Check some specific expected values
        case_expectations = {
            "hc_5p1x2p1_s0p05": 0.998805,
            "hc_5p1x2p1_s1p0": 0.70274,
            "hc_5x2_s3p8": 0.17735,
        }
        
        for row in rows:
            case_id = row[0]
            if case_id in case_expectations and row[3]:
                expected_from_csv = float(row[3])
                expected_from_dict = case_expectations[case_id]
                assert abs(expected_from_csv - expected_from_dict) < 1e-6, \
                    f"Expected value mismatch for {case_id}: {expected_from_csv} vs {expected_from_dict}"


if __name__ == "__main__":
    pytest.main([__file__])
