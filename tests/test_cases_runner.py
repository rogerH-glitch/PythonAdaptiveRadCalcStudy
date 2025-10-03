import os
import tempfile
import yaml
import csv
from pathlib import Path
import pytest
from src.cli import run_cases

def test_cases_runner_basic():
    """Test that cases runner creates summary CSV with correct fields."""
    
    # Create a tiny YAML with 2 simple cases
    test_cases = {
        'version': 1,
        'cases': [
            {
                'id': 'test_case_1',
                'enabled': True,
                'description': 'Simple test case 1',
                'geometry': {
                    'emitter': {'w': 2.0, 'h': 1.0},
                    'receiver': {'w': 2.0, 'h': 1.0},
                    'setback': 1.0,
                    'angle': 0.0
                },
                'expected': {'F12': 0.5},
                'method_overrides': {
                    'rel_tol': 0.01,
                    'max_depth': 5
                }
            },
            {
                'id': 'test_case_2',
                'enabled': True,
                'description': 'Simple test case 2',
                'geometry': {
                    'emitter': {'w': 3.0, 'h': 2.0},
                    'receiver': {'w': 3.0, 'h': 2.0},
                    'setback': 2.0,
                    'angle': 0.0
                },
                'expected': {'F12': 0.3},
                'method_overrides': {
                    'rel_tol': 0.02,
                    'max_depth': 6
                }
            }
        ]
    }
    
    # Create temporary directory and YAML file
    with tempfile.TemporaryDirectory() as temp_dir:
        yaml_path = os.path.join(temp_dir, 'test_cases.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(test_cases, f)
        
        # Run cases
        exit_code = run_cases(yaml_path, temp_dir, plot=False)
        
        # Check exit code
        assert exit_code == 0
        
        # Check that summary CSV was created
        summary_csv = os.path.join(temp_dir, 'cases_summary.csv')
        assert os.path.exists(summary_csv)
        
        # Read and validate CSV content
        with open(summary_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Check header
        expected_headers = [
            "id", "method", "vf", "ci95", "expected", "rel_err", "status", 
            "iterations", "achieved_tol", "notes"
        ]
        assert rows[0] == expected_headers
        
        # Check that we have 2 data rows (plus header)
        assert len(rows) == 3
        
        # Check data rows
        for i, row in enumerate(rows[1:], 1):
            case_id = row[0]
            method = row[1]
            vf = row[2]
            status = row[6]
            
            # Basic field validation
            assert case_id in ['test_case_1', 'test_case_2']
            assert method in ['analytical', 'adaptive', 'fixedgrid', 'montecarlo']
            assert vf != ""  # Should have a view factor value
            assert status in ['converged', 'reached_limits', 'failed']
            
            # Check that vf is a valid number
            try:
                vf_float = float(vf)
                assert 0.0 <= vf_float <= 1.0  # View factor should be in [0,1]
            except ValueError:
                pytest.fail(f"Invalid view factor value: {vf}")

def test_cases_runner_with_plot():
    """Test that cases runner generates plots when requested."""
    
    # Create a simple test case
    test_cases = {
        'version': 1,
        'cases': [
            {
                'id': 'plot_test_case',
                'enabled': True,
                'description': 'Test case for plotting',
                'geometry': {
                    'emitter': {'w': 2.0, 'h': 1.0},
                    'receiver': {'w': 2.0, 'h': 1.0},
                    'setback': 1.0,
                    'angle': 0.0
                },
                'expected': {'F12': 0.5}
            }
        ]
    }
    
    # Create temporary directory and YAML file
    with tempfile.TemporaryDirectory() as temp_dir:
        yaml_path = os.path.join(temp_dir, 'test_cases.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(test_cases, f)
        
        # Run cases with plotting
        exit_code = run_cases(yaml_path, temp_dir, plot=True)
        
        # Check exit code
        assert exit_code == 0
        
        # Check that plots directory was created
        plots_dir = os.path.join(temp_dir, 'plots')
        assert os.path.exists(plots_dir)
        
        # Check that summary CSV was created
        summary_csv = os.path.join(temp_dir, 'cases_summary.csv')
        assert os.path.exists(summary_csv)
        
        # Read CSV and check that plot filename is mentioned in notes
        with open(summary_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Check that notes column contains plot information
        notes = rows[1][9]  # notes column
        assert 'plot=' in notes

def test_cases_runner_disabled_case():
    """Test that disabled cases are handled correctly."""
    
    # Create test cases with one disabled
    test_cases = {
        'version': 1,
        'cases': [
            {
                'id': 'enabled_case',
                'enabled': True,
                'description': 'Enabled test case',
                'geometry': {
                    'emitter': {'w': 2.0, 'h': 1.0},
                    'receiver': {'w': 2.0, 'h': 1.0},
                    'setback': 1.0,
                    'angle': 0.0
                }
            },
            {
                'id': 'disabled_case',
                'enabled': False,
                'description': 'Disabled test case',
                'geometry': {
                    'emitter': {'w': 2.0, 'h': 1.0},
                    'receiver': {'w': 2.0, 'h': 1.0},
                    'setback': 1.0,
                    'angle': 0.0
                }
            }
        ]
    }
    
    # Create temporary directory and YAML file
    with tempfile.TemporaryDirectory() as temp_dir:
        yaml_path = os.path.join(temp_dir, 'test_cases.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(test_cases, f)
        
        # Run cases
        exit_code = run_cases(yaml_path, temp_dir, plot=False)
        
        # Check exit code
        assert exit_code == 0
        
        # Read CSV and check both cases are present
        summary_csv = os.path.join(temp_dir, 'cases_summary.csv')
        with open(summary_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Should have header + 2 data rows
        assert len(rows) == 3
        
        # Check enabled case
        enabled_row = next(row for row in rows[1:] if row[0] == 'enabled_case')
        assert enabled_row[6] in ['converged', 'reached_limits', 'failed']  # status
        
        # Check disabled case
        disabled_row = next(row for row in rows[1:] if row[0] == 'disabled_case')
        assert disabled_row[6] == 'skipped'  # status
        assert 'disabled' in disabled_row[9]  # notes