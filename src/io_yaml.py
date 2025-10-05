"""
YAML input/output utilities for configuration and results.

This module handles loading configuration files and saving results
in various formats for the view factor validation tool.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import csv
import json
import os
from datetime import datetime

from .geometry import Rectangle


class YamlError(Exception):
    """Exception raised for YAML loading and validation errors."""
    pass


REQUIRED_GEOM_KEYS = {"emitter", "receiver", "setback", "angle"}


def load_cases(path: str) -> List[Dict[str, Any]]:
    """
    Load a YAML file of validation cases.
    Returns a list of cases (dicts). Raises YamlError on problems.
    """
    if not os.path.isfile(path):
        raise YamlError(f"YAML not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        raise YamlError(f"Failed to parse YAML: {e}") from e
    if not isinstance(data, dict) or "cases" not in data or not isinstance(data["cases"], list):
        raise YamlError("Invalid YAML structure: top-level 'cases' list required")
    return data["cases"]


def validate_case_schema(case: Dict[str, Any]) -> None:
    """
    Minimal schema check for a case.
    Required: id (str), enabled (bool), geometry/emitter/receiver/setback/angle
    """
    if "id" not in case or not isinstance(case["id"], str):
        raise YamlError("Case missing 'id' (str)")
    if "enabled" not in case or not isinstance(case["enabled"], bool):
        raise YamlError(f"Case {case.get('id')} missing 'enabled' (bool)")
    if "geometry" not in case or not isinstance(case["geometry"], dict):
        raise YamlError(f"Case {case['id']}: 'geometry' dict required")
    geom = case["geometry"]
    for k in REQUIRED_GEOM_KEYS:
        if k not in geom:
            raise YamlError(f"Case {case['id']}: geometry missing key '{k}'")
    for surf in ("emitter", "receiver"):
        if surf not in geom or not isinstance(geom[surf], dict):
            raise YamlError(f"Case {case['id']}: geometry.{surf} dict required")
        for dim in ("w", "h"):
            if dim not in geom[surf]:
                raise YamlError(f"Case {case['id']}: geometry.{surf}.{dim} missing")


def coerce_case_to_cli_kwargs(case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a YAML case dict into kwargs compatible with the CLI execution layer.
    """
    method = case.get("method", "adaptive")
    geom = case["geometry"]
    k = {
        "method": method,
        "emitter": (float(geom["emitter"]["w"]), float(geom["emitter"]["h"])),
        "receiver": (float(geom["receiver"]["w"]), float(geom["receiver"]["h"])),
        "setback": float(geom["setback"]),
        "angle": float(geom.get("angle", 0.0)),
        "expected": (case.get("expected", {}) or {}).get("F12"),
        "expected_tol": (case.get("expected", {}).get("tolerance", {}) or {}),
        "id": case["id"],
    }
    # Optional per-method overrides
    if "method_overrides" in case and isinstance(case["method_overrides"], dict):
        k["overrides"] = case["method_overrides"]
    else:
        k["overrides"] = {}
    return k


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML config: {e}") from e


def save_results(results: Dict[str, Any], 
                output_path: Path,
                format: str = 'csv') -> None:
    """Save calculation results to file.
    
    Args:
        results: Dictionary of method name to result data
        output_path: Output file path
        format: Output format ('csv', 'json', or 'yaml')
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine format from file extension if not specified
    if format == 'csv' or output_path.suffix.lower() == '.csv':
        _save_results_csv(results, output_path)
    elif format == 'json' or output_path.suffix.lower() == '.json':
        _save_results_json(results, output_path)
    elif format == 'yaml' or output_path.suffix.lower() in ['.yaml', '.yml']:
        _save_results_yaml(results, output_path)
    else:
        raise ValueError(f"Unsupported output format: {format}")


def _save_results_csv(results: Dict[str, Any], output_path: Path) -> None:
    """Save results in CSV format."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'timestamp',
            'method',
            'view_factor',
            'uncertainty',
            'converged',
            'iterations',
            'computation_time_s'
        ])
        
        # Write results
        timestamp = datetime.now().isoformat()
        for method_name, result in results.items():
            # Handle both object attributes and dictionary keys
            if hasattr(result, 'value'):
                view_factor = result.value
                uncertainty = getattr(result, 'uncertainty', 0.0)
                converged = getattr(result, 'converged', True)
                iterations = getattr(result, 'iterations', 0)
                computation_time = getattr(result, 'computation_time', 0.0)
            elif isinstance(result, dict):
                view_factor = result.get('value', result.get('view_factor', 0.0))
                uncertainty = result.get('uncertainty', 0.0)
                converged = result.get('converged', True)
                iterations = result.get('iterations', 0)
                computation_time = result.get('computation_time', 0.0)
            else:
                # Fallback for simple values
                view_factor = float(result) if isinstance(result, (int, float)) else 0.0
                uncertainty = 0.0
                converged = True
                iterations = 0
                computation_time = 0.0
            
            writer.writerow([
                timestamp,
                method_name,
                f"{view_factor:.8f}",
                f"{uncertainty:.8f}",
                converged,
                iterations,
                f"{computation_time:.4f}"
            ])


def _save_results_json(results: Dict[str, Any], output_path: Path) -> None:
    """Save results in JSON format."""
    data = {
        'timestamp': datetime.now().isoformat(),
        'results': {}
    }
    
    for method_name, result in results.items():
        # Handle both object attributes and dictionary keys
        if hasattr(result, 'value'):
            view_factor = result.value
            uncertainty = getattr(result, 'uncertainty', 0.0)
            converged = getattr(result, 'converged', True)
            iterations = getattr(result, 'iterations', 0)
            computation_time = getattr(result, 'computation_time', 0.0)
            method_used = getattr(result, 'method_used', method_name)
        elif isinstance(result, dict):
            view_factor = result.get('value', result.get('view_factor', 0.0))
            uncertainty = result.get('uncertainty', 0.0)
            converged = result.get('converged', True)
            iterations = result.get('iterations', 0)
            computation_time = result.get('computation_time', 0.0)
            method_used = result.get('method_used', method_name)
        else:
            # Fallback for simple values
            view_factor = float(result) if isinstance(result, (int, float)) else 0.0
            uncertainty = 0.0
            converged = True
            iterations = 0
            computation_time = 0.0
            method_used = method_name
        
        data['results'][method_name] = {
            'view_factor': view_factor,
            'uncertainty': uncertainty,
            'converged': converged,
            'iterations': iterations,
            'computation_time_s': computation_time,
            'method_used': method_used
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def _save_results_yaml(results: Dict[str, Any], output_path: Path) -> None:
    """Save results in YAML format."""
    data = {
        'timestamp': datetime.now().isoformat(),
        'results': {}
    }
    
    for method_name, result in results.items():
        # Handle both object attributes and dictionary keys
        if hasattr(result, 'value'):
            view_factor = result.value
            uncertainty = getattr(result, 'uncertainty', 0.0)
            converged = getattr(result, 'converged', True)
            iterations = getattr(result, 'iterations', 0)
            computation_time = getattr(result, 'computation_time', 0.0)
            method_used = getattr(result, 'method_used', method_name)
        elif isinstance(result, dict):
            view_factor = result.get('value', result.get('view_factor', 0.0))
            uncertainty = result.get('uncertainty', 0.0)
            converged = result.get('converged', True)
            iterations = result.get('iterations', 0)
            computation_time = result.get('computation_time', 0.0)
            method_used = result.get('method_used', method_name)
        else:
            # Fallback for simple values
            view_factor = float(result) if isinstance(result, (int, float)) else 0.0
            uncertainty = 0.0
            converged = True
            iterations = 0
            computation_time = 0.0
            method_used = method_name
        
        data['results'][method_name] = {
            'view_factor': float(view_factor),
            'uncertainty': float(uncertainty),
            'converged': bool(converged),
            'iterations': int(iterations),
            'computation_time_s': float(computation_time),
            'method_used': str(method_used)
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, indent=2)


def create_sample_config() -> Dict[str, Any]:
    """Create a sample configuration dictionary.
    
    Returns:
        Sample configuration for documentation purposes
    """
    return {
        'geometry': {
            'emitter': {
                'width': 5.1,
                'height': 2.1,
                'centre': [2.55, 1.05, 0.0],
                'normal': [0.0, 0.0, 1.0]
            },
            'receiver': {
                'width': 5.1,
                'height': 2.1,
                'centre': [2.55, 1.05, 1.0],
                'normal': [0.0, 0.0, -1.0]
            }
        },
        'methods': {
            'adaptive': {
                'enabled': True,
                'tolerance': 3e-3,
                'max_depth': 12,
                'max_cells': 200000
            },
            'fixed_grid': {
                'enabled': True,
                'nx': 100,
                'ny': 100,
                'quadrature': 'centroid'
            },
            'montecarlo': {
                'enabled': True,
                'samples': 200000,
                'seed': 42
            },
            'analytical': {
                'enabled': True
            }
        },
        'output': {
            'format': 'csv',
            'path': 'results/viewfactor_results.csv',
            'include_plots': False
        }
    }


def save_sample_config(output_path: Path) -> None:
    """Save a sample configuration file.
    
    Args:
        output_path: Path where to save the sample config
    """
    config = create_sample_config()
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def load_validation_cases(cases_path: Path) -> Dict[str, Any]:
    """Load validation test cases from YAML file.
    
    Args:
        cases_path: Path to validation cases YAML file
        
    Returns:
        Dictionary containing validation test cases
    """
    return load_config(cases_path)


def save_results(dataframe_or_dict, outdir: str, filename: str):
    """Save results to file with improved path handling.
    
    Args:
        dataframe_or_dict: Data to save (pandas DataFrame or dict)
        outdir: Output directory path
        filename: Output filename
    """
    # Always ensure directory exists
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    path = str(p / filename)
    
    # Determine file type and save accordingly
    if hasattr(dataframe_or_dict, 'to_csv'):
        # Pandas DataFrame
        dataframe_or_dict.to_csv(path, index=False)
    elif isinstance(dataframe_or_dict, dict):
        # Dictionary - save as JSON
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dataframe_or_dict, f, indent=2)
    else:
        # Fallback - save as text
        with open(path, 'w', encoding='utf-8') as f:
            f.write(str(dataframe_or_dict))