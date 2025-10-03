"""
Smoke tests for the radiation view factor validation tool.

These tests verify that all modules can be imported without errors
and basic functionality works as expected.
"""

import pytest
import numpy as np


def test_import_all_modules():
    """Test that all modules can be imported without error."""
    # Core modules
    import src
    import src.geometry
    import src.cli
    import src.io_yaml
    import src.analytical
    import src.fixed_grid
    import src.adaptive
    import src.montecarlo
    
    # Verify main functions are available (only analytical is implemented so far)
    assert hasattr(src, 'local_peak_vf_analytic_approx')
    assert hasattr(src, 'validate_geometry')
    assert hasattr(src, 'get_analytical_info')


def test_rectangle_creation():
    """Test basic rectangle creation and properties."""
    from src.geometry import Rectangle
    
    # Create a simple rectangle
    rect = Rectangle(
        origin=np.array([0.0, 0.0, 0.0]),
        u_vector=np.array([5.0, 0.0, 0.0]),
        v_vector=np.array([0.0, 2.0, 0.0])
    )
    
    # Check basic properties
    assert rect.area == 10.0
    assert rect.width == 5.0
    assert rect.height == 2.0
    assert np.allclose(rect.centroid, [2.5, 1.0, 0.0])
    assert np.allclose(rect.normal, [0.0, 0.0, 1.0])


def test_view_factor_result_creation():
    """Test ViewFactorResult creation and validation."""
    from src.geometry import ViewFactorResult
    
    # Valid result
    result = ViewFactorResult(
        value=0.5,
        uncertainty=0.01,
        converged=True,
        iterations=100,
        computation_time=1.5,
        method_used="test"
    )
    
    assert result.value == 0.5
    assert result.uncertainty == 0.01
    assert result.converged is True
    
    # Test string representation
    str_repr = str(result)
    assert "0.500000" in str_repr
    assert "converged" in str_repr


def test_analytical_function_availability():
    """Test that analytical functions are available and callable."""
    from src.analytical import local_peak_vf_analytic_approx, validate_geometry, get_analytical_info
    
    # Test that functions are callable
    assert callable(local_peak_vf_analytic_approx)
    assert callable(validate_geometry)
    assert callable(get_analytical_info)
    
    # Test basic functionality without errors
    info = get_analytical_info()
    assert isinstance(info, str)
    assert len(info) > 0
    
    # Test validation function
    is_valid, msg = validate_geometry(1.0, 1.0, 1.0, 1.0, 1.0)
    assert is_valid
    assert msg == ""


def test_geometry_validation():
    """Test geometry validation functions."""
    from src.geometry import Rectangle, validate_geometry, GeometryError
    
    # Valid rectangles
    emitter = Rectangle(
        origin=np.array([0.0, 0.0, 0.0]),
        u_vector=np.array([5.0, 0.0, 0.0]),
        v_vector=np.array([0.0, 2.0, 0.0])
    )
    receiver = Rectangle(
        origin=np.array([0.0, 0.0, 1.0]),
        u_vector=np.array([5.0, 0.0, 0.0]),
        v_vector=np.array([0.0, 2.0, 0.0])
    )
    
    # Should not raise exception
    validate_geometry(emitter, receiver)
    
    # Invalid rectangle (zero area)
    with pytest.raises((ValueError, GeometryError)):
        invalid_rect = Rectangle(
            origin=np.array([0.0, 0.0, 0.0]),
            u_vector=np.array([0.0, 0.0, 0.0]),  # Zero vector
            v_vector=np.array([0.0, 2.0, 0.0])
        )


def test_cli_parser_creation():
    """Test that CLI parser can be created."""
    from src.cli import create_parser
    
    parser = create_parser()
    assert parser is not None
    
    # Test help doesn't crash
    help_text = parser.format_help()
    assert "view factor" in help_text.lower()


def test_cli_argument_parsing():
    """Test CLI argument parsing with various combinations."""
    from src.cli import create_parser
    import argparse
    
    parser = create_parser()
    
    # Test basic adaptive method
    args = parser.parse_args([
        '--method', 'adaptive',
        '--emitter', '5.1', '2.1',
        '--setback', '1.0'
    ])
    assert args.method == 'adaptive'
    assert args.emitter == [5.1, 2.1]
    assert args.setback == 1.0
    assert args.angle == 0.0  # default
    assert args.receiver is None  # should default to emitter
    
    # Test with receiver dimensions
    args = parser.parse_args([
        '--method', 'fixedgrid',
        '--emitter', '20.02', '1.05',
        '--receiver', '15.0', '0.8',
        '--setback', '0.81',
        '--angle', '15.0'
    ])
    assert args.method == 'fixedgrid'
    assert args.emitter == [20.02, 1.05]
    assert args.receiver == [15.0, 0.8]
    assert args.setback == 0.81
    assert args.angle == 15.0
    
    # Test Monte Carlo with custom parameters
    args = parser.parse_args([
        '--method', 'montecarlo',
        '--emitter', '5.0', '2.0',
        '--setback', '3.8',
        '--samples', '500000',
        '--seed', '123'
    ])
    assert args.method == 'montecarlo'
    assert args.samples == 500000
    assert args.seed == 123
    
    # Test with cases file
    args = parser.parse_args([
        '--method', 'analytical',
        '--cases', 'test_cases.yaml'
    ])
    assert args.method == 'analytical'
    assert str(args.cases) == 'test_cases.yaml'


def test_cli_defaults():
    """Test that CLI defaults are set correctly."""
    from src.cli import create_parser
    
    parser = create_parser()
    
    args = parser.parse_args([
        '--method', 'adaptive',
        '--emitter', '5.1', '2.1',
        '--setback', '1.0'
    ])
    
    # Test adaptive defaults
    assert args.rel_tol == 3e-3
    assert args.abs_tol == 1e-6
    assert args.max_depth == 12
    assert args.min_cell_area_frac == 1e-8
    assert args.max_cells == 200000
    assert args.time_limit_s == 60.0
    assert args.init_grid == '4x4'
    
    # Test fixed grid defaults
    assert args.grid_nx == 100
    assert args.grid_ny == 100
    assert args.quadrature == 'centroid'
    
    # Test Monte Carlo defaults
    assert args.samples == 200000
    assert args.target_rel_ci == 0.02
    assert args.max_iters == 50
    assert args.seed == 42
    
    # Test general defaults
    assert args.angle == 0.0
    assert str(args.outdir) == 'results'
    assert args.plot is False
    assert args.verbose is False


def test_cli_validation():
    """Test CLI argument validation."""
    from src.cli import validate_args, normalize_args
    import argparse
    from pathlib import Path
    
    # Test valid arguments
    args = argparse.Namespace(
        method='adaptive',
        emitter=[5.1, 2.1],
        receiver=None,
        setback=1.0,
        angle=0.0,
        cases=None,
        plot=False,
        outdir=Path('./results'),
        rel_tol=3e-3,
        abs_tol=1e-6,
        max_depth=12,
        min_cell_area_frac=1e-8,
        max_cells=200000,
        time_limit_s=60.0,
        init_grid='4x4',
        grid_nx=100,
        grid_ny=100,
        quadrature='centroid',
        samples=200000,
        target_rel_ci=0.02,
        max_iters=50,
        seed=42,
        verbose=False
    )
    
    # Should not raise exception
    validate_args(args)
    
    # Test normalization
    normalized = normalize_args(args)
    assert normalized.receiver == [5.1, 2.1]  # Should default to emitter
    assert hasattr(normalized, 'init_grid_nx')
    assert normalized.init_grid_nx == 4
    assert normalized.init_grid_ny == 4


def test_cli_validation_errors():
    """Test CLI validation error cases."""
    from src.cli import validate_args
    import argparse
    from pathlib import Path
    
    # Test missing emitter when not using cases
    args = argparse.Namespace(
        method='adaptive',
        emitter=None,
        receiver=None,
        setback=1.0,
        cases=None,
        rel_tol=3e-3,
        abs_tol=1e-6,
        max_depth=12,
        max_cells=200000,
        time_limit_s=60.0,
        grid_nx=100,
        grid_ny=100,
        samples=200000,
        target_rel_ci=0.02,
        max_iters=50
    )
    
    with pytest.raises(ValueError, match="--emitter is required"):
        validate_args(args)
    
    # Test missing setback
    args.emitter = [5.1, 2.1]
    args.setback = None
    
    with pytest.raises(ValueError, match="--setback is required"):
        validate_args(args)
    
    # Test negative dimensions
    args.setback = 1.0
    args.emitter = [-5.1, 2.1]
    
    with pytest.raises(ValueError, match="Emitter dimensions must be positive"):
        validate_args(args)
    
    # Test negative setback
    args.emitter = [5.1, 2.1]
    args.setback = -1.0
    
    with pytest.raises(ValueError, match="Setback distance must be positive"):
        validate_args(args)


def test_cli_main_with_args():
    """Test the main_with_args function."""
    from src.cli import main_with_args
    import argparse
    from pathlib import Path
    
    # Test successful execution
    args = argparse.Namespace(
        method='analytical',
        emitter=[5.1, 2.1],
        receiver=None,
        setback=1.0,
        angle=0.0,
        cases=None,
        plot=False,
        outdir=Path('./results'),
        analytical_nx=240,
        analytical_ny=240,
        rc_mode='center',
        rc_grid_n=21,
        rc_search_rel_tol=3e-3,
        rc_search_max_iters=200,
        rc_search_multistart=8,
        rc_search_time_limit_s=10.0,
        rc_bounds='auto',
        rel_tol=3e-3,
        abs_tol=1e-6,
        max_depth=12,
        min_cell_area_frac=1e-8,
        max_cells=200000,
        time_limit_s=60.0,
        init_grid='4x4',
        grid_nx=100,
        grid_ny=100,
        quadrature='centroid',
        samples=200000,
        target_rel_ci=0.02,
        max_iters=50,
        seed=42,
        verbose=False
    )
    
    # Should return 0 for success
    exit_code = main_with_args(args)
    assert exit_code == 0
    
    # Test with invalid arguments
    args.emitter = None
    exit_code = main_with_args(args)
    assert exit_code == 1  # Should return 1 for validation error


def test_cli_init_grid_parsing():
    """Test init-grid format parsing."""
    from src.cli import normalize_args
    import argparse
    from pathlib import Path
    
    # Test "4x4" format
    args = argparse.Namespace(
        emitter=[5.1, 2.1],
        receiver=None,
        outdir=Path('./results'),
        init_grid='4x4'
    )
    
    normalized = normalize_args(args)
    assert normalized.init_grid_nx == 4
    assert normalized.init_grid_ny == 4
    
    # Test "8x6" format
    args.init_grid = '8x6'
    normalized = normalize_args(args)
    assert normalized.init_grid_nx == 8
    assert normalized.init_grid_ny == 6
    
    # Test single number format
    args.init_grid = '10'
    normalized = normalize_args(args)
    assert normalized.init_grid_nx == 10
    assert normalized.init_grid_ny == 10
    
    # Test invalid format
    args.init_grid = 'invalid'
    with pytest.raises(ValueError, match="Invalid init-grid format"):
        normalize_args(args)


def test_yaml_io_functions():
    """Test YAML I/O utility functions."""
    from src.io_yaml import create_sample_config
    
    # Should create valid config
    config = create_sample_config()
    assert isinstance(config, dict)
    assert 'geometry' in config
    assert 'methods' in config
    assert 'output' in config


def test_numpy_imports():
    """Test that numpy functions work correctly."""
    # Basic numpy operations used in the code
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    
    # Cross product
    c = np.cross(a, b)
    assert len(c) == 3
    
    # Norm
    norm = np.linalg.norm(a)
    assert norm > 0
    
    # Dot product
    dot = np.dot(a, b)
    assert isinstance(dot, (int, float, np.number))


def test_scipy_imports():
    """Test that scipy functions are available."""
    from numpy.polynomial.legendre import leggauss
    
    # Test Gauss-Legendre quadrature
    points, weights = leggauss(5)
    assert len(points) == 5
    assert len(weights) == 5
    assert np.allclose(np.sum(weights), 2.0)  # Integral of 1 over [-1,1]


if __name__ == "__main__":
    pytest.main([__file__])
