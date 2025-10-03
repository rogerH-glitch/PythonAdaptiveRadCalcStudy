"""
Command-line interface for the radiation view factor validation tool.

This module provides the main CLI entry point and argument parsing
for the view factor calculation tool.
"""

import argparse
import sys
import os
import csv
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

def _ensure_headless_matplotlib():
    """
    Use a headless backend when no display is available (Windows/CI safe).
    Call this right before importing matplotlib.pyplot if plotting is requested.
    """
    try:
        import matplotlib  # noqa
    except Exception:
        # Will be handled later if plotting is requested
        return
    # Force Agg on Windows or when DISPLAY is missing
    try:
        import matplotlib
        if (os.name == "nt") or (not os.environ.get("DISPLAY")):
            matplotlib.use("Agg")  # headless backend
    except Exception:
        pass

try:
    import numpy as np
except ImportError as e:
    raise SystemExit("Missing dependency 'numpy'. Run: pip install -r requirements.txt") from e

try:
    import yaml  # used by YAML mode
except ImportError as e:
    raise SystemExit("Missing dependency 'pyyaml'. Run: pip install -r requirements.txt") from e

from .analytical import local_peak_vf_analytic_approx, validate_geometry, get_analytical_info

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Local Python tool for radiation view factor validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --method adaptive --emitter 5.1 2.1 --setback 1.0
  python main.py --method fixedgrid --emitter 20.02 1.05 --setback 0.81 --plot
  python main.py --method montecarlo --emitter 5.1 2.1 --receiver 4.0 1.8 --setback 2.0
  python main.py --cases validation_cases.yaml --outdir ./output

Default assumptions:
  - Surfaces face each other, centres aligned
  - Receiver dimensions default to emitter dimensions
  - Parallel orientation (angle = 0°)
        """
    )
    
    # Core calculation method (required unless using --cases)
    parser.add_argument(
        '--method',
        choices=['analytical', 'fixedgrid', 'adaptive', 'montecarlo'],
        help='Calculation method to use (required unless using --cases)'
    )
    
    # Geometry parameters - emitter is required unless using --cases
    parser.add_argument(
        '--emitter',
        nargs=2,
        type=float,
        metavar=('W', 'H'),
        help='Emitter dimensions (width height) in metres'
    )
    parser.add_argument(
        '--receiver',
        nargs=2,
        type=float,
        metavar=('W', 'H'),
        help='Receiver dimensions (width height) in metres (default: same as emitter)'
    )
    parser.add_argument(
        '--setback',
        type=float,
        metavar='S',
        help='Setback distance in metres'
    )
    parser.add_argument(
        '--angle',
        type=float,
        default=0.0,
        metavar='DEG',
        help='Rotation angle in degrees (default: 0 for parallel)'
    )
    
    # Test cases and output options
    parser.add_argument(
        '--cases',
        type=str,
        metavar='PATH',
        help='YAML test suite file path (optional)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate plots'
    )
    parser.add_argument(
        '--outdir',
        type=Path,
        default=Path('./results'),
        metavar='PATH',
        help='Output directory (default: ./results)'
    )
    
    # Analytical method tuning parameters
    analytical_group = parser.add_argument_group('Analytical method tuning')
    analytical_group.add_argument(
        '--analytical-nx',
        type=int,
        default=240,
        help='Emitter grid Nx for analytical approx (default: 240)'
    )
    analytical_group.add_argument(
        '--analytical-ny',
        type=int,
        default=240,
        help='Emitter grid Ny for analytical approx (default: 240)'
    )
    
    # Adaptive method tuning parameters
    adaptive_group = parser.add_argument_group('Adaptive method tuning')
    adaptive_group.add_argument(
        '--rel-tol',
        type=float,
        default=3e-3,
        help='Relative tolerance (default: 3e-3)'
    )
    adaptive_group.add_argument(
        '--abs-tol',
        type=float,
        default=1e-6,
        help='Absolute tolerance (default: 1e-6)'
    )
    adaptive_group.add_argument(
        '--max-depth',
        type=int,
        default=12,
        help='Maximum recursion depth (default: 12)'
    )
    adaptive_group.add_argument(
        '--min-cell-area-frac',
        type=float,
        default=1e-8,
        help='Minimum cell area fraction (default: 1e-8)'
    )
    adaptive_group.add_argument(
        '--max-cells',
        type=int,
        default=200000,
        help='Maximum number of cells (default: 200000)'
    )
    adaptive_group.add_argument(
        '--time-limit-s',
        type=float,
        default=60.0,
        help='Time limit in seconds (default: 60)'
    )
    adaptive_group.add_argument(
        '--init-grid',
        type=str,
        default='4x4',
        help='Initial grid size (default: 4x4)'
    )
    
    # Fixed grid method tuning parameters
    fixed_group = parser.add_argument_group('Fixed grid method tuning')
    fixed_group.add_argument(
        '--grid-nx',
        type=int,
        default=100,
        help='Grid points in x-direction (default: 100)'
    )
    fixed_group.add_argument(
        '--grid-ny',
        type=int,
        default=100,
        help='Grid points in y-direction (default: 100)'
    )
    fixed_group.add_argument(
        '--quadrature',
        choices=['centroid', '2x2'],
        default='centroid',
        help='Quadrature method (default: centroid)'
    )
    
    # Monte Carlo method tuning parameters
    mc_group = parser.add_argument_group('Monte Carlo method tuning')
    mc_group.add_argument(
        '--samples',
        type=int,
        default=200000,
        help='Number of samples (default: 200000)'
    )
    mc_group.add_argument(
        '--target-rel-ci',
        type=float,
        default=0.02,
        help='Target relative confidence interval (default: 0.02)'
    )
    mc_group.add_argument(
        '--max-iters',
        type=int,
        default=50,
        help='Maximum iterations (default: 50)'
    )
    mc_group.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    # Peak locator options
    peak_group = parser.add_argument_group('Peak locator options')
    peak_group.add_argument(
        '--rc-mode',
        choices=['center', 'grid', 'search'],
        default='center',
        help='Receiver peak search mode (default: center)'
    )
    peak_group.add_argument(
        '--rc-grid-n',
        type=int,
        default=21,
        help='Grid resolution for coarse sampling (default: 21)'
    )
    peak_group.add_argument(
        '--rc-search-rel-tol',
        type=float,
        default=3e-3,
        help='Target relative improvement tolerance (default: 3e-3)'
    )
    peak_group.add_argument(
        '--rc-search-max-iters',
        type=int,
        default=200,
        help='Max local-optimizer iterations (default: 200)'
    )
    peak_group.add_argument(
        '--rc-search-multistart',
        type=int,
        default=8,
        help='Number of multi-start seeds (default: 8)'
    )
    peak_group.add_argument(
        '--rc-search-time-limit-s',
        type=float,
        default=10.0,
        help='Wall clock cap for search phase (default: 10.0)'
    )
    peak_group.add_argument(
        '--rc-bounds',
        choices=['auto', 'explicit'],
        default='auto',
        help='Bounds mode (default: auto)'
    )
    
    # General options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate parsed command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Raises:
        ValueError: If arguments are invalid or missing
    """
    # If using test cases file, emitter/setback not required
    if args.cases:
        if not os.path.isfile(args.cases):
            raise ValueError(f"Test cases file not found: {args.cases}")
        return
    
    # For direct geometry specification, method, emitter and setback are required
    if not args.method:
        raise ValueError("--method is required when not using --cases")
    
    if not args.emitter:
        raise ValueError("--emitter is required when not using --cases")
    
    if args.setback is None:
        raise ValueError("--setback is required when not using --cases")
    
    # Validate emitter dimensions
    emitter_width, emitter_height = args.emitter
    if emitter_width <= 0 or emitter_height <= 0:
        raise ValueError(f"Emitter dimensions must be positive, got {emitter_width} × {emitter_height}")
    
    # Validate receiver dimensions if provided
    if args.receiver:
        receiver_width, receiver_height = args.receiver
        if receiver_width <= 0 or receiver_height <= 0:
            raise ValueError(f"Receiver dimensions must be positive, got {receiver_width} × {receiver_height}")
    
    # Validate setback distance
    if args.setback <= 0:
        raise ValueError(f"Setback distance must be positive, got {args.setback}")
    
    # Validate tuning parameters
    if args.rel_tol <= 0:
        raise ValueError(f"Relative tolerance must be positive, got {args.rel_tol}")
    
    if args.abs_tol <= 0:
        raise ValueError(f"Absolute tolerance must be positive, got {args.abs_tol}")
    
    if args.max_depth < 1:
        raise ValueError(f"Maximum depth must be at least 1, got {args.max_depth}")
    
    if args.max_cells < 1:
        raise ValueError(f"Maximum cells must be at least 1, got {args.max_cells}")
    
    if args.time_limit_s <= 0:
        raise ValueError(f"Time limit must be positive, got {args.time_limit_s}")
    
    if args.grid_nx < 1 or args.grid_ny < 1:
        raise ValueError(f"Grid dimensions must be at least 1, got {args.grid_nx} × {args.grid_ny}")
    
    if args.samples < 1:
        raise ValueError(f"Number of samples must be at least 1, got {args.samples}")
    
    if not (0 < args.target_rel_ci < 1):
        raise ValueError(f"Target relative CI must be between 0 and 1, got {args.target_rel_ci}")
    
    if args.max_iters < 1:
        raise ValueError(f"Maximum iterations must be at least 1, got {args.max_iters}")


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """Normalize and set defaults for parsed arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Normalized arguments with defaults applied
    """
    # If receiver dimensions not provided, use emitter dimensions
    if args.emitter and not args.receiver:
        args.receiver = args.emitter.copy()
        logger.debug(f"Receiver dimensions defaulted to emitter: {args.receiver}")
    
    # Ensure output directory exists
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    # Parse init-grid format (e.g., "4x4" -> (4, 4))
    if 'x' in args.init_grid.lower():
        try:
            nx, ny = map(int, args.init_grid.lower().split('x'))
            args.init_grid_nx = nx
            args.init_grid_ny = ny
        except ValueError:
            raise ValueError(f"Invalid init-grid format: {args.init_grid}. Expected format: NxM")
    else:
        # Assume square grid if single number
        try:
            n = int(args.init_grid)
            args.init_grid_nx = n
            args.init_grid_ny = n
        except ValueError:
            raise ValueError(f"Invalid init-grid format: {args.init_grid}. Expected format: N or NxM")
    
    return args


def print_parsed_args(args: argparse.Namespace) -> None:
    """Print parsed arguments in a formatted way.
    
    Args:
        args: Parsed and normalized arguments
    """
    print("=== Parsed Arguments ===")
    
    # Core parameters
    print(f"Method: {args.method}")
    
    if args.cases:
        print(f"Test cases file: {args.cases}")
    else:
        print(f"Emitter: {args.emitter[0]:.3f} × {args.emitter[1]:.3f} m")
        print(f"Receiver: {args.receiver[0]:.3f} × {args.receiver[1]:.3f} m")
        print(f"Setback: {args.setback:.3f} m")
        print(f"Angle: {args.angle:.1f}°")
    
    print(f"Output directory: {args.outdir}")
    print(f"Generate plots: {args.plot}")
    
    # Method-specific parameters
    if args.method == 'adaptive':
        print("\n--- Adaptive Method Parameters ---")
        print(f"Relative tolerance: {args.rel_tol:.2e}")
        print(f"Absolute tolerance: {args.abs_tol:.2e}")
        print(f"Maximum depth: {args.max_depth}")
        print(f"Minimum cell area fraction: {args.min_cell_area_frac:.2e}")
        print(f"Maximum cells: {args.max_cells:,}")
        print(f"Time limit: {args.time_limit_s:.1f} s")
        print(f"Initial grid: {args.init_grid} ({args.init_grid_nx}×{args.init_grid_ny})")
    
    elif args.method == 'fixedgrid':
        print("\n--- Fixed Grid Method Parameters ---")
        print(f"Grid size: {args.grid_nx} × {args.grid_ny}")
        print(f"Quadrature: {args.quadrature}")
    
    elif args.method == 'montecarlo':
        print("\n--- Monte Carlo Method Parameters ---")
        print(f"Samples: {args.samples:,}")
        print(f"Target relative CI: {args.target_rel_ci:.3f}")
        print(f"Maximum iterations: {args.max_iters}")
        print(f"Random seed: {args.seed}")
    
    elif args.method == 'analytical':
        print("\n--- Analytical Method ---")
        print("No additional parameters")
    
    print("\n--- Default Assumptions ---")
    print("- Surfaces face each other")
    print("- Centres are aligned")
    print("- Parallel orientation (unless angle specified)")
    print("="*50)


def run_cases(cases_path: str, outdir: str) -> int:
    """Run validation cases from YAML file and generate summary CSV.
    
    Args:
        cases_path: Path to YAML cases file
        outdir: Output directory for results
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    from .io_yaml import load_cases, validate_case_schema, coerce_case_to_cli_kwargs
    
    try:
        cases = load_cases(cases_path)
        os.makedirs(outdir, exist_ok=True)
        summary_csv = os.path.join(outdir, "cases_summary.csv")
        
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id","method","vf","expected","rel_err","status","notes"])
            
            for case in cases:
                try:
                    validate_case_schema(case)
                    if not case["enabled"]:
                        writer.writerow([case.get("id"), case.get("method","adaptive"), "", "", "", "skipped", "disabled"])
                        continue
                    kw = coerce_case_to_cli_kwargs(case)
                except Exception as e:
                    writer.writerow([case.get("id","<unknown>"), case.get("method","adaptive"), "", "", "", "invalid", str(e)])
                    continue

                # Placeholder solve (replace in later steps)
                # For now, produce a deterministic dummy vf
                start = time.time()
                vf = 0.123456
                status = "pending"
                
                # Compare against expected if present
                expected = kw.get("expected")
                rel_err = ""
                if expected is not None:
                    rel_err = abs(vf - expected)/expected if expected != 0 else ""
                
                writer.writerow([kw["id"], kw["method"], f"{vf:.8f}", expected if expected is not None else "", rel_err, status, "placeholder"])
        
        print(f"Wrote: {summary_csv}")
        return 0
        
    except Exception as e:
        logger.error(f"Error running cases: {e}")
        return 1


def run_calculation(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run the view factor calculation based on the specified method.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary containing calculation results
    """
    # Extract geometry parameters
    em_w, em_h = args.emitter
    rc_w, rc_h = args.receiver
    setback = args.setback
    angle = args.angle
    
    # Import peak locator
    from .peak_locator import find_local_peak, create_vf_evaluator
    
    # Create method-specific evaluator
    method_params = {
        'analytical_nx': getattr(args, 'analytical_nx', 240),
        'analytical_ny': getattr(args, 'analytical_ny', 240),
        'rel_tol': getattr(args, 'rel_tol', 3e-3),
        'abs_tol': getattr(args, 'abs_tol', 1e-6),
        'max_depth': getattr(args, 'max_depth', 12),
        'max_cells': getattr(args, 'max_cells', 200000),
        'min_cell_area_frac': getattr(args, 'min_cell_area_frac', 1e-8),
        'init_grid': getattr(args, 'init_grid', '4x4'),
        'grid_nx': getattr(args, 'grid_nx', 100),
        'grid_ny': getattr(args, 'grid_ny', 100),
        'quadrature': getattr(args, 'quadrature', 'centroid'),
        'samples': getattr(args, 'samples', 200000),
        'target_rel_ci': getattr(args, 'target_rel_ci', 0.02),
        'max_iters': getattr(args, 'max_iters', 50),
        'seed': getattr(args, 'seed', 42),
        'time_limit_s': getattr(args, 'time_limit_s', 60.0)
    }
    
    # Create evaluator function
    vf_evaluator = create_vf_evaluator(
        args.method, em_w, em_h, rc_w, rc_h, setback, angle, **method_params
    )
    
    # Find local peak
    start_time = time.time()
    peak_result = find_local_peak(
        em_w, em_h, rc_w, rc_h, setback, angle, vf_evaluator,
        rc_mode=args.rc_mode,
        rc_grid_n=args.rc_grid_n,
        rc_search_rel_tol=args.rc_search_rel_tol,
        rc_search_max_iters=args.rc_search_max_iters,
        rc_search_multistart=args.rc_search_multistart,
        rc_search_time_limit_s=args.rc_search_time_limit_s,
        rc_bounds=args.rc_bounds
    )
    calc_time = time.time() - start_time
    
    # Generate grid data for plotting if requested
    grid_data = None
    if args.plot and args.rc_mode in ['grid', 'search']:
        from .plotting import generate_grid_data_for_plotting
        grid_data = generate_grid_data_for_plotting(
            em_w, em_h, rc_w, rc_h, setback, angle,
            args.method, method_params, args.rc_grid_n
        )
    
    # Build result dictionary
    result = {
        'method': args.method,
        'vf': peak_result['vf_peak'],
        'calc_time': calc_time,
        'x_peak': peak_result['x_peak'],
        'y_peak': peak_result['y_peak'],
        'rc_mode': args.rc_mode,
        'status': peak_result['status'],
        'geometry': {
            'emitter': (em_w, em_h),
            'receiver': (rc_w, rc_h),
            'setback': setback,
            'angle': angle
        },
        'search_metadata': peak_result.get('search_metadata', {}),
        'info': f"{args.method} {args.rc_mode} mode, peak at ({peak_result['x_peak']:.3f}, {peak_result['y_peak']:.3f})",
        'grid_data': grid_data
    }
    
    return result


def print_results(result: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    Print calculation results to console.
    
    Args:
        result: Calculation results dictionary
        args: Parsed command-line arguments
    """
    print("\n" + "="*50)
    print("CALCULATION RESULTS")
    print("="*50)
    
    method = result['method']
    vf = result['vf']
    calc_time = result['calc_time']
    
    print(f"Method: {method.title()}")
    print(f"Local Peak View Factor: {vf:.8f}")
    print(f"Peak Location: ({result.get('x_peak', 0.0):.3f}, {result.get('y_peak', 0.0):.3f}) m")
    print(f"RC Mode: {result.get('rc_mode', 'center')}")
    print(f"Calculation Time: {calc_time:.3f} seconds")
    
    if 'info' in result:
        print(f"\nMethod Info:")
        print(f"  {result['info']}")
    
    # Show search metadata if available
    search_metadata = result.get('search_metadata', {})
    if search_metadata:
        print(f"\nSearch Details:")
        print(f"  Evaluations: {search_metadata.get('evaluations', 0)}")
        print(f"  Search Time: {search_metadata.get('time_s', 0.0):.3f} s")
        if 'seeds_used' in search_metadata:
            print(f"  Seeds Used: {search_metadata['seeds_used']}")
    
    print("="*50)


def save_results(result: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    Save calculation results to CSV file.
    
    Args:
        result: Calculation results dictionary
        args: Parsed command-line arguments
    """
    # Ensure output directory exists
    os.makedirs(args.outdir, exist_ok=True)
    
    # Generate output filename based on method
    method = result['method']
    csv_filename = f"{method}.csv"
    csv_path = os.path.join(args.outdir, csv_filename)
    
    # Extract geometry and peak information
    geom = result['geometry']
    em_w, em_h = geom['emitter']
    rc_w, rc_h = geom['receiver']
    setback = geom['setback']
    angle = geom['angle']
    vf = result['vf']
    x_peak = result.get('x_peak', 0.0)
    y_peak = result.get('y_peak', 0.0)
    rc_mode = result.get('rc_mode', 'center')
    status = result.get('status', 'unknown')
    
    # Extract search metadata
    search_metadata = result.get('search_metadata', {})
    evaluations = search_metadata.get('evaluations', 0)
    search_time = search_metadata.get('time_s', 0.0)
    
    # Write CSV with peak locator fields
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header row
            writer.writerow([
                'method', 'emitter_w', 'emitter_h', 'receiver_w', 'receiver_h', 'setback', 'angle',
                'rc_mode', 'x_peak', 'y_peak', 'vf_peak', 'status', 'evaluations', 'search_time_s'
            ])
            
            # Data row
            writer.writerow([
                method, em_w, em_h, rc_w, rc_h, setback, angle,
                rc_mode, f"{x_peak:.6f}", f"{y_peak:.6f}", f"{vf:.8f}", 
                status, evaluations, f"{search_time:.3f}"
            ])
        
        print(f"\nResults saved to: {csv_path}")
        
    except PermissionError:
        # Fallback to timestamped filename if permission denied
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_filename = f"{method}_{timestamp}.csv"
        fallback_path = os.path.join(args.outdir, fallback_filename)
        
        with open(fallback_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header row
            writer.writerow([
                'method', 'emitter_w', 'emitter_h', 'receiver_w', 'receiver_h', 'setback', 'angle',
                'rc_mode', 'x_peak', 'y_peak', 'vf_peak', 'status', 'evaluations', 'search_time_s'
            ])
            
            # Data row
            writer.writerow([
                method, em_w, em_h, rc_w, rc_h, setback, angle,
                rc_mode, f"{x_peak:.6f}", f"{y_peak:.6f}", f"{vf:.8f}", 
                status, evaluations, f"{search_time:.3f}"
            ])
        
        print(f"\nResults saved to: {fallback_path}")


def main_with_args(args: argparse.Namespace) -> int:
    """Main function that takes parsed arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Set logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Handle plotting setup if requested
        if args.plot:
            _ensure_headless_matplotlib()
            try:
                import matplotlib.pyplot as plt  # noqa
            except Exception as e:
                raise SystemExit("Plotting requested but matplotlib is not available. "
                                 "Run: pip install matplotlib") from e
        
        # Validate arguments
        validate_args(args)
        
        # If using cases file, run cases and exit
        if args.cases:
            return run_cases(args.cases, str(args.outdir))
        
        # Normalize arguments and apply defaults for single case
        args = normalize_args(args)
        
        # Print parsed arguments
        print_parsed_args(args)
        
        # Run calculation
        result = run_calculation(args)
        
        # Print and save results
        print_results(result, args)
        save_results(result, args)
        
        # Generate plots if requested
        if args.plot:
            from .plotting import create_heatmap_plot
            create_heatmap_plot(result, args, result.get('grid_data'))
        
        return 0
        
    except ValueError as e:
        logger.error(f"Invalid arguments: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 2


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    
    # If no arguments provided, show help and exit with code 2
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(2)
    
    try:
        args = parser.parse_args()
        exit_code = main_with_args(args)
        sys.exit(exit_code)
        
    except SystemExit:
        # Re-raise SystemExit (from argparse errors or explicit sys.exit calls)
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
