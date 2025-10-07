"""
Command-line interface for the radiation view factor validation tool.

This module provides the main CLI entry point and orchestrates the
view factor calculation workflow following SOLID principles.
"""

import sys
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Import refactored modules
from .cli_parser import create_parser, validate_args, normalize_args
from .cli_cases import run_cases
from .cli_results import print_parsed_args, print_results, save_and_report_csv

# Logger will be configured after argument parsing
logger = logging.getLogger(__name__)


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


def run_calculation(args) -> Dict[str, Any]:
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
    method_params = _create_method_params(args)
    
    # Resolve centre-to-centre offsets (receiver - emitter)
    dy, dz = _resolve_offsets(args)

    # Create geometry configuration (using new offset system)
    geom_cfg = _create_geometry_config(args, dy, dz)
    
    # Optional: geometry preview (for plotting/search bounds)
    _preview_geometry(args, dy, dz)

    # Create evaluator function
    vf_evaluator = create_vf_evaluator(
        args.method, em_w, em_h, rc_w, rc_h, setback, angle, geom_cfg, **method_params
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
    
    dy, dz = _resolve_offsets(args)
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
        'info': f"{args.method} {args.rc_mode} mode, peak at ({float(peak_result['x_peak']):.3f}, {float(peak_result['y_peak']):.3f})",
        'grid_data': grid_data
    }
    result.update({
        'We': em_w, 'He': em_h, 'Wr': rc_w, 'Hr': rc_h,
        'emitter_center': (setback, 0.0, 0.0),
        'receiver_center': (0.0, float(dy), float(dz)),
        'R_emitter': [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
        'R_receiver': [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
    })
    
    return result


def _create_method_params(args) -> Dict[str, Any]:
    """Create method-specific parameters from arguments."""
    return {
        'analytical_nx': getattr(args, 'analytical_nx', 240),
        'analytical_ny': getattr(args, 'analytical_ny', 240),
        'rel_tol': getattr(args, 'rel_tol', 3e-3),
        'abs_tol': getattr(args, 'abs_tol', 1e-6),
        'max_depth': getattr(args, 'max_depth', 12),
        'max_cells': getattr(args, 'max_cells', 200000),
        'min_cell_area_frac': getattr(args, 'min_cell_area_frac', 1e-8),
        'min_cells': getattr(args, 'min_cells', 16),
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


def _resolve_offsets(args) -> tuple[float, float]:
    """Resolve centre-to-centre offsets (receiver - emitter)."""
    dy = dz = 0.0
    if args.align_centres:
        dy = dz = 0.0
    elif args.receiver_offset:
        dy, dz = args.receiver_offset
    elif args.emitter_offset:
        # convert to (receiver - emitter) convention
        dy, dz = (-args.emitter_offset[0], -args.emitter_offset[1])
    return dy, dz


def _create_geometry_config(args, dy: float, dz: float) -> Dict[str, Any]:
    """Create geometry configuration."""
    return {
        "emitter_offset": (0.0, 0.0),  # Legacy compatibility
        "receiver_offset": (dy, dz),    # New offset system (receiver - emitter)
        "angle_deg": float(args.angle),
        "angle_pivot": args.angle_pivot,
        "rotate_target": args.rotate_target,
        # Augmented fields for orientation-aware evaluators
        "emitter_width": float(args.emitter[0]),
        "emitter_height": float(args.emitter[1]),
        "setback": float(args.setback),
        "rotate_axis": args.rotate_axis,   # 'z' or 'y'
        "angle": float(args.angle),        # duplicate in degrees for consumers
        "dy": float(dy),
        "dz": float(dz),
    }


def _preview_geometry(args, dy: float, dz: float) -> None:
    """Preview geometry for plotting/search bounds."""
    try:
        from .geometry import PanelSpec, PlacementOpts, place_panels
        em = PanelSpec(args.emitter[0], args.emitter[1])
        rc = PanelSpec(args.receiver[0], args.receiver[1])
        opts = PlacementOpts(
            angle_deg=args.angle,
            rotate_axis=args.rotate_axis,
            rotate_target=args.rotate_target,
            pivot=args.angle_pivot,
            offset_dy=dy, offset_dz=dz,
            align_centres=args.align_centres,
        )
        _ = place_panels(em, rc, setback=args.setback, opts=opts)
    except Exception as e:
        logger.warning(f"Geometry preview failed: {e}")


def main_with_args(args) -> int:
    """Main function that takes parsed arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Backfill defaults for tests that pass a partial Namespace
    _backfill_defaults(args)

    try:
        # Handle version flag
        if args.version:
            from . import __version__
            print(__version__)
            return 0
        
        # Set up logging
        _setup_logging(args)
        
        # Handle plotting setup if requested
        if args.plot:
            _ensure_headless_matplotlib()
            try:
                import matplotlib.pyplot as plt  # noqa
            except Exception as e:
                raise SystemExit("Plotting requested but matplotlib is not available. "
                                 "Run: pip install matplotlib") from e
        
        # Resolve output directory early so everything downstream uses a concrete path
        _resolve_output_directory(args)

        # Provide missing geometry defaults for legacy callers
        _provide_geometry_defaults(args)

        from .cli_parser import map_eval_mode_args
        map_eval_mode_args(args)
        # Validate arguments
        validate_args(args)
        
        # If using cases file, run cases and exit
        if args.cases:
            return run_cases(args.cases, str(args.outdir), plot=args.plot)
        
        # Normalize arguments and apply defaults for single case
        args = normalize_args(args)
        
        # Print parsed arguments
        print_parsed_args(args)
        
        # Run calculation
        result = run_calculation(args)
        
        # Print and save results
        print_results(result, args)
        save_and_report_csv(result, args)
        
        # Generate plots if requested
        if args.plot:
            from .plotting import create_heatmap_plot
            create_heatmap_plot(result, args, result.get('grid_data'))
            try:
                from .viz.plots import plot_geometry_and_heatmap
                from pathlib import Path as _P
                plot_geometry_and_heatmap(
                    result=result,
                    eval_mode=args.eval_mode,
                    method=args.method,
                    setback=float(args.setback),
                    out_png=_P(args.outdir) / 'figure.png',
                )
            except Exception as _e:
                logger.warning(f"Combined geometry/heatmap plot skipped: {_e}")

        return 0
        
    except ValueError as e:
        logger.error(f"Invalid arguments: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 2


def _backfill_defaults(args) -> None:
    """Backfill defaults for tests that pass a partial Namespace."""
    for name, default in (
        ('log_level', 'INFO'),
        ('outdir', 'results'),
        ('test_run', False),
        ('rotate_axis', 'z'),
        ('rotate_target', 'emitter'),
        ('angle_pivot', 'toe'),
        ('angle', 0.0),
        ('rc_mode', 'center'),
        ('method', 'analytical'),
    ):
        if not hasattr(args, name) or getattr(args, name) in (None, ''):
            setattr(args, name, default)
    try:
        args.angle = float(args.angle)
    except Exception:
        args.angle = 0.0


def _setup_logging(args) -> None:
    """Set up logging configuration."""
    # Set logging level
    if getattr(args, 'verbose', False):
        args.log_level = "DEBUG"
    
    level = getattr(logging, getattr(args, 'log_level', 'INFO'), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    if args.log_level != "DEBUG":
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("fontTools").setLevel(logging.WARNING)


def _resolve_output_directory(args) -> None:
    """Resolve output directory early so everything downstream uses a concrete path."""
    try:
        from .paths import resolve_outdir
        out_dir_path = resolve_outdir(args.outdir, test_run=args.test_run)
        args.outdir = str(out_dir_path)
    except Exception as e:
        # fallback to user-supplied or default; creation will be attempted later
        logger.warning(f"Outdir resolution failed: {e}")
        if args.outdir is None:
            args.outdir = "results"


def _provide_geometry_defaults(args) -> None:
    """Provide missing geometry defaults for legacy callers."""
    if not hasattr(args, 'align_centres'):
        args.align_centres = False
    if not hasattr(args, 'receiver_offset'):
        args.receiver_offset = (0.0, 0.0)


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
