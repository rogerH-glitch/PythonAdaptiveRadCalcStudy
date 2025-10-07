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
from .cli_parser import create_parser, validate_args, normalize_args, map_eval_mode_args
from dataclasses import asdict
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
    
    # Derive offsets for geometry panel centers (receiver at origin by convention)
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
    # Geometry info for plotting (wireframes/labels)
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


def _isclose(a, b, eps=1e-12):  # tiny helper for floats
    try:
        return abs(float(a) - float(b)) <= eps
    except Exception:
        return a == b


def _print_user_relevant_summary(args):
    """Only print options the user actually changed or that materially affect the run."""
    print("=== Parsed Arguments ===")
    print(f"Method: {args.method}")
    print(f"Emitter: {args.emitter[0]:.3f} × {args.emitter[1]:.3f} m")
    print(f"Receiver: {args.receiver[0]:.3f} × {args.receiver[1]:.3f} m")
    print(f"Setback: {float(args.setback):.3f} m")

    # Orientation: only if non-zero angle or non-default rotation options
    has_angle = not _isclose(getattr(args, "angle", 0.0), 0.0)
    non_default_rot = (
        getattr(args, "rotate_axis", "z") != "z" or
        getattr(args, "rotate_target", "emitter") != "emitter" or
        getattr(args, "angle_pivot", "toe") != "toe"
    )
    if has_angle or non_default_rot:
        print(f"Angle: {getattr(args,'angle',0.0):.1f}°")
        print(f"Rotate axis: {getattr(args,'rotate_axis','z')}")
        print(f"Rotate target: {getattr(args,'rotate_target','emitter')}")
        print(f"Angle pivot: {getattr(args,'angle_pivot','toe')}")

    # Offsets: print only if not centred or if user forced align-centres
    r_off = getattr(args, "receiver_offset", None)
    e_off = getattr(args, "emitter_offset", None)
    align = bool(getattr(args, "align_centres", False))
    if align:
        print("Align centres: True")
    if r_off and (not _isclose(r_off[0], 0.0) or not _isclose(r_off[1], 0.0)):
        print(f"Receiver offset (receiver - emitter): ({r_off[0]:.3f}, {r_off[1]:.3f}) m  [y,z]")
    elif e_off and (not _isclose(e_off[0], 0.0) or not _isclose(e_off[1], 0.0)):
        # Mirror info as receiver_offset for clarity
        print(f"Emitter offset given; equivalent receiver offset: ({-e_off[0]:.3f}, {-e_off[1]:.3f}) m  [y,z]")

    # Output dir & plotting flags (concise)
    # Print exactly what the user provided, not any internal resolved path
    outdir_raw = getattr(args, "_outdir_user", args.outdir)
    print(f"Output directory: {outdir_raw}")
    if getattr(args, "plot", False):
        pmode = getattr(args, "plot_geom", "2d")
        print(f"Generate plots: True (geom={pmode})")


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
        resolved_outdir = _resolve_output_directory(args)

        # Provide missing geometry defaults for legacy callers
        _provide_geometry_defaults(args)

        # Map --eval-mode / --rc-mode aliases
        map_eval_mode_args(args)
        
        # Validate arguments
        validate_args(args)
        
        # Map deprecated --rc-mode -> --eval-mode
        if getattr(args, 'rc_mode', None) and not getattr(args, 'eval_mode', None):
            args.eval_mode = args.rc_mode
            print("[deprecation] --rc-mode is deprecated; use --eval-mode.", file=sys.stderr)
        if not getattr(args, 'eval_mode', None):
            args.eval_mode = args.rc_mode  # fallback to existing default
        
        # If using cases file, run cases and exit
        if args.cases:
            return run_cases(args.cases, str(args.outdir), plot=args.plot)
        
        # Normalize arguments and apply defaults for single case
        args = normalize_args(args)
        
        # Guardrail: if anyone mutated args.outdir after parse, restore it to the raw user input.
        if hasattr(args, "_outdir_user"):
            args.outdir = args._outdir_user
        
        # Concise, relevant summary
        _print_user_relevant_summary(args)
        
        # Run calculation
        result = run_calculation(args)
        
        # Print and save results
        print_results(result, args)
        # Extra safety: warn once if outdir was mutated mid-run
        if hasattr(args, "_outdir_user") and args.outdir != args._outdir_user:
            import sys
            print(f"[warning] --outdir was mutated from '{args._outdir_user}' to '{args.outdir}'. "
                  f"Using user value.", file=sys.stderr)
            args.outdir = args._outdir_user
        save_and_report_csv(result, args)

        # If grid/search produced a field, print the peak (y,z). Keep quiet for center.
        try:
            if getattr(args, 'eval_mode', None) in ("grid", "search"):
                Fpk = float(result.get('vf', result.get('F_peak', 0.0)))
                ypk = float(result.get('x_peak', result.get('y_peak', 0.0)))
                zpk = float(result.get('y_peak', result.get('z_peak', 0.0)))
                print(f"Peak VF = {Fpk:.6f} at (y,z) = ({ypk:.3f}, {zpk:.3f}) m")
        except Exception:
            pass
        
        # Generate plots if requested
        if args.plot:
            from .plotting import create_heatmap_plot
            from .util.paths import get_outdir
            # Re-assert outdir before any writers
            if hasattr(args, "_outdir_user"): args.outdir = args._outdir_user
            outdir = get_outdir(args.outdir)
            create_heatmap_plot(result, args, result.get('grid_data'))
            # Combined 2D geometry + heatmap (PNG) if requested
            if getattr(args, "plot_geom", "2d") in ("2d", "both"):
                try:
                    # Build receiver centre using the SAME offset logic used by the engine,
                    # so --emitter-offset (without --receiver-offset) works.
                    dy, dz = _resolve_offsets(args)  # receiver_center - emitter_center
                    from .viz.plots import plot_geometry_and_heatmap
                    from .util.filenames import join_with_ts
                    from .util.paths import get_outdir
                    if hasattr(args, "_outdir_user"): args.outdir = args._outdir_user
                    out_png = join_with_ts(get_outdir(args.outdir), "geom2d.png")
                    plot_geometry_and_heatmap(
                        result={**result, **{
                            "emitter_center": (float(args.setback), 0.0, 0.0),
                            "receiver_center": (0.0, float(dy), float(dz)),
                            "We": args.emitter[0], "He": args.emitter[1],
                            "Wr": args.receiver[0], "Hr": args.receiver[1],
                        }},
                        eval_mode=getattr(args, "eval_mode", getattr(args, "rc_mode", "center")),
                        method=args.method,
                        setback=float(args.setback),
                        out_png=out_png,
                    )
                    print(f"Combined geometry/heatmap saved to: {out_png}")
                except Exception as _e:
                    # Don't fail the run just because plotting failed
                    logger.warning(f"2D geometry/heatmap plot skipped: {_e}")
            # Interactive 3D (HTML) if requested
            if getattr(args, "plot_geom", "2d") in ("3d", "both"):
                try:
                    from .viz.plot3d import geometry_3d_html
                    from .util.filenames import join_with_ts
                    from .util.paths import get_outdir
                    if hasattr(args, "_outdir_user"): args.outdir = args._outdir_user
                    import numpy as _np
                    geom = result.get("geometry", {})
                    (We, He) = geom.get("emitter", args.emitter)
                    (Wr, Hr) = geom.get("receiver", args.receiver)
                    # Canonical centres (receiver at origin; emitter at +x=setback). Apply receiver offset dy,dz.
                    try:
                        dy, dz = _resolve_offsets(args)
                    except Exception:
                        dy = dz = 0.0
                    E = (float(args.setback), 0.0, 0.0)
                    R = (0.0, float(dy), float(dz))
                    out_html = str(join_with_ts(get_outdir(args.outdir), "geom3d.html"))
                    geometry_3d_html(
                        emitter_center=E, receiver_center=R,
                        We=We, He=He, Wr=Wr, Hr=Hr,
                        R_emitter=_np.eye(3), R_receiver=_np.eye(3),
                        out_html=out_html, include_plotlyjs="cdn"
                    )
                except ImportError as _ie:
                    logger.warning(str(_ie))
                except Exception as _e:
                    logger.warning(f"3D geometry plot skipped: {_e}")
        
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


def _resolve_output_directory(args):
    # Always return the resolved Path; never mutate args.outdir here.
    from .util.paths import get_outdir
    return get_outdir(args.outdir)


def _provide_geometry_defaults(args) -> None:
    """Provide missing geometry defaults for legacy callers."""
    if not hasattr(args, 'align_centres'):
        args.align_centres = False
    if not hasattr(args, 'receiver_offset'):
        args.receiver_offset = (0.0, 0.0)
    if not hasattr(args, 'eval_mode'):
        args.eval_mode = 'search'


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
