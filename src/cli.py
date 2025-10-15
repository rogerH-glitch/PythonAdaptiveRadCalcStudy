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
from .util.plot_payload import attach_grid_field, attach_field_from_tap, has_field
from .util.grid_tap import drain as _drain_grid
from .util.offsets import get_receiver_offset
from .viz.field_sampler import sample_receiver_field

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


def _want_plots(args) -> tuple[bool, bool]:
    """Return (want_2d, want_3d) based on parsed args."""
    want_2d = bool(args.plot) or bool(getattr(args, 'plot_both', False))
    want_3d = bool(getattr(args, 'plot_3d', False)) or bool(getattr(args, 'plot_both', False))
    return want_2d, want_3d


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
    # Make orientation/offsets visible to plotting code
    result.update({
        'We': em_w, 'He': em_h, 'Wr': rc_w, 'Hr': rc_h,
        'emitter_center': (float(setback), 0.0, 0.0),
        'receiver_center': (0.0, float(dy), float(dz)),
        'R_emitter': [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
        'R_receiver': [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
        'rotate_axis': str(getattr(args, 'rotate_axis', 'z')),
        'rotate_target': str(getattr(args, 'rotate_target', 'emitter')),
        'angle_pivot': str(getattr(args, 'angle_pivot', 'toe')),
        'angle': float(getattr(args, 'angle', 0.0)),
        'dy': float(dy),
        'dz': float(dz),
    })
    
    # Attach grid field data to result if available (for grid/search modes)
    if getattr(args, "eval_mode", None) in ("grid", "search") or args.rc_mode in ("grid", "search"):
        # If grid_data contains Y,Z,F, extract and attach them
        if grid_data and isinstance(grid_data, dict):
            Y = grid_data.get("Y")
            Z = grid_data.get("Z")
            F = grid_data.get("F")
            # Attach to result if we have valid data
            if Y is not None and Z is not None and F is not None:
                attach_grid_field(result, Y, Z, F)
        # Also attach any tapped field captured during evaluation, regardless of plotting flags
        try:
            tapped = _drain_grid()
            if tapped is not None:
                Yt, Zt, Ft = tapped
                attach_grid_field(result, Yt, Zt, Ft)
        except Exception:
            pass
        # If still no field and in pure grid mode, generate a dense grid via the evaluator
        if args.rc_mode == 'grid' and not has_field(result):
            try:
                # Build fast-path point evaluator (no peak search)
                from .peak_locator import create_vf_evaluator
                # Override method params for fast grid evaluation (no peak search)
                fast_params = (method_params or {}).copy()
                fast_params.update({
                    'max_cells': 1500,
                    'max_iters': 40,
                    'rel_tol': 1e-2,  # Looser tolerance for speed
                    'abs_tol': 1e-4,
                    'peak_search': False,  # Disable peak search
                    'multistart': 1,       # Single evaluation per point
                })
                vf_evaluator = create_vf_evaluator(
                    args.method, em_w, em_h, rc_w, rc_h, setback, angle, geom_cfg, **fast_params
                )
                print("[grid] fast-path point evaluator active")
                import numpy as _np
                import time as _time
                from .eval.watchdog import check_time as _check_time, Timeout as _Timeout
                from .eval.cancel import CancelToken as _CancelToken
                ny = int(getattr(args, 'rc_grid_n', 41))
                nz = ny
                y_coords = _np.linspace(-rc_w/2.0, rc_w/2.0, ny)
                z_coords = _np.linspace(-rc_h/2.0, rc_h/2.0, nz)
                Yg, Zg = _np.meshgrid(y_coords, z_coords, indexing='xy')
                Fg = _np.empty_like(Yg, dtype=float)
                start_ts = _time.time()
                every_n = 25
                time_limit_s = float(getattr(args, 'rc_search_time_limit_s', 3.0) or 3.0)
                token = _CancelToken(timeout_s=time_limit_s)
                # Per-point time budget
                per_point_budget = max(0.003, 0.25 * time_limit_s / (ny * nz))
                try:
                    for j in range(Zg.shape[0]):
                        # Cooperative global timeout check
                        if token.expired():
                            print("[warn] grid timeout -> partial return")
                            Fg[j:, :] = float("nan")
                            break
                        for i in range(Yg.shape[1]):
                            # Use fast-path point evaluation with per-point budget
                            try:
                                vf_val, _meta = vf_evaluator(float(Yg[0, i]), float(Zg[j, 0]))
                            except Exception as _e:
                                # If point evaluation fails, use NaN
                                vf_val = float('nan')
                            Fg[j, i] = float(vf_val)
                            _check_time(start_ts, time_limit_s, every_n, j*Yg.shape[1] + i)
                except _Timeout as _e:
                    # Fill remainder with NaN and warn
                    Fg[j:, i:] = float('nan')
                    print(f"[warn] grid evaluation timed out: {_e}")
                attach_grid_field(result, Yg, Zg, Fg)
                # Also expose 1-D axes for downstream consumers
                result['grid_y'] = y_coords
                result['grid_z'] = z_coords
                # Ensure vf_field is available for compatibility
                result['vf_field'] = Fg
                
                # Generate denser heatmap grid if requested
                heatmap_n = getattr(args, 'heatmap_n', None)
                if heatmap_n is not None and heatmap_n > ny:
                    # Auto-downshift if budget too small
                    max_points = 1600 if time_limit_s <= 1.5 else 3600
                    if heatmap_n * heatmap_n > max_points:
                        heatmap_n = int(_np.sqrt(max_points))
                        print(f"[warn] reduced heatmap_n to {heatmap_n} due to time budget; values unchanged")
                    
                    if heatmap_n > ny:
                        print(f"[heatmap] generating denser grid: {heatmap_n}x{heatmap_n}")
                        # Use same fast evaluator for denser grid
                        hm_y_coords = _np.linspace(-rc_w/2.0, rc_w/2.0, heatmap_n)
                        hm_z_coords = _np.linspace(-rc_h/2.0, rc_h/2.0, heatmap_n)
                        hm_Yg, hm_Zg = _np.meshgrid(hm_y_coords, hm_z_coords, indexing='xy')
                        hm_Fg = _np.empty_like(hm_Yg, dtype=float)
                        
                        # Use same timeout and cooperative cancellation
                        hm_token = _CancelToken(timeout_s=time_limit_s)
                        hm_every_n = max(25, heatmap_n // 4)  # Check more frequently for larger grids
                        
                        try:
                            for j in range(hm_Zg.shape[0]):
                                if hm_token.expired():
                                    print("[warn] heatmap grid timeout -> partial return")
                                    hm_Fg[j:, :] = float("nan")
                                    break
                                for i in range(hm_Yg.shape[1]):
                                    try:
                                        vf_val, _meta = vf_evaluator(float(hm_Yg[0, i]), float(hm_Zg[j, 0]))
                                    except Exception as _e:
                                        vf_val = float('nan')
                                    hm_Fg[j, i] = float(vf_val)
                                    _check_time(start_ts, time_limit_s, hm_every_n, j*hm_Yg.shape[1] + i)
                        except _Timeout as _e:
                            hm_Fg[j:, i:] = float('nan')
                            print(f"[warn] heatmap grid evaluation timed out: {_e}")
                        
                        # Store denser grid for heatmap
                        result['heatmap_field'] = hm_Fg
                        result['heatmap_y'] = hm_y_coords
                        result['heatmap_z'] = hm_z_coords
            except Exception as _e:
                logger.debug("dense grid generation failed: %s", _e)
    
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

    # Orientation: print rotation details only when angle is non-zero
    has_angle = not _isclose(getattr(args, "angle", 0.0), 0.0)
    if has_angle:
        print(f"Angle: {getattr(args,'angle',0.0):.1f}°")
        print(f"Rotate axis: {getattr(args,'rotate_axis','z')}")
        print(f"Rotate target: {getattr(args,'rotate_target','emitter')}")
        print(f"Angle pivot: {getattr(args,'angle_pivot','toe')}")

    # Offsets: prefer printing only the one the user provided and only if non-zero
    r_off = getattr(args, "receiver_offset", None)
    e_off = getattr(args, "emitter_offset", None)
    if e_off and (not _isclose(e_off[0], 0.0) or not _isclose(e_off[1], 0.0)):
        print(f"Emitter offset (dy,dz): ({e_off[0]:.3f}, {e_off[1]:.3f}) m")
    elif r_off and (not _isclose(r_off[0], 0.0) or not _isclose(r_off[1], 0.0)):
        print(f"Receiver offset (dy,dz): ({r_off[0]:.3f}, {r_off[1]:.3f}) m")

    # Output dir & plotting flags (concise)
    # Print exactly what the user provided, not any internal resolved path
    outdir_raw = getattr(args, "_outdir_user", args.outdir)
    print(f"Output directory: {outdir_raw}")
    plot_mode = getattr(args, "_plot_mode", "none")
    if plot_mode != "none":
        print(f"Generate plots: True (mode={plot_mode})")


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

        # Check if we already have field data from run_calculation
        if getattr(args, "eval_mode", None) in ("grid", "search"):
            if has_field(result):
                F = result["F"]
                print(f"[plot] using dense grid field: shape={getattr(F, 'shape', None)}")
                
                # Generate denser heatmap grid if requested
                heatmap_n = getattr(args, 'heatmap_n', None)
                if heatmap_n is not None and hasattr(F, 'shape') and len(F.shape) == 2:
                    current_n = F.shape[0]  # Assuming square grid
                    if heatmap_n > current_n:
                        # Auto-downshift if budget too small
                        time_limit_s = float(getattr(args, 'rc_search_time_limit_s', 3.0) or 3.0)
                        max_points = 1600 if time_limit_s <= 1.5 else 3600
                        if heatmap_n * heatmap_n > max_points:
                            import numpy as np
                            heatmap_n = int(np.sqrt(max_points))
                            print(f"[warn] reduced heatmap_n to {heatmap_n} due to time budget; values unchanged")
                        
                        if heatmap_n > current_n:
                            print(f"[heatmap] generating denser grid: {heatmap_n}x{heatmap_n}")
                            # Generate denser heatmap grid using the same fast evaluator
                            try:
                                from .peak_locator import create_vf_evaluator
                                from .eval.cancel import CancelToken as _CancelToken
                                from .eval.watchdog import check_time as _check_time, Timeout as _Timeout
                                import numpy as _np
                                import time as _time
                                
                                # Get geometry parameters
                                em_w, em_h = args.emitter
                                rc_w, rc_h = args.receiver
                                setback = float(args.setback)
                                angle = float(getattr(args, 'angle', 0.0))
                                
                                # Create fast evaluator (same as in run_calculation)
                                method_params = _create_method_params(args)
                                fast_params = (method_params or {}).copy()
                                fast_params.update({
                                    'max_cells': 1500,
                                    'max_iters': 40,
                                    'rel_tol': 1e-2,
                                    'abs_tol': 1e-4,
                                    'peak_search': False,
                                    'multistart': 1,
                                })
                                
                                # Get geometry config
                                geom_cfg = result.get('geometry', {})
                                vf_evaluator = create_vf_evaluator(
                                    args.method, em_w, em_h, rc_w, rc_h, setback, angle, geom_cfg, **fast_params
                                )
                                
                                # Generate denser grid
                                hm_y_coords = _np.linspace(-rc_w/2.0, rc_w/2.0, heatmap_n)
                                hm_z_coords = _np.linspace(-rc_h/2.0, rc_h/2.0, heatmap_n)
                                hm_Yg, hm_Zg = _np.meshgrid(hm_y_coords, hm_z_coords, indexing='xy')
                                hm_Fg = _np.empty_like(hm_Yg, dtype=float)
                                
                                # Use same timeout and cooperative cancellation
                                hm_token = _CancelToken(timeout_s=time_limit_s)
                                hm_every_n = max(25, heatmap_n // 4)
                                start_ts = _time.time()
                                
                                try:
                                    for j in range(hm_Zg.shape[0]):
                                        if hm_token.expired():
                                            print("[warn] heatmap grid timeout -> partial return")
                                            hm_Fg[j:, :] = float("nan")
                                            break
                                        for i in range(hm_Yg.shape[1]):
                                            try:
                                                vf_val, _meta = vf_evaluator(float(hm_Yg[0, i]), float(hm_Zg[j, 0]))
                                            except Exception as _e:
                                                vf_val = float('nan')
                                            hm_Fg[j, i] = float(vf_val)
                                            _check_time(start_ts, time_limit_s, hm_every_n, j*hm_Yg.shape[1] + i)
                                except _Timeout as _e:
                                    hm_Fg[j:, i:] = float('nan')
                                    print(f"[warn] heatmap grid evaluation timed out: {_e}")
                                
                                # Store denser grid for heatmap
                                result['heatmap_field'] = hm_Fg
                                result['heatmap_y'] = hm_y_coords
                                result['heatmap_z'] = hm_z_coords
                                print(f"[heatmap] generated denser grid: shape={hm_Fg.shape}")
                            except Exception as _e:
                                logger.debug("denser heatmap grid generation failed: %s", _e)
            else:
                # Try to get field from tap as fallback
                tapped = _drain_grid()
                if tapped is not None:
                    Y, Z, F = tapped
                    try:
                        attach_grid_field(result, Y, Z, F)
                        print(f"[plot] using captured field: shape={getattr(F, 'shape', None)}")
                    except Exception as _e:
                        logger.debug("attach_grid_field skipped: %s", _e)
                else:
                    # Fallback sampler (plotting only)
                    try:
                        result = sample_receiver_field(args, result)
                        if has_field(result):
                            F = result["F"]
                            print(f"[plot] sampled coarse receiver field: shape={getattr(F,'shape',None)}")
                        else:
                            print("[plot] note: no receiver field was captured and sampler unavailable; heat-map may be empty")
                    except Exception as e:
                        logger.debug("coarse sampler failed: %s", e)
        
        # Print and save results
        print_results(result, args)
        # Extra safety: warn once if outdir was mutated mid-run
        if hasattr(args, "_outdir_user") and args.outdir != args._outdir_user:
            import sys
            print(f"[warning] --outdir was mutated from '{args._outdir_user}' to '{args.outdir}'. "
                  f"Using user value.", file=sys.stderr)
            args.outdir = args._outdir_user
        save_and_report_csv(result, args)

        # Build a single structured console summary (presentation only)
        try:
            # Geometry/eval metadata
            method = str(getattr(args, 'method', result.get('method', 'adaptive')))
            yaw = float(getattr(args, 'angle', result.get('angle', 0.0)))
            pitch = float(getattr(args, 'pitch', result.get('pitch', 0.0) or 0.0))
            pivot = str(getattr(args, 'angle_pivot', result.get('angle_pivot', 'toe')))
            target = str(getattr(args, 'rotate_target', result.get('rotate_target', 'emitter')))

            # Offsets
            dy, dz = _resolve_offsets(args)
            setback = float(getattr(args, 'setback', result.get('setback', 0.0)))

            # Grid usage
            used_ny = used_nz = None
            base_n = int(getattr(args, 'rc_grid_n', result.get('rc_grid_n', 0)) or 0)
            # Prefer denser heatmap grid when present
            if isinstance(result.get('heatmap_field'), (list, tuple, np.ndarray)):
                F_used = np.asarray(result.get('heatmap_field'))
                if F_used.ndim == 2:
                    used_nz, used_ny = F_used.shape  # note: imshow used F.T; report as ny x nz
            elif isinstance(result.get('F'), (list, tuple, np.ndarray)):
                F_used = np.asarray(result.get('F'))
                if F_used.ndim == 2:
                    used_nz, used_ny = F_used.shape

            # Peaks: adaptive (scalar) and grid-argmax when field present
            F_adapt = float(result.get('vf', result.get('F_peak', np.nan)))
            y_adapt = float(result.get('x_peak', result.get('y_peak', np.nan)))
            z_adapt = float(result.get('y_peak', result.get('z_peak', np.nan)))

            grid_peak_str = None
            if isinstance(result.get('vf_field'), (list, tuple, np.ndarray)) or isinstance(result.get('F'), (list, tuple, np.ndarray)):
                Fg = np.asarray(result.get('vf_field') if result.get('vf_field') is not None else result.get('F'))
                gy = np.asarray(result.get('grid_y') if result.get('grid_y') is not None else result.get('heatmap_y', []))
                gz = np.asarray(result.get('grid_z') if result.get('grid_z') is not None else result.get('heatmap_z', []))
                try:
                    jj, ii = np.unravel_index(np.nanargmax(Fg), Fg.shape)
                    # Map to physical coordinates; ii indexes y, jj indexes z
                    y_star = float(gy[ii]) if gy.size else float('nan')
                    z_star = float(gz[jj]) if gz.size else float('nan')
                    Fg_pk = float(np.nanmax(Fg))
                    grid_peak_str = f"grid F≈{Fg_pk:.3f} at (y,z)=({y_star:.2f}, {z_star:.2f})"
                except Exception:
                    grid_peak_str = None

            # Solver status (best-effort)
            status = str(result.get('status', 'converged'))
            tol = result.get('achieved_tol', result.get('tol', None))
            iters = result.get('iterations', result.get('iters', None))
            cells = result.get('cells', None)
            wall_s = result.get('time', result.get('elapsed_s', None))

            # Artifact paths (if generated)
            artifacts = []
            # We rely on previous local variables if set
            try:
                if 'out_png' in locals():
                    artifacts.append(f"2D={out_png}")
            except Exception:
                pass
            try:
                if 'out_html' in locals():
                    artifacts.append(f"3D={out_html}")
            except Exception:
                pass
            # CSV path (conventional)
            artifacts.append("CSV=results/adaptive.csv")

            # Print summary lines
            print(f"[eval] method={method} | yaw={yaw:.0f}° pitch={pitch:.0f}° | pivot={pivot} target={target}")
            print(f"[geom] setback={setback:.3f} m | offset(dy,dz)=({dy:.3f},{dz:.3f}) m")
            if used_ny and used_nz:
                if base_n and (used_ny != base_n or used_nz != base_n):
                    print(f"[grid] used={used_ny}x{used_nz} (base={base_n}x{base_n})")
                elif base_n:
                    print(f"[grid] used={used_ny}x{used_nz} (base={base_n}x{base_n})")
                else:
                    print(f"[grid] used={used_ny}x{used_nz}")
            # Peak line
            if grid_peak_str:
                print(f"[peak] adaptive F={F_adapt:.6f} at (y,z)=({y_adapt:.3f}, {z_adapt:.3f}) | {grid_peak_str}")
            else:
                print(f"[peak] adaptive F={F_adapt:.6f} at (y,z)=({y_adapt:.3f}, {z_adapt:.3f})")
            # Status line
            tol_str = (f"{float(tol):.2e}" if tol is not None else "n/a")
            it_str = (str(int(iters)) if iters is not None else "n/a")
            ce_str = (str(int(cells)) if cells is not None else "n/a")
            tm_str = (f"{float(wall_s):.2f} s" if wall_s is not None else "n/a")
            print(f"[status] {status} tol={tol_str} iters={it_str} cells={ce_str} time={tm_str}")
            # Artifacts
            if artifacts:
                print(f"[artifacts] {' | '.join(artifacts)}")
        except Exception:
            # Never fail the run due to summary printing
            pass
        
        # Generate plots if requested
        if args.plot:
            from .util.paths import get_outdir
            # Re-assert outdir before any writers
            if hasattr(args, "_outdir_user"): args.outdir = args._outdir_user
            outdir = get_outdir(args.outdir)
        # Determine what plots to generate
        want_2d, want_3d = _want_plots(args)
        
        if want_2d or want_3d:
            # Always try to attach field data from tap first
            result = attach_field_from_tap(result)
            
            # If no field data, try fallback sampler
            if not has_field(result):
                try:
                    result = sample_receiver_field(args, result)
                    if has_field(result):
                        F = result["F"]
                        print(f"[plot] using fallback sampled field: shape={getattr(F, 'shape', None)}")
                    else:
                        print("[plot] note: no field data available; heatmap may be empty")
                except Exception as e:
                    logger.debug("fallback sampler failed: %s", e)
                    print("[plot] note: no field data available; heatmap may be empty")
        
        # 2-D plotting when requested
        if want_2d:
            try:
                # Build receiver centre using the SAME offset logic used by the engine,
                # so --emitter-offset (without --receiver-offset) works.
                dy, dz = _resolve_offsets(args)  # receiver_center - emitter_center
                from .viz.plots import plot_geometry_and_heatmap
                from .util.filenames import join_with_ts
                from .util.paths import get_outdir
                raw = getattr(args, "_outdir_user", args.outdir)
                out_png = join_with_ts(get_outdir(raw), "geom2d.png")
                # Use denser heatmap grid if available, otherwise fall back to regular grid
                heatmap_field = result.get("heatmap_field")
                if heatmap_field is not None:
                    vf_field = heatmap_field
                    vf_grid = {"y": result.get("heatmap_y"), "z": result.get("heatmap_z")}
                else:
                    vf_field = result.get("F", None)
                    vf_grid = {"y": result.get("grid_y"), "z": result.get("grid_z")}
                
                plot_geometry_and_heatmap(
                    result={**result, **{
                        "emitter_center": (float(args.setback), 0.0, 0.0),
                        "receiver_center": (0.0, float(dy), float(dz)),
                        "We": args.emitter[0], "He": args.emitter[1],
                        "Wr": args.receiver[0], "Hr": args.receiver[1],
                        "rotate_axis": getattr(args, "rotate_axis", "z"),
                        "rotate_target": getattr(args, "rotate_target", "emitter"),
                        "angle_pivot": getattr(args, "angle_pivot", "toe"),
                        "angle": float(getattr(args, "angle", 0.0)),
                    }},
                    eval_mode=getattr(args, "eval_mode", getattr(args, "rc_mode", "center")),
                    method=args.method,
                    setback=float(args.setback),
                    out_png=out_png,
                    # Use denser heatmap grid if available
                    vf_field=vf_field,
                    vf_grid=vf_grid,
                    prefer_eval_field=True,
                    heatmap_interp=getattr(args, "heatmap_interp", "bilinear"),
                    marker_mode=getattr(args, "heatmap_marker", "both"),
                    adaptive_peak_yz=(
                        float(result.get('x_peak', result.get('y_peak', 0.0))),
                        float(result.get('y_peak', result.get('z_peak', 0.0)))
                    ),
                    subcell_fit=True,
                    debug_plots=bool(getattr(args, 'debug_plots', False)),
                )
                print(f"Combined geometry/heatmap saved to: {out_png}")
            except Exception as _e:
                # Don't fail the run just because plotting failed
                logger.warning(f"2D geometry/heatmap plot skipped: {_e}")
        
        # 3-D plotting when requested
        if want_3d:
            try:
                from .viz.plot3d import plot_geometry_3d
                from .util.filenames import join_with_ts
                from .util.paths import get_outdir
                raw = getattr(args, "_outdir_user", args.outdir)
                out_html = str(join_with_ts(get_outdir(raw), "geom3d.html"))
                plot_geometry_3d(result, out_html, debug_plots=bool(getattr(args, 'debug_plots', False)))
                print(f"3-D interactive plot saved to: {out_html}")
            except Exception as _e:
                logger.warning(f"3D plot skipped: {_e}")
        
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
