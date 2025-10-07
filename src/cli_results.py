"""
Result processing and output for the radiation view factor validation tool.

This module handles result printing, saving, and formatting,
following the Single Responsibility Principle.
"""

import os
import csv
import time
from typing import Dict, Any
import argparse
import logging

logger = logging.getLogger(__name__)


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
        print(f"Rotate axis: {args.rotate_axis}")
        print(f"Rotate target: {args.rotate_target}")
        print(f"Angle pivot: {args.angle_pivot}")
        
        # resolve centre-to-centre offsets (receiver - emitter)
        dy = dz = 0.0
        if args.align_centres:
            dy = dz = 0.0
        elif args.receiver_offset:
            dy, dz = args.receiver_offset
        elif args.emitter_offset:
            # convert to (receiver - emitter) convention
            dy, dz = (-args.emitter_offset[0], -args.emitter_offset[1])
        
        print(f"Receiver offset (receiver - emitter): ({dy:.3f}, {dz:.3f}) m  [y,z]")
        print(f"Align centres: {args.align_centres}")
    
    print(f"Output directory: {args.outdir}")
    print(f"Generate plots: {args.plot}")
    
    # Method-specific parameters
    _print_method_parameters(args)
    
    # resolve centre-to-centre offsets (receiver - emitter)
    dy = dz = 0.0
    if args.align_centres:
        dy = dz = 0.0
    elif args.receiver_offset:
        dy, dz = args.receiver_offset
    elif args.emitter_offset:
        # convert to (receiver - emitter) convention
        dy, dz = (-args.emitter_offset[0], -args.emitter_offset[1])

    print(f"\n--- Offsets & Orientation ---")
    print(f"Receiver offset (receiver - emitter): ({dy:.3f}, {dz:.3f}) m  [y,z]")
    print(f"Align centres: {args.align_centres}")
    print(f"Rotate target: {args.rotate_target}")
    print(f"Angle pivot: {args.angle_pivot}")

    print("\n--- Default Assumptions ---")
    print("- Surfaces face each other")
    print("- Centres are aligned")
    print("- Parallel orientation (unless angle specified)")
    print("="*50)


def _print_method_parameters(args: argparse.Namespace) -> None:
    """Print method-specific parameters."""
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


def print_single_line_summary(result: Dict[str, Any], args: argparse.Namespace) -> None:
    """Print a single-line summary of calculation results.
    
    Args:
        result: Calculation results dictionary
        args: Parsed command-line arguments
    """
    method = result['method']
    vf = result['vf']
    status = result.get('status', 'unknown')
    calc_time = result['calc_time']
    
    # Extract method-specific metrics
    if method == 'adaptive':
        iterations = result.get('search_metadata', {}).get('iterations', 0)
        depth = result.get('search_metadata', {}).get('depth', 0)
        cells = result.get('search_metadata', {}).get('cells', 0)
        achieved_tol = result.get('search_metadata', {}).get('achieved_tol', 0.0)
        print(f"[method={method}] vf={vf:.6f}, achieved_tol={achieved_tol:.3e}, status={status}, iterations={iterations}, cells={cells}, time={calc_time:.3f}s")
    elif method == 'fixedgrid':
        iterations = result.get('search_metadata', {}).get('evaluations', 0)
        samples_emitter = result.get('search_metadata', {}).get('samples_emitter', 0)
        samples_receiver = result.get('search_metadata', {}).get('samples_receiver', 0)
        print(f"[method={method}] vf={vf:.6f}, status={status}, iterations={iterations}, emitter_samples={samples_emitter}, receiver_samples={samples_receiver}, time={calc_time:.3f}s")
    elif method == 'montecarlo':
        samples = result.get('search_metadata', {}).get('samples', 0)
        ci95 = result.get('search_metadata', {}).get('ci95', 0.0)
        iterations = result.get('search_metadata', {}).get('iterations', 0)
        print(f"[method={method}] vf={vf:.6f}, ci95={ci95:.6f}, status={status}, samples={samples}, iterations={iterations}, time={calc_time:.3f}s")
    else:  # analytical
        iterations = result.get('search_metadata', {}).get('evaluations', 0)
        print(f"[method={method}] vf={vf:.6f}, status={status}, iterations={iterations}, time={calc_time:.3f}s")

    # NEW: For grid/search modes, always print the peak coordinates plainly
    rc_mode = result.get('rc_mode', 'center')
    if rc_mode in ('grid', 'search'):
        ypk = float(result.get('x_peak', 0.0))  # keys follow historical (x_peak,y_peak) naming
        zpk = float(result.get('y_peak', 0.0))
        print(f"Peak VF = {vf:.6f} at (y,z) = ({ypk:.3f}, {zpk:.3f}) m")


def print_results(result: Dict[str, Any], args: argparse.Namespace) -> None:
    """Print calculation results to console.
    
    Args:
        result: Calculation results dictionary
        args: Parsed command-line arguments
    """
    # Always print single-line summary
    print_single_line_summary(result, args)
    
    # Only print detailed results if verbose
    if args.verbose:
        print("\n" + "="*50)
        print("DETAILED RESULTS")
        print("="*50)
        
        method = result['method']
        vf = result['vf']
        calc_time = result['calc_time']
        
        print(f"Method: {method.title()}")
        
        # Use different labels based on eval_mode
        # Harmonise on eval_mode for display
        eval_mode = getattr(args, 'eval_mode', None) or result.get('rc_mode', 'center')
        if eval_mode == "center":
            print(f"View Factor at Selected Point: {vf:.8f}")
            print(f"Selected Location: ({float(result.get('x_peak', 0.0)):.3f}, {float(result.get('y_peak', 0.0)):.3f}) m")
        else:  # grid or search
            print(f"Local Peak View Factor: {vf:.8f}")
            print(f"Peak Location: ({float(result.get('x_peak', 0.0)):.3f}, {float(result.get('y_peak', 0.0)):.3f}) m")
        
        print(f"Eval Mode: {eval_mode}")
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
    """Save calculation results to CSV file.
    
    Args:
        result: Calculation results dictionary
        args: Parsed command-line arguments
    """
    # Ensure output directory exists and normalize path
    from .util.paths import get_outdir
    outdir = get_outdir(args.outdir)
    
    # Generate output filename based on method
    method = result['method']
    csv_filename = f"{method}.csv"
    csv_path = outdir / csv_filename
    
    # Extract geometry and peak information
    geom = result['geometry']
    em_w, em_h = geom['emitter']
    rc_w, rc_h = geom['receiver']
    setback = geom['setback']
    angle = geom['angle']
    vf = result['vf']
    x_peak = float(result.get('x_peak', 0.0))
    y_peak = float(result.get('y_peak', 0.0))
    rc_mode = result.get('rc_mode', 'center')
    status = result.get('status', 'unknown')
    
    # Extract search metadata
    search_metadata = result.get('search_metadata', {})
    evaluations = search_metadata.get('evaluations', 0)
    search_time = search_metadata.get('time_s', 0.0)
    
    # Write CSV with peak locator fields
    try:
        _write_csv_file(csv_path, method, em_w, em_h, rc_w, rc_h, setback, angle,
                       vf, search_metadata, search_time)
        print(f"\nResults saved to: {csv_path}")
        
    except PermissionError:
        _write_csv_file_fallback(method, em_w, em_h, rc_w, rc_h, setback, angle,
                                vf, search_metadata, search_time, outdir)


def _write_csv_file(csv_path: str, method: str, em_w: float, em_h: float, rc_w: float, 
                   rc_h: float, setback: float, angle: float, vf: float, 
                   search_metadata: Dict[str, Any], search_time: float) -> None:
    """Write CSV file with results."""
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header row - stable schema as per requirements
        writer.writerow([
            'method', 'emitter_w', 'emitter_h', 'receiver_w', 'receiver_h', 'setback', 'angle',
            'vf', 'vf_mean', 'ci95', 'status', 'iterations', 'samples', 'achieved_tol', 'time_s', 'cells'
        ])
        
        # Extract method-specific metadata
        iterations = search_metadata.get('iterations', '')
        samples = search_metadata.get('samples', '')
        achieved_tol = search_metadata.get('achieved_tol', '')
        cells = search_metadata.get('cells', '')
        
        # For Monte Carlo, use vf_mean and ci95; for other methods, use empty strings
        vf_mean = ''
        ci95 = ''
        
        # Data row - stable schema
        writer.writerow([
            method, em_w, em_h, rc_w, rc_h, setback, angle,
            f"{vf:.8f}", vf_mean, ci95, 'converged', iterations, samples, achieved_tol, f"{search_time:.3f}", cells
        ])


def _write_csv_file_fallback(method: str, em_w: float, em_h: float, rc_w: float, 
                            rc_h: float, setback: float, angle: float, vf: float, 
                            search_metadata: Dict[str, Any], search_time: float, outdir: str) -> None:
    """Write CSV file with timestamped filename as fallback."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fallback_filename = f"{method}_{timestamp}.csv"
    fallback_path = os.path.join(outdir, fallback_filename)
    
    with open(fallback_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header row - stable schema as per requirements
        writer.writerow([
            'method', 'emitter_w', 'emitter_h', 'receiver_w', 'receiver_h', 'setback', 'angle',
            'vf', 'vf_mean', 'ci95', 'status', 'iterations', 'samples', 'achieved_tol', 'time_s', 'cells'
        ])
        
        # Extract method-specific metadata
        iterations = search_metadata.get('iterations', '')
        samples = search_metadata.get('samples', '')
        achieved_tol = search_metadata.get('achieved_tol', '')
        cells = search_metadata.get('cells', '')
        
        # For Monte Carlo, use vf_mean and ci95; for other methods, use empty strings
        vf_mean = ''
        ci95 = ''
        
        # Data row - stable schema
        writer.writerow([
            method, em_w, em_h, rc_w, rc_h, setback, angle,
            f"{vf:.8f}", vf_mean, ci95, 'converged', iterations, samples, achieved_tol, f"{search_time:.3f}", cells
        ])
    
    print(f"\nResults saved to: {fallback_path}")
