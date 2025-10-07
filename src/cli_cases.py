"""
Test cases execution for the radiation view factor validation tool.

This module handles running validation cases from YAML files,
following the Single Responsibility Principle.
"""

import os
import csv
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def run_cases(cases_path: str, outdir: str, plot: bool = False) -> int:
    """Run validation cases from YAML file and generate summary CSV.
    
    Args:
        cases_path: Path to YAML cases file
        outdir: Output directory for results
        plot: Whether to generate plots
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    from .io_yaml import load_cases, validate_case_schema, coerce_case_to_cli_kwargs
    from .peak_locator import find_local_peak, create_vf_evaluator
    
    try:
        cases = load_cases(cases_path)
        os.makedirs(outdir, exist_ok=True)
        
        # Normalize output directory and create plots directory if plotting is requested
        from .util.paths import get_outdir
        outdir_path = get_outdir(outdir)
        
        plots_dir = None
        if plot:
            plots_dir = outdir_path / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
        
        summary_csv = outdir_path / "cases_summary.csv"
        
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Enhanced CSV headers
            writer.writerow([
                "id", "method", "vf", "ci95", "expected", "rel_err", "status", 
                "iterations", "achieved_tol", "validation", "validation_rel_err", 
                "ref_analytical", "rel_err_to_ref", "attempts", 
                "vf_point_center", "vf_receiver_avg", "compare_to", "avg_gt_center", "notes"
            ])
            
            for case in cases:
                try:
                    validate_case_schema(case)
                    if not case["enabled"]:
                        writer.writerow([
                            case.get("id"), case.get("method","adaptive"), "", "", "", "", 
                            "skipped", "", "", "", "", "", "", "", "", "", "", "", "disabled"
                        ])
                        continue
                    
                    _process_single_case(case, writer, plot, plots_dir)
                except Exception as e:
                    writer.writerow([
                        case.get("id","<unknown>"), case.get("method","adaptive"), 
                        "", "", "", "", "failed", "", "", "", "", "", "", "", "", "", "", "", str(e)
                    ])
                    continue
        
        print(f"Wrote: {summary_csv}")
        if plot:
            print(f"Plots saved to: {plots_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Error running cases: {e}")
        return 1


def _process_single_case(case: Dict[str, Any], writer: csv.writer, plot: bool, plots_dir: Optional[str]) -> None:
    """Process a single test case and write results to CSV.
    
    Args:
        case: Test case dictionary
        writer: CSV writer object
        plot: Whether to generate plots
        plots_dir: Directory for plot files
    """
    from .io_yaml import coerce_case_to_cli_kwargs
    from .peak_locator import find_local_peak, create_vf_evaluator
    
    kw = coerce_case_to_cli_kwargs(case)
    
    # Extract geometry and method parameters
    em_w, em_h = kw["emitter"]
    rc_w, rc_h = kw["receiver"]
    setback = kw["setback"]
    angle = kw["angle"]
    method = kw["method"]
    
    # Apply method-specific overrides from case
    overrides = kw.get("overrides", {})
    
    # Create method-specific evaluator
    method_params = _create_method_params(overrides)
    
    # Create geometry configuration (defaults for cases)
    geom_cfg = _create_geometry_config(em_w, em_h, setback, angle)
    
    # Create evaluator function
    vf_evaluator = create_vf_evaluator(
        method, em_w, em_h, rc_w, rc_h, setback, angle, geom_cfg, **method_params
    )
    
    # Find local peak (use center mode for cases)
    start_time = time.time()
    peak_result = find_local_peak(
        em_w, em_h, rc_w, rc_h, setback, angle, vf_evaluator,
        rc_mode='center'  # Use center mode for batch processing
    )
    calc_time = time.time() - start_time
    
    # Extract results
    vf = peak_result['vf_peak']
    status = peak_result['status']
    iterations = peak_result.get('search_metadata', {}).get('iterations', 0)
    achieved_tol = peak_result.get('search_metadata', {}).get('achieved_tol', 0.0)
    
    # For Monte Carlo, extract CI if available
    ci95 = ""
    if method == 'montecarlo' and 'ci95' in peak_result:
        ci95 = f"{peak_result['ci95']:.6f}"
    
    # Extract peak coordinates
    x_peak = peak_result.get('x_peak', 0.0)
    y_peak = peak_result.get('y_peak', 0.0)
    
    # Compare against expected if present
    expected = kw.get("expected")
    rel_err = ""
    if expected is not None and expected != "":
        rel_err = f"{abs(vf - expected)/expected:.6f}" if expected != 0 else ""
    
    # Generate plot if requested
    plot_filename = ""
    if plot and plots_dir is not None:
        plot_filename = _generate_case_plot(case, kw, peak_result, plots_dir, method, method_params)
    
    # Analytical cross-check (diagnostics only)
    ref_analytical, rel_err_to_ref = _calculate_analytical_reference(em_w, em_h, setback, angle, vf, expected)
    
    # Point vs area-average diagnostics
    vf_point_center, vf_receiver_avg, avg_gt_center, compare_to = _calculate_area_average_diagnostics(
        em_w, em_h, rc_w, rc_h, setback, angle, vf, expected, case
    )
    
    # Auto-retry on validation failure (bounded)
    attempts = _attempt_retry_if_needed(
        case, expected, rel_err, status, method, method_params,
        em_w, em_h, rc_w, rc_h, setback, angle, geom_cfg
    )
    
    # Calculate validation status using correct comparison value
    validation, validation_rel_err = _calculate_validation_status(
        expected, compare_to, vf_receiver_avg, vf_point_center
    )
    
    # Write results to CSV
    _write_case_results(
        writer, kw, method, vf, ci95, expected, rel_err, status,
        iterations, achieved_tol, validation, validation_rel_err,
        ref_analytical, rel_err_to_ref, attempts,
        vf_point_center, vf_receiver_avg, compare_to, avg_gt_center,
        calc_time, x_peak, y_peak, plot_filename
    )


def _create_method_params(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Create method parameters from overrides."""
    return {
        'analytical_nx': overrides.get('analytical_nx', 240),
        'analytical_ny': overrides.get('analytical_ny', 240),
        'rel_tol': overrides.get('rel_tol', 3e-3),
        'abs_tol': overrides.get('abs_tol', 1e-6),
        'max_depth': overrides.get('max_depth', 12),
        'max_cells': overrides.get('max_cells', 200000),
        'min_cell_area_frac': overrides.get('min_cell_area_frac', 1e-8),
        'init_grid': overrides.get('init_grid', '4x4'),
        'grid_nx': overrides.get('grid_nx', 100),
        'grid_ny': overrides.get('grid_ny', 100),
        'quadrature': overrides.get('quadrature', 'centroid'),
        'samples': overrides.get('samples', 200000),
        'target_rel_ci': overrides.get('target_rel_ci', 0.02),
        'max_iters': overrides.get('max_iters', 50),
        'seed': overrides.get('seed', 42),
        'time_limit_s': overrides.get('time_limit_s', 60.0)
    }


def _create_geometry_config(em_w: float, em_h: float, setback: float, angle: float) -> Dict[str, Any]:
    """Create geometry configuration for cases."""
    return {
        "emitter_offset": (0.0, 0.0),
        "receiver_offset": (0.0, 0.0),
        "angle_deg": float(angle),
        "angle_pivot": "toe",
        "rotate_target": "emitter",
        # Augmented fields for orientation-aware evaluators
        "emitter_width": float(em_w),
        "emitter_height": float(em_h),
        "setback": float(setback),
        "rotate_axis": "z",
        "angle": float(angle),
        "dy": 0.0,
        "dz": 0.0,
    }


def _generate_case_plot(case: Dict[str, Any], kw: Dict[str, Any], peak_result: Dict[str, Any], 
                       plots_dir: str, method: str, method_params: Dict[str, Any]) -> str:
    """Generate plot for a test case."""
    try:
        from .plotting import create_heatmap_plot, generate_grid_data_for_plotting
        
        # Generate grid data for plotting
        grid_data = generate_grid_data_for_plotting(
            kw["emitter"][0], kw["emitter"][1], kw["receiver"][0], kw["receiver"][1], 
            kw["setback"], kw["angle"], method, method_params, 21  # Use 21x21 grid for plots
        )
        
        # Create result dict for plotting
        result = {
            'method': method,
            'vf': peak_result['vf_peak'],
            'x_peak': peak_result['x_peak'],
            'y_peak': peak_result['y_peak'],
            'rc_mode': 'center',
            'status': peak_result['status'],
            'geometry': {
                'emitter': (kw["emitter"][0], kw["emitter"][1]),
                'receiver': (kw["receiver"][0], kw["receiver"][1]),
                'setback': kw["setback"],
                'angle': kw["angle"]
            },
            'grid_data': grid_data
        }
        
        # Create mock args for plotting
        class MockArgs:
            def __init__(self):
                self.outdir = str(plots_dir) if plots_dir else str(outdir_path)
                self.plot = True
                # Provide fields expected by legacy/new plotting helpers
                self.method = method
                self.setback = kw["setback"]
                # For titles: prefer eval_mode if present
                self.eval_mode = 'center'
                self.rc_mode = 'center'
        
        mock_args = MockArgs()
        plot_filename = f"{kw['id']}_{method}.png"
        create_heatmap_plot(result, mock_args, grid_data)
        
        # Rename the plot file to use case-specific name
        old_plot_path = os.path.join(plots_dir, f"{method}_peak_heatmap.png")
        new_plot_path = os.path.join(plots_dir, plot_filename)
        if os.path.exists(old_plot_path):
            os.rename(old_plot_path, new_plot_path)
        
        return plot_filename
        
    except Exception as plot_error:
        logger.warning(f"Failed to generate plot for case {kw['id']}: {plot_error}")
        return ""


def _calculate_analytical_reference(em_w: float, em_h: float, setback: float, angle: float, 
                                  vf: float, expected: Any) -> tuple[str, str]:
    """Calculate analytical reference for comparison."""
    ref_analytical = ""
    rel_err_to_ref = ""
    if expected is not None and expected > 0 and angle == 0:
        # Only for parallel, concentric cases
        try:
            from .analytical import vf_point_rect_to_point_parallel
            F_ref = vf_point_rect_to_point_parallel(em_w, em_h, setback, rx=0.0, ry=0.0, nx=420, ny=420)
            ref_analytical = f"{F_ref:.8f}"
            rel_err_to_ref = f"{abs(vf - F_ref)/max(F_ref, 1e-12):.6f}"
        except Exception:
            pass  # Skip if analytical fails
    return ref_analytical, rel_err_to_ref


def _calculate_area_average_diagnostics(em_w: float, em_h: float, rc_w: float, rc_h: float, 
                                      setback: float, angle: float, vf: float, 
                                      expected: Any, case: Dict[str, Any]) -> tuple[float, str, bool, str]:
    """Calculate area average diagnostics."""
    vf_point_center = vf  # The solver's returned point value at rc=(0,0)
    vf_receiver_avg = ""
    avg_gt_center = False
    compare_to = "point"  # Default
    
    if angle == 0:  # Only for parallel cases
        try:
            from .validators import receiver_area_average_point_integrand
            vf_receiver_avg = receiver_area_average_point_integrand(
                em_w, em_h, rc_w, rc_h, setback, angle_deg=0.0, 
                nx_rc=45, ny_rc=19, nx_em=180, ny_em=180
            )
            avg_gt_center = vf_receiver_avg > vf_point_center * 1.005  # >0.5% higher
        except Exception:
            pass  # Skip if area-average computation fails
    
    # Determine comparison type from YAML
    if expected is not None:
        compare_to = case.get("expected", {}).get("type", "point")
    
    return vf_point_center, vf_receiver_avg, avg_gt_center, compare_to


def _attempt_retry_if_needed(case: Dict[str, Any], expected: Any, rel_err: str, status: str, 
                           method: str, method_params: Dict[str, Any], em_w: float, em_h: float, 
                           rc_w: float, rc_h: float, setback: float, angle: float, 
                           geom_cfg: Dict[str, Any]) -> int:
    """Attempt retry if validation fails."""
    attempts = 1
    if expected is not None and expected > 0:
        tolerance_value = case.get("expected", {}).get("tolerance", {}).get("value", 0.01)
        rel_err_float = float(rel_err) if rel_err else 0.0
        
        if rel_err_float > tolerance_value and status != "failed" and method == "adaptive":
            # Build stricter config for retry
            retry_params = method_params.copy()
            retry_params['rel_tol'] = min(retry_params.get('rel_tol', 3e-3), 1e-3)
            retry_params['init_grid'] = "8x8"  # Bump to at least 8x8
            retry_params['max_cells'] = max(retry_params.get('max_cells', 200000), 100000)
            retry_params['max_depth'] = max(retry_params.get('max_depth', 12), 14)
            retry_params['time_limit_s'] = min(retry_params.get('time_limit_s', 60) + 5, 120)
            retry_params['min_cells'] = max(retry_params.get('min_cells', 16), 64)
            
            try:
                from .peak_locator import create_vf_evaluator, find_local_peak
                # Create retry evaluator with stricter settings
                retry_evaluator = create_vf_evaluator(
                    method, em_w, em_h, rc_w, rc_h, setback, angle, geom_cfg, **retry_params
                )
                
                # Re-run with stricter settings
                retry_result = find_local_peak(
                    em_w, em_h, rc_w, rc_h, setback, angle, retry_evaluator,
                    rc_mode="center"
                )
                
                if retry_result['status'] != "failed":
                    retry_vf = retry_result['vf_peak']
                    retry_rel_err = abs(retry_vf - expected) / expected if expected != 0 else 0
                    
                    # If retry improves, use retry result
                    if retry_rel_err < rel_err_float:
                        attempts = 2
                        
            except Exception:
                pass  # Keep original result if retry fails
    
    return attempts


def _calculate_validation_status(expected: Any, compare_to: str, vf_receiver_avg: str, 
                               vf_point_center: float) -> tuple[str, str]:
    """Calculate validation status."""
    validation = ""
    validation_rel_err = ""
    if expected is not None and expected > 0:
        # Choose comparison value based on YAML type
        if compare_to == "area_avg" and vf_receiver_avg != "":
            cmp_val = vf_receiver_avg
        else:
            cmp_val = vf_point_center
        
        # Compute relative error against expected
        rel_err_float = abs(cmp_val - expected) / expected if expected != 0 else 0
        validation_rel_err = f"{rel_err_float:.6f}"
        tolerance_value = 0.01  # Default tolerance
        
        if rel_err_float <= tolerance_value:
            validation = "pass"
        else:
            validation = "out_of_spec"
    
    return validation, validation_rel_err


def _write_case_results(writer: csv.writer, kw: Dict[str, Any], method: str, vf: float, 
                       ci95: str, expected: Any, rel_err: str, status: str, iterations: int, 
                       achieved_tol: float, validation: str, validation_rel_err: str,
                       ref_analytical: str, rel_err_to_ref: str, attempts: int,
                       vf_point_center: float, vf_receiver_avg: str, compare_to: str, 
                       avg_gt_center: bool, calc_time: float, x_peak: float, y_peak: float, 
                       plot_filename: str) -> None:
    """Write case results to CSV."""
    # Calculate notes
    x_peak_float = float(x_peak) if x_peak is not None else 0.0
    y_peak_float = float(y_peak) if y_peak is not None else 0.0
    notes = f"calc_time={calc_time:.3f}s, rc=({x_peak_float:.3f},{y_peak_float:.3f})"
    if plot_filename:
        notes += f", plot={plot_filename}"
    
    # Add warning if area-average is significantly higher than center
    if avg_gt_center:
        notes += ", WARN: avg>center; check geometry/expected type"
    
    writer.writerow([
        kw["id"], method, f"{vf:.8f}", ci95, 
        expected if expected is not None else "", rel_err, status,
        iterations, achieved_tol, validation, validation_rel_err,
        ref_analytical, rel_err_to_ref, attempts,
        f"{vf_point_center:.8f}", f"{vf_receiver_avg:.8f}" if vf_receiver_avg != "" else "",
        compare_to, avg_gt_center, notes
    ])
