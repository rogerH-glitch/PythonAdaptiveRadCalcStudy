"""
Local peak view factor locator for general geometries.

This module implements a coarse-to-fine search algorithm to find the location
on the receiver where the point view factor is maximized, supporting non-concentric,
rotated, and future occluded geometries.
"""

from __future__ import annotations
import time
import math
import numpy as np
from typing import Dict, Tuple, List, Callable, Optional
from scipy.optimize import minimize
from .constants import EPS, STATUS_CONVERGED, STATUS_REACHED_LIMITS, STATUS_FAILED


def find_local_peak(
    em_w: float, em_h: float, rc_w: float, rc_h: float, setback: float, angle: float,
    vf_evaluator: Callable[[float, float], Tuple[float, Dict]], 
    rc_mode: str = "center",
    rc_grid_n: int = 21,
    rc_search_rel_tol: float = 3e-3,
    rc_search_max_iters: int = 200,
    rc_search_multistart: int = 8,
    rc_search_time_limit_s: float = 10.0,
    rc_bounds: str = "auto"
) -> Dict:
    """
    Find the local peak view factor location on the receiver.
    
    Args:
        em_w, em_h: Emitter dimensions
        rc_w, rc_h: Receiver dimensions  
        setback: Setback distance
        angle: Rotation angle (degrees)
        vf_evaluator: Function that evaluates F(x,y) at a receiver point
        rc_mode: Search mode ("center", "grid", "search")
        rc_grid_n: Grid resolution for coarse sampling
        rc_search_rel_tol: Relative improvement tolerance
        rc_search_max_iters: Max local optimizer iterations
        rc_search_multistart: Number of multi-start seeds
        rc_search_time_limit_s: Time limit for search phase
        rc_bounds: Bounds mode ("auto" or "explicit")
        
    Returns:
        Dictionary with peak location, value, and search metadata
    """
    start_time = time.perf_counter()
    
    # Input validation
    if em_w <= 0 or em_h <= 0 or rc_w <= 0 or rc_h <= 0 or setback <= 0:
        return {
            "x_peak": 0.0,
            "y_peak": 0.0, 
            "vf_peak": 0.0,
            "status": STATUS_FAILED,
            "search_metadata": {"time_s": 0.0, "iterations": 0, "evaluations": 0}
        }
    
    if rc_mode == "center":
        # Fast path: assume peak at center for concentric parallel case
        x_peak, y_peak = 0.0, 0.0
        vf_peak, metadata = vf_evaluator(x_peak, y_peak)
        
        return {
            "x_peak": x_peak,
            "y_peak": y_peak,
            "vf_peak": vf_peak,
            "status": STATUS_CONVERGED,
            "search_metadata": {
                "time_s": time.perf_counter() - start_time,
                "evaluations": 1,
                "method": "center",
                **metadata  # Include method-specific metrics from evaluator
            }
        }
    
    elif rc_mode == "grid":
        # Coarse grid sampling only
        return _coarse_grid_search(
            rc_w, rc_h, vf_evaluator, rc_grid_n, start_time
        )
    
    elif rc_mode == "search":
        # Full coarse-to-fine search
        return _coarse_to_fine_search(
            rc_w, rc_h, vf_evaluator, rc_grid_n, rc_search_rel_tol,
            rc_search_max_iters, rc_search_multistart, rc_search_time_limit_s,
            start_time
        )
    
    else:
        raise ValueError(f"Unknown rc_mode: {rc_mode}")


def _coarse_grid_search(
    rc_w: float, rc_h: float, vf_evaluator: Callable[[float, float], Tuple[float, Dict]],
    grid_n: int, start_time: float
) -> Dict:
    """Coarse grid sampling to find peak."""
    # Create uniform grid over receiver
    x_coords = np.linspace(-rc_w/2, rc_w/2, grid_n)
    y_coords = np.linspace(-rc_h/2, rc_h/2, grid_n)
    
    best_vf = -1.0
    best_x, best_y = 0.0, 0.0
    best_metadata = {}
    evaluations = 0
    
    for x in x_coords:
        for y in y_coords:
            vf, metadata = vf_evaluator(x, y)
            evaluations += 1
            
            if vf > best_vf:
                best_vf = vf
                best_x, best_y = x, y
                best_metadata = metadata
    
    return {
        "x_peak": best_x,
        "y_peak": best_y,
        "vf_peak": best_vf,
        "status": STATUS_CONVERGED,
        "search_metadata": {
            "time_s": time.perf_counter() - start_time,
            "evaluations": evaluations,
            "method": "grid",
            **best_metadata  # Include method-specific metrics from best evaluation
        }
    }


def _coarse_to_fine_search(
    rc_w: float, rc_h: float, vf_evaluator: Callable[[float, float], Tuple[float, Dict]],
    grid_n: int, rel_tol: float, max_iters: int, multistart: int, time_limit: float,
    start_time: float
) -> Dict:
    """Coarse-to-fine search with multi-start local optimization."""
    
    # Step 1: Coarse grid sampling
    x_coords = np.linspace(-rc_w/2, rc_w/2, grid_n)
    y_coords = np.linspace(-rc_h/2, rc_h/2, grid_n)
    
    grid_values = []
    best_metadata = {}
    for x in x_coords:
        for y in y_coords:
            vf, metadata = vf_evaluator(x, y)
            grid_values.append((vf, x, y))
            if vf > grid_values[0][0] if grid_values else True:  # Track best metadata
                best_metadata = metadata
    
    # Sort by view factor (descending) and select top seeds
    grid_values.sort(key=lambda x: x[0], reverse=True)
    
    # Deduplicate nearby seeds (within 10% of receiver size)
    seeds = []
    min_distance = 0.1 * min(rc_w, rc_h)
    
    for vf, x, y in grid_values:
        if not seeds or all(math.sqrt((x-sx)**2 + (y-sy)**2) > min_distance 
                           for sx, sy in seeds):
            seeds.append((x, y))
            if len(seeds) >= multistart:
                break
    
    # Step 2: Local optimization from each seed
    best_result = None
    total_evaluations = len(grid_values)
    
    for seed_x, seed_y in seeds:
        if time.perf_counter() - start_time > time_limit:
            break
            
        # Local optimization using Nelder-Mead
        def objective(params):
            x, y = params
            # Constrain to receiver bounds
            x = max(-rc_w/2, min(rc_w/2, x))
            y = max(-rc_h/2, min(rc_h/2, y))
            
            vf, metadata = vf_evaluator(x, y)
            # Store metadata for the best result
            nonlocal best_metadata
            if vf > (best_result["vf_peak"] if best_result else -1):
                best_metadata = metadata
            return -vf  # Minimize negative for maximization
        
        try:
            result = minimize(
                objective, 
                [seed_x, seed_y],
                method='Nelder-Mead',
                options={'maxiter': max_iters, 'xatol': rel_tol * 1e-3}
            )
            
            if result.success or result.nit >= max_iters:
                x_opt, y_opt = result.x
                vf_opt = -result.fun
                
                if best_result is None or vf_opt > best_result["vf_peak"]:
                    best_result = {
                        "x_peak": x_opt,
                        "y_peak": y_opt,
                        "vf_peak": vf_opt,
                        "status": STATUS_CONVERGED if result.success else STATUS_REACHED_LIMITS
                    }
                    
        except Exception:
            # Skip failed optimizations
            continue
    
    if best_result is None:
        # Fallback to best grid point
        best_vf, best_x, best_y = grid_values[0]
        # Check if we hit time limit or other limits
        elapsed_time = time.perf_counter() - start_time
        if elapsed_time >= time_limit:
            status = STATUS_REACHED_LIMITS
        else:
            status = STATUS_FAILED
        best_result = {
            "x_peak": best_x,
            "y_peak": best_y,
            "vf_peak": best_vf,
            "status": status
        }
    
    # Add search metadata
    best_result["search_metadata"] = {
        "time_s": time.perf_counter() - start_time,
        "iterations": len(seeds),
        "evaluations": total_evaluations,
        "method": "search",
        "seeds_used": len(seeds),
        **best_metadata  # Include method-specific metrics from best evaluation
    }
    
    return best_result


def create_vf_evaluator(
    method: str, em_w: float, em_h: float, rc_w: float, rc_h: float, 
    setback: float, angle: float, geom_cfg: Optional[Dict] = None, **method_params
) -> Callable[[float, float], Tuple[float, Dict]]:
    """
    Create a point view factor evaluator for the specified method.
    
    Args:
        method: Method name ("analytical", "adaptive", "fixedgrid", "montecarlo")
        em_w, em_h: Emitter dimensions
        rc_w, rc_h: Receiver dimensions
        setback: Setback distance
        angle: Rotation angle (degrees)
        geom_cfg: Geometry configuration dict with offsets and rotation settings
        **method_params: Method-specific parameters
        
    Returns:
        Function that evaluates F(x,y) at a receiver point
    """
    
    if method == "analytical":
        from .analytical import vf_point_rect_to_point
        from .geometry import to_emitter_frame
        
        def evaluator(x: float, y: float) -> Tuple[float, Dict]:
            # Transform receiver point to emitter frame if geometry config provided
            if geom_cfg:
                rx_local, ry_local = to_emitter_frame(
                    (x, y),  # receiver point in receiver local coordinates
                    geom_cfg.get('emitter_offset', (0.0, 0.0)),
                    geom_cfg.get('receiver_offset', (0.0, 0.0)),
                    geom_cfg.get('angle_deg', 0.0),
                    geom_cfg.get('rotate_target', 'emitter')
                )
            else:
                rx_local, ry_local = x, y
            
            # For analytical method, we can evaluate at a specific point
            vf = vf_point_rect_to_point(
                em_w, em_h, rc_w, rc_h, setback, angle,
                rx_local, ry_local, 
                nx=method_params.get('analytical_nx', 240),
                ny=method_params.get('analytical_ny', 240)
            )
            return vf, {'method': 'analytical'}
        
        return evaluator
    
    elif method == "adaptive":
        from .adaptive import vf_adaptive
        from .geometry import to_emitter_frame
        
        def evaluator(x: float, y: float) -> Tuple[float, Dict]:
            # Transform receiver point to emitter frame if geometry config provided
            if geom_cfg:
                rx_local, ry_local = to_emitter_frame(
                    (x, y),  # receiver point in receiver local coordinates
                    geom_cfg.get('emitter_offset', (0.0, 0.0)),
                    geom_cfg.get('receiver_offset', (0.0, 0.0)),
                    geom_cfg.get('angle_deg', 0.0),
                    geom_cfg.get('rotate_target', 'emitter')
                )
            else:
                rx_local, ry_local = x, y
            
            # For adaptive method, evaluate at the specific receiver point
            result = vf_adaptive(
                em_w, em_h, rc_w, rc_h, setback,
                rel_tol=method_params.get('rel_tol', 3e-3),
                abs_tol=method_params.get('abs_tol', 1e-6),
                max_depth=method_params.get('max_depth', 8),
                max_cells=method_params.get('max_cells', 10000),
                min_cells=method_params.get('min_cells', 16),
                time_limit_s=method_params.get('time_limit_s', 60.0),
                rc_point=(rx_local, ry_local)
            )
            return result['vf'], {
                'iterations': result.get('iterations', 0),
                'achieved_tol': result.get('achieved_tol', 0.0),
                'cells': result.get('cells', 0),
                'depth': result.get('depth', 0)
            }
        
        return evaluator
    
    elif method == "fixedgrid":
        from .fixed_grid import vf_fixed_grid
        from .geometry import to_emitter_frame
        
        def evaluator(x: float, y: float) -> Tuple[float, Dict]:
            # Transform receiver point to emitter frame if geometry config provided
            if geom_cfg:
                rx_local, ry_local = to_emitter_frame(
                    (x, y),  # receiver point in receiver local coordinates
                    geom_cfg.get('emitter_offset', (0.0, 0.0)),
                    geom_cfg.get('receiver_offset', (0.0, 0.0)),
                    geom_cfg.get('angle_deg', 0.0),
                    geom_cfg.get('rotate_target', 'emitter')
                )
            else:
                rx_local, ry_local = x, y
            
            # For fixed grid, we can evaluate at a specific point
            result = vf_fixed_grid(
                em_w, em_h, rc_w, rc_h, setback,
                grid_nx=method_params.get('grid_nx', 100),
                grid_ny=method_params.get('grid_ny', 100),
                quadrature=method_params.get('quadrature', 'centroid'),
                time_limit_s=method_params.get('time_limit_s', 60.0),
                rc_point=(rx_local, ry_local)
            )
            return result['vf'], {
                'samples_emitter': result.get('samples_emitter', 0),
                'samples_receiver': result.get('samples_receiver', 0)
            }
        
        return evaluator
    
    elif method == "montecarlo":
        from .montecarlo import vf_montecarlo
        from .geometry import to_emitter_frame
        
        def evaluator(x: float, y: float) -> Tuple[float, Dict]:
            # Transform receiver point to emitter frame if geometry config provided
            if geom_cfg:
                rx_local, ry_local = to_emitter_frame(
                    (x, y),  # receiver point in receiver local coordinates
                    geom_cfg.get('emitter_offset', (0.0, 0.0)),
                    geom_cfg.get('receiver_offset', (0.0, 0.0)),
                    geom_cfg.get('angle_deg', 0.0),
                    geom_cfg.get('rotate_target', 'emitter')
                )
            else:
                rx_local, ry_local = x, y
            
            # For Monte Carlo, we can evaluate at a specific point
            result = vf_montecarlo(
                em_w, em_h, rc_w, rc_h, setback,
                samples=method_params.get('samples', 10000),
                target_rel_ci=method_params.get('target_rel_ci', 0.05),
                max_iters=method_params.get('max_iters', 10),
                seed=method_params.get('seed', 42),
                time_limit_s=method_params.get('time_limit_s', 60.0),
                rc_mode="center",  # Force center mode for point evaluation
                rc_point=(rx_local, ry_local)
            )
            return result['vf_mean'], {
                'samples': result.get('samples', 0),
                'vf_ci95': result.get('vf_ci95', 0.0)
            }
        
        return evaluator
    
    else:
        raise ValueError(f"Unknown method: {method}")
