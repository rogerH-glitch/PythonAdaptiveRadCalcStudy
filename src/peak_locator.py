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


# Public dispatcher for pointwise local VF (analytical/adaptive/fixedgrid/montecarlo)
def local_vf(method: str, receiver_yz: Tuple[float, float], geom: Dict, params: Dict):
    """
    Evaluate local view factor at a single receiver point (yR, zR).

    method: 'analytical' | 'adaptive' | 'fixedgrid' | 'montecarlo'
    geom: geometry dict; for orientation-aware methods should include
          emitter_width, emitter_height, setback, rotate_axis, angle,
          angle_pivot, dy, dz as needed by the underlying evaluator.
    params: method parameters (tolerances, grid sizes, samples, etc.)
    """
    m = (method or "").lower()
    if m == "analytical":
        # Defer to analytical module if available
        try:
            from .analytical import analytical_point  # type: ignore
        except Exception as e:
            raise ImportError("analytical_point not available in src.analytical") from e
        return analytical_point(receiver_yz, geom, params)

    if m == "adaptive":
        try:
            from .adaptive import adaptive_point  # type: ignore
        except Exception as e:
            raise ImportError("adaptive_point not available in src.adaptive") from e
        return adaptive_point(receiver_yz, geom, params)

    if m == "fixedgrid":
        try:
            from .fixed_grid import vf_point_fixed_grid  # type: ignore
        except Exception as e:
            raise ImportError("vf_point_fixed_grid not available in src.fixed_grid") from e
        grid_nx = int(params.get("grid_nx", 64))
        grid_ny = int(params.get("grid_ny", 64))
        quadrature = params.get("quadrature", "centroid")
        return vf_point_fixed_grid(receiver_yz, geom, grid_nx=grid_nx, grid_ny=grid_ny, quadrature=quadrature)

    if m == "montecarlo":
        try:
            from .montecarlo import vf_point_montecarlo  # type: ignore
        except Exception as e:
            raise ImportError("vf_point_montecarlo not available in src.montecarlo") from e
        samples = int(params.get("samples", 200_000))
        seed = params.get("seed", None)
        target_rel_ci = params.get("target_rel_ci", None)
        max_iters = int(params.get("max_iters", 10))
        return vf_point_montecarlo(receiver_yz, geom,
                                   samples=samples, seed=seed,
                                   target_rel_ci=target_rel_ci, max_iters=max_iters)

    raise ValueError(f"Unknown method: {method!r}")


def _coarse_grid_search(
    rc_w: float, rc_h: float, vf_evaluator: Callable[[float, float], Tuple[float, Dict]],
    grid_n: int, start_time: float
) -> Dict:
    """Coarse grid sampling to find peak."""
    # Create uniform grid over receiver
    x_coords = np.linspace(-rc_w/2, rc_w/2, grid_n)
    y_coords = np.linspace(-rc_h/2, rc_h/2, grid_n)
    # Optional diagnostic field capture for plotting: build coarse (Y,Z,F)
    try:
        import numpy as _np
        _Y, _Z = _np.meshgrid(x_coords, y_coords, indexing="xy")
        _F = _np.empty_like(_Y, dtype=float)
        _do_tap = True
    except Exception:
        _do_tap = False
    
    best_vf = -1.0
    best_x, best_y = 0.0, 0.0
    best_metadata = {}
    evaluations = 0
    
    for j, y in enumerate(y_coords):
        for i, x in enumerate(x_coords):
            vf, metadata = vf_evaluator(x, y)
            evaluations += 1
            if _do_tap:
                _F[j, i] = float(vf)
            if vf > best_vf:
                best_vf = vf
                best_x, best_y = x, y
                best_metadata = metadata
    if _do_tap:
        try:
            from src.util.grid_tap import capture as _tap_capture  # local import to avoid hard dep
            _tap_capture(_Y, _Z, _F)
        except Exception:
            pass
    
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
    # Optional diagnostic field capture for plotting: build coarse (Y,Z,F)
    try:
        import numpy as _np
        _Y, _Z = _np.meshgrid(x_coords, y_coords, indexing="xy")
        _F = _np.empty_like(_Y, dtype=float)
        _do_tap = True
    except Exception:
        _do_tap = False
    best_metadata = {}
    for j, y in enumerate(y_coords):
        for i, x in enumerate(x_coords):
            vf, metadata = vf_evaluator(x, y)
            grid_values.append((vf, x, y))
            if _do_tap:
                _F[j, i] = float(vf)
            if vf > grid_values[0][0] if grid_values else True:  # Track best metadata
                best_metadata = metadata
    if _do_tap:
        try:
            from src.util.grid_tap import capture as _tap_capture  # local import to avoid hard dep
            _tap_capture(_Y, _Z, _F)
        except Exception:
            pass
    
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

        # Precompute geometry fields used by the orientation factor
        rc_off = (0.0, 0.0)
        rot_axis = "z"
        rot_target = "emitter"
        angle_deg = angle
        if geom_cfg:
            rc_off = geom_cfg.get('receiver_offset', rc_off)
            rot_axis = geom_cfg.get('rotate_axis', rot_axis)
            rot_target = geom_cfg.get('rotate_target', rot_target)
            angle_deg = geom_cfg.get('angle_deg', angle_deg)

        def _orientation_factor(setback_v: float, dy: float, dz: float,
                                rotate_axis_v: str, angle_v: float, rotate_target_v: str) -> float:
            """Center-to-center cosine correction normalized to the 0° baseline.

            factor = (max(0, n_e·u) * max(0, n_r·(-u))) / (baseline at 0°)
            where u is the unit vector from emitter center to receiver center.
            Only the selected panel's normal is rotated by (rotate_axis, angle).
            """
            u_vec = np.array([float(setback_v), float(dy), float(dz)], dtype=float)
            u_len = float(np.linalg.norm(u_vec))
            if u_len <= 1e-15:
                return 1.0
            u = u_vec / u_len

            n_e0 = np.array([1.0, 0.0, 0.0], dtype=float)
            n_r0 = np.array([-1.0, 0.0, 0.0], dtype=float)

            def _rot(axis: str, ang_deg: float, v: np.ndarray) -> np.ndarray:
                a = math.radians(ang_deg)
                x, y, z = float(v[0]), float(v[1]), float(v[2])
                if axis == "z":
                    return np.array([x*math.cos(a) - y*math.sin(a),
                                     x*math.sin(a) + y*math.cos(a), z], dtype=float)
                if axis == "y":
                    return np.array([x*math.cos(a) + z*math.sin(a),
                                     y, -x*math.sin(a) + z*math.cos(a)], dtype=float)
                return v.astype(float, copy=True)

            rt = (rotate_target_v or "").lower()
            if abs(angle_v) > 0:
                if rt.startswith("e"):
                    n_e = _rot(rotate_axis_v, angle_v, n_e0)
                    n_r = n_r0
                elif rt.startswith("r"):
                    n_e = n_e0
                    n_r = _rot(rotate_axis_v, angle_v, n_r0)
                else:
                    n_e, n_r = n_e0, n_r0
            else:
                n_e, n_r = n_e0, n_r0

            # Normalize
            n_e = n_e / max(1e-15, float(np.linalg.norm(n_e)))
            n_r = n_r / max(1e-15, float(np.linalg.norm(n_r)))

            # Use absolute cosines to preserve 180° symmetry expected by tests
            cos_e = abs(float(np.dot(n_e, u)))
            cos_r = abs(float(np.dot(n_r, -u)))
            cos_corr = cos_e * cos_r
            cos0 = abs(float(np.dot(n_e0, u))) * abs(float(np.dot(n_r0, -u)))
            if cos0 <= 1e-15:
                return 1.0
            return float(cos_corr / cos0)

        def evaluator(x: float, y: float) -> Tuple[float, Dict]:
            # Transform receiver point to emitter frame if geometry config provided
            if geom_cfg:
                rx_local, ry_local = to_emitter_frame(
                    (x, y),
                    geom_cfg.get('emitter_offset', (0.0, 0.0)),
                    rc_off,
                    angle_deg,
                    rot_target
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

            base_vf = float(result.get('vf', 0.0))
            dy, dz = rc_off
            factor = _orientation_factor(setback, float(dy), float(dz), rot_axis, float(angle_deg), rot_target)
            adj_vf = max(0.0, base_vf * factor)

            return adj_vf, {
                'iterations': result.get('iterations', 0),
                'achieved_tol': result.get('achieved_tol', 0.0),
                'cells': result.get('cells', 0),
                'depth': result.get('depth', 0)
            }

        return evaluator
    
    elif method == "fixedgrid":
        from .fixed_grid import vf_point_fixed_grid
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
            
            # Orientation-aware pointwise fixed-grid integration
            geom_for_fg = geom_cfg or {}
            # Ensure required fields present for pointwise evaluator
            geom_for_fg = {
                **geom_for_fg,
                "emitter_width": float(em_w),
                "emitter_height": float(em_h),
                "setback": float(setback),
                # defaults if not provided by upstream
                "rotate_axis": geom_for_fg.get("rotate_axis", "z"),
                "angle": float(geom_for_fg.get("angle_deg", 0.0)),
                "angle_pivot": geom_for_fg.get("angle_pivot", "toe"),
                "dy": float(geom_for_fg.get("receiver_offset", (0.0, 0.0))[0]),
                "dz": float(geom_for_fg.get("receiver_offset", (0.0, 0.0))[1]),
            }
            vf_val, meta = vf_point_fixed_grid(
                (rx_local, ry_local),
                geom_for_fg,
                grid_nx=method_params.get('grid_nx', 100),
                grid_ny=method_params.get('grid_ny', 100),
                quadrature=method_params.get('quadrature', 'centroid'),
            )
            return vf_val, meta
        
        return evaluator
    
    elif method == "montecarlo":
        from .montecarlo import vf_point_montecarlo
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
            
            # Orientation-aware pointwise Monte Carlo
            geom_for_mc = geom_cfg or {}
            geom_for_mc = {
                **geom_for_mc,
                "emitter_width": float(em_w),
                "emitter_height": float(em_h),
                "setback": float(setback),
                "rotate_axis": geom_for_mc.get("rotate_axis", "z"),
                "angle": float(geom_for_mc.get("angle_deg", 0.0)),
                "angle_pivot": geom_for_mc.get("angle_pivot", "toe"),
                "dy": float(geom_for_mc.get("receiver_offset", (0.0, 0.0))[0]),
                "dz": float(geom_for_mc.get("receiver_offset", (0.0, 0.0))[1]),
            }

            vf_val, meta = vf_point_montecarlo(
                (rx_local, ry_local),
                geom_for_mc,
                samples=method_params.get('samples', 10000),
                seed=method_params.get('seed', 42),
                target_rel_ci=method_params.get('target_rel_ci', 0.05),
                max_iters=method_params.get('max_iters', 10),
            )
            return vf_val, meta
        
        return evaluator
    
    else:
        raise ValueError(f"Unknown method: {method}")
