"""
Fixed grid view factor calculations.

This module implements view factor calculations using uniform grid
subdivision with regular quadrature points for local peak view factor computation.
"""

import time
import numpy as np
import math
from typing import Dict, Tuple
import logging

from .constants import EPS, STATUS_CONVERGED, STATUS_REACHED_LIMITS, STATUS_FAILED
try:
    # Optional orientation support for pointwise fixed-grid evaluation
    from .orientation import make_emitter_frame
except Exception:  # pragma: no cover - optional import guard
    make_emitter_frame = None  # type: ignore

logger = logging.getLogger(__name__)


def vf_fixed_grid(
    em_w: float, em_h: float, rc_w: float, rc_h: float, setback: float,
    grid_nx: int = 100, grid_ny: int = 100, quadrature: str = "centroid",
    time_limit_s: float = 60, eps: float = EPS
) -> Dict:
    """
    Compute local peak view factor using fixed grid method.
    
    Samples the receiver on a grid and for each receiver point, integrates
    over emitter with a fixed emitter grid using centroid quadrature.
    
    Args:
        em_w: Emitter width (m)
        em_h: Emitter height (m) 
        rc_w: Receiver width (m)
        rc_h: Receiver height (m)
        setback: Setback distance (m)
        grid_nx: Emitter grid points in x-direction
        grid_ny: Emitter grid points in y-direction
        quadrature: Quadrature method ("centroid" only for now)
        time_limit_s: Time limit in seconds
        eps: Small value for numerical stability
        
    Returns:
        Dictionary with:
        - vf: Maximum pointwise view factor over receiver grid
        - status: "converged" | "reached_limits" | "failed"
        - samples_emitter: Number of emitter grid points
        - samples_receiver: Number of receiver grid points
        - iterations: Total iterations (receiver points)
    """
    start_time = time.perf_counter()
    
    # Enhanced input validation
    if not all(x > 0 for x in [em_w, em_h, rc_w, rc_h, setback]):
        return {
            "vf": 0.0,
            "status": STATUS_FAILED,
            "samples_emitter": 0,
            "samples_receiver": 0,
            "iterations": 0
        }
    
    if not all(x > 0 for x in [grid_nx, grid_ny]):
        return {
            "vf": 0.0,
            "status": STATUS_FAILED,
            "samples_emitter": 0,
            "samples_receiver": 0,
            "iterations": 0
        }
    
    # Clamp grid sizes to reasonable ranges
    grid_nx = min(max(grid_nx, 10), 1000)  # 10 to 1000 points
    grid_ny = min(max(grid_ny, 10), 1000)
    
    try:
        # For parallel surfaces, the local peak typically occurs at receiver centre
        # Sample receiver grid to find maximum pointwise view factor
        max_vf = 0.0
        
        # Receiver grid for sampling (coarse grid for efficiency)
        rc_nx = max(10, grid_nx // 10)  # Coarse receiver grid
        rc_ny = max(10, grid_ny // 10)
        
        # Grid spacing
        em_dx = em_w / grid_nx
        em_dy = em_h / grid_ny
        rc_dx = rc_w / rc_nx
        rc_dy = rc_h / rc_ny
        
        iterations = 0
        max_iterations = rc_nx * rc_ny
        
        # Sample receiver points
        for rc_i in range(rc_nx):
            for rc_j in range(rc_ny):
                # Check time limit
                elapsed = time.perf_counter() - start_time
                if elapsed > time_limit_s:
                    return {
                        "vf": max_vf,
                        "status": STATUS_REACHED_LIMITS,
                        "samples_emitter": grid_nx * grid_ny,
                        "samples_receiver": rc_nx * rc_ny,
                        "iterations": iterations
                    }
                
                # Check iteration limit
                if iterations >= max_iterations:
                    return {
                        "vf": max_vf,
                        "status": STATUS_REACHED_LIMITS,
                        "samples_emitter": grid_nx * grid_ny,
                        "samples_receiver": rc_nx * rc_ny,
                        "iterations": iterations
                    }
                
                # Receiver point (centre of cell)
                rc_x = (rc_i + 0.5) * rc_dx - rc_w / 2  # Centre at origin
                rc_y = (rc_j + 0.5) * rc_dy - rc_h / 2
                rc_z = setback
                
                # Calculate pointwise view factor for this receiver point
                point_vf = _calculate_pointwise_vf(
                    rc_x, rc_y, rc_z, em_w, em_h, grid_nx, grid_ny, eps
                )
                
                max_vf = max(max_vf, point_vf)
                iterations += 1
        
        # Clamp to physical bounds
        max_vf = max(0.0, min(max_vf, 1.0))
        
        return {
            "vf": max_vf,
            "status": STATUS_CONVERGED,
            "samples_emitter": grid_nx * grid_ny,
            "samples_receiver": rc_nx * rc_ny,
            "iterations": iterations
        }
        
    except Exception as e:
        logger.error(f"Fixed grid calculation failed: {e}")
        return {
            "vf": 0.0,
            "status": STATUS_FAILED,
            "samples_emitter": grid_nx * grid_ny,
            "samples_receiver": rc_nx * rc_ny,
            "iterations": 0
        }


def _calculate_pointwise_vf(
    rc_x: float, rc_y: float, rc_z: float,
    em_w: float, em_h: float, grid_nx: int, grid_ny: int, eps: float
) -> float:
    """
    Calculate pointwise view factor from receiver point to emitter.
    
    Uses differential exchange formula: dF = (cosθ1 cosθ2)/(π r²) dA_emitter
    For parallel surfaces: cosθ1 = cosθ2 = setback / r
    
    Args:
        rc_x, rc_y, rc_z: Receiver point coordinates
        em_w, em_h: Emitter dimensions
        grid_nx, grid_ny: Emitter grid resolution
        eps: Small value for numerical stability
        
    Returns:
        Pointwise view factor value
    """
    # Grid spacing
    em_dx = em_w / grid_nx
    em_dy = em_h / grid_ny
    
    total_vf = 0.0
    
    # Integrate over emitter grid
    for em_i in range(grid_nx):
        for em_j in range(grid_ny):
            # Emitter point (centre of cell)
            em_x = (em_i + 0.5) * em_dx - em_w / 2  # Centre at origin
            em_y = (em_j + 0.5) * em_dy - em_h / 2
            em_z = 0.0
            
            # Distance vector
            dx = rc_x - em_x
            dy = rc_y - em_y
            dz = rc_z - em_z
            r_squared = dx*dx + dy*dy + dz*dz
            
            if r_squared > eps:  # Avoid division by zero
                r = np.sqrt(r_squared)
                
                # For parallel surfaces: cosθ1 = cosθ2 = setback / r
                # Clamp cos_theta to prevent numerical issues
                cos_theta = max(0.0, min(1.0, rc_z / r))
                
                # View factor kernel: (cosθ1 cosθ2)/(π r²)
                kernel = (cos_theta * cos_theta) / (np.pi * r_squared)
                
                # Cell area contribution
                cell_area = em_dx * em_dy
                total_vf += kernel * cell_area
    
    return total_vf


def _fg_int_uv(u: float, v: float, yR: float, zR: float, frame) -> float:
    """
    Oriented differential VF integrand at emitter local (u,v) to receiver point (S,yR,zR):
        dF = max(0, n1·r̂) * max(0, n2·(-r̂)) / (π r²)  du dv
    where p1 = C + j*u + k*v, p2 = (S, yR, zR), r = p2 - p1.
    """
    C, j, k, n1, n2, W, H, S = frame.C, frame.j, frame.k, frame.n1, frame.n2, frame.W, frame.H, frame.S
    p1 = C + j * u + k * v
    p2 = np.array([S, yR, zR], dtype=float)
    r = p2 - p1
    r2 = float(r.dot(r))
    if r2 <= 0.0:
        return 0.0
    rinv = 1.0 / math.sqrt(r2)
    rhat = r * rinv
    cos1 = float(n1.dot(rhat))
    cos2 = float(n2.dot(-rhat))
    if cos1 <= 0.0 or cos2 <= 0.0:
        return 0.0
    return (cos1 * cos2) / (math.pi * r2)


def vf_point_fixed_grid(
    receiver_yz: Tuple[float, float],
    geom: Dict,
    grid_nx: int = 64,
    grid_ny: int = 64,
    quadrature: str = "centroid",
) -> Tuple[float, Dict]:
    """
    Orientation-aware fixed-grid estimate of local view factor at a single
    receiver point (yR,zR), integrating over the emitter rectangle in (u,v).

    Required in `geom`:
      - emitter_width, emitter_height, setback
      - rotate_axis in {'z','y'}, angle (deg), angle_pivot in {'toe','center'}
      - dy, dz (receiver - emitter centre offsets in y,z)
    """
    if make_emitter_frame is None:  # pragma: no cover - defensive
        raise RuntimeError("orientation.make_emitter_frame is unavailable; add src/orientation.py first.")

    yR, zR = receiver_yz
    W = float(geom["emitter_width"])  # meters
    H = float(geom["emitter_height"])  # meters
    S = float(geom["setback"])  # meters
    axis = geom.get("rotate_axis", "z")
    angle = float(geom.get("angle", 0.0))
    pivot = geom.get("angle_pivot", "toe")
    dy = float(geom.get("dy", 0.0))
    dz = float(geom.get("dz", 0.0))

    frame = make_emitter_frame(W, H, S, axis=axis, angle_deg=angle, pivot=pivot, dy=dy, dz=dz)

    # Uniform grid in (u,v) over [-W/2,W/2] × [-H/2,H/2]
    us = np.linspace(-W / 2, W / 2, grid_nx + 1)
    vs = np.linspace(-H / 2, H / 2, grid_ny + 1)
    F = 0.0

    if quadrature == "centroid":
        for i in range(grid_nx):
            for j in range(grid_ny):
                uc = 0.5 * (us[i] + us[i + 1])
                vc = 0.5 * (vs[j] + vs[j + 1])
                val = _fg_int_uv(uc, vc, yR, zR, frame)
                F += val * (us[i + 1] - us[i]) * (vs[j + 1] - vs[j])
    elif quadrature == "2x2":
        w = [-1 / math.sqrt(3), +1 / math.sqrt(3)]
        for i in range(grid_nx):
            for j in range(grid_ny):
                u0, u1 = us[i], us[i + 1]
                v0, v1 = vs[j], vs[j + 1]
                du, dv = (u1 - u0) / 2.0, (v1 - v0) / 2.0
                uc, vc = (u0 + u1) / 2.0, (v0 + v1) / 2.0
                cell = 0.0
                for a in w:
                    for b in w:
                        uu = uc + a * du
                        vv = vc + b * dv
                        cell += _fg_int_uv(uu, vv, yR, zR, frame)
                F += cell * du * dv
    else:
        raise ValueError("quadrature must be 'centroid' or '2x2'")

    meta = {"grid_nx": grid_nx, "grid_ny": grid_ny, "quadrature": quadrature}
    return F, meta
