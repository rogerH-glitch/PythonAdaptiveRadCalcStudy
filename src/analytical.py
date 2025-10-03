"""
Analytical view factor calculations.

This module implements analytical and semi-analytical view factor calculations
using numerical integration over emitter grids for point evaluation.
"""

from __future__ import annotations
import math
import numpy as np
from .constants import EPS


def _kernel_parallel(setback: float, r2: np.ndarray) -> np.ndarray:
    """
    Differential exchange kernel for parallel, facing planes:
      dF = (cosθ1 cosθ2) / (π r^2) dA  with cosθ1=cosθ2=s/r  =>  s^2 / (π r^4)
    We compute (cos1*cos2)/(π r^2) directly for stability: (s/r * s/r)/(π r^2) = s^2/(π r^4)
    """
    r2 = np.maximum(r2, EPS)
    r = np.sqrt(r2)
    cos = np.maximum(0.0, np.minimum(1.0, setback / r))  # parallel, facing, clamped
    return (cos * cos) / (math.pi * r2)


def vf_point_rect_to_point_parallel(
    em_w: float, em_h: float,
    setback: float,
    rx: float, ry: float,
    nx: int = 240, ny: int = 240,
) -> float:
    """
    Compute point view factor at receiver point (rx, ry) due to a rectangular emitter,
    using centroid quadrature on a uniform emitter grid.

    Coordinates:
      - Emitter centered at (0,0) in its plane, spanning x∈[-em_w/2, +em_w/2], y∈[-em_h/2, +em_h/2]
      - Receiver centered & parallel at distance 'setback' along +z; (rx, ry) is expressed
        in the same XY frame (centres aligned → receiver center is (0,0)).

    Returns:
      F_point ∈ [0,1]
    """
    # Enhanced input validation
    if not all(x > 0 for x in [em_w, em_h, setback]):
        raise ValueError("Emitter dimensions and setback must be positive")
    
    if not all(x > 0 for x in [nx, ny]):
        raise ValueError("Grid dimensions must be positive")
    
    # Clamp grid sizes to reasonable ranges
    nx = min(max(nx, 10), 1000)  # 10 to 1000 points
    ny = min(max(ny, 10), 1000)
    
    # grid cell centers on emitter
    dx = em_w / nx
    dy = em_h / ny
    xs = np.linspace(-em_w/2 + 0.5*dx, em_w/2 - 0.5*dx, nx)
    ys = np.linspace(-em_h/2 + 0.5*dy, em_h/2 - 0.5*dy, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    # distance from emitter cell center to receiver point (rx, ry)
    DX = X - rx
    DY = Y - ry
    r2 = DX*DX + DY*DY + setback*setback

    K = _kernel_parallel(setback, r2)  # shape (ny, nx)
    dA = dx * dy
    F = float(np.sum(K) * dA)
    
    # Clamp to physical bounds
    F = max(0.0, min(1.0, F))
    return F


def local_peak_vf_analytic_approx(
    em_w: float, em_h: float,
    rc_w: float, rc_h: float,
    setback: float,
    rc_point: tuple[float, float] | None = None,
    nx: int = 240, ny: int = 240,
) -> float:
    """
    Wrapper to keep prior API but with receiver-point support.
    If rc_point is None → use receiver center (0,0).
    """
    # Input validation
    if em_w <= 0 or em_h <= 0 or rc_w <= 0 or rc_h <= 0 or setback <= 0:
        raise ValueError("All dimensions and setback must be positive")
    
    rx, ry = (0.0, 0.0) if rc_point is None else rc_point
    return vf_point_rect_to_point_parallel(em_w, em_h, setback, rx, ry, nx=nx, ny=ny)


# Optional: raise for non-zero angle until implemented
def vf_point_rect_to_point(
    em_w: float, em_h: float, rc_w: float, rc_h: float,
    setback: float, angle_deg: float,
    rx: float, ry: float, nx: int = 240, ny: int = 240
) -> float:
    """
    Future-proof entry point. Currently supports angle=0 (parallel).
    """
    if abs(angle_deg) > 1e-9:
        raise NotImplementedError("Analytical point evaluator currently supports angle=0 (parallel) only.")
    return vf_point_rect_to_point_parallel(em_w, em_h, setback, rx, ry, nx=nx, ny=ny)


def validate_geometry(em_w: float, em_h: float, rc_w: float, rc_h: float, setback: float) -> tuple[bool, str]:
    """
    Validate geometry parameters for analytical calculations.
    
    Args:
        em_w, em_h: Emitter dimensions
        rc_w, rc_h: Receiver dimensions
        setback: Setback distance
        
    Returns:
        (is_valid, error_message)
    """
    if em_w <= 0:
        return False, "Emitter width must be positive"
    if em_h <= 0:
        return False, "Emitter height must be positive"
    if rc_w <= 0:
        return False, "Receiver width must be positive"
    if rc_h <= 0:
        return False, "Receiver height must be positive"
    if setback <= 0:
        return False, "Setback must be positive"
    
    return True, ""


def get_analytical_info() -> str:
    """Get information string for analytical method."""
    return "Analytical baseline approximation using 200x200 Walton-style numerical integration"