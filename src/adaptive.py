"""
Adaptive integration view factor calculations.

This module implements the adaptive integration method (1AI) per 
NISTIR 6925 (Walton, 2002) with recursive mesh refinement.
"""

import time
import heapq
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from .constants import EPS, STATUS_CONVERGED, STATUS_REACHED_LIMITS, STATUS_FAILED

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveCell:
    """Represents a cell in the adaptive mesh.
    
    Attributes:
        x0, x1: Cell bounds in x-direction (emitter coordinates)
        y0, y1: Cell bounds in y-direction (emitter coordinates)
        est: Estimated integral over this cell
        err: Estimated error for this cell
        depth: Subdivision depth level
    """
    x0: float
    x1: float
    y0: float
    y1: float
    est: float = 0.0
    err: float = 0.0
    depth: int = 0
    
    @property
    def area(self) -> float:
        """Cell area in physical coordinates."""
        return (self.x1 - self.x0) * (self.y1 - self.y0)
    
    @property
    def centre(self) -> Tuple[float, float]:
        """Cell centre coordinates."""
        return (0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1))


def vf_adaptive(
    em_w: float, em_h: float, rc_w: float, rc_h: float, setback: float,
    rel_tol: float = 3e-3, abs_tol: float = 1e-6, max_depth: int = 12,
    min_cell_area_frac: float = 1e-8, max_cells: int = 200000, time_limit_s: float = 60,
    init_grid: str = "4x4", min_cells: int = 16, eps: float = EPS,
    rc_point: Tuple[float, float] = (0.0, 0.0)
) -> Dict:
    """
    Compute local peak view factor using Walton-style adaptive integration.
    
    Uses proper 1AI refinement with error-driven subdivision based on
    centroid vs 2x2 quadrature error estimation.
    
    Args:
        em_w: Emitter width (m)
        em_h: Emitter height (m) 
        rc_w: Receiver width (m)
        rc_h: Receiver height (m)
        setback: Setback distance (m)
        rel_tol: Relative tolerance for convergence
        abs_tol: Absolute tolerance for convergence
        max_depth: Maximum subdivision depth
        min_cell_area_frac: Minimum cell area fraction
        max_cells: Maximum number of cells
        time_limit_s: Time limit in seconds
        init_grid: Initial grid size (e.g., "4x4")
        min_cells: Minimum number of cells before convergence
        eps: Small value for numerical stability
        rc_point: Receiver point (rx, ry) for local VF evaluation
        
    Returns:
        Dictionary with:
        - vf: Pointwise view factor at rc_point
        - status: "converged" | "reached_limits" | "failed"
        - iterations: Number of subdivisions performed
        - achieved_tol: Achieved relative tolerance
        - depth: Maximum depth reached
        - cells: Total cells processed
    """
    start_time = time.perf_counter()
    
    # Enhanced input validation
    if not all(x > 0 for x in [em_w, em_h, rc_w, rc_h, setback]):
        return {
            "vf": 0.0,
            "status": STATUS_FAILED,
            "iterations": 0,
            "achieved_tol": float('inf'),
            "depth": 0,
            "cells": 0
        }
    
    if not all(x > 0 for x in [rel_tol, abs_tol]) or max_depth < 1 or max_cells < 1:
        return {
            "vf": 0.0,
            "status": STATUS_FAILED,
            "iterations": 0,
            "achieved_tol": float('inf'),
            "depth": 0,
            "cells": 0
        }
    
    # Clamp tolerances to reasonable ranges
    rel_tol = max(1e-6, min(rel_tol, 0.1))
    abs_tol = max(1e-9, min(abs_tol, 1e-3))
    max_depth = min(max_depth, 20)  # Prevent excessive depth
    max_cells = min(max_cells, 1000000)  # Prevent excessive memory usage
    min_cells = max(min_cells, 16)  # Ensure minimum refinement
    
    try:
        # Parse initial grid
        if 'x' in init_grid.lower():
            try:
                nx, ny = map(int, init_grid.lower().split('x'))
            except ValueError:
                nx, ny = 4, 4
        else:
            try:
                nx = ny = int(init_grid)
            except ValueError:
                nx, ny = 4, 4
        
        # Extract receiver point
        rc_x, rc_y = rc_point
        rc_z = setback
        
        # Calculate adaptive view factor for the specified receiver point
        result = _calculate_adaptive_vf_walton(
            rc_x, rc_y, rc_z, em_w, em_h, nx, ny,
            rel_tol, abs_tol, max_depth, min_cell_area_frac, 
            max_cells, min_cells, time_limit_s, eps
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Adaptive calculation failed: {e}")
        return {
            "vf": 0.0,
            "status": STATUS_FAILED,
            "iterations": 0,
            "achieved_tol": float('inf'),
            "depth": 0,
            "cells": 0
        }


def _calculate_adaptive_vf_walton(
    rc_x: float, rc_y: float, rc_z: float,
    em_w: float, em_h: float, init_nx: int, init_ny: int,
    rel_tol: float, abs_tol: float, max_depth: int, 
    min_cell_area_frac: float, max_cells: int, min_cells: int, 
    time_limit_s: float, eps: float
) -> Dict:
    """
    Calculate adaptive view factor using proper Walton-style 1AI refinement.
    
    Uses error-driven subdivision based on centroid vs 2x2 quadrature error estimation.
    
    Args:
        rc_x, rc_y, rc_z: Receiver point coordinates
        em_w, em_h: Emitter dimensions
        init_nx, init_ny: Initial grid size
        rel_tol, abs_tol: Convergence tolerances
        max_depth: Maximum subdivision depth
        min_cell_area_frac: Minimum cell area fraction
        max_cells: Maximum number of cells
        min_cells: Minimum number of cells before convergence
        time_limit_s: Time limit in seconds
        eps: Small value for numerical stability
        
    Returns:
        Dictionary with vf, iterations, status, depth, cells, achieved_tol
    """
    start_time = time.perf_counter()
    
    # Create initial cells in physical coordinates
    cells = []
    dx = em_w / init_nx
    dy = em_h / init_ny
    
    for i in range(init_nx):
        for j in range(init_ny):
            x0 = -em_w/2 + i * dx
            x1 = -em_w/2 + (i + 1) * dx
            y0 = -em_h/2 + j * dy
            y1 = -em_h/2 + (j + 1) * dy
            
            cell = AdaptiveCell(x0, x1, y0, y1)
            cells.append(cell)
    
    # Priority queue for refinement (max-heap by error)
    refinement_queue = []
    sum_est = 0.0
    sum_err = 0.0
    
    # Calculate initial estimates using proper error estimation
    for cell in cells:
        est, err = _estimate_cell_integral_walton(cell, rc_x, rc_y, rc_z, eps)
        cell.est = est
        cell.err = err
        sum_est += est
        sum_err += err
        
        # Add to refinement queue (negative error for max-heap)
        heapq.heappush(refinement_queue, (-err, id(cell), cell))
    
    iteration = 0
    min_cell_area = min_cell_area_frac * em_w * em_h
    max_depth_reached = 0
    
    # Main refinement loop
    while refinement_queue:
        # Check time limit
        if time.perf_counter() - start_time > time_limit_s:
            break
        
        # Count total cells (leaves in the tree)
        total_cells = len(refinement_queue)
        if total_cells >= max_cells:
            break
        
        # Calculate current tolerance
        tol = max(rel_tol * abs(sum_est), abs_tol)
        
        # Check convergence: error <= tolerance AND minimum cells satisfied
        if sum_err <= tol and total_cells >= min_cells:
            break
        
        # Get cell with largest error
        neg_error, _, current_cell = heapq.heappop(refinement_queue)
        current_error = -neg_error
        
        # Update max depth reached
        max_depth_reached = max(max_depth_reached, current_cell.depth)
        
        # Check depth limit
        if current_cell.depth >= max_depth:
            # Put cell back in queue (it won't be refined further)
            heapq.heappush(refinement_queue, (neg_error, id(current_cell), current_cell))
            break
        
        # Check minimum cell area
        if current_cell.area < min_cell_area:
            # Put cell back in queue (it won't be refined further)
            heapq.heappush(refinement_queue, (neg_error, id(current_cell), current_cell))
            break
        
        # Subdivide cell into 4 subcells
        subcells = _subdivide_cell_walton(current_cell)
        
        # Remove current cell from sum
        sum_est -= current_cell.est
        sum_err -= current_cell.err
        
        # Calculate estimates for subcells
        for subcell in subcells:
            est, err = _estimate_cell_integral_walton(subcell, rc_x, rc_y, rc_z, eps)
            subcell.est = est
            subcell.err = err
            sum_est += est
            sum_err += err
            
            # Add to refinement queue
            heapq.heappush(refinement_queue, (-err, id(subcell), subcell))
        
        iteration += 1
    
    # Calculate final result
    final_est = sum_est
    
    # Clamp to physical bounds
    vf = max(0.0, min(final_est, 1.0))
    
    # Calculate achieved tolerance
    achieved_tol = sum_err / max(abs(sum_est), 1e-300)
    
    # Determine status
    total_cells = len(refinement_queue)
    final_tol = max(rel_tol * abs(sum_est), abs_tol)
    if sum_err <= final_tol and total_cells >= min_cells:
        status = STATUS_CONVERGED
    else:
        status = STATUS_REACHED_LIMITS
    
    return {
        "vf": vf,
        "iterations": iteration,
        "status": status,
        "depth": max_depth_reached,
        "cells": total_cells,
        "achieved_tol": achieved_tol
    }


def _estimate_cell_integral_walton(
    cell: AdaptiveCell, rc_x: float, rc_y: float, rc_z: float, eps: float
) -> Tuple[float, float]:
    """
    Estimate integral and error for a cell using Walton-style 1AI.
    
    Uses centroid estimate vs 2x2 quadrature for error estimation.
    
    Args:
        cell: Cell to estimate
        rc_x, rc_y, rc_z: Receiver point coordinates
        eps: Small value for numerical stability
        
    Returns:
        Tuple of (integral_2x2, error_estimate)
    """
    # Centroid estimate (1-point)
    cx, cy = cell.centre
    est_centroid = _evaluate_kernel(cx, cy, rc_x, rc_y, rc_z, eps) * cell.area
    
    # 2x2 estimate (4-point)
    dx = (cell.x1 - cell.x0) / 2
    dy = (cell.y1 - cell.y0) / 2
    
    # Four subcell centers
    subcells = [
        (cell.x0 + dx/2, cell.y0 + dy/2),  # Bottom-left
        (cell.x1 - dx/2, cell.y0 + dy/2),  # Bottom-right
        (cell.x0 + dx/2, cell.y1 - dy/2),  # Top-left
        (cell.x1 - dx/2, cell.y1 - dy/2),  # Top-right
    ]
    
    est_2x2 = 0.0
    for sx, sy in subcells:
        kernel_val = _evaluate_kernel(sx, sy, rc_x, rc_y, rc_z, eps)
        subcell_area = dx * dy
        est_2x2 += kernel_val * subcell_area
    
    # Error estimate: |I_2x2 - I_centroid|
    error = abs(est_2x2 - est_centroid)
    
    # Return the better estimate (2x2) and error
    return est_2x2, error


def _evaluate_kernel(em_x: float, em_y: float, rc_x: float, rc_y: float, rc_z: float, eps: float) -> float:
    """
    Evaluate the view factor kernel at a point.
    
    For parallel surfaces: K = s²/(π r⁴) where s = setback, r = distance
    
    Args:
        em_x, em_y: Emitter point coordinates
        rc_x, rc_y, rc_z: Receiver point coordinates
        eps: Small value for numerical stability
        
    Returns:
        Kernel value
    """
    dx = rc_x - em_x
    dy = rc_y - em_y
    dz = rc_z - 0.0  # Emitter is at z=0
    r_squared = dx*dx + dy*dy + dz*dz
    
    if r_squared > eps:
        r = np.sqrt(r_squared)
        
        # For parallel surfaces: cosθ1 = cosθ2 = setback / r
        # Clamp cos_theta to prevent numerical issues
        cos_theta = max(0.0, min(1.0, rc_z / r))
        
        # View factor kernel: (cosθ1 cosθ2)/(π r²) = s²/(π r⁴)
        kernel = (cos_theta * cos_theta) / (np.pi * r_squared)
        
        return kernel
    else:
        return 0.0


def _estimate_cell_error(
    cell: AdaptiveCell, rc_x: float, rc_y: float, rc_z: float,
    em_w: float, em_h: float, eps: float
) -> float:
    """
    Estimate error for a cell using corner sampling.
    
    Args:
        cell: Cell to estimate error for
        rc_x, rc_y, rc_z: Receiver point coordinates
        em_w, em_h: Emitter dimensions
        eps: Small value for numerical stability
        
    Returns:
        Error estimate
    """
    # Sample corners of the cell
    corners = [
        (cell.s_min, cell.t_min),
        (cell.s_max, cell.t_min),
        (cell.s_min, cell.t_max),
        (cell.s_max, cell.t_max)
    ]
    
    corner_values = []
    for s, t in corners:
        # Convert to world coordinates
        em_x = (s - 0.5) * em_w  # Centre at origin
        em_y = (t - 0.5) * em_h
        em_z = 0.0
        
        # Calculate view factor kernel
        dx = rc_x - em_x
        dy = rc_y - em_y
        dz = rc_z - em_z
        r_squared = dx*dx + dy*dy + dz*dz
        
        if r_squared > eps:
            r = np.sqrt(r_squared)
            cos_theta = rc_z / r
            kernel = (cos_theta * cos_theta) / (np.pi * r_squared)
            corner_values.append(kernel)
        else:
            corner_values.append(0.0)
    
    # Error estimate as standard deviation of corner values
    if len(corner_values) > 1:
        error = np.std(corner_values) * cell.area * em_w * em_h
    else:
        error = 0.0
    
    return error


def _subdivide_cell_walton(cell: AdaptiveCell) -> List[AdaptiveCell]:
    """
    Subdivide a cell into 4 subcells.
    
    Args:
        cell: Cell to subdivide
        
    Returns:
        List of 4 subcells
    """
    x_mid = 0.5 * (cell.x0 + cell.x1)
    y_mid = 0.5 * (cell.y0 + cell.y1)
    
    subcells = [
        AdaptiveCell(cell.x0, x_mid, cell.y0, y_mid, depth=cell.depth + 1),
        AdaptiveCell(x_mid, cell.x1, cell.y0, y_mid, depth=cell.depth + 1),
        AdaptiveCell(cell.x0, x_mid, y_mid, cell.y1, depth=cell.depth + 1),
        AdaptiveCell(x_mid, cell.x1, y_mid, cell.y1, depth=cell.depth + 1),
    ]
    
    return subcells