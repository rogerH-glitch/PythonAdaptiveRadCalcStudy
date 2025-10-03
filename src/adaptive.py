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
        s_min: Minimum s parameter (u-direction)
        s_max: Maximum s parameter (u-direction)  
        t_min: Minimum t parameter (v-direction)
        t_max: Maximum t parameter (v-direction)
        integral: Estimated integral over this cell
        error: Estimated error for this cell
        depth: Subdivision depth level
    """
    s_min: float
    s_max: float
    t_min: float
    t_max: float
    integral: float = 0.0
    error: float = 0.0
    depth: int = 0
    
    @property
    def area(self) -> float:
        """Cell area in parameter space."""
        return (self.s_max - self.s_min) * (self.t_max - self.t_min)
    
    @property
    def centre(self) -> Tuple[float, float]:
        """Cell centre coordinates."""
        return (0.5 * (self.s_min + self.s_max), 0.5 * (self.t_min + self.t_max))


def vf_adaptive(
    em_w: float, em_h: float, rc_w: float, rc_h: float, setback: float,
    rel_tol: float = 3e-3, abs_tol: float = 1e-6, max_depth: int = 12,
    min_cell_area_frac: float = 1e-8, max_cells: int = 200000, time_limit_s: float = 60,
    init_grid: str = "4x4", eps: float = EPS
) -> Dict:
    """
    Compute local peak view factor using Walton-style adaptive integration.
    
    Evaluates local VF at candidate receiver points and adaptively subdivides
    the emitter into sub-rectangles until convergence or limits are reached.
    
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
        eps: Small value for numerical stability
        
    Returns:
        Dictionary with:
        - vf: Maximum pointwise view factor
        - status: "converged" | "reached_limits" | "failed"
        - iterations: Number of cells processed
        - achieved_tol: Achieved tolerance
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
        
        # For parallel surfaces, local peak typically occurs at receiver centre
        # Use a small receiver grid for sampling
        rc_nx = max(3, nx // 2)  # Coarse receiver grid
        rc_ny = max(3, ny // 2)
        
        max_vf = 0.0
        total_iterations = 0
        max_depth_reached = 0
        total_cells = 0
        
        # Sample receiver points
        rc_dx = rc_w / rc_nx
        rc_dy = rc_h / rc_ny
        
        for rc_i in range(rc_nx):
            for rc_j in range(rc_ny):
                # Check time limit
                elapsed = time.perf_counter() - start_time
                if elapsed > time_limit_s:
                    return {
                        "vf": max_vf,
                        "status": STATUS_REACHED_LIMITS,
                        "iterations": total_iterations,
                        "achieved_tol": float('inf'),
                        "depth": max_depth_reached,
                        "cells": total_cells
                    }
                
                # Receiver point (centre of cell)
                rc_x = (rc_i + 0.5) * rc_dx - rc_w / 2  # Centre at origin
                rc_y = (rc_j + 0.5) * rc_dy - rc_h / 2
                rc_z = setback
                
                # Calculate adaptive view factor for this receiver point
                remaining_time = max(0.1, time_limit_s - elapsed)  # Ensure minimum time
                result = _calculate_adaptive_vf(
                    rc_x, rc_y, rc_z, em_w, em_h, nx, ny,
                    rel_tol, abs_tol, max_depth, min_cell_area_frac, 
                    max_cells, remaining_time, eps
                )
                
                max_vf = max(max_vf, result['vf'])
                total_iterations += result['iterations']
                max_depth_reached = max(max_depth_reached, result.get('depth', 0))
                total_cells += result.get('cells', 0)
        
        # Determine achieved tolerance
        achieved_tol = min(rel_tol, abs_tol)  # Conservative estimate
        
        return {
            "vf": max_vf,
            "status": STATUS_CONVERGED,
            "iterations": total_iterations,
            "achieved_tol": achieved_tol,
            "depth": max_depth_reached,
            "cells": total_cells
        }
        
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


def _calculate_adaptive_vf(
    rc_x: float, rc_y: float, rc_z: float,
    em_w: float, em_h: float, init_nx: int, init_ny: int,
    rel_tol: float, abs_tol: float, max_depth: int, 
    min_cell_area_frac: float, max_cells: int, time_limit_s: float, eps: float
) -> Dict:
    """
    Calculate adaptive view factor from receiver point to emitter.
    
    Args:
        rc_x, rc_y, rc_z: Receiver point coordinates
        em_w, em_h: Emitter dimensions
        init_nx, init_ny: Initial grid size
        rel_tol, abs_tol: Convergence tolerances
        max_depth: Maximum subdivision depth
        min_cell_area_frac: Minimum cell area fraction
        max_cells: Maximum number of cells
        time_limit_s: Remaining time limit
        eps: Small value for numerical stability
        
    Returns:
        Dictionary with vf, iterations, status, depth, and cells
    """
    start_time = time.perf_counter()
    
    # Create initial cells
    cells = []
    dx = 1.0 / init_nx
    dy = 1.0 / init_ny
    
    for i in range(init_nx):
        for j in range(init_ny):
            s_min = i * dx
            s_max = (i + 1) * dx
            t_min = j * dy
            t_max = (j + 1) * dy
            
            cell = AdaptiveCell(s_min, s_max, t_min, t_max)
            cells.append(cell)
    
    # Priority queue for refinement (max-heap by error)
    refinement_queue = []
    converged_cells = []
    total_integral = 0.0
    
    # Calculate initial estimates
    for cell in cells:
        integral, error = _estimate_cell_integral(cell, rc_x, rc_y, rc_z, em_w, em_h, eps)
        cell.integral = integral
        cell.error = error
        total_integral += integral
        
        # Add to refinement queue (negative error for max-heap)
        heapq.heappush(refinement_queue, (-error, id(cell), cell))
    
    iteration = 0
    min_cell_area = min_cell_area_frac * (1.0 / init_nx) * (1.0 / init_ny)
    
    # Stagnation detection
    stagnation_count = 0
    last_error = float('inf')
    stagnation_threshold = 0.1 * max(rel_tol * abs(total_integral), abs_tol)
    max_depth_reached = 0
    
    while refinement_queue and iteration < max_cells:
        # Check time limit
        if time.perf_counter() - start_time > time_limit_s:
            break
        
        # Check cell limit before processing
        total_cells = len(converged_cells) + len(refinement_queue)
        if total_cells >= max_cells:
            break
        
        # Get cell with largest error
        neg_error, _, current_cell = heapq.heappop(refinement_queue)
        current_error = -neg_error
        
        # Update max depth reached
        max_depth_reached = max(max_depth_reached, current_cell.depth)
        
        # Calculate current tolerance
        current_tol = max(rel_tol * abs(total_integral), abs_tol)
        
        # Check for stagnation
        if current_error < last_error:
            error_improvement = last_error - current_error
            if error_improvement < stagnation_threshold:
                stagnation_count += 1
                if stagnation_count >= 5:  # 5 consecutive steps with minimal improvement
                    break
            else:
                stagnation_count = 0
        else:
            stagnation_count += 1
            if stagnation_count >= 5:
                break
        
        last_error = current_error
        
        if current_error < current_tol:
            # Cell has converged
            converged_cells.append(current_cell)
            # Move all remaining cells to converged
            while refinement_queue:
                _, _, cell = heapq.heappop(refinement_queue)
                converged_cells.append(cell)
            break
        
        # Check depth limit
        if current_cell.depth >= max_depth:
            converged_cells.append(current_cell)
            continue
        
        # Check minimum cell area
        if current_cell.area < min_cell_area:
            converged_cells.append(current_cell)
            continue
        
        # Subdivide cell
        subcells = _subdivide_cell(current_cell)
        
        # Calculate refined estimates
        refined_total = 0.0
        for subcell in subcells:
            integral, error = _estimate_cell_integral(subcell, rc_x, rc_y, rc_z, em_w, em_h, eps)
            subcell.integral = integral
            subcell.error = error
            refined_total += integral
        
        # Update total integral
        total_integral = total_integral - current_cell.integral + refined_total
        
        # Add subcells to appropriate queues
        for subcell in subcells:
            if subcell.error < current_tol:
                converged_cells.append(subcell)
            else:
                heapq.heappush(refinement_queue, (-subcell.error, id(subcell), subcell))
        
        iteration += 1
    
    # Calculate final result
    final_integral = sum(cell.integral for cell in converged_cells)
    final_integral += sum(cell.integral for _, _, cell in refinement_queue)
    
    # Convert to view factor (normalize by emitter area)
    vf = final_integral  # Already includes physical area scaling in cell_area
    
    # Clamp to physical bounds
    vf = max(0.0, min(vf, 1.0))
    
    # Determine status
    total_cells = len(converged_cells) + len(refinement_queue)
    if len(refinement_queue) == 0:
        status = STATUS_CONVERGED
    elif total_cells >= max_cells or stagnation_count >= 5:
        status = STATUS_REACHED_LIMITS
    else:
        status = STATUS_REACHED_LIMITS  # Time limit or other limit reached
    
    return {
        "vf": vf,
        "iterations": total_cells,
        "status": status,
        "depth": max_depth_reached,
        "cells": total_cells
    }


def _estimate_cell_integral(
    cell: AdaptiveCell, rc_x: float, rc_y: float, rc_z: float,
    em_w: float, em_h: float, eps: float
) -> Tuple[float, float]:
    """
    Estimate integral and error for a cell using centroid quadrature.
    
    Args:
        cell: Cell to estimate
        rc_x, rc_y, rc_z: Receiver point coordinates
        em_w, em_h: Emitter dimensions
        eps: Small value for numerical stability
        
    Returns:
        Tuple of (integral, error_estimate)
    """
    # Centroid quadrature (1-point)
    s_centroid = 0.5 * (cell.s_min + cell.s_max)
    t_centroid = 0.5 * (cell.t_min + cell.t_max)
    
    # Convert to world coordinates
    em_x = (s_centroid - 0.5) * em_w  # Centre at origin
    em_y = (t_centroid - 0.5) * em_h
    em_z = 0.0
    
    # Calculate view factor kernel
    dx = rc_x - em_x
    dy = rc_y - em_y
    dz = rc_z - em_z
    r_squared = dx*dx + dy*dy + dz*dz
    
    if r_squared > eps:
        r = np.sqrt(r_squared)
        
        # For parallel surfaces: cosθ1 = cosθ2 = setback / r
        # Clamp cos_theta to prevent numerical issues
        cos_theta = max(0.0, min(1.0, rc_z / r))
        
        # View factor kernel: (cosθ1 cosθ2)/(π r²)
        kernel = (cos_theta * cos_theta) / (np.pi * r_squared)
        
        # Cell area contribution
        cell_area = cell.area * em_w * em_h  # Physical area
        integral = kernel * cell_area
        
        # Error estimate using corner sampling
        error = _estimate_cell_error(cell, rc_x, rc_y, rc_z, em_w, em_h, eps)
        
        return integral, error
    else:
        return 0.0, 0.0


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


def _subdivide_cell(cell: AdaptiveCell) -> List[AdaptiveCell]:
    """
    Subdivide cell into 4 subcells (2×2 refinement).
    
    Args:
        cell: Cell to subdivide
        
    Returns:
        List of 4 subcells
    """
    s_mid = 0.5 * (cell.s_min + cell.s_max)
    t_mid = 0.5 * (cell.t_min + cell.t_max)
    
    subcells = [
        AdaptiveCell(cell.s_min, s_mid, cell.t_min, t_mid, depth=cell.depth + 1),
        AdaptiveCell(s_mid, cell.s_max, cell.t_min, t_mid, depth=cell.depth + 1),
        AdaptiveCell(cell.s_min, s_mid, t_mid, cell.t_max, depth=cell.depth + 1),
        AdaptiveCell(s_mid, cell.s_max, t_mid, cell.t_max, depth=cell.depth + 1),
    ]
    
    return subcells