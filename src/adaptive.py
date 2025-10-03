"""
Adaptive integration view factor calculations.

This module implements the adaptive integration method (1AI) per 
NISTIR 6925 (Walton, 2002) with recursive mesh refinement.
"""

import time
import heapq
import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

from .constants import EPS, STATUS_CONVERGED, STATUS_REACHED_LIMITS, STATUS_FAILED
from .geometry import Rectangle, ViewFactorResult, validate_geometry

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


class AdaptiveCalculator:
    """Calculator using adaptive integration (1AI method).
    
    Implements the single-area integration method with recursive 
    mesh refinement based on local error estimation per NISTIR 6925.
    """
    
    def __init__(self, 
                 tolerance: float = 3e-3,
                 max_depth: int = 12,
                 max_cells: int = 200000,
                 n_src_gl: int = 6,
                 n_tgt_gl: int = 28) -> None:
        """Initialise adaptive calculator.
        
        Args:
            tolerance: Relative convergence tolerance
            max_depth: Maximum recursion depth
            max_cells: Maximum number of cells
            n_src_gl: Gauss-Legendre points for source integration
            n_tgt_gl: Gauss-Legendre points for target integration
        """
        self._tolerance = tolerance
        self._max_depth = max_depth
        self._max_cells = max_cells
        self._n_src_gl = n_src_gl
        self._n_tgt_gl = n_tgt_gl
        self._name = "adaptive"
        
        # Pre-compute Gauss-Legendre quadrature points and weights
        self._setup_quadrature()
    
    def _setup_quadrature(self) -> None:
        """Pre-compute Gauss-Legendre quadrature points and weights."""
        # Source quadrature (for cell integration)
        xi_src, wi_src = leggauss(self._n_src_gl)
        self._x_src = 0.5 * (xi_src + 1.0)  # Transform to [0,1]
        self._w_src = 0.5 * wi_src
        
        # Target quadrature (for receiver integration)
        xi_tgt, wi_tgt = leggauss(self._n_tgt_gl)
        self._x_tgt = 0.5 * (xi_tgt + 1.0)
        self._w_tgt = 0.5 * wi_tgt
    
    def calculate(self, emitter: Rectangle, receiver: Rectangle) -> ViewFactorResult:
        """Calculate view factor using adaptive integration.
        
        Args:
            emitter: Source surface geometry
            receiver: Target surface geometry
            
        Returns:
            ViewFactorResult with calculated value and metadata
        """
        start_time = time.perf_counter()
        
        # Validate geometry
        validate_geometry(emitter, receiver)
        
        try:
            # Initialise with coarse 4×4 grid
            initial_cells = self._create_initial_mesh(grid_size=4)
            
            # Run adaptive refinement
            final_integral, converged, total_cells = self._adaptive_refinement(
                initial_cells, emitter, receiver
            )
            
            computation_time = time.perf_counter() - start_time
            
            # Calculate final view factor
            emitter_area = emitter.area
            receiver_area = receiver.area
            view_factor = (emitter_area * receiver_area * final_integral) / emitter_area
            
            # Estimate uncertainty based on tolerance
            uncertainty = self._tolerance * view_factor if converged else 0.1 * view_factor
            
            return ViewFactorResult(
                value=view_factor,
                uncertainty=uncertainty,
                converged=converged,
                iterations=total_cells,
                computation_time=computation_time,
                method_used=self._name
            )
            
        except Exception as e:
            logger.error(f"Adaptive calculation failed: {e}")
            raise
    
    def _create_initial_mesh(self, grid_size: int = 4) -> List[AdaptiveCell]:
        """Create initial uniform mesh.
        
        Args:
            grid_size: Number of cells per dimension
            
        Returns:
            List of initial cells
        """
        cells = []
        cell_size = 1.0 / grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                cell = AdaptiveCell(
                    s_min=i * cell_size,
                    s_max=(i + 1) * cell_size,
                    t_min=j * cell_size,
                    t_max=(j + 1) * cell_size,
                    depth=0
                )
                cells.append(cell)
        
        return cells
    
    def _adaptive_refinement(self, 
                           initial_cells: List[AdaptiveCell],
                           emitter: Rectangle,
                           receiver: Rectangle) -> Tuple[float, bool, int]:
        """Perform adaptive mesh refinement.
        
        Args:
            initial_cells: Initial mesh cells
            emitter: Source rectangle
            receiver: Target rectangle
            
        Returns:
            Tuple of (final_integral, converged, total_cells)
        """
        # Priority queue for cells needing refinement (max-heap by error)
        refinement_queue = []
        converged_cells = []
        total_integral = 0.0
        
        # Calculate initial estimates
        for cell in initial_cells:
            integral, error = self._estimate_cell_integral(cell, emitter, receiver)
            cell.integral = integral
            cell.error = error
            total_integral += integral
            
            # Add to refinement queue (negative error for max-heap)
            heapq.heappush(refinement_queue, (-error, id(cell), cell))
        
        iteration = 0
        max_iterations = 1000
        start_time = time.perf_counter()
        
        while refinement_queue and iteration < max_iterations:
            # Check time limit
            if hasattr(self, '_time_limit') and (time.perf_counter() - start_time) > self._time_limit:
                logger.warning(f"Reached time limit: {self._time_limit:.1f}s")
                break
            # Get cell with largest error
            neg_error, _, current_cell = heapq.heappop(refinement_queue)
            current_error = -neg_error
            
            # Check convergence
            if len(converged_cells) + len(refinement_queue) > self._max_cells:
                logger.warning(f"Reached maximum cells limit: {self._max_cells}")
                break
            
            relative_error = current_error / max(abs(total_integral), EPS)
            if relative_error < self._tolerance:
                # Cell has converged
                converged_cells.append(current_cell)
                # Move all remaining cells to converged
                while refinement_queue:
                    _, _, cell = heapq.heappop(refinement_queue)
                    converged_cells.append(cell)
                break
            
            # Check depth limit
            if current_cell.depth >= self._max_depth:
                converged_cells.append(current_cell)
                continue
            
            # Subdivide cell
            subcells = self._subdivide_cell(current_cell)
            
            # Calculate refined estimates
            refined_total = 0.0
            for subcell in subcells:
                integral, error = self._estimate_cell_integral(subcell, emitter, receiver)
                subcell.integral = integral
                subcell.error = error
                refined_total += integral
            
            # Update total integral
            total_integral = total_integral - current_cell.integral + refined_total
            
            # Add subcells to appropriate queues
            for subcell in subcells:
                subcell_relative_error = subcell.error / max(abs(total_integral), EPS)
                if subcell_relative_error < self._tolerance:
                    converged_cells.append(subcell)
                else:
                    heapq.heappush(refinement_queue, (-subcell.error, id(subcell), subcell))
            
            iteration += 1
        
        # Calculate final result
        final_integral = sum(cell.integral for cell in converged_cells)
        final_integral += sum(cell.integral for _, _, cell in refinement_queue)
        
        converged = len(refinement_queue) == 0 and iteration < max_iterations
        status = STATUS_CONVERGED if converged else STATUS_FAILED
        if not converged and iteration >= max_iterations:
            status = STATUS_REACHED_LIMITS
        elif not converged and hasattr(self, '_time_limit') and (time.perf_counter() - start_time) > self._time_limit:
            status = STATUS_REACHED_LIMITS
        total_cells = len(converged_cells) + len(refinement_queue)
        
        return final_integral, converged, total_cells
    
    def _subdivide_cell(self, cell: AdaptiveCell) -> List[AdaptiveCell]:
        """Subdivide cell into 4 subcells (2×2 refinement).
        
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
    
    def _estimate_cell_integral(self, 
                              cell: AdaptiveCell,
                              emitter: Rectangle,
                              receiver: Rectangle) -> Tuple[float, float]:
        """Estimate integral and error for a cell.
        
        Args:
            cell: Cell to integrate over
            emitter: Source rectangle
            receiver: Target rectangle
            
        Returns:
            Tuple of (integral_estimate, error_estimate)
        """
        # Use Gauss-Legendre quadrature over the cell
        integral = 0.0
        
        # Map quadrature points to cell
        ds = cell.s_max - cell.s_min
        dt = cell.t_max - cell.t_min
        
        for i in range(self._n_src_gl):
            for j in range(self._n_src_gl):
                # Source point in parameter space
                s = cell.s_min + ds * self._x_src[i]
                t = cell.t_min + dt * self._x_src[j]
                
                # Convert to world coordinates
                p1 = emitter.origin + s * emitter.u_vector + t * emitter.v_vector
                
                # Integrate over receiver
                inner_integral = self._integrate_over_receiver(p1, emitter, receiver)
                
                # Add weighted contribution
                weight = self._w_src[i] * self._w_src[j] * ds * dt
                integral += inner_integral * weight
        
        # Estimate error as fraction of integral (heuristic)
        error = abs(integral) * 0.1 / (2 ** cell.depth)  # Error decreases with refinement
        
        return integral, error
    
    def _integrate_over_receiver(self, 
                               source_point: np.ndarray,
                               emitter: Rectangle,
                               receiver: Rectangle) -> float:
        """Integrate view factor kernel over receiver surface.
        
        Args:
            source_point: Point on emitter surface
            emitter: Source rectangle
            receiver: Target rectangle
            
        Returns:
            Integral of view factor kernel over receiver
        """
        integral = 0.0
        
        # Get surface normals
        n1 = emitter.normal
        n2 = receiver.normal
        
        # Auto-orient normals to face each other
        centre_direction = receiver.centroid - emitter.centroid
        if np.dot(n1, centre_direction) < 0:
            n1 = -n1
        if np.dot(n2, -centre_direction) < 0:
            n2 = -n2
        
        # Gauss-Legendre quadrature over receiver
        for i in range(self._n_tgt_gl):
            for j in range(self._n_tgt_gl):
                # Receiver point in parameter space
                u = self._x_tgt[i]
                v = self._x_tgt[j]
                
                # Convert to world coordinates
                p2 = receiver.origin + u * receiver.u_vector + v * receiver.v_vector
                
                # Calculate view factor kernel
                r_vec = p2 - source_point
                r = np.linalg.norm(r_vec)
                
                if r > EPS:  # Avoid division by zero
                    r_hat = r_vec / r
                    
                    cos1 = np.dot(n1, r_hat)
                    cos2 = -np.dot(n2, r_hat)
                    
                    if cos1 > 0 and cos2 > 0:
                        kernel = (cos1 * cos2) / (np.pi * r * r)
                        weight = self._w_tgt[i] * self._w_tgt[j]
                        integral += kernel * weight
        
        return integral
