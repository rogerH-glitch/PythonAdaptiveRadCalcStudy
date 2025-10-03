"""
Fixed grid view factor calculations.

This module implements view factor calculations using uniform grid
subdivision with regular quadrature points.
"""

import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from typing import Tuple
import logging

from .constants import EPS, STATUS_CONVERGED, STATUS_REACHED_LIMITS, STATUS_FAILED
from .geometry import Rectangle, ViewFactorResult, validate_geometry

logger = logging.getLogger(__name__)


class FixedGridCalculator:
    """Calculator using fixed uniform grid subdivision.
    
    Implements view factor calculation by subdividing the emitter surface
    into a regular grid and using numerical quadrature at each grid point.
    """
    
    def __init__(self, nx: int = 100, ny: int = 100, quadrature: str = 'centroid') -> None:
        """Initialise fixed grid calculator.
        
        Args:
            nx: Number of grid points in x-direction
            ny: Number of grid points in y-direction  
            quadrature: Quadrature method ('centroid' or '2x2')
        """
        self._nx = nx
        self._ny = ny
        self._quadrature = quadrature
        self._name = "fixed_grid"
        
        if quadrature not in ['centroid', '2x2']:
            raise ValueError(f"Unknown quadrature method: {quadrature}")
    
    def calculate(self, emitter: Rectangle, receiver: Rectangle) -> ViewFactorResult:
        """Calculate view factor using fixed grid method.
        
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
            if self._quadrature == 'centroid':
                view_factor = self._calculate_centroid_quadrature(emitter, receiver)
            else:  # '2x2'
                view_factor = self._calculate_2x2_quadrature(emitter, receiver)
            
            computation_time = time.perf_counter() - start_time
            
            return ViewFactorResult(
                value=view_factor,
                uncertainty=0.0,  # No error estimation for fixed grid
                converged=True,
                iterations=self._nx * self._ny,
                computation_time=computation_time,
                method_used=self._name
            )
            
        except Exception as e:
            logger.error(f"Fixed grid calculation failed: {e}")
            raise
    
    def _calculate_centroid_quadrature(self, 
                                     emitter: Rectangle, 
                                     receiver: Rectangle) -> float:
        """Calculate using centroid quadrature (1-point per cell).
        
        Args:
            emitter: Source rectangle
            receiver: Target rectangle
            
        Returns:
            View factor value
        """
        total_integral = 0.0
        
        # Grid spacing
        dx = 1.0 / self._nx
        dy = 1.0 / self._ny
        
        # Cell area in parameter space
        cell_area = dx * dy
        
        for i in range(self._nx):
            for j in range(self._ny):
                # Cell centre in parameter space [0,1] × [0,1]
                s = (i + 0.5) * dx
                t = (j + 0.5) * dy
                
                # Convert to world coordinates
                p1 = emitter.origin + s * emitter.u_vector + t * emitter.v_vector
                
                # Calculate integral over receiver for this source point
                inner_integral = self._integrate_over_receiver(p1, emitter, receiver)
                
                # Add contribution (weighted by cell area)
                total_integral += inner_integral * cell_area
        
        # Multiply by physical areas
        emitter_area = emitter.area
        receiver_area = receiver.area
        
        return (emitter_area * receiver_area * total_integral) / emitter_area
    
    def _calculate_2x2_quadrature(self, 
                                emitter: Rectangle, 
                                receiver: Rectangle) -> float:
        """Calculate using 2×2 Gauss-Legendre quadrature per cell.
        
        Args:
            emitter: Source rectangle
            receiver: Target rectangle
            
        Returns:
            View factor value
        """
        # Get 2-point Gauss-Legendre quadrature on [0,1]
        xi, wi = leggauss(2)
        x_quad = 0.5 * (xi + 1.0)  # Transform to [0,1]
        w_quad = 0.5 * wi
        
        total_integral = 0.0
        
        # Grid spacing
        dx = 1.0 / self._nx
        dy = 1.0 / self._ny
        
        for i in range(self._nx):
            for j in range(self._ny):
                # Cell bounds in parameter space
                s_min = i * dx
                s_max = (i + 1) * dx
                t_min = j * dy
                t_max = (j + 1) * dy
                
                # 2×2 quadrature over this cell
                cell_integral = 0.0
                for qi in range(2):
                    for qj in range(2):
                        # Quadrature point in parameter space
                        s = s_min + (s_max - s_min) * x_quad[qi]
                        t = t_min + (t_max - t_min) * x_quad[qj]
                        
                        # Convert to world coordinates
                        p1 = emitter.origin + s * emitter.u_vector + t * emitter.v_vector
                        
                        # Calculate integral over receiver
                        inner_integral = self._integrate_over_receiver(p1, emitter, receiver)
                        
                        # Add weighted contribution
                        weight = w_quad[qi] * w_quad[qj] * dx * dy
                        cell_integral += inner_integral * weight
                
                total_integral += cell_integral
        
        # Multiply by physical areas
        emitter_area = emitter.area
        receiver_area = receiver.area
        
        return (emitter_area * receiver_area * total_integral) / emitter_area
    
    def _integrate_over_receiver(self, 
                               source_point: np.ndarray,
                               emitter: Rectangle,
                               receiver: Rectangle) -> float:
        """Integrate view factor kernel over receiver surface.
        
        Args:
            source_point: Point on emitter surface
            emitter: Source rectangle (for normal vector)
            receiver: Target rectangle
            
        Returns:
            Integral of view factor kernel over receiver
        """
        # Use fixed quadrature over receiver (moderate resolution)
        n_recv = 20  # Fixed resolution for receiver integration
        
        # Get quadrature points and weights
        xi, wi = leggauss(n_recv)
        x_quad = 0.5 * (xi + 1.0)
        w_quad = 0.5 * wi
        
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
        
        for i in range(n_recv):
            for j in range(n_recv):
                # Receiver point in parameter space
                u = x_quad[i]
                v = x_quad[j]
                
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
                        weight = w_quad[i] * w_quad[j]
                        integral += kernel * weight
        
        return integral
