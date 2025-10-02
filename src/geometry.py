"""
Geometric utilities and data structures for view factor calculations.

This module provides the core geometric classes and validation functions
for fire safety radiation calculations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class Rectangle:
    """Immutable rectangle geometry for fire safety calculations.
    
    Represents a rectangular surface in 3D space defined by origin point
    and two edge vectors. Used for emitter and receiver surfaces in
    view factor calculations.
    
    Attributes:
        origin: 3D coordinates of rectangle corner (metres)
        u_vector: First edge vector (metres)
        v_vector: Second edge vector (metres)
        
    Example:
        >>> # 5.1m × 2.1m rectangle at origin
        >>> rect = Rectangle(
        ...     origin=np.array([0.0, 0.0, 0.0]),
        ...     u_vector=np.array([5.1, 0.0, 0.0]),
        ...     v_vector=np.array([0.0, 2.1, 0.0])
        ... )
        >>> print(f"Area: {rect.area:.2f} m²")
        Area: 10.71 m²
    """
    
    origin: np.ndarray
    u_vector: np.ndarray  
    v_vector: np.ndarray
    
    def __post_init__(self) -> None:
        """Validate rectangle geometry after initialisation."""
        # Convert to numpy arrays if needed
        object.__setattr__(self, 'origin', np.asarray(self.origin, dtype=float))
        object.__setattr__(self, 'u_vector', np.asarray(self.u_vector, dtype=float))
        object.__setattr__(self, 'v_vector', np.asarray(self.v_vector, dtype=float))
        
        if self.area <= 0:
            raise ValueError(f"Rectangle must have positive area, got {self.area}")
        
        if np.any(np.isnan(self.origin)) or np.any(np.isinf(self.origin)):
            raise ValueError("Rectangle origin contains NaN or Inf values")
    
    @property
    def area(self) -> float:
        """Calculate rectangle area in square metres."""
        return float(np.linalg.norm(np.cross(self.u_vector, self.v_vector)))
    
    @property
    def normal(self) -> np.ndarray:
        """Calculate unit normal vector using right-hand rule."""
        cross_product = np.cross(self.u_vector, self.v_vector)
        norm = np.linalg.norm(cross_product)
        if norm == 0:
            raise ValueError("Cannot calculate normal for degenerate rectangle")
        return cross_product / norm
    
    @property
    def centroid(self) -> np.ndarray:
        """Calculate rectangle centroid coordinates."""
        return self.origin + 0.5 * (self.u_vector + self.v_vector)
    
    @property
    def width(self) -> float:
        """Width of rectangle (length of u_vector)."""
        return float(np.linalg.norm(self.u_vector))
    
    @property
    def height(self) -> float:
        """Height of rectangle (length of v_vector)."""
        return float(np.linalg.norm(self.v_vector))
    
    @classmethod
    def from_dimensions(cls, 
                       centre: np.ndarray,
                       width: float, 
                       height: float,
                       normal: np.ndarray) -> Rectangle:
        """Create rectangle from centre point, dimensions, and normal vector.
        
        Args:
            centre: Centre point of rectangle
            width: Width in metres
            height: Height in metres  
            normal: Normal vector direction
            
        Returns:
            Rectangle instance
        """
        # Normalise the normal vector
        n = np.asarray(normal, dtype=float)
        n = n / np.linalg.norm(n)
        
        # Create orthogonal basis vectors
        # Choose initial tangent vector
        if abs(n[0]) < 0.9:
            t = np.array([1.0, 0.0, 0.0])
        else:
            t = np.array([0.0, 1.0, 0.0])
        
        # Create orthonormal basis
        u_dir = np.cross(n, t)
        u_dir = u_dir / np.linalg.norm(u_dir)
        v_dir = np.cross(n, u_dir)
        
        # Scale by dimensions
        u_vector = width * u_dir
        v_vector = height * v_dir
        
        # Calculate origin from centre
        origin = centre - 0.5 * (u_vector + v_vector)
        
        return cls(origin=origin, u_vector=u_vector, v_vector=v_vector)


@dataclass
class ViewFactorResult:
    """Result of view factor calculation with metadata.
    
    This class encapsulates the result of a view factor calculation,
    including the calculated value, convergence information, and
    performance metrics.
    
    Attributes:
        value: Calculated view factor (dimensionless, 0 ≤ F ≤ 1)
        uncertainty: Estimated uncertainty in the calculation
        converged: Whether the calculation converged to tolerance
        iterations: Number of iterations performed
        computation_time: Wall-clock time for calculation (seconds)
        method_used: Name of calculation method employed
    """
    
    value: float
    uncertainty: float = 0.0
    converged: bool = True
    iterations: int = 0
    computation_time: float = 0.0
    method_used: str = "unknown"
    
    def __post_init__(self) -> None:
        """Validate result after initialisation."""
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(f"View factor must be in range [0, 1], got {self.value}")
        if self.uncertainty < 0.0:
            raise ValueError(f"Uncertainty must be non-negative, got {self.uncertainty}")
    
    def __str__(self) -> str:
        """String representation of result."""
        status = "converged" if self.converged else "not converged"
        return (f"ViewFactorResult(value={self.value:.6f}, "
                f"uncertainty=±{self.uncertainty:.6f}, "
                f"status={status})")
    
    def is_within_tolerance(self, reference_value: float, tolerance: float = 0.003) -> bool:
        """Check if result is within tolerance of reference value.
        
        Args:
            reference_value: Reference value for comparison
            tolerance: Relative tolerance (default: 0.003 for ±0.3%)
            
        Returns:
            True if within tolerance, False otherwise
        """
        if reference_value == 0.0:
            return abs(self.value) <= tolerance
        
        relative_error = abs(self.value - reference_value) / abs(reference_value)
        return relative_error <= tolerance


class GeometryError(Exception):
    """Exception raised for invalid geometry."""
    pass


class CalculationError(Exception):
    """Exception raised when calculation fails."""
    pass


def validate_geometry(emitter: Rectangle, receiver: Rectangle) -> None:
    """Validate geometry for fire safety calculations.
    
    Args:
        emitter: Source surface geometry
        receiver: Target surface geometry
        
    Raises:
        GeometryError: If geometry is invalid for calculation
    """
    # Check positive areas
    if emitter.area <= 0:
        raise GeometryError(f"Emitter area must be positive, got {emitter.area}")
    if receiver.area <= 0:
        raise GeometryError(f"Receiver area must be positive, got {receiver.area}")
    
    # Check reasonable dimensions for fire safety (max 1000m)
    max_dimension = 1000.0
    if max(emitter.width, emitter.height) > max_dimension:
        raise GeometryError(f"Emitter dimensions exceed {max_dimension}m limit")
    if max(receiver.width, receiver.height) > max_dimension:
        raise GeometryError(f"Receiver dimensions exceed {max_dimension}m limit")
    
    # Check minimum separation (1cm minimum for numerical stability)
    separation = np.linalg.norm(receiver.centroid - emitter.centroid)
    if separation < 0.01:
        raise GeometryError("Surfaces too close for reliable calculation (< 1cm)")


def calculate_separation_distance(emitter: Rectangle, receiver: Rectangle) -> float:
    """Calculate separation distance between rectangle centroids.
    
    Args:
        emitter: Source surface
        receiver: Target surface
        
    Returns:
        Distance between centroids in metres
    """
    return float(np.linalg.norm(receiver.centroid - emitter.centroid))
