# Coding Standards for Fire Safety Radiation View Factor Tool

**Document:** Coding Standards & Best Practices  
**Version:** 1.0  
**Date:** October 2, 2025  
**Applies to:** All Python code in the Fire Safety Radiation View Factor project  

---

## Overview

This document establishes coding standards for AI tools and developers working on the Fire Safety Radiation View Factor validation tool. These standards ensure code quality, maintainability, safety, and performance in fire safety engineering applications.

## Core Principles

### 1. **SOLID Principles** 🏗️

All code must strictly follow SOLID principles:

#### Single Responsibility Principle (SRP)
```python
# ✅ GOOD: Each class has one responsibility
class ViewFactorCalculator:
    """Calculates view factors using specified method."""
    
    def calculate_adaptive(self, emitter: Rectangle, receiver: Rectangle) -> float:
        """Calculate view factor using adaptive integration."""
        pass

class GeometryValidator:
    """Validates geometric inputs for view factor calculations."""
    
    def validate_rectangle(self, rect: Rectangle) -> ValidationResult:
        """Validate rectangle geometry parameters."""
        pass

# ❌ BAD: Multiple responsibilities in one class
class ViewFactorProcessor:
    def calculate_view_factor(self): pass
    def validate_input(self): pass
    def save_results(self): pass
    def plot_graphs(self): pass
```

#### Open/Closed Principle (OCP)
```python
# ✅ GOOD: Open for extension, closed for modification
from abc import ABC, abstractmethod

class ViewFactorMethod(ABC):
    """Abstract base for view factor calculation methods."""
    
    @abstractmethod
    def calculate(self, emitter: Rectangle, receiver: Rectangle) -> ViewFactorResult:
        """Calculate view factor between surfaces."""
        pass

class AdaptiveIntegrationMethod(ViewFactorMethod):
    """Adaptive integration implementation."""
    
    def calculate(self, emitter: Rectangle, receiver: Rectangle) -> ViewFactorResult:
        # Implementation specific to adaptive method
        pass

class MonteCarloMethod(ViewFactorMethod):
    """Monte Carlo implementation."""
    
    def calculate(self, emitter: Rectangle, receiver: Rectangle) -> ViewFactorResult:
        # Implementation specific to Monte Carlo method
        pass
```

#### Liskov Substitution Principle (LSP)
```python
# ✅ GOOD: Derived classes can replace base classes
def calculate_view_factors(method: ViewFactorMethod, 
                          emitter: Rectangle, 
                          receiver: Rectangle) -> ViewFactorResult:
    """Calculate using any valid method implementation."""
    return method.calculate(emitter, receiver)

# Works with any ViewFactorMethod implementation
adaptive_method = AdaptiveIntegrationMethod()
monte_carlo_method = MonteCarloMethod()

result1 = calculate_view_factors(adaptive_method, emitter, receiver)
result2 = calculate_view_factors(monte_carlo_method, emitter, receiver)
```

#### Interface Segregation Principle (ISP)
```python
# ✅ GOOD: Specific interfaces for different capabilities
class Calculable(Protocol):
    """Interface for calculation capability."""
    def calculate(self) -> float: ...

class Visualisable(Protocol):
    """Interface for visualisation capability."""
    def plot(self) -> matplotlib.Figure: ...

class Exportable(Protocol):
    """Interface for export capability."""
    def export_csv(self, filepath: Path) -> None: ...

# ❌ BAD: Fat interface forcing unnecessary dependencies
class ViewFactorProcessor(Protocol):
    def calculate(self) -> float: ...
    def plot(self) -> matplotlib.Figure: ...
    def export_csv(self) -> None: ...
    def send_email(self) -> None: ...  # Not all implementations need this
```

#### Dependency Inversion Principle (DIP)
```python
# ✅ GOOD: Depend on abstractions, not concretions
class ViewFactorEngine:
    """High-level view factor calculation engine."""
    
    def __init__(self, 
                 calculator: ViewFactorMethod,
                 validator: GeometryValidator,
                 exporter: ResultExporter):
        self._calculator = calculator
        self._validator = validator
        self._exporter = exporter
    
    def process(self, emitter: Rectangle, receiver: Rectangle) -> None:
        """Process view factor calculation with validation and export."""
        validation_result = self._validator.validate_geometry(emitter, receiver)
        if not validation_result.is_valid:
            raise GeometryValidationError(validation_result.errors)
        
        result = self._calculator.calculate(emitter, receiver)
        self._exporter.export(result)
```

### 2. **DRY (Don't Repeat Yourself)** 🔄

Eliminate code duplication through proper abstraction:

```python
# ✅ GOOD: Common functionality extracted
class GeometryUtils:
    """Utility functions for geometric calculations."""
    
    @staticmethod
    def calculate_area(u_vector: np.ndarray, v_vector: np.ndarray) -> float:
        """Calculate area from two vectors defining a rectangle."""
        return np.linalg.norm(np.cross(u_vector, v_vector))
    
    @staticmethod
    def calculate_centroid(origin: np.ndarray, u_vector: np.ndarray, v_vector: np.ndarray) -> np.ndarray:
        """Calculate centroid of rectangle."""
        return origin + 0.5 * (u_vector + v_vector)

# Usage in multiple classes
class AdaptiveMethod:
    def _calculate_cell_area(self, cell: RectangleCell) -> float:
        return GeometryUtils.calculate_area(cell.u_vector, cell.v_vector)

class FixedGridMethod:
    def _calculate_grid_area(self, grid_cell: GridCell) -> float:
        return GeometryUtils.calculate_area(grid_cell.u_vector, grid_cell.v_vector)
```

### 3. **KISS (Keep It Simple, Stupid)** 💡

Favour simple, clear implementations:

```python
# ✅ GOOD: Simple, clear implementation
def is_valid_setback_distance(distance: float) -> bool:
    """Check if setback distance is valid for fire safety calculations."""
    return distance > 0.0 and distance < 1000.0  # Reasonable engineering limits

# ❌ BAD: Overly complex for simple validation
def is_valid_setback_distance(distance: float) -> bool:
    """Check if setback distance is valid."""
    validation_rules = {
        'positive': lambda x: x > 0.0,
        'reasonable_upper': lambda x: x < 1000.0,
        'not_nan': lambda x: not np.isnan(x),
        'not_inf': lambda x: not np.isinf(x)
    }
    
    return all(rule(distance) for rule in validation_rules.values())
```

### 4. **Python Style Guide** (Adapted from Google Standards) 📝

Following PEP 8 with Google Python Style Guide adaptations:

```python
# ✅ GOOD: Proper Python naming and structure
class ViewFactorCalculationEngine:
    """Engine for calculating view factors in fire safety applications.
    
    This class implements the primary calculation engine following
    NISTIR 6925 adaptive integration methodology.
    
    Attributes:
        tolerance: Convergence tolerance for adaptive integration
        max_depth: Maximum recursion depth for mesh refinement
    """
    
    def __init__(self, tolerance: float = 3e-3, max_depth: int = 12) -> None:
        """Initialise calculation engine with specified parameters.
        
        Args:
            tolerance: Relative tolerance for convergence (default: 3e-3)
            max_depth: Maximum recursion depth (default: 12)
            
        Raises:
            ValueError: If tolerance is not positive or max_depth < 1
        """
        if tolerance <= 0:
            raise ValueError("Tolerance must be positive")
        if max_depth < 1:
            raise ValueError("Maximum depth must be at least 1")
            
        self._tolerance = tolerance
        self._max_depth = max_depth
    
    def calculate_view_factor(self, 
                            emitter: Rectangle, 
                            receiver: Rectangle) -> ViewFactorResult:
        """Calculate view factor between emitter and receiver surfaces.
        
        Args:
            emitter: Source surface geometry
            receiver: Target surface geometry
            
        Returns:
            ViewFactorResult containing calculated value and metadata
            
        Raises:
            GeometryError: If surface geometries are invalid
            ConvergenceError: If calculation fails to converge
        """
        # Implementation here
        pass
```

### 5. **Australian English** 🇦🇺

Use Australian English spelling and terminology throughout:

```python
# ✅ GOOD: Australian English spelling
class ColourMap:
    """Colour mapping utilities for visualisation."""
    
    def normalise_values(self, values: List[float]) -> List[float]:
        """Normalise values for colour mapping."""
        pass
    
    def centre_colorbar(self, figure: plt.Figure) -> None:
        """Centre the colourbar on the figure."""
        pass

# Fire safety terminology
class FireSafetyCalculator:
    """Calculator for fire safety engineering analyses."""
    
    def calculate_radiant_heat_flux(self, view_factor: float, 
                                  fire_temperature: float) -> float:
        """Calculate radiant heat flux in kW/m²."""
        pass
    
    def assess_building_separation(self, separation_distance: float) -> str:
        """Assess adequacy of building separation per Australian standards."""
        pass

# ❌ BAD: American English spelling
class ColorMap:  # Should be ColourMap
    def normalize_values(self): pass  # Should be normalise_values
    def center_colorbar(self): pass   # Should be centre_colorbar
```

### 6. **Comprehensive Testing** 🧪

All code must include thorough tests:

```python
# test_view_factor_calculator.py
import pytest
import numpy as np
from unittest.mock import Mock, patch

class TestViewFactorCalculator:
    """Comprehensive tests for ViewFactorCalculator."""
    
    @pytest.fixture
    def calculator(self) -> ViewFactorCalculator:
        """Fixture providing calculator instance."""
        return ViewFactorCalculator(tolerance=1e-3, max_depth=10)
    
    @pytest.fixture
    def parallel_rectangles(self) -> Tuple[Rectangle, Rectangle]:
        """Fixture providing parallel rectangle configuration."""
        emitter = Rectangle(
            origin=np.array([0.0, 0.0, 0.0]),
            u_vector=np.array([5.1, 0.0, 0.0]),
            v_vector=np.array([0.0, 2.1, 0.0])
        )
        receiver = Rectangle(
            origin=np.array([0.0, 0.0, 1.0]),
            u_vector=np.array([5.1, 0.0, 0.0]),
            v_vector=np.array([0.0, 2.1, 0.0])
        )
        return emitter, receiver
    
    def test_calculate_view_factor_parallel_surfaces(self, 
                                                   calculator: ViewFactorCalculator,
                                                   parallel_rectangles: Tuple[Rectangle, Rectangle]) -> None:
        """Test view factor calculation for parallel surfaces."""
        emitter, receiver = parallel_rectangles
        
        result = calculator.calculate_view_factor(emitter, receiver)
        
        # Validate against known reference value
        expected_value = 0.70274  # From validation test case UC-002
        assert abs(result.value - expected_value) < 0.003  # Within ±0.3%
        assert result.converged is True
        assert result.iterations > 0
    
    def test_invalid_geometry_raises_error(self, calculator: ViewFactorCalculator) -> None:
        """Test that invalid geometry raises appropriate error."""
        # Zero-area rectangle
        invalid_emitter = Rectangle(
            origin=np.array([0.0, 0.0, 0.0]),
            u_vector=np.array([0.0, 0.0, 0.0]),  # Zero vector
            v_vector=np.array([0.0, 2.1, 0.0])
        )
        valid_receiver = Rectangle(
            origin=np.array([0.0, 0.0, 1.0]),
            u_vector=np.array([5.1, 0.0, 0.0]),
            v_vector=np.array([0.0, 2.1, 0.0])
        )
        
        with pytest.raises(GeometryError, match="Zero area surface"):
            calculator.calculate_view_factor(invalid_emitter, valid_receiver)
    
    @pytest.mark.parametrize("setback,expected", [
        (0.05, 0.998805),
        (1.0, 0.70274),
        (3.8, 0.17735),
    ])
    def test_validation_cases(self, 
                            calculator: ViewFactorCalculator,
                            setback: float, 
                            expected: float) -> None:
        """Test against validation test cases from PRD."""
        emitter = Rectangle.create_from_dimensions(
            centre=np.array([2.55, 1.05, 0.0]),
            width=5.1, height=2.1,
            normal=np.array([0.0, 0.0, 1.0])
        )
        receiver = Rectangle.create_from_dimensions(
            centre=np.array([2.55, 1.05, setback]),
            width=5.1, height=2.1,
            normal=np.array([0.0, 0.0, -1.0])
        )
        
        result = calculator.calculate_view_factor(emitter, receiver)
        
        relative_error = abs(result.value - expected) / expected
        assert relative_error < 0.003, f"Error {relative_error:.4f} exceeds ±0.3% tolerance"
```

### 7. **Documentation Standards** 📚

All public APIs must be thoroughly documented:

```python
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
        
    Example:
        >>> result = calculator.calculate_view_factor(emitter, receiver)
        >>> print(f"View factor: {result.value:.6f} ± {result.uncertainty:.6f}")
        >>> if result.converged:
        ...     print(f"Converged in {result.iterations} iterations")
    """
    
    def __init__(self, 
                 value: float,
                 uncertainty: float = 0.0,
                 converged: bool = True,
                 iterations: int = 0,
                 computation_time: float = 0.0,
                 method_used: str = "unknown") -> None:
        """Initialise view factor result.
        
        Args:
            value: Calculated view factor value
            uncertainty: Estimated uncertainty (default: 0.0)
            converged: Convergence status (default: True)
            iterations: Number of iterations (default: 0)
            computation_time: Computation time in seconds (default: 0.0)
            method_used: Name of calculation method (default: "unknown")
            
        Raises:
            ValueError: If value is not in valid range [0, 1]
            ValueError: If uncertainty is negative
        """
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"View factor must be in range [0, 1], got {value}")
        if uncertainty < 0.0:
            raise ValueError(f"Uncertainty must be non-negative, got {uncertainty}")
            
        self.value = value
        self.uncertainty = uncertainty
        self.converged = converged
        self.iterations = iterations
        self.computation_time = computation_time
        self.method_used = method_used
    
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
            
        Example:
            >>> result = ViewFactorResult(0.70274, uncertainty=0.001)
            >>> result.is_within_tolerance(0.70200, tolerance=0.01)  # ±1%
            True
        """
        if reference_value == 0.0:
            return abs(self.value) <= tolerance
        
        relative_error = abs(self.value - reference_value) / abs(reference_value)
        return relative_error <= tolerance
```

### 8. **Safety-Critical Reliability** ⚠️

All changes must maintain safety-critical reliability:

```python
class SafetyValidatedCalculator:
    """Safety-validated view factor calculator for fire engineering.
    
    This calculator includes multiple safety checks and validation
    layers to ensure reliable results for fire safety applications.
    """
    
    def __init__(self, enable_safety_checks: bool = True) -> None:
        """Initialise with safety validation enabled by default."""
        self._safety_checks_enabled = enable_safety_checks
        self._calculation_history: List[CalculationRecord] = []
    
    def calculate_view_factor(self, 
                            emitter: Rectangle, 
                            receiver: Rectangle) -> ViewFactorResult:
        """Calculate view factor with comprehensive safety validation.
        
        Safety checks performed:
        1. Geometry validation (positive areas, valid orientations)
        2. Physical bounds checking (0 ≤ F ≤ 1)
        3. Numerical stability verification
        4. Cross-validation with alternative method
        5. Result logging for audit trail
        """
        if self._safety_checks_enabled:
            self._validate_geometry_safety(emitter, receiver)
        
        # Primary calculation
        primary_result = self._calculate_adaptive(emitter, receiver)
        
        if self._safety_checks_enabled:
            # Cross-validation with fixed grid method
            validation_result = self._calculate_fixed_grid(emitter, receiver)
            self._cross_validate_results(primary_result, validation_result)
            
            # Physical bounds check
            self._validate_physical_bounds(primary_result)
            
            # Log for audit trail
            self._log_calculation(emitter, receiver, primary_result)
        
        return primary_result
    
    def _validate_geometry_safety(self, emitter: Rectangle, receiver: Rectangle) -> None:
        """Validate geometry for safety-critical applications."""
        # Check for degenerate geometries
        if emitter.area < 1e-10:
            raise SafetyError("Emitter area too small for reliable calculation")
        if receiver.area < 1e-10:
            raise SafetyError("Receiver area too small for reliable calculation")
        
        # Check for reasonable engineering dimensions
        max_dimension = 1000.0  # metres
        if (emitter.max_dimension > max_dimension or 
            receiver.max_dimension > max_dimension):
            raise SafetyError(f"Surface dimensions exceed {max_dimension}m limit")
        
        # Check separation distance
        separation = self._calculate_separation(emitter, receiver)
        if separation < 0.01:  # 1cm minimum
            raise SafetyError("Surfaces too close for reliable calculation")
    
    def _cross_validate_results(self, 
                              primary: ViewFactorResult, 
                              validation: ViewFactorResult) -> None:
        """Cross-validate results between methods."""
        if primary.value == 0.0 and validation.value == 0.0:
            return  # Both zero is acceptable
        
        if primary.value == 0.0 or validation.value == 0.0:
            raise SafetyError("Method disagreement: one result is zero")
        
        relative_difference = abs(primary.value - validation.value) / max(primary.value, validation.value)
        if relative_difference > 0.10:  # 10% disagreement threshold
            raise SafetyError(f"Method disagreement: {relative_difference:.1%} difference")
    
    def _validate_physical_bounds(self, result: ViewFactorResult) -> None:
        """Validate result is within physical bounds."""
        if not (0.0 <= result.value <= 1.0):
            raise SafetyError(f"View factor {result.value} outside physical bounds [0,1]")
        
        if np.isnan(result.value) or np.isinf(result.value):
            raise SafetyError("View factor calculation produced NaN or Inf")

class SafetyError(Exception):
    """Exception raised for safety-critical validation failures."""
    pass
```

### 9. **Performance Requirements** ⚡

Must meet PRD performance requirements (≤3s single case):

```python
import time
from functools import wraps
from typing import Callable, Any

def performance_monitor(max_time_seconds: float = 3.0):
    """Decorator to monitor and enforce performance requirements."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                elapsed_time = time.perf_counter() - start_time
                
                if elapsed_time > max_time_seconds:
                    logger.warning(
                        f"{func.__name__} exceeded performance target: "
                        f"{elapsed_time:.2f}s > {max_time_seconds:.2f}s"
                    )
                
                # Add timing metadata if result supports it
                if hasattr(result, 'computation_time'):
                    result.computation_time = elapsed_time
                
                return result
                
            except Exception as e:
                elapsed_time = time.perf_counter() - start_time
                logger.error(f"{func.__name__} failed after {elapsed_time:.2f}s: {e}")
                raise
        
        return wrapper
    return decorator

class PerformanceOptimisedCalculator:
    """Calculator optimised for performance requirements."""
    
    @performance_monitor(max_time_seconds=3.0)
    def calculate_view_factor(self, 
                            emitter: Rectangle, 
                            receiver: Rectangle) -> ViewFactorResult:
        """Calculate view factor within performance bounds."""
        # Use performance-optimised algorithm selection
        if self._should_use_fast_approximation(emitter, receiver):
            return self._calculate_fast_approximation(emitter, receiver)
        else:
            return self._calculate_adaptive_optimised(emitter, receiver)
    
    def _should_use_fast_approximation(self, 
                                     emitter: Rectangle, 
                                     receiver: Rectangle) -> bool:
        """Determine if fast approximation is appropriate."""
        # Use fast method for simple geometries
        separation = self._calculate_separation(emitter, receiver)
        max_dimension = max(emitter.max_dimension, receiver.max_dimension)
        
        # If separation >> surface dimensions, use far-field approximation
        return separation > 5.0 * max_dimension
```

### 10. **Complex Algorithm Comments** 💭

Complex algorithms must include comprehensive inline comments:

```python
def adaptive_integration_1ai(self, 
                           emitter: Rectangle, 
                           receiver: Rectangle,
                           tolerance: float = 3e-3) -> ViewFactorResult:
    """
    Adaptive integration method per NISTIR 6925 (Walton, 2002).
    
    This implements the single-area integration (1AI) method with
    recursive mesh refinement based on local error estimation.
    """
    # Initialise adaptive mesh with coarse grid
    # Start with 4x4 subdivision as per NISTIR 6925 recommendations
    initial_cells = self._create_initial_mesh(emitter, grid_size=4)
    
    # Priority queue for cells needing refinement (max-heap by error)
    refinement_queue = []
    converged_cells = []
    total_integral = 0.0
    
    # Calculate initial estimates for all cells
    for cell in initial_cells:
        # Estimate integral over this cell using Gauss-Legendre quadrature
        # Higher-order quadrature (6-point) for initial estimate
        cell_integral, cell_error = self._estimate_cell_integral(
            cell, receiver, quadrature_order=6
        )
        
        # Store cell with error estimate for potential refinement
        # Use negative error for max-heap behaviour (heapq is min-heap)
        heapq.heappush(refinement_queue, (-cell_error, cell, cell_integral))
        total_integral += cell_integral
    
    iteration = 0
    max_iterations = 1000  # Safety limit to prevent infinite loops
    
    while refinement_queue and iteration < max_iterations:
        # Get cell with largest error estimate
        neg_error, current_cell, old_integral = heapq.heappop(refinement_queue)
        current_error = -neg_error
        
        # Check if this cell's error is below tolerance
        # Use relative error: |error| / |total_integral|
        relative_error = current_error / max(abs(total_integral), 1e-12)
        if relative_error < tolerance:
            # This cell (and all remaining) are within tolerance
            converged_cells.append((current_cell, old_integral))
            # Move all remaining cells to converged list
            while refinement_queue:
                _, cell, integral = heapq.heappop(refinement_queue)
                converged_cells.append((cell, integral))
            break
        
        # Subdivide cell into 4 subcells (2x2 refinement)
        # This is the core of the adaptive algorithm
        subcells = self._subdivide_cell(current_cell)
        
        # Calculate refined estimates for each subcell
        subcell_integrals = []
        subcell_errors = []
        refined_total = 0.0
        
        for subcell in subcells:
            # Use higher-order quadrature for refined calculation
            # 8-point Gauss-Legendre for better accuracy
            subcell_integral, subcell_error = self._estimate_cell_integral(
                subcell, receiver, quadrature_order=8
            )
            subcell_integrals.append(subcell_integral)
            subcell_errors.append(subcell_error)
            refined_total += subcell_integral
        
        # Update total integral (remove old, add refined)
        total_integral = total_integral - old_integral + refined_total
        
        # Add subcells back to refinement queue if they need further refinement
        for i, subcell in enumerate(subcells):
            if subcell_errors[i] > tolerance * abs(total_integral):
                heapq.heappush(refinement_queue, 
                             (-subcell_errors[i], subcell, subcell_integrals[i]))
            else:
                # Subcell has converged
                converged_cells.append((subcell, subcell_integrals[i]))
        
        iteration += 1
    
    # Check convergence status
    converged = len(refinement_queue) == 0 and iteration < max_iterations
    
    # Calculate final result from converged cells
    final_integral = sum(integral for _, integral in converged_cells)
    
    # Estimate overall uncertainty from remaining unconverged cells
    remaining_error = sum(-neg_error for neg_error, _, _ in refinement_queue)
    uncertainty = remaining_error / max(abs(final_integral), 1e-12)
    
    return ViewFactorResult(
        value=final_integral,
        uncertainty=uncertainty,
        converged=converged,
        iterations=iteration,
        method_used="adaptive_1ai"
    )
```

### 11-25. **Additional Standards**

#### 11. Single Responsibility
Each method and class has one well-defined purpose (see SOLID examples above).

#### 12. Clear Naming
```python
# ✅ GOOD: Descriptive, self-documenting names
def calculate_differential_view_factor_at_point(
    source_point: np.ndarray,
    target_point: np.ndarray,
    source_normal: np.ndarray,
    target_normal: np.ndarray
) -> float:
    """Calculate differential view factor between two points."""
    pass

# ❌ BAD: Unclear abbreviations
def calc_dvf(sp, tp, sn, tn):
    pass
```

#### 13. No Hard-coded Secrets
```python
# ✅ GOOD: Configuration-based settings
class Configuration:
    """Application configuration with environment-based settings."""
    
    def __init__(self):
        self.nist_validation_url = os.getenv(
            'NIST_VALIDATION_URL', 
            'https://nvlpubs.nist.gov/nistpubs/ir/2002/NIST.IR.6925.pdf'
        )
        self.results_directory = Path(os.getenv('RESULTS_DIR', './results'))
        
# ❌ BAD: Hard-coded sensitive information
API_KEY = "sk-1234567890abcdef"  # Never do this!
DATABASE_PASSWORD = "secret123"   # Never do this!
```

#### 14. Comments First
```python
# Write comments before implementing complex logic
def calculate_gauss_legendre_quadrature(n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Gauss-Legendre quadrature points and weights.
    
    Algorithm:
    1. Generate Legendre polynomial roots (quadrature points)
    2. Calculate corresponding weights using derivative formula
    3. Transform from [-1,1] to [0,1] interval for rectangle integration
    4. Apply Jacobian transformation for proper scaling
    
    Mathematical basis:
    - Legendre polynomials are orthogonal on [-1,1] with weight function w(x)=1
    - n-point Gauss-Legendre quadrature exactly integrates polynomials up to degree 2n-1
    - Transformation: x_[0,1] = (x_[-1,1] + 1) / 2, weight scaling by 1/2
    """
    # Step 1: Generate Legendre polynomial roots
    # Use numpy's built-in function which implements the Golub-Welsch algorithm
    points_standard, weights_standard = np.polynomial.legendre.leggauss(n_points)
    
    # Step 2: Transform from [-1,1] to [0,1] interval
    # Linear transformation: x_new = (x_old + 1) / 2
    points_unit = 0.5 * (points_standard + 1.0)
    
    # Step 3: Scale weights for new interval
    # Jacobian of transformation is 1/2
    weights_unit = 0.5 * weights_standard
    
    return points_unit, weights_unit
```

#### 15. Test-Driven Development
```python
# Write tests first, then implement
class TestViewFactorValidation:
    """Test-driven development for view factor validation."""
    
    def test_nistir_6925_case_should_match_reference_within_tolerance(self):
        """
        NISTIR 6925 validation case must match published results.
        
        Reference: Table 1, ε=1e-4 → F₁,₂ = 0.11562055
        """
        # Arrange: Set up NISTIR 6925 geometry
        surface_1 = Rectangle.unit_square(origin=[0, 0, 0])
        surface_2 = Rectangle.unit_square(origin=[0, 0, 1])
        occluder_3 = Rectangle.from_dimensions(0.5, 0.5, origin=[0.25, 0.25, 0.75])
        occluder_4 = Rectangle.from_dimensions(0.5, 0.5, origin=[0.25, 0.25, 0.75])
        
        calculator = AdaptiveViewFactorCalculator(tolerance=1e-4)
        
        # Act: Calculate with occluders
        result = calculator.calculate_with_occluders(
            emitter=surface_1,
            receiver=surface_2,
            occluders=[occluder_3, occluder_4]
        )
        
        # Assert: Must match reference within ±0.3%
        expected = 0.11562055
        assert abs(result.value - expected) / expected < 0.003
        assert result.converged is True
        
    # Now implement the actual calculation method to make test pass
```

#### 16. Explicit Error Handling
```python
class ViewFactorError(Exception):
    """Base exception for view factor calculations."""
    pass

class GeometryError(ViewFactorError):
    """Raised when geometry is invalid for calculation."""
    pass

class ConvergenceError(ViewFactorError):
    """Raised when calculation fails to converge."""
    pass

class PerformanceError(ViewFactorError):
    """Raised when calculation exceeds performance limits."""
    pass

def calculate_view_factor_with_explicit_error_handling(
    emitter: Rectangle, 
    receiver: Rectangle
) -> ViewFactorResult:
    """Calculate view factor with comprehensive error handling."""
    
    try:
        # Validate inputs
        if emitter.area <= 0:
            raise GeometryError(f"Emitter has invalid area: {emitter.area}")
        if receiver.area <= 0:
            raise GeometryError(f"Receiver has invalid area: {receiver.area}")
        
        # Perform calculation with timeout
        with timeout_context(seconds=3.0):
            result = _perform_adaptive_calculation(emitter, receiver)
        
        # Validate result
        if not result.converged:
            raise ConvergenceError(
                f"Failed to converge after {result.iterations} iterations. "
                f"Final error: {result.uncertainty:.2e}"
            )
        
        return result
        
    except TimeoutError as e:
        raise PerformanceError(f"Calculation exceeded 3-second limit: {e}")
    except np.linalg.LinAlgError as e:
        raise GeometryError(f"Linear algebra error (degenerate geometry?): {e}")
    except Exception as e:
        # Log unexpected errors for debugging
        logger.error(f"Unexpected error in view factor calculation: {e}")
        raise ViewFactorError(f"Calculation failed: {e}") from e
```

#### 17-25. Additional Standards Summary

- **Secure by Default**: Input validation, bounds checking, safe defaults
- **Performance Boundaries**: Monitoring, limits, early termination
- **Minimal Dependencies**: Only numpy, scipy, matplotlib, pytest as core
- **YAGNI**: Implement only required features from PRD
- **Code for Readers**: Clear structure, good naming, comprehensive docs
- **Explicit Types**: Full type hints throughout
- **No "Any" Types**: Specific types or Union types instead
- **Internationalisation**: Support for metric units, Australian terminology
- **File Length Limit**: Maximum 700 lines per file, split larger modules

---

## Project Structure Following Standards

```
src/
├── __init__.py
├── core/                          # Core calculation engines
│   ├── __init__.py
│   ├── adaptive_calculator.py     # Adaptive integration (1AI)
│   ├── fixed_grid_calculator.py   # Fixed grid method
│   ├── monte_carlo_calculator.py  # Monte Carlo method
│   └── analytical_calculator.py   # Analytical solutions
├── geometry/                      # Geometric utilities
│   ├── __init__.py
│   ├── rectangle.py              # Rectangle class and utilities
│   ├── validators.py             # Geometry validation
│   └── transformations.py        # Coordinate transformations
├── results/                       # Result handling
│   ├── __init__.py
│   ├── view_factor_result.py     # Result data class
│   ├── exporters.py              # CSV, JSON export
│   └── visualisation.py          # Plotting utilities
├── validation/                    # Validation and testing
│   ├── __init__.py
│   ├── nistir_cases.py           # NISTIR 6925 test cases
│   ├── reference_values.py       # Known reference values
│   └── test_runners.py           # Automated validation
└── utils/                         # Utilities
    ├── __init__.py
    ├── performance.py            # Performance monitoring
    ├── safety.py                 # Safety validation
    └── configuration.py          # Configuration management
```

---

## Enforcement and Review

### Code Review Checklist

- [ ] Follows SOLID principles
- [ ] No code duplication (DRY)
- [ ] Simple, clear implementation (KISS)
- [ ] Australian English spelling
- [ ] Comprehensive tests included
- [ ] All public APIs documented
- [ ] Safety checks implemented
- [ ] Performance requirements met
- [ ] Complex algorithms commented
- [ ] Single responsibility per class/method
- [ ] Clear, descriptive naming
- [ ] No hard-coded secrets
- [ ] Explicit error handling
- [ ] Type hints throughout
- [ ] File under 700 lines

### Automated Checks

```bash
# Style and formatting
black --check src/
flake8 src/
mypy src/

# Testing
pytest tests/ --cov=src --cov-report=html

# Performance testing
pytest tests/performance/ --benchmark-only

# Security scanning
bandit -r src/
```

---

**This document is mandatory for all code contributions to the Fire Safety Radiation View Factor project. Non-compliance will result in code review rejection.**
