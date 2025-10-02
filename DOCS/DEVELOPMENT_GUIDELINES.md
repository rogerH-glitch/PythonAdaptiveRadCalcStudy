# Development Guidelines for AI Tools and Developers

**Document:** Development Guidelines  
**Version:** 1.0  
**Date:** October 2, 2025  
**Target Audience:** AI Tools, Automated Code Generators, Human Developers  

---

## Quick Reference for AI Tools

### 🚀 Essential Checklist for Code Generation

When generating code for this fire safety engineering project, ensure:

1. **✅ SOLID Compliance**: Every class has single responsibility, follows open/closed principle
2. **✅ Australian English**: Use "colour", "centre", "normalise", "analyse" 
3. **✅ Type Safety**: All functions have explicit type hints, no `Any` types
4. **✅ Error Handling**: Specific exception types with descriptive messages
5. **✅ Performance**: Must complete single calculations within 3 seconds
6. **✅ Safety Validation**: Include bounds checking and cross-validation
7. **✅ Documentation**: Docstrings for all public methods with examples
8. **✅ Testing**: Generate corresponding test cases with fixtures

### 🎯 Fire Safety Context Awareness

This is a **safety-critical fire engineering application**. Generated code must:

- **Validate physical bounds**: View factors must be 0 ≤ F ≤ 1
- **Handle edge cases**: Near-contact geometries, degenerate surfaces
- **Cross-validate results**: Compare multiple calculation methods
- **Maintain audit trails**: Log all calculations for safety compliance
- **Use engineering units**: Metres, kW/m², degrees Celsius
- **Follow Australian standards**: AS 3959, NCC, fire safety terminology

---

## Code Generation Templates

### 1. Class Template (SOLID Compliant)

```python
from abc import ABC, abstractmethod
from typing import Protocol, TypeVar, Generic
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T')

class CalculationMethod(Protocol):
    """Protocol defining calculation method interface."""
    
    def calculate(self, emitter: 'Rectangle', receiver: 'Rectangle') -> 'ViewFactorResult':
        """Calculate view factor between surfaces."""
        ...

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

class ViewFactorCalculator(ABC):
    """Abstract base class for view factor calculation methods.
    
    Implements common functionality and defines interface for
    specific calculation methods (adaptive, fixed grid, Monte Carlo).
    """
    
    def __init__(self, name: str) -> None:
        """Initialise calculator with method name."""
        self._name = name
        self._calculation_count = 0
    
    @abstractmethod
    def _calculate_implementation(self, 
                                emitter: Rectangle, 
                                receiver: Rectangle) -> 'ViewFactorResult':
        """Implement specific calculation method."""
        pass
    
    def calculate(self, emitter: Rectangle, receiver: Rectangle) -> 'ViewFactorResult':
        """Calculate view factor with validation and error handling.
        
        Args:
            emitter: Source surface geometry
            receiver: Target surface geometry
            
        Returns:
            ViewFactorResult with calculated value and metadata
            
        Raises:
            GeometryError: If surface geometries are invalid
            CalculationError: If calculation fails
        """
        # Validate inputs
        self._validate_geometry(emitter, receiver)
        
        # Perform calculation with error handling
        try:
            start_time = time.perf_counter()
            result = self._calculate_implementation(emitter, receiver)
            computation_time = time.perf_counter() - start_time
            
            # Validate result
            self._validate_result(result)
            
            # Update metadata
            result.computation_time = computation_time
            result.method_used = self._name
            
            self._calculation_count += 1
            logger.info(f"Calculated view factor: {result.value:.6f} using {self._name}")
            
            return result
            
        except Exception as e:
            logger.error(f"Calculation failed with {self._name}: {e}")
            raise CalculationError(f"View factor calculation failed: {e}") from e
    
    def _validate_geometry(self, emitter: Rectangle, receiver: Rectangle) -> None:
        """Validate geometry for fire safety calculations."""
        # Check positive areas
        if emitter.area <= 0:
            raise GeometryError(f"Emitter area must be positive, got {emitter.area}")
        if receiver.area <= 0:
            raise GeometryError(f"Receiver area must be positive, got {receiver.area}")
        
        # Check reasonable dimensions for fire safety
        max_dimension = 1000.0  # metres
        if emitter.max_dimension > max_dimension:
            raise GeometryError(f"Emitter dimension {emitter.max_dimension}m exceeds limit")
        if receiver.max_dimension > max_dimension:
            raise GeometryError(f"Receiver dimension {receiver.max_dimension}m exceeds limit")
    
    def _validate_result(self, result: 'ViewFactorResult') -> None:
        """Validate calculation result."""
        if not (0.0 <= result.value <= 1.0):
            raise CalculationError(f"View factor {result.value} outside physical bounds [0,1]")
        
        if np.isnan(result.value) or np.isinf(result.value):
            raise CalculationError("View factor calculation produced NaN or Inf")

# Custom exceptions
class ViewFactorError(Exception):
    """Base exception for view factor calculations."""
    pass

class GeometryError(ViewFactorError):
    """Raised when geometry is invalid."""
    pass

class CalculationError(ViewFactorError):
    """Raised when calculation fails."""
    pass
```

### 2. Test Template

```python
import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import Tuple

class TestViewFactorCalculator:
    """Comprehensive tests for view factor calculations."""
    
    @pytest.fixture
    def valid_rectangles(self) -> Tuple[Rectangle, Rectangle]:
        """Fixture providing valid rectangle pair for testing."""
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
    
    @pytest.fixture
    def calculator(self) -> ViewFactorCalculator:
        """Fixture providing calculator instance."""
        return ConcreteViewFactorCalculator()
    
    def test_calculate_returns_valid_result(self, 
                                          calculator: ViewFactorCalculator,
                                          valid_rectangles: Tuple[Rectangle, Rectangle]) -> None:
        """Test that calculation returns valid result within bounds."""
        emitter, receiver = valid_rectangles
        
        result = calculator.calculate(emitter, receiver)
        
        # Validate result properties
        assert isinstance(result, ViewFactorResult)
        assert 0.0 <= result.value <= 1.0
        assert result.computation_time > 0
        assert result.method_used == calculator._name
    
    def test_invalid_geometry_raises_error(self, calculator: ViewFactorCalculator) -> None:
        """Test that invalid geometry raises appropriate error."""
        # Zero-area emitter
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
        
        with pytest.raises(GeometryError, match="positive area"):
            calculator.calculate(invalid_emitter, valid_receiver)
    
    @pytest.mark.parametrize("setback,expected,tolerance", [
        (0.05, 0.998805, 0.003),
        (1.0, 0.70274, 0.003),
        (3.8, 0.17735, 0.003),
    ])
    def test_validation_cases(self, 
                            calculator: ViewFactorCalculator,
                            setback: float, 
                            expected: float,
                            tolerance: float) -> None:
        """Test against PRD validation cases."""
        # Create parallel rectangles with specified setback
        emitter = Rectangle.create_parallel_pair(
            width=5.1, height=2.1, setback=setback
        )[0]
        receiver = Rectangle.create_parallel_pair(
            width=5.1, height=2.1, setback=setback
        )[1]
        
        result = calculator.calculate(emitter, receiver)
        
        relative_error = abs(result.value - expected) / expected
        assert relative_error < tolerance, (
            f"Validation case failed: setback={setback}m, "
            f"expected={expected}, got={result.value}, "
            f"error={relative_error:.4f} > {tolerance}"
        )
    
    def test_performance_requirement(self, 
                                   calculator: ViewFactorCalculator,
                                   valid_rectangles: Tuple[Rectangle, Rectangle]) -> None:
        """Test that calculation meets 3-second performance requirement."""
        emitter, receiver = valid_rectangles
        
        start_time = time.perf_counter()
        result = calculator.calculate(emitter, receiver)
        elapsed_time = time.perf_counter() - start_time
        
        assert elapsed_time < 3.0, f"Calculation took {elapsed_time:.2f}s > 3.0s limit"
        assert result.computation_time < 3.0
    
    def test_australian_english_terminology(self) -> None:
        """Test that Australian English is used in error messages and docs."""
        # Check docstrings use Australian spelling
        assert "colour" in SomeClass.__doc__.lower()
        assert "centre" in SomeClass.__doc__.lower()
        
        # Check error messages use Australian terminology
        with pytest.raises(GeometryError) as exc_info:
            invalid_operation()
        
        error_message = str(exc_info.value).lower()
        assert "metre" in error_message or "analysis" in error_message
```

### 3. Performance Monitoring Template

```python
import time
import functools
import logging
from typing import Callable, Any, Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for fire safety calculations."""
    
    method_name: str
    execution_time: float
    memory_usage: float = 0.0
    iterations: int = 0
    converged: bool = True
    performance_grade: str = field(init=False)
    
    def __post_init__(self) -> None:
        """Calculate performance grade based on execution time."""
        if self.execution_time < 1.0:
            self.performance_grade = "Excellent"
        elif self.execution_time < 2.0:
            self.performance_grade = "Good"  
        elif self.execution_time < 3.0:
            self.performance_grade = "Acceptable"
        else:
            self.performance_grade = "Poor - Exceeds Requirements"

def monitor_performance(max_time_seconds: float = 3.0):
    """Decorator to monitor calculation performance."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                
                # Log performance metrics
                metrics = PerformanceMetrics(
                    method_name=func.__name__,
                    execution_time=execution_time,
                    iterations=getattr(result, 'iterations', 0),
                    converged=getattr(result, 'converged', True)
                )
                
                logger.info(f"Performance: {metrics.performance_grade} - "
                           f"{execution_time:.3f}s for {func.__name__}")
                
                if execution_time > max_time_seconds:
                    logger.warning(f"Performance requirement exceeded: "
                                 f"{execution_time:.2f}s > {max_time_seconds}s")
                
                # Attach metrics to result if possible
                if hasattr(result, 'performance_metrics'):
                    result.performance_metrics = metrics
                
                return result
                
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                logger.error(f"Function {func.__name__} failed after {execution_time:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator
```

---

## AI Tool Integration Guidelines

### For Code Generation AI Tools

1. **Context Awareness**: Always consider fire safety engineering context
2. **Safety First**: Include validation and bounds checking in every function
3. **Australian Standards**: Use Australian English and fire safety terminology
4. **Performance**: Generate code that meets 3-second calculation requirement
5. **Testing**: Generate comprehensive test cases alongside implementation
6. **Documentation**: Include detailed docstrings with fire safety examples

### For Code Review AI Tools

1. **SOLID Compliance**: Check for single responsibility, proper abstraction
2. **Type Safety**: Ensure all functions have explicit type hints
3. **Error Handling**: Verify specific exception types and descriptive messages
4. **Performance**: Flag any code that might exceed 3-second limit
5. **Safety Validation**: Ensure physical bounds checking and cross-validation
6. **Australian English**: Check spelling and terminology consistency

### For Automated Testing AI Tools

1. **Validation Cases**: Include PRD test cases (UC-001 through UC-006)
2. **Edge Cases**: Test near-contact geometries, degenerate surfaces
3. **Performance Tests**: Verify 3-second calculation requirement
4. **Safety Tests**: Check bounds validation and error handling
5. **Integration Tests**: Test method combinations and cross-validation
6. **Property-Based Tests**: Use hypothesis for geometric property testing

---

## Common Patterns and Anti-Patterns

### ✅ Recommended Patterns

```python
# Pattern 1: Immutable data classes for geometry
@dataclass(frozen=True)
class Rectangle:
    origin: np.ndarray
    u_vector: np.ndarray
    v_vector: np.ndarray

# Pattern 2: Protocol-based interfaces
class CalculationMethod(Protocol):
    def calculate(self, emitter: Rectangle, receiver: Rectangle) -> ViewFactorResult: ...

# Pattern 3: Explicit error types
class GeometryError(ViewFactorError):
    """Specific error for geometry validation failures."""
    pass

# Pattern 4: Performance monitoring
@monitor_performance(max_time_seconds=3.0)
def calculate_view_factor(self, emitter: Rectangle, receiver: Rectangle) -> ViewFactorResult:
    pass

# Pattern 5: Safety validation
def _validate_physical_bounds(self, result: ViewFactorResult) -> None:
    if not (0.0 <= result.value <= 1.0):
        raise CalculationError(f"View factor {result.value} outside bounds [0,1]")
```

### ❌ Anti-Patterns to Avoid

```python
# Anti-pattern 1: Mutable geometry (unsafe for concurrent access)
class Rectangle:
    def __init__(self):
        self.origin = [0, 0, 0]  # Mutable list
        self.width = 0           # Can be modified after creation

# Anti-pattern 2: Generic exceptions
def calculate():
    if error_condition:
        raise Exception("Something went wrong")  # Too generic

# Anti-pattern 3: No type hints
def calculate_view_factor(emitter, receiver):  # Missing types
    return some_value  # Unknown return type

# Anti-pattern 4: Hard-coded values
def validate_geometry(rect):
    if rect.area > 100:  # Magic number, should be configurable
        raise ValueError("Too large")

# Anti-pattern 5: No performance consideration
def slow_calculation():
    # Nested loops without time limits
    for i in range(1000000):
        for j in range(1000000):
            expensive_operation()  # Could run forever
```

---

## File Organisation Standards

### Directory Structure
```
src/
├── core/                    # Core calculation engines (max 700 lines each)
├── geometry/               # Geometric utilities and validation
├── results/                # Result handling and export
├── validation/             # Test cases and validation
└── utils/                  # Common utilities and configuration
```

### File Naming Conventions
- Use snake_case for Python files: `adaptive_calculator.py`
- Use descriptive names: `nistir_6925_validation.py` not `test1.py`
- Include purpose in name: `geometry_validator.py` not `validator.py`

### Import Organisation
```python
# Standard library imports
import time
import logging
from typing import Protocol, TypeVar
from dataclasses import dataclass

# Third-party imports  
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from .geometry import Rectangle
from .results import ViewFactorResult
from .utils import PerformanceMonitor
```

---

This document provides comprehensive guidance for AI tools and developers working on the Fire Safety Radiation View Factor project. Following these guidelines ensures code quality, safety, and maintainability in this critical fire safety engineering application.
