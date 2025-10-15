from __future__ import annotations
import numpy as np
from typing import Any, Dict
from .grid_tap import drain as _drain_grid

def attach_grid_field(result: Dict[str, Any], Y, Z, F) -> None:
    """Attach arrays for plotting to the result in a consistent schema."""
    if Y is None or Z is None or F is None:
        return
    if not isinstance(result, dict):
        return
    gd = result.setdefault("grid_data", {})
    if gd is None:
        result["grid_data"] = {}
        gd = result["grid_data"]
    gd.update({"Y": Y, "Z": Z, "F": F})
    result["Y"] = Y
    result["Z"] = Z
    result["F"] = F
    
    # Also store 1D grid arrays for plotting compatibility
    if Y.ndim == 2 and Z.ndim == 2:
        # Extract 1D arrays from 2D meshgrid
        result["grid_y"] = Y[0, :]  # First row of Y meshgrid
        result["grid_z"] = Z[:, 0]  # First column of Z meshgrid
    else:
        # Already 1D arrays
        result["grid_y"] = Y
        result["grid_z"] = Z

def attach_field_from_tap(result: Dict[str, Any]) -> Dict[str, Any]:
    """Try to attach field data from the grid tap, return updated result."""
    tapped = _drain_grid()
    if tapped is not None:
        Y, Z, F = tapped
        attach_grid_field(result, Y, Z, F)
    return result

def has_field_data(result: Dict[str, Any]) -> bool:
    """Check if result has field data (Y, Z, F)."""
    return all(key in result for key in ["Y", "Z", "F"]) and all(result[key] is not None for key in ["Y", "Z", "F"])

def has_field(result: Dict[str, Any]) -> bool:
    """
    Safe check for attached field arrays without triggering NumPy truth-value errors.
    Requires Y,Z,F to be numpy arrays with identical shapes and non-zero size.
    """
    Y = result.get("Y", None)
    Z = result.get("Z", None)
    F = result.get("F", None)
    if (Y is None) or (Z is None) or (F is None):
        return False
    try:
        # Shape and size checks (avoid boolean context on arrays)
        return (
            hasattr(Y, "shape") and hasattr(Z, "shape") and hasattr(F, "shape")
            and Y.shape == Z.shape == F.shape
            and np.size(Y) > 0 and np.size(Z) > 0 and np.size(F) > 0
        )
    except Exception:
        return False
