from __future__ import annotations
from typing import Any, Dict
import numpy as np

def attach_grid_field(result: Dict[str, Any], Y, Z, F) -> None:
    """
    Attach a plottable receiver-plane field to `result`, in a consistent schema:
      result['grid_data'] = {'Y': Y, 'Z': Z, 'F': F}
      result['Y'], result['Z'], result['F'] (mirrors for looser callers)
    Shapes:
      - Y, Z, F must be 2D arrays with identical shapes (nz, ny) using indexing='xy'
    """
    if Y is None or Z is None or F is None:
        return
    # Best-effort validation
    if getattr(Y, "shape", None) != getattr(Z, "shape", None) or getattr(Z, "shape", None) != getattr(F, "shape", None):
        raise ValueError("attach_grid_field: Y, Z, F must have identical 2D shapes")
    result.setdefault("grid_data", {})
    result["grid_data"].update({"Y": Y, "Z": Z, "F": F})
    result["Y"] = Y
    result["Z"] = Z
    result["F"] = F
