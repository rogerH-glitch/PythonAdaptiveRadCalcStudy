from __future__ import annotations
from typing import Any, Dict

def attach_grid_field(result: Dict[str, Any], Y, Z, F) -> None:
    """Attach arrays for plotting to the result in a consistent schema."""
    if Y is None or Z is None or F is None:
        return
    gd = result.setdefault("grid_data", {})
    gd.update({"Y": Y, "Z": Z, "F": F})
    result["Y"] = Y
    result["Z"] = Z
    result["F"] = F
