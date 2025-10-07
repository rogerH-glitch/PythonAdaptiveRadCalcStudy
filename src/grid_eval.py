from __future__ import annotations

# Shim to preserve old import paths; re-export tapped implementations
from .rc_eval.grid_eval import (
    sample_receiver_grid,
    evaluate_grid,
    peak_from_field,
)

__all__ = ["sample_receiver_grid", "evaluate_grid", "peak_from_field"]


