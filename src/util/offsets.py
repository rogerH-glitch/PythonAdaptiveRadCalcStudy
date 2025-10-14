from __future__ import annotations
import warnings

warnings.warn(
    "Module 'src.util.offsets' is deprecated; offset extraction has moved into src.viz.geometry_utils (translate_to_setback/extract offsets). This shim will be removed in v1.2.",
    DeprecationWarning,
    stacklevel=2,
)
from typing import Tuple

def get_receiver_offset(args) -> Tuple[float, float]:
    """
    Return (dy, dz) as receiver_center − emitter_center from CLI args.
    Falls back to the negation of emitter_offset if only that is given.
    """
    # Prefer explicit receiver_offset if present
    ro = getattr(args, "receiver_offset", None)
    if ro is not None:
        dy, dz = ro
        return float(dy), float(dz)
    # Else use -emitter_offset if present
    eo = getattr(args, "emitter_offset", None)
    if eo is not None:
        dy, dz = eo
        return -float(dy), -float(dz)
    return 0.0, 0.0

