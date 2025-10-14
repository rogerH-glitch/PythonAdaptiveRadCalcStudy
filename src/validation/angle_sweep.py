from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple
import numpy as np

# Import the individual method modules
from src.peak_locator import create_vf_evaluator
from src.montecarlo import vf_point_montecarlo
from src.fixed_grid import vf_point_fixed_grid
from src.analytical import vf_point_rect_to_point


@dataclass(frozen=True)
class SweepConfig:
    emitter_w: float = 2.0
    emitter_h: float = 1.0
    receiver_w: float = 2.0
    receiver_h: float = 1.0
    setback: float = 5.0
    dy: float = 0.0
    dz: float = 0.0
    rotate_target: str = "emitter"  # or "receiver"
    rotate_axis: str = "z"          # "z" (yaw) or "y" (pitch)
    pivot: str = "toe"              # keep minimum setback


def angle_sweep(angles_deg: Iterable[float], method: str, cfg: SweepConfig) -> List[Tuple[float, float]]:
    """
    Compute a pointwise (center) VF across angles.
    Returns list of (angle_deg, vf) for convenience.
    """
    results: List[Tuple[float, float]] = []
    
    for ang in angles_deg:
        # Create geometry configuration for methods that need it
        geom = {
            'emitter_width': cfg.emitter_w,
            'emitter_height': cfg.emitter_h,
            'setback': cfg.setback,
            'rotate_axis': cfg.rotate_axis,
            'angle': ang,
            'angle_pivot': cfg.pivot,
            'dy': cfg.dy,
            'dz': cfg.dz
        }
        
        # Create geometry config for evaluator-based methods
        geom_cfg = {
            'emitter_offset': (0.0, 0.0),
            'receiver_offset': (cfg.dy, cfg.dz),
            'angle_deg': ang,
            'rotate_target': cfg.rotate_target
        }
        
        # Receiver center point (0, 0) for center evaluation
        receiver_yz = (0.0, 0.0)
        
        # Call appropriate method
        if method == "adaptive":
            # Use the proper evaluator approach
            evaluator = create_vf_evaluator(
                method="adaptive",
                em_w=cfg.emitter_w, em_h=cfg.emitter_h,
                rc_w=cfg.receiver_w, rc_h=cfg.receiver_h,
                setback=cfg.setback, angle=ang, geom_cfg=geom_cfg
            )
            vf, _ = evaluator(receiver_yz[0], receiver_yz[1])
        elif method == "montecarlo":
            vf, _ = vf_point_montecarlo(receiver_yz, geom)
        elif method == "fixed_grid":
            vf, _ = vf_point_fixed_grid(receiver_yz, geom)
        elif method == "analytical":
            vf, _ = create_vf_evaluator(
                method="analytical",
                em_w=cfg.emitter_w, em_h=cfg.emitter_h,
                rc_w=cfg.receiver_w, rc_h=cfg.receiver_h,
                setback=cfg.setback, angle=ang, geom_cfg=geom_cfg
            )(receiver_yz[0], receiver_yz[1])
        else:
            raise ValueError(f"Unknown method: {method}")
        
        results.append((float(ang), float(vf)))
    
    return results
