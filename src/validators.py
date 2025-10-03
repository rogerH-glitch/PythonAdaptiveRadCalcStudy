"""
Validation helpers for point vs area-average view factor diagnostics.

This module provides utilities to compute both point view factors at specific
receiver locations and area-averaged view factors across the receiver surface.
"""

from __future__ import annotations
import numpy as np
from .analytical import vf_point_rect_to_point_parallel


def receiver_area_average_point_integrand(em_w: float, em_h: float,
                                          rc_w: float, rc_h: float,
                                          setback: float,
                                          angle_deg: float = 0.0,
                                          nx_rc: int = 45, ny_rc: int = 19,
                                          nx_em: int = 180, ny_em: int = 180) -> float:
    """
    Approximate receiver area-averaged F by sampling a receiver grid (nx_rc x ny_rc),
    evaluating the *point* VF at each receiver node, then averaging.
    Uses the analytical point evaluator (parallel, angle=0) for speed.
    
    Args:
        em_w, em_h: Emitter dimensions
        rc_w, rc_h: Receiver dimensions
        setback: Setback distance
        angle_deg: Rotation angle (degrees) - currently only supports 0
        nx_rc, ny_rc: Receiver grid resolution
        nx_em, ny_em: Emitter integration resolution for each point
        
    Returns:
        Area-averaged view factor across receiver surface
    """
    # Only support parallel cases for now
    if abs(angle_deg) > 1e-6:
        raise NotImplementedError("Area-averaged view factor only supported for parallel cases (angle=0)")
    
    # Receiver grid centered at (0,0)
    dx_r = rc_w / nx_rc
    dy_r = rc_h / ny_rc
    xs_r = np.linspace(-rc_w/2 + 0.5*dx_r, rc_w/2 - 0.5*dx_r, nx_rc)
    ys_r = np.linspace(-rc_h/2 + 0.5*dy_r, rc_h/2 - 0.5*dy_r, ny_rc)

    vals = []
    for ry in ys_r:
        for rx in xs_r:
            Fp = vf_point_rect_to_point_parallel(em_w, em_h, setback, rx=rx, ry=ry,
                                                 nx=nx_em, ny=nx_em)
            vals.append(Fp)
    return float(np.mean(vals)) if vals else 0.0


def compute_point_vs_area_diagnostics(em_w: float, em_h: float, rc_w: float, rc_h: float,
                                      setback: float, angle_deg: float = 0.0,
                                      vf_point_center: float = None) -> dict:
    """
    Compute comprehensive point vs area-average diagnostics.
    
    Args:
        em_w, em_h: Emitter dimensions
        rc_w, rc_h: Receiver dimensions  
        setback: Setback distance
        angle_deg: Rotation angle (degrees)
        vf_point_center: Point VF at receiver center (if None, will compute)
        
    Returns:
        Dictionary with diagnostic values:
        - vf_point_center: Point VF at receiver center
        - vf_receiver_avg: Area-averaged VF across receiver
        - avg_gt_center: Boolean if area-avg > center by >0.5%
        - rel_diff: Relative difference (avg - center) / center
    """
    # Compute point VF at center if not provided
    if vf_point_center is None:
        vf_point_center = vf_point_rect_to_point_parallel(
            em_w, em_h, setback, rx=0.0, ry=0.0, nx=200, ny=200
        )
    
    # Compute area-averaged VF
    vf_receiver_avg = receiver_area_average_point_integrand(
        em_w, em_h, rc_w, rc_h, setback, angle_deg=angle_deg
    )
    
    # Compute relative difference
    rel_diff = (vf_receiver_avg - vf_point_center) / max(vf_point_center, 1e-12)
    avg_gt_center = vf_receiver_avg > vf_point_center * 1.005  # >0.5% higher
    
    return {
        'vf_point_center': vf_point_center,
        'vf_receiver_avg': vf_receiver_avg,
        'avg_gt_center': avg_gt_center,
        'rel_diff': rel_diff
    }
