"""
Analytical baseline approximations for radiation view factors.

This module provides baseline analytical approximations for unobstructed,
parallel, centre-aligned rectangular surfaces. These are NOT the sophisticated
Walton 1AI adaptive integration methods, but rather simple fixed-grid
numerical integrations of the standard differential view factor formula.

Author: Fire Safety Engineer
Date: 2024
"""

from __future__ import annotations
import numpy as np
from typing import Tuple

# Numerical stability epsilon
EPS = 1e-12


def local_peak_vf_analytic_approx(
    em_w: float, 
    em_h: float,
    rc_w: float, 
    rc_h: float,
    setback: float
) -> float:
    """
    Compute the local peak (point-to-point differential) view factor at the 
    receiver centre from a rectangular emitter surface.
    
    This is a BASELINE APPROXIMATION using fixed-grid numerical integration
    of the standard differential view factor formula:
        dF = (cosθ1 * cosθ2) / (π * r²) dA_emitter
    
    For parallel, centre-aligned surfaces: cosθ1 = cosθ2 = setback / r
    
    This is NOT the sophisticated Walton 1AI adaptive integration method.
    It uses a modest fixed grid (200×200) for baseline validation purposes only.
    
    Args:
        em_w: Emitter width (m)
        em_h: Emitter height (m) 
        rc_w: Receiver width (m)
        rc_h: Receiver height (m)
        setback: Separation distance between surfaces (m)
        
    Returns:
        Local peak view factor at receiver centre (dimensionless, 0 ≤ F ≤ 1)
        
    Raises:
        ValueError: If any dimension is non-positive
        
    Notes:
        - Assumes parallel surfaces, centres aligned, facing each other
        - Uses 200×200 fixed grid for emitter discretisation
        - Clamps denominators with EPS=1e-12 for numerical stability
        - Result is clamped to [0, 1] range
    """
    # Validate inputs
    if em_w <= 0 or em_h <= 0 or rc_w <= 0 or rc_h <= 0 or setback <= 0:
        raise ValueError("All dimensions and setback must be positive")
    
    # Fixed grid resolution for baseline approximation
    nx, ny = 200, 200
    
    # Emitter grid spacing
    dx = em_w / nx
    dy = em_h / ny
    dA = dx * dy  # Differential area element
    
    # Receiver centre coordinates (origin at emitter centre)
    rc_x, rc_y, rc_z = 0.0, 0.0, setback
    
    # Emitter bounds (centred at origin)
    em_x_min, em_x_max = -em_w / 2, em_w / 2
    em_y_min, em_y_max = -em_h / 2, em_h / 2
    
    # Integrate over emitter surface
    vf_total = 0.0
    
    for i in range(nx):
        for j in range(ny):
            # Emitter element centre coordinates
            em_x = em_x_min + (i + 0.5) * dx
            em_y = em_y_min + (j + 0.5) * dy
            em_z = 0.0
            
            # Vector from emitter element to receiver point
            r_vec = np.array([rc_x - em_x, rc_y - em_y, rc_z - em_z])
            r_mag = np.linalg.norm(r_vec)
            
            # Avoid division by zero
            if r_mag < EPS:
                continue
                
            # Unit vector
            r_hat = r_vec / r_mag
            
            # Surface normals (both pointing towards each other)
            em_normal = np.array([0.0, 0.0, 1.0])  # Emitter normal (+z)
            rc_normal = np.array([0.0, 0.0, -1.0])  # Receiver normal (-z)
            
            # Cosines of angles between normals and connecting vector
            cos_theta1 = np.dot(em_normal, r_hat)  # Emitter angle
            cos_theta2 = np.dot(-r_hat, rc_normal)  # Receiver angle (note sign)
            
            # Ensure cosines are non-negative (surfaces facing each other)
            cos_theta1 = max(0.0, cos_theta1)
            cos_theta2 = max(0.0, cos_theta2)
            
            # Differential view factor contribution
            # dF = (cosθ1 * cosθ2) / (π * r²) * dA
            dF = (cos_theta1 * cos_theta2 * dA) / (np.pi * (r_mag**2 + EPS))
            
            vf_total += dF
    
    # Clamp result to valid range [0, 1]
    vf_total = max(0.0, min(1.0, vf_total))
    
    return vf_total


def validate_geometry(
    em_w: float, 
    em_h: float,
    rc_w: float, 
    rc_h: float,
    setback: float
) -> Tuple[bool, str]:
    """
    Validate geometry parameters for analytical calculations.
    
    Args:
        em_w: Emitter width (m)
        em_h: Emitter height (m)
        rc_w: Receiver width (m) 
        rc_h: Receiver height (m)
        setback: Separation distance (m)
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if em_w <= 0:
        return False, f"Emitter width must be positive, got {em_w}"
    if em_h <= 0:
        return False, f"Emitter height must be positive, got {em_h}"
    if rc_w <= 0:
        return False, f"Receiver width must be positive, got {rc_w}"
    if rc_h <= 0:
        return False, f"Receiver height must be positive, got {rc_h}"
    if setback <= 0:
        return False, f"Setback must be positive, got {setback}"
        
    return True, ""


def get_analytical_info() -> str:
    """
    Return information string about the analytical method.
    
    Returns:
        Description of the analytical baseline method
    """
    return (
        "Analytical Baseline: Fixed-grid (200×200) numerical integration of "
        "differential view factor formula for parallel, centre-aligned rectangles. "
        "This is a baseline approximation, not the sophisticated Walton 1AI method."
    )