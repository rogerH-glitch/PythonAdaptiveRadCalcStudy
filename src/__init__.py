"""
Radiation View Factor Validation Tool

A local Python tool for calculating local peak view factors between rectangular 
surfaces under fire conditions using multiple numerical methods.
"""

__version__ = "1.0.0"
__author__ = "Fire Safety Engineer"

# Import only what's currently implemented
from .analytical import local_peak_vf_analytic_approx, validate_geometry, get_analytical_info

__all__ = [
    "local_peak_vf_analytic_approx",
    "validate_geometry", 
    "get_analytical_info",
]
