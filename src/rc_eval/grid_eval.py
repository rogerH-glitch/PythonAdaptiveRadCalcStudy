from __future__ import annotations
import numpy as np
from src.util.grid_tap import capture as _tap_capture

def sample_receiver_grid(width: float, height: float, ny: int = 81, nz: int = 61):
    """
    Uniform Y×Z grid over the receiver (coords relative to the receiver centre).
    Returns (Y, Z) shaped (nz, ny) using indexing='xy' convention.
    """
    ys = np.linspace(-width/2.0, width/2.0, ny)
    zs = np.linspace(-height/2.0, height/2.0, nz)
    Y, Z = np.meshgrid(ys, zs, indexing="xy")
    return Y, Z

def evaluate_grid(make_point_vf, Y, Z, dy: float, dz: float):
    """
    Evaluate field at receiver grid points, correctly shifted by the receiver offset.
    make_point_vf must support vectorised (y,z) arrays measured relative to the *emitter centre*.
    """
    F = make_point_vf(Y + dy, Z + dz)
    # One-liner tap: capture the actual evaluated field for downstream plotting
    try:
        _tap_capture(Y, Z, F)
    except Exception:
        # Best-effort capture; never interfere with evaluation
        pass
    return F

def peak_from_field(F, Y, Z):
    """
    Return (F_peak, y_peak, z_peak) where y/z are relative to the receiver centre.
    """
    idx = np.nanargmax(F)
    i, j = np.unravel_index(idx, F.shape)
    return float(F[i, j]), float(Y[i, j]), float(Z[i, j])


