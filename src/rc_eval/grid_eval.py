from __future__ import annotations
import logging
from typing import Tuple

import numpy as np
from src.util.grid_tap import capture as _tap_capture

log = logging.getLogger(__name__)


def sample_receiver_grid(width: float, height: float, ny: int = 81, nz: int = 61) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a uniform Y×Z grid over the receiver (coords relative to the receiver centre).
    Returns (Y, Z) shaped (nz, ny) using indexing='xy' convention.
    """
    ys = np.linspace(-width / 2.0, width / 2.0, ny)
    zs = np.linspace(-height / 2.0, height / 2.0, nz)
    Y, Z = np.meshgrid(ys, zs, indexing="xy")
    return Y, Z


def evaluate_grid(make_point_vf, Y: np.ndarray, Z: np.ndarray, dy: float, dz: float) -> np.ndarray:
    """
    Evaluate field at receiver grid points, correctly shifted by the receiver offset.
    make_point_vf must support vectorised (y,z) arrays measured relative to the *emitter centre*.
    Captures (Y,Z,F) for downstream plotting via src.util.grid_tap.
    """
    # VF field evaluated at receiver coords shifted by (dy,dz) (receiver_center − emitter_center).
    F = make_point_vf(Y + dy, Z + dz)

    # Tap once: capture the *actual* evaluated field for plotting later.
    try:
        _tap_capture(Y, Z, F)
        shp = getattr(F, "shape", None)
        log.info("[grid_eval] captured field via evaluate_grid() shape=%s", shp)
    except Exception as e:
        log.debug("[grid_eval] capture skipped: %s", e)

    return F


def peak_from_field(F: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Tuple[float, float, float]:
    """
    Return (F_peak, y_peak, z_peak) where y/z are relative to the receiver centre.
    """
    idx = np.nanargmax(F)
    i, j = np.unravel_index(idx, F.shape)
    return float(F[i, j]), float(Y[i, j]), float(Z[i, j])
