from __future__ import annotations
"""
Field sampler for plotting only.
If a receiver field (Y,Z,F) was not captured by the solver, we sample a coarse
grid on the receiver plane and evaluate a *pointwise* view-factor kernel to
produce a heatmap for visualization. Physics is unchanged; this is plotting-only.
"""
from typing import Callable, Optional, Tuple
import numpy as np

# --- Kernel resolution --------------------------------------------------------
def _try_resolve_point_kernel() -> Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """
    Try to import a vectorized pointwise kernel from known locations.
    Expected signature: kernel(y, z) -> F (same shape), with y,z measured
    relative to the *emitter centre*. If the available function needs offsets,
    we will add them outside (y+dy, z+dz).
    Returns None if nothing resolvable.
    """
    # 1) src.analytical: prefer a dedicated pointwise kernel if present
    try:
        from src.analytical import point_vf as _point_vf  # y,z -> F
        return lambda y, z: _point_vf(y, z)
    except Exception:
        pass
    # 2) src.analytical.make_point_vf(...) → kernel(y,z)
    try:
        from src.analytical import make_point_vf as _mk
        try:
            k = _mk()
            return lambda y, z: k(y, z)
        except TypeError:
            # Try a very permissive call (some impls accept no args)
            return lambda y, z: _mk()(y, z)
    except Exception:
        pass
    # 3) As a last resort, look for a very generic name in analytical
    try:
        import src.analytical as ana
        for name in ("pointwise_vf", "vf_point", "point_kernel", "kernel"):
            if hasattr(ana, name):
                fn = getattr(ana, name)
                return lambda y, z, _fn=fn: _fn(y, z)
    except Exception:
        pass
    return None

# --- Public API ---------------------------------------------------------------
def sample_receiver_field(
    *,
    Wr: float,
    Hr: float,
    dy: float,
    dz: float,
    ny: int = 81,
    nz: int = 61,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build a coarse (ny×nz) field purely for plotting.
    Coordinates y,z are relative to the receiver centre; kernel expects emitter-centred,
    so we evaluate kernel(y+dy, z+dz).
    Returns (Y,Z,F) or None if no kernel can be resolved.
    """
    kernel = _try_resolve_point_kernel()
    if kernel is None:
        return None
    ys = np.linspace(-Wr/2.0, Wr/2.0, ny)
    zs = np.linspace(-Hr/2.0, Hr/2.0, nz)
    Y, Z = np.meshgrid(ys, zs, indexing="xy")
    F = kernel(Y + dy, Z + dz)
    return Y, Z, F

