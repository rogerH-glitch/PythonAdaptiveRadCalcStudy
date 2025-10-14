from __future__ import annotations
"""
Field sampler for plotting only.
If a receiver field (Y,Z,F) was not captured by the solver, we sample a coarse
grid on the receiver plane and evaluate a *pointwise* view-factor kernel to
produce a heatmap for visualization. Physics is unchanged; this is plotting-only.
"""
from typing import Callable, Optional, Tuple, Dict, Any
import numpy as np
from ..util.plot_payload import has_field

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
    # 3) Create a vectorized wrapper for vf_point_rect_to_point_parallel
    try:
        from src.analytical import vf_point_rect_to_point_parallel
        def create_vectorized_kernel():
            def kernel(y, z):
                # y, z are arrays of the same shape
                # We need to evaluate the function for each point
                result = np.zeros_like(y)
                for i in range(y.shape[0]):
                    for j in range(y.shape[1]):
                        # Use default emitter dimensions and setback
                        # This is a simplified version for plotting only
                        result[i, j] = vf_point_rect_to_point_parallel(
                            em_w=5.0, em_h=2.0, setback=3.0,
                            rx=y[i, j], ry=z[i, j]
                        )
                return result
            return kernel
        return create_vectorized_kernel()
    except Exception:
        pass
    # 4) As a last resort, look for a very generic name in analytical
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
def sample_receiver_field(args, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sample a coarse field for plotting if Y/Z/F not present in result.
    Only runs if Y/Z/F not present, populates Y,Z using receiver grid sizing,
    fills F via make_point_vf(Y+dy, Z+dz), sets result["field_is_sampled"]=True.
    Returns updated result.
    """
    # Check if field data already exists
    if has_field(result):
        return result
    
    # Extract parameters from args and result
    Wr = result.get("Wr", args.receiver[0])
    Hr = result.get("Hr", args.receiver[1])
    dy = result.get("dy", 0.0)
    dz = result.get("dz", 0.0)
    
    # Try to resolve point kernel
    kernel = _try_resolve_point_kernel()
    if kernel is None:
        return result
    
    # Build coarse grid
    ny, nz = 81, 61
    ys = np.linspace(-Wr/2.0, Wr/2.0, ny)
    zs = np.linspace(-Hr/2.0, Hr/2.0, nz)
    Y, Z = np.meshgrid(ys, zs, indexing="xy")
    F = kernel(Y + dy, Z + dz)
    
    # Attach to result
    result["Y"] = Y
    result["Z"] = Z
    result["F"] = F
    result["field_is_sampled"] = True
    
    return result


