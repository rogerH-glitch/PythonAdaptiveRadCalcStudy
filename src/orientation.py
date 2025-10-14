from __future__ import annotations
from dataclasses import dataclass
import warnings
import math
import numpy as np

@dataclass
class Frame:
    C: np.ndarray        # emitter centre in world coords (x,y,z)
    j: np.ndarray        # emitter local width axis (unit)
    k: np.ndarray        # emitter local height axis (unit)
    n1: np.ndarray       # emitter normal (unit) = j × k
    n2: np.ndarray       # receiver normal (unit), faces emitter
    W: float
    H: float
    S: float             # receiver plane x = S
warnings.warn(
    "Module 'src.orientation' is deprecated and will be replaced by geometry utils and display-based transforms in v1.2.",
    DeprecationWarning,
    stacklevel=2,
)


def _rot_z(theta: float):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=float)


def _rot_y(theta: float):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=float)


def make_emitter_frame(
    W: float, H: float, S: float,
    *, axis: str, angle_deg: float,
    pivot: str,           # 'toe' or 'center' (x-centre formula for min gap is same here)
    dy: float, dz: float, # receiver - emitter in (y,z); if align_centres: dy=dz=0
) -> Frame:
    """
    Build world frame for the emitter rectangle:
      - local coordinates (u,v) in [-W/2,W/2] × [-H/2,H/2]
      - world point: p1(u,v) = C + j*u + k*v
      - j and k are unit axes; n1 = j × k
      - receiver plane at x = S with normal n2 = (-1,0,0) facing emitter
    Enforces min-gap-at-edge ('toe'): chooses C.x so the nearest edge sits at x=0.
    """
    th = math.radians(angle_deg)
    axis = axis.lower()
    # Base local axes before rotation: j0 = +y, k0 = +z
    j0 = np.array([0.0, 1.0, 0.0])
    k0 = np.array([0.0, 0.0, 1.0])
    # n0 = np.array([1.0, 0.0, 0.0])  # emitter normal when parallel: +x (unused)

    if abs(th) < 1e-15:
        R = np.eye(3)
    elif axis == "z":
        R = _rot_z(th)
    elif axis == "y":
        R = _rot_y(th)
    else:
        raise ValueError("rotate-axis must be 'z' or 'y'")

    j = (R @ j0).astype(float)       # width axis
    k = (R @ k0).astype(float)       # height axis
    n1 = np.cross(j, k)              # emitter normal (unit; rotation preserves orthonormality)
    n1 /= np.linalg.norm(n1)

    # Receiver normal: faces emitter → (-1,0,0)
    n2 = np.array([-1.0, 0.0, 0.0])

    # Choose emitter centre so the min x across the panel equals 0 (toe convention).
    # x(u,v) = Cx + (j·ex)*u + (k·ex)*v, with u∈[-W/2,W/2], v∈[-H/2,H/2].
    # For yaw (z-axis), k·ex≈0, min along u=W/2 if j·ex<0 → Cx = - (j·ex)*(W/2).
    # For pitch (y-axis), j·ex≈0, min along v=H/2 if k·ex>0 → Cx = - (k·ex)*(H/2).
    jx, kx = j[0], k[0]
    Cx = -jx * (W / 2.0) if axis == "z" else -kx * (H / 2.0)

    # Centre offsets: receiver centre is at (S, 0, 0); receiver - emitter = (dy,dz)
    # → emitter centre (y,z) = (0 - dy, 0 - dz)
    Cy = -dy
    Cz = -dz
    C = np.array([Cx, Cy, Cz], dtype=float)

    return Frame(C=C, j=j, k=k, n1=n1, n2=n2, W=W, H=H, S=S)
