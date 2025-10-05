"""
Geometric utilities and data structures for view factor calculations.

This module provides the core geometric classes and validation functions
for fire safety radiation calculations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
import math
import numpy as np
import inspect

@dataclass
class PanelSpec:
    width: float   # true width (y-extent in emitter-local frame)
    height: float  # true height (z-extent in emitter-local frame)

@dataclass
class PlacementOpts:
    angle_deg: float = 0.0       # rotation angle
    rotate_axis: str = "z"       # "z" (yaw) or "y" (pitch)
    rotate_target: str = "emitter"  # currently only "emitter" supported for rotation
    pivot: str = "toe"           # "toe" (edge pivot) or "center"
    # offsets between centres in global (y,z): receiver_center - emitter_center
    offset_dy: float = 0.0
    offset_dz: float = 0.0
    align_centres: bool = False  # force centres aligned (dy=dz=0), overrides offsets

def _rect_center_yz(width: float, height: float) -> tuple[float,float]:
    return 0.0, 0.0

def _emitter_segment_yaw(width: float, angle_rad: float, pivot_edge: str):
    """
    Top view (x–y). Build finite segment whose length equals TRUE width.
    pivot_edge: "top" => y=+W/2; "bottom" => y=-W/2.
    Returns two endpoints in (x,y) BEFORE x-translation for setback.
    """
    W = width
    y0 = +W/2 if pivot_edge == "top" else -W/2
    # direction from top->bottom after yaw by +angle about z:
    u = np.array([math.sin(angle_rad), -math.cos(angle_rad)])  # (ux, uy)
    P0 = np.array([0.0, y0])       # pivot endpoint (nearest edge)
    P1 = P0 - W * u                # finite length = W
    return P0, P1

def _emitter_segment_pitch(height: float, angle_rad: float, pivot_edge: str):
    """
    Side view (x–z). Build finite segment whose length equals TRUE height.
    pivot_edge: "top" => z=H; "bottom" => z=0.
    Returns two endpoints in (x,z) BEFORE x-translation for setback.
    """
    H = height
    z0 = H if pivot_edge == "top" else 0.0
    v = np.array([math.sin(angle_rad), -math.cos(angle_rad)])  # (vx, vz)
    Q0 = np.array([0.0, z0])       # pivot endpoint (nearest edge)
    Q1 = Q0 - H * v                # finite length = H
    return Q0, Q1

def place_panels(
    emitter: PanelSpec,
    receiver: PanelSpec,
    setback: float,
    opts: PlacementOpts,
) -> dict:
    """
    Build a minimal geometric description sufficient for kernels/search bounds:
      - Receiver is kept axis-aligned, vertical, at plane x = setback.
      - Emitter is vertical, rotated around z (yaw) or y (pitch).
      - If opts.pivot=="toe": rotate about the *nearest edge* and translate along +x so that
        that edge is the min-gap location and min gap equals 'setback'.
      - If opts.pivot=="center": rotate around emitter centre; then translate along +x so that
        the *nearest endpoint* touches x=0 (preserves min setback), and optionally align centres.
      - Offsets (dy,dz) are applied as centre-to-centre translations in (y,z), unless align_centres=True.
    Returns a dict with:
      {
        "receiver_plane_x": setback,
        "emitter_segment_xy": (P0, P1)  # for yaw (top view),
        "emitter_segment_xz": (Q0, Q1)  # for pitch (side view),
        "centres": {"emitter": (y_e, z_e), "receiver": (y_r, z_r)},
      }
    Notes:
      - This is a geometric *scaffold* for positioning & plotting, not the full 3D mesh.
      - VF kernels should still use the panel dimensions (true width/height) and transforms.
    """
    angle = math.radians(opts.angle_deg)
    # receiver stays centred at (y,z) = (0,0) unless offsets/align override
    y_r, z_r = 0.0, 0.0

    if opts.align_centres:
        dy, dz = 0.0, 0.0
    else:
        dy, dz = opts.offset_dy, opts.offset_dz

    # emitter centre initially at (0,0), then shifted to match dy,dz sign convention:
    y_e, z_e = y_r - dy, z_r - dz  # receiver - emitter = (dy,dz)  => emitter = receiver - (dy,dz)

    geom = {
        "receiver_plane_x": float(setback),
        "centres": {"emitter": (y_e, z_e), "receiver": (y_r, z_r)},
        "emitter_segment_xy": None,
        "emitter_segment_xz": None,
    }

    if abs(angle) < 1e-12:
        # parallel case: no segment tilt needed; kernels can treat as aligned
        return geom

    if opts.rotate_axis.lower() == "z":
        pivot_edge = "top" if opts.pivot == "toe" else "center"
        if pivot_edge == "top":
            P0, P1 = _emitter_segment_yaw(emitter.width, angle, "top")
            # translate along +x so pivot endpoint x==0 (min gap preserved at that edge)
            dx = -P0[0]
            P0 += np.array([dx, 0.0])
            P1 += np.array([dx, 0.0])
        else:
            # rotate around centre: build around (0,0) then push along +x so nearest endpoint touches x=0
            # half-vector h = (W/2)*(sinθ, cosθ) in (x,y)
            W = emitter.width
            hx, hy = (W/2)*math.sin(angle), (W/2)*math.cos(angle)
            low  = np.array([-hx, -hy])
            high = np.array([+hx, +hy])
            # pick endpoint with larger y as "nearest" and set its x to 0
            target = high if high[1] >= low[1] else low
            dx = -target[0]
            low  += np.array([dx, 0.0])
            high += np.array([dx, 0.0])
            P0, P1 = low, high
        # finally apply centre translations in y (and z is irrelevant in top view)
        P0[1] += y_e
        P1[1] += y_e
        geom["emitter_segment_xy"] = (P0, P1)

    elif opts.rotate_axis.lower() == "y":
        pivot_edge = "top" if opts.pivot == "toe" else "center"
        if pivot_edge == "top":
            Q0, Q1 = _emitter_segment_pitch(emitter.height, angle, "top")
            dx = -Q0[0]
            Q0 += np.array([dx, 0.0])
            Q1 += np.array([dx, 0.0])
        else:
            H = emitter.height
            kx, kz = (H/2)*math.sin(angle), (H/2)*math.cos(angle)
            low  = np.array([-kx, -kz])
            high = np.array([+kx, +kz])
            target = high if high[1] >= low[1] else low
            dx = -target[0]
            low  += np.array([dx, 0.0])
            high += np.array([dx, 0.0])
            Q0, Q1 = low, high
        # apply centre translations in z (top/side share same centre z)
        Q0[1] += z_e
        Q1[1] += z_e
        geom["emitter_segment_xz"] = (Q0, Q1)
    else:
        raise ValueError("rotate_axis must be 'z' or 'y'")

    return geom

# ------------------------------------------------------------------
# Test-compatibility shims (non-invasive) expected by smoke tests
# ------------------------------------------------------------------

class GeometryError(Exception):
    """Raised on invalid geometry inputs (compat for tests)."""
    pass


def validate_geometry(em_w: float, em_h: float, rc_w: float, rc_h: float, setback: float) -> bool:
    """Very light validation used by smoke tests."""
    if em_w <= 0 or em_h <= 0 or rc_w <= 0 or rc_h <= 0:
        raise GeometryError("Widths/heights must be positive.")
    if setback <= 0:
        raise GeometryError("Setback must be positive.")
    return True


def _patch_rectangle_init_for_uv(RectangleClass):
    """
    Monkey-patch Rectangle.__init__ to accept u_vector/v_vector/**kwargs
    without breaking existing constructor signatures.
    """
    orig = RectangleClass.__init__
    def _init(self, center, width, height, *, normal=None, u_vector=None, v_vector=None, **kwargs):
        # Try most explicit signature first, then fall back
        try:
            orig(self, center=center, width=width, height=height, normal=normal)
        except TypeError:
            try:
                orig(self, center, width, height)  # older positional form
            except TypeError:
                orig(self, center=center, width=width, height=height)  # minimal
        # Attach u/v vectors if provided or set sensible defaults
        if not hasattr(self, "u_vector") or u_vector is not None:
            self.u_vector = np.array(u_vector if u_vector is not None else (0.0, 1.0, 0.0), dtype=float)
        if not hasattr(self, "v_vector") or v_vector is not None:
            self.v_vector = np.array(v_vector if v_vector is not None else (0.0, 0.0, 1.0), dtype=float)
        if not hasattr(self, "normal"):
            self.normal = np.array(normal if normal is not None else (1.0, 0.0, 0.0), dtype=float)
    RectangleClass.__init__ = _init
    return RectangleClass

# If a Rectangle already exists, patch it; otherwise create a minimal one.
if "Rectangle" in globals():
    try:
        sig = inspect.signature(Rectangle.__init__)
        if "u_vector" not in sig.parameters or "v_vector" not in sig.parameters:
            Rectangle = _patch_rectangle_init_for_uv(Rectangle)  # type: ignore
    except Exception:
        Rectangle = _patch_rectangle_init_for_uv(Rectangle)  # type: ignore
else:
    class Rectangle:  # minimal fallback (used only if no Rectangle existed)
        def __init__(self, center, width, height, *, normal=None, u_vector=None, v_vector=None, **kwargs):
            self.center = tuple(center)
            self.width = float(width); self.height = float(height)
            self.normal = np.array(normal if normal is not None else (1.0,0.0,0.0), dtype=float)
            self.u_vector = np.array(u_vector if u_vector is not None else (0.0,1.0,0.0), dtype=float)
            self.v_vector = np.array(v_vector if v_vector is not None else (0.0,0.0,1.0), dtype=float)

class ViewFactorResult:
    """
    Compat container: accepts either `vf=` or legacy `value=`,
    plus optional `ci95`, `status`, `meta`.
    """

    def __init__(
        self,
        vf: Optional[float] = None,
        ci95: Optional[float] = None,
        status: str = "converged",
        meta: Optional[Dict[str, Any]] = None,
        value: Optional[float] = None,
        **kwargs,
    ):
        if vf is None and value is not None:
            vf = float(value)
        self.vf = float(vf if vf is not None else 0.0)
        self.ci95 = ci95
        self.status = status
        self.meta = meta or {}


# Legacy compatibility functions (keeping existing API)
@dataclass(frozen=True)
class Rectangle:
    """Immutable rectangle geometry for fire safety calculations.
    
    Represents a rectangular surface in 3D space defined by origin point
    and two edge vectors. Used for emitter and receiver surfaces in
    view factor calculations.
    """
    origin: np.ndarray
    edge_u: np.ndarray
    edge_v: np.ndarray
    
    def __post_init__(self):
        """Validate rectangle geometry after initialization."""
        if self.origin.shape != (3,):
            raise ValueError("Origin must be a 3D point")
        if self.edge_u.shape != (3,) or self.edge_v.shape != (3,):
            raise ValueError("Edge vectors must be 3D vectors")
        
        # Check for zero-length edges
        if np.linalg.norm(self.edge_u) < 1e-12:
            raise ValueError("Edge U has zero length")
        if np.linalg.norm(self.edge_v) < 1e-12:
            raise ValueError("Edge V has zero length")
    
    @property
    def area(self) -> float:
        """Calculate rectangle area."""
        return np.linalg.norm(np.cross(self.edge_u, self.edge_v))
    
    @property
    def normal(self) -> np.ndarray:
        """Calculate unit normal vector."""
        cross = np.cross(self.edge_u, self.edge_v)
        return cross / np.linalg.norm(cross)
    
    def contains_point(self, point: np.ndarray, tolerance: float = 1e-12) -> bool:
        """Check if point lies within rectangle (within tolerance)."""
        if point.shape != (3,):
            return False
        
        # Vector from origin to point
        v = point - self.origin
        
        # Project onto edge vectors
        u_proj = np.dot(v, self.edge_u) / np.dot(self.edge_u, self.edge_u)
        v_proj = np.dot(v, self.edge_v) / np.dot(self.edge_v, self.edge_v)
        
        # Check if projections are within [0,1] with tolerance
        return (-tolerance <= u_proj <= 1 + tolerance and 
                -tolerance <= v_proj <= 1 + tolerance)


def create_rectangle_from_corners(
    corner1: np.ndarray, 
    corner2: np.ndarray, 
    corner3: np.ndarray
) -> Rectangle:
    """Create rectangle from three corner points.
    
    Args:
        corner1: First corner point
        corner2: Second corner point (adjacent to first)
        corner3: Third corner point (adjacent to second)
    
    Returns:
        Rectangle with origin at corner1 and edges along corner1->corner2 and corner1->corner3
    """
    if corner1.shape != (3,) or corner2.shape != (3,) or corner3.shape != (3,):
        raise ValueError("All corners must be 3D points")
    
    edge_u = corner2 - corner1
    edge_v = corner3 - corner1
    
    return Rectangle(origin=corner1, edge_u=edge_u, edge_v=edge_v)


def create_parallel_rectangles(
    emitter_w: float, 
    emitter_h: float, 
    receiver_w: float, 
    receiver_h: float, 
    setback: float
) -> tuple[Rectangle, Rectangle]:
    """Create parallel rectangular surfaces for view factor calculation.
    
    Args:
        emitter_w: Emitter width (y-direction)
        emitter_h: Emitter height (z-direction)  
        receiver_w: Receiver width (y-direction)
        receiver_h: Receiver height (z-direction)
        setback: Distance between surfaces (x-direction)
    
    Returns:
        Tuple of (emitter_rectangle, receiver_rectangle)
    """
    # Emitter at x=0, centered at origin
    emitter_origin = np.array([0.0, -emitter_w/2, -emitter_h/2])
    emitter_edge_u = np.array([0.0, emitter_w, 0.0])
    emitter_edge_v = np.array([0.0, 0.0, emitter_h])
    
    # Receiver at x=setback, centered at origin
    receiver_origin = np.array([setback, -receiver_w/2, -receiver_h/2])
    receiver_edge_u = np.array([0.0, receiver_w, 0.0])
    receiver_edge_v = np.array([0.0, 0.0, receiver_h])
    
    emitter = Rectangle(origin=emitter_origin, edge_u=emitter_edge_u, edge_v=emitter_edge_v)
    receiver = Rectangle(origin=receiver_origin, edge_u=receiver_edge_u, edge_v=receiver_edge_v)
    
    return emitter, receiver


def validate_geometry(
    emitter_w: float, 
    emitter_h: float, 
    receiver_w: float, 
    receiver_h: float, 
    setback: float
) -> tuple[bool, str]:
    """Validate geometry parameters for view factor calculations.
    
    Args:
        emitter_w: Emitter width
        emitter_h: Emitter height
        receiver_w: Receiver width  
        receiver_h: Receiver height
        setback: Distance between surfaces
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if emitter_w <= 0 or emitter_h <= 0:
        return False, "Emitter dimensions must be positive"
    if receiver_w <= 0 or receiver_h <= 0:
        return False, "Receiver dimensions must be positive"
    if setback <= 0:
        return False, "Setback must be positive"
    
    return True, ""


# Legacy rotation and transformation functions (keeping existing API)
def Rz(theta_deg: float) -> np.ndarray:
    """Create 2D rotation matrix about z-axis.
    
    Args:
        theta_deg: Rotation angle in degrees
        
    Returns:
        2x2 rotation matrix
    """
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]], dtype=float)


def to_emitter_frame(
    rc_point_local: tuple[float, float],
    em_offset: tuple[float, float],
    rc_offset: tuple[float, float],
    angle_deg: float,
    rotate_target: str = "emitter"
) -> tuple[float, float]:
    """
    Map a receiver-local point (rx, ry) into the emitter frame for angle/offset handling.
    - Offsets are x–y translations of each surface center.
    - If rotate_target == 'emitter', the emitter is rotated by +angle (z); to express rc in emitter frame,
      apply inverse rotation to the world vector (rc_local + rc_offset - em_offset).
    - If rotate_target == 'receiver', rotate the receiver-local point forward by +angle, then translate.
    """
    rx, ry = rc_point_local
    ex, ey = em_offset
    qx, qy = rc_offset

    if abs(angle_deg) < 1e-12:
        return (rx + qx - ex, ry + qy - ey)

    if rotate_target == "emitter":
        Rinv = Rz(-angle_deg)
        v = np.array([rx + qx - ex, ry + qy - ey], dtype=float)
        vv = (Rinv @ v.reshape(2, 1)).ravel()
        return float(vv[0]), float(vv[1])
    else:
        R = Rz(angle_deg)
        v = (R @ np.array([rx, ry]).reshape(2, 1)).ravel()
        vv = v + np.array([qx - ex, qy - ey])
        return float(vv[0]), float(vv[1])


def toe_pivot_adjust_emitter_center(
    angle_deg: float, 
    em_offset: tuple[float, float]
) -> tuple[float, float]:
    """
    For pure z-rotation and zero thickness surfaces, the plane of the emitter stays at x=0 in 3D.
    The 'toe' (nearest-face) convention is equivalent to keeping the emitter's min x-envelope 
    aligned with the separation line. In our in-plane 2D integration (emitter frame), this does 
    not change the local (y,z) parametrization. Therefore, for now, return em_offset unchanged.
    (This hook exists to adjust center if in future you model finite thickness.)
    """
    return em_offset