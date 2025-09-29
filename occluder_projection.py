# occluder_projection.py
import numpy as np

# ---------- small vector helpers ----------
def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def _orthonormal_basis(n):
    n = _unit(n)
    # pick a vector not parallel to n
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = _unit(np.cross(n, tmp))
    t2 = _unit(np.cross(n, t1))
    return t1, t2, n

def _rect_corners(o, u, v):  # 5 points (closed)
    return np.array([o, o+u, o+u+v, o+v, o])

# ---------- target-plane 2D mapping ----------
def target_plane_frame(o_tgt, n_tgt):
    """Return local frame (t1, t2, n) and origin for the target plane."""
    t1, t2, n = _orthonormal_basis(n_tgt)
    return o_tgt, t1, t2, n

def world_to_plane_2d(P, o_plane, t1, t2):
    """Project 3D points P onto plane coords (u,v) by dotting with in-plane axes."""
    UV = np.column_stack(((P - o_plane) @ t1, (P - o_plane) @ t2))
    return UV

# ---------- perspective projection ----------
def project_poly_from_point_to_plane(P0, plane_point, plane_normal, poly3d):
    """
    Project a list/array of 3D points (poly3d) from eye P0 onto plane (plane_point, plane_normal).
    Returns Nx3 projected points in 3D.
    """
    n = _unit(plane_normal)
    num = (plane_point - P0) @ n
    projs = []
    for Q in poly3d:
        d = (Q - P0)
        den = d @ n
        if abs(den) < 1e-12:
            # ray parallel to plane: "send far away" or skip; we place at large distance
            t = 1e12
        else:
            t = num / den
        projs.append(P0 + t * d)
    return np.vstack(projs)

# ---------- user-facing occluder builders ----------
def make_rect_occluder_from_params(emitter_origin, emitter_normal,
                                   setback, width, height,
                                   orientation="parallel", angle_deg=0.0,
                                   thickness=0.0, center_offset_xy=(0.0, 0.0)):
    """
    Build a rectangle (or a thin slab if thickness>0) occluder near the 'emitter' (source plate).
    Returns one or two rectangles (for slab) as tuples (origin, u, v).

    emitter_origin: a point on emitter plane (e.g., rect1 centroid)
    emitter_normal: unit normal of emitter (+outward)
    setback: distance along emitter_normal (positive "in front of" the emitter)
    width: along local x (t1)
    height: along local y (t2)
    orientation:
      - "parallel": occluder plane || emitter plane
      - "perpendicular": rotate 90° about local x (so it’s like a vertical fence)
      - "angled": rotate by angle_deg about local x
    thickness: optional slab thickness measured along occluder normal
    center_offset_xy: (du, dv) offsets in emitter’s in-plane coordinates
    """
    # Emitter local frame
    t1, t2, n = _orthonormal_basis(emitter_normal)
    center = emitter_origin + setback * n + center_offset_xy[0]*t1 + center_offset_xy[1]*t2

    # Occluder plane normal/orientation
    if orientation == "parallel":
        n_occ = n
    elif orientation == "perpendicular":
        # rotate +90° about t1: n -> n*0 + t2
        n_occ = t2
    elif orientation == "angled":
        # rotate by angle about t1
        a = np.deg2rad(angle_deg)
        n_occ = _unit(np.cos(a)*n + np.sin(a)*t2)
    else:
        raise ValueError("orientation must be 'parallel', 'perpendicular', or 'angled'")

    # In-plane axes for occluder
    occ_x = t1
    occ_y = _unit(np.cross(n_occ, occ_x))  # ensure orthogonal to normal

    # Build rectangle as (origin, u, v) centered at 'center'
    u = width * occ_x
    v = height * occ_y
    o = center - 0.5*(u + v)

    if thickness and thickness > 0:
        # build a thin slab: two parallel faces separated along n_occ
        o_front = o + 0.5*thickness * n_occ
        o_back  = o - 0.5*thickness * n_occ
        return [(o_front, u, v), (o_back, u, v)]
    else:
        return [(o, u, v)]

def make_circular_occluder(center, normal, radius, n_sides=48):
    """
    Approximate a circular disk as an N-gon; return polygon 3D points (closed).
    """
    t1, t2, n = _orthonormal_basis(normal)
    angles = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
    pts = [center + radius*np.cos(a)*t1 + radius*np.sin(a)*t2 for a in angles]
    pts.append(pts[0])
    return np.vstack(pts)
