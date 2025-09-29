# viewfactor_1ai_obstructed_general.py
import numpy as np
from numpy.polynomial.legendre import leggauss
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from occluder_projection import (
    _rect_corners, _unit, target_plane_frame, world_to_plane_2d,
    project_poly_from_point_to_plane
)

def _rect_area(u, v):
    return np.linalg.norm(np.cross(u, v))

def viewfactor_1ai_rects_with_occluders(rect_src, rect_tgt, occluder_list,
                                        n_src=14, n_tgt=20):
    """
    General obstructed 1AI between two rectangles with any number of occluders.

    rect_src, rect_tgt: (origin, u, v) in 3D.
    occluder_list: list of occluders, each either:
        - ("rect", (origin, u, v))
        - ("poly", np.array(Nx3 closed points))
    """
    o1, u1, v1 = rect_src
    o2, u2, v2 = rect_tgt
    A1 = _rect_area(u1, v1)
    A2 = _rect_area(u2, v2)
    if A1 == 0 or A2 == 0:
        return 0.0

    # raw normals
    n1 = _unit(np.cross(u1, v1))
    n2 = _unit(np.cross(u2, v2))

    # face each other
    c1 = o1 + 0.5*(u1 + v1)
    c2 = o2 + 0.5*(u2 + v2)
    if np.dot(n1, c2 - c1) < 0: n1 = -n1
    if np.dot(n2, c1 - c2) < 0: n2 = -n2

    # target plane local frame for 2D shapely clipping
    o_plane, t1p, t2p, n_plane = target_plane_frame(o2, n2)
    # target outline in 3D and 2D
    T3 = _rect_corners(o2, u2, v2)
    T2 = world_to_plane_2d(T3, o_plane, t1p, t2p)  # (5x2)
    poly_target_2d = Polygon(T2[:-1])

    # GL nodes/weights on [0,1]
    xi_s, wi_s = leggauss(n_src); xs = 0.5*(xi_s + 1.0); ws = 0.5*wi_s
    xi_t, wi_t = leggauss(n_tgt); xt = 0.5*(xi_t + 1.0); wt = 0.5*wi_t

    total = 0.0
    for i, wx in enumerate(ws):
        sx = xs[i]
        for j, wy in enumerate(ws):
            sy = xs[j]
            p1 = o1 + sx*u1 + sy*v1

            # ---- project *each occluder* from p1 onto the target plane ----
            occ_polys_2d = []
            for kind, data in occluder_list:
                if kind == "rect":
                    o, u, v = data
                    O3 = _rect_corners(o, u, v)  # 5x3
                elif kind == "poly":
                    O3 = data  # (Nx3, closed)
                else:
                    continue

                proj3 = project_poly_from_point_to_plane(p1, o_plane, n_plane, O3)
                proj2 = world_to_plane_2d(proj3, o_plane, t1p, t2p)
                # Robustify polygon orientation
                occ_polys_2d.append(Polygon(proj2[:-1]))

            # union of all projected occluders
            occ_union = unary_union(occ_polys_2d) if occ_polys_2d else None

            # visible region on target
            if occ_union and not occ_union.is_empty:
                visible = poly_target_2d.difference(occ_union.buffer(0))
            else:
                visible = poly_target_2d

            if visible.is_empty:
                continue

            # ---- inner (target) integral: keep only visible target sample points ----
            inner = 0.0
            for a, wtx in enumerate(wt):
                tx = 0.5*(xi_t[a] + 1.0)
                for b, wty in enumerate(wt):
                    ty = 0.5*(xi_t[b] + 1.0)

                    p2 = o2 + tx*u2 + ty*v2   # 3D point on target
                    # In-plane 2D coords for visibility test
                    uv = world_to_plane_2d(p2[None,:], o_plane, t1p, t2p)[0]
                    if not visible.contains(Point(uv[0], uv[1])):
                        continue

                    r_vec = p2 - p1
                    r = np.linalg.norm(r_vec)
                    if r == 0:
                        continue
                    r_hat = r_vec / r
                    cos1 = np.dot(n1, r_hat)
                    cos2 = -np.dot(n2, r_hat)
                    if (cos1 <= 0.0) or (cos2 <= 0.0):
                        continue

                    inner += (cos1 * cos2) / (np.pi * r*r) * wtx * wty

            total += inner * wx * wy

    # Scale by areas (GL maps [0,1]x[0,1] to rectangles)
    F = (A1 * A2 * total) / A1
    return F
