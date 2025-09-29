# viewfactor_1ai_adaptive_obstructed.py
import numpy as np
from numpy.polynomial.legendre import leggauss
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

from occluder_projection import (
    _rect_corners, _unit, target_plane_frame, world_to_plane_2d,
    project_poly_from_point_to_plane
)

def _area(u, v):
    return np.linalg.norm(np.cross(u, v))

def _rect_point(o, u, v, s, t):
    return o + s*u + t*v

def _n_of(u, v):
    n = np.cross(u, v); n /= np.linalg.norm(n)
    return n

def _visible_polygon_for_source_point(p1, rect_tgt, occluder_list):
    """Project occluders from p1 to target plane -> visible polygon on target (2D shapely)."""
    o2, u2, v2 = rect_tgt
    n2 = _n_of(u2, v2)
    o_plane, t1p, t2p, n_plane = target_plane_frame(o2, n2)

    T3 = _rect_corners(o2, u2, v2)
    T2 = world_to_plane_2d(T3, o_plane, t1p, t2p)
    poly_target_2d = Polygon(T2[:-1])

    occ_polys_2d = []
    for kind, data in occluder_list:
        if kind == "rect":
            o, u, v = data
            O3 = _rect_corners(o, u, v)
        elif kind == "poly":
            O3 = data
        else:
            continue
        proj3 = project_poly_from_point_to_plane(p1, o_plane, n_plane, O3)
        proj2 = world_to_plane_2d(proj3, o_plane, t1p, t2p)
        occ_polys_2d.append(Polygon(proj2[:-1]))

    if occ_polys_2d:
        occ_union = unary_union(occ_polys_2d)
        vis = poly_target_2d.difference(occ_union.buffer(0))
    else:
        vis = poly_target_2d
    return vis, (o_plane, t1p, t2p, n_plane)

def _inner_target_integral_visible(p1, n1, rect_tgt, visible_poly_2d, plane_frame, n_tgt_gl):
    """I(p1) over VISIBLE part of target using GL on [0,1]^2 and point-in-polygon test."""
    o2, u2, v2 = rect_tgt
    (o_plane, t1p, t2p, _) = plane_frame
    n2 = _n_of(u2, v2)

    xi, wi = leggauss(n_tgt_gl)
    xt, wt = 0.5*(xi + 1.0), 0.5*wi

    inner = 0.0
    for a, wtx in enumerate(wt):
        tx = xt[a]
        for b, wty in enumerate(wt):
            ty = xt[b]
            p2 = _rect_point(o2, u2, v2, tx, ty)
            # visibility
            uv = world_to_plane_2d(p2[None,:], o_plane, t1p, t2p)[0]
            if not visible_poly_2d.contains(Point(uv[0], uv[1])):
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

    A2 = _area(u2, v2)
    return A2 * inner

def viewfactor_1ai_adaptive_obstructed(rect_src, rect_tgt, occluder_list,
                                       tol=1e-3, max_depth=6, n_tgt_gl=22):
    """
    Adaptive single-area F_{1->2} with general occluders.
    - Subdivide SOURCE param space where the cell's coarse vs refined contribution differs.
    - At each source-cell centroid, project occluders to target to get the visible polygon.
    """
    o1, u1, v1 = rect_src
    A1 = _area(u1, v1)
    if A1 == 0:
        return 0.0

    # orient n1 to face the target
    n1 = _n_of(u1, v1)
    c1 = o1 + 0.5*(u1 + v1)
    c2 = rect_tgt[0] + 0.5*(rect_tgt[1] + rect_tgt[2])
    if np.dot(n1, c2 - c1) < 0: n1 = -n1

    # simple cache so repeated nearby points reuse visibility (rounded coords)
    vis_cache = {}

    def visible_at(p):
        key = tuple(np.round(p, 5))  # 1e-5 m grid
        if key in vis_cache:
            return vis_cache[key]
        vis, frame = _visible_polygon_for_source_point(p, rect_tgt, occluder_list)
        vis_cache[key] = (vis, frame)
        return vis, frame

    def I_at(p):
        vis_poly, frame = visible_at(p)
        if vis_poly.is_empty:
            return 0.0
        return _inner_target_integral_visible(p, n1, rect_tgt, vis_poly, frame, n_tgt_gl)

    total = 0.0

    def recurse(s0, t0, s1, t1, depth):
        nonlocal total
        # cell area on source
        cell_A = A1 * (s1 - s0) * (t1 - t0)

        # four subcell centroids (use their mean as "coarse" to reduce bias)
        sm, tm = 0.5*(s0 + s1), 0.5*(t0 + t1)
        sub = [
            (s0 + 0.25*(s1 - s0), t0 + 0.25*(t1 - t0)),
            (sm + 0.25*(s1 - s0), t0 + 0.25*(t1 - t0)),
            (s0 + 0.25*(s1 - s0), tm + 0.25*(t1 - t0)),
            (sm + 0.25*(s1 - s0), tm + 0.25*(t1 - t0)),
        ]
        I_vals = []
        for (ss, tt) in sub:
            p = _rect_point(o1, u1, v1, ss, tt)
            I_vals.append(I_at(p))
        I_avg4 = 0.25 * sum(I_vals)        # "coarse" (less biased)
        # refined = sum of the 4 subcells, each using its own centroid (same I_vals),
        # but scaled by quarter area -> also equals I_avg4 here.
        refined_val = I_avg4 * cell_A
        # build a cheaper error proxy by sampling *edges* midpoints too
        edge_pts = [
            (0.5*(s0+sm), t0 + 0.25*(t1 - t0)),
            (0.5*(sm+s1), t0 + 0.25*(t1 - t0)),
            (0.5*(s0+sm), tm + 0.25*(t1 - t0)),
            (0.5*(sm+s1), tm + 0.25*(t1 - t0)),
        ]
        edge_I = []
        for (ss, tt) in edge_pts:
            p = _rect_point(o1, u1, v1, ss, tt)
            edge_I.append(I_at(p))
        # variation measure
        var = (max(I_vals + edge_I) - min(I_vals + edge_I))
        err = var * cell_A * 0.25  # heuristic

        if (err <= tol) or (depth >= max_depth):
            total += refined_val
            return
        # subdivide
        recurse(s0, t0, sm, tm, depth+1)
        recurse(sm, t0, s1, tm, depth+1)
        recurse(s0, tm, sm, s1, depth+1)
        recurse(sm, tm, s1, s1, depth+1)  # (typo-proof: last args = sm,tm -> s1,t1)
    # fix last call (above line) to correct bounds:
    def recurse(s0, t0, s1, t1, depth):  # redefine with correct four children
        nonlocal total
        cell_A = A1 * (s1 - s0) * (t1 - t0)
        sm, tm = 0.5*(s0 + s1), 0.5*(t0 + t1)
        sub = [
            (s0 + 0.25*(s1 - s0), t0 + 0.25*(t1 - t0)),
            (sm + 0.25*(s1 - s0), t0 + 0.25*(t1 - t0)),
            (s0 + 0.25*(s1 - s0), tm + 0.25*(t1 - t0)),
            (sm + 0.25*(s1 - s0), tm + 0.25*(t1 - t0)),
        ]
        I_vals = [I_at(_rect_point(o1, u1, v1, ss, tt)) for (ss, tt) in sub]
        I_avg4 = 0.25 * sum(I_vals)
        refined_val = I_avg4 * cell_A

        # edge-variation heuristic for error
        edge_pts = [
            (0.5*(s0+sm), t0 + 0.25*(t1 - t0)),
            (0.5*(sm+s1), t0 + 0.25*(t1 - t0)),
            (0.5*(s0+sm), tm + 0.25*(t1 - t0)),
            (0.5*(sm+s1), tm + 0.25*(t1 - t0)),
        ]
        edge_I = [I_at(_rect_point(o1, u1, v1, ss, tt)) for (ss, tt) in edge_pts]
        var = (max(I_vals + edge_I) - min(I_vals + edge_I))
        err = var * cell_A * 0.25

        if (err <= tol) or (depth >= max_depth):
            total += refined_val
            return
        recurse(s0, t0, sm, tm, depth+1)
        recurse(sm, t0, s1, tm, depth+1)
        recurse(s0, tm, sm, t1, depth+1)
        recurse(sm, tm, s1, t1, depth+1)

    recurse(0.0, 0.0, 1.0, 1.0, 0)
    return total / A1
