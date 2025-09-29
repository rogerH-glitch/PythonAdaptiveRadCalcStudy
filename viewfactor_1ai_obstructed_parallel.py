# viewfactor_1ai_obstructed_parallel.py
import numpy as np
from numpy.polynomial.legendre import leggauss
from shapely.geometry import Polygon, Point

def _rect_polygon_xy(origin, u, v):
    """Return shapely Polygon for a rectangle projected to XY (z ignored)."""
    o = origin
    a = origin + u
    b = origin + u + v
    c = origin + v
    # drop z for 2D polygon
    return Polygon([(o[0], o[1]), (a[0], a[1]), (b[0], b[1]), (c[0], c[1])])

def viewfactor_1ai_parallel_with_occluder(rect_src, rect_tgt, rect_occ,
                                          n_src=14, n_tgt=22):
    """
    Two parallel 1×? rectangles facing each other (normals ±z),
    with a thin rectangular occluder in a plane parallel to them.
    We project the occluder onto the TARGET plane (orthographic in z),
    clip the visible target region, and integrate only visible points.

    rect_* = (origin, u, v), arbitrary sizes, but all three rectangles' planes are parallel.
    """
    o1, u1, v1 = rect_src
    o2, u2, v2 = rect_tgt
    oo, uo, vo = rect_occ

    # normals (assume parallel planes aligned with z axis direction)
    n1 = np.cross(u1, v1); n1 /= np.linalg.norm(n1)
    n2 = np.cross(u2, v2); n2 /= np.linalg.norm(n2)

    # auto-orient to face each other (like before)
    c1 = o1 + 0.5*(u1 + v1)
    c2 = o2 + 0.5*(u2 + v2)
    if np.dot(n1, c2 - c1) < 0: n1 = -n1
    if np.dot(n2, c1 - c2) < 0: n2 = -n2

    # separation (signed along the normal direction)
    # assume target is "above" source in +n1 direction
    sep = np.dot((c2 - c1), n1)

    # Build shapely polygons in the target plane's XY (orthographic projection)
    # For parallel planes, we can project by just dropping z and using XY coords.
    poly_tgt = _rect_polygon_xy(o2, u2, v2)
    poly_occ = _rect_polygon_xy(oo, uo, vo)
    poly_visible = poly_tgt.difference(poly_occ.buffer(0))  # buffer(0) fixes orientation issues

    # Gauss-Legendre nodes/weights on [0,1]
    xi_s, wi_s = leggauss(n_src); xs = 0.5*(xi_s + 1.0); ws = 0.5*wi_s
    xi_t, wi_t = leggauss(n_tgt); xt = 0.5*(xi_t + 1.0); wt = 0.5*wi_t

    A1 = np.linalg.norm(np.cross(u1, v1))
    A2 = np.linalg.norm(np.cross(u2, v2))

    total = 0.0
    for i, wx in enumerate(ws):
        sx = xs[i]
        for j, wy in enumerate(ws):
            sy = xs[j]
            p1 = o1 + sx*u1 + sy*v1  # 3D source sample

            inner = 0.0
            for a, wtx in enumerate(wt):
                tx = xt[a]
                for b, wty in enumerate(wt):
                    ty = xt[b]
                    p2 = o2 + tx*u2 + ty*v2  # 3D target point

                    # visibility test: is (x,y) of p2 inside visible region?
                    if not poly_visible.contains(Point(p2[0], p2[1])):
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

    # scale by areas (GL maps unit square to rectangles)
    F = (A1 * A2 * total) / A1
    return F
