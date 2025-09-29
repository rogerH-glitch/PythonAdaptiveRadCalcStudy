# viewfactor_1ai_general.py
import numpy as np
from numpy.polynomial.legendre import leggauss

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def _rect_area(u, v):
    return np.linalg.norm(np.cross(u, v))

def _rect_centroid(rect):
    o, u, v = rect
    return o + 0.5*(u + v)

def viewfactor_1ai_rect_to_rect(rect_src, rect_tgt, n_src=16, n_tgt=20):
    """
    Single-area structure: integrate over the SOURCE only with GL points; at each source point,
    evaluate the correct kernel by GL quadrature over the TARGET (unobstructed).
      rect = (origin, u_vec, v_vec) in 3D
      F_1->2 = (1/A1) ∬_{A1} ∬_{A2} [cosθ1 cosθ2/(π r^2)] dA2 dA1
    We perform:
      outer (A1): n_src × n_src GL points on the source
      inner (A2): n_tgt × n_tgt GL points on the target
    """
    o1, u1, v1 = rect_src
    o2, u2, v2 = rect_tgt

    A1 = _rect_area(u1, v1)
    if A1 == 0: return 0.0

    # mathematical (raw) normals
    n1_raw = _unit(np.cross(u1, v1))
    n2_raw = _unit(np.cross(u2, v2))

    # Auto-orient normals so they face each other for the math
    c1 = _rect_centroid(rect_src)
    c2 = _rect_centroid(rect_tgt)
    n1 = n1_raw.copy()
    n2 = n2_raw.copy()
    if np.dot(n1, c2 - c1) < 0: n1 = -n1
    if np.dot(n2, c1 - c2) < 0: n2 = -n2

    # Gauss-Legendre nodes/weights on [0,1]
    xi_s, wi_s = leggauss(n_src); xs = 0.5*(xi_s + 1.0); ws = 0.5*wi_s
    xi_t, wi_t = leggauss(n_tgt); xt = 0.5*(xi_t + 1.0); wt = 0.5*wi_t

    # Jacobians for mapping unit square -> actual rectangles
    J1 = A1   # but we use sampling weights, so included implicitly by weights on source
    A2 = _rect_area(u2, v2)

    total = 0.0

    # Outer (source) integral
    for i, wx in enumerate(ws):
        sx = xs[i]
        for j, wy in enumerate(ws):
            sy = xs[j]
            # source sample point p1 and its local differential area via weights (already wt factors)
            p1 = o1 + sx*u1 + sy*v1

            # Inner (target) integral at this source point
            inner = 0.0
            for a, wtx in enumerate(wt):
                tx = xt[a]
                for b, wty in enumerate(wt):
                    ty = xt[b]
                    p2 = o2 + tx*u2 + ty*v2

                    r_vec = p2 - p1
                    r = np.linalg.norm(r_vec)
                    if r == 0:  # coincident (degenerate) guard
                        continue
                    r_hat = r_vec / r

                    cos1 = np.dot(n1, r_hat)          # leaving source
                    cos2 = -np.dot(n2, r_hat)         # arriving at target
                    if (cos1 <= 0.0) or (cos2 <= 0.0):
                        continue

                    inner += (cos1 * cos2) / (np.pi * r*r) * wtx * wty

            # accumulate with source weights
            total += inner * wx * wy

    # Account for physical areas of rectangles:
    # - Our GL weights integrate over [0,1]×[0,1]; multiply by areas A1 and A2.
    F = (A1 * A2 * total) / A1   # divide by A1 per definition
    return F
