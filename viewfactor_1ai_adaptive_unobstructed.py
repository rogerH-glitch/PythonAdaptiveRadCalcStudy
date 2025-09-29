# viewfactor_1ai_adaptive_unobstructed.py
import numpy as np
from numpy.polynomial.legendre import leggauss

# ---------- small helpers ----------
def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def _area(u, v):
    return np.linalg.norm(np.cross(u, v))

def _rect_point(o, u, v, s, t):
    """Map (s,t) in [0,1]^2 to 3D point o + s*u + t*v."""
    return o + s*u + t*v

# ---------- kernel over target by Gauss–Legendre ----------
def _target_integral_at_source_point(p1, n1, rect_tgt, n_tgt_gl=20):
    """
    Evaluate I(p1) = ∬_{A2} [cosθ1 cosθ2 / (π r^2)] dA2
    by GL on target. This is the 'inner' integral.
    """
    o2, u2, v2 = rect_tgt
    n2 = _unit(np.cross(u2, v2))

    # Orient to face each other (robust for skew cases)
    # (this is consistent with your other routines)
    c2 = o2 + 0.5*(u2 + v2)
    c1_approx = p1                             # a local 'centroid' for check
    if np.dot(n2, c1_approx - c2) < 0: n2 = -n2

    # GL nodes/weights on [0,1]
    xi, wi = leggauss(n_tgt_gl)
    xt = 0.5*(xi + 1.0); wt = 0.5*wi

    inner = 0.0
    for a, wtx in enumerate(wt):
        tx = xt[a]
        for b, wty in enumerate(wt):
            ty = xt[b]
            p2 = _rect_point(o2, u2, v2, tx, ty)
            r_vec = p2 - p1
            r = np.linalg.norm(r_vec)
            if r == 0:
                continue
            r_hat = r_vec / r
            cos1 = np.dot(n1, r_hat)          # leaving source
            cos2 = -np.dot(n2, r_hat)         # arriving target
            if (cos1 <= 0.0) or (cos2 <= 0.0):
                continue
            inner += (cos1 * cos2) / (np.pi * r*r) * wtx * wty

    # scale to physical target area (GL integrates unit square)
    A2 = _area(u2, v2)
    return A2 * inner

# ---------- adaptive 1AI over source ----------
def viewfactor_1ai_adaptive_unobstructed(rect_src, rect_tgt,
                                         tol=5e-4,          # absolute error target on F
                                         max_depth=6,       # maximum subdivision levels
                                         n_src_gl=6,        # GL order per cell center estimate
                                         n_tgt_gl=20):      # GL order for inner integral
    """
    Adaptive single-area view factor F_{1->2} for two unobstructed rectangles.

    Strategy:
      - Integrate over the SOURCE only (1AI).
      - Start with whole (s,t) in [0,1]^2. For each cell:
          (1) Coarse estimate: evaluate I(p1) at the cell centroid (GL on target only).
          (2) Refined estimate: split the cell into 4 quarters; evaluate each quarter’s centroid.
          (3) Error = |refined - coarse| * (cell area in source).
          (4) If error > local_tolerance, recurse on that cell; else accept refined value.
      - Sum contributions and divide by A1.

    Notes:
      * local_tolerance = tol * A1, distributed by a simple heuristic:
        we compare each cell’s absolute contribution; this works well in practice.
      * n_src_gl is used only to place centroid (we just use the true centroid here);
        kept for signature consistency and future upgrades (e.g., multi-point per cell).

    Returns:
      F1->2 (float)
    """
    o1, u1, v1 = rect_src
    A1 = _area(u1, v1)
    if A1 == 0:
        return 0.0

    # source normal oriented to face target (robust)
    n1 = _unit(np.cross(u1, v1))
    c1 = o1 + 0.5*(u1 + v1)
    o2, u2, v2 = rect_tgt
    c2 = o2 + 0.5*(u2 + v2)
    if np.dot(n1, c2 - c1) < 0: n1 = -n1

    # local tolerance per cell will be compared against absolute contribution
    # We simply check absolute error per cell and ensure the sum stays under tol.
    target_abs_tol = tol

    def cell_estimate(s0, t0, s1, t1):
        """Return (coarse, refined, err) for a single source cell in [s0,s1]x[t0,t1]."""
        # cell centroid in parametric space
        sc = 0.5*(s0 + s1)
        tc = 0.5*(t0 + t1)
        p_center = _rect_point(o1, u1, v1, sc, tc)

        # Coarse: inner integral at the centroid
        I_c = _target_integral_at_source_point(p_center, n1, rect_tgt, n_tgt_gl=n_tgt_gl)

        # Refined: 4 subcenters
        sm = 0.5*(s0 + s1)
        tm = 0.5*(t0 + t1)
        sub_centers = [(s0, t0), (sm, t0), (s0, tm), (sm, tm)]  # lower-left subcenters in param space
        I_ref = 0.0
        for (sa, ta) in sub_centers:
            # each subcell centroid:
            sc_sub = sa + 0.25*(s1 - s0)
            tc_sub = ta + 0.25*(t1 - t0)
            p_sub = _rect_point(o1, u1, v1, sc_sub, tc_sub)
            I_ref += _target_integral_at_source_point(p_sub, n1, rect_tgt, n_tgt_gl=n_tgt_gl)
        I_ref *= 0.25  # average of 4 subcells

        # Contribution from this cell to the double integral:
        # ∬_{cell} I(p1) dA1  ≈  I_* * (cell area in source)
        cell_A = A1 * (s1 - s0) * (t1 - t0)
        coarse_val  = I_c   * cell_A
        refined_val = I_ref * cell_A
        err = abs(refined_val - coarse_val)
        return coarse_val, refined_val, err

    # recursive subdivision
    total = 0.0

    def recurse(s0, t0, s1, t1, depth):
        nonlocal total
        coarse_val, refined_val, err = cell_estimate(s0, t0, s1, t1)
        # Accept if improved value is close enough or depth limit reached
        if (err <= target_abs_tol) or (depth >= max_depth):
            total += refined_val
            return
        # else split into 4 children
        sm = 0.5*(s0 + s1)
        tm = 0.5*(t0 + t1)
        recurse(s0, t0, sm, tm, depth+1)
        recurse(sm, t0, s1, tm, depth+1)
        recurse(s0, tm, sm, s1, depth+1)
        recurse(sm, tm, s1, t1, depth+1)

    recurse(0.0, 0.0, 1.0, 1.0, 0)

    # F = (1/A1) ∬_{A1} ∬_{A2} K dA2 dA1
    F = total / A1
    return F
# --- adaptive tracer (unobstructed) ---
import numpy as np
from numpy.polynomial.legendre import leggauss

def vf1ai_adaptive_unobs_trace(rect_src, rect_tgt,
                               tol=5e-4, max_depth=6, n_tgt_gl=22):
    """
    Same result as your adaptive 1AI but also returns a list of cells:
    each cell = (s0, t0, s1, t1, I_avg, err_est).
    """
    o1, u1, v1 = rect_src
    o2, u2, v2 = rect_tgt
    A1 = np.linalg.norm(np.cross(u1, v1))
    n1 = np.cross(u1, v1); n1 /= np.linalg.norm(n1)

    # orient to face target
    c1 = o1 + 0.5*(u1+v1)
    c2 = o2 + 0.5*(u2+v2)
    if np.dot(n1, c2 - c1) < 0: n1 = -n1

    xi, wi = leggauss(n_tgt_gl)
    xt, wt = 0.5*(xi+1.0), 0.5*wi
    n2 = np.cross(u2, v2); n2 /= np.linalg.norm(n2)

    def rect_point(o,u,v,s,t): return o + s*u + t*v
    def area(u,v): return np.linalg.norm(np.cross(u,v))

    def inner_I_at(p1):
        inner = 0.0
        for a, wtx in enumerate(wt):
            tx = xt[a]
            for b, wty in enumerate(wt):
                ty = xt[b]
                p2 = rect_point(o2,u2,v2,tx,ty)
                r_vec = p2 - p1
                r = np.linalg.norm(r_vec)
                if r == 0:
                    continue
                r_hat = r_vec / r
                cos1 = np.dot(n1, r_hat)
                cos2 = -np.dot(n2, r_hat)
                if (cos1 <= 0) or (cos2 <= 0):
                    continue
                inner += (cos1*cos2)/(np.pi*r*r) * wtx * wty
        return area(u2,v2) * inner

    cells = []
    total = 0.0

    def recurse(s0,t0,s1,t1,depth):
        nonlocal total
        # 4 subcell centroids
        sm, tm = 0.5*(s0+s1), 0.5*(t0+t1)
        sub = [
            (s0 + 0.25*(s1-s0), t0 + 0.25*(t1-t0)),
            (sm + 0.25*(s1-s0), t0 + 0.25*(t1-t0)),
            (s0 + 0.25*(s1-s0), tm + 0.25*(t1-t0)),
            (sm + 0.25*(s1-s0), tm + 0.25*(t1-t0)),
        ]
        I_vals = [inner_I_at(rect_point(o1,u1,v1,ss,tt)) for (ss,tt) in sub]
        I_avg = 0.25 * sum(I_vals)
        cell_A = A1 * (s1-s0) * (t1-t0)
        refined_val = I_avg * cell_A

        # cheap variation-based error estimate (span of samples)
        var = (max(I_vals) - min(I_vals))
        err_est = var * cell_A * 0.25

        # record for plotting (store in [0,1]x[0,1] param space)
        cells.append((s0,t0,s1,t1, I_avg, err_est, depth))

        if (err_est <= tol) or (depth >= max_depth):
            total += refined_val
            return
        # split 2x2
        recurse(s0,t0,sm,tm,depth+1)
        recurse(sm,t0,s1,tm,depth+1)
        recurse(s0,tm,sm,t1,depth+1)
        recurse(sm,tm,s1,t1,depth+1)

    recurse(0.0,0.0,1.0,1.0,0)
    F = total / A1
    return F, cells
