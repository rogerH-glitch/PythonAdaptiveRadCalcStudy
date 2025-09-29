# viewfactor_1ai_parallel.py
import numpy as np
from numpy.polynomial.legendre import leggauss

def viewfactor_1ai_parallel_unit(sep, n_src=16, n_tgt=24):
    """
    Correct single-area structure for two parallel, aligned 1×1 squares:
      - Source: z=0,   [0,1]×[0,1], normal +z
      - Target: z=sep, [0,1]×[0,1], normal -z
    F = (1/π) ∬_{A1} [ cosθ1 * ∬_{A2} (cosθ2 / r^2) dA2 ] dA1
    We integrate source with n_src×n_src GL points, and target with n_tgt×n_tgt GL points.
    """
    # GL nodes/weights on [0,1]
    xi_s, wi_s = leggauss(n_src); xs = 0.5*(xi_s + 1.0); ws = 0.5*wi_s
    xi_t, wi_t = leggauss(n_tgt); xt = 0.5*(xi_t + 1.0); wt = 0.5*wi_t

    z = float(sep)
    total = 0.0

    for i, wx in enumerate(ws):
        x1 = xs[i]
        for j, wy in enumerate(ws):
            y1 = xs[j]
            # source point p1 = (x1, y1, 0), source normal = +z
            # For a target point p2 = (x2, y2, z), r = p2 - p1 = (dx, dy, z)
            # cosθ1 = (n1 · r̂) = z / r ; cosθ2 = (-n2 · r̂) = z / r
            # kernel = cosθ1 * cosθ2 / r^2 = z^2 / r^4
            inner = 0.0
            for a, wtx in enumerate(wt):
                x2 = xt[a]
                for b, wty in enumerate(wt):
                    y2 = xt[b]
                    dx = x2 - x1
                    dy = y2 - y1
                    r2 = dx*dx + dy*dy + z*z
                    inner += (z*z) / (r2*r2) * wtx * wty
            # accumulate with source weights
            total += inner * wx * wy

    # Multiply by Jacobians for [0,1]×[0,1] mappings: already accounted via 0.5 scaling in ws/wt
    # Final factor 1/π per definition
    F = total / np.pi
    return F

if __name__ == "__main__":
    for d in [0.25, 0.5, 1.0, 1.5, 2.0]:
        F = viewfactor_1ai_parallel_unit(d, n_src=18, n_tgt=30)
        print(f"sep={d:>4.2f}  F1->2 (correct 1AI numeric kernel) ≈ {F:.6f}")
