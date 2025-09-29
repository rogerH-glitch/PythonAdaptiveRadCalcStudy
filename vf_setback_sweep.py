# vf_setback_sweep.py
import numpy as np

# --- numeric solvers you already have ---
from viewfactor_1ai_general import viewfactor_1ai_rect_to_rect
try:
    from viewfactor_1ai_adaptive_unobstructed import viewfactor_1ai_adaptive_unobstructed
    _HAS_ADAPTIVE = True
except Exception:
    _HAS_ADAPTIVE = False


# ---------- small geometry helper ----------
def make_rect_from_size(center, normal, width, height):
    """
    Build (origin, u_vec, v_vec) for a rectangle with given center, plane normal,
    and full edge lengths width (u) and height (v).
    """
    n = np.array(normal, dtype=float)
    n /= np.linalg.norm(n)
    t = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, t);  t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1); t2 /= np.linalg.norm(t2)
    u = width  * t1
    v = height * t2
    origin = np.array(center, dtype=float) - 0.5*(u + v)
    return (origin, u, v)


# ---------- runner ----------
def run_case(label, W, H, setbacks, sfpe_vals=None,
             n_src_fixed=18, n_tgt_fixed=30,
             tol=3e-4, max_depth=8, n_tgt_gl=28):
    """
    Prints a table of F1->2 for a given emitter size and setbacks.
    Receiver: parallel, same size, centered, facing the emitter.
    Optionally compares against SFPE values (list matching 'setbacks').
    """
    print(f"\n=== {label} ===")
    print(f"Emitter: {W:.2f} m × {H:.2f} m; receiver parallel, same size, centered")

    # emitter @ z=0, facing +Z
    emitter_center = np.array([W/2, H/2, 0.0])
    emitter_normal = np.array([0.0, 0.0, 1.0])
    rect1 = make_rect_from_size(emitter_center, emitter_normal, W, H)

    # header
    cols = ["Setback (m)", "1AI fixed-grid"]
    if _HAS_ADAPTIVE:
        cols += ["1AI adaptive", "|Δ| (abs)", "Rel. Δ"]
    if sfpe_vals is not None:
        cols.append("SFPE")
    widths = [12, 15] + ([14, 11, 9] if _HAS_ADAPTIVE else []) + ([10] if sfpe_vals else [])
    header = " | ".join([f"{c:>{w}}" for c, w in zip(cols, widths)])
    print(header)
    print("-" * len(header))

    for i, d in enumerate(setbacks):
        # receiver is d along +Z from emitter, faces back (-Z)
        receiver_center = emitter_center + d * emitter_normal
        rect2 = make_rect_from_size(receiver_center, -emitter_normal, W, H)

        F_fixed = viewfactor_1ai_rect_to_rect(rect1, rect2,
                                              n_src=n_src_fixed, n_tgt=n_tgt_fixed)

        row_vals = [f"{d:12.2f}", f"{F_fixed:15.6f}"]

        if _HAS_ADAPTIVE:
            F_adapt = viewfactor_1ai_adaptive_unobstructed(
                rect1, rect2,
                tol=tol, max_depth=max_depth,
                n_src_gl=6, n_tgt_gl=n_tgt_gl
            )
            diff = F_adapt - F_fixed
            rel  = (diff / F_fixed) if F_fixed > 0 else 0.0
            row_vals += [f"{F_adapt:14.6f}", f"{abs(diff):11.6f}", f"{rel:8.2%}"]

        if sfpe_vals is not None:
            # assume sfpe_vals aligned with setbacks
            row_vals += [f"{sfpe_vals[i]:10.4f}"]

        print(" | ".join(row_vals))

    if not _HAS_ADAPTIVE:
        print("Note: adaptive integrator not found; only fixed-grid results printed.")
        print("To enable it, add `viewfactor_1ai_adaptive_unobstructed.py` to your project.")


# ---------- main ----------
if __name__ == "__main__":
    # Case A: 20.02 × 1.05 m
    setbacks_A = [0.81, 1.80, 3.80, 6.80]
    # Your SFPE (treated as final view factors you've chosen to use):
    sfpe_A = [0.543752, 0.279308, 0.134132, 0.070500]
    run_case("Case A (20.02 × 1.05 m)", 20.02, 1.05, setbacks_A, sfpe_vals=sfpe_A,
             n_src_fixed=18, n_tgt_fixed=30, tol=3e-4, max_depth=8, n_tgt_gl=28)

    # Case B: 21.0 × 1.0 m
    setbacks_B = [0.68, 1.67, 3.67, 6.67]
    # Your SFPE values for 21×1 (as provided):
    sfpe_B = [4*0.1481, 4*0.0716, 4*0.0332, 4*0.0173]
    run_case("Case B (21.0 × 1.0 m)", 21.0, 1.0, setbacks_B, sfpe_vals=sfpe_B,
             n_src_fixed=18, n_tgt_fixed=30, tol=3e-4, max_depth=8, n_tgt_gl=28)
