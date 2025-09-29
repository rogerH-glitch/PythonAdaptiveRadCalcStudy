# run_viewfactor_scenarios.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- solvers ---
from viewfactor_1ai_general import viewfactor_1ai_rect_to_rect
from viewfactor_1ai_adaptive_unobstructed import (
    viewfactor_1ai_adaptive_unobstructed,
    vf1ai_adaptive_unobs_trace,         # tracer that records the adaptive mesh
)

# ======================== CONFIG ========================
# Add/edit cases here. Each case is (label, width_m, height_m, setbacks_m, sfpe_vals_or_None)
CASES = [
    # ("A: 20.02 × 1.05 m", 20.02, 1.05, [0.81, 1.80, 3.80, 6.80],
    #  [0.543752, 0.279308, 0.134132, 0.070500]),
    #
    # ("B: 21.0 × 1.0 m",   21.00, 1.00, [0.68, 1.67, 3.67, 6.67],
    #  [0.5924, 0.2864, 0.1328, 0.0692]),
    ("A: 5.1 × 2.1 m",   5.10, 2.10, [0.81, 1.80, 3.80, 6.80],
     [0.776961, 0.45287, 0.177354, 0.066576]),

    ("B: 5.1 × 2.1 m",   5.10, 2.10, [0.68, 1.67, 3.67, 6.67],
     [0.828574, 0.485836, 0.18696, 0.068939]),

    # # ← Your new test: change these numbers to try any emitter, e.g. 5.1 × 2.1 m
    # ("C: 5.1 × 2.1 m",   5.10, 2.10, [0.05, 0.2, 2.00, 35.00],
    #  None),  # set to a list if you have SFPE refs
    #
    # # ← Your new test: change these numbers to try any emitter, e.g. 5.1 × 2.1 m
    # ("D: 5.1 × 2.1 m",   5.10, 2.10, [0.50, 1.00, 2.00, 4.00],
    #  None),  # set to a list if you have SFPE refs
]

# Numeric “quality” knobs (bump up for accuracy, down for speed)
N_SRC_FIXED = 18
N_TGT_FIXED = 30
ADAPT_TOL   = 3e-4
ADAPT_DEPTH = 8
ADAPT_TGTGL = 28
# ========================================================

def make_rect_from_size(center, normal, width, height):
    n = np.array(normal, dtype=float); n /= np.linalg.norm(n)
    t = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, t);  t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1); t2 /= np.linalg.norm(t2)
    u = width  * t1
    v = height * t2
    origin = np.array(center, dtype=float) - 0.5*(u + v)
    return (origin, u, v)

def run_case(label, W, H, setbacks, sfpe_vals=None):
    print(f"\n=== {label} ===")
    print(f"Emitter: {W:.2f} m × {H:.2f} m; receiver parallel, same size, centered")

    emitter_center = np.array([W/2, H/2, 0.0])
    emitter_normal = np.array([0.0, 0.0, 1.0])
    rect1 = make_rect_from_size(emitter_center, emitter_normal, W, H)

    cols = ["Setback (m)", "1AI fixed-grid", "1AI adaptive", "|Δ| (abs)", "Rel. Δ"]
    if sfpe_vals is not None:
        cols.append("SFPE")
    header = " | ".join([f"{c:>14}" for c in cols])
    print(header)
    print("-"*len(header))

    fixed_vals, adapt_vals = [], []
    for i, d in enumerate(setbacks):
        rect2 = make_rect_from_size(emitter_center + d*emitter_normal, -emitter_normal, W, H)

        F_fixed = viewfactor_1ai_rect_to_rect(rect1, rect2, n_src=N_SRC_FIXED, n_tgt=N_TGT_FIXED)
        F_adapt = viewfactor_1ai_adaptive_unobstructed(
            rect1, rect2, tol=ADAPT_TOL, max_depth=ADAPT_DEPTH, n_src_gl=6, n_tgt_gl=ADAPT_TGTGL
        )
        diff = F_adapt - F_fixed
        rel  = (diff / F_fixed) if F_fixed > 0 else 0.0
        row = [f"{d:14.2f}", f"{F_fixed:14.6f}", f"{F_adapt:14.6f}", f"{abs(diff):14.6f}", f"{rel:13.2%}"]
        if sfpe_vals is not None:
            row.append(f"{sfpe_vals[i]:>14.4f}")
        print(" | ".join(row))
        fixed_vals.append(F_fixed); adapt_vals.append(F_adapt)

    return np.array(setbacks, float), np.array(fixed_vals), np.array(adapt_vals), np.array(sfpe_vals) if sfpe_vals is not None else None

def overlay_plot(all_series):
    # all_series = list of (title, setbacks, fixed, adapt, sfpe_or_None)
    n = len(all_series)
    fig, axs = plt.subplots(1, n, figsize=(6*n, 4), sharey=True)
    if n == 1: axs = [axs]
    for ax, (title, x, y_fixed, y_adapt, y_sfpe) in zip(axs, all_series):
        ax.plot(x, y_fixed, 'o-', label='1AI fixed-grid')
        ax.plot(x, y_adapt, 's--', label='1AI adaptive')
        if y_sfpe is not None:
            ax.plot(x, y_sfpe, '^:', label='SFPE')
        ax.set_xlabel('Setback (m)'); ax.set_ylabel('View factor F1→2'); ax.grid(True, alpha=0.3)
        ax.set_title(title); ax.legend()
    fig.tight_layout(); plt.show()

def plot_adaptive_mesh(rect1, rect2, title_left='Uniform (conceptual 8×6)', title_right='Adaptive cells'):
    # Get traced cells from the adaptive routine (colored by local inner integral)
    F_adapt, cells = vf1ai_adaptive_unobs_trace(rect1, rect2, tol=ADAPT_TOL, max_depth=ADAPT_DEPTH, n_tgt_gl=ADAPT_TGTGL)

    fig, axs = plt.subplots(1,2, figsize=(10,4))

    # Left: conceptual uniform grid for contrast
    ax = axs[0]; ax.set_aspect('equal'); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title(title_left); ax.set_xticks([]); ax.set_yticks([])
    NX, NY = 8, 6
    for i in range(NX):
        for j in range(NY):
            ax.add_patch(Rectangle((i/NX, j/NY), 1/NX, 1/NY, fill=False, linewidth=1.0))

    # Right: actual adaptive cells (color by I_avg magnitude)
    ax = axs[1]; ax.set_aspect('equal'); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title(f"{title_right}\nF≈{F_adapt:.3f}"); ax.set_xticks([]); ax.set_yticks([])
    Ivals = np.array([c[4] for c in cells])
    vmin, vmax = np.percentile(Ivals, 5), np.percentile(Ivals, 95)
    for (s0,t0,s1,t1, I_avg, err, depth) in cells:
        color = plt.cm.viridis((np.clip(I_avg, vmin, vmax) - vmin) / (vmax - vmin + 1e-12))
        ax.add_patch(Rectangle((s0,t0), s1-s0, t1-t0, facecolor=color, edgecolor='k', linewidth=0.4))
    sm = plt.cm.ScalarMappable(cmap='viridis'); sm.set_array(Ivals)
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04); cb.set_label("local inner integral I(p1)")
    fig.tight_layout(); plt.show()

if __name__ == "__main__":
    # 1) Tables + collect series for plots
    series = []
    for (label, W, H, dlist, sfpe) in CASES:
        x, y_fixed, y_adapt, y_sfpe = run_case(label, W, H, dlist, sfpe_vals=sfpe)
        series.append((label, x, y_fixed, y_adapt, y_sfpe))

    # 2) Overlay plot for all cases
    overlay_plot(series)

    # 3) Show an adaptive mesh example that is NOT trivial:
    #    tilt the receiver 20° about the emitter's local u-axis (width axis)
    label, W, H, dlist, _ = CASES[-1]
    d = min(dlist)
    emitter_center = np.array([W / 2, H / 2, 0.0])
    n_emit = np.array([0.0, 0.0, 1.0])

    # Source rect (emitter)
    rect1 = make_rect_from_size(emitter_center, n_emit, W, H)
    o1, u1, v1 = rect1

    # Build a tilted receiver:
    tilt_deg = 20.0
    theta = np.deg2rad(tilt_deg)
    ux = u1 / np.linalg.norm(u1)  # rotate about emitter's u-axis through the receiver center
    c_recv = emitter_center + d * n_emit

    # Rodrigues rotation around axis ux
    K = np.array([[0, -ux[2], ux[1]],
                  [ux[2], 0, -ux[0]],
                  [-ux[1], ux[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    n_recv = R @ (-n_emit)  # start facing the emitter, then tilt
    # Make the receiver with the tilted normal (same size)
    rect2 = make_rect_from_size(c_recv, n_recv, W, H)

    plot_adaptive_mesh(
        rect1, rect2,
        title_left='1AI fixed-grid (uniform conceptual)',
        title_right=f'1AI adaptive (tilted {tilt_deg:.0f}°), {label}, setback={d} m'
    )
