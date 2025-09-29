# vf_compare_plots.py
import numpy as np
import matplotlib.pyplot as plt

from viewfactor_1ai_general import viewfactor_1ai_rect_to_rect
try:
    from viewfactor_1ai_adaptive_unobstructed import viewfactor_1ai_adaptive_unobstructed
    HAS_ADAPTIVE = True
except Exception:
    HAS_ADAPTIVE = False


def make_rect_from_size(center, normal, width, height):
    n = np.array(normal, dtype=float); n /= np.linalg.norm(n)
    t = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, t); t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1); t2 /= np.linalg.norm(t2)
    u = width  * t1
    v = height * t2
    origin = np.array(center, dtype=float) - 0.5*(u + v)
    return (origin, u, v)


def compute_series(W, H, setbacks, n_src=18, n_tgt=30, tol=3e-4, max_depth=8, n_tgt_gl=28):
    """Return dict with fixed/adaptive arrays for given emitter size and setbacks."""
    emitter_center = np.array([W/2, H/2, 0.0])
    emitter_normal = np.array([0.0, 0.0, 1.0])
    rect1 = make_rect_from_size(emitter_center, emitter_normal, W, H)

    F_fixed = []
    F_adapt = [] if HAS_ADAPTIVE else None

    for d in setbacks:
        receiver_center = emitter_center + d * emitter_normal
        rect2 = make_rect_from_size(receiver_center, -emitter_normal, W, H)

        Ff = viewfactor_1ai_rect_to_rect(rect1, rect2, n_src=n_src, n_tgt=n_tgt)
        F_fixed.append(Ff)

        if HAS_ADAPTIVE:
            Fa = viewfactor_1ai_adaptive_unobstructed(
                rect1, rect2, tol=tol, max_depth=max_depth, n_src_gl=6, n_tgt_gl=n_tgt_gl
            )
            F_adapt.append(Fa)

    return {
        "setbacks": np.array(setbacks, float),
        "fixed": np.array(F_fixed, float),
        "adaptive": np.array(F_adapt, float) if HAS_ADAPTIVE else None,
    }


def plot_overlay(ax, setbacks, sfpe, fixed, adaptive, title):
    ax.plot(setbacks, fixed, marker='o', linestyle='-', label='1AI fixed-grid')
    if adaptive is not None:
        ax.plot(setbacks, adaptive, marker='s', linestyle='--', label='1AI adaptive')
    if sfpe is not None:
        ax.plot(setbacks, sfpe, marker='^', linestyle=':', label='SFPE')
    ax.set_xlabel('Setback (m)')
    ax.set_ylabel('View factor F$_{1\\to 2}$')
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.legend()


if __name__ == "__main__":
    # ----- Case A: 20.02 x 1.05 m -----
    setbacks_A = [0.81, 1.80, 3.80, 6.80]
    # Your SFPE (treated as final "×4" values):
    sfpe_A = np.array([0.543752, 0.279308, 0.134132, 0.070500])

    series_A = compute_series(20.02, 1.05, setbacks_A,
                              n_src=18, n_tgt=30, tol=3e-4, max_depth=8, n_tgt_gl=28)

    # ----- Case B: 21.0 x 1.0 m -----
    setbacks_B = [0.68, 1.67, 3.67, 6.67]
    # Your SFPE "×4" values:
    sfpe_B = np.array([0.5924, 0.2864, 0.1328, 0.0692])

    series_B = compute_series(21.0, 1.0, setbacks_B,
                              n_src=18, n_tgt=30, tol=3e-4, max_depth=8, n_tgt_gl=28)

    # ----- Plot both cases -----
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    plot_overlay(axs[0], series_A["setbacks"], sfpe_A, series_A["fixed"], series_A["adaptive"],
                 'Case A: 20.02 × 1.05 m')
    plot_overlay(axs[1], series_B["setbacks"], sfpe_B, series_B["fixed"], series_B["adaptive"],
                 'Case B: 21.0 × 1.0 m')

    plt.tight_layout()
    plt.show()

# vf_sampling_diagram.py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def draw_fixed_grid(ax, nx=6, ny=4):
    ax.set_aspect('equal')
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_title('1AI fixed-grid (uniform)')
    for i in range(nx):
        for j in range(ny):
            ax.add_patch(Rectangle((i/nx, j/ny), 1/nx, 1/ny,
                                   fill=False, linewidth=1.0))
    ax.set_xticks([]); ax.set_yticks([])

def draw_adaptive_grid(ax, max_depth=4):
    ax.set_aspect('equal'); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_title('1AI adaptive (refine where needed)')
    ax.set_xticks([]); ax.set_yticks([])
    # simple heuristic: refine near (1,1) corner (as if kernel is steep there)
    def needs_refine(x0,y0,x1,y1,depth):
        # "variation" proxy = closeness to (1,1)
        xc, yc = 0.5*(x0+x1), 0.5*(y0+y1)
        d = ((1-xc)**2 + (1-yc)**2)**0.5
        # refine more when closer to corner and depth allows
        return (d < 0.6/(depth+1)) and (depth < max_depth)

    def subdivide(x0,y0,x1,y1,depth):
        if needs_refine(x0,y0,x1,y1,depth):
            xm, ym = 0.5*(x0+x1), 0.5*(y0+y1)
            subdivide(x0,y0,xm,ym,depth+1)
            subdivide(xm,y0,x1,ym,depth+1)
            subdivide(x0,ym,xm,y1,depth+1)
            subdivide(xm,ym,x1,y1,depth+1)
        else:
            ax.add_patch(Rectangle((x0,y0), x1-x0, y1-y0, fill=False, linewidth=1.2))

    subdivide(0,0,1,1,0)

if __name__ == "__main__":
    fig, axs = plt.subplots(1,2, figsize=(10,4))
    draw_fixed_grid(axs[0], nx=6, ny=4)
    draw_adaptive_grid(axs[1], max_depth=4)
    plt.tight_layout()
    plt.show()
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def plot_adaptive_vs_fixed(cells, nx=8, ny=6, title_left='Fixed-grid', title_right='Adaptive'):
    # Left: uniform grid
    fig, axs = plt.subplots(1,2, figsize=(10,4))
    ax = axs[0]
    ax.set_title(title_left); ax.set_aspect('equal'); ax.set_xlim(0,1); ax.set_ylim(0,1)
    for i in range(nx):
        for j in range(ny):
            ax.add_patch(Rectangle((i/nx, j/ny), 1/nx, 1/ny, fill=False, linewidth=1.0))
    ax.set_xticks([]); ax.set_yticks([])

    # Right: adaptive quadtree, color by I_avg (integrand magnitude)
    ax = axs[1]
    ax.set_title(title_right); ax.set_aspect('equal'); ax.set_xlim(0,1); ax.set_ylim(0,1)
    Ivals = np.array([c[4] for c in cells])
    vmin, vmax = np.percentile(Ivals, 5), np.percentile(Ivals, 95)
    for (s0,t0,s1,t1, I_avg, err, depth) in cells:
        color = plt.cm.viridis((np.clip(I_avg, vmin, vmax) - vmin) / (vmax - vmin + 1e-12))
        ax.add_patch(Rectangle((s0,t0), s1-s0, t1-t0, fill=True, facecolor=color, edgecolor='k', linewidth=0.4))
    sm = plt.cm.ScalarMappable(cmap='viridis')
    sm.set_array(Ivals); cb = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("local inner integral I(p1)")
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.show()
