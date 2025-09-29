# viewfactor_sweep.py
import numpy as np
import matplotlib.pyplot as plt
from viewfactor_demo_plot import coarse_view_factor

def make_rects(sep):
    rect1 = (np.array([0.,0.,0.]),
             np.array([1.,0.,0.]),
             np.array([0.,1.,0.]))
    rect2 = (np.array([0.,0.,sep]),
             np.array([1.,0.,0.]),
             np.array([0.,1.,0.]))
    return rect1, rect2

if __name__ == "__main__":
    seps = np.linspace(0.25, 3.0, 18)
    Fvals = []
    for s in seps:
        r1, r2 = make_rects(s)
        F = coarse_view_factor(r1, r2, nu=18, nv=18)
        Fvals.append(F)
        print(f"sep={s:>4.2f} m  F1->2≈ {F:.4f}")

    plt.plot(seps, Fvals, marker='o')
    plt.xlabel('Separation (m)')
    plt.ylabel('F1→2')
    plt.title('View factor vs. separation (two 1×1 squares)')
    plt.grid(True)
    plt.show()
