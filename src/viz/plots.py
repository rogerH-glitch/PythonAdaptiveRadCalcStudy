from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe headless default
import matplotlib.pyplot as plt

def _rect_wire_points(center_xyz, w, h, R=np.eye(3)):
    """
    Build a rectangle lying in a local Y–Z plane at x=0, then rotate/translate into world.
    Returns arrays for X, Y, Z forming a closed loop (5 points).
    """
    c = np.array([[0, -w/2, -h/2],
                  [0,  w/2, -h/2],
                  [0,  w/2,  h/2],
                  [0, -w/2,  h/2],
                  [0, -w/2, -h/2]], dtype=float)
    pts = (np.asarray(R, float) @ c.T).T + np.asarray(center_xyz, float)
    return pts[:,0], pts[:,1], pts[:,2]

def _add_wire(ax, x, y, color, label=None):
    ax.plot(x, y, color=color, lw=2.0, label=label)
    if label:
        ax.legend(loc="best", frameon=True)

def _extract_YZF_from_grid(grid_data):
    """
    Try a few common shapes for grid_data -> (Y, Z, F).
    """
    if grid_data is None:
        return None, None, None
    # Dict with uppercase keys
    if isinstance(grid_data, dict):
        Y = grid_data.get("Y") or grid_data.get("y")
        Z = grid_data.get("Z") or grid_data.get("z")
        F = grid_data.get("F") or grid_data.get("vf") or grid_data.get("field")
        return Y, Z, F
    # Tuple/list
    if isinstance(grid_data, (tuple, list)) and len(grid_data) == 3:
        return grid_data[0], grid_data[1], grid_data[2]
    return None, None, None

def _heatmap(ax, Y, Z, F, ypk, zpk, title="View Factor Heatmap"):
    cs = ax.contourf(Y, Z, F, levels=30)
    ax.plot([ypk], [zpk], marker="*", ms=12, mfc="white", mec="red")
    ax.set_title(title)
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("Z (m)")
    ax.figure.colorbar(cs, ax=ax, label="View Factor")

def plot_geometry_and_heatmap(*, result, eval_mode, method, setback, out_png):
    """
    Draw Plan (X–Y), Elevation (X–Z) wireframes (Emitter red, Receiver black) and the Y–Z heatmap.
    Uses centres & rotations from `result` if available; otherwise falls back to a canonical layout.

    Expected optional keys in `result`:
      - 'emitter_center', 'receiver_center' as (x,y,z)
      - 'R_emitter', 'R_receiver' as 3x3 rotation matrices (world)
      - 'We','He','Wr','Hr' sizes
      - grid data: result['grid_data'] with Y,Z,F
      - 'x_peak','y_peak','vf' for the marker
    """
    We = result.get("We"); He = result.get("He")
    Wr = result.get("Wr"); Hr = result.get("Hr")
    E = np.asarray(result.get("emitter_center", (setback, 0.0, 0.0)), float)
    Rcv = np.asarray(result.get("receiver_center", (0.0, 0.0, 0.0)), float)
    REm = np.asarray(result.get("R_emitter", np.eye(3)), float)
    RRc = np.asarray(result.get("R_receiver", np.eye(3)), float)

    # Receiver-plane data
    Y, Z, F = _extract_YZF_from_grid(result.get("grid_data"))
    ypk = float(result.get("x_peak", 0.0))
    zpk = float(result.get("y_peak", 0.0))
    Fpk = float(result.get("vf", np.nan))

    fig = plt.figure(figsize=(14, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.3], wspace=0.35)
    ax_xy = fig.add_subplot(gs[0,0])
    ax_xz = fig.add_subplot(gs[0,1])
    ax_hm = fig.add_subplot(gs[0,2])

    # Wireframes
    xE, yE, zE = _rect_wire_points(E, We, He, REm)
    xR, yR, zR = _rect_wire_points(Rcv, Wr, Hr, RRc)
    _add_wire(ax_xy, xE, yE, "red", "Emitter")
    _add_wire(ax_xy, xR, yR, "black", "Receiver")
    ax_xy.set_aspect("equal"); ax_xy.set_xlabel("X (m)"); ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("Plan (X–Y)")

    _add_wire(ax_xz, xE, zE, "red")
    _add_wire(ax_xz, xR, zR, "black")
    ax_xz.set_aspect("equal"); ax_xz.set_xlabel("X (m)"); ax_xz.set_ylabel("Z (m)")
    ax_xz.set_title("Elevation (X–Z)")

    # Heatmap (if available)
    if Y is not None and Z is not None and F is not None:
        _heatmap(ax_hm, Y, Z, F, ypk, zpk)
    else:
        ax_hm.text(0.5, 0.5, "No field data available",
                   ha="center", va="center", transform=ax_hm.transAxes)
        ax_hm.set_axis_off()

    sup = f"{method.title()} – Peak VF: {Fpk:.6f} at (y,z)=({ypk:.3f},{zpk:.3f}) m | Eval Mode: {eval_mode} | Setback: {setback:.3f} m" \
          if np.isfinite(Fpk) else \
          f"{method.title()} | Eval Mode: {eval_mode} | Setback: {setback:.3f} m"
    fig.suptitle(sup)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


