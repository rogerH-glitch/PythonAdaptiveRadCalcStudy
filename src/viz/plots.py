# PLOTTING PRINCIPLE: never massage visuals. Render computed geometry/fields as-is.

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe headless default
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .display_geom import build_display_geom
from ..util.plot_payload import has_field

def _compute_bounds_from_panels(p_em_xy, p_rec_xy, p_em_xz, p_rec_xz, pad=0.02):
    """Compute axis bounds to include both emitter and receiver panels with padding."""
    def _bounds(a, b):
        both = np.vstack([a, b])
        xmin, ymin = both.min(axis=0); xmax, ymax = both.max(axis=0)
        dx, dy = xmax - xmin, ymax - ymin
        ex = pad if dx < 1e-12 else pad * dx
        ey = pad if dy < 1e-12 else pad * dy
        return (xmin - ex, xmax + ex), (ymin - ey, ymax + ey)
    (xlim_xy, ylim_xy) = _bounds(p_em_xy, p_rec_xy)
    (xlim_xz, zlim_xz) = _bounds(p_em_xz, p_rec_xz)
    return {"xy": (xlim_xy, ylim_xy), "xz": (xlim_xz, zlim_xz)}

PLACEHOLDER_NY = 40
PLACEHOLDER_NZ = 40

def _build_placeholder_field(receiver_w: float, receiver_h: float):
    """
    Return (Y, Z, F) placeholder arrays when no field is available.
    True-scale axes: Y spans [-W/2, +W/2], Z spans [-H/2, +H/2].
    F is zeros with a single max at center to keep peak marker behavior stable.
    """
    w, h = float(receiver_w), float(receiver_h)
    y = np.linspace(-w/2.0, w/2.0, PLACEHOLDER_NY)
    z = np.linspace(-h/2.0, h/2.0, PLACEHOLDER_NZ)
    Y, Z = np.meshgrid(y, z)  # Z rows, Y cols shape
    F = np.zeros_like(Y, dtype=float)
    # Seed a tiny nonzero at the geometric center so "peak" logic has a location
    F[Z.shape[0]//2, Y.shape[1]//2] = 1e-12
    return Y, Z, F

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

def _extract_YZF_from_result_fallback(result):
    """
    Fallback extractor that looks directly on the result dict for common keys.
    Tries ('Y','Z','F'), then ('grid_Y','grid_Z','field'), then ('y','z','vf_field').
    """
    if not isinstance(result, dict):
        return None, None, None
    for yk, zk, fk in (("Y","Z","F"), ("grid_Y","grid_Z","field"), ("y","z","vf_field")):
        Y = result.get(yk); Z = result.get(zk); F = result.get(fk)
        if Y is not None and Z is not None and F is not None:
            return Y, Z, F
    return None, None, None

def _heatmap(ax, Y, Z, F, ypk, zpk, title="View Factor Heatmap"):
    cs = ax.contourf(Y, Z, F, levels=30)
    ax.plot([ypk], [zpk], marker="*", ms=12, mfc="white", mec="red")
    ax.set_title(title)
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("Z (m)")
    ax.figure.colorbar(cs, ax=ax, label="View Factor")

def plot_geometry_and_heatmap(*, result, eval_mode, method, setback, out_png, return_fig: bool=False,
                              vf_field=None, vf_grid=None, prefer_eval_field: bool=False, heatmap_interp: str="bilinear"):
    """
    Draw Plan (X–Y), Elevation (X–Z) wireframes (Emitter red, Receiver black) and the Y–Z heatmap.
    Uses centralized display geometry for consistent rotation/translation.
    """
    # Create a mock args object for build_display_geom
    class MockArgs:
        def __init__(self, result):
            self.emitter = (result.get("We", 5.0), result.get("He", 2.0))
            self.receiver = (result.get("Wr", 5.0), result.get("Hr", 2.0))
            self.setback = setback
            self.rotate_axis = result.get("rotate_axis", "z")
            self.angle = result.get("angle", 0.0)
            self.angle_pivot = result.get("angle_pivot", "toe")
            self.rotate_target = result.get("rotate_target", "emitter")
    
    args = MockArgs(result)
    display_geom = build_display_geom(args, result)
    
    # Extract field data safely
    ypk = float(result.get("x_peak", 0.0))
    zpk = float(result.get("y_peak", 0.0))
    Fpk = float(result.get("vf", np.nan))

    fig = plt.figure(figsize=(14, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.3], wspace=0.35)
    ax_xy = fig.add_subplot(gs[0,0])
    ax_xz = fig.add_subplot(gs[0,1])
    ax_hm = fig.add_subplot(gs[0,2])

    # Plan view (X-Y) - use edge segments
    emitter_xy = display_geom["xy"]["emitter"]
    receiver_xy = display_geom["xy"]["receiver"]
    
    # Convert to numpy arrays for bounds computation
    em_xy = np.array(emitter_xy) if emitter_xy else np.array([[0, 0], [0, 0]])
    rec_xy = np.array(receiver_xy) if receiver_xy else np.array([[0, 0], [0, 0]])
    
    if emitter_xy:
        (x0, y0), (x1, y1) = emitter_xy
        ax_xy.plot([x0, x1], [y0, y1], color="red", lw=2.0, label=f"Emitter {result.get('We', 5.0):.3g}×{result.get('He', 2.0):.3g} m")
    
    if receiver_xy:
        (x0, y0), (x1, y1) = receiver_xy
        ax_xy.plot([x0, x1], [y0, y1], color="black", lw=2.0, label=f"Receiver {result.get('Wr', 5.0):.3g}×{result.get('Hr', 2.0):.3g} m")
    
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("Plan (X–Y)")
    ax_xy.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

    # Elevation view (X-Z) - use thin rectangles
    emitter_xz = display_geom["xz"]["emitter"]
    receiver_xz = display_geom["xz"]["receiver"]
    
    # Convert to numpy arrays for bounds computation
    em_xz = np.array([[emitter_xz["x"], emitter_xz["z0"]], [emitter_xz["x"], emitter_xz["z1"]]]) if emitter_xz else np.array([[0, 0], [0, 0]])
    rec_xz = np.array([[receiver_xz["x"], receiver_xz["z0"]], [receiver_xz["x"], receiver_xz["z1"]]]) if receiver_xz else np.array([[0, 0], [0, 0]])
    
    if emitter_xz:
        rect = Rectangle((emitter_xz["x"] - 0.01, emitter_xz["z0"]), 0.02, 
                        emitter_xz["z1"] - emitter_xz["z0"], 
                        facecolor="red", edgecolor="red", linewidth=1.0, alpha=0.7,
                        label=f"Emitter {result.get('We', 5.0):.3g}×{result.get('He', 2.0):.3g} m")
        ax_xz.add_patch(rect)
    
    if receiver_xz:
        rect = Rectangle((receiver_xz["x"] - 0.01, receiver_xz["z0"]), 0.02,
                        receiver_xz["z1"] - receiver_xz["z0"],
                        facecolor="black", edgecolor="black", linewidth=1.0, alpha=0.7,
                        label=f"Receiver {result.get('Wr', 5.0):.3g}×{result.get('Hr', 2.0):.3g} m")
        ax_xz.add_patch(rect)
    
    ax_xz.set_aspect("equal")
    ax_xz.set_xlabel("X (m)")
    ax_xz.set_ylabel("Z (m)")
    ax_xz.set_title("Elevation (X–Z)")
    ax_xz.legend(loc="upper left", frameon=False)
    
    # Set axis bounds to include both panels with increased padding
    b = _compute_bounds_from_panels(em_xy, rec_xy, em_xz, rec_xz, pad=0.05)
    (xlim_xy, ylim_xy) = b["xy"]
    (xlim_xz, zlim_xz) = b["xz"]
    ax_xy.set_xlim(*xlim_xy)
    ax_xy.set_ylim(*ylim_xy)
    ax_xz.set_xlim(*xlim_xz)
    ax_xz.set_ylim(*zlim_xz)
    
    # Add cosmetic margins for better visual presentation
    ax_xy.margins(x=0.02, y=0.02)
    ax_xz.margins(x=0.02, y=0.02)

    # If an explicit dense grid field is provided and preferred, use it first
    if prefer_eval_field and vf_field is not None and isinstance(vf_field, np.ndarray):
        F = vf_field
        if isinstance(vf_grid, dict):
            gy = vf_grid.get("y", None)
            gz = vf_grid.get("z", None)
            if gy is not None and gz is not None:
                # Build meshgrid matching F orientation (nz, ny)
                Y, Z = np.meshgrid(np.asarray(gy, float), np.asarray(gz, float), indexing="xy")
    else:
        # Field capture (tap preferred) or fallback to result field
        field = getattr(result, "field", None)
        Y = getattr(field, "Y", None) if field is not None else None
        Z = getattr(field, "Z", None) if field is not None else None
        F = getattr(field, "F", None) if field is not None else None

    # If field missing, always ensure we have something to render.
    if Y is None or Z is None or F is None:
        rW = getattr(result, "receiver_w", getattr(result, "receiver_width", 1.0))
        rH = getattr(result, "receiver_h", getattr(result, "receiver_height", 1.0))
        # Prefer sampler if available, regardless of eval_mode; else placeholder
        try:
            from src.rc_eval.grid_eval import sample_receiver_grid  # type: ignore
            Y, Z, F = sample_receiver_grid(result, method=method,
                                           ny=PLACEHOLDER_NY, nz=PLACEHOLDER_NZ,
                                           receiver_w=rW, receiver_h=rH)
        except Exception:
            Y, Z, F = _build_placeholder_field(rW, rH)

    # Heatmap rendering (no extra normalisation; vf_field is already 0..1 view factor)
    heatmap_title = "View Factor Heatmap"
    
    # Derive extent from receiver physical coordinates
    y_min, y_max = float(np.min(rec_xy[:,1])), float(np.max(rec_xy[:,1]))
    z_min, z_max = float(np.min(rec_xz[:,1])), float(np.max(rec_xz[:,1]))
    extent = [y_min, y_max, z_min, z_max]
    
    try:
        # Use imshow with proper extent for physical coordinates and interpolation
        hm = ax_hm.imshow(F.T, origin="lower", extent=extent, aspect="auto", cmap="inferno", interpolation=heatmap_interp)
    except Exception as e:
        # Fallback: draw a blank heatmap grid to avoid crashing the CLI
        # and surface a friendly message in the figure title.
        from matplotlib.colors import Normalize
        hm = ax_hm.imshow(np.zeros_like(F.T), origin="lower", extent=extent, aspect="auto", cmap="inferno", norm=Normalize(0, 1), interpolation=heatmap_interp)
        heatmap_title = f"Heatmap (fallback; data unavailable: {type(e).__name__})"
    # peak marker + label (existing behavior uses argmax on F)
    ax_hm.plot([ypk], [zpk], marker="*", ms=12, mfc="white", mec="red")
    ax_hm.set_title(heatmap_title)
    ax_hm.set_xlabel("Y (m)")
    ax_hm.set_ylabel("Z (m)")
    fig.colorbar(hm, ax=ax_hm, label="View Factor")

    # Plausibility check for accidental re-normalisation
    try:
        if F is not None:
            vmax = float(np.nanmax(F))
            if vmax < 1e-6:
                print("[warn] heatmap vf_field peak is extremely small; check normalisation")
    except Exception:
        pass

    # Title with offset information
    # Calculate offset from receiver_center if not explicitly provided
    if "dy" in result and "dz" in result:
        dy, dz = result["dy"], result["dz"]
    else:
        # Extract from receiver_center
        rc_center = result.get("receiver_center", (0.0, 0.0, 0.0))
        dy, dz = rc_center[1], rc_center[2]
    
    offset_text = f" | Offset (dy,dz)=({dy:.3f},{dz:.3f}) m" if dy != 0 or dz != 0 else ""
    
    sup = f"{method.title()} – Peak VF: {Fpk:.6f} at (y,z)=({ypk:.3f},{zpk:.3f}) m | Eval Mode: {eval_mode} | Setback: {setback:.3f} m{offset_text}" \
          if np.isfinite(Fpk) else \
          f"{method.title()} | Eval Mode: {eval_mode} | Setback: {setback:.3f} m{offset_text}"
    fig.suptitle(sup)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    if return_fig:
        return fig, (ax_xy, ax_xz, ax_hm)
    plt.close(fig)


