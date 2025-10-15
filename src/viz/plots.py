# PLOTTING PRINCIPLE: never massage visuals. Render computed geometry/fields as-is.

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe headless default
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .display_geom import build_display_geom
from ..util.plot_payload import has_field

def _xz_silhouette(corners):
    """Orthographic projection onto XZ, silhouette as axis-aligned bbox."""
    x = corners[:, 0]
    z = corners[:, 2]
    x0, x1 = float(np.min(x)), float(np.max(x))
    z0, z1 = float(np.min(z)), float(np.max(z))
    # rectangle in order
    return np.array([[x0, z0], [x0, z1], [x1, z1], [x1, z0], [x0, z0]], dtype=float)

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


def subcell_quadratic_peak(F, gy, gz, j, i):
    """Estimate sub-cell peak around (j,i) using a 2D quadratic fit over a 3×3 neighborhood.
    Returns (y_hat, z_hat) in physical units. If ill-conditioned, falls back to grid node.
    """
    F = np.asarray(F)
    gy = np.asarray(gy, float)
    gz = np.asarray(gz, float)
    ny, nz = F.shape
    if j <= 0 or j >= ny - 1 or i <= 0 or i >= nz - 1:
        return float(gy[i]), float(gz[j])
    Ys, Zs, Vs = [], [], []
    for jj in (j - 1, j, j + 1):
        for ii in (i - 1, i, i + 1):
            v = F[jj, ii]
            if np.isfinite(v):
                Ys.append(gy[ii])
                Zs.append(gz[jj])
                Vs.append(v)
    if len(Vs) < 6:
        return float(gy[i]), float(gz[j])
    Y = np.array(Ys, float)
    Z = np.array(Zs, float)
    V = np.array(Vs, float)
    X = np.column_stack([Y * Y, Z * Z, Y * Z, Y, Z, np.ones_like(Y)])
    try:
        coeffs, *_ = np.linalg.lstsq(X, V, rcond=None)
    except np.linalg.LinAlgError:
        return float(gy[i]), float(gz[j])
    a, b, c, d, e, f = coeffs
    A = np.array([[2 * a, c], [c, 2 * b]], float)
    bvec = -np.array([d, e], float)
    try:
        y_hat, z_hat = np.linalg.solve(A, bvec)
        y_hat = float(np.clip(y_hat, gy.min(), gy.max()))
        z_hat = float(np.clip(z_hat, gz.min(), gz.max()))
    except np.linalg.LinAlgError:
        y_hat, z_hat = float(gy[i]), float(gz[j])
    return y_hat, z_hat

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
                              vf_field=None, vf_grid=None, prefer_eval_field: bool=False, heatmap_interp: str="bilinear",
                              marker_mode: str="adaptive", adaptive_peak_yz=None, subcell_fit: bool=True, title: str | None=None,
                              debug_plots: bool=False):
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
    
    # Extract field data safely - allow explicit adaptive peak override
    if adaptive_peak_yz is not None and isinstance(adaptive_peak_yz, (tuple, list)) and len(adaptive_peak_yz) == 2:
        ypk = float(adaptive_peak_yz[0])
        zpk = float(adaptive_peak_yz[1])
    else:
        ypk = float(result.get("x_peak", 0.0))
        zpk = float(result.get("y_peak", 0.0))
    Fpk = float(result.get("vf", np.nan))

    # --- G1: layout & scaling ---
    # Larger canvas with tuned widths; preserve true scale within axes
    fig = plt.figure(
        figsize=(17, 5.5),
        constrained_layout=True,
        dpi=plt.rcParams.get("figure.dpi", 100),
    )
    gs = fig.add_gridspec(
        nrows=1,
        ncols=4,                 # XY, XZ, Heatmap, Colorbar
        width_ratios=[1.2, 1.0, 1.8, 0.08],
        wspace=0.15,
    )
    ax_xy = fig.add_subplot(gs[0, 0])   # Plan (X–Y)
    ax_xz = fig.add_subplot(gs[0, 1])   # Elevation (X–Z)
    ax_hm = fig.add_subplot(gs[0, 2])   # Heatmap (Y–Z)
    cax   = fig.add_subplot(gs[0, 3])   # Colorbar axis
    # --- end G1 ---

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
    ax_xy.legend(loc="upper left", bbox_to_anchor=(0, 1.02), frameon=False)

    # Elevation view (X-Z) - use orthographic silhouette rectangles
    emitter_corners = np.array(display_geom["corners3d"]["emitter"])
    receiver_corners = np.array(display_geom["corners3d"]["receiver"])
    
    # Compute orthographic silhouettes
    em_xz_outline = _xz_silhouette(emitter_corners)
    rc_xz_outline = _xz_silhouette(receiver_corners)
    
    # Convert to numpy arrays for bounds computation
    em_xz = em_xz_outline
    rec_xz = rc_xz_outline
    
    # Plot silhouettes as polylines
    ax_xz.plot(em_xz_outline[:, 0], em_xz_outline[:, 1], 
               color="red", linewidth=2.0, alpha=0.8,
                        label=f"Emitter {result.get('We', 5.0):.3g}×{result.get('He', 2.0):.3g} m")
    
    ax_xz.plot(rc_xz_outline[:, 0], rc_xz_outline[:, 1], 
               color="black", linewidth=2.0, alpha=0.8,
                        label=f"Receiver {result.get('Wr', 5.0):.3g}×{result.get('Hr', 2.0):.3g} m")
    
    ax_xz.set_aspect("equal")
    ax_xz.set_xlabel("X (m)")
    ax_xz.set_ylabel("Z (m)")
    ax_xz.set_title("Elevation (X–Z)")
    ax_xz.legend(loc="upper left", bbox_to_anchor=(0, 1.02), frameon=False)
    
    # Set axis bounds to include both panels with increased padding
    b = _compute_bounds_from_panels(em_xy, rec_xy, em_xz, rec_xz, pad=0.08)
    (xlim_xy, ylim_xy) = b["xy"]
    (xlim_xz, zlim_xz) = b["xz"]
    ax_xy.set_xlim(*xlim_xy)
    ax_xy.set_ylim(*ylim_xy)
    ax_xz.set_xlim(*xlim_xz)
    ax_xz.set_ylim(*zlim_xz)
    
    # Set true-scale aspect ratios (equal scale, square panels)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xz.set_aspect("equal", adjustable="box")
    
    # Add cosmetic margins for better visual presentation
    ax_xy.margins(x=0.02, y=0.02)
    ax_xz.margins(x=0.02, y=0.02)

    # Denser ticks and minor ticks for readability
    try:
        from matplotlib.ticker import MaxNLocator, AutoMinorLocator
        for _ax in (ax_xy, ax_xz):
            _ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
            _ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            _ax.xaxis.set_minor_locator(AutoMinorLocator())
            _ax.yaxis.set_minor_locator(AutoMinorLocator())
    except Exception:
        pass

    # --- G2: legends below x-axis, outside the plots ---
    try:
        fig.set_constrained_layout(True)
        def _legend_below(_ax):
            lg = _ax.get_legend()
            if lg is not None:
                _ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(0.0, -0.18),
                    frameon=False,
                    borderaxespad=0.0,
                    handlelength=2.0,
                    handletextpad=0.8,
                )
        _legend_below(ax_xy)
        _legend_below(ax_xz)
    except Exception:
        pass

    # If an explicit dense grid field is provided and preferred, use it first
    if prefer_eval_field and vf_field is not None and isinstance(vf_field, np.ndarray):
        F = vf_field
        if isinstance(vf_grid, dict):
            gy = vf_grid.get("y", None)
            gz = vf_grid.get("z", None)
            if gy is not None and gz is not None:
                # Build meshgrid matching F orientation (nz, ny)
                Y, Z = np.meshgrid(np.asarray(gy, float), np.asarray(gz, float), indexing="xy")
                # Compute peak from the dense field
                j, i = np.unravel_index(np.nanargmax(F), F.shape)
                ypk = float(gy[i])  # gy is 1D array, i is the column index
                zpk = float(gz[j])  # gz is 1D array, j is the row index
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
        hm = ax_hm.imshow(F.T, origin="lower", extent=extent, aspect="equal", cmap="inferno", interpolation=heatmap_interp)
    except Exception as e:
        # Fallback: draw a blank heatmap grid to avoid crashing the CLI
        # and surface a friendly message in the figure title.
        from matplotlib.colors import Normalize
        hm = ax_hm.imshow(np.zeros_like(F.T), origin="lower", extent=extent, aspect="equal", cmap="inferno", norm=Normalize(0, 1), interpolation=heatmap_interp)
        heatmap_title = f"Heatmap (fallback; data unavailable: {type(e).__name__})"
    # Markers and diagnostics
    # --- P0 DEBUG START ---
    try:
        if debug_plots and vf_field is not None and vf_grid is not None and isinstance(vf_grid, dict):
            import numpy as _np
            gy_dbg, gz_dbg = vf_grid.get("y"), vf_grid.get("z")
            try:
                jj_dbg, ii_dbg = _np.unravel_index(_np.nanargmax(vf_field), _np.asarray(vf_field).shape)
                _y_grid_dbg = float(gy_dbg[jj_dbg]) if gy_dbg is not None and jj_dbg is not None else None
                _z_grid_dbg = float(gz_dbg[ii_dbg]) if gz_dbg is not None and ii_dbg is not None else None
            except Exception as _e:
                jj_dbg = ii_dbg = None
                _y_grid_dbg = _z_grid_dbg = None
            adaptive_peak_yz = (float(ypk), float(zpk)) if _np.isfinite(ypk) and _np.isfinite(zpk) else None
            print(f"[p0] marker_mode={marker_mode} | grid_idx={(jj_dbg, ii_dbg)} grid_yz=({_y_grid_dbg},{_z_grid_dbg}) | "
                  f"adaptive_yz={adaptive_peak_yz} | field_shape={None if vf_field is None else _np.asarray(vf_field).shape}")
    except Exception:
        pass
    # --- P0 DEBUG END ---
    try:
        show_grid = marker_mode in ("grid", "both")
        show_adapt = marker_mode in ("adaptive", "both")
        # Compute grid-argmax if we have a dense field and grid axes
        grid_y = vf_grid.get("y") if isinstance(vf_grid, dict) else None
        grid_z = vf_grid.get("z") if isinstance(vf_grid, dict) else None
        if show_grid and isinstance(F, np.ndarray) and grid_y is not None and grid_z is not None:
            gy = np.asarray(grid_y, float)
            gz = np.asarray(grid_z, float)
            jj, ii = np.unravel_index(np.nanargmax(F), F.shape)
            # Map grid node, then optional sub-cell quadratic refinement for marker only
            if subcell_fit:
                y_star, z_star = subcell_quadratic_peak(F, gy, gz, jj, ii)
            else:
                y_star, z_star = float(gy[ii]), float(gz[jj])
            ax_hm.plot([y_star], [z_star], marker=(5, 1, 0), markersize=6, markeredgewidth=0.8,
                       color="crimson", markeredgecolor="white", linestyle="None", zorder=10)
            y_grid, z_grid = y_star, z_star
        if show_adapt:
            ax_hm.plot([ypk], [zpk], marker="x", markersize=6, markeredgewidth=1.0,
                       color="white", markeredgecolor="black", linestyle="None", zorder=11)
        # Diagnostics: grid spacing and distance from adaptive peak to nearest node
        if grid_y is not None and grid_z is not None:
            gy = np.asarray(grid_y, float)
            gz = np.asarray(grid_z, float)
            if gy.size >= 2 and gz.size >= 2 and np.isfinite(ypk) and np.isfinite(zpk):
                dy = float((gy[-1] - gy[0]) / (gy.size - 1))
                dz = float((gz[-1] - gz[0]) / (gz.size - 1))
                jn = int(np.argmin(np.abs(gy - ypk)))
                in_ = int(np.argmin(np.abs(gz - zpk)))
                d = float(np.hypot(ypk - gy[jn], zpk - gz[in_]))
                print(f"[diag] grid Δy≈{dy:.3f} Δz≈{dz:.3f} | dist_to_nearest_node≈{d:.3f} m")
        # Attach meta for tests
        placed_grid = None
        if show_grid:
            try:
                if (y_grid is not None) and np.isfinite(y_grid) and np.isfinite(z_grid):
                    placed_grid = (float(y_grid), float(z_grid))
            except Exception:
                placed_grid = None
        placed_adaptive = None
        if show_adapt and np.isfinite(ypk) and np.isfinite(zpk):
            placed_adaptive = (float(ypk), float(zpk))
        try:
            fig._vf_plot_meta = {
                "markers": {"grid": placed_grid, "adaptive": placed_adaptive},
                "field_shape": None if vf_field is None else tuple(np.asarray(vf_field).shape),
            }
        except Exception:
            pass
    except Exception:
        pass
    ax_hm.set_title(heatmap_title)
    ax_hm.set_xlabel("Y (m)")
    ax_hm.set_ylabel("Z (m)")
    fig.colorbar(hm, cax=cax, label="View Factor")

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
    
    # Include yaw/pivot/target in title for clarity
    yaw = float(result.get("angle", 0.0))
    piv = str(result.get("angle_pivot", "toe"))
    tgt = str(result.get("rotate_target", "emitter"))
    yaw_text = f" | Yaw {yaw:.0f}° (pivot={piv}, target={tgt})" if abs(yaw) > 1e-9 else ""
    if title is not None:
        fig.suptitle(title)
    else:
        # Build a clean two-line title: line1 (peak/eval/setback), line2 (yaw/pivot/target/offset)
        line1 = (
            f"{method.title()} — Peak VF: {Fpk:.6f} at (y,z)=({ypk:.3f},{zpk:.3f}) m | Eval Mode: {eval_mode} | Setback: {setback:.3f} m"
            if np.isfinite(Fpk)
            else f"{method.title()} | Eval Mode: {eval_mode} | Setback: {setback:.3f} m"
        )
        line2 = f"Yaw {yaw:.0f}° (pivot={piv}, target={tgt}){offset_text}"
        fig.suptitle(line1 + "\n" + line2, fontsize=18)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    if return_fig:
        return fig, (ax_xy, ax_xz, ax_hm)
    plt.close(fig)


