from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from src.util.plot_payload import has_field


def _compute_limits(F: np.ndarray, vmin: float | None, vmax: float | None) -> Tuple[float, float]:
    if vmin is None:
        try:
            vmin = float(np.nanmin(F))
        except Exception:
            vmin = float(np.min(F))
    if vmax is None:
        try:
            vmax = float(np.nanmax(F))
        except Exception:
            vmax = float(np.max(F))
    return vmin, vmax


def render_receiver_heatmap(
    ax,
    Y: np.ndarray,
    Z: np.ndarray,
    F: np.ndarray,
    ypk: float | None = None,
    zpk: float | None = None,
    *,
    levels: int = 60,
    vmin: float | None = None,
    vmax: float | None = None,
    add_colorbar: bool = True,
    colorbar_axes=None,
):
    """Draw a high-detail receiver heatmap with consistent styling.

    Returns (QuadContourSet, vmin, vmax).
    """
    # True scale between Y and Z axes
    ax.set_aspect("equal")
    # Robust vmin/vmax
    if vmin is None or vmax is None:
        arr = np.asarray(F)
        vmin = np.nanmin(arr) if vmin is None else vmin
        vmax = np.nanmax(arr) if vmax is None else vmax
    cs = ax.contourf(Y, Z, F, levels=levels, vmin=vmin, vmax=vmax)
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("Z (m)")
    if add_colorbar:
        cax = colorbar_axes if colorbar_axes is not None else None
        cb = ax.figure.colorbar(cs, ax=ax if cax is None else None, cax=cax)
        cb.set_label("View Factor")
    if (ypk is not None) and (zpk is not None) and np.isfinite(ypk) and np.isfinite(zpk):
        y_s, z_s = float(ypk), float(zpk)
        ax.plot([y_s], [z_s], marker="*", ms=14, mfc="white", mec="red", zorder=10)
        ax.annotate(f"Peak: ({y_s:.3f}, {z_s:.3f})",
                    xy=(y_s, z_s), xycoords="data",
                    xytext=(8, 8), textcoords="offset points",
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
                    fontsize=9)
    return cs, vmin, vmax


def draw_heatmap(ax, Y, Z, F, *, ypk=None, zpk=None):
    """
    Draw a true-scale receiver heatmap; Y (width-axis), Z (height-axis).
    """
    # Avoid ambiguous truth on numpy arrays; validate shape/size
    if not has_field({"Y": Y, "Z": Z, "F": F}):
        ax.text(0.5, 0.5, "No field data available", ha="center", va="center",
                transform=ax.transAxes)
        return
    
    # Use the existing render_receiver_heatmap function for actual drawing
    render_receiver_heatmap(ax, Y, Z, F, ypk=ypk, zpk=zpk)



