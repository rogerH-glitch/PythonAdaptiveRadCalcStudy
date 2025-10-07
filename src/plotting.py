"""
Plotting utilities for view factor calculations and peak visualization.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
import os
from pathlib import Path
from .util.filenames import join_with_ts
from .util.paths import get_outdir


def _extract_grid_YZF(grid_data: Any, result: dict | None = None) -> Tuple[Any, Any, Any]:
    """
    Try hard to obtain (Y, Z, F) in this order:
      1) grid_data dict with 'Y','Z','F' (or common aliases),
      2) grid_data as (Y,Z,F) tuple/list,
      3) fields directly on result dict ('Y','Z','F' or 'grid_Y','grid_Z','field').
    """
    # 1/2: look inside grid_data
    if grid_data is not None:
        if isinstance(grid_data, dict):
            Y = grid_data.get("Y")
            if Y is None:
                Y = grid_data.get("y")
            Z = grid_data.get("Z")
            if Z is None:
                Z = grid_data.get("z")
            F = grid_data.get("F")
            if F is None:
                F = grid_data.get("vf")
            if F is None:
                F = grid_data.get("field")
            if Y is not None and Z is not None and F is not None:
                return Y, Z, F
        if isinstance(grid_data, (tuple, list)) and len(grid_data) == 3:
            return grid_data[0], grid_data[1], grid_data[2]
    # 3) fall back to the result dict
    if isinstance(result, dict):
        for yk, zk, fk in (("Y","Z","F"), ("grid_Y","grid_Z","field"), ("y","z","vf_field")):
            Y = result.get(yk); Z = result.get(zk); F = result.get(fk)
            if Y is not None and Z is not None and F is not None:
                return Y, Z, F
    return None, None, None


def create_heatmap_plot(
    result: Dict[str, Any], 
    args: Any,
    grid_data: Optional[Dict[str, np.ndarray]] = None
) -> None:
    """
    Legacy figure: keep the left X–Z 'Geometry Overview' panel, but fix the right heat-map to plot Y–Z,
    apply receiver offset if available, and place the peak star consistently.
    """
    from pathlib import Path
    if not args.plot:
        return
    
    # Ensure headless matplotlib
    try:
        import matplotlib
        if os.name == "nt" or not os.environ.get("DISPLAY"):
            matplotlib.use("Agg")
    except ImportError:
        return
    
    # Extract geometry
    geom = result['geometry']
    em_w, em_h = geom['emitter']
    rc_w, rc_h = geom['receiver']
    setback = geom['setback']
    angle = geom['angle']
    
    # Create figure
    fig, (ax_geom, ax_hm) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.0, 1.3]})
    
    # Left plot: Geometry overview (X–Z)
    _plot_geometry_overview(ax_geom, em_w, em_h, rc_w, rc_h, setback, angle)
    
    # Right: Y–Z heat-map using evaluated field (preferred)
    Y, Z, F = _extract_grid_YZF(grid_data, result)
    ypk = result.get("x_peak") or result.get("y_peak")  # historical naming: x_peak/y_peak represent (y,z)
    zpk = result.get("y_peak") or result.get("z_peak")
    if Y is not None and Z is not None and F is not None:
        cs = ax_hm.contourf(Y, Z, F, levels=30)
        if ypk is not None and zpk is not None:
            ax_hm.plot([float(ypk)], [float(zpk)], marker="*", ms=12, mfc="white", mec="red")
        ax_hm.set_xlabel("Y (m)")
        ax_hm.set_ylabel("Z (m)")
        fig.colorbar(cs, ax=ax_hm, label="View Factor")
    else:
        # Fallback: draw a centered placeholder and warn in-title
        ax_hm.text(0.5, 0.5, "Field not provided to plotting", ha="center", va="center", transform=ax_hm.transAxes)
        ax_hm.set_axis_off()
    
    method = result['method'].title()
    sup = f"{method} Method - Peak VF: {result.get('vf', float('nan')):.6f} at ({float(ypk):.3f}, {float(zpk):.3f}) m" \
          if (ypk is not None and zpk is not None) else f"{method} Method"
    eval_mode = getattr(args,'eval_mode',getattr(args,'rc_mode','center'))
    fig.suptitle(sup + f"\nEval Mode: {eval_mode} | Setback: {float(setback):.3f} m")
    
    # Save plot
    # use normalized outdir (might be Path)
    from .util.paths import get_outdir
    out = join_with_ts(get_outdir(args.outdir), "heatmap.png")
    
    try:
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"\nHeatmap plot saved to: {out}")
    except Exception as e:
        print(f"Warning: Could not save plot: {e}")
    finally:
        plt.close(fig)


def _plot_geometry_overview(ax, em_w: float, em_h: float, rc_w: float, rc_h: float, 
                          setback: float, angle: float) -> None:
    """Plot geometry overview showing emitter and receiver positions."""
    ax.set_aspect('equal')
    
    # Emitter (top)
    emitter_rect = plt.Rectangle((-em_w/2, setback), em_w, 0.1, 
                               facecolor='red', alpha=0.7, label='Emitter')
    ax.add_patch(emitter_rect)
    
    # Receiver (bottom)
    receiver_rect = plt.Rectangle((-rc_w/2, 0), rc_w, 0.1, 
                                facecolor='blue', alpha=0.7, label='Receiver')
    ax.add_patch(receiver_rect)
    
    # Connection lines
    ax.plot([-em_w/2, -rc_w/2], [setback, 0], 'k--', alpha=0.5)
    ax.plot([em_w/2, rc_w/2], [setback, 0], 'k--', alpha=0.5)
    
    ax.set_xlim(-max(em_w, rc_w)/2 - 0.5, max(em_w, rc_w)/2 + 0.5)
    ax.set_ylim(-0.5, setback + 0.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Geometry Overview')
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_view_factor_heatmap(ax, result: Dict[str, Any], grid_data: Dict[str, np.ndarray], 
                            rc_w: float, rc_h: float) -> None:
    """Plot heatmap of view factor values across receiver surface."""
    x_coords = grid_data['x_coords']
    y_coords = grid_data['y_coords']
    values = grid_data['values']
    
    # Create meshgrid for contour plot
    X, Y = np.meshgrid(x_coords, y_coords)
    Z = values.reshape(X.shape)
    
    # Create heatmap
    im = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('View Factor', rotation=270, labelpad=15)
    
    # Mark peak location
    x_peak, y_peak = result.get('x_peak', 0.0), result.get('y_peak', 0.0)
    ax.plot(x_peak, y_peak, 'r*', markersize=15, markeredgecolor='white', 
            markeredgewidth=2, label=f'Peak: ({x_peak:.3f}, {y_peak:.3f})')
    
    # Set limits and labels
    ax.set_xlim(-rc_w/2, rc_w/2)
    ax.set_ylim(-rc_h/2, rc_h/2)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('View Factor Heatmap')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


def _plot_peak_location(ax, result: Dict[str, Any], rc_w: float, rc_h: float) -> None:
    """Plot simple peak location when no grid data is available."""
    x_peak, y_peak = result.get('x_peak', 0.0), result.get('y_peak', 0.0)
    vf_peak = result['vf']
    
    # Draw receiver outline
    receiver_rect = plt.Rectangle((-rc_w/2, -rc_h/2), rc_w, rc_h, 
                                facecolor='lightblue', alpha=0.3, edgecolor='blue')
    ax.add_patch(receiver_rect)
    
    # Mark peak location
    ax.plot(x_peak, y_peak, 'r*', markersize=20, markeredgecolor='white', 
            markeredgewidth=2, label=f'Peak VF: {vf_peak:.6f}')
    
    # Add text annotation
    ax.annotate(f'({x_peak:.3f}, {y_peak:.3f})', 
                xy=(x_peak, y_peak), xytext=(10, 10), 
                textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Set limits and labels
    ax.set_xlim(-rc_w/2 - 0.1, rc_w/2 + 0.1)
    ax.set_ylim(-rc_h/2 - 0.1, rc_h/2 + 0.1)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Peak Location')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


def generate_grid_data_for_plotting(
    em_w: float, em_h: float, rc_w: float, rc_h: float, setback: float, angle: float,
    method: str, method_params: Dict[str, Any], grid_n: int = 21
) -> Optional[Dict[str, np.ndarray]]:
    """
    Generate grid data for heatmap plotting.
    
    Args:
        em_w, em_h: Emitter dimensions
        rc_w, rc_h: Receiver dimensions
        setback: Setback distance
        angle: Rotation angle
        method: Calculation method
        method_params: Method-specific parameters
        grid_n: Grid resolution
        
    Returns:
        Dictionary with x_coords, y_coords, and values arrays, or None if not available
    """
    try:
        from .peak_locator import create_vf_evaluator
        
        # Create evaluator
        evaluator = create_vf_evaluator(method, em_w, em_h, rc_w, rc_h, setback, angle, **method_params)
        
        # Generate grid coordinates
        x_coords = np.linspace(-rc_w/2, rc_w/2, grid_n)
        y_coords = np.linspace(-rc_h/2, rc_h/2, grid_n)
        
        # Evaluate at each grid point
        values = np.zeros((grid_n, grid_n))
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                vf, _ = evaluator(x, y)
                values[j, i] = vf  # Note: j,i for proper orientation
        
        return {
            'x_coords': x_coords,
            'y_coords': y_coords,
            'values': values.flatten()
        }
        
    except Exception as e:
        print(f"Warning: Could not generate grid data for plotting: {e}")
        return None
