"""
Plotting utilities for view factor calculations and peak visualization.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
import os


def create_heatmap_plot(
    result: Dict[str, Any], 
    args: Any,
    grid_data: Optional[Dict[str, np.ndarray]] = None
) -> None:
    """
    Create a heatmap visualization of view factor distribution across the receiver.
    
    Args:
        result: Calculation results dictionary
        args: Parsed command-line arguments
        grid_data: Optional grid data for heatmap (x_coords, y_coords, values)
    """
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left plot: Geometry overview
    _plot_geometry_overview(ax1, em_w, em_h, rc_w, rc_h, setback, angle)
    
    # Right plot: View factor heatmap
    if grid_data is not None:
        _plot_view_factor_heatmap(ax2, result, grid_data, rc_w, rc_h)
    else:
        _plot_peak_location(ax2, result, rc_w, rc_h)
    
    # Add title and save
    method = result['method'].title()
    rc_mode = result.get('rc_mode', 'center')
    vf_peak = result['vf']
    x_peak, y_peak = result.get('x_peak', 0.0), result.get('y_peak', 0.0)
    
    fig.suptitle(f'{method} Method - Peak VF: {vf_peak:.6f} at ({x_peak:.3f}, {y_peak:.3f}) m\n'
                 f'RC Mode: {rc_mode} | Setback: {setback:.3f} m', fontsize=12)
    
    # Save plot
    os.makedirs(args.outdir, exist_ok=True)
    plot_filename = f"{method.lower()}_peak_heatmap.png"
    plot_path = os.path.join(args.outdir, plot_filename)
    
    try:
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nHeatmap plot saved to: {plot_path}")
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
