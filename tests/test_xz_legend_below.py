"""
Test that XZ legend appears below the plot area.
This test is marked as xfail because the current implementation has legend overlap issues.
"""
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.viz.display_geom import build_display_geom
from src.viz.plots import plot_geometry_and_heatmap


# @pytest.mark.xfail(reason="XZ legend overlaps with plot area instead of appearing below it")  # FIXED
def test_xz_legend_below_axes():
    """Test that XZ legend bbox y-coordinate < 0 in axes coordinates."""
    # Create geometry
    g = build_display_geom(width=5.1, height=2.1, setback=3.0, angle=30.0, pitch=0.0,
                          angle_pivot="toe", rotate_target="emitter", dy=0.25)
    
    result = {
        "We": 5.1, "He": 2.1, "Wr": 5.1, "Hr": 2.1,
        "setback": 3.0, "angle": 30.0, "rotate_axis": "z",
        "rotate_target": "emitter", "angle_pivot": "toe"
    }
    
    # Plot and get the figure
    fig, (ax_plan, ax_elev, ax_hm) = plot_geometry_and_heatmap(
        result=result, eval_mode="center", method="adaptive",
        setback=3.0, out_png="test_legend.png", return_fig=True,
        vf_field=None, vf_grid=None, prefer_eval_field=False
    )
    
    # Get the XZ legend
    xz_legend = ax_elev.get_legend()
    assert xz_legend is not None, "XZ legend not found"
    
    # Get legend bbox in axes coordinates
    # The legend bbox_to_anchor is set to (0.0, -0.18) in axes coordinates
    # We need to check the actual legend position by looking at its window extent
    # and converting to axes coordinates
    legend_window = xz_legend.get_window_extent()
    bbox_axes = ax_elev.transAxes.inverted().transform(legend_window)
    legend_y = bbox_axes[0][1]  # bottom-left y coordinate in axes units
    
    # Legend should be below the axes (y < 0 in axes coordinates)
    assert legend_y < 0, f"Legend y-coordinate {legend_y} should be < 0 (below axes)"
    
    # Check that legend is not overlapping with plot content
    # Get the plot area bounds
    plot_bounds = ax_elev.get_position()
    plot_bottom = plot_bounds.y0
    
    # Legend should be well below the plot area
    assert legend_y < plot_bottom - 0.1, f"Legend too close to plot area: y={legend_y}, plot_bottom={plot_bottom}"
    
    plt.close(fig)


# @pytest.mark.xfail(reason="Constrained layout may not be properly applied")  # FIXED
def test_constrained_layout_applied():
    """Test that constrained layout is applied to the figure."""
    # Create geometry
    g = build_display_geom(width=5.1, height=2.1, setback=3.0, angle=30.0, pitch=0.0,
                          angle_pivot="toe", rotate_target="emitter", dy=0.25)
    
    result = {
        "We": 5.1, "He": 2.1, "Wr": 5.1, "Hr": 2.1,
        "setback": 3.0, "angle": 30.0, "rotate_axis": "z",
        "rotate_target": "emitter", "angle_pivot": "toe"
    }
    
    # Plot and get the figure
    fig, (ax_plan, ax_elev, ax_hm) = plot_geometry_and_heatmap(
        result=result, eval_mode="center", method="adaptive",
        setback=3.0, out_png="test_constrained.png", return_fig=True,
        vf_field=None, vf_grid=None, prefer_eval_field=False
    )
    
    # Check that constrained layout is enabled
    assert fig.get_constrained_layout(), "Constrained layout should be enabled"
    
    plt.close(fig)
