def test_plotting_principle_banner_exists():
    """Test that the plotting principle banner exists in both plotting files."""
    banner = "PLOTTING PRINCIPLE: never massage visuals. Render computed geometry/fields as-is."
    
    # Check src/viz/plots.py
    with open("src/viz/plots.py", "r") as f:
        plots_content = f.read()
    assert banner in plots_content, "Plotting principle banner missing from src/viz/plots.py"
    
    # Check src/viz/plot3d.py
    with open("src/viz/plot3d.py", "r") as f:
        plot3d_content = f.read()
    assert banner in plot3d_content, "Plotting principle banner missing from src/viz/plot3d.py"
