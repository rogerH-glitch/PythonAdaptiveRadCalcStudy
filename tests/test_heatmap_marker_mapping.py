"""
Test heatmap marker mapping to ensure adaptive peak marker lands on correct coordinates.
This test is marked as xfail because the current implementation has marker misalignment issues.
"""
import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.viz.display_geom import build_display_geom
from src.viz.plots import plot_geometry_and_heatmap


# @pytest.mark.xfail(reason="Heatmap marker misalignment - adaptive peak marker doesn't align with actual coordinates in image space")  # FIXED
def test_heatmap_marker_mapping():
    """Test that adaptive peak marker lands exactly on the same coordinate as adaptive (y,z) in image space."""
    # Create a small anisotropic field to test marker placement
    g = build_display_geom(width=5.1, height=2.1, setback=3.0, angle=30.0, pitch=0.0,
                          angle_pivot="toe", rotate_target="emitter", dy=0.25)
    
    # Get the actual heatmap extent from the geometry
    rec_xy = g["receiver"]["xy"]
    rec_xz = g["receiver"]["xz"]
    y_min, y_max = float(np.min(rec_xy[:,1])), float(np.max(rec_xy[:,1]))
    z_min, z_max = float(np.min(rec_xz[:,1])), float(np.max(rec_xz[:,1]))
    
    # Create a mock result with a known peak location
    result = {
        "We": 5.1, "He": 2.1, "Wr": 5.1, "Hr": 2.1,
        "setback": 3.0, "angle": 30.0, "rotate_axis": "z",
        "rotate_target": "emitter", "angle_pivot": "toe",
        "x_peak": 0.0, "y_peak": 0.0, "vf": 0.3  # Known peak location at center
    }
    
    # Create a small anisotropic field with peak at center of the heatmap extent
    ny, nz = 21, 21
    y_coords = np.linspace(y_min, y_max, ny)
    z_coords = np.linspace(z_min, z_max, nz)
    Y, Z = np.meshgrid(y_coords, z_coords, indexing='xy')
    
    # Create field with peak at center of the field grid
    center_y = y_coords[ny // 2]  # Center of the field grid
    center_z = z_coords[nz // 2]  # Center of the field grid
    F = np.exp(-((Y - center_y)**2 + (Z - center_z)**2) / 0.1)
    
    # Plot and get the figure
    fig, (ax_plan, ax_elev, ax_hm) = plot_geometry_and_heatmap(
        result=result, eval_mode="center", method="adaptive",
        setback=3.0, out_png="test_marker.png", return_fig=True,
        vf_field=F, vf_grid={"y": y_coords, "z": z_coords}, 
        prefer_eval_field=True, marker_mode="adaptive",
        adaptive_peak_yz=(center_y, center_z)
    )
    
    # Get the heatmap extent and origin
    im = ax_hm.get_images()[0]
    extent = im.get_extent()  # [xmin, xmax, ymin, ymax]
    origin = im.origin  # 'lower' or 'upper'
    
    # Get the marker position in data coordinates
    lines = ax_hm.get_lines()
    marker_line = None
    for line in lines:
        if line.get_marker() == 'x':  # Adaptive peak marker
            marker_line = line
            break
    
    assert marker_line is not None, "Adaptive peak marker not found"
    
    # Get marker position in data coordinates
    marker_x, marker_y = marker_line.get_data()
    marker_data_x, marker_data_y = marker_x[0], marker_y[0]
    
    # Convert to image coordinates using the same transform as imshow
    # Note: F is transposed (F.T) in imshow, so x and y are swapped
    xmin, xmax, ymin, ymax = extent
    if origin == 'lower':
        # For 'lower' origin, y-axis is not flipped
        # F.T means x maps to z-coords and y maps to y-coords
        img_x = (marker_data_y - ymin) / (ymax - ymin) * (nz - 1)  # y maps to x in image
        img_y = (marker_data_x - xmin) / (xmax - xmin) * (ny - 1)  # x maps to y in image
    else:
        # For 'upper' origin, y-axis is flipped
        img_x = (marker_data_y - ymin) / (ymax - ymin) * (nz - 1)
        img_y = (ny - 1) - (marker_data_x - xmin) / (xmax - xmin) * (ny - 1)
    
    # Find the actual peak in the field
    # F.T means we need to swap the indices
    peak_j, peak_i = np.unravel_index(np.argmax(F), F.shape)
    # Swap indices because F is transposed
    peak_i, peak_j = peak_j, peak_i
    
    # Debug output
    print(f"Field peak at indices: ({peak_j}, {peak_i})")
    print(f"Marker image coordinates: ({img_x}, {img_y})")
    print(f"Field shape: {F.shape}")
    print(f"Extent: {extent}")
    print(f"Origin: {origin}")
    print(f"Center coordinates: ({center_y}, {center_z})")
    print(f"Marker data coordinates: ({marker_data_x}, {marker_data_y})")
    print(f"Y coords: {y_coords}")
    print(f"Z coords: {z_coords}")
    print(f"Y coords center index: {ny // 2}, value: {y_coords[ny // 2]}")
    print(f"Z coords center index: {nz // 2}, value: {z_coords[nz // 2]}")
    
    # The marker should land on the same pixel as the field peak
    # Allow small tolerance for floating point precision and coordinate transformation
    assert abs(img_x - peak_i) < 1.0, f"Marker x-coordinate {img_x} doesn't match field peak {peak_i}"
    assert abs(img_y - peak_j) < 1.0, f"Marker y-coordinate {img_y} doesn't match field peak {peak_j}"
    
    plt.close(fig)
