# tests/test_heatmap_star_aligns.py
import numpy as np
from src.viz.display_geom import build_display_geom
from main import eval_case

def test_star_aligns_with_grid_peak():
    """Test that the heatmap star marks the peak of the vf_field when using grid mode."""
    g = build_display_geom(width=5.1, height=2.1, setback=3.0, angle=30.0, pitch_deg=0.0, 
                          angle_pivot="toe", rotate_target="emitter", dy=0.25)
    out = eval_case(method="adaptive", eval_mode="grid", geom=g, plot=False)
    
    F = out["vf_field"]
    gy, gz = out["grid_y"], out["grid_z"]
    
    # Compute peak from the field
    j, i = np.unravel_index(np.nanargmax(F), F.shape)
    y_star, z_star = gy[i], gz[j]
    
    # Verify the peak coordinates are valid floats
    assert isinstance(y_star, float) and isinstance(z_star, float)
    
    # Verify the peak is within the expected range
    assert -3.0 <= y_star <= 3.0, f"y_star {y_star} out of expected range"
    assert -1.5 <= z_star <= 1.5, f"z_star {z_star} out of expected range"
    
    # Verify the peak value is reasonable
    peak_value = float(np.nanmax(F))
    assert 0.1 < peak_value < 0.4, f"peak value {peak_value} out of expected range"
    
    print(f"Grid peak at (y={y_star:.3f}, z={z_star:.3f}) with value {peak_value:.3f}")
