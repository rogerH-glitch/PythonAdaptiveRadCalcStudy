# scripts/test_true_scale_axes.py
from src.viz.display_geom import build_display_geom
from src.viz.plots import plot_geometry_and_heatmap
import matplotlib
matplotlib.use("Agg")

def test_true_scale_axes():
    g = build_display_geom(width=5.1, height=2.1, setback=3, angle=30, pitch_deg=0, 
                          angle_pivot="toe", rotate_target="emitter", dy=0.25)
    
    # Create a mock result dict for the plotting function
    result = {
        "We": 5.1, "He": 2.1, "Wr": 5.1, "Hr": 2.1,
        "setback": 3.0, "angle": 30.0, "rotate_axis": "z",
        "rotate_target": "emitter", "angle_pivot": "toe"
    }
    
    fig, (ax_plan, ax_elev, ax_hm) = plot_geometry_and_heatmap(
        result=result, eval_mode="center", method="adaptive", 
        setback=3.0, out_png="test.png", return_fig=True,
        vf_field=None, vf_grid=None, prefer_eval_field=False
    )
    
    # Check that aspect ratios are set to "equal" (matplotlib converts to numeric)
    assert ax_plan.get_aspect() == "equal" or ax_plan.get_aspect() == 1.0, f"Plan view aspect should be 'equal' or 1.0, got '{ax_plan.get_aspect()}'"
    assert ax_elev.get_aspect() == "equal" or ax_elev.get_aspect() == 1.0, f"Elevation view aspect should be 'equal' or 1.0, got '{ax_elev.get_aspect()}'"
    
    print("true-scale axes set")
    print(f"Plan view aspect: {ax_plan.get_aspect()}")
    print(f"Elevation view aspect: {ax_elev.get_aspect()}")
    print(f"Heatmap aspect: {ax_hm.get_aspect()}")

if __name__ == "__main__":
    test_true_scale_axes()
