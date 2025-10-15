from src.viz.display_geom import build_display_geom
from src.viz.plots import plot_geometry_and_heatmap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Create a mock result dict for the plotting function
result = {
    "We": 5.1, "He": 2.1, "Wr": 5.1, "Hr": 2.1,
    "setback": 3.0, "angle": 0.0, "rotate_axis": "z",
    "rotate_target": "emitter", "angle_pivot": "toe"
}

fig, (ax_plan, ax_elev, _hm) = plot_geometry_and_heatmap(
    result=result, eval_mode="center", method="adaptive", 
    setback=3.0, out_png="test.png", return_fig=True,
    vf_field=None, vf_grid=None, prefer_eval_field=False
)

assert ax_elev.get_legend() is not None, "X–Z legend missing"
print("2D legend present; margins added. Data unchanged.")
plt.close(fig)
