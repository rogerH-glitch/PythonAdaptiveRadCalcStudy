import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

# Minimal fake geometry just to let the plot run (4-corner rectangles)
geom = {
    "xy": {
        "emitter": np.array([[0, -1], [0, 1]], float),
        "receiver": np.array([[3, -1], [3, 1]], float),
    },
    "corners3d": {
        "emitter": np.array([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]], float),
        "receiver": np.array([[3, -1, -1], [3, 1, -1], [3, 1, 1], [3, -1, 1]], float),
    },
}

# 5x5 grid with a peak off-center
gy = np.linspace(-2.5, 2.5, 5)
gz = np.linspace(-1.0, 1.0, 5)
Y, Z = np.meshgrid(gy, gz, indexing="ij")
F = np.exp(-((Y + 0.5) ** 2 + (Z - 0.2) ** 2))  # peak near (-0.5, 0.2)

# Adaptive peak (pretend evaluator result)
adaptive_peak = (-0.52, 0.18)

from src.viz.plots import plot_geometry_and_heatmap

fig = plot_geometry_and_heatmap(
    result={
        "We": 2.0, "He": 2.0, "Wr": 2.0, "Hr": 2.0,
        "rotate_axis": "z", "rotate_target": "emitter", "angle_pivot": "toe", "angle": 0.0,
    },
    eval_mode="grid", method="adaptive", setback=3.0, out_png="ignore.png",
    return_fig=True,
    vf_field=F,
    vf_grid={"y": gy, "z": gz},
    prefer_eval_field=True,
    adaptive_peak_yz=adaptive_peak,
    marker_mode="both",
    subcell_fit=True,
    title="wiring test",
)
print("OK: plot function returned a figure; markers computed.")
plt.close(fig[0] if isinstance(fig, tuple) else fig)


