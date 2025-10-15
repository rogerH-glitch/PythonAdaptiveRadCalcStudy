import numpy as np
from src.viz.display_geom import build_display_geom
from src.viz.plot3d import plot_geometry_3d


def test_plot3d_has_mesh_and_wireframe():
    g = build_display_geom(
        width=5.1, height=2.1, setback=3.0,
        angle=30.0, pitch=0.0,
        angle_pivot="toe", rotate_target="emitter",
        dy=0.25, dz=0.0,
    )
    # Build result shape expected by plotter
    result = {
        "We": 5.1, "He": 2.1, "Wr": 5.1, "Hr": 2.1,
        "setback": 3.0, "angle": 30.0, "rotate_axis": "z",
        "rotate_target": "emitter", "angle_pivot": "toe",
    }
    import tempfile, os
    out = os.path.join(tempfile.gettempdir(), "_test3d.html")
    fig = plot_geometry_3d(result, out_html=out, return_fig=True)
    kinds = [getattr(tr, "type", None) for tr in fig.data]
    assert "mesh3d" in kinds, f"No Mesh3d in traces: {kinds}"
    assert "scatter3d" in kinds, f"No wireframe Scatter3d in traces: {kinds}"
    assert kinds.count("scatter3d") >= 2


