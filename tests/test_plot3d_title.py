from src.viz.plot3d import plot_geometry_3d

def test_plot3d_title_contains_angle_and_offset(tmp_path):
    res = {
        "angle": 20.0, "rotate_axis": "z", "rotate_target": "emitter", "angle_pivot": "toe",
        "receiver_center": (0.0, 0.6, 0.4)
    }
    html = tmp_path / "t.html"
    fig = plot_geometry_3d(res, str(html), return_fig=True)
    title = fig.layout.title.text
    assert "Yaw 20°" in title and "Offset (dy,dz)=(0.600,0.400)" in title
