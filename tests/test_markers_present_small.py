from main import eval_case
from src.viz.display_geom import build_display_geom


def test_markers_present_small():
    g = build_display_geom(width=5.1, height=2.1, setback=3, angle=30, angle_pivot="toe", rotate_target="emitter", dy=0.25)
    out = eval_case(method="adaptive", eval_mode="grid", geom=g, plot=False)
    assert out.get("vf_field") is not None
    assert "grid_y" in out and "grid_z" in out


