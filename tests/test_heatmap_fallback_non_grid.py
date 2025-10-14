import tempfile
from pathlib import Path

from src.viz.plots import plot_geometry_and_heatmap


class _R(dict):
    """Minimal result stub without field tap to force fallback."""
    def __init__(self):
        super().__init__({
            'We': 1.2, 'He': 0.8, 'Wr': 1.2, 'Hr': 0.8,
            'setback': 3.0, 'angle': 0.0, 'rotate_axis': 'z',
            'rotate_target': 'emitter', 'receiver_offset': (0.0, 0.0),
            'x_peak': 0.0, 'y_peak': 0.0, 'vf': 0.5
        })
        # No 'field' attribute on purpose


def _tmp_png():
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    p = Path(tmp.name)
    tmp.close()
    return p


def test_heatmap_renders_for_center_mode_without_field():
    res = _R()
    out = _tmp_png()
    try:
        fig, _ = plot_geometry_and_heatmap(
            result=res, eval_mode="center", method="adaptive", setback=res['setback'], out_png=str(out), return_fig=True
        )
        assert out.exists() and out.stat().st_size > 0
    finally:
        if out.exists():
            out.unlink()


def test_heatmap_renders_for_search_mode_without_field():
    res = _R()
    out = _tmp_png()
    try:
        fig, _ = plot_geometry_and_heatmap(
            result=res, eval_mode="search", method="adaptive", setback=res['setback'], out_png=str(out), return_fig=True
        )
        assert out.exists() and out.stat().st_size > 0
    finally:
        if out.exists():
            out.unlink()
