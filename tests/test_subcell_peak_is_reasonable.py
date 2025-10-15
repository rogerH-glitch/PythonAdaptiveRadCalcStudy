import numpy as np
from src.viz.display_geom import build_display_geom
from main import eval_case


def test_quadratic_peak_runs():
    g = build_display_geom(width=5.1, height=2.1, setback=3, angle=30, angle_pivot="toe", rotate_target="emitter", dy=0.25)
    out = eval_case(method="adaptive", eval_mode="grid", geom=g, plot=False)
    F = out["vf_field"]; gy = out["grid_y"]; gz = out["grid_z"]
    j, i = np.unravel_index(np.nanargmax(F), F.shape)
    from src.viz.plots import subcell_quadratic_peak
    y_hat, z_hat = subcell_quadratic_peak(F, gy, gz, j, i)
    assert np.isfinite(y_hat) and np.isfinite(z_hat)
    assert (np.min(gy) - 1e-9) <= y_hat <= (np.max(gy) + 1e-9)
    assert (np.min(gz) - 1e-9) <= z_hat <= (np.max(gz) + 1e-9)


