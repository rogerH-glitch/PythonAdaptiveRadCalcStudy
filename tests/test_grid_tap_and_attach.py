import numpy as np

def test_grid_tap_drain_and_attach(tmp_path, monkeypatch):
    from src.util.grid_tap import capture, drain
    from src.util.plot_payload import attach_grid_field
    from src.plotting import create_heatmap_plot
    class A: pass
    args = A()
    args.method="adaptive"; args.setback=3.0; args.outdir=str(tmp_path)
    args.eval_mode="grid"; args.rc_mode="grid"; args.plot=True
    # Simulate solver producing a field
    y = np.linspace(-2,2,41); z = np.linspace(-1,1,21)
    Y,Z = np.meshgrid(y,z, indexing="xy")
    F = np.exp(-((Y-0.5)**2 + (Z-0.4)**2))
    capture(Y,Z,F)
    tapped = drain()
    result = {
        "method": "adaptive",
        "vf": float(F.max()), 
        "x_peak": 0.5, 
        "y_peak": 0.4,
        "geometry": {
            "emitter": (5.0, 2.0),
            "receiver": (5.0, 2.0),
            "setback": 3.0,
            "angle": 0.0
        }
    }
    if tapped:
        Yt, Zt, Ft = tapped
        attach_grid_field(result, Yt, Zt, Ft)
    # Plot should write a heatmap file
    create_heatmap_plot(result, args, result.get("grid_data"))
    import os
    assert any(p.name.endswith("_heatmap.png") for p in tmp_path.iterdir())
