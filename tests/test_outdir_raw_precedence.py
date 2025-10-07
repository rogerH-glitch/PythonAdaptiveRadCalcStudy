from types import SimpleNamespace
from pathlib import Path

def test_csv_uses_raw_outdir(tmp_path):
    from src.cli_results import _csv_path
    args = SimpleNamespace(outdir=str(tmp_path/"mutated"/"results"),
                           _outdir_user=str(tmp_path/"results"),
                           method="adaptive")
    p = _csv_path(args, "adaptive")
    # must be under the raw path, not the mutated one
    assert p == tmp_path/"results"/"adaptive.csv"

def test_plotting_uses_raw_outdir(tmp_path):
    from src.plotting import create_heatmap_plot
    import numpy as np
    class A: pass
    args = A()
    args.method="adaptive"; args.setback=6.0; args.eval_mode="grid"; args.rc_mode="grid"
    args.plot=True  # Required for plotting
    args.outdir=str(tmp_path/"mutated"/"results")
    args._outdir_user=str(tmp_path/"results")
    Y,Z = np.zeros((3,3)), np.zeros((3,3)); F = np.zeros((3,3))
    result = {
        "method": "adaptive",
        "vf":0.0,"x_peak":0.0,"y_peak":0.0,
        "geometry": {"emitter": (5.0, 2.0), "receiver": (5.0, 2.0), "setback": 6.0, "angle": 0.0}
    }
    create_heatmap_plot(result, args, {"Y":Y,"Z":Z,"F":F})
    files = list((tmp_path/"results").glob("*_heatmap.png"))
    assert files, "expected a heatmap in the RAW outdir"
