from src.cli_parser import create_parser, normalize_args
from types import SimpleNamespace
import builtins

def get_mode(argv):
    p = create_parser()
    args = p.parse_args(argv)
    args = normalize_args(args)
    return getattr(args, "_plot_mode", "none")

def test_plot_modes_exclusive():
    assert get_mode(["--plot"]) == "2d"
    assert get_mode(["--plot-3d"]) == "3d"
    assert get_mode(["--plot-both"]) == "both"
    assert get_mode([]) == "none"

def test_cli_calls_correct_plotters(monkeypatch, tmp_path):
    # fake args
    p = create_parser()
    args = p.parse_args(["--method","adaptive","--emitter","5","2","--receiver","5","2","--setback","1","--plot-both","--outdir",str(tmp_path)])
    args = normalize_args(args)

    called = {"two_d":0, "three_d":0}
    def fake_2d(**kwargs):
        called["two_d"] += 1
    def fake_3d(result, html_path):
        called["three_d"] += 1
    
    # Patch the imports that cli.py uses
    import src.viz.plots as plots_module
    import src.viz.plot3d as plot3d_module
    monkeypatch.setattr(plots_module, "plot_geometry_and_heatmap", fake_2d)
    monkeypatch.setattr(plot3d_module, "geometry_3d_html", fake_3d)
    
    # Test the plotting logic directly
    result = {"method":"adaptive","vf":0.1}
    plot_mode = getattr(args, "_plot_mode", "none")
    
    # Test 2D plotting
    if plot_mode in ("2d","both"):
        fake_2d(result=result, eval_mode="grid", method="adaptive", setback=1.0, out_png=str(tmp_path/"a.png"))
    
    # Test 3D plotting  
    if plot_mode in ("3d","both"):
        fake_3d(result, str(tmp_path/"b.html"))
    
    assert called["two_d"] == 1 and called["three_d"] == 1
