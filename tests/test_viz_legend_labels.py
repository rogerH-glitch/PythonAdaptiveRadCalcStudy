from src.viz.plots import plot_geometry_and_heatmap

def test_legend_contains_dimensions(tmp_path):
    out = tmp_path/"g.png"
    res = {"We":5.1,"He":2.1,"Wr":5.1,"Hr":2.1,
           "emitter_center":(0.0,0.0,0.0), "receiver_center":(0.0,0.0,0.0),
           "vf":0.1,"y_peak":0.0,"z_peak":0.0}
    fig,(ax_xy,ax_xz,_) = plot_geometry_and_heatmap(result=res, eval_mode="grid", method="adaptive",
                                                    setback=3.0, out_png=str(out), return_fig=True)
    lx = [txt.get_text() for txt in ax_xy.get_legend().get_texts()]
    assert any("Emitter" in s and "×" in s for s in lx)
    assert any("Receiver" in s and "×" in s for s in lx)
