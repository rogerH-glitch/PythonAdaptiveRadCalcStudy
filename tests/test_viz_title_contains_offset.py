from src.viz.plots import plot_geometry_and_heatmap

def test_title_shows_offset(tmp_path):
    out = tmp_path/"g.png"
    result = {"We":5.0,"He":2.0,"Wr":5.0,"Hr":2.0,
              "emitter_center":(0.0,0.0,0.0),"receiver_center":(0.0,0.6,0.4),
              "vf":0.2,"y_peak":0.5,"z_peak":0.4}
    fig,axes = plot_geometry_and_heatmap(result=result, eval_mode="grid", method="adaptive",
                                         setback=3.0, out_png=str(out), return_fig=True)
    assert "Offset (dy,dz)=(0.600,0.400)" in fig._suptitle.get_text()
