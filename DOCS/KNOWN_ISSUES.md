# Known Issues (v1.0.1)

1) Elevation (X–Z) depiction under yaw
   - Expectation: yaw (rotate-axis=z) should not change the X–Z outline (it should remain a thin vertical rectangle/line; pitch would change X–Z).
   - Current: we draw a thin rectangle, but the geometry may not reflect toe/center translation exactly; also the apparent thickness can look too large.
   - Next steps:
     - In plotting-only rotation, clamp XZ thickness to a fixed pixel width in display space OR compute exact post-rotation toe-translation and use true thickness=0 (line).
     - Add test: yaw should not change the Z-span or the X position of the emitter's midline.

2) Heatmap sometimes missing
   - Cause: field capture is not attached for non-grid evals, or fallback sampler isn't invoked in some paths.
   - Next steps:
     - Ensure `rc_eval.grid_eval.evaluate_grid()` always `_tap_capture(Y,Z,F)` and `viz/plots.plot_geometry_and_heatmap()` always tries sampler when tap is empty.
     - Add test: `--eval-mode grid` with emitter-offset produces a non-empty heatmap (PNG must include colorbar and a peak star).

3) 3-D plot rotation (yaw/pitch) not visible on geometry
   - Current: title shows angle, but coordinates sent to Plotly are not rotated for the target + pivot.
   - Next steps:
     - Apply the same lightweight 2-D rotation used in viz/plots to the line-loop coordinates in `viz/plot3d.py` (for the selected axis and target).
     - Tests: (a) yaw 20° moves emitter's 3D corner X/Y; (b) pitch 15° changes Z correctly.
