# Known Issues / Next Steps (v1.1.0)

## v1.0.2 Resolved Issues
- Toe/center pivot logic unified; setback preserved; offsets handled; canonical corner ordering.
- Degenerate plot limits: epsilon padding avoids zero-span.
- Receiver toe-pivot with x/y offsets: min/max edge selection keeps correct setback.
- Dual API geometry refactor; legacy keys & shapes supported via shims.
- All tests pass locally: 189 passed, 1 skipped.

## Current Issues (v1.1.0)

1) **Heatmap marker misalignment**
   - The adaptive peak marker (×) on heatmaps doesn't align with the actual adaptive peak coordinates in image space.
   - See `docs/TODO_v1.1.0.md` for detailed analysis and acceptance criteria.

2) **3D panels appear as bow-ties at high yaw angles**
   - At yaw angles around 89°, 3D panels render as X-shaped bow-ties instead of rectangles.
   - See `docs/TODO_v1.1.0.md` for detailed analysis and acceptance criteria.

3) **XZ legend overlaps with plot area**
   - The XZ elevation legend appears inside the plot area instead of below it.
   - See `docs/TODO_v1.1.0.md` for detailed analysis and acceptance criteria.

4) **Console output is scattered/duplicated**
   - Console output has multiple scattered print statements instead of a single structured summary.
   - See `docs/TODO_v1.1.0.md` for detailed analysis and acceptance criteria.

## Legacy Issues (v1.0.2)

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
