# Architecture Overview

## Geometry pipeline
1. Input parse (new or legacy) → normalized config.
2. Helpers (`src/viz/geometry_utils.py`):
   - `order_quad`, `bbox_size_2d`, `translate_to_setback`, small math (rotz/roty).
3. Construction in `src/viz/display_geom.py`:
   - Build emitter/receiver quads in 3D → rotate (yaw/pitch) → translate via pivot/offset/setback.
   - Emit canonical `corners` and legacy-compatible structures (`emitter_xy`, `xz_limits`, `corners3d`, etc.).
4. Viz (`src/viz/plots.py`, `src/viz/plot3d.py`):
   - 2D panels and heatmap fallback.
   - 3D plot (to be aligned with 2D transforms per KNOWN_ISSUES).

## Deprecation strategy
- Legacy modules remain as shims with `DeprecationWarning`; removal in v1.2.
- Tests cover both new and legacy APIs to prevent regressions.
