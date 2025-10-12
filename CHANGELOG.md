# Changelog

All notable changes to this project will be documented in this file.

## v1.0.1 — 2025-10-12
### Added
- 2-D combined geometry+heatmap figure: true-scale heatmap, peak star/label, offset in title.
- Legends show panel dimensions; anchored to avoid overlap.
- New flags: `--plot-3d` (3-D HTML only) and `--plot-both` (2-D PNG + 3-D HTML).
- CSV writer appends rows and adds `timestamp` (first column).

### Fixed
- 2-D geometry now always draws both emitter and receiver in Plan/Elevation; Plan limits auto-fit rotated emitters.

### Internal
- Consolidated plotting modules under `src/viz/*`; removed legacy duplicates.
- Introduced `viz/heatmap_core.py` for consistent heatmap styling.

### Known issues
- Elevation X–Z under yaw should remain a line; refine plot-only rotation/translation.
- Heatmap may not appear if field tap/sampler is skipped in certain paths.
- 3-D plot does not yet rotate coordinates; only the title shows angle.
(See docs/KNOWN_ISSUES.md)
