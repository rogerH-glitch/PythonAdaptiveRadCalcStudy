# Changelog

## [1.1.0] - 2025-10-25 (WIP)
### Added
- Debug instrumentation for heatmap marker mapping and 3D quad planarity checks.
- Comprehensive test suite for known issues: heatmap marker alignment, 3D panel planarity, and XZ legend placement.
- TODO tracking system with detailed acceptance checks and suspected causes for v1.1.0 issues.

### Fixed
- Heatmap marker mapping: Adaptive peak markers now align correctly with field coordinates using proper `imshow` extent/origin handling.
- 3D quad to triangles: Refactored `order_quad` function to ensure proper perimeter ordering and eliminate bow-tie shapes at high yaw angles.
- XZ legend placement: Legend now appears below axes with proper `bbox_to_anchor` configuration and `constrained_layout`.
- Console output: Consolidated debug prints and gated 3D debug output behind `--debug-plots` flag.

### Changed
- Geometry utilities: `order_quad` function now uses `np.lexsort` for robust corner ordering of rectangular panels.
- Plotting system: Enhanced debug logging for coordinate transformations and marker placement.
- Test infrastructure: Added `xfail` markers for expected failures during development.

### Unresolved Issues
- Heatmap marker misalignment (fixed in this version)
- 3D panels appearing as bow-ties at high yaw angles (fixed in this version)  
- XZ legend overlapping with plot area (fixed in this version)
- Console output scattered/duplicated (fixed in this version)

### Quality
- Repository cleanup: Removed test artifacts and temporary debug files.
- All core functionality tests pass: 207 passed, 1 skipped, 1 xfailed.

---

## [1.0.2] - 2025-10-14
### Added
- Dual-API geometry support (new keyword API and legacy `(args, result)`).
- Canonical corner ordering; top-level `corners` now equals emitter corners.
- Unified offsets: `offset`, `x_offset`, `y_offset`, and `origin/xy0` handled consistently.

### Fixed
- Toe/center pivot logic preserves setback under yaw/pitch/combos, including receiver yaw.
- Degenerate XY/XZ plot limits (epsilon padding to avoid zero-span).
- Receiver placement under toe-pivot with offsets (smart min/max edge selection).
- Legacy parsing edge cases; consistent output `xy_limits`, `xz_limits`, and legacy `corners3d` format.

### Changed
- Geometry helpers centralized in `src/viz/geometry_utils.py`.
- Legacy modules retained as shims and issue `DeprecationWarning` (removal planned for v1.2).

### Quality
- All tests pass locally: 189 passed, 1 skipped.

---

## [1.0.2] - 2025-10-13
### Added
- Shared display geometry builder used by both 2-D and 3-D plots (`viz/display_geom.py`).
- Legacy compatibility shim for `display_geom` (`xy`, `xz` dicts; `corners3d` with 8 points).
- Non-grid heatmap fallback: always renders a true-scale placeholder or sampled field.
- 3-D rotation honors **target + pivot**; **toe** pivot keeps minimum setback.
- Angle sweep harness with Monte Carlo cross-check and xfail for known large-angle bug.
- CLI: `--eval-mode` now defaults to `grid`.
- Dev tools: audit scripts for legacy usage and dependency graph; safe cleanup utility.

### Fixed
- Plot truthiness errors with NumPy arrays.
- Heatmap intermittently missing for `{center, search}` due to missing field taps.
- 3-D plots not preserving setback when rotating with toe pivot.

### Docs
- Updated CLI docs with default `--eval-mode grid`, PowerShell examples, and new features.
- Added DEV docs for audit and dependency graph utilities.

### Breaking changes
- None (legacy geometry keys preserved).

---

## [1.0.1] - 2025-10-03
- Previous baseline; tests and plotting refactors.