# TODO v1.1.0 - Known Issues & Fixes

## Known Issues

### 1. Heatmap marker misalignment
**Description**: The adaptive peak marker (×) on heatmaps doesn't align with the actual adaptive peak coordinates in image space.

**Acceptance Checks**:
- [ ] Adaptive marker lands exactly on the same coordinate as adaptive `(y,z)` in image space
- [ ] No magic shifts or coordinate transformations that aren't mathematically justified
- [ ] Works with both `imshow` extent and `transform` used by marker placement

**Suspected Causes**:
- [ ] Inconsistent `imshow` extent vs marker coordinate system
- [ ] Transpose or sign flip in coordinate mapping
- [ ] Missing pixel-center offset that's mathematically justified
- [ ] Different coordinate systems between field data and marker placement

### 2. 3D panels appear as bow-ties at high yaw angles
**Description**: At yaw angles around 89°, 3D panels render as X-shaped bow-ties instead of rectangles.

**Acceptance Checks**:
- [ ] Corners are coplanar (plane fit residual < 1e-6)
- [ ] Two triangles cover the quad with consistent winding (no bow-tie)
- [ ] Plotly trace has exactly 2 triangles per quad
- [ ] Edges close the loop properly

**Suspected Causes**:
- [ ] `_order_quad` doesn't return corners in proper perimeter order
- [ ] Triangle indices `(0,1,2)` and `(0,2,3)` are incorrect
- [ ] Missing planarity check for rectangles
- [ ] Corner ordering doesn't account for rotation

### 3. XZ legend overlaps with plot area
**Description**: The XZ elevation legend appears inside the plot area instead of below it.

**Acceptance Checks**:
- [ ] Legend bbox y-coordinate < 0 in axes coordinates
- [ ] Legend appears below the X-axis
- [ ] No overlap with plot content

**Suspected Causes**:
- [ ] `bbox_to_anchor=(0, -0.18)` not properly applied
- [ ] `fig.set_constrained_layout(True)` not called
- [ ] Legend positioning logic incorrect

### 4. Console output is scattered/duplicated
**Description**: Console output has multiple scattered print statements instead of a single structured summary.

**Acceptance Checks**:
- [ ] Single summary block with `[eval]`, `[geom]`, optional `[grid]`, `[peak]`, `[status]`, optional `[artifacts]`
- [ ] `[diag]` line preserved (useful signal)
- [ ] 3D debug prints (`[p0-3d]`) only shown with `--debug-plots`
- [ ] No duplicate or scattered print statements

**Suspected Causes**:
- [ ] Multiple print statements throughout code instead of centralized summary
- [ ] Debug prints not properly gated behind `--debug-plots`
- [ ] Missing structured summary block

## Implementation Plan

1. **Add failing tests first** (xfail) to capture current behavior
2. **Add debug instrumentation** to understand root causes
3. **Fix incrementally** - one issue per patch
4. **Verify with user scenarios** after each fix

## Test Scenarios

### S1: Standard case
```bash
python main.py --emitter 5.1 2.1 --receiver 5.1 2.1 --setback 3 --receiver-offset 0.25 0 --angle 30 --eval-mode grid --plot-both --heatmap-n 41
```

### S2: High yaw case
```bash
python main.py --emitter 5.1 2.1 --receiver 1.1 2.1 --setback 1 --receiver-offset 0.25 0 --angle 89 --eval-mode grid --plot-both --heatmap-n 41
```

## Success Criteria

- [ ] All 4 issues resolved
- [ ] Tests pass (no xfail)
- [ ] User scenarios produce correct visual output
- [ ] No performance regressions
- [ ] Clean console output
