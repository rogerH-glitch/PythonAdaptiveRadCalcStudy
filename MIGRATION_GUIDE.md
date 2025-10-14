# Migration Guide (v1.0.2)

## TL;DR
- Prefer: `build_display_geom(width=..., height=..., angle=..., angle_pivot=..., offsets=...)`.
- Legacy calls `build_display_geom(args, result)` still work; a shim preserves old keys/shapes.
- Deprecations emit `DeprecationWarning` and will be removed in v1.2.

## Input API
New (preferred):
```python
build_display_geom(
  width=W, height=H,
  angle=yaw_deg, pitch=pitch_deg,
  angle_pivot="toe"|"center",
  rotate_target="emitter"|"receiver",
  setback=...,
  # offsets via any alias:
  offset=(dx, dy)  # or x_offset=..., y_offset=..., origin=(dx,dy), xy0=(dx,dy)
)
```

Legacy (kept):
```python
build_display_geom(args, result)
# args/result tuple/dict/namespace accepted
# legacy keys like emitter_w/h, receiver_offset, emitter/receiver tuples supported
```

## Outputs
- corners: equals emitter corners (canonical order).
- emitter["xy"], emitter["xz"] are 4x2 arrays.
- corners3d["emitter"] is 8 points (4 repeated) for legacy parity.
- xy_limits, xz_limits are numpy arrays supporting .shape.

## Deprecations (removal in v1.2)
- Legacy modules shimmed: src.constants, src.orientation, src.paths, src.util.offsets.
- Legacy aliases/keys still accepted; warnings may be added on access in future releases.
