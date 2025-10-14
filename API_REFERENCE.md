# API Reference

## build_display_geom

Signature (new API):
```python
build_display_geom(
  *,
  width: float,
  height: float,
  yaw_deg: float = 0.0,
  pitch_deg: float = 0.0,
  rotate_target: str = "emitter",   # "emitter"|"receiver"|"both"
  angle_pivot: str = "toe",         # "toe"|"center"
  setback: float = 2.0,
  # offsets (any alias): offset=(dx,dy), x_offset=..., y_offset=..., origin=(dx,dy), xy0=(dx,dy)
  **kwargs,
) -> dict
```

Legacy API:
```python
build_display_geom(args_like, result_like) -> dict
```

### Returns (keys)
- emitter: { corners:(4,3), xy:(4,2), xz:(4,2), x_span, y_span, z_span }
- receiver: { corners:(4,3), xy:(4,2), xz:(4,2), x_span, y_span, z_span }
- xy: { emitter: [(x,y),(x,y)], receiver: [(x,y),(x,y)] }
- xz: { emitter: {x,z0,z1,zmin,zmax}, receiver: {...} }
- corners3d: { emitter:(8,3 list), receiver:(8,3 list) }
- bounds: { xy:(xmin,xmax,ymin,ymax), xz:(xmin,xmax,zmin,zmax) }
- xy_limits: np.ndarray shape (4,)
- xz_limits: np.ndarray shape (4,)
- corners: np.ndarray (4,3) — equals emitter corners (canonical)

### Helpers
- See `src/viz/geometry_utils.py`.
