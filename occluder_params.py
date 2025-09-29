# occluder_params.py
import numpy as np
from typing import List, Dict, Tuple, Any

from occluder_projection import (
    make_rect_occluder_from_params,
    make_circular_occluder,
)

def build_occluders_from_specs(
    emitter_origin: np.ndarray,
    emitter_normal: np.ndarray,
    specs: List[Dict[str, Any]],
) -> List[Tuple[str, Any]]:
    """
    Convert a list of human-friendly occluder 'specs' into the occluder_list
    expected by viewfactor_1ai_obstructed_general.viewfactor_1ai_rects_with_occluders().

    Each spec is a dict with keys depending on 'shape':

    Common keys:
      shape: "rect" | "disk" (circle) | "polygon" (optional; for polygon, pass 'points3d')
      setback: float (m)  distance from emitter along emitter_normal (ignored for 'polygon' if points3d given)
      center_offset_xy: (du, dv) offset in emitter’s local in-plane axes (default (0,0))
      orientation: "parallel" | "perpendicular" | "angled"   (rect only)
      angle_deg: float (deg)   (rect only, used when orientation == "angled")
      thickness: float (m)     (rect only; 0 = sheet, >0 = slab → two faces)

    Rectangle-specific:
      width: float (m)
      height: float (m)

    Circle-specific:
      radius: float (m)
      n_sides: int (default 48) polygon approximation

    Polygon-specific:
      points3d: np.ndarray (Nx3 closed) — use this only if you already have a 3D polygon.

    Returns:
      occluder_list: List of ("rect", (o,u,v)) and/or ("poly", np.ndarray Nx3 closed)
    """
    occluders = []

    for i, spec in enumerate(specs):
        shape = spec.get("shape", "rect").lower()

        if shape == "rect":
            width   = float(spec["width"])
            height  = float(spec["height"])
            setback = float(spec.get("setback", 0.0))
            orient  = spec.get("orientation", "parallel").lower()
            angle   = float(spec.get("angle_deg", 0.0))
            thick   = float(spec.get("thickness", 0.0))
            du, dv  = spec.get("center_offset_xy", (0.0, 0.0))

            rects = make_rect_occluder_from_params(
                emitter_origin=emitter_origin,
                emitter_normal=emitter_normal,
                setback=setback,
                width=width, height=height,
                orientation=orient,
                angle_deg=angle,
                thickness=thick,
                center_offset_xy=(du, dv),
            )
            # one face (sheet) or two faces (slab)
            occluders += [("rect", r) for r in rects]

        elif shape == "disk":
            radius  = float(spec["radius"])
            setback = float(spec.get("setback", 0.0))
            du, dv  = spec.get("center_offset_xy", (0.0, 0.0))
            n_sides = int(spec.get("n_sides", 48))

            # place disk center at emitter_origin + setback*n + offsets in plane
            from occluder_projection import _orthonormal_basis  # local axes
            t1, t2, n = _orthonormal_basis(emitter_normal)
            center = emitter_origin + setback*n + du*t1 + dv*t2

            poly3d = make_circular_occluder(center=center, normal=emitter_normal,
                                            radius=radius, n_sides=n_sides)
            occluders.append(("poly", poly3d))

        elif shape == "polygon":
            pts = spec["points3d"]
            occluders.append(("poly", pts))

        else:
            raise ValueError(f"Unknown occluder shape '{shape}' in spec #{i+1}")

    return occluders
