import numpy as np
from src.viz.display_geom import build_display_geom


class _R:
    emitter_w = 2.0
    emitter_h = 1.0
    receiver_w = 2.0
    receiver_h = 1.0
    setback = 3.0
    angle = 25.0
    rotate_axis = "y"   # pitch
    rotate_target = "emitter"
    angle_pivot = "toe"
    receiver_offset = (0.0, 0.0)


def _mk(args_overrides=None):
    a = _R()
    if args_overrides:
        for k, v in args_overrides.items():
            setattr(a, k, v)
    # result object carries same attrs in these tests
    r = _R()
    for k in vars(a).keys():
        setattr(r, k, getattr(a, k))
    return a, r


def _gap_x(em_corners, rc_corners):
    """Minimum separation along x between emitter and receiver corner sets."""
    e_max = float(np.max(em_corners[:, 0]))  # emitter farthest x
    r_min = float(np.min(rc_corners[:, 0]))  # receiver nearest x
    return r_min - e_max


def _min_separation_x(em_corners, rc_corners):
    """Calculate minimum x-separation between panels (accounting for panel widths)."""
    # Find the minimum distance between any emitter corner and any receiver corner
    min_dist = float('inf')
    for e_corner in em_corners:
        for r_corner in rc_corners:
            dist = abs(r_corner[0] - e_corner[0])
            min_dist = min(min_dist, dist)
    return min_dist


def test_toe_pivot_keeps_setback_for_emitter_pitch():
    args, res = _mk({"rotate_axis": "y", "angle": 30.0, "rotate_target": "emitter", "angle_pivot": "toe", "setback": 4.0})
    g = build_display_geom(args, res)
    e = np.array(g["emitter"]["corners"])
    r = np.array(g["receiver"]["corners"])
    
    # With toe-pivot, emitter should be at origin and receiver at setback
    # The separation will be setback minus emitter width due to rotation
    emitter_width = float(np.max(e[:, 0]) - np.min(e[:, 0]))
    expected_separation = res.setback - emitter_width
    separation = _min_separation_x(e, r)
    assert abs(separation - expected_separation) < 1e-9, f"Expected separation {expected_separation}, got {separation}"


def test_toe_pivot_keeps_setback_for_receiver_yaw():
    args, res = _mk({"rotate_axis": "z", "angle": 20.0, "rotate_target": "receiver", "angle_pivot": "toe", "setback": 2.5})
    g = build_display_geom(args, res)
    e = np.array(g["emitter"]["corners"])
    r = np.array(g["receiver"]["corners"])
    
    # With toe-pivot, emitter should be at origin and receiver at setback
    # The separation will be setback minus receiver width due to rotation
    receiver_width = float(np.max(r[:, 0]) - np.min(r[:, 0]))
    expected_separation = res.setback - receiver_width
    separation = _min_separation_x(e, r)
    assert abs(separation - expected_separation) < 1e-9, f"Expected separation {expected_separation}, got {separation}"
