import numpy as np
from src.viz.display_geom import build_display_geom


class _R:
    emitter_w = 1.0
    emitter_h = 1.0
    receiver_w = 1.0
    receiver_h = 1.0
    setback = 3.0
    angle = 0.0
    rotate_axis = "z"
    rotate_target = "emitter"
    angle_pivot = "center"
    receiver_offset = (0.0, 0.0)


def _mk():
    args = _R()
    res = _R()
    return args, res


def test_rotating_emitter_leaves_receiver_unchanged():
    args, res = _mk()
    g0 = build_display_geom(args, res)
    # rotate emitter 20 deg yaw
    args.angle = 20.0
    args.rotate_axis = "z"
    args.rotate_target = "emitter"
    g1 = build_display_geom(args, res)

    e0 = np.array(g0["emitter"]["corners"])
    e1 = np.array(g1["emitter"]["corners"])
    r0 = np.array(g0["receiver"]["corners"])
    r1 = np.array(g1["receiver"]["corners"])

    assert not np.allclose(e0, e1), "Emitter should change when rotated"
    assert np.allclose(r0, r1), "Receiver should remain unchanged when emitter rotates"


def test_rotating_receiver_leaves_emitter_unchanged():
    # Start with receiver as target to avoid position swapping
    args, res = _mk()
    args.rotate_target = "receiver"  # Set receiver as target from start
    g0 = build_display_geom(args, res)
    
    # rotate receiver 15 deg pitch
    args.angle = 15.0
    args.rotate_axis = "y"
    args.rotate_target = "receiver"  # Keep receiver as target
    g1 = build_display_geom(args, res)

    e0 = np.array(g0["emitter"]["corners"])
    e1 = np.array(g1["emitter"]["corners"])
    r0 = np.array(g0["receiver"]["corners"])
    r1 = np.array(g1["receiver"]["corners"])

    assert np.allclose(e0, e1), "Emitter should remain unchanged when receiver rotates"
    assert not np.allclose(r0, r1), "Receiver should change when rotated"
