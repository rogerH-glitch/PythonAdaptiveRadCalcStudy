import time
from src.viz.display_geom import build_display_geom


class _R:
    emitter_w = 2.0
    emitter_h = 1.0
    receiver_w = 2.0
    receiver_h = 1.0
    setback = 5.0
    receiver_offset = (0.5, 0.25)
    angle = 35.0
    rotate_axis = "z"
    rotate_target = "emitter"
    angle_pivot = "toe"


def test_build_display_geom_is_reasonably_fast():
    # Generous threshold to avoid flakiness on CI/Windows — target < 3s in practice.
    args = _R()
    res = _R()
    t0 = time.perf_counter()
    for _ in range(400):
        _ = build_display_geom(args, res)
    dt = time.perf_counter() - t0
    assert dt < 5.0, f"display geom took {dt:.2f}s for 400 builds (should be < 5s)"

