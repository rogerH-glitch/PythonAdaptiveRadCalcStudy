import numpy as np
from src.viz.display_geom import build_display_geom


def plane_residual(pts):
    pts = np.asarray(pts, float)
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    n = vt[-1]
    return float(np.max(np.abs((pts - c) @ n)))


def projected_area_any(pts):
    pts = np.asarray(pts, float)
    # Build local plane basis and compute area in that basis
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0]
    def _norm(a):
        n = np.linalg.norm(a)
        return a / n if n > 0 else a
    u = _norm(v1)
    v2p = v2 - np.dot(v2, u) * u
    v = _norm(v2p) if np.linalg.norm(v2p) > 0 else _norm(v2)
    P2 = np.column_stack([ (pts - pts[0]) @ u, (pts - pts[0]) @ v ])
    x2, y2 = P2[:, 0], P2[:, 1]
    return 0.5 * np.abs(np.sum(x2 * np.roll(y2, -1) - y2 * np.roll(x2, -1)))


def test_panels_planar_and_ordered():
    g = build_display_geom(
        width=5.1, height=2.1, setback=3.0,
        angle=30.0, pitch=0.0,
        angle_pivot="toe", rotate_target="emitter",
        dy=0.25, dz=0.0,
    )
    for key in ("emitter", "receiver"):
        p = g[key]["corners"]
        assert plane_residual(p) < 1e-9
        assert projected_area_any(p) > 0.0


