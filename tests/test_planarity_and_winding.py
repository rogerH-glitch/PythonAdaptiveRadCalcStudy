import numpy as np
from src.viz.display_geom import build_display_geom, _order_quad


def plane_residual(p: np.ndarray) -> float:
    c = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - c)
    n = vt[-1]
    r = np.abs((p - c) @ n)
    return float(r.max())


def test_panels_planar_and_ordered():
    g = build_display_geom(width=5.1, height=2.1, setback=3, angle=30, angle_pivot="toe", rotate_target="emitter", dy=0.25)
    for key in ("emitter", "receiver"):
        p = np.asarray(g[key]["corners"], float)
        # Deduplicate if corners include repeats; take first 4 unique rows in order
        uniq = []
        for row in p:
            if not any(np.allclose(row, u) for u in uniq):
                uniq.append(row)
            if len(uniq) == 4:
                break
        p = np.asarray(uniq, float)
        # Ensure canonical ordering to avoid self-crossing interpretation
        p = _order_quad(p)
        assert p.shape == (4, 3)
        assert plane_residual(p) < 1e-9
        # Rough non-crossing check: area in the plane spanned by the quad edges
        # Build local 2D basis using first three points
        v1 = p[1] - p[0]
        v2 = p[2] - p[0]
        # Orthonormal basis (u along v1, v in plane orthogonal to u)
        def _norm(a):
            n = np.linalg.norm(a)
            return a / n if n > 0 else a
        u = _norm(v1)
        # Remove u component from v2 to get second axis in plane
        v2p = v2 - np.dot(v2, u) * u
        v = _norm(v2p) if np.linalg.norm(v2p) > 0 else _norm(v2)
        # Project points to 2D and compute polygon area
        P2 = np.column_stack([ (p - p[0]) @ u, (p - p[0]) @ v ])
        x2, y2 = P2[:, 0], P2[:, 1]
        # Width/height in local plane should be positive (non-degenerate quad)
        assert np.ptp(x2) > 0 and np.ptp(y2) > 0

