from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np


def bbox_size_2d(points_xy: np.ndarray | Iterable[Tuple[float, float]]) -> Tuple[float, float]:
    P = np.asarray(points_xy, dtype=float)
    if P.size == 0:
        return 0.0, 0.0
    xmin, xmax = np.min(P[:, 0]), np.max(P[:, 0])
    ymin, ymax = np.min(P[:, 1]), np.max(P[:, 1])
    return float(xmax - xmin), float(ymax - ymin)


def order_quad(corners3d: np.ndarray) -> np.ndarray:
    pts = corners3d[:, :2]
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(ang)
    ordered = corners3d[order]
    pts_ord = ordered[:, :2]
    start = np.lexsort((pts_ord[:, 0], pts_ord[:, 1]))[0]
    return np.roll(ordered, -start, axis=0)


def span_with_epsilon(vmin: float, vmax: float, *, ref: float = 1.0, eps: float = 1e-9) -> tuple[float, float]:
    if vmax > vmin:
        return vmin, vmax
    pad = max(eps, 0.01 * float(ref))
    mid = 0.5 * (vmin + vmax)
    return mid - pad, mid + pad


def deg2rad(a: float) -> float:
    return float(np.deg2rad(a))


def rotz(deg: float) -> np.ndarray:
    c, s = np.cos(deg2rad(deg)), np.sin(deg2rad(deg))
    R = np.eye(3)
    R[0, 0], R[0, 1] = c, -s
    R[1, 0], R[1, 1] = s, c
    return R


def roty(deg: float) -> np.ndarray:
    c, s = np.cos(deg2rad(deg)), np.sin(deg2rad(deg))
    R = np.eye(3)
    R[0, 0], R[0, 2] = c, s
    R[2, 0], R[2, 2] = -s, c
    return R


def apply_about_pivot(points_3xN: np.ndarray, R: np.ndarray, pivot3: np.ndarray) -> np.ndarray:
    P = np.asarray(points_3xN)
    pivot = np.asarray(pivot3).reshape(3, 1)
    return R @ (P - pivot) + pivot


def toe_pivot_point(target: str, W: float, H: float, setback: float, base: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    x0 = 0.0 if target == "emitter" else float(setback)
    is_pure_yaw = abs(float(pitch)) <= 1e-12 and abs(float(yaw)) > 1e-12
    if is_pure_yaw:
        return base + np.array([x0, -W / 2.0, -H / 2.0], dtype=float)
    else:
        return base + np.array([x0, 0.0, -H / 2.0], dtype=float)


def translate_to_setback(
    corners3d: np.ndarray,
    *,
    mode: str,
    setback: float,
    treat_as: str,
    dx0: float,
) -> np.ndarray:
    xs = corners3d[:, 0]
    if mode == "center":
        cx = xs.mean()
        target = dx0 + float(setback)
        dx = (target - cx)
        return corners3d + np.array([dx, 0.0, 0.0])
    if treat_as == "emitter":
        target = dx0
        xmin = xs.min()
        dx = (target - xmin)
        return corners3d + np.array([dx, 0.0, 0.0])
    else:
        target = dx0 + float(setback)
        xmin, xmax = xs.min(), xs.max()
        if abs(target - xmin) <= abs(target - xmax):
            dx = (target - xmin)
        else:
            dx = (target - xmax)
        return corners3d + np.array([dx, 0.0, 0.0])


