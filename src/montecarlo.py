"""
Monte Carlo view factor calculations.

This module implements view factor calculations using Monte Carlo
ray tracing with statistical sampling and uncertainty estimation.
"""

from __future__ import annotations
import math, time
import numpy as np
from typing import Tuple, Dict
try:
    # Orientation frame (optional). If unavailable, pointwise MC will raise.
    from .orientation import make_emitter_frame
except Exception:  # pragma: no cover - optional import guard
    make_emitter_frame = None  # type: ignore
from .constants import EPS, STATUS_CONVERGED, STATUS_REACHED_LIMITS, STATUS_FAILED


def _vf_point_mc_center(em_w: float, em_h: float, rc_w: float, rc_h: float, setback: float,
                        samples: int, seed: int) -> tuple[float, float, dict]:
    """
    MC estimate of local (point) view factor at the receiver center.
    Samples uniformly on the emitter; integrates kernel over emitter.
    Returns (vf_mean, ci95, extras).
    """
    rng = np.random.default_rng(seed)
    # receiver center in emitter coordinates (centres aligned)
    rx, ry = 0.0, 0.0

    xs = rng.uniform(-em_w/2.0,  em_w/2.0,  samples)
    ys = rng.uniform(-em_h/2.0,  em_h/2.0,  samples)

    dx = xs - rx
    dy = ys - ry
    r2 = dx*dx + dy*dy + setback*setback
    # guards
    r2 = np.maximum(r2, EPS)
    r  = np.sqrt(r2)
    cos = np.maximum(0.0, np.minimum(1.0, setback / r))  # parallel planes, clamped
    K   = (cos*cos) / (math.pi * r2)       # == s^2 / (π r^4)

    # Integral over emitter = A_em * mean(K) for uniform sampling
    A_em = em_w * em_h
    meanK = float(np.mean(K))
    stdK  = float(np.std(K, ddof=1)) if samples > 1 else 0.0

    vf_mean = A_em * meanK
    # Propagate CI from mean(K): CI(meanK) * A_em
    ci95 = 1.96 * (stdK / math.sqrt(samples)) * A_em if samples > 1 else 0.0

    # clamp
    vf_mean = max(0.0, min(1.0, vf_mean))

    extras = {
        "kernel_min": float(np.min(K)),
        "kernel_max": float(np.max(K)),
        "A_em": A_em,
        "meanK": meanK
    }
    return vf_mean, ci95, extras


def vf_montecarlo(em_w, em_h, rc_w, rc_h, setback,
                  samples=200000, target_rel_ci=0.02, max_iters=50, seed=42, time_limit_s=60,
                  eps: float = EPS, rc_mode: str = "center", **_):
    """
    Monte Carlo local-peak estimator. Default rc_mode='center'.
    
    Args:
        em_w, em_h: Emitter dimensions (m)
        rc_w, rc_h: Receiver dimensions (m)
        setback: Setback distance (m)
        samples: Maximum number of samples
        target_rel_ci: Target relative confidence interval
        max_iters: Maximum number of iterations
        seed: Random seed
        time_limit_s: Time limit in seconds
        eps: Small value for numerical stability
        rc_mode: Receiver mode (only "center" supported)
        
    Returns:
        Dictionary with vf_mean, vf_ci95, samples, status, iterations
    """
    # Enhanced input validation
    if not all(x > 0 for x in [em_w, em_h, rc_w, rc_h, setback]):
        return {
            "vf_mean": 0.0,
            "vf_ci95": 0.0,
            "samples": 0,
            "status": STATUS_FAILED,
            "iterations": 0
        }
    
    if not all(x > 0 for x in [samples, target_rel_ci, max_iters]):
        return {
            "vf_mean": 0.0,
            "vf_ci95": 0.0,
            "samples": 0,
            "status": STATUS_FAILED,
            "iterations": 0
        }
    
    # Clamp parameters to reasonable ranges
    samples = min(max(samples, 1000), 10000000)  # 1K to 10M samples
    target_rel_ci = max(0.001, min(target_rel_ci, 0.1))  # 0.1% to 10%
    max_iters = min(max(max_iters, 1), 1000)  # 1 to 1000 iterations
    time_limit_s = max(0.1, min(time_limit_s, 3600))  # 0.1s to 1 hour
    
    start = time.time()
    rng = np.random.default_rng(seed)

    total_samples = 0
    # batch in chunks for early stopping
    batch = max(1000, min(samples, 50000))  # Smaller batches for better testing
    acc_vals = []
    iteration = 0
    
    # Stagnation detection
    stagnation_count = 0
    last_rel_ci = float('inf')
    stagnation_threshold = 0.1 * target_rel_ci

    while total_samples < samples and iteration < max_iters:
        # Check time limit
        elapsed = time.time() - start
        if elapsed >= time_limit_s:
            break
            
        n = min(batch, samples - total_samples)
        vf_mean, ci95, extras = _vf_point_mc_center(em_w, em_h, rc_w, rc_h, setback, n, rng.integers(0, 2**31-1))
        acc_vals.append(vf_mean)
        total_samples += n
        iteration += 1

        # combine batches by running mean & std:
        m = np.mean(acc_vals)
        s = np.std(acc_vals, ddof=1) if len(acc_vals) > 1 else 0.0
        # conservative CI over batch means
        ci = 1.96 * (s / math.sqrt(len(acc_vals)))
        rel_ci = (ci / m) if m > 0 else float("inf")
        
        # Check for stagnation
        if rel_ci < last_rel_ci:
            improvement = last_rel_ci - rel_ci
            if improvement < stagnation_threshold:
                stagnation_count += 1
                if stagnation_count >= 5:  # 5 consecutive steps with minimal improvement
                    break
            else:
                stagnation_count = 0
        else:
            stagnation_count += 1
            if stagnation_count >= 5:
                break
        
        last_rel_ci = rel_ci

        if rel_ci <= target_rel_ci:
            return {
                "vf_mean": float(m),
                "vf_ci95": float(ci),
                "samples": int(total_samples),
                "status": STATUS_CONVERGED,
                "iterations": iteration
            }

    # Determine why we stopped
    elapsed = time.time() - start
    if elapsed >= time_limit_s:
        status = STATUS_REACHED_LIMITS
    elif iteration >= max_iters:
        status = STATUS_REACHED_LIMITS
    elif stagnation_count >= 5:
        status = STATUS_REACHED_LIMITS
    else:
        status = STATUS_REACHED_LIMITS  # Hit sample limit
    
    m = float(np.mean(acc_vals)) if acc_vals else 0.0
    s = float(np.std(acc_vals, ddof=1)) if len(acc_vals) > 1 else 0.0
    ci = 1.96 * (s / math.sqrt(max(1, len(acc_vals))))
    
    # Clamp to physical bounds
    m = max(0.0, min(m, 1.0))
    
    return {
        "vf_mean": m,
        "vf_ci95": ci,
        "samples": int(total_samples),
        "status": status,
        "iterations": iteration
    }


def _mc_contrib(u: float, v: float, yR: float, zR: float, frame) -> float:
    """
    Oriented differential VF integrand at emitter local (u,v) to receiver point (S,yR,zR):
        dF = max(0, n1·r̂) * max(0, n2·(-r̂)) / (π r²)  du dv
    with p1 = C + j*u + k*v, p2 = (S, yR, zR).
    """
    C, j, k, n1, n2, W, H, S = frame.C, frame.j, frame.k, frame.n1, frame.n2, frame.W, frame.H, frame.S
    p1 = C + j * u + k * v
    p2 = np.array([S, yR, zR], dtype=float)
    r = p2 - p1
    r2 = float(r.dot(r))
    if r2 <= 0.0:
        return 0.0
    rinv = 1.0 / math.sqrt(r2)
    rhat = r * rinv
    cos1 = float(n1.dot(rhat))
    cos2 = float(n2.dot(-rhat))
    if cos1 <= 0.0 or cos2 <= 0.0:
        return 0.0
    return (cos1 * cos2) / (math.pi * r2)


def vf_point_montecarlo(
    receiver_yz: Tuple[float, float],
    geom: Dict,
    samples: int = 200_000,
    seed: int | None = None,
    target_rel_ci: float | None = None,
    max_iters: int = 10,
) -> Tuple[float, Dict]:
    """
    Orientation-aware Monte Carlo estimate of local view factor at a single
    receiver point (yR, zR), integrating over the emitter rectangle (u,v).

    Required in `geom`:
      - emitter_width, emitter_height, setback
      - rotate_axis in {'z','y'}, angle (deg), angle_pivot in {'toe','center'}
      - dy, dz (receiver - emitter centre offsets in y,z)
    """
    if make_emitter_frame is None:  # pragma: no cover - defensive
        raise RuntimeError("orientation.make_emitter_frame not available; add src/orientation.py first.")

    rng = np.random.default_rng(seed)
    yR, zR = receiver_yz
    W = float(geom["emitter_width"])  # meters
    H = float(geom["emitter_height"])  # meters
    S = float(geom["setback"])  # meters
    axis = geom.get("rotate_axis", "z")
    angle = float(geom.get("angle", 0.0))
    pivot = geom.get("angle_pivot", "toe")
    dy = float(geom.get("dy", 0.0))
    dz = float(geom.get("dz", 0.0))

    frame = make_emitter_frame(W, H, S, axis=axis, angle_deg=angle, pivot=pivot, dy=dy, dz=dz)

    area = W * H
    total = 0.0
    total2 = 0.0
    iters = 0

    while True:
        U = rng.uniform(-W / 2, W / 2, size=samples)
        V = rng.uniform(-H / 2, H / 2, size=samples)
        vals = np.empty(samples, dtype=float)
        for i, (u, v) in enumerate(zip(U, V)):
            vals[i] = _mc_contrib(u, v, yR, zR, frame)

        s = float(vals.sum())
        s2 = float((vals ** 2).sum())
        total += s
        total2 += s2
        iters += 1

        N = samples * iters
        mean_cell = total / N  # mean integrand
        var_cell = max(total2 / N - (total / N) ** 2, 0.0)
        mean = mean_cell * area
        std = math.sqrt(var_cell) * area
        ci95 = 1.96 * std / math.sqrt(N) if N > 0 else float("inf")

        if not target_rel_ci or iters >= max_iters:
            return mean, {"samples": samples, "iters": iters, "ci95": ci95}
        if mean != 0.0 and (ci95 / abs(mean)) <= target_rel_ci:
            return mean, {"samples": samples, "iters": iters, "ci95": ci95}