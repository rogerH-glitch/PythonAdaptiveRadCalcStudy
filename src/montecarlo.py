"""
Monte Carlo view factor calculations.

This module implements view factor calculations using Monte Carlo
ray tracing with statistical sampling and uncertainty estimation.
"""

from __future__ import annotations
import math, time
import numpy as np
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
    cos = setback / r                      # parallel planes
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
    """
    # Input validation
    if em_w <= 0 or em_h <= 0 or rc_w <= 0 or rc_h <= 0 or setback <= 0:
        return {
            "vf_mean": 0.0,
            "vf_ci95": 0.0,
            "samples": 0,
            "status": STATUS_FAILED
        }
    
    if samples <= 0 or target_rel_ci <= 0 or max_iters <= 0:
        return {
            "vf_mean": 0.0,
            "vf_ci95": 0.0,
            "samples": 0,
            "status": STATUS_FAILED
        }
    
    start = time.time()
    rng = np.random.default_rng(seed)

    total_samples = 0
    # batch in chunks for early stopping
    batch = max(1000, min(samples, 50000))  # Smaller batches for better testing
    acc_vals = []
    iteration = 0

    while total_samples < samples and (time.time() - start) < time_limit_s and iteration < max_iters:
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

        if rel_ci <= target_rel_ci:
            return {
                "vf_mean": float(m),
                "vf_ci95": float(ci),
                "samples": int(total_samples),
                "status": STATUS_CONVERGED
            }

    # Determine why we stopped
    if (time.time() - start) >= time_limit_s:
        status = STATUS_REACHED_LIMITS
    elif iteration >= max_iters:
        status = STATUS_REACHED_LIMITS
    else:
        status = STATUS_REACHED_LIMITS  # Hit sample limit
    
    m = float(np.mean(acc_vals)) if acc_vals else 0.0
    s = float(np.std(acc_vals, ddof=1)) if len(acc_vals) > 1 else 0.0
    ci = 1.96 * (s / math.sqrt(max(1, len(acc_vals))))
    return {
        "vf_mean": m,
        "vf_ci95": ci,
        "samples": int(total_samples),
        "status": status
    }