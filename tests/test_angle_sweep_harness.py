import math
from src.validation.angle_sweep import angle_sweep, SweepConfig


def _is_monotone_nonincreasing(seq, tol=1e-9):
    return all(seq[i+1] <= seq[i] + tol for i in range(len(seq)-1))


def test_adaptive_vf_decreases_from_0_to_60_deg():
    cfg = SweepConfig(setback=5.0, rotate_axis="z", rotate_target="emitter", pivot="toe")
    angles = [0, 10, 20, 30, 45, 60]
    sweep = angle_sweep(angles, method="adaptive", cfg=cfg)
    vfs = [v for _, v in sweep]
    # Expect a general drop as yaw increases in this moderate range
    assert _is_monotone_nonincreasing(vfs, tol=1e-6), f"Expected nonincreasing VF 0→60°, got {vfs}"
    # strictly less at 60 vs 0 (not just equal everywhere)
    assert vfs[-1] < vfs[0] - 1e-8


def test_montecarlo_matches_adaptive_within_loose_tolerance():
    cfg = SweepConfig(setback=5.0, rotate_axis="y", rotate_target="emitter", pivot="toe")  # pitch case
    angles = [0, 15, 30, 45]
    sweep_ad = angle_sweep(angles, method="adaptive", cfg=cfg)
    sweep_mc = angle_sweep(angles, method="montecarlo", cfg=cfg)
    for (_, v_ad), (_, v_mc) in zip(sweep_ad, sweep_mc):
        # Loose tolerance; we just want a sanity guard across methods
        assert abs(v_ad - v_mc) < 5e-3, f"Adaptive vs MC diverged at angle: {v_ad} vs {v_mc}"


def test_adaptive_not_constant_at_very_large_angles():
    cfg = SweepConfig(setback=5.0, rotate_axis="z", rotate_target="emitter", pivot="toe")
    angles = [75, 80, 85]
    sweep = angle_sweep(angles, method="adaptive", cfg=cfg)
    vfs = [v for _, v in sweep]
    # Expect some drop (not a flat series)
    assert any(abs(vfs[i+1] - vfs[i]) > 1e-5 for i in range(len(vfs)-1)), f"VF appears ~constant at {angles}: {vfs}"
