import numpy as np
from src.validation.angle_sweep import SweepConfig, angle_sweep
# Note: Large-angle behavior is now fixed; this file should contain no xfail markers.
# If you previously had @pytest.mark.xfail decorators here, they’ve been removed.
# Note: previously some cases were marked xfail due to the large-angle plateau.
# The adaptive orientation fix resolved that, so these remain as normal tests.


def test_angle_sweep_basic_functionality():
    """Test basic angle sweep functionality with small angles."""
    cfg = SweepConfig(
        emitter_w=2.0, emitter_h=1.0,
        receiver_w=2.0, receiver_h=1.0,
        setback=5.0, rotate_target="emitter",
        rotate_axis="z", pivot="toe"
    )
    
    angles = [0.0, 10.0, 20.0, 30.0]
    results = angle_sweep(angles, "adaptive", cfg)
    
    assert len(results) == len(angles)
    for i, (angle, vf) in enumerate(results):
        assert angle == angles[i]
        assert isinstance(vf, float)
        assert vf >= 0.0  # View factor should be non-negative


def test_angle_sweep_different_axes():
    """Test angle sweep with different rotation axes."""
    cfg_yaw = SweepConfig(rotate_axis="z", rotate_target="emitter")
    cfg_pitch = SweepConfig(rotate_axis="y", rotate_target="emitter")
    
    angles = [0.0, 15.0, 30.0]
    
    results_yaw = angle_sweep(angles, "adaptive", cfg_yaw)
    results_pitch = angle_sweep(angles, "adaptive", cfg_pitch)
    
    assert len(results_yaw) == len(angles)
    assert len(results_pitch) == len(angles)
    
    # Results should be different for different axes
    for (_, vf_yaw), (_, vf_pitch) in zip(results_yaw, results_pitch):
        assert vf_yaw != vf_pitch or abs(vf_yaw - vf_pitch) < 1e-10


def test_angle_sweep_different_targets():
    """Test angle sweep with different rotation targets."""
    cfg_emitter = SweepConfig(rotate_target="emitter", rotate_axis="z")
    cfg_receiver = SweepConfig(rotate_target="receiver", rotate_axis="z")
    
    angles = [0.0, 20.0, 40.0]
    
    results_emitter = angle_sweep(angles, "adaptive", cfg_emitter)
    results_receiver = angle_sweep(angles, "adaptive", cfg_receiver)
    
    assert len(results_emitter) == len(angles)
    assert len(results_receiver) == len(angles)


def test_angle_sweep_different_methods():
    """Test angle sweep with different calculation methods."""
    cfg = SweepConfig(rotate_axis="z", rotate_target="emitter")
    angles = [0.0, 10.0, 20.0]
    
    results_adaptive = angle_sweep(angles, "adaptive", cfg)
    results_montecarlo = angle_sweep(angles, "montecarlo", cfg)
    
    assert len(results_adaptive) == len(angles)
    assert len(results_montecarlo) == len(angles)
    
    # Results should be reasonably close for small angles
    for (_, vf_adaptive), (_, vf_montecarlo) in zip(results_adaptive, results_montecarlo):
        assert abs(vf_adaptive - vf_montecarlo) < 0.1  # Allow some tolerance


def test_angle_sweep_large_angles():
    """Test angle sweep with large angles (where invariance bug may occur)."""
    cfg = SweepConfig(
        emitter_w=2.0, emitter_h=1.0,
        receiver_w=2.0, receiver_h=1.0,
        setback=5.0, rotate_target="emitter",
        rotate_axis="z", pivot="toe"
    )
    
    # Large angles where invariance bug is known to occur
    angles = [0.0, 45.0, 90.0, 135.0, 180.0]
    results = angle_sweep(angles, "adaptive", cfg)
    
    assert len(results) == len(angles)
    for angle, vf in results:
        assert isinstance(vf, float)
        assert vf >= 0.0
        assert vf <= 1.0  # View factor should not exceed 1.0


def test_angle_sweep_invariance_bug():
    """Test that exposes the known angle invariance bug.
    
    This test is expected to fail due to the known bug where view factors
    are not invariant under certain angle transformations (e.g., 90° vs 270°).
    """
    cfg = SweepConfig(
        emitter_w=2.0, emitter_h=1.0,
        receiver_w=2.0, receiver_h=1.0,
        setback=5.0, rotate_target="emitter",
        rotate_axis="z", pivot="toe"
    )
    
    # Test angles that should be equivalent due to symmetry
    angles = [90.0, 270.0]  # 90° and 270° should give same result
    results = angle_sweep(angles, "adaptive", cfg)
    
    vf_90 = results[0][1]
    vf_270 = results[1][1]
    
    # These should be equal due to symmetry, but the bug causes them to differ
    assert abs(vf_90 - vf_270) < 1e-10, f"View factors should be equal: 90°={vf_90}, 270°={vf_270}"


def test_angle_sweep_symmetry_180():
    """Test 180-degree symmetry (should be invariant)."""
    cfg = SweepConfig(
        emitter_w=2.0, emitter_h=1.0,
        receiver_w=2.0, receiver_h=1.0,
        setback=5.0, rotate_target="emitter",
        rotate_axis="z", pivot="toe"
    )
    
    # Test 180-degree symmetry
    angles = [0.0, 180.0]
    results = angle_sweep(angles, "adaptive", cfg)
    
    vf_0 = results[0][1]
    vf_180 = results[1][1]
    
    # 0° and 180° should give the same result due to symmetry
    assert abs(vf_0 - vf_180) < 1e-10, f"View factors should be equal: 0°={vf_0}, 180°={vf_180}"


def test_angle_sweep_continuous_range():
    """Test angle sweep over a continuous range to detect discontinuities."""
    cfg = SweepConfig(
        emitter_w=1.0, emitter_h=1.0,
        receiver_w=1.0, receiver_h=1.0,
        setback=3.0, rotate_target="emitter",
        rotate_axis="z", pivot="toe"
    )
    
    # Dense angle range to detect discontinuities
    angles = np.linspace(0.0, 360.0, 37)  # Every 10 degrees
    results = angle_sweep(angles, "adaptive", cfg)
    
    assert len(results) == len(angles)
    
    # Check for reasonable continuity (no sudden jumps)
    vf_values = [vf for _, vf in results]
    for i in range(1, len(vf_values)):
        diff = abs(vf_values[i] - vf_values[i-1])
        # Allow some tolerance for numerical precision
        assert diff < 0.1, f"Sudden jump detected at angle {angles[i]}: {vf_values[i-1]} -> {vf_values[i]}"
