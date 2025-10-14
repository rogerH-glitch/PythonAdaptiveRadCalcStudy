# Validation Guide

This document describes the validation procedures and test harnesses used to ensure the accuracy and reliability of the radiation view factor calculations.

## Angle sweep checks (new)

To validate orientation handling and catch regressions, run the harness tests:

- Monotone drop in VF from **0° → 60°** for yaw/pitch with toe pivot.
- Cross-check **adaptive** vs **Monte Carlo** for a few angles.
- Large-angle (≈80°+) sweeps now **decrease** as expected; tests assert non-constant behavior.

The harness is implemented in `src/validation/angle_sweep.py` and tested by
`tests/test_angle_sweep_harness.py`.
