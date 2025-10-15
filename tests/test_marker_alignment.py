import numpy as np
from src.viz._marker_utils import grid_argmax_to_yz, subcell_quadratic_peak


def test_grid_argmax_mapping_matches_field_indices():
    gy = np.linspace(-2.55, 2.55, 41)
    gz = np.linspace(-1.05, 1.05, 41)
    Y, Z = np.meshgrid(gy, gz, indexing="ij")
    F = np.exp(-((Y + 0.25) ** 2 + (Z - 0.0) ** 2) / (0.2 ** 2))
    yg, zg, (j, i) = grid_argmax_to_yz(F, gy, gz)
    j0, i0 = np.unravel_index(np.nanargmax(F), F.shape)
    assert (j, i) == (j0, i0)
    assert np.isclose(yg, gy[j])
    assert np.isclose(zg, gz[i])


def test_subcell_refines_but_stays_within_extent():
    gy = np.linspace(-2.55, 2.55, 21)
    gz = np.linspace(-1.05, 1.05, 21)
    Y, Z = np.meshgrid(gy, gz, indexing="ij")
    F = np.exp(-((Y + 0.52) ** 2 + (Z - 0.18) ** 2) / (0.25 ** 2))
    yg, zg, (j, i) = grid_argmax_to_yz(F, gy, gz)
    yh, zh = subcell_quadratic_peak(F, gy, gz, j, i)
    assert np.isfinite(yh) and np.isfinite(zh)
    assert (gy.min() - 1e-9) <= yh <= (gy.max() + 1e-9)
    assert (gz.min() - 1e-9) <= zh <= (gz.max() + 1e-9)
    dy = (gy[-1] - gy[0]) / (len(gy) - 1)
    dz = (gz[-1] - gz[0]) / (len(gz) - 1)
    assert abs(yh - yg) <= 1.5 * dy
    assert abs(zh - zg) <= 1.5 * dz


