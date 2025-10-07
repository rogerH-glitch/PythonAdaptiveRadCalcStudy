import numpy as np

def test_grid_eval_shim_exports_tapped_impl():
    # Import via shim path
    from src.grid_eval import evaluate_grid
    # Drain tap before test (in case a prior test filled it)
    from src.util.grid_tap import drain
    drain()

    # Make a tiny synthetic grid and a trivial vectorized kernel
    Y, Z = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-0.5, 0.5, 3), indexing="xy")
    def point_kernel(y, z):
        return np.exp(-(y**2 + z**2))

    # Call through shim: this should invoke the tapped evaluate_grid
    F = evaluate_grid(point_kernel, Y, Z, dy=0.1, dz=-0.2)
    assert F.shape == Y.shape

    # The tap should now hold (Y, Z, F)
    tapped = drain()
    assert tapped is not None, "Expected tapped field from evaluate_grid via shim"
    Yt, Zt, Ft = tapped
    assert Yt.shape == Y.shape and Zt.shape == Z.shape and Ft.shape == F.shape


