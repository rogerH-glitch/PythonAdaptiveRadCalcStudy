import numpy as np


def grid_argmax_to_yz(F: np.ndarray, gy: np.ndarray, gz: np.ndarray):
    j, i = np.unravel_index(np.nanargmax(F), F.shape)
    return float(gy[j]), float(gz[i]), (int(j), int(i))


def subcell_quadratic_peak(F: np.ndarray, gy: np.ndarray, gz: np.ndarray, j: int, i: int):
    ny, nz = F.shape
    if j <= 0 or j >= ny - 1 or i <= 0 or i >= nz - 1:
        return float(gy[j]), float(gz[i])
    Y, Z, V = [], [], []
    for jj in (j - 1, j, j + 1):
        for ii in (i - 1, i, i + 1):
            v = F[jj, ii]
            if np.isfinite(v):
                Y.append(gy[jj]); Z.append(gz[ii]); V.append(v)
    if len(V) < 6:
        return float(gy[j]), float(gz[i])
    Y = np.asarray(Y); Z = np.asarray(Z); V = np.asarray(V)
    X = np.column_stack([Y*Y, Z*Z, Y*Z, Y, Z, np.ones_like(Y)])
    try:
        coeffs, *_ = np.linalg.lstsq(X, V, rcond=None)
    except np.linalg.LinAlgError:
        return float(gy[j]), float(gz[i])
    a, b, c, d, e, f = coeffs
    A = np.array([[2*a, c], [c, 2*b]], dtype=float)
    bvec = -np.array([d, e], dtype=float)
    try:
        sol = np.linalg.solve(A, bvec)
        y_hat, z_hat = float(sol[0]), float(sol[1])
    except np.linalg.LinAlgError:
        y_hat, z_hat = float(gy[j]), float(gz[i])
    return y_hat, z_hat


