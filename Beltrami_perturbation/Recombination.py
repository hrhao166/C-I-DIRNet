import numpy as np

def recombination(T: np.ndarray, phyx: np.ndarray, phyy: np.ndarray) -> np.ndarray:
    """
    Compose deformation by bilinear interpolation of T at deformed coordinates.

    Args:
        T    : 2D array of shape (129, 129), original image/field
        phyx : 2D array of shape (129, 129), x-coordinate mapping (float)
        phyy : 2D array of shape (129, 129), y-coordinate mapping (float)

    Returns:
        D    : 2D array of shape (129, 129), deformed image/field
    """
    N = 129
    D = np.zeros((N, N), dtype=T.dtype)

    # Loop over interior points (MATLAB i=2:128, j=2:128)
    for i in range(1, N-1):
        for j in range(1, N-1):
            x = phyx[i, j]
            y = phyy[i, j]
            # Integer part (MATLAB fix)
            m1 = int(np.floor(x))
            m2 = int(np.floor(y))
            # Clamp to [1, 128]
            m1 = min(max(m1, 1), N-1)
            m2 = min(max(m2, 1), N-1)
            # Fractional deltas
            deltx = x - m1
            delty = y - m2
            # Convert to 0-based indices
            i1, j1 = m1 - 1, m2 - 1
            # Bilinear interpolation
            D[i, j] = (
                (1 - deltx) * (1 - delty) * T[i1,     j1]     +
                deltx       * (1 - delty) * T[i1 + 1, j1]     +
                (1 - deltx) * delty       * T[i1,     j1 + 1] +
                deltx       * delty       * T[i1 + 1, j1 + 1]
            )
    return D
