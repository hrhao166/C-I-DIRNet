import numpy as np

def custom_multi_peak_function(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Generate a custom multi-peak surface over the grid defined by X and Y.

    Args:
        X, Y: 2D arrays of the same shape, representing coordinates.

    Returns:
        Z: 2D array of the same shape as X and Y, containing the multi-peak function values.
    """
    # Initialize output
    Z = np.zeros_like(X, dtype=float)

    # Global smooth peak parameters
    base_amplitude = 0.5
    base_frequency = 1.0
    Z += base_amplitude * np.sin(base_frequency * X) * np.cos(base_frequency * Y)

    # Sharp peak parameters
    sharp_peak_x = [5, -3]
    sharp_peak_y = [0, 4]
    sharp_amplitude = 10.0
    sharp_frequency = 1.0

    # Add sharp peaks
    for xc, yc in zip(sharp_peak_x, sharp_peak_y):
        Z += sharp_amplitude * np.exp(-(((X - xc)**2 + (Y - yc)**2) / 0.1)) \
             * np.sin(sharp_frequency * (X - xc)) * np.cos(sharp_frequency * (Y - yc))

    # Optional: add global noise
    # Z += 0.1 * np.random.randn(*X.shape)

    return Z

