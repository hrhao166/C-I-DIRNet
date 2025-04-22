import numpy as np

def sharp_peak_function(X: np.ndarray, Y: np.ndarray, peak_params: dict) -> np.ndarray:
    """
    Generate a sharp Gaussian peak over a 2D grid.

    Args:
        X, Y: 2D coordinate arrays of the same shape.
        peak_params: dict with keys:
            - x_center: float, center x-coordinate of the peak
            - y_center: float, center y-coordinate of the peak
            - sigma_x: float, standard deviation in x-direction
            - sigma_y: float, standard deviation in y-direction
            - amplitude: float, peak amplitude (positive)
            - sign: int, +1 or -1 (optional; if missing, selected randomly)

    Returns:
        Z: 2D array of same shape as X and Y, containing the sharp peak.
    """
    # Determine sign factor
    sign_factor = peak_params.get('sign')
    if sign_factor is None:
        sign_factor = np.random.choice([-1, 1])

    # Extract parameters
    xc = peak_params['x_center']
    yc = peak_params['y_center']
    sx = peak_params['sigma_x']
    sy = peak_params['sigma_y']
    amp = peak_params['amplitude']

    # Compute the Gaussian peak
    Z = sign_factor * amp * np.exp(
        -(((X - xc)**2) / (2 * sx**2) + ((Y - yc)**2) / (2 * sy**2))
    )
    return Z

