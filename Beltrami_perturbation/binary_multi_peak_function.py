import numpy as np

def binary_multi_peak_function(X: np.ndarray,
                                Y: np.ndarray,
                                num_peaks: int,
                                amplitude_factor: float,
                                frequency_factor: float,
                                noise_level: float) -> np.ndarray:
    """
    Generate a binary multi-peak function over grid (X, Y).

    Args:
        X, Y            : 2D coordinate arrays of the same shape.
        num_peaks       : Number of peaks to sum.
        amplitude_factor: Base factor for random amplitude.
        frequency_factor: Base factor for frequencies.
        noise_level     : Standard deviation of added Gaussian noise.

    Returns:
        Z               : 2D array of the same shape as X and Y, with summed peaks and noise.
    """
    # Initialize output array
    Z = np.zeros_like(X, dtype=float)

    # Sum num_peaks sinusoidal peaks
    for i in range(num_peaks):  # MATLAB i=1:num_peaks
        amplitude = amplitude_factor * (1 + 0.5 * np.random.rand())
        frequency_x = frequency_factor * (i + 1)
        frequency_y = frequency_factor * (i + 2)
        phase_x = 2 * np.pi * np.random.rand()
        phase_y = 2 * np.pi * np.random.rand()

        Z += amplitude * np.sin(frequency_x * X + phase_x) * np.sin(frequency_y * Y + phase_y)

    # Add Gaussian noise
    Z += noise_level * np.random.randn(*X.shape)

    return Z

