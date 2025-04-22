import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter

def sharp_peak_function(X: np.ndarray, Y: np.ndarray, params: dict) -> np.ndarray:
    """
    Generate a sharp Gaussian peak with given parameters.
    params keys: x_center, y_center, sigma_x, sigma_y, amplitude, sign
    """
    xc = params['x_center']
    yc = params['y_center']
    sx = params['sigma_x']
    sy = params['sigma_y']
    amp = params['amplitude']
    sign = params['sign']
    return sign * amp * np.exp(-(((X - xc)**2) / (2*sx**2) + ((Y - yc)**2) / (2*sy**2)))


def beltrami_generation_function(r: np.ndarray,
                                  t: np.ndarray,
                                  k1: int,
                                  k2: int) -> (np.ndarray, np.ndarray):
    """
    Python translation of MATLAB beltrami_generation_function.m
    Inputs:
      r    : array shape (M, N)
      t    : array shape (M, N)
      k1   : number of grid points in x-direction for peaks
      k2   : number of grid points in y-direction for peaks
    Outputs:
      rou  : perturbation of r
      tau  : perturbation of t
    """
    # average filter (unused)
    # filter1 = uniform_filter
    # Gaussian filter parameters
    gauss_size = 7
    gauss_sigma = 1e6

    # build grid for sharp peaks
    x = np.linspace(-10, 10, k1)
    y = np.linspace(-10, 10, k2)
    X, Y = np.meshgrid(x, y)

    # random peak parameters
    peak_params1 = {
        'x_center': np.random.rand()*16 - 8,
        'y_center': np.random.rand()*16 - 8,
        'sigma_x': np.random.rand()*1.5 + 0.5,
        'sigma_y': np.random.rand()*1.5 + 0.5,
        'amplitude': np.random.rand()*0.6 + 0.2,
        'sign': np.random.randint(0,2)*2 - 1
    }
    peak_params2 = {
        'x_center': np.random.rand()*16 - 8,
        'y_center': np.random.rand()*16 - 8,
        'sigma_x': np.random.rand()*1.5 + 0.5,
        'sigma_y': np.random.rand()*1.5 + 0.5,
        'amplitude': np.random.rand()*0.6 + 0.2,
        'sign': np.random.randint(0,2)*2 - 1
    }

    # sharp peaks
    Z1 = sharp_peak_function(X, Y, peak_params1)
    Z2 = sharp_peak_function(X, Y, peak_params2)

    # global smooth perturbation (assumes r.shape == (129,129))
    rr = (2*np.random.rand(*r.shape) - 1) / 2
    tt = (2*np.random.rand(*t.shape) - 1) / 2

    # zero border of width 5
    rr[:5, :] = 0
    rr[-5:, :] = 0
    rr[:, :5] = 0
    rr[:, -5:] = 0
    tt[:5, :] = 0
    tt[-5:, :] = 0
    tt[:, :5] = 0
    tt[:, -5:] = 0

    # apply gaussian filter twice
    rr = gaussian_filter(rr, sigma=gauss_sigma, truncate=3)
    rr = gaussian_filter(rr, sigma=gauss_sigma, truncate=3)
    tt = gaussian_filter(tt, sigma=gauss_sigma, truncate=3)
    tt = gaussian_filter(tt, sigma=gauss_sigma, truncate=3)

    # combine perturbations
    rr = rr + Z1
    tt = tt + Z2

    # update inputs
    rou = r + rr
    tau = t + tt

    # enforce constraint rou^2 + tau^2 < 1 (with small epsilon)
    mask = rou**2 + tau**2 >= 1
    denom = np.sqrt(rou**2 + tau**2 + 0.05)
    rou[mask] /= denom[mask]
    tau[mask] /= denom[mask]

    return rou, tau


