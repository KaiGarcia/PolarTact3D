import numpy as np
from scipy.interpolate import interp1d

def rho_spec(rho, n):
    """
    Compute zenith angle (theta_s) from degree of polarization (rho) for specular pixels.
    Uses a lookup table and interpolation.

    Parameters:
        rho (numpy.ndarray or float): Degree of polarization values.
        n (float): Index of refraction.

    Returns:
        numpy.ndarray or float: Zenith angle estimates (same size as rho).
    """
    theta = np.arange(0, np.pi/2, 0.01)  # Generate theta values from 0 to pi/2

    # Compute rho_s values
    rho_s = (2 * np.sin(theta)**2 * np.cos(theta) * np.sqrt(n**2 - np.sin(theta)**2)) / (
            n**2 - np.sin(theta)**2 - n**2 * np.sin(theta)**2 + 2 * np.sin(theta)**4)

    # Find the maximum valid position
    maxpos = np.argmax(rho_s)  # Equivalent to MATLAB's `find(rho_s==max(rho_s))`

    # Trim theta and rho_s up to the max position
    theta = theta[:maxpos+1]
    rho_s = rho_s[:maxpos+1]

    # Create interpolation function
    interp_func = interp1d(rho_s, theta, bounds_error=False, fill_value="extrapolate")

    # Interpolate to find theta_s for given rho values
    theta_s = interp_func(rho)

    return theta_s

# Example usage
rho = np.linspace(0, 1, 100)  # Example degree of polarization values
n = 1.5  # Example refractive index
theta_s_values = rho_spec(rho, n)
