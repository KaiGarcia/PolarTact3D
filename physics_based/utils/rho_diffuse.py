import numpy as np

def rho_diffuse(rho, n):
    """
    Compute zenith angle (theta) from degree of polarization (rho) for diffuse pixels.

    Parameters:
        rho (numpy.ndarray or float): Degree of polarization values.
        n (float): Index of refraction.

    Returns:
        numpy.ndarray or float: Zenith angle estimates (same size as rho).
    """
    temp = ((2 * rho + 2 * n**2 * rho - 2 * n**2 + n**4 + rho**2 + 4 * n**2 * rho**2 - n**4 * rho**2 
             - 4 * n**3 * rho * np.sqrt(np.maximum(-(rho - 1) * (rho + 1), 0)) + 1) /
            (n**4 * rho**2 + 2 * n**4 * rho + n**4 + 6 * n**2 * rho**2 + 4 * n**2 * rho 
             - 2 * n**2 + rho**2 + 2 * rho + 1)) ** 0.5
    
    # Ensure the result stays within the valid range to avoid complex numbers due to numerical precision
    temp = np.clip(np.real(temp), 0, 1)

    return np.arccos(temp)

# Example usage
rho = np.linspace(0, 1, 100)  # Example degree of polarization values
n = 1.5  # Example refractive index
theta_values = rho_diffuse(rho, n)
