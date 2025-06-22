import numpy as np
from scipy.optimize import least_squares

def trs_fit(angles, I):
    """
    Nonlinear least squares optimisation to fit sinusoid

    Inputs:
        angles : np.ndarray
            Vector of polarising filter angles (in radians)
        I : np.ndarray
            Vector of measured intensities

    Returns:
        Iun : float
            Unpolarised intensity component
        rho : float
            Degree of polarisation
        phi : float
            Phase angle (polarisation angle)
    """
    # Initial guess
    b0 = [
        np.mean(I),  # Iun
        np.sqrt(np.mean((I - np.mean(I)) ** 2)) * np.sqrt(2),  # modulation
        0.0  # phi
    ]

    # Bounds for [Iun, modulation, phi]
    bounds = ([0, 0, -np.pi], [np.inf, np.inf, np.pi])

    # Least squares optimization
    result = least_squares(
        lambda b: trs_fit_obj(b, angles, I),
        b0,
        bounds=bounds,
        method='trf'  # trust-region reflective
    )

    b = result.x

    # Adjust phase if negative
    if b[2] < 0:
        b[2] += np.pi

    Iun = b[0]
    Imax = Iun + b[1]
    Imin = Iun - b[1]
    rho = (Imax - Imin) / (Imax + Imin)
    phi = b[2]

    return Iun, rho, phi

def trs_fit_obj(b, angles, I):
    """
    Objective function for least squares fitting.
    """
    I_fit = b[0] + b[1] * np.cos(2 * angles - 2 * b[2])
    return I - I_fit
