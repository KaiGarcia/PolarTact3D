import numpy as np
from numpy.linalg import norm
from scipy.ndimage import gaussian_filter

def lambertian_sfp(rho, phi, mask, n, s, albedo, Iun):
    """
    Shape-from-polarisation with Lambertian model.
    
    Args:
        rho (ndarray): Degree of polarisation (rows x cols)
        phi (ndarray): Phase angle (rows x cols)
        mask (ndarray): Binary mask (rows x cols)
        n (float): Refractive index
        s (ndarray): Light source direction (3,)
        albedo (ndarray or float): Diffuse albedo
        Iun (ndarray): Unpolarised intensity

    Returns:
        N (ndarray): Surface normals (rows x cols x 3)
        height (ndarray): Height map (rows x cols)
    """
    # Zenith angle from DOP
    numerator = (
        2*rho + 2*n**2*rho - 2*n**2 + n**4 + rho**2 + 4*n**2*rho**2 -
        n**4*rho**2 - 4*n**3*rho*np.sqrt(1 - rho**2) + 1
    )
    denominator = (
        n**4*rho**2 + 2*n**4*rho + n**4 + 6*n**2*rho**2 +
        4*n**2*rho - 2*n**2 + rho**2 + 2*rho + 1
    )
    temp = np.sqrt(np.maximum(np.real(numerator / denominator), 0))
    temp = np.minimum(temp, 1)
    theta = np.arccos(temp)

    # Handle Iun > albedo
    Iun = np.where((Iun / albedo) > 1, albedo, Iun)
    Iun = Iun / albedo

    # Ambiguous polarisation normals
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    N1 = np.stack([
        np.sin(phi) * sin_theta,
        np.cos(phi) * sin_theta,
        cos_theta
    ], axis=-1)

    N2 = np.stack([
        np.sin(phi + np.pi) * sin_theta,
        np.cos(phi + np.pi) * sin_theta,
        cos_theta
    ], axis=-1)

    N = np.full_like(N1, np.nan)

    rows, cols = mask.shape
    a, b, c3 = s

    for row in range(rows):
        for col in range(cols):
            if not mask[row, col]:
                continue

            theta_rc = theta[row, col]
            stheta2 = np.sin(theta_rc) ** 2
            c = c3 * np.cos(theta_rc) - Iun[row, col]
            d = -stheta2

            discrim = -d*a**2 - d*b**2 - c**2
            if discrim < 0:
                continue

            root = np.sqrt(discrim)

            denom = a**2 + b**2
            ny1 = -(b*c + a*root) / denom
            ny2 = -(b*c - a*root) / denom

            nx1 = np.sqrt(stheta2 - ny1**2)
            nx2 = -np.sqrt(stheta2 - ny1**2)

            n1 = np.array([nx1, ny1, np.cos(theta_rc)])
            n2 = np.array([nx2, ny2, np.cos(theta_rc)])

            n1_dot1 = np.dot(n1, N1[row, col])
            n1_dot2 = np.dot(n1, N2[row, col])
            n2_dot1 = np.dot(n2, N1[row, col])
            n2_dot2 = np.dot(n2, N2[row, col])

            n1best = max(n1_dot1, n1_dot2)
            n2best = max(n2_dot1, n2_dot2)

            na = n1 if n1best > n2best else n2

            nb = N1[row, col] if np.dot(na, N1[row, col]) > np.dot(na, N2[row, col]) else N2[row, col]

            n = na + nb
            if norm(n[:2]) != 0:
                n[:2] = (n[:2] / norm(n[:2])) * np.sin(theta_rc)
            n[2] = np.cos(theta_rc)

            N[row, col] = np.real(n)

    height = lsq_integration(N, mask)
    return N, height


def lsq_integration(N, mask, fill=False, smoothing_sigma=None):
    """
    Least squares integration of surface normals into a height map.
    
    Args:
        N (ndarray): Surface normals (rows x cols x 3)
        mask (ndarray): Binary mask
        fill (bool): Fill in outside region (default False)
        smoothing_sigma (float): Optional Gaussian smoothing
    
    Returns:
        height (ndarray): Integrated height map
    """
    rows, cols, _ = N.shape
    zx = -N[:, :, 0] / N[:, :, 2]
    zy = -N[:, :, 1] / N[:, :, 2]

    zx[~mask] = 0
    zy[~mask] = 0

    fx = np.zeros((rows, cols))
    fy = np.zeros((rows, cols))

    fx[:, 1:] = zx[:, 1:] - zx[:, :-1]
    fy[1:, :] = zy[1:, :] - zy[:-1, :]

    height = np.zeros((rows, cols))
    for i in range(1, rows):
        height[i, 0] = height[i - 1, 0] + zy[i, 0]
    for j in range(1, cols):
        height[:, j] = height[:, j - 1] + zx[:, j]

    height[~mask] = 0
    if smoothing_sigma:
        height = gaussian_filter(height, sigma=smoothing_sigma)

    return height
