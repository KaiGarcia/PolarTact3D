import numpy as np
from scipy.ndimage import binary_erosion
from skimage.measure import label, regionprops
from math import acos, pi, atan2

def Propagation(rho, phi, mask, n):
    """
    Shape-from-polarization by boundary propagation
    Parameters:
        rho    - 2D matrix of DOP values
        phi    - 2D matrix of phase angles
        mask   - 2D binary foreground mask
        n      - refractive index
    Returns:
        N      - 3D matrix of surface normals
        height - height map obtained by integrating N using lsqintegration
    """
    # Invert degree of diffuse polarization expression to compute zenith angle
    temp = ((2 * rho + 2 * n ** 2 * rho - 2 * n ** 2 + n ** 4 + rho ** 2 + 4 * n ** 2 * rho ** 2 - n ** 4 * rho ** 2 - 4 * n ** 3 * rho * np.sqrt((rho - 1) * (rho + 1)) + 1) / 
            (n ** 4 * rho ** 2 + 2 * n ** 4 * rho + n ** 4 + 6 * n ** 2 * rho ** 2 + 4 * n ** 2 * rho - 2 * n ** 2 + rho ** 2 + 2 * rho + 1)) ** 0.5
    temp = np.minimum(np.real(temp), 1)
    theta = np.arccos(temp)

    # Pad matrices to avoid boundary problems
    mask = np.pad(mask, pad_width=2, mode='constant', constant_values=0)
    theta = np.pad(theta, pad_width=2, mode='constant', constant_values=0)
    phi = np.pad(phi, pad_width=2, mode='constant', constant_values=0)
    
    rows, cols = mask.shape
    N = np.full((rows, cols, 3), np.nan)

    # Find boundary of the mask
    boundary = mask != mask
    available_estimates = mask != mask

    # Fill in boundary normals with disambiguation closest to outward facing normal
    boundaries = label(mask)
    for region in regionprops(boundaries):
        for coord in region.coords:
            r, c = coord
            available_estimates[r, c] = True
            boundary[r, c] = True
            azi = atan2(region.coords[1, 1] - region.coords[0, 1], region.coords[1, 0] - region.coords[0, 0])
            n1 = np.array([np.sin(phi[r, c]) * np.sin(theta[r, c]), np.cos(phi[r, c]) * np.sin(theta[r, c]), np.cos(theta[r, c])])
            n2 = np.array([np.sin(phi[r, c] + pi) * np.sin(theta[r, c]), np.cos(phi[r, c] + pi) * np.sin(theta[r, c]), np.cos(theta[r, c])])
            nb = np.array([np.cos(azi) * np.sin(theta[r, c]), np.sin(azi) * np.sin(theta[r, c]), np.cos(theta[r, c])])

            if np.dot(n1, nb) > np.dot(n2, nb):
                N[r, c, :] = n1
            else:
                N[r, c, :] = n2

    # Consider interior pixels
    interior = mask & ~boundary
    sorted_idx = np.argsort(theta[interior], axis=None)[::-1]
    row, col = np.meshgrid(np.arange(1, cols + 1), np.arange(1, rows + 1))

    # Iterate over sorted zenith angles
    while sorted_idx.size > 0:
        flag = False
        selected = 0
        r, c = row[interior], col[interior]
        while not flag:
            neighbourhood = []
            for i in range(-3, 4):
                for j in range(-3, 4):
                    if i != 0 or j != 0:
                        if available_estimates[r[sorted_idx[selected]] + i, c[sorted_idx[selected]] + j]:
                            neighbourhood.append([r[sorted_idx[selected]] + i, c[sorted_idx[selected]] + j])
            if neighbourhood:
                flag = True
            else:
                selected += 1

        r, c = r[sorted_idx[selected]], c[sorted_idx[selected]]
        Ns = []
        for n in neighbourhood:
            Ns.append([N[n[0], n[1], 0], N[n[0], n[1], 1], N[n[0], n[1], 2]])

        Ns = np.array(Ns)
        n1 = np.array([np.sin(phi[r, c]) * np.sin(theta[r, c]), np.cos(phi[r, c]) * np.sin(theta[r, c]), np.cos(theta[r, c])])
        n2 = np.array([np.sin(phi[r, c] + pi) * np.sin(theta[r, c]), np.cos(phi[r, c] + pi) * np.sin(theta[r, c]), np.cos(theta[r, c])])

        if np.mean(np.arccos(np.dot(Ns, n1))) < np.mean(np.arccos(np.dot(Ns, n2))):
            N[r, c, :] = n1
        else:
            N[r, c, :] = n2

        available_estimates[r, c] = True
        sorted_idx = np.delete(sorted_idx, selected)

    # Unpad estimated normals and mask
    N = N[2:-2, 2:-2, :]
    N = np.real(N)
    mask = mask[2:-2, 2:-2]

    # Integrate normals into height map (assuming lsqintegration is defined elsewhere)
    height = lsqintegration(N, mask, False, None)

    return N, height
