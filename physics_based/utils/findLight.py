import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import lsq_linear

def find_light(theta, phi, diffuse, mask, ldim=3, l=None):
    """
    Estimate illumination from polarisation data.

    Parameters:
        theta (numpy.ndarray): Zenith angle estimates from degree of polarisation.
        phi (numpy.ndarray): Phase angle from polarisation image.
        diffuse (numpy.ndarray): Unpolarised intensity.
        mask (numpy.ndarray): Binary foreground mask.
        ldim (int): Set to 3, 4 or 9 depending on the lighting model.
        l (numpy.ndarray): Known light direction (optional).

    Returns:
        l (numpy.ndarray): Lighting coefficient vector.
        T (numpy.ndarray): Ambiguous transformation matrix.
        B (numpy.ndarray): Basis consistent with l, reshaped to (H, 3, W).
    """
    H, W = mask.shape
    print(f"Input mask shape: {mask.shape}")

     # Flatten all input images
    theta_flat = theta.flatten()
    phi_flat = phi.flatten()
    diffuse_flat = diffuse.flatten()
    mask_flat = mask.flatten()

    # Apply mask correctly
    i = diffuse_flat[mask_flat]
    theta = theta_flat[mask_flat]
    phi = phi_flat[mask_flat]

    print(f"Masked diffuse intensity shape: {i.shape}")
    print(f"Masked theta shape: {theta.shape}")
    print(f"Masked phi shape: {phi.shape}")

     # Compute surface normals N = [nx, ny, nz]
    N = np.column_stack([
        np.sin(phi) * np.sin(theta),
        np.cos(phi) * np.sin(theta),
        np.cos(theta)
    ])

    print(f"Normal matrix N shape: {N.shape}")

    # Set up the basis and transformation matrices
    if ldim == 3:
        B = N
        T = np.diag([-1, -1, 1])
    elif ldim == 4:
        B = np.column_stack([N, np.ones(N.shape[0])])
        T = np.diag([-1, -1, 1, 1])
    elif ldim == 9:
        B = np.column_stack([np.ones(N.shape[0]), N,
                             3 * N[:, 2]**2 - 1,
                             N[:, 0] * N[:, 1],
                             N[:, 0] * N[:, 2],
                             N[:, 1] * N[:, 2],
                             N[:, 0]**2 - N[:, 1]**2])
        T = np.diag([1, -1, -1, 1, 1, 1, -1, -1, 1])

    print(f"Initial B shape (before column slicing): {B.shape}")

    if B.shape[1] != 3 and ldim == 3:
        B = B[:, :3]

    print(f"B shape after ensuring 3 columns: {B.shape}")

    tau = 1e-9
    maxiter = 100
    niter = 0
    converged = False

    if l is not None:
        print("Using provided initial light direction:", l)
        while not converged:
            B_l = B @ l
            print(f"Iteration {niter}: Shape of B_l: {B_l.shape}")

            # Decide which pixels to transform
            idx = (np.square(B_l - i) > np.square((B @ T) @ l - i))
            P = B.copy()
            P[idx] = (P[idx] @ T)

            # Solve for new l
            l_new = l * lstsq(P, i)[0]
            if np.linalg.norm(l - l_new) < tau:
                converged = True
            l = l_new
            niter += 1
            if niter > maxiter:
                print("Max iterations reached.")
                break
    else:
        l = np.random.randn(ldim)
        if ldim == 4:
            l[3] = abs(l[3])

        while not converged:
            B_l = B @ l
            print(f"Iteration {niter}: Shape of B_l: {B_l.shape}")

            idx = (np.square(B_l - i) > np.square((B @ T) @ l - i))
            P = B.copy()
            P[idx] = (P[idx] @ T)

            if ldim == 3 or ldim == 9:
                l_new = lstsq(P, i)[0]
            elif ldim == 4:
                result = lsq_linear(P, i, bounds=(None, np.inf))
                l_new = result.x

            if np.linalg.norm(l - l_new) < tau:
                converged = True
            l = l_new
            niter += 1
            if niter > maxiter:
                print("Max iterations reached.")
                break

    print(f"Number of iterations = {niter}")
    print(f"Residual = {np.linalg.norm(P @ l - i)**2}")

    # Optional: return B in reshaped form later if needed
    return l, T, P