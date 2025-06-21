import numpy as np
from scipy.optimize import least_squares

def polarisation_image(images, angles, mask=None, method='linear'):
    rows, cols, n_images = images.shape
    
    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.astype(bool).reshape(-1)
        I = images.reshape(rows * cols, n_images)[mask_flat]
    else:
        I = images.reshape(-1, n_images)
        mask_flat = np.ones(I.shape[0], dtype=bool)

    # Debugging: print mask and flattened shapes
    print(f"Original mask shape: {mask.shape}" if mask is not None else "No mask provided")
    print(f"Flattened mask shape: {mask_flat.shape}")
    print(f"Shape of I (after applying mask): {I.shape}")
    
    if method == 'nonlinear':
        def trs_fit(params, angles, I):
            Iun, rho, phi = params
            return Iun * (1 + rho * np.cos(2 * angles - 2 * phi)) - I

        Iun = np.zeros(I.shape[0])
        rho = np.zeros(I.shape[0])
        phi = np.zeros(I.shape[0])
        
        # Debugging: Nonlinear method progress
        print("Starting nonlinear fitting...")
        for i in range(I.shape[0]):
            res = least_squares(trs_fit, [np.mean(I[i]), 0, 0], args=(angles, I[i]))
            Iun[i], rho[i], phi[i] = res.x
            if i % 1000 == 0:  # Print progress every 1000 iterations
                print(f"Nonlinear fitting progress: {i}/{I.shape[0]}")

        phi = np.mod(phi, np.pi)

    elif method == 'linear':
        A = np.column_stack([np.ones(n_images), np.cos(2 * angles), np.sin(2 * angles)])
        x = np.linalg.lstsq(A, I.T, rcond=None)[0].T
        Imax = x[:, 0] + np.sqrt(x[:, 1]**2 + x[:, 2]**2)
        Imin = x[:, 0] - np.sqrt(x[:, 1]**2 + x[:, 2]**2)
        Iun = (Imin + Imax) / 2
        rho = (Imax - Imin) / (Imax + Imin)
        phi = 0.5 * np.arctan2(x[:, 2], x[:, 1])
        phi = np.mod(phi - np.pi / 2, np.pi)

    # Debugging: print sizes of results
    print(f"Size of rho: {rho.size}")
    print(f"Size of phi: {phi.size}")
    print(f"Size of Iun: {Iun.size}")
    
    # Reshape back to original image size if mask is provided
    if mask is not None:
        # Ensure the size of rho matches the number of foreground pixels
        if len(rho) != mask_flat.sum():
            raise ValueError(f"Size mismatch: rho size ({len(rho)}) and mask size ({mask_flat.sum()}) do not match.")
        
        # Debugging: Mask processing
        print(f"Number of foreground pixels (mask): {mask_flat.sum()}")

        # Create full images of the same shape as the input images
        rho_full = np.zeros((rows, cols))
        phi_full = np.zeros((rows, cols))
        Iun_full = np.zeros((rows, cols))

        # Debugging: Check shapes before inserting into the full arrays
        print(f"Shape of full images: {rho_full.shape}")

        # Put the computed values into the corresponding masked positions
        rho_full.ravel()[mask_flat] = rho
        phi_full.ravel()[mask_flat] = phi
        Iun_full.ravel()[mask_flat] = Iun

        # Now return the full images (with the mask applied)
        rho, phi, Iun = rho_full, phi_full, Iun_full
    else:
        # If no mask is provided, just reshape the results back to the image dimensions
        rho = rho.reshape(rows, cols)
        phi = phi.reshape(rows, cols)
        Iun = Iun.reshape(rows, cols)

    # Debugging: Final reshaped results
    print(f"Final rho shape: {rho.shape}")
    print(f"Final phi shape: {phi.shape}")
    print(f"Final Iun shape: {Iun.shape}")

    return rho, phi, Iun
