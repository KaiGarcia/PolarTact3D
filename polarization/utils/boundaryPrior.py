import numpy as np
import cv2
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter

def boundary_prior(mask, weight=5):
    """
    Compute convexity prior from a binary foreground mask.

    Parameters:
        mask (numpy.ndarray): Binary mask (rows x cols).
        weight (float): Determines how fast the per-pixel weight falls off from the boundary (default: 5).

    Returns:
        azi (numpy.ndarray): Azimuth angle prior (rows x cols).
        Bdist (numpy.ndarray): Weight associated with each azimuth estimate.
    """
    print(f"Input mask shape: {mask.shape}")

    # Find boundary of mask
    B = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    print(f"Number of contours found: {len(B)}")

    bpoints = np.vstack([cnt[:, 0, :] for cnt in B])  # Convert contours to list of points
    print(f"Shape of boundary points array: {bpoints.shape}")

    brow, bcol = bpoints[:, 1], bpoints[:, 0]

    # Compute distance to closest point on boundary
    mask_indices = np.argwhere(mask)
    print(f"Number of foreground pixels: {len(mask_indices)}")

    kdtree = KDTree(np.column_stack((brow, bcol)))
    _, D = kdtree.query(mask_indices)

    # Create Bdist array
    Bdist = np.full(mask.shape, np.nan)
    mask_flat_indices = tuple(mask_indices.T)
    Bdist[mask_flat_indices] = -D
    Bdist[mask_flat_indices] -= np.nanmin(Bdist)
    Bdist /= np.nanmax(Bdist)
    Bdist **= weight
    print("Bdist array computed")

    # Initialize azimuth array
    azi = np.zeros_like(mask, dtype=np.float32)
    mask2 = mask.copy()
    iteration = 0

    # Repeatedly erode boundary to propagate into interior
    while np.sum(mask2) > 0:
        contours, _ = cv2.findContours(mask2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print("No more contours found; stopping.")
            break

        print(f"Iteration {iteration}: Number of contours = {len(contours)}")

        contour = contours[0][:, 0, :]  # shape (N, 2)
        if contour.shape[0] < 2:
            print("Contour too small; stopping.")
            break

        # Close the contour loop
        contour = np.vstack([contour, contour[0][None, :]])

        for i in range(len(contour) - 1):
            dy = contour[i + 1, 1] - contour[i, 1]
            dx = contour[i + 1, 0] - contour[i, 0]
            azi[contour[i, 1], contour[i, 0]] = np.arctan2(dx, dy)

        mask2 = cv2.erode(mask2.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        iteration += 1

    # Convert azimuth angles to vectors
    dx = np.cos(azi)
    dy = np.sin(azi)
    dx[~mask] = 0
    dy[~mask] = 0

    # Smooth vector field using a Gaussian filter
    dx = gaussian_filter(dx, sigma=2)
    dy = gaussian_filter(dy, sigma=2)

    # Transform back to azimuth angle
    azi = np.arctan2(dy, dx)
    azi = np.mod(-azi + np.pi - np.pi / 2, 2 * np.pi)

    print("Azimuth (azi) computation complete")

    return azi, Bdist

# Example test
if __name__ == "__main__":
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True  # Example square mask
    azi, Bdist = boundary_prior(mask)
