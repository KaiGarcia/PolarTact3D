import numpy as np
from scipy.ndimage import convolve
from scipy.sparse import lil_matrix

def lsqintegration(N, mask, verbose=True, guidez=None):
    """
    LSQINTEGRATION Least squares surface integration with optional guide z
    Input:
        N      - rows by cols by 3 matrix containing surface normals
        mask   - rows by cols binary foreground mask
        guidez - rows by cols guide depth map
    Output:
        z      - estimated depth map
    """
    
    rows, cols, _ = N.shape
    
    # Pad to avoid boundary problems
    N2 = np.zeros((rows + 2, cols + 2, 3))
    N2[1:rows+1, 1:cols+1, :] = N
    N = N2
    
    mask2 = np.zeros((rows + 2, cols + 2))
    mask2[1:rows+1, 1:cols+1] = mask
    mask = mask2
    
    rows, cols = rows + 2, cols + 2
    
    # Build lookup table relating x,y coordinate of valid pixels to index position in vectorised representation
    count = 0
    indices = np.zeros(mask.shape, dtype=int)
    for row in range(rows):
        for col in range(cols):
            if mask[row, col]:
                count += 1
                indices[row, col] = count
    
    # Create mask for smoothed central difference filter
    h = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
    maskSCDx = convolve(mask, h, mode='constant', cval=0) == 6
    maskSCDy = convolve(mask, h.T, mode='constant', cval=0) == 6
    SCDx = (1/12) * np.array([[-1, 0, 1], [-4, 0, 4], [-1, 0, 1]])
    SCDy = np.flipud(SCDx.T)
    
    # Create mask for SavGol gradient filter
    h = np.ones((5, 5))
    h[:, 2] = 0
    maskSGx = convolve(mask, h, mode='constant', cval=0) == 20
    maskSGy = convolve(mask, h.T, mode='constant', cval=0) == 20
    SG = SavGol(3, 5)  # Assuming you have this function implemented
    SGx = SG[:, :, 1].T
    SGy = np.flipud(SGx)
    
    # Create mask for 4-neighbours for Laplacian smoothing
    h = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    mask4n = convolve(mask, h, mode='constant', cval=0) == 4
    
    # Smoothing weight. lambda=0 => no smoothing.
    lambda_smooth = 0.2
    
    npix = np.sum(mask)  # The number of usable pixels
    if verbose:
        print(f"Using {npix} pixels")
    
    # Preallocate maximum required space
    # This would be if all valid pixels had equations for all 8 neighbours for all possible pairs of images - it will be less in practice
    i, j, s = [], [], []
    d = np.zeros(npix * 2)
    
    NumEq = 0  # number of rows in matrix
    k = 0  # total number of non-zero entries in matrix
    
    for row in range(rows):
        for col in range(cols):
            if mask[row, col]:
                if mask4n[row, col] and lambda_smooth != 0:
                    # Add Laplacian smoothing term
                    NumEq += 1
                    d[NumEq - 1] = 0
                    k += 1
                    i.append(NumEq - 1)
                    j.append(indices[row, col] - 1)
                    s.append(-4 * lambda_smooth)
                    # Edge neighbours
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        k += 1
                        i.append(NumEq - 1)
                        j.append(indices[row + dr, col + dc] - 1)
                        s.append(lambda_smooth)
                
                # X equations
                if maskSCDx[row, col]:
                    NumEq += 1
                    d[NumEq - 1] = P[row, col]  # Assuming P is defined similarly
                    for a in range(3):
                        for b in range(3):
                            if SCDx[a, b] != 0:
                                k += 1
                                i.append(NumEq - 1)
                                j.append(indices[row + a - 2, col + b - 2] - 1)
                                s.append(SCDx[a, b])
                # Similar processing for Y equations goes here
                
    if verbose:
        print(f"System contains {NumEq} linear equations with {k} non-zero entries in C")
    
    # Build matrix +1 (for constraint on pixel 1)
    C = lil_matrix((NumEq, npix))
    for idx in range(k):
        C[i[idx], j[idx]] = s[idx]
    
    C = C.tocsr()
    
    # Solve system
    z = np.linalg.lstsq(C.toarray(), d, rcond=None)[0]
    
    # Copy vector of height values back into appropriate pixel positions
    height = np.full(mask.shape, np.nan)
    count = 0
    for row in range(rows):
        for col in range(cols):
            if mask[row, col]:
                count += 1
                height[row, col] = z[count - 1]
    
    # Unpad
    height = height[1:rows-1, 1:cols-1]
    
    return height
