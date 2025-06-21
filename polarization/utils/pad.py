import numpy as np

def pad(X):
    """
    Pad a matrix with an additional row and column on each edge.

    Parameters:
        X : np.ndarray
            Input matrix (2D array)

    Returns:
        X2 : np.ndarray
            Padded matrix
    """
    # Determine if X is logical (boolean)
    is_logical = X.dtype == bool

    # Create padded matrix of zeros with appropriate dtype
    X2 = np.zeros((X.shape[0] + 2, X.shape[1] + 2), dtype=bool if is_logical else X.dtype)

    # Copy the original matrix into the center
    X2[1:-1, 1:-1] = X

    return X2
