import numpy as np
import scipy.sparse
import scipy.sparse.linalg

def hfpol(n, Iun_est, phi_est_combined, s, mask, is_spec, spec):
    # n is a scalar refractive index, so no need to unpack shape
    # Iun_est, phi_est_combined, s, mask, is_spec, spec are used appropriately
    mask = mask.astype(np.bool_)
    is_spec = is_spec.astype(np.bool_)

    h, w = mask.shape  # Only use the shape of the mask for height and width
    pad = lambda x: np.pad(x, ((1, 1), (1, 1)), mode='constant')
    mask = pad(mask)
    is_spec = pad(is_spec)
    if spec is not None:
        spec = pad(spec)

    # Setup for pixel indices
    idx = np.zeros_like(mask, dtype=np.int32)
    idx[mask] = np.arange(1, np.count_nonzero(mask) + 1)
    idx = idx - 1  # 0-based indexing

    total = np.count_nonzero(mask)
    maxeq = 8 * total
    data, row, col, d = [], [], [], []
    eqn = 0

    for i in range(1, h + 1):
        for j in range(1, w + 1):
            if not mask[i, j]:
                continue

            center = idx[i, j]
            nx, ny, nz = s  # Using the light source direction (s) for normals

            add_equation = lambda r, c, val: (row.append(eqn), col.append(c), data.append(val))

            if is_spec[i, j]:
                # Specular constraint
                for di, dj, coeff, deriv in [(-1, 0, 1, 'x'), (1, 0, -1, 'x'), (0, -1, 1, 'y'), (0, 1, -1, 'y')]:
                    ni, nj = i + di, j + dj
                    if mask[ni, nj]:
                        neighbor = idx[ni, nj]
                        if deriv == 'x':
                            val = coeff / 2
                            add_equation(eqn, neighbor, val)
                        elif deriv == 'y':
                            val = coeff / 2
                            add_equation(eqn, neighbor, val)
                add_equation(eqn, center, 0.0)
                if phi_est_combined is not None:
                    ax = -np.cos(phi_est_combined[i, j])
                    ay = -np.sin(phi_est_combined[i, j])
                    val = ax * nx + ay * ny
                    d.append(val)
                else:
                    d.append(-nx if eqn % 2 == 0 else -ny)
                eqn += 1
            else:
                # Diffuse constraint
                for (di, dj, coeff) in [(-1, 0, 1), (1, 0, -1)]:
                    if mask[i + di, j + dj]:
                        neighbor = idx[i + di, j + dj]
                        add_equation(eqn, neighbor, coeff / 2 * nx)
                for (di, dj, coeff) in [(0, -1, 1), (0, 1, -1)]:
                    if mask[i + di, j + dj]:
                        neighbor = idx[i + di, j + dj]
                        add_equation(eqn, neighbor, coeff / 2 * ny)
                add_equation(eqn, center, 0.0)
                d.append(nz)
                eqn += 1

    # Sparse matrix setup
    C = scipy.sparse.csr_matrix((data, (row, col)), shape=(eqn, total))
    d = np.array(d)

    # Solve least squares problem
    z = scipy.sparse.linalg.lsqr(C, d)[0]

    # Map solution to full image
    Z = np.zeros_like(mask, dtype=np.float64)
    Z[mask] = z
    Z = Z[1:-1, 1:-1]  # Remove padding
    return Z
