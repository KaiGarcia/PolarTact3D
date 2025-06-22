import numpy as np
from scipy.fftpack import dct, idct

def dct2(a: np.ndarray) -> np.ndarray:
    """2D Discrete Cosine Transform (DCT-II) with orthonormal normalization."""
    return dct(dct(a, type=2, norm='ortho', axis=0),
               type=2, norm='ortho', axis=1)

def idct2(a: np.ndarray) -> np.ndarray:
    """2D Inverse Discrete Cosine Transform (IDCT-II) with orthonormal normalization."""
    return idct(idct(a, type=2, norm='ortho', axis=0),
                type=2, norm='ortho', axis=1)

def dct_poisson(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Solve ∇² z = div([p, q]) with Neumann boundary conditions via DCT.
    p, q : M×N gradient fields (∂z/∂x, ∂z/∂y)
    Returns height map z of shape M×N, offset so min(z)=0.
    """
    M, N = p.shape

    # 1) Compute divergence f = ∂p/∂x + ∂q/∂y using central differences
    #    with replicated-edge (Neumann) boundary conditions
    px = 0.5 * (
        np.vstack((p[1:, :],   p[-1:, :])) -
        np.vstack((p[0:1, :],  p[:-1, :] ))
    )
    qy = 0.5 * (
        np.hstack((q[:, 1:],   q[:, -1:])) -
        np.hstack((q[:, 0:1],  q[:, :-1]))
    )
    f = px + qy

    # 2) Build boundary-term matrix b
    b = np.zeros_like(p)
    # edges (excluding corners)
    b[0,    1:-1] = -p[0,    1:-1]
    b[-1,   1:-1] =  p[-1,   1:-1]
    b[1:-1, 0   ] = -q[1:-1, 0   ]
    b[1:-1,-1  ] =  q[1:-1,-1  ]
    # corners
    sqrt2 = np.sqrt(2)
    b[0,   0   ] = (1/sqrt2)*(-p[0,0]   - q[0,0])
    b[0,   -1  ] = (1/sqrt2)*(-p[0,-1]  + q[0,-1])
    b[-1,  -1  ] = (1/sqrt2)*( p[-1,-1] + q[-1,-1])
    b[-1,  0   ] = (1/sqrt2)*( p[-1,0]  - q[-1,0])

    # 3) Modify f near boundaries (Eqs. 53–54 in Queau et al.)
    f[0,    1:-1] -= b[0,    1:-1]
    f[-1,   1:-1] -= b[-1,   1:-1]
    f[1:-1, 0   ] -= b[1:-1, 0   ]
    f[1:-1,-1  ] -= b[1:-1,-1  ]

    f[0,   -1  ] -= sqrt2 * b[0,   -1  ]
    f[-1,  -1  ] -= sqrt2 * b[-1,  -1  ]
    f[-1,   0  ] -= sqrt2 * b[-1,   0  ]
    f[0,    0  ] -= sqrt2 * b[0,    0  ]

    # 4) DCT of the right-hand side
    f_cos = dct2(f)

    # 5) Build eigenvalue denominator
    x = np.arange(N)
    y = np.arange(M)
    X, Y = np.meshgrid(x, y)
    denom = 4 * ((np.sin(0.5*np.pi*X/N))**2 + (np.sin(0.5*np.pi*Y/M))**2)
    # avoid division by zero at (0,0)
    denom[0,0] = np.finfo(float).eps

    # 6) Solve in DCT domain and invert
    z_bar = -f_cos / denom
    z = idct2(z_bar)

    # 7) Offset so minimum is zero
    z -= z.min()
    return z