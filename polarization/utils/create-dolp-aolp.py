import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def compute_dolp_aolp(I0, I45, I90, I135):
    # Convert to float for calculation
    I0 = I0.astype(np.float32)
    I45 = I45.astype(np.float32)
    I90 = I90.astype(np.float32)
    I135 = I135.astype(np.float32)
    
    # Compute Stokes parameters
    S0 = 0.5 * (I0 + I90 + I45 + I135)
    S1 = I0 - I90
    S2 = I45 - I135

    # Degree of Linear Polarization (DoLP)
    DoLP = np.sqrt(S1**2 + S2**2) / (S0 + 1e-8)  # Add epsilon to avoid divide by zero
    DoLP = np.clip(DoLP, 0, 1)  # Clip to [0,1] range

    # Angle of Linear Polarization (AoLP), in radians [-π/2, π/2]
    AoLP = 0.5 * np.arctan2(S2, S1)

    return DoLP, AoLP

I0 = cv2.imread('data/teapot_0.png', cv2.IMREAD_GRAYSCALE)
I45 = cv2.imread('data/teapot_45.png', cv2.IMREAD_GRAYSCALE)
I90 = cv2.imread('data/teapot_90.png', cv2.IMREAD_GRAYSCALE)
I135 = cv2.imread('data/teapot_135.png', cv2.IMREAD_GRAYSCALE)

DoLP, AoLP = compute_dolp_aolp(I0, I45, I90, I135)

#visualizer

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# DoLP
axs[0].set_title("DoLP")
im1 = axs[0].imshow(DoLP, cmap='gray', vmax=0.2)
axs[0].set_xticks([])
axs[0].set_yticks([])

divider1 = make_axes_locatable(axs[0])
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax1)

# AoLP
axs[1].set_title("AoLP (radians)")
im2 = axs[1].imshow(AoLP, cmap='hsv')
axs[1].set_xticks([])
axs[1].set_yticks([])

divider2 = make_axes_locatable(axs[1])
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax2)

plt.tight_layout()
plt.show()

