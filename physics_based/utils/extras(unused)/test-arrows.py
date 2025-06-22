import matplotlib.pyplot as plt
import cv2
import numpy as np

from loadData import load_data
from rho_spec import rho_spec
from plot_utils import visualize_normals_rgb

# === Load Data ===
file_path = '../data/spheres1.png'
data = load_data(file_path)

images = data['images']
angles = data['angles']
mask = data.get('mask', None)

# === Select grayscale image for background (I_0 by default) ===
background_img = images[0]
if background_img.max() > 1:
    background_img = background_img.astype(np.float32) / 255.0

# === Get DoLP and AoLP ===
if 'dolp' in data and 'aolp' in data:
    dolp = data['dolp']
    aolp = data['aolp']
else:
    import polanalyser as pa
    stokes = pa.calcStokes(images, np.deg2rad(angles))
    dolp = pa.cvtStokesToDoLP(stokes)
    aolp = pa.cvtStokesToAoLP(stokes)

# === Estimate Zenith Angle (theta) from DoLP ===
n = 1.5
theta = rho_spec(dolp, n)

# === Compute Surface Normals ===
def surface_normal_from_zenith_azimuth(zenith, azimuth):
    """
    Calculate surface normals from zenith and azimuth,
    adjusted to match the synthetic sphere’s color mapping:
      - Red channel: x = sin(θ)*cos(adjusted_azimuth)
      - Green channel: y = sin(θ)*sin(adjusted_azimuth)
      - Blue channel: z = cos(θ)
    """
    # Try subtracting pi/2 to adjust azimuth
    adjusted_azimuth = azimuth - np.pi / 2

    x = np.sin(zenith) * np.cos(adjusted_azimuth)
    y = np.sin(zenith) * np.sin(adjusted_azimuth)
    z = np.cos(zenith)

    normals = np.stack([x, -y, z], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8

    return normals

# Get the surface normals for the entire image
normals = surface_normal_from_zenith_azimuth(theta, aolp)
if mask is not None:
    normals[mask == 0] = 0

# === Sample arrows for plotting ===
step = 40  # spacing between arrows
scale = 100  # scaling factor for arrow length

H, W = background_img.shape
Y, X = np.mgrid[0:H, 0:W]

# Sample every `step` pixels
X_s = X[::step, ::step]
Y_s = Y[::step, ::step]

# Extract x and y components of the normal (arrows' directions)
U = normals[::step, ::step, 0]  # x component
V = -normals[::step, ::step, 1]  # y component (flip for image coordinates)
# We can keep the z-component for visual debugging or additional 3D visualizations
Z = normals[::step, ::step, 2]  # z component (if needed)

# Compute vector magnitude to debug
mag = np.sqrt(U**2 + V**2)
print(f"Arrow magnitude range: min={mag.min()}, max={mag.max()}, mean={mag.mean()}")

# Normalize U, V to avoid zero-length arrows
epsilon = 1e-6
U_norm = U / (mag + epsilon)
V_norm = V / (mag + epsilon)

# === Plot overlay on image ===
plt.figure(figsize=(10, 8))
plt.imshow(background_img, cmap='gray')
plt.quiver(
    X_s, Y_s, U_norm, V_norm,
    color='red', scale=scale, scale_units='width',  # Keep scale consistent
    angles='xy', width=0.003, headwidth=3
)
plt.title('Surface Normals (Arrows) Overlayed on Input Image')
plt.axis('off')
plt.tight_layout()
plt.show()
