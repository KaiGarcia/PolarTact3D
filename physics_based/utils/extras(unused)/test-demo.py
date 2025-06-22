import numpy as np
import matplotlib.pyplot as plt
from loadData import load_data
from rho_spec import rho_spec
from surfaceNormal import (
    surface_normal_from_zenith_azimuth,
    compute_surface_normals
)
from plot_utils import (
    show_reference_normal_colors,
    visualize_normals_rgb,
    visualize_aolp,
    visualize_dolp
)


# === Load Data ===
file_path = '../data/spheres1.png'  # or path to a .mat file
data = load_data(file_path)

images = data['images']
angles = data['angles']
mask = data.get('mask', None)

# === Get DoLP and AoLP ===
if 'dolp' in data and 'aolp' in data:
    dolp = data['dolp']
    aolp = data['aolp']
else:
    import polanalyser as pa
    stokes = pa.calcStokes(images, np.deg2rad(angles))
    dolp = pa.cvtStokesToDoLP(stokes)
    aolp = pa.cvtStokesToAoLP(stokes)

# === Refractive Index ===
n = 1.5

# Estimate zenith angle (theta) from DoLP
theta = rho_spec(dolp, n)  # or rho_diffuse(dolp, n)


# === Compute Surface Normals ===
normals = surface_normal_from_zenith_azimuth(theta, aolp)
if mask is not None:
    normals[mask == 0] = 0

# === Visualize Normal Map ===
visualize_normals_rgb(normals)

# === Overlay Quiver Plot on Image ===
# Use the first input polarization image as the background
background = images[0]  # shape: (H, W), or (H, W, 3) if RGB
H, W = background.shape[:2]

# Arrow sampling
k = 55  # spacing in pixels
y_coords = np.arange(0, H, k)
x_coords = np.arange(0, W, k)
x_grid, y_grid = np.meshgrid(x_coords, y_coords)
x_grid = x_grid + 0.5
y_grid = y_grid + 0.5

# Extract x and y components of normals
nx = normals[y_coords[:, None], x_coords, 0]
ny = normals[y_coords[:, None], x_coords, 1]

# If mask is present, mask out arrows where mask = 0
if mask is not None:
    mask_sampled = mask[y_coords[:, None], x_coords]
    nx[mask_sampled == 0] = 0
    ny[mask_sampled == 0] = 0

# Plot
plt.figure(figsize=(6, 6))
plt.imshow(background, cmap='gray')  # Flip image to match quiver
plt.quiver(x_grid, y_grid, nx, -ny, color='red', scale=25, headwidth=3)
plt.axis('off')
plt.title("Surface Normals Overlaid on Image")
plt.show()
