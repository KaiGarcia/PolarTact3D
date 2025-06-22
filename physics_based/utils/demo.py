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

"""
# === Print Sample Normals ===
print("Sample surface normal vectors:")
for i in range(0, normals.shape[0], 100):
    for j in range(0, normals.shape[1], 100):
        print(f"normal[{i},{j}] = {normals[i, j]}")
   

print("Theta: min =", theta.min(), "max =", theta.max(), "mean =", theta.mean())
print("Azimuth: min =", aolp.min(), "max =", aolp.max(), "mean =", aolp.mean())
"""

# === Visualize ===
#visualize_dolp(dolp)
#visualize_aolp(aolp)


visualize_normals_rgb(normals)
#show_reference_normal_colors()
