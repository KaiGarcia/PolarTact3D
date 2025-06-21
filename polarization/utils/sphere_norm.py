import numpy as np
from PIL import Image

W = H = 512                            # resolution
u = np.linspace(0, W-1, W)
v = np.linspace(0, H-1, H)
x, y = np.meshgrid(u, v)
x = (2*x + 1)/W - 1                    # [-1,1]
y = (2*y + 1)/H - 1
r2 = x**2 + y**2

mask = r2 <= 1.0
z = np.zeros_like(x)
z[mask] = np.sqrt(1.0 - r2[mask])

n = np.stack([x, y, z], axis=-1)       # shape (H,W,3)
rgb = ((n * 0.5 + 0.5) * 255).astype(np.uint8)
rgb[~mask] = 0                         # optional: make outside black

Image.fromarray(rgb).save("sphere_normal_disc.png")