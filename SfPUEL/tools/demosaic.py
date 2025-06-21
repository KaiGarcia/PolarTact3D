import cv2
import polanalyser as pa
import numpy as np
import os

# --- Load grayscale polarized image ---
image_path = "pdms_airtag2.png"
raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if raw is None:
    raise ValueError("Image not found or path is incorrect.")

# --- Demosaic using polanalyser ---
pol_imgs = pa.demosaicing(raw, pa.COLOR_PolarMono)

# --- Save images with suffixes ---
angles = ["000", "045", "090", "135"]

# Get base filename without extension
base_name = os.path.splitext(os.path.basename(image_path))[0]
output_dir = os.path.dirname(image_path)  # Save next to input

for img, angle in zip(pol_imgs, angles):
    out_path = os.path.join(output_dir, f"{base_name}_{angle}.png")
    norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(out_path, norm_img)
    print(f"Saved: {out_path}")
