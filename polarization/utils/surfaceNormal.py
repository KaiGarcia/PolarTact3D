import numpy as np
import cv2
import matplotlib.pyplot as plt
import polanalyser as pa
from rho_spec import rho_spec
from rho_diffuse import rho_diffuse

def compute_surface_normals(images, angles, specdiff, mask=None, n=1.5):
    """
    Compute surface normals from polarization images.

    Args:
        images (list or np.ndarray): List of intensity images at different polarization angles.
        angles (np.ndarray): Array of polarization angles.
        mask (np.ndarray, optional): Mask to apply to the images. Defaults to None.
        n (float, optional): Refractive index. Defaults to 1.5.

    Returns:
        np.ndarray: Surface normals.
    """
    # Compute the Stokes parameters
    img_stokes = pa.calcStokes(images, np.deg2rad(angles))

    # Split the Stokes parameters into individual components
    S0, S1, S2 = cv2.split(img_stokes)

    # Compute Degree of Linear Polarization (DoLP) and Angle of Linear Polarization (AoLP)
    dolp = pa.calcDoLP(S0, S1, S2)
    aolp = pa.calcAoLP(S1, S2)

    # Estimate the zenith angle (theta) from DoLP
    n = 1.5  # Index of refraction
    if specdiff == spec:
        theta = rho_spec(dolp, n)  
    elif specdiff == diff:
        theta = rho_diffuse(dolp, n)
    else: 
        theta = np.arccos(np.sqrt((dolp - dolp.min()) / (dolp.max() - dolp.min())))

    # Compute surface normals
    normals = surface_normal_from_zenith_azimuth(theta, aolp)

    if mask is not None:
        normals[mask == 0] = 0

    return normals

def surface_normal_from_zenith_azimuth(zenith, azimuth):
    """
    Calculate surface normals from zenith and azimuth,
    adjusted to match the synthetic sphere’s color mapping:
      - Red channel: x = sin(θ)*cos(adjusted_azimuth)
      - Green channel: y = sin(θ)*sin(adjusted_azimuth)
      - Blue channel: z = cos(θ)
    """
    # Try subtracting pi/2
    adjusted_azimuth = azimuth - np.pi / 2

    x = np.sin(zenith) * np.cos(adjusted_azimuth)
    y = np.sin(zenith) * np.sin(adjusted_azimuth)
    z = np.cos(zenith)

    normals = np.stack([x, -y, z], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8

    # Visualize normals with proper RGB mapping
    normals_rgb = (normals + 1) / 2  # Ensure range is between 0 and 1
    normals_rgb = np.clip(normals_rgb, 0, 1)  # Clip values to [0, 1] range

    # Print the range of RGB values to check if the output is reasonable
    # print("RGB Normalized min:", normals_rgb.min())
    # print("RGB Normalized max:", normals_rgb.max())
    return normals



