import os
import numpy as np
import scipy.io as sio
import cv2
import polanalyser as pa

def load_data(file_path):
    """
    Load data from a .mat file or a polarization image file.

    Args:
        file_path (str): Path to the .mat or polarization image file.

    Returns:
        dict: Contains images, angles, mask, and spec if a .mat file is provided.
              Contains the demosaiced images, angles, and Stokes parameters if a polarization image file is provided.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")

    _, ext = os.path.splitext(file_path)

    if ext == '.mat':
        data = sio.loadmat(file_path)
        images = data.get('images')
        angles = data.get('angles')
        mask = data.get('mask')
        spec = data.get('spec')

        if images is None or angles is None or mask is None or spec is None:
            raise ValueError("One or more required variables are missing in the .mat file.")

        if angles.ndim == 2 and angles.shape[0] == 1:
            angles = angles.flatten()

        return {'images': images, 'angles': angles, 'mask': mask, 'spec': spec}

    elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
        raw_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if raw_image is None:
            raise ValueError(f"Failed to load image file: '{file_path}'")

        # Demosaic the raw polarization image
        I_0, I_45, I_90, I_135 = pa.demosaicing(raw_image, pa.COLOR_PolarMono)

        # Define the corresponding angles for the polarization images
        angles = np.array([0, 45, 90, 135])

        # Calculate the Stokes vector per-pixel
        image_list = [I_0, I_45, I_90, I_135]
        angles_rad = np.deg2rad(angles)
        img_stokes = pa.calcStokes(image_list, angles_rad)

        # Decompose the Stokes vector into its components
        img_s0, img_s1, img_s2 = cv2.split(img_stokes)

        # Convert the Stokes vector to Intensity, DoLP, and AoLP
        img_intensity = pa.cvtStokesToIntensity(img_stokes)
        img_dolp = pa.cvtStokesToDoLP(img_stokes)
        img_aolp = pa.cvtStokesToAoLP(img_stokes)

        return {
            'images': image_list,
            'angles': angles,
            'stokes': img_stokes,
            'intensity': img_intensity,
            'dolp': img_dolp,
            'aolp': img_aolp
        }

    else:
        raise ValueError(f"Unsupported file extension: '{ext}'")
