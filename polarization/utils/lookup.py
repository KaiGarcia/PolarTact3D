import numpy as np
import scipy.io as sio
import cv2
import os
from scipy.interpolate import LinearNDInterpolator
import polanalyser as pa
from concurrent.futures import ThreadPoolExecutor
import tqdm
import matplotlib.pyplot as plt


# === Load Data Function ===
def load_data(file_path):
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

        I_0, I_45, I_90, I_135 = pa.demosaicing(raw_image, pa.COLOR_PolarMono)
        angles = np.array([0, 45, 90, 135])
        image_list = [I_0, I_45, I_90, I_135]
        angles_rad = np.deg2rad(angles)
        img_stokes = pa.calcStokes(image_list, angles_rad)

        img_s0, img_s1, img_s2 = cv2.split(img_stokes)
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


# === Normal Computation from Polar Angles ===
def surface_normal_from_zenith_azimuth(zenith, azimuth):
    x = np.sin(zenith) * np.cos(azimuth)
    y = np.sin(zenith) * np.sin(azimuth)
    z = np.cos(zenith)
    return np.array([x, y, z], dtype=np.float32)


# === Create Lookup Table ===
def create_lookup_table(dolp, aolp, num_samples=10000):
    h, w = dolp.shape
    lookup_table = {}

    indices = np.random.choice(h * w, size=num_samples, replace=False)
    for idx in indices:
        i = idx // w
        j = idx % w
        d = dolp[i, j]
        a = aolp[i, j]

        if np.isnan(d) or np.isnan(a) or d < 0 or d > 1:
            continue

        try:
            zenith = np.arccos(d)  # Approximate
            azimuth = a
            normal = surface_normal_from_zenith_azimuth(zenith, azimuth)
            lookup_table[(d, a)] = normal
        except:
            continue

    print(f"Lookup table generated with {len(lookup_table)} points.")
    return lookup_table


# === Interpolate Surface Normal ===
def create_interpolator(lookup_table):
    keys = np.array(list(lookup_table.keys()))
    values = np.array(list(lookup_table.values()))
    return LinearNDInterpolator(keys, values)


# === Process a Single Pixel ===
def process_pixel(i, j, dolp_image, aolp_image, interpolator):
    dolp_query = dolp_image[i, j]
    aolp_query = aolp_image[i, j]

    try:
        normal = interpolator((dolp_query, aolp_query))
    except Exception:
        normal = None

    if (
        normal is None or
        not isinstance(normal, np.ndarray) or
        normal.shape != (3,) or
        np.any(np.isnan(normal))
    ):
        normal = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    return (i, j, normal)


# === Generate Normals from Image ===
def get_surface_normals_from_image(image, interpolator):
    dolp_image = image['dolp']
    aolp_image = image['aolp']

    h, w = dolp_image.shape
    surface_normals = np.zeros((h, w, 3), dtype=np.float32)

    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(h):
            for j in range(w):
                futures.append(executor.submit(process_pixel, i, j, dolp_image, aolp_image, interpolator))

        for future in tqdm.tqdm(futures, desc="Processing Pixels", unit="pixel"):
            i, j, normal = future.result()
            surface_normals[i, j] = normal

    print("Surface normals computed.")
    return surface_normals


# === Visualize Normals as RGB ===
def visualize_normals_rgb(normals):
    normals = np.nan_to_num(normals)
    normals = (normals + 1.0) / 2.0  # Normalize to [0, 1]
    normals = (normals * 255).clip(0, 255).astype(np.uint8)
    return normals


# === Main ===
if __name__ == "__main__":
    file_path = "../data/spheres1.png"  # Replace with your image path

    data = load_data(file_path)
    dolp = data['dolp']
    aolp = data['aolp']

    print(f"Loaded image data: {data.keys()}")
    print(f"DoLP shape: {dolp.shape}, AoLP shape: {aolp.shape}")

    lookup_table = create_lookup_table(dolp, aolp, num_samples=10000)
    interpolator = create_interpolator(lookup_table)
    normals = get_surface_normals_from_image(data, interpolator)

    rgb_image = visualize_normals_rgb(normals)
    print("RGB image dtype:", rgb_image.dtype)
    print("RGB image shape:", rgb_image.shape)

    plt.imshow(rgb_image)
    plt.title("Surface Normals Visualized (RGB)")
    plt.axis('off')
    plt.show()

    # Save image
    out_path = "output_surface_normals_rgb.png"
    cv2.imwrite(out_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    print(f"Saved surface normal visualization to: {out_path}")
