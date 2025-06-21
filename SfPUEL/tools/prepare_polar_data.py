import os
from PIL import Image
import numpy as np
import glob
from tqdm import tqdm

# Constants — resize to width 512, preserving original 6:5 aspect ratio
target_width = 512
target_height = int((2048 / 2448) * target_width)  # ≈ 427
target_size = (target_width, target_height)

# Directory setup
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
target_base = os.path.join(root_dir, "SfPUEL_test_data", "real1")

# Canonical suffixes map
canonical_suffixes = {
    "_000.png": "pol000",
    "_045.png": "pol045",
    "_090.png": "pol090",
    "_135.png": "pol135",
    "_0.png": "pol000", "_0°.png": "pol000",
    "_45.png": "pol045", "_45°.png": "pol045",
    "_90.png": "pol090", "_90°.png": "pol090",
    "_135.png": "pol135", "_135°.png": "pol135"
}

# Collect image paths with matching suffixes
image_paths = []
for suffix in canonical_suffixes:
    image_paths.extend(glob.glob(os.path.join(root_dir, f"*{suffix}")))

# Group images by base name with normalized suffixes
grouped = {}
for path in image_paths:
    for variant_suffix, canonical_dir in canonical_suffixes.items():
        if path.endswith(variant_suffix):
            base = os.path.basename(path).replace(variant_suffix, "")
            grouped.setdefault(base, {})[canonical_dir] = path

# Process each complete set
print("📦 Preparing datasets...")
for base_name in tqdm(grouped.keys(), desc="Processing sets"):
    imgs = grouped[base_name]

    if set(imgs.keys()) == {"pol000", "pol045", "pol090", "pol135"}:
        # Determine target resize size based on pol000 image
        ref_img = Image.open(imgs["pol000"])
        w_orig, h_orig = ref_img.size
        target_width = 512
        target_height = int((h_orig / w_orig) * target_width)
        target_size = (target_width, target_height)

        resized_imgs = []

        # Process each polarization image
        for angle_dir in ["pol000", "pol045", "pol090", "pol135"]:
            dest_dir = os.path.join(target_base, angle_dir)
            os.makedirs(dest_dir, exist_ok=True)

            src_img = Image.open(imgs[angle_dir]).convert("RGB").resize(target_size)
            dest_path = os.path.join(dest_dir, base_name + ".png")
            src_img.save(dest_path)
            resized_imgs.append(np.array(src_img))
            print(f"✅ Saved resized image to {angle_dir}/")

        # Create averaged image (unpolar)
        avg_img = np.mean(resized_imgs, axis=0).astype(np.uint8)
        unpolar_dir = os.path.join(target_base, "unpolar")
        os.makedirs(unpolar_dir, exist_ok=True)
        Image.fromarray(avg_img).save(os.path.join(unpolar_dir, base_name + ".png"))
        print(f"🟢 Saved averaged image to unpolar/")

        # Copy and resize mask.png to match this image set
        mask_src = os.path.join(root_dir, "mask.png")
        mask_dir = os.path.join(target_base, "mask")
        os.makedirs(mask_dir, exist_ok=True)
        mask_img = Image.open(mask_src).convert("L").resize(target_size)
        mask_img.save(os.path.join(mask_dir, base_name + ".png"))
        print(f"🟡 Copied and resized mask.png to mask/")

        # Copy and resize ballblack.png to match this image set
        normal_src = os.path.join(root_dir, "ballblack.png")
        normal_dir = os.path.join(target_base, "normal")
        os.makedirs(normal_dir, exist_ok=True)
        normal_img = Image.open(normal_src).convert("RGB").resize(target_size)
        normal_img.save(os.path.join(normal_dir, base_name + ".png"))
        print(f"🔵 Copied and resized ballblack.png to normal/")

    else:
        print(f"⚠️ Skipped incomplete set: {base_name} (found: {list(imgs.keys())})")
