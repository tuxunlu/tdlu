import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

mask_folder = "/beacon-scratch/tuxunlu/git/tdlu/dataset/LIBRA_Masks_npy"
img_folder = "/beacon-scratch/tuxunlu/git/tdlu/dataset/WUSTL_png_nomarker_16"
save_folder = "/beacon-scratch/tuxunlu/git/tdlu/dataset/WUSTL_png_nomarker_16_cropped"

os.makedirs(save_folder, exist_ok=True)

img_files = [f for f in os.listdir(img_folder) if f.endswith(".png")]

for fname in img_files:
    id = fname.replace(".png", "")
    print(f"Processing {fname}")

    mask_path = f"{mask_folder}/Masks_{id}-new.npy"
    img_path = f"{img_folder}/{id}.png"
    save_path = f"{save_folder}/{id}.png"

    # Load mask (boolean)
    mask = np.load(mask_path).astype(bool)

    # Load image and convert to numpy array
    img = np.array(Image.open(img_path))

    # If grayscale: shape (H,W), if RGB: shape (H,W,3)
    if img.ndim == 2:
        cropped = img * mask

    # Option A: Save with black background (no transparency)
    Image.fromarray(cropped).save(save_path, format="PNG")
