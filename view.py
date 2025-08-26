import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os

folder = "/beacon-scratch/tuxunlu/git/tdlu/dataset/LIBRA_Masks"
save = "/beacon-scratch/tuxunlu/git/tdlu/dataset/LIBRA_Masks_npy"

mat_files = [f for f in os.listdir(folder) if f.endswith(".mat")]

for fname in mat_files:
    print(f"Processing {fname}")
    fpath = os.path.join(folder, fname)

    # Load .mat file
    mat = sio.loadmat(fpath, squeeze_me=True, struct_as_record=False)

    # Extract the struct
    res = mat['res']

    dense_mask = res.DenseMask.astype(bool)

    with open(f'{save}/{fname.replace("mat", "npy")}', 'wb') as f:
        np.save(f, dense_mask)