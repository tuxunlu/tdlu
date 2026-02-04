#!/usr/bin/env python3
"""
extract_breastmask_to_npy_and_png.py

Recursively scan a folder for LIBRA-style .mat files and export BreastMask
as both .npy and .png. Supports MATLAB v7 (scipy.io.loadmat) and v7.3 (HDF5 via h5py).

Usage:
    python extract_breastmask_to_npy_and_png.py \
        --input_dir /path/to/LIBRA_Masks \
        --output_dir /path/to/LIBRA_BreastMasks_out
"""

import os
import argparse
import numpy as np
import cv2


def _load_breastmask_from_mat(mat_path: str) -> np.ndarray:
    """
    Load BreastMask from a .mat file. Tries:
      1) res.BreastMask (MAT v7)
      2) BreastMask     (top-level)
      3) v7.3 via h5py: /res/BreastMask or /BreastMask (following object refs)
    Returns a NumPy array (not yet binarized).
    """
    # --- Try SciPy (MAT v7) ---
    try:
        import scipy.io as sio  # type: ignore
        md = sio.loadmat(mat_path, simplify_cells=True)
        if isinstance(md.get("res"), dict) and "BreastMask" in md["res"]:
            return np.asarray(md["res"]["BreastMask"])
        if "BreastMask" in md:
            return np.asarray(md["BreastMask"])
    except Exception:
        pass  # Fall through to h5py path

    # --- Try h5py (MAT v7.3 / HDF5) ---
    try:
        import h5py  # type: ignore
        with h5py.File(mat_path, "r") as f:
            # Direct dataset at root
            if "BreastMask" in f:
                return f["BreastMask"][()]

            # Under 'res' group, possibly stored as an object reference
            if "res" in f:
                res = f["res"]
                if "BreastMask" in res:
                    node = res["BreastMask"][()]  # may be ref or array
                    # If it's a reference array/scalar, resolve it
                    if hasattr(node, "dtype") and node.dtype == h5py.ref_dtype:
                        ref = node[()] if node.shape == () else node.flat[0]
                        return f[ref][()]
                    return node
    except Exception:
        pass

    raise RuntimeError(f"BreastMask not found in {mat_path}")


def _binarize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert mask to uint8 0/1 (foreground > 0).
    """
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.dtype == np.bool_:
        return mask.astype(np.uint8)
    return (mask > 0).astype(np.uint8)


def _save_png(mask01: np.ndarray, out_png: str) -> None:
    """
    Save 0/1 mask as a visible 0/255 uint8 PNG.
    """
    img = (mask01.astype(np.uint8) * 255)
    img = np.ascontiguousarray(img)  # ensure C-contiguous for cv2
    ok = cv2.imwrite(out_png, img)
    if not ok:
        raise IOError(f"cv2.imwrite failed for {out_png}")


def process_directory(input_dir: str, output_dir: str) -> None:
    """
    Walk `input_dir`, find .mat files, extract BreastMask, and save as both .npy and .png
    to `output_dir`. Output names:
        <mat_basename>_BreastMask.npy
        <mat_basename>_BreastMask.png
    """
    os.makedirs(output_dir, exist_ok=True)

    total = 0
    saved = 0
    failures = []

    for root, _, files in os.walk(input_dir):
        for fn in files:
            if fn.lower().endswith(".mat"):
                total += 1
                fpath = os.path.join(root, fn)
                base = os.path.splitext(fn)[0]
                out_npy = os.path.join(output_dir, f"{base}_BreastMask.npy")
                out_png = os.path.join(output_dir, f"{base}_BreastMask.png")
                try:
                    mask = _load_breastmask_from_mat(fpath)
                    mask01 = _binarize_mask(mask)
                    # Save .npy
                    np.save(out_npy, mask01)
                    # Save .png (0/255 grayscale)
                    _save_png(mask01, out_png)
                    saved += 1
                    h, w = mask01.shape[:2]
                    ones = int(mask01.sum())
                    print(f"[OK] {fn} -> {out_npy}, {out_png} | shape={h}x{w} | ones={ones}")
                except Exception as e:
                    failures.append((fpath, str(e)))
                    print(f"[FAIL] {fn}: {e}")

    print(f"\nDone. Scanned {total} .mat | Saved {saved} pairs (.npy + .png) | Failed {len(failures)}")
    if failures:
        print("Examples of failures:")
        for p, msg in failures[:5]:
            print(f" - {os.path.basename(p)} -> {msg}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract BreastMask from LIBRA .mat files and save as both .npy and .png"
    )
    parser.add_argument("--input_dir", required=True, help="Folder that contains .mat files (recursively searched)")
    parser.add_argument("--output_dir", required=True, help="Folder to save exported .npy and .png masks")
    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
