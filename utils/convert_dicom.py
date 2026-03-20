#!/usr/bin/env python3
import os
import sys
import subprocess
import csv
import pydicom
import numpy as np
import cv2
from pydicom.uid import JPEG2000, JPEG2000Lossless

def _get_gdcm():
    try:
        import gdcm
        return gdcm
    except ImportError:
        return None

def decompress_dicom_with_gdcm(dicom_path):
    """
    Decompress a DICOM file using gdcmconv via the gdcm Python wrapper.
    The decompressed file will overwrite the original.
    """
    gdcm = _get_gdcm()
    if gdcm is None:
        raise RuntimeError("gdcm is required for decompressing JPEG2000 DICOMs; install with conda install gdcm")
    path = os.path.dirname(gdcm.__file__)
    gdcmconv = os.path.join(path, "_gdcm", "gdcmconv")
    # Overwrite the original file with decompressed version
    cmd = [gdcmconv, "--raw", dicom_path, dicom_path]
    subprocess.call(cmd)

def read_subject_data(csv_file):
    """
    Reads subject data from a CSV file with columns 'subject_id', 'Manufacturer', and 'tdlu_density'.
    Returns a dictionary mapping subject_id to its data.
    """
    subject_data = {}
    try:
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename", "").strip()
                subject_id = row["subject_id"].strip()
                manufacturer = row.get("Manufacturer", "").strip()
                tdlu_density = row.get("tdlu_density", "").strip()
                WindowCenter = row.get("WindowCenter", "").strip()
                WindowWidth = row.get("WindowWidth", "").strip()
                PhotometricInterpretation = row.get("PhotometricInterpretation", "").strip()
                subject_data[filename] = {
                    "subject_id": subject_id,
                    "Manufacturer": manufacturer,
                    "tdlu_density": tdlu_density,
                    "WindowCenter": WindowCenter,
                    "WindowWidth": WindowWidth,
                    "PhotometricInterpretation": PhotometricInterpretation
                }
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}", file=sys.stderr)
    return subject_data

def append_missing_tags(dicom_file, tag, value):
    """
    Appends a missing tag to the DICOM file using dcmodify.
    Expects TDLU_DCMTK_ROOT from the environment (set by sourcing tdlu/activate.sh).
    """
    dcmtk_root = os.environ.get(
        "TDLU_DCMTK_ROOT",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dcmtk-3.6.9-linux-x86_64-static"),
    )
    dcmodify_bin = os.path.join(dcmtk_root, "bin", "dcmodify")
    try:
        cmd = [dcmodify_bin, "-i", f"{tag}={value}", dicom_file]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error appending tag {tag} to {dicom_file}: {e}", file=sys.stderr)

def convert_dicom_to_png(dicom_file, output_file, extra_flags=None):
    """
    Uses dcm2img to convert a DICOM file to a PNG image.
    Optionally, extra_flags can be provided to adjust the command-line options.
    Expects DCMDICTPATH and TDLU_DCMTK_ROOT from the environment (set by sourcing tdlu/activate.sh).
    Returns a tuple (True, None) if conversion succeeds,
    otherwise returns (False, error_message) if conversion fails.
    """
    dcmtk_root = os.environ.get(
        "TDLU_DCMTK_ROOT",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dcmtk-3.6.9-linux-x86_64-static"),
    )
    dcm2img_bin = os.path.join(dcmtk_root, "bin", "dcm2img")
    try:
        cmd = [dcm2img_bin]
        if extra_flags:
            cmd.extend(extra_flags)
        cmd.extend([dicom_file, output_file])
        print("dcm2img:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return (True, None)
    except subprocess.CalledProcessError as e:
        error_msg = f"Error converting {dicom_file}: {e}"
        print(error_msg, file=sys.stderr)
        return (False, str(e))


def _load_breastmask_from_mat(mat_path):
    """
    Load BreastMask from a LIBRA MAT file.
    Supports MAT v7 (scipy) and MAT v7.3/HDF5 (h5py).
    """
    try:
        import scipy.io as sio  # type: ignore
        md = sio.loadmat(mat_path, simplify_cells=True)
        if isinstance(md.get("res"), dict) and "BreastMask" in md["res"]:
            return np.asarray(md["res"]["BreastMask"])
        if "BreastMask" in md:
            return np.asarray(md["BreastMask"])
    except Exception:
        pass

    try:
        import h5py  # type: ignore
        with h5py.File(mat_path, "r") as f:
            if "BreastMask" in f:
                return np.asarray(f["BreastMask"][()])
            if "res" in f and "BreastMask" in f["res"]:
                node = f["res"]["BreastMask"][()]
                if hasattr(node, "dtype") and node.dtype == h5py.ref_dtype:
                    ref = node[()] if node.shape == () else node.flat[0]
                    return np.asarray(f[ref][()])
                return np.asarray(node)
    except Exception:
        pass

    raise RuntimeError(f"BreastMask not found in MAT file: {mat_path}")


def _binarize_mask(mask):
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    return (mask > 0).astype(np.uint8) * 255


def get_breast_mask_from_mat(mat_path, target_shape=None):
    """
    Load breast mask from LIBRA MAT file and return a binary uint8 mask (0/255).
    """
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"Mask MAT file not found: {mat_path}")
    breast_mask = _binarize_mask(_load_breastmask_from_mat(mat_path))

    if target_shape is not None:
        h, w = target_shape[:2]
        if breast_mask.shape[0] != h or breast_mask.shape[1] != w:
            if (breast_mask.shape[0], breast_mask.shape[1]) == (w, h):
                breast_mask = cv2.resize(breast_mask.T, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                breast_mask = cv2.resize(breast_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return breast_mask


def _resolve_mask_mat_path(result_images_dir, stem):
    """
    Resolve MAT mask path for one image stem.
    Expected naming: Masks_<stem>.mat
    """
    candidates = [
        os.path.join(result_images_dir, f"Masks_{stem}.mat"),
        os.path.join(result_images_dir, f"{stem}.mat"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def apply_mask_to_image(img, breast_mask, try_flip_if_low=True):
    """
    Apply binary breast mask to an image (2D or 3D). Zeros out pixels outside the mask.
    img and breast_mask must have the same (H, W); img may be (H, W) or (H, W, C).
    Returns (masked_img, mask_used). If try_flip_if_low: when mean intensity inside the
    mask is very low (wrong orientation), retry with left-right flipped mask.
    """
    if img.shape[:2] != breast_mask.shape[:2]:
        raise ValueError(
            f"img shape {img.shape} and mask shape {breast_mask.shape} must match"
        )
    mask_bool = (breast_mask > 0)
    if not np.any(mask_bool):
        raise ValueError("Empty breast mask")
    out = img.copy()
    if out.ndim == 2:
        out[~mask_bool] = 0
    else:
        out[~mask_bool, :] = 0
    mean_inside = float(np.mean(out[mask_bool])) if np.any(mask_bool) else 0.0
    max_img = np.max(img)
    if try_flip_if_low and mean_inside < 1.0 and max_img > 50:
        mask_flip = np.fliplr(breast_mask)
        mask_flip_bool = (mask_flip > 0)
        out_flip = img.copy()
        if out_flip.ndim == 2:
            out_flip[~mask_flip_bool] = 0
        else:
            out_flip[~mask_flip_bool, :] = 0
        mean_flip = float(np.mean(out_flip[mask_flip_bool])) if np.any(mask_flip_bool) else 0.0
        if mean_flip > mean_inside:
            return out_flip, mask_flip
    return out, breast_mask


# Default dcm2img flags for STAMP.
# +Wr ROI parameters are appended per image using MAT mask bbox.
STAMP_DCM2IMG_FLAGS = ["+on2"]


def _bbox_from_mask(mask):
    """
    Compute ROI rectangle (left, top, width, height) from binary mask.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Empty breast mask; cannot compute ROI bbox for +Wr")
    left = int(xs.min())
    top = int(ys.min())
    right = int(xs.max())
    bottom = int(ys.max())
    width = int(right - left + 1)
    height = int(bottom - top + 1)
    return left, top, width, height


def _compute_ww_from_roi(ds, left, top, width, height):
    """
    Compute +Ww center/width from ROI p5/p95 values.
    """
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    h, w = arr.shape[:2]
    l = max(0, int(left))
    t = max(0, int(top))
    r = min(w, l + int(width))
    b = min(h, t + int(height))
    if r <= l or b <= t:
        raise ValueError("Invalid ROI bounds for p5/p95 window computation")

    roi = arr[t:b, l:r]
    roi_vals = roi.reshape(-1)
    # Ignore very bright outliers as requested.
    roi_vals = roi_vals[roi_vals <= 3000]
    # Fallback if all values were filtered out.
    if roi_vals.size == 0:
        roi_vals = roi.reshape(-1)
    p5 = float(np.percentile(roi_vals, 5))
    p95 = float(np.percentile(roi_vals, 95))
    center = (p5 + p95) / 2.0
    win_width = max(1.0, p95 - p5)
    return center, win_width


def process_stamp_with_libra_crop(
    anonymizations_root,
    libra_result_root,
    output_dir,
    limit=None,
    use_windowed_original=False,
):
    """
    Discover DICOMs under anonymizations_root (e.g. Batch1, Batch2, ...), find matching
    LIBRA MAT masks (Masks_<stem>.mat) under libra_result_root/BatchN/Result_Images/,
    If use_windowed_original is False (default), convert each DICOM to PNG with
    convert_dicom_to_png using +Ww derived from ROI p5/p95 (ROI from MAT mask bbox).
    If use_windowed_original is True, use LIBRA *_Windowed_Original.jpg as source image.
    Then apply the breast mask and save masked PNG + mask PNG.
    If limit is set (e.g. 1), stop after that many successful processed scans.
    """
    base_output_dir = (
        os.path.join(output_dir, "windowed_original_source")
        if use_windowed_original
        else output_dir
    )
    processed_dir = os.path.join(base_output_dir, "processed_png")
    masks_dir = os.path.join(base_output_dir, "masks_png")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    processed = 0
    skipped_no_mask = []
    errors = []
    for batch_name in sorted(os.listdir(anonymizations_root)):
        batch_dir = os.path.join(anonymizations_root, batch_name)
        if not os.path.isdir(batch_dir):
            continue
        result_images_dir = os.path.join(
            libra_result_root, batch_name, "Result_Images"
        )
        if not os.path.isdir(result_images_dir):
            continue
        for fn in os.listdir(batch_dir):
            if not fn.lower().endswith(".dcm"):
                continue
            dicom_path = os.path.join(batch_dir, fn)
            stem = os.path.splitext(fn)[0]
            mat_path = _resolve_mask_mat_path(result_images_dir, stem)
            if mat_path is None:
                skipped_no_mask.append(f"{batch_name}/{fn}")
                continue
            out_png_path = os.path.join(processed_dir, f"{stem}.png")
            try:
                if use_windowed_original:
                    src_img_path = os.path.join(result_images_dir, f"{stem}_Windowed_Original.jpg")
                    img = cv2.imread(src_img_path, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        errors.append(f"{fn}: failed to read {src_img_path}")
                        continue
                    breast_mask = get_breast_mask_from_mat(mat_path, target_shape=img.shape[:2])
                    left = top = width = height = None
                    center = ww = None
                else:
                    ds = pydicom.dcmread(dicom_path, force=True)
                    if hasattr(ds, "file_meta") and hasattr(ds.file_meta, "TransferSyntaxUID"):
                        ts_uid = ds.file_meta.TransferSyntaxUID
                        if ts_uid in (JPEG2000, JPEG2000Lossless) or (
                            hasattr(ts_uid, "name") and "JPEG2000" in ts_uid.name
                        ):
                            decompress_dicom_with_gdcm(dicom_path)
                            ds = pydicom.dcmread(dicom_path, force=True)
                    if not hasattr(ds, "Rows") or not hasattr(ds, "Columns"):
                        errors.append(f"{fn}: missing Rows/Columns in DICOM")
                        continue
                    dicom_shape = (int(ds.Rows), int(ds.Columns))
                    breast_mask = get_breast_mask_from_mat(mat_path, target_shape=dicom_shape)
                    left, top, width, height = _bbox_from_mask(breast_mask)
                    center, ww = _compute_ww_from_roi(ds, left, top, width, height)
                    flags = STAMP_DCM2IMG_FLAGS + [
                        "+Ww", f"{center:.3f}", f"{ww:.3f}",
                    ]
                    success, err_msg = convert_dicom_to_png(
                        dicom_path, out_png_path, extra_flags=flags
                    )
                    if not success:
                        errors.append(f"{fn}: dcm2png failed: {err_msg}")
                        continue
                    img = cv2.imread(out_png_path, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        errors.append(f"{fn}: failed to read converted PNG")
                        continue
                    if breast_mask.shape[:2] != img.shape[:2]:
                        breast_mask = cv2.resize(
                            breast_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
                        )
                # MAT mask is the source of truth; keep ROI(+Wr) and mask in the same
                # coordinate system (origin: top-left), so disable auto flip here.
                masked_img, mask_used = apply_mask_to_image(
                    img, breast_mask, try_flip_if_low=False
                )
                cv2.imwrite(out_png_path, masked_img)
                out_mask_png = os.path.join(masks_dir, f"{stem}_mask.png")
                cv2.imwrite(out_mask_png, mask_used)
                if processed == 0 or (limit and processed <= 1):
                    n_pixels_inside = np.sum(mask_used > 0)
                    n_pixels = mask_used.shape[0] * mask_used.shape[1]
                    nonzero = masked_img[masked_img > 0]
                    mask_shape = mask_used.shape[:2]
                    if len(nonzero) > 0:
                        print(
                            f"[debug] {fn}: mask_shape={mask_shape} png_shape={img.shape[:2]} "
                            f"mask_frac={n_pixels_inside / n_pixels:.3f} inside min/max/mean={np.min(nonzero):.2f}/{np.max(nonzero):.2f}/{np.mean(nonzero):.2f}"
                        )
                        if not use_windowed_original:
                            print(
                                f"[debug] {fn}: ROI left={left} top={top} width={width} height={height}"
                            )
                            print(
                                f"[debug] {fn}: +Ww center={center:.3f} width={ww:.3f} (from ROI p5/p95)"
                            )
                    else:
                        print(f"[debug] {fn}: mask_shape={mask_shape} png_shape={img.shape[:2]} mask_frac=0 (no pixels kept)")
                processed += 1
                if limit is not None and processed >= limit:
                    print(f"Processed {processed} (limit={limit}), stopping.", flush=True)
                    break
                elif processed % 50 == 0:
                    print(f"Processed {processed} ...", flush=True)
            except Exception as e:
                errors.append(f"{fn}: {e}")
        if limit is not None and processed >= limit:
            break
    if skipped_no_mask:
        print(f"Skipped (no LIBRA MAT mask): {len(skipped_no_mask)}", file=sys.stderr)
        for s in skipped_no_mask[:20]:
            print(f"  {s}", file=sys.stderr)
        if len(skipped_no_mask) > 20:
            print(f"  ... and {len(skipped_no_mask) - 20} more", file=sys.stderr)
    if errors:
        print(f"Errors: {len(errors)}", file=sys.stderr)
        for e in errors[:30]:
            print(f"  {e}", file=sys.stderr)
    print(
        f"Saved {processed} masked mammograms to {processed_dir} and masks to {masks_dir}"
    )
    return processed


def main():
    # Define directories and CSV file path.
    data_dir = "/fs/nexus-scratch/tuxunlu/git/tdlu/data"
    raw_dir = os.path.join(data_dir, "WUSTL_Unmodified_mammograms_selected")
    output_dir = os.path.join(data_dir, "WUSTL_png")
    csv_file = os.path.join(data_dir, "umd_annot_md_TDLU_y2025m04d11.csv")

    excluded_ids = ["9497"]

    # Create the output directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read subject data from CSV.
    subject_data = read_subject_data(csv_file)
    if not subject_data:
        print("No subject data was found in the CSV file. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Lists to record issues.
    missing_subject_ids = []  # subject_ids missing in CSV.
    conversion_errors = []    # subject_ids that encountered conversion errors.

    # Single walk through all subdirectories.
    for root, dirs, files in os.walk(raw_dir):
        # Skip the output directory to avoid re-processing converted files.
        if os.path.abspath(root).startswith(os.path.abspath(output_dir)):
            continue

        for filename in files:
            # Process only DICOM files.
            if not filename.lower().endswith(".dcm"):
                continue

            dicom_path = os.path.join(root, filename)

            # Attempt to read the DICOM file with pydicom.
            try:
                ds = pydicom.dcmread(dicom_path, force=True)
            except Exception as e:
                print(f"Error reading DICOM file {dicom_path}: {e}", file=sys.stderr)
                continue

            # Check if decompression is needed based on TransferSyntaxUID.
            if hasattr(ds, 'file_meta') and hasattr(ds.file_meta, 'TransferSyntaxUID'):
                ts_uid = ds.file_meta.TransferSyntaxUID
                # Check if the transfer syntax indicates JPEG2000 compression.
                if ts_uid in (JPEG2000, JPEG2000Lossless) or "JPEG2000" in ts_uid.name:
                    print(f"Decompressing {dicom_path}...")
                    decompress_dicom_with_gdcm(dicom_path)
                    # Re-read the dataset after decompression.
                    try:
                        ds = pydicom.dcmread(dicom_path, force=True)
                    except Exception as e:
                        print(f"Error reading decompressed DICOM file {dicom_path}: {e}", file=sys.stderr)
                        continue

            # Extract subject identifiers from the filename.
            subject_id = filename.split('-')[0]

            # Check if the subject_id exists in CSV; skip if not.
            if filename not in subject_data:
                print(f"Subject ID {subject_id} not found in CSV. Skipping {filename}.", file=sys.stderr)
                missing_subject_ids.append(filename)
                continue

            # Check if the subject is excluded.
            if subject_id in excluded_ids:
                print(f"Subject ID {subject_id} is excluded. Skipping {filename}.", file=sys.stderr)
                continue

            # Check for and append missing DICOM tags using pydicom.
            try:
                if "WindowCenter" not in ds:
                    append_missing_tags(dicom_path, "0028,1050", subject_data[filename]["WindowCenter"])
                if "WindowWidth" not in ds:
                    append_missing_tags(dicom_path, "0028,1051", subject_data[filename]["WindowWidth"])
                if "PhotometricInterpretation" not in ds:
                    append_missing_tags(dicom_path, "0028,0004", subject_data[filename]["PhotometricInterpretation"])
            except Exception as e:
                print(f"Error processing tags for {dicom_path}: {e}", file=sys.stderr)
                continue

            # Determine conversion flags based on manufacturer.
            manufacturer = subject_data[filename]["Manufacturer"].upper()
            if manufacturer in ("GE MEDICAL SYSTEMS", "GE"):
                flags = ["+on2", "--sigmoid-function", "--use-window", "1"]
            elif manufacturer in ("FISCHER IMAGING CORPORATION"):
                flags = ["+on2", "--sigmoid-function", "--use-window", "1"]
            else:
                flags = ["+on2", "--min-max-window"]

            output_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_dir, output_filename)

            # print(f"Converting {dicom_path} to {output_path} with flags: {flags}")
            success, error_msg = convert_dicom_to_png(dicom_path, output_path, extra_flags=flags)
            if not success:
                conversion_errors.append(f"{filename}: {error_msg}")
            # else:
            #     print(f"Converted {file_id} successfully.")

    # Report issues.
    if missing_subject_ids:
        print("\nThe following subject_ids were not found in the CSV file:")
        print(", ".join(missing_subject_ids))
    
    if conversion_errors:
        print("\nThe following subject_ids encountered conversion errors:")
        print("\n".join(conversion_errors))
    else:
        print("\nNo conversion errors were encountered.")

def main_stamp(
    anonymizations_dir=None,
    libra_dir=None,
    output_dir=None,
    stamp_root=None,
    limit=None,
    use_windowed_original=False,
):
    """Run STAMP pipeline and apply breast mask; source can be DICOM or LIBRA Windowed_Original."""
    if stamp_root is None:
        # Default: STAMP dataset dir relative to this file (tdlu/utils/ -> tdlu/dataset/STAMP)
        stamp_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dataset",
            "STAMP",
        )
    anonymizations_dir = anonymizations_dir or os.path.join(
        stamp_root, "Anonymizations_2020-Aug-18_09-14-30. PatientAge added"
    )
    libra_dir = libra_dir or os.path.join(
        anonymizations_dir, "LIBRA result images"
    )
    output_dir = output_dir or os.path.join(stamp_root, "breast_cropped_float32")
    return process_stamp_with_libra_crop(
        anonymizations_dir,
        libra_dir,
        output_dir,
        limit=limit,
        use_windowed_original=use_windowed_original,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DICOM conversion and STAMP breast cropping")
    parser.add_argument(
        "--stamp",
        action="store_true",
        help="Run STAMP pipeline: convert DICOM to PNG, apply breast mask, save masked PNG under STAMP",
    )
    parser.add_argument(
        "--stamp-output",
        default=None,
        help="Output directory for STAMP cropped images (default: STAMP/breast_cropped_float32)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only this many scans (for testing)",
    )
    parser.add_argument(
        "--use-windowed-original",
        action="store_true",
        help="Use LIBRA *_Windowed_Original.jpg as processed source image instead of dcm2img output",
    )
    args = parser.parse_args()
    if args.stamp:
        main_stamp(
            output_dir=args.stamp_output,
            limit=args.limit,
            use_windowed_original=args.use_windowed_original,
        )
    else:
        main()
