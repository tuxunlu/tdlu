#!/usr/bin/env python3
"""
remove_marker.py

This script processes PNG16 mammogram images to remove letter markers by extracting
and masking the largest contour from the binarized image, while preserving the full
16-bit depth of the input.

Inspired by:
https://www.kaggle.com/code/davidbroberts/mammography-remove-letter-markers/notebook

Usage:
    python remove_marker.py --input_dir <path_to_input_pngs> --output_dir <path_to_save_processed_images> 

Optional arguments:
    --thresh_val        Threshold value for binary thresholding (default=200)
"""

import cv2
import numpy as np
import os
import sys
import argparse


def remove_marker_from_image(img: np.ndarray, thresh_val: int = 200) -> np.ndarray:
    """
    Removes letter markers from a 16-bit mammogram image using the largest contour technique,
    preserving the original bit-depth.

    Parameters:
      img (numpy.ndarray): Input mammogram image (uint16 or uint8), single- or multi-channel.
      thresh_val (int): Threshold value for binary thresholding (applied in original bit-depth).

    Returns:
      numpy.ndarray: Image with markers removed, same dtype and channels as input.
    """
    # Determine if image is grayscale or color
    if img.ndim == 2:
        # already single-channel
        gray = img
    else:
        # convert color to grayscale (preserve depth)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Choose maxval based on bit-depth
    maxval = int(np.iinfo(gray.dtype).max)

    # Binary threshold to isolate bright regions (markers and breast)
    _, bin_img = cv2.threshold(gray, thresh_val, maxval, cv2.THRESH_BINARY)
    # Convert to 8-bit mask for contour detection
    mask8 = (bin_img > 0).astype(np.uint8) * 255

    # Find external contours in the mask
    contours, _ = cv2.findContours(mask8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print("No contours found in image, returning original.", file=sys.stderr)
        return img

    # Keep only the largest contour (assumed to be the breast region)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a filled mask from the largest contour (8-bit)
    roi_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(roi_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Apply the mask to each channel of the original image
    # bitwise_and supports masking with 8-bit mask and preserves dtype of src
    result = cv2.bitwise_and(img, img, mask=roi_mask)

    return result


def process_directory(input_dir: str, output_dir: str, thresh_val: int = 200) -> None:
    """
    Recursively processes all PNG16 images in the input directory, removes markers,
    and writes results with the original bit-depth to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(".png"):
                input_path = os.path.join(root, filename)
                # Read with unchanged flag to preserve bit-depth (16-bit)
                img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"Error reading image {input_path}", file=sys.stderr)
                    continue
                processed_img = remove_marker_from_image(img, thresh_val)
                # Write back as PNG; dtype preserved by cv2.imwrite
                rel_path = os.path.relpath(root, input_dir)
                out_subdir = os.path.join(output_dir, rel_path)
                os.makedirs(out_subdir, exist_ok=True)
                output_path = os.path.join(out_subdir, filename)
                cv2.imwrite(output_path, processed_img)
                print(f"Processed {input_path} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Remove letter markers from PNG16 mammograms")
    parser.add_argument("--input_dir", required=True,
                        help="Path to input PNG16 images")
    parser.add_argument("--output_dir", required=True,
                        help="Path to save processed images")
    parser.add_argument("--thresh_val", type=int, default=20,
                        help="Threshold value (in original bit-depth) for masking")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, args.thresh_val)


if __name__ == "__main__":
    main()
