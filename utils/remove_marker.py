#!/usr/bin/env python3
"""
remove_marker.py

This script processes PNG mammogram images to remove letter markers by extracting
the largest contour from the binarized image and using that contour as a mask.
This approach assumes that the largest contour corresponds to the main region of 
interest (i.e. the breast) while excluding extraneous markers.

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

def remove_marker_from_image(img, thresh_val=200):
    """
    Removes letter marker from a mammogram image using the largest contour technique.

    Steps:
      - Convert the image to grayscale.
      - Apply binary thresholding to create a binary image.
      - Find contours in the binary image.
      - Keep only the largest contour (assumed to be the main region of interest).
      - Create a mask from the largest contour.
      - Use bitwise_and to extract the main region from the original image.

    Parameters:
      img (numpy.ndarray): Input mammogram image in BGR format.
      thresh_val (int): Threshold value for binary thresholding.
      
    Returns:
      numpy.ndarray: Image with markers removed.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding to isolate bright regions
    _, bin_img = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    
    # Find external contours in the binarized image
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print("No contours found in image, returning original.", file=sys.stderr)
        return img

    # Keep only the largest contour (assumed to be the breast region)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a mask using the largest contour
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)
    
    # Use bitwise_and to mask the original image, keeping only the main region
    result = cv2.bitwise_and(img, img, mask=mask)
    
    return result

def process_directory(input_dir, output_dir, thresh_val=200):
    """
    Recursively processes all PNG images in the input directory using the marker removal
    technique and saves the processed images to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Walk through the input directory and process each PNG file
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(".png"):
                input_path = os.path.join(root, filename)
                img = cv2.imread(input_path)
                if img is None:
                    print(f"Error reading image {input_path}", file=sys.stderr)
                    continue
                processed_img = remove_marker_from_image(img, thresh_val)
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, processed_img)
                print(f"Processed {input_path} -> {output_path}")

def main():
    input_dir = "/fs/nexus-scratch/tuxunlu/git/tdlu/WUSTL_Unmodified_mammograms_selected/WUSTL_png_GE+minmax"
    output_dir = "/fs/nexus-scratch/tuxunlu/git/tdlu/WUSTL_Unmodified_mammograms_selected/WUSTL_png_GE+minmax_nomarker"
    thresh_val = 20
    process_directory(input_dir, output_dir, thresh_val)

if __name__ == "__main__":
    main()
