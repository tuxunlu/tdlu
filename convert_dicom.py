#!/usr/bin/env python3
import os
import sys
import subprocess
import pandas as pd
import csv

def read_subject_data(csv_file):
    """
    Reads subject data from a CSV file with columns 'subject_id', 'Manufacturer' and 'tdlu_count_final'.
    Returns a dictionary mapping subject_id to Manufacturer.
    """
    subject_data = {}
    try:
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                subject_id = row["subject_id"].strip()
                manufacturer = row.get("Manufacturer", "").strip()
                tdlu_count_final = row.get("tdlu_count_final", "").strip()
                subject_data[subject_id] = {"Manufacturer": manufacturer, "tdlu_count_final": tdlu_count_final}
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}", file=sys.stderr)
    return subject_data

def convert_dicom_to_png(dicom_file, output_file, extra_flags=None):
    """
    Uses dcm2img to convert a DICOM file to a PNG image.
    Optionally, extra_flags can be provided to adjust the command-line options.
    """
    try:
        cmd = ["dcm2img"]
        if extra_flags:
            cmd.extend(extra_flags)
        cmd.extend([dicom_file, output_file])
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting {dicom_file}: {e}", file=sys.stderr)

def main():
    # Define directories and CSV file
    input_dir = "/Volumes/PRO-G40/WUSTL. Unmodified mammograms-selected/Batch 1"   # Directory containing the DICOM files
    output_dir = "/Volumes/PRO-G40/WUSTL. Unmodified mammograms-selected/Batch 1_png"     # Directory for the output PNG images
    csv_file = "/Volumes/PRO-G40/WUSTL. Unmodified mammograms-selected/umd_annot_md_TDLU_y2025m03d13.csv"  # CSV file with subject data

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the subject data from CSV (subject_id and Manufacturer)
    subject_data = read_subject_data(csv_file)
    if not subject_data:
        print("No subject data was found in the CSV file. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Iterate over DICOM files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".dcm"):
            # Extract subject_id from filename (assumes filename is in the format "subject_id-num.dcm")
            subject_id = filename.split('-')[0]
            # If subject_id is not found in the CSV, skip the file.
            if subject_id not in subject_data:
                print(f"Subject ID {subject_id} not found in CSV. Skipping {filename}.")
                continue
            if subject_data[subject_id]["tdlu_count_final"] == "N":
                print(f"Subject ID {subject_id} has no TDLU count. Skipping {filename}.")
                continue

            dicom_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_dir, output_filename)

            # Check the Manufacturer from CSV (case insensitive comparison)
            manufacturer = subject_data[subject_id]["Manufacturer"].upper()
            if manufacturer == "GE MEDICAL SYSTEMS":
                flags = ["+on2", "-W"]
            else:
                flags = ["+on2", "-W"]

            print(f"Converting {dicom_path} to {output_path} with flags: {flags}")
            convert_dicom_to_png(dicom_path, output_path, extra_flags=flags)
            print(f"Converted {dicom_path} successfully.")

if __name__ == "__main__":
    main()
