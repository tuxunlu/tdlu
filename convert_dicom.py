#!/usr/bin/env python3
import os
import sys
import subprocess
import csv

def read_subject_data(csv_file):
    """
    Reads subject data from a CSV file with columns 'subject_id', 'Manufacturer' and 'tdlu_density'.
    Returns a dictionary mapping subject_id to its data.
    """
    subject_data = {}
    try:
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                subject_id = row["subject_id"].strip()
                manufacturer = row.get("Manufacturer", "").strip()
                tdlu_density = row.get("tdlu_density", "").strip()
                subject_data[subject_id] = {"Manufacturer": manufacturer, "tdlu_density": tdlu_density}
    except Exception as e:
        print(f"Error reading CSV file {csv_file}: {e}", file=sys.stderr)
    return subject_data

def convert_dicom_to_png(dicom_file, output_file, extra_flags=None):
    """
    Uses dcm2img to convert a DICOM file to a PNG image.
    Optionally, extra_flags can be provided to adjust the command-line options.
    """
    try:
        cmd = ["/fs/nexus-scratch/tuxunlu/git/tdlu/dcmtk-3.6.9-linux-x86_64-static/bin/dcm2img"]
        if extra_flags:
            cmd.extend(extra_flags)
        cmd.extend([dicom_file, output_file])
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting {dicom_file}: {e}", file=sys.stderr)

def main():
    # Define the parent directory containing all data folders and the CSV file.
    parent_input_dir = "/fs/nexus-scratch/tuxunlu/git/tdlu/WUSTL. Unmodified mammograms-selected"
    output_dir = os.path.join(parent_input_dir, "WUSTL_png")
    csv_file = os.path.join(parent_input_dir, "umd_annot_md_TDLU_y2025m03d13.csv")

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the subject data from CSV (subject_id, Manufacturer, and tdlu_density)
    subject_data = read_subject_data(csv_file)
    if not subject_data:
        print("No subject data was found in the CSV file. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Missing subject_ids
    missing_subject_ids = []

    # Walk through all subdirectories under parent_input_dir
    for root, dirs, files in os.walk(parent_input_dir):
        # Skip the output directory to avoid processing converted files.
        if os.path.abspath(root).startswith(os.path.abspath(output_dir)):
            continue

        for filename in files:
            if filename.lower().endswith(".dcm"):
                # Extract subject_id from filename (assumes filename is in the format "subject_id-num.dcm")
                subject_id = filename.split('-')[0]
                # if subject_id not in subject_data:
                #     print(f"Subject ID {subject_id} not found in CSV. Skipping {filename}.")
                #     continue
                # if subject_data[subject_id]["tdlu_density"] in {"N", "0"}:
                #     print(f"Subject ID {subject_id} has no or zero TDLU density. Skipping {filename}.")
                #     continue

                dicom_path = os.path.join(root, filename)
                output_filename = os.path.splitext(filename)[0] + ".png"
                output_path = os.path.join(output_dir, output_filename)

                # Check the Manufacturer from CSV (case insensitive comparison)
                # Check if subject_id exists in subject_data
                if subject_id not in subject_data:
                    print(f"Subject ID {subject_id} not found in CSV. Skipping {filename}.")
                    missing_subject_ids.append(subject_id)
                    continue
                manufacturer = subject_data[subject_id]["Manufacturer"].upper()
                # Adjust flags based on manufacturer if needed (currently same flags for all)
                if manufacturer == "GE MEDICAL SYSTEMS":
                    flags = ["+on2", "-W"]
                else:
                    flags = ["+on2", "-W"]

                print(f"Converting {dicom_path} to {output_path} with flags: {flags}")
                convert_dicom_to_png(dicom_path, output_path, extra_flags=flags)
                print(f"Converted {dicom_path} successfully.")

if __name__ == "__main__":
    main()
