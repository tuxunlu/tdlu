#!/usr/bin/env python3
import os
import sys
import subprocess
import csv
import gdcm
import pydicom
from pydicom.uid import JPEG2000, JPEG2000Lossless

def decompress_dicom_with_gdcm(dicom_path):
    """
    Decompress a DICOM file using gdcmconv via the gdcm Python wrapper.
    The decompressed file will overwrite the original.
    """
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
    """
    try:
        cmd = [
            "/fs/nexus-scratch/tuxunlu/git/tdlu/dcmtk-3.6.9-linux-x86_64-static/bin/dcmodify",
            "-i", f"{tag}={value}", dicom_file
        ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error appending tag {tag} to {dicom_file}: {e}", file=sys.stderr)

def convert_dicom_to_png(dicom_file, output_file, extra_flags=None):
    """
    Uses dcm2img to convert a DICOM file to a PNG image.
    Optionally, extra_flags can be provided to adjust the command-line options.
    Returns a tuple (True, None) if conversion succeeds,
    otherwise returns (False, error_message) if conversion fails.
    """
    try:
        cmd = ["/fs/nexus-scratch/tuxunlu/git/tdlu/dcmtk-3.6.9-linux-x86_64-static/bin/dcm2img"]
        if extra_flags:
            cmd.extend(extra_flags)
        cmd.extend([dicom_file, output_file])
        subprocess.run(cmd, check=True)
        return (True, None)
    except subprocess.CalledProcessError as e:
        error_msg = f"Error converting {dicom_file}: {e}"
        print(error_msg, file=sys.stderr)
        return (False, str(e))

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
            if manufacturer in ("GE MEDICAL SYSTEMS"):
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

if __name__ == "__main__":
    main()
