import pandas as pd
import matplotlib.pyplot as plt

# Specify the path to your CSV file
csv_path = "/fs/nexus-scratch/tuxunlu/git/tdlu/WUSTL. Unmodified mammograms-selected/umd_annot_md_TDLU_y2025m03d13.csv"  # Replace with your actual CSV path

# Load the CSV into a DataFrame
df = pd.read_csv(csv_path)

# Filter out rows where tdlu_density is "N" or "0"
df_filtered = df[(df["tdlu_density"] != "N") & (df["tdlu_density"] != "0")].copy()

# Convert tdlu_density column from string to float
df_filtered["tdlu_density"] = df_filtered["tdlu_density"].astype(float)

# Plot the histogram of the tdlu_density column
plt.figure(figsize=(10, 6))
plt.hist(df_filtered["tdlu_density"], bins=30, color='blue', edgecolor='black', alpha=0.7)
plt.title("Histogram of TDLU Density (Excluding 'N' and Zeros)")
plt.xlabel("TDLU Density")
plt.ylabel("Frequency")
plt.savefig("tdlu_density_histogram.png")  # Save the histogram as an image
