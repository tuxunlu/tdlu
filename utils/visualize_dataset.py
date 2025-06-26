import os
import matplotlib.pyplot as plt
from collections import Counter
import torchvision.transforms as transforms
from data.TDLUDataset import TDLUDataset

def visualize_class_distribution(dataset):
    """
    Visualizes the distribution of binned classes using the precomputed label_map.
    This avoids loading each image, resulting in a much faster execution.
    """
    label_list = []
    # Iterate over image filenames and use the label_map to retrieve the label.
    for img_file in dataset.image_files:
        subject_id = img_file.split("-")[0]
        if subject_id in dataset.label_map:
            label_list.append(dataset.label_map[subject_id])
    
    # Count the frequency of each label using collections.Counter.
    label_counts = Counter(label_list)
    print("Label Counts:", label_counts)
    
    # For plotting, sort the bins.
    bins = sorted(label_counts.keys())
    counts = [label_counts[b] for b in bins]
    
    # Plot the class distribution.
    plt.figure(figsize=(10, 6))
    plt.bar(bins, counts)
    plt.xlabel("Class (Bin)")
    plt.ylabel("Count")
    plt.title("Class Distribution in Dataset")
    plt.xticks(bins)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # Update these paths and parameters as needed.
    image_dir = '/fs/nexus-scratch/tuxunlu/git/tdlu/WUSTL_Unmodified_mammograms_selected/WUSTL_png_minmax'
    csv_path = '/fs/nexus-scratch/tuxunlu/git/tdlu/WUSTL_Unmodified_mammograms_selected/umd_annot_md_TDLU_y2025m03d13.csv'
    
    # Specify the number of bins.
    num_bins = 4  # For example: 1 bin reserved for zero values + 3 bins for non-zero values.
    target = "tdlu_density"  # Replace with your CSV target column name.
    
    # Use a simple transform to convert images to tensor (not used in visualization).
    transform = transforms.ToTensor()

    # Create the dataset instance.
    dataset = TDLUDataset(
        image_dir=image_dir,
        csv_path=csv_path,
        num_bins=num_bins,
        transform=transform,
        augment=False,            # No augmentation needed for visualization.
        weights_json_path=None,   # Not required for this task.
        target=target
    )
    
    # Visualize the class distribution.
    visualize_class_distribution(dataset)
