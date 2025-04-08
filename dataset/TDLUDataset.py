import os
import json
import random
import pandas as pd
import numpy as np
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms
import torch
from tqdm import tqdm

def bin_value_quantile(value, thresholds):
    """
    Convert a non-zero numeric value into a bin index based on quantile thresholds.
    The returned index is zero-indexed, so that later we can add 1 to shift bins to start at 1.
    Args:
        value (float): The continuous, non-zero value to bin.
        thresholds (list): Sorted list of quantile threshold values computed from non-zero values.
    Returns:
        int: A zero-based bin index in the range [0, len(thresholds)].
    """
    for i, t in enumerate(thresholds):
        if value <= t:
            return i
    return len(thresholds)  # assign to last bin if value exceeds all thresholds

class TDLUDataset(Dataset):
    def __init__(self, image_dir, csv_path, num_bins, transform=None, augment=False, weights_json_path=None, target=None):
        """
        Args:
            image_dir (str): Directory containing .png images.
            csv_path (str): Path to CSV with columns ['subject_id', <target>, 'BreastDensity_avg'].
            transform (callable): Optional transform pipeline.
            augment (bool): If True, apply simple augmentation.
            weights_json_path (str): Optional JSON path for sample weights.
            target (str): Column name in CSV for target labels.
            num_bins (int): Total number of bins (one of which, bin 0, will be reserved for zero values).
                         Non-zero values will be binned into num_bins-1 bins.
        """
        self.image_dir = image_dir
        
        # Define transforms
        if transform is not None:
            self.transform = transform
        else:
            if augment:
                self.transform = transforms.Compose([
                    transforms.Resize((1024, 1024)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize((0.3881, 0.3881, 0.3881), (0.4424, 0.4424, 0.4424))
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.3881, 0.3881, 0.3881), (0.4424, 0.4424, 0.4424))
                ])

        # Read CSV and ensure the subject ID is a string.
        self.csv = pd.read_csv(csv_path)
        self.csv["subject_id"] = self.csv["subject_id"].astype(str)

        # Compute quantile thresholds for non-zero target values (ignoring rows with "N")
        non_zero_bins = num_bins - 1  # reserve bin 0 for zeros
        # Use pd.to_numeric with errors="coerce" to turn "N" into NaN, then drop NaN values.
        numeric_target = pd.to_numeric(self.csv[target], errors='coerce')
        valid_values = numeric_target[(numeric_target.notna()) & (numeric_target != 0)]
        quantiles = [i / non_zero_bins for i in range(1, non_zero_bins)]
        self.quantile_thresholds = valid_values.quantile(quantiles).tolist()

        all_image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
        
        self.label_map = {}
        self.breast_density_map = {}  # Dictionary to store the continuous breast density value.
        self.image_files = []
        for img_file in all_image_files:
            subject_id = img_file.split("-")[0]
            row = self.csv[self.csv["subject_id"] == subject_id]
            if not row.empty:
                target_value = row.iloc[0][target]
                breast_density = row.iloc[0]["BreastDensity_avg"]
                if target_value == "N":
                    print(f"Warning: subject_id {subject_id} has no {target}. Skipping.")
                    continue

                target_value_float = float(target_value)
                if target_value_float == 0:
                    # Assign zero to bin 0 when target value is zero.
                    self.label_map[subject_id] = 0
                else:
                    # Non-zero values are binned using the quantile thresholds.
                    # We add 1 so that non-zero bins start from 1.
                    self.label_map[subject_id] = bin_value_quantile(target_value_float, self.quantile_thresholds) + 1
                self.breast_density_map[subject_id] = float(breast_density)
                self.image_files.append(img_file)
            else:
                print(f"Warning: subject_id {subject_id} not found in CSV.")

        # Compute or load sample weights for WeightedRandomSampler
        self.sample_weights = []
        if weights_json_path is not None and os.path.exists(weights_json_path):
            with open(weights_json_path, "r") as f:
                self.sample_weights = json.load(f)
            if len(self.sample_weights) != len(self.image_files):
                print("Warning: Loaded sample weights length does not match number of image files. Recomputing weights.")
                self.sample_weights = self.compute_sample_weights()
                with open(weights_json_path, "w") as f:
                    json.dump(self.sample_weights, f)
        else:
            self.sample_weights = self.compute_sample_weights()
            if weights_json_path is not None:
                with open(weights_json_path, "w") as f:
                    json.dump(self.sample_weights, f)

    def compute_sample_weights(self):
        """
        Computes per-sample weights based on the inverse frequency of the bin each sample belongs to.
        Returns:
            list: A list of sample weights corresponding to self.image_files.
        """
        labels = [self.label_map[f.split("-")[0]] for f in self.image_files if f.split("-")[0] in self.label_map]
        bin_counts = Counter(labels)
        weights = []
        for f in self.image_files:
            sid = f.split("-")[0]
            if sid in self.label_map:
                b = self.label_map[sid]
                weights.append(1.0 / bin_counts[b])
            else:
                weights.append(0.0)
        return weights

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_filename = self.image_files[idx]
        subject_id = image_filename.split("-")[0]
        image_path = os.path.join(self.image_dir, image_filename)

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if subject_id not in self.label_map:
            raise ValueError(f"Subject ID {subject_id} not found in the label map.")
        
        label_bin = self.label_map[subject_id]
        # Retrieve the continuous breast density value.
        breast_density = self.breast_density_map[subject_id]
        # Return image, density value, and binned label.
        return image, breast_density, label_bin

    def get_weights(self):
        """
        Returns the per-sample weights for use with WeightedRandomSampler.
        """
        return self.sample_weights

    def get_dataloaders(self, train_split=0.8, batch_size=1, num_workers=0, pin_memory=False):
        """
        Splits the dataset into training and test subsets and returns their respective DataLoaders.

        """
        dataset_size = len(self)
        train_size = int(train_split * dataset_size)
        test_size = dataset_size - train_size
        train_dataset, test_dataset = random_split(self, [train_size, test_size])
        
        if train_split == 0:
            train_dataloader = None
        else:
            train_weights = [self.sample_weights[i] for i in train_dataset.indices]
            train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_dataset), replacement=True)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                        num_workers=num_workers, pin_memory=pin_memory)
        
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=pin_memory)
        
        return train_dataloader, test_dataloader

def get_dataset_stats(dataset):
    """
    Compute per-channel mean and standard deviation for the dataset.
    This function iterates over the dataset using batch_size=1 so that images are not collated.
    It collects per-image statistics and prints intermediate results.
    """
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # Using dictionaries to collect per-channel stats.
    means = {0: [], 1: [], 2: []}
    stds = {0: [], 1: [], 2: []}
    
    for idx, batch in enumerate(tqdm(data_loader, desc="Computing dataset stats"), start=1):
        # Each batch is a tuple: (image, breast_density, label)
        image, _, _ = batch
        # Remove the batch dimension; image shape becomes [3, H, W]
        image = image.squeeze(0)
        for channel in range(3):
            channel_data = image[channel].float()
            means[channel].append(torch.mean(channel_data))
            stds[channel].append(torch.std(channel_data))
        
        if idx % max(1, (len(data_loader) // 20)) == 0:
            current_means = [torch.mean(torch.stack(means[ch])) for ch in range(3)]
            current_stds = [torch.mean(torch.stack(stds[ch])) for ch in range(3)]
            print(f"At image {idx}/{len(data_loader)}: current means: {current_means}, current stds: {current_stds}")
    
    final_means = [torch.mean(torch.stack(means[ch])) for ch in range(3)]
    final_stds = [torch.mean(torch.stack(stds[ch])) for ch in range(3)]
    return final_means, final_stds

if __name__ == "__main__":
    # Update these paths and parameters as needed.
    image_dir = '/fs/nexus-scratch/tuxunlu/git/tdlu/WUSTL_Unmodified_mammograms_selected/WUSTL_png_minmax'
    csv_path = '/fs/nexus-scratch/tuxunlu/git/tdlu/WUSTL_Unmodified_mammograms_selected/umd_annot_md_TDLU_y2025m03d13.csv'
    num_bins = 5                           # Example value; adjust as necessary
    target = "tdlu_density"                # Replace with the actual target column name in your CSV

    # Use a basic transform that converts an image to a tensor without resizing.
    transform = transforms.ToTensor()

    # Create the dataset instance.
    dataset = TDLUDataset(
        image_dir=image_dir,
        csv_path=csv_path,
        num_bins=num_bins,
        transform=transform,
        augment=False,            # No augmentation when computing raw statistics
        weights_json_path=None,   # Not needed for this computation
        target=target
    )

    # Compute and print dataset statistics without stacking/collating images.
    final_means, final_stds = get_dataset_stats(dataset)
    print("Final Dataset Mean (per channel):", final_means)
    print("Final Dataset Std (per channel):", final_stds)
