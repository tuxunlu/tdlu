import os
import json
import random
import pandas as pd
import numpy as np
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms

def bin_value_quantile(value, thresholds):
    """
    Convert a numeric value into a bin index based on quantile thresholds.
    Args:
        value (float): The continuous value to bin.
        thresholds (list): Sorted list of quantile threshold values.
    Returns:
        int: Bin index in the range [0, len(thresholds)].
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
            csv_path (str): Path to CSV with columns ['subject_id', <target>].
            transform (callable): Optional transform pipeline.
            augment (bool): If True, apply simple augmentation.
            weights_json_path (str): Optional JSON path for sample weights.
            target (str): Column name in CSV for target labels.
            num_bins (int): Number of bins for classification.
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
                    transforms.Normalize((0.4195, 0.4195, 0.4195), (0.4363, 0.4363, 0.4363))
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4195, 0.4195, 0.4195), (0.4363, 0.4363, 0.4363))
                ])

        self.csv = pd.read_csv(csv_path)
        self.csv["subject_id"] = self.csv["subject_id"].astype(str)

        # Compute quantile thresholds based on the target column (ignoring "N" values)
        valid_values = self.csv[self.csv[target] != "N"][target].astype(float)
        quantiles = [i / num_bins for i in range(1, num_bins)]
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
                # Use quantile-based binning instead of uniform binning.
                self.label_map[subject_id] = bin_value_quantile(float(target_value), self.quantile_thresholds)
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
