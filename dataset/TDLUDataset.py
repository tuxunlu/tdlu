import os
import json
import random
import pandas as pd
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
import torchvision.transforms as transforms

def bin_value(value, num_bins, max_val):
    """
    Convert a numeric value into an integer bin index in [0, num_bins-1].
    Each bin covers a range of size (max_val / num_bins).
    """
    bin_size = max_val / num_bins  # e.g., if max_val=100 and num_bins=10, bin_size = 10
    bin_index = int(value // bin_size)
    if bin_index >= num_bins:
        bin_index = num_bins - 1  # clamp to last bin
    return bin_index

class TDLUDataset(Dataset):
    def __init__(self, image_dir, csv_path, num_bins, transform=None, augment=False, weights_json_path=None, target=None):
        """
        Args:
            image_dir (str): Directory containing .png images.
            csv_path (str): Path to CSV with columns ['subject_id', <target>].
            transform (callable): Optional transform pipeline.
            augment (bool): If True and no custom transform is provided, apply simple augmentation.
            weights_json_path (str): Optional path to a JSON file for storing/retrieving sample weights.
            target (str): Column name in CSV to use for labels.
            num_bins (int): Number of bins to use when binning the density value.
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
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4195, 0.4195, 0.4195),
                                         (0.4363, 0.4363, 0.4363))
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4195, 0.4195, 0.4195),
                                         (0.4363, 0.4363, 0.4363))
                ])

        # Load CSV and ensure subject_id is string
        self.csv = pd.read_csv(csv_path)
        self.csv["subject_id"] = self.csv["subject_id"].astype(str)

        # Compute the maximum density value from the CSV for the given target column,
        # ignoring rows where the density is "N"
        valid_values = self.csv[self.csv[target] != "N"][target].astype(float)
        self.max_density = valid_values.max() if not valid_values.empty else 200

        # Gather all .png files and sort them for stable ordering
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

        # Build a map from subject_id -> binned label
        self.label_map = {}
        for img_file in self.image_files:
            subject_id = img_file.split("-")[0]
            row = self.csv[self.csv["subject_id"] == subject_id]
            if not row.empty:
                target_value = row.iloc[0][target]
                if target_value == "N":
                    print(f"Warning: subject_id {subject_id} has no {target}. Skipping.")
                    continue
                self.label_map[subject_id] = bin_value(float(target_value), num_bins=num_bins, max_val=self.max_density)
            else:
                print(f"Warning: subject_id {subject_id} not found in CSV.")

        # Compute or load sample weights for WeightedRandomSampler
        self.sample_weights = []
        if weights_json_path is not None and os.path.exists(weights_json_path):
            # Load precomputed weights from the JSON file
            with open(weights_json_path, "r") as f:
                self.sample_weights = json.load(f)
            # Optionally, you could add a check that the length matches self.image_files
            if len(self.sample_weights) != len(self.image_files):
                print("Warning: Loaded sample weights length does not match number of image files. Recomputing weights.")
                self.sample_weights = self.compute_sample_weights()
                with open(weights_json_path, "w") as f:
                    json.dump(self.sample_weights, f)
        else:
            # Compute sample weights and save them if a path is provided
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
        # Get the bin label for each image (only if available in label_map)
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
        return image, label_bin

    def get_weights(self):
        """
        Returns the per-sample weights for use with WeightedRandomSampler.
        """
        return self.sample_weights

    def get_dataloaders(self, train_split=0.8, batch_size=1, num_workers=0, pin_memory=False):
        """
        Splits the dataset into training and test subsets and returns respective DataLoaders.
        The training dataloader uses a WeightedRandomSampler based on the sample weights,
        while the test dataloader uses the default (unweighted) sampling.
        
        Args:
            train_split (float): Proportion of the dataset to use for training (e.g., 0.8).
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses to use for data loading.
            pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory.
        
        Returns:
            tuple: (train_dataloader, test_dataloader)
        """
        dataset_size = len(self)
        train_size = int(train_split * dataset_size)
        test_size = dataset_size - train_size
        train_dataset, test_dataset = random_split(self, [train_size, test_size])
        
        if train_split == 0:
            train_dataloader = None
        else:
            # Create a weighted sampler for the training subset using only the weights for selected indices.
            train_weights = [self.sample_weights[i] for i in train_dataset.indices]
            train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_dataset), replacement=True)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                        num_workers=num_workers, pin_memory=pin_memory)
        
        # For the test subset, simply use a dataloader with default (sequential) sampling.
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=pin_memory)
        
        return train_dataloader, test_dataloader
