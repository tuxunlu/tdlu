import os
import json
import random
import pandas as pd
import numpy as np
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.ToTensor16RGB import ToTensor16RGB
from torchvision.transforms import ConvertImageDtype
import torchio as tio
import torchvision.transforms.functional as TF
import time
import torchio as tio
from torchvision import transforms
from torchvision.transforms import Lambda


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

class TorchIOWrapper:
    def __init__(self, tio_transform: tio.Transform):
        self.tio_transform = tio_transform

    def __call__(self, img_tensor):
        # img_tensor: (C, H, W), possibly int16
        vol = img_tensor.unsqueeze(0)  # → (1, C, H, W)
        subject = tio.Subject(image=tio.ScalarImage(tensor=vol))
        transformed = self.tio_transform(subject)
        # cast to float before returning
        return transformed.image.data.squeeze(0).float()



class TdludatasetTorchio(Dataset):
    def __init__(
        self,
        image_dir,
        csv_path,
        num_bins,
        target,
        split_ratio=[0.8, 0.1, 0.1],
        purpose="train",
        aug_prob=0.5,
        use_augmentation=True,
        mamm_only=False,
        num_workers=4,
        pin_memory=True,
        batch_size=1,
        weights_json_path=None
    ):
        """
        Args:
            image_dir (str): Directory containing .png images.
            csv_path (str): Path to CSV with columns ['subject_id', <target>, 'BreastDensity_avg', ...].
            num_bins (int): Total number of bins (bin 0 is reserved for zero values;
                            non-zero values are binned into num_bins-1 bins).
            target (str): Column name in CSV for target labels.
            split_ratio (list): [train, validation, test] split fractions.
            purpose (str): One of "train", "validation", or "test"; determines augmentation and split.
            aug_prob (float): Probability for applying augmentation transforms.
            use_augmentation (bool): Whether to apply augmentation (only if purpose == "train").
            mamm_only (bool): If True, __getitem__ returns only (image, label_onehot).
            num_workers (int): For DataLoader compatibility.
            pin_memory (bool): For DataLoader compatibility.
            batch_size (int): For DataLoader compatibility.
            weights_json_path (str): Optional path for sample weights JSON.
        """
        self.image_dir = image_dir
        self.csv_path = csv_path
        self.num_bins = num_bins
        self.target = target
        self.purpose = purpose
        self.aug_prob = aug_prob
        self.use_augmentation = use_augmentation
        self.mamm_only = mamm_only
        self.split_ratio = split_ratio
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size

        # Configure augmentation/normalization.
        self.augmentation = self.__configure_augmentation()

        # Read CSV and ensure filename is string
        self.csv = pd.read_csv(csv_path)
        self.csv["filename"] = self.csv["filename"].astype(str)
        self.csv["stem"] = self.csv["filename"].str.replace(r"\.[^.]+$", "", regex=True)
        df_stem = self.csv.set_index("stem")

        # Compute quantile thresholds for non-zero target values
        non_zero_bins = num_bins - 1
        numeric_target = pd.to_numeric(self.csv[target], errors='coerce')
        valid_values = numeric_target[(numeric_target.notna()) & (numeric_target != 0)]
        quantiles = [i / non_zero_bins for i in range(1, non_zero_bins)]
        self.quantile_thresholds = valid_values.quantile(quantiles).tolist()
        print(f"Quantile thresholds for {target}: {self.quantile_thresholds}")

        # Gather image files
        all_image_files = sorted(f for f in os.listdir(image_dir) if f.endswith(".png"))
        self.label_map = {}
        self.breast_density_map = {}
        self.age_map = {}
        self.bmi_map = {}
        self.image_files = []

        for img_file in all_image_files:
            stem = os.path.splitext(img_file)[0]
            if stem not in df_stem.index:
                print(f"Warning: no CSV row for stem {stem}, skipping {img_file}")
                continue
            row = df_stem.loc[stem]
            density = float(row["BreastDensity_avg"])
            age = float(row["mamm_age"])
            bmi = float(row["cbmi_donation"])
            raw_target = row[self.target]
            if raw_target == "N":
                continue
            tval = float(raw_target)
            bin_idx = 0 if tval == 0 else bin_value_quantile(tval, self.quantile_thresholds) + 1
            self.label_map[img_file] = bin_idx
            self.breast_density_map[img_file] = density
            self.age_map[img_file] = age
            self.bmi_map[img_file] = bmi
            self.image_files.append(img_file)

        # Map subjects
        self.subject_map = {f: str(df_stem.loc[os.path.splitext(f)[0], "subject_id"]) for f in self.image_files}

        # Split dataset
        if self.purpose in ["train", "validation", "test"]:
            self._split_dataset()

        # Compute sample weights
        self.sample_weights = self.compute_sample_weights()

    def __configure_augmentation(self):
        mean, std = (0.113, 0.113, 0.113), (0.185, 0.185, 0.185)
        tio_augs = tio.Compose([
            tio.RandomBiasField(coefficients=(0.3, 0.7), p=0.5),
            tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5),
            tio.RandomNoise(std=(0, 0.05), p=0.5),
            tio.RandomBlur(std=(0, 1.5), p=0.5),
            tio.RandomElasticDeformation(num_control_points=5, max_displacement=5, p=0.3),
            tio.RandomAffine(degrees=(-15, 15), scales=(0.9, 1.1), p=0.5),
            tio.RandomFlip(axes=('LR',), p=0.5),
        ])
        tio_wrapper = TorchIOWrapper(tio_augs)
        if self.purpose == 'train' and self.use_augmentation:
            return transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.PILToTensor(),                # yields integer tensor
                ConvertImageDtype(torch.float32),       # cast & scale to float
                tio_wrapper,                            # your TorchIO augmentations
                Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.PILToTensor(),                # yields integer tensor
                ConvertImageDtype(torch.float32),       # cast & scale to float
                Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize(mean, std),
            ])

    def compute_sample_weights(self):
        labels = [self.label_map[f] for f in self.image_files]
        bin_counts = Counter(labels)
        return [1.0 / bin_counts[self.label_map[f]] for f in self.image_files]

    def _split_dataset(self):
        random.seed(42)
        all_subjects = sorted(list(set(self.subject_map.values())))
        random.shuffle(all_subjects)
        total = len(all_subjects)
        train_end = int(self.split_ratio[0] * total)
        val_end = train_end + int(self.split_ratio[1] * total)
        train_subjs = all_subjects[:train_end]
        val_subjs = all_subjects[train_end:val_end]
        test_subjs = all_subjects[val_end:]
        chosen = {'train': train_subjs, 'validation': val_subjs, 'test': test_subjs}[self.purpose]

        if self.purpose == "train":
            # Save chosen subject ID to JSON file
            with open(f"train_subjects_{time.time()}.json", "w") as f:
                json.dump(list(chosen), f)
        elif self.purpose == "validation":
            # Save chosen subject ID from JSON file
            with open(f"val_subjects_{time.time()}.json", "w") as f:
                json.dump(list(chosen), f)
        if self.purpose == "test":
            # Save chosen subject ID from JSON file
            with open(f"test_subjects_{time.time()}.json", "w") as f:
                json.dump(list(chosen), f)
        self.image_files = [f for f in self.image_files if self.subject_map[f] in chosen]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img = Image.open(os.path.join(self.image_dir, img_file))
        img = self.augmentation(img)
        if img_file not in self.label_map:
            raise ValueError(f"Filename {img_file} not found in label map.")
        label_bin = self.label_map[img_file]
        label_onehot = F.one_hot(torch.tensor(label_bin), num_classes=self.num_bins).float()

        if self.mamm_only:
            return img, label_onehot

        density = self.breast_density_map[img_file]
        age = self.age_map[img_file]
        bmi = self.bmi_map[img_file]
        return img, torch.tensor(density, dtype=torch.float32), \
               torch.tensor(age, dtype=torch.float32), torch.tensor(bmi, dtype=torch.float32), label_onehot


def get_dataset_stats(dataset, batch_size=32, num_workers=12):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    total_pixels = 0
    for items in tqdm(loader, desc="Recomputing dataset stats"):
        imgs = items[0] if isinstance(items, (list, tuple)) else items
        img = imgs[0]
        c, h, w = img.shape
        pixels = h * w
        channel_sum += img.view(c, -1).sum(dim=1)
        channel_sq_sum += (img.view(c, -1) ** 2).sum(dim=1)
        total_pixels += pixels
    means = channel_sum / total_pixels
    vars_ = channel_sq_sum / total_pixels - means ** 2
    stds = torch.sqrt(vars_)
    return means.tolist(), stds.tolist()


if __name__ == "__main__":
    # Example usage
    train_dataset = TDLUDataset_torchio(
        image_dir="/fs/nexus-scratch/tuxunlu/git/tdlu/dataset/WUSTL_png_nomarker_16",
        csv_path="/fs/nexus-scratch/tuxunlu/git/tdlu/dataset/umd_annot_md_TDLU_y2025m04d28.csv",
        num_bins=4,
        target="tdlu_density",
        purpose="train",
        use_augmentation=False,
        mamm_only=True,
        batch_size=8
    )

    # Get dataset stats
    means, stds = get_dataset_stats(train_dataset, batch_size=8, num_workers=12)
    print("Means:", means)
    print("Stds:", stds)

    val_dataset = TDLUDataset_torchio(
        image_dir="/fs/nexus-scratch/tuxunlu/git/tdlu/dataset/WUSTL_png_nomarker_16",
        csv_path="/fs/nexus-scratch/tuxunlu/git/tdlu/dataset/umd_annot_md_TDLU_y2025m04d28.csv",
        num_bins=4,
        target="tdlu_density",
        purpose="validation",
        use_augmentation=False,
        mamm_only=True,
        batch_size=8
    )

    test_dataset = TDLUDataset_torchio(
        image_dir="/fs/nexus-scratch/tuxunlu/git/tdlu/dataset/WUSTL_png_nomarker_16",
        csv_path="/fs/nexus-scratch/tuxunlu/git/tdlu/dataset/umd_annot_md_TDLU_y2025m04d28.csv",
        num_bins=4,
        target="tdlu_density",
        purpose="test",
        use_augmentation=False,
        mamm_only=True,
        batch_size=8
    )
    
    train_images = set(train_dataset.image_files)
    val_images   = set(val_dataset.image_files)
    test_images  = set(test_dataset.image_files)

    # Check overlaps
    print("Train ∩ Val:", train_images & val_images)     # should be set()
    print("Train ∩ Test:", train_images & test_images)    # should be set()
    print("Val ∩ Test:", val_images & test_images)        # should be set()

    # Or assert that they’re disjoint
    assert train_images.isdisjoint(val_images)
    assert train_images.isdisjoint(test_images)
    assert val_images.isdisjoint(test_images)
    print("All splits are mutually exclusive!")
