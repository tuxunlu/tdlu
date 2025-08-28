import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import RandomApply
from torch.nn import ModuleList
from torchvision.transforms import ConvertImageDtype, Lambda
import torchio as tio
import json
import time

from utils.ToTensor16RGB import ToTensor16RGB

import matplotlib.pyplot as plt


def bin_value_quantile(value, thresholds):
    """
    Return which bin value belongs to based on thresholds
    """
    for i, t in enumerate(thresholds):
        if value <= t:
            return i
    return len(thresholds)

def compute_class_weights(subjects, subjects_data, thresholds, label_type):
    """
    Compute the inverse-frequency weights for each class. The weights are used as focal loss alpha
    """
    # 1) Build a list of class indices for every sample
    bins = []
    for sid in subjects:
        raw = subjects_data[sid]['target']

        if label_type == "label":
            bin_idx = int(raw)
        elif label_type == "label":
            bin_idx = 0 if raw == 0 else bin_value_quantile(raw, thresholds) + 1
        else:
            bin_idx = bin_value_quantile(raw, thresholds)

        bins.append(bin_idx)

    # 2) Count how many samples in each class
    counts = Counter(bins)
    print(counts)
    num_classes = max(counts.keys()) + 1  # assumes classes are 0..C-1

    # 3) Compute inverse‐frequency weights
    total = len(bins)
    class_weights = []
    for c in range(num_classes):
        # if a class is missing, you can set weight 0 or 1.0 as you prefer.
        cnt = counts.get(c, 0)
        if cnt > 0:
            w = total / (num_classes * cnt)
        else:
            w = 0.0
        class_weights.append(w)
    
    return class_weights


class TorchIOWrapper:
    def __init__(self, tio_transform: tio.Transform):
        self.tio_transform = tio_transform

    def __call__(self, img_tensor):
        vol = img_tensor.unsqueeze(0)
        subject = tio.Subject(image=tio.ScalarImage(tensor=vol))
        transformed = self.tio_transform(subject)
        return transformed.image.data.squeeze(0).float()

class RandomOrderTorchIO(tio.transforms.Transform):
    def __init__(self, transforms, p=1.0, copy=True):
        super().__init__(p=p, copy=copy)
        # Ensure a mutable list copy
        self.transforms = list(transforms)

    def apply_transform(self, subject):
        # Shuffle in place
        random.shuffle(self.transforms)
        # Apply each in new order
        for transform in self.transforms:
            subject = transform(subject)
        return subject


class TdludatasetTorchioFourViewMetaAvgFolds(Dataset):
    """
    Dataset for four-view mammograms: CC/L, MLO/L, MLO/R, CC/R.
    Supports optional 10-fold cross-validation with separate validation and test folds.
    Accepts `cross_val_fold` for test fold and optional `val_fold` for validation fold (0-9).
    Returns views_tensor, meta, label, file_names
    """
    def __init__(
        self,
        image_dir: str,
        csv_path: str,
        num_bins: int,
        target: str,
        meta_cols: list,
        label_type: str,
        purpose: str = 'train',  # 'train', 'validation', or 'test'
        cross_val_fold: int = None,  # test fold index 0-9
        split_ratio=(0.8, 0.1, 0.1),  # used if cross_val_fold is None
        use_augmentation: bool = True
    ):
        self.image_dir = image_dir
        self.num_bins = num_bins
        self.target = target
        self.meta_cols = meta_cols
        self.label_type = label_type
        self.purpose = purpose
        self.use_augmentation = use_augmentation

        # Allowed view-laterality combinations
        self.allowed_combos = [
            ('CC', 'L'),
            ('MLO', 'L'),
            ('MLO', 'R'),
            ('CC', 'R'),
        ]

        # Load and filter CSV
        df = pd.read_csv(csv_path)
        # Drop unknown geanc_Race rows
        df = df[df['geanc_Race'] != 'N']

        # Filter to rows where (view, side) is one of {('CC','L'), ('MLO','L'), ('MLO','R'), ('CC','R')}”.
        df = df[df[['ViewPosition_comp2', 'ImageLaterality_comp2']]
                .apply(lambda r: (r['ViewPosition_comp2'], r['ImageLaterality_comp2']) in self.allowed_combos, axis=1)]

        df['stem'] = df['filename'].str.replace(r"\.[^.]+$", "", regex=True)
        df['subject_id'] = df['subject_id'].astype(str)

        # Extract meta-data per subject
        meta_df = df[['subject_id'] + self.meta_cols].drop_duplicates('subject_id')
        meta_map = {row.subject_id: row[self.meta_cols].values.astype(float) for _, row in meta_df.iterrows()}

        # Compute quantile thresholds
        vals = pd.to_numeric(df[target], errors='coerce')
        
        if self.label_type == "zero":
            # Reserve one bin for zeros
            num_bins = num_bins - 1
            valid = vals[(vals.notna()) & (vals > 0)]
        else:
            valid = vals[(vals.notna()) & (vals > 0)]

        # Compute the thresholds for each quantile
        quantiles = [i / num_bins for i in range(1, num_bins)]
        self.thresholds = valid.quantile(quantiles).tolist()

        all_data = {}
        for sid, group in df.groupby('subject_id'):
            # 1) build view → filename map
            combos = {}
            for _, row in group.iterrows():
                key = (row['ViewPosition_comp2'], row['ImageLaterality_comp2'])
                combos[key] = row['filename'].replace('.dcm', '.png')

            # only keep subjects with all 4 views present
            if set(combos.keys()) != set(self.allowed_combos):
                continue
            # skip if missing target
            raw_target = float(group.iloc[0][target])
            if np.isnan(raw_target):
                continue

            # 3) extract the 4 shared meta-features
            subj_meta = group.iloc[0][self.meta_cols].astype(float).values  # → shape (4,)

            all_data[sid] = {
                'combos': combos,
                'target': raw_target,
                'subject_meta': subj_meta
            }

        # Cross-validation splitting
        assert 0 <= cross_val_fold < 10, "cross_val_fold must be in [0,9]"
        subs = list(all_data.keys())
        random.seed(1234)
        random.shuffle(subs)
        folds = np.array_split(subs, 10)
        test_ids = list(folds[cross_val_fold])
        # determine validation fold
        val_idx = (cross_val_fold + 1) % 10
        assert 0 <= val_idx < 10 and val_idx != cross_val_fold, "val_fold must differ from cross_val_fold"
        val_ids = list(folds[val_idx])
        # training = remaining folds
        train_ids = [sid for i, f in enumerate(folds) if i not in (cross_val_fold, val_idx) for sid in f]

        if purpose == 'train':
            selected = train_ids
        elif purpose == 'validation':
            selected = val_ids
        elif purpose == 'test':
            selected = test_ids
        else:
            raise ValueError("purpose must be 'train', 'validation', or 'test'")

        # finalize
        self.subjects = selected
        self.subject_data = {sid: all_data[sid] for sid in self.subjects}
        self.transform = self._make_transform()
        
        self.class_weights = compute_class_weights(
            self.subjects,
            self.subject_data,
            self.thresholds,
            self.label_type
        )
        print(f"{self.purpose} Class weights: {self.class_weights}")


    def _make_transform(self):
        mean, std = (0.113,)*3, (0.185,)*3

        # 1) Build RandomOrder pipeline
        tio_augs = [
            tio.RandomBiasField(p=0.3),
            tio.RandomGamma(log_gamma=(-0.3,0.3), p=0.3),
            tio.RandomNoise(std=(0,0.25), p=0.3),
            tio.RandomBlur(std=(0,2), p=0.3),
        ]
        random_order_aug = RandomOrderTorchIO(tio_augs, p=0.4)
        wrapper = TorchIOWrapper(random_order_aug)

        # 2) Put it behind a RandomApply gate:
        aug_gate = RandomApply([wrapper], p=1)

        # Base transformations
        base = [
            transforms.Resize((1024,1024)),
            transforms.PILToTensor(),
            ConvertImageDtype(torch.float32),
        ]

        # Pytorch augmentations
        torch_augs = transforms.RandomApply(torch.nn.ModuleList([
            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
                shear=10,
            ),
            transforms.RandomVerticalFlip(p=0.5),
        ]), p=0.4)

        if self.purpose == 'train' and self.use_augmentation:
            base.append(aug_gate)
            base.append(torch_augs)
        base += [
            Lambda(lambda x: x.repeat(3,1,1)),
            transforms.Normalize(mean, std),
        ]

        return transforms.Compose(base)


    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        sid = self.subjects[idx]
        data = self.subject_data[sid]
        views = []
        file_names = []
        for combo in self.allowed_combos:
            fname = data['combos'][combo]          # get the filename
            img_path = os.path.join(self.image_dir, fname)
            img = Image.open(img_path)
            views.append(self.transform(img))
            file_names.append(fname)
        views_tensor = torch.stack(views, 0)

        raw = data['target']

        
        if self.label_type == "raw":
            bin_idx = raw
        elif self.label_type == "label":
            # Provided raw data are labels already
            bin_idx = raw
        elif self.label_type == "zero":
            # Provided raw data are NOT labels and need to reserve zeros as a separate class
            bin_idx = 0 if raw == 0 else bin_value_quantile(raw, self.thresholds) + 1
        else:
            # Provided raw data are NOT labels
            bin_idx = bin_value_quantile(raw, self.thresholds)

        label = torch.tensor(bin_idx, dtype=torch.long) if not self.label_type == "raw" else torch.tensor(bin_idx, dtype=torch.float32)

        # the 4 shared meta-features
        meta = torch.tensor(
            data['subject_meta'],
            dtype=torch.float32
        )  # → shape [4]

        return views_tensor, meta, label, file_names

if __name__ == "__main__":
    train_ds = TdludatasetTorchioFourViewMetaAvgFolds(
        image_dir='/beacon-scratch/tuxunlu/git/tdlu/dataset/WUSTL_png_nomarker_16',
        csv_path='/beacon-scratch/tuxunlu/git/tdlu/dataset/umd_annot_md_TDLU_y2025m07d09.csv',
        num_bins=3,
        target='tdlu_density',
        label_type='raw',
        meta_cols=['BreastDensity', 'mamm_age', 'cbmi_donation', 'geanc_Race'],
        purpose='train',
        cross_val_fold=0,
    )

    val_ds = TdludatasetTorchioFourViewMetaAvgFolds(
        image_dir='/beacon-scratch/tuxunlu/git/tdlu/dataset/WUSTL_png_nomarker_16',
        csv_path='/beacon-scratch/tuxunlu/git/tdlu/dataset/umd_annot_md_TDLU_y2025m07d09.csv',
        num_bins=3,
        target='tdlu_density',
        label_type='raw',
        meta_cols=['BreastDensity', 'mamm_age', 'cbmi_donation', 'geanc_Race'],
        purpose='validation',
        cross_val_fold=0,
    )

    idx=1
    
    views, meta, label, filenames = train_ds[idx]
    print("Train: ", views.shape, meta, label, filenames)
    # statistics used in your Normalize
    mean = torch.tensor([0.113, 0.113, 0.113])[:, None, None]
    std  = torch.tensor([0.185, 0.185, 0.185])[:, None, None]

    fig, axes = plt.subplots(1, len(views), figsize=(16,4))
    for i, (v, fname) in enumerate(zip(views, filenames)):
        # un-normalize
        img = v * std + mean               # still [3,H,W]
        img = img.clamp(0,1)               # ensure in [0,1]
        img = img.permute(1,2,0).cpu().numpy()  # → H×W×3

        # if you want grayscale, average the channels:
        gray = img.mean(-1)

        axes[i].imshow(gray, cmap='gray')
        axes[i].set_title(fname.split('.')[0])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig('train views.png', dpi=300)


    views, meta, label, filenames = val_ds[idx]
    print("Validation: ", views.shape, meta, label, filenames)
    # statistics used in your Normalize
    mean = torch.tensor([0.113, 0.113, 0.113])[:, None, None]
    std  = torch.tensor([0.185, 0.185, 0.185])[:, None, None]

    fig, axes = plt.subplots(1, len(views), figsize=(16,4))
    for i, (v, fname) in enumerate(zip(views, filenames)):
        # un-normalize
        img = v * std + mean               # still [3,H,W]
        img = img.clamp(0,1)               # ensure in [0,1]
        img = img.permute(1,2,0).cpu().numpy()  # → H×W×3

        # if you want grayscale, average the channels:
        gray = img.mean(-1)

        axes[i].imshow(gray, cmap='gray')
        axes[i].set_title(fname.split('.')[0])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig('val views.png', dpi=300)
