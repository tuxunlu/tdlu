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


def bin_value_quantile(value, thresholds):
    for i, t in enumerate(thresholds):
        if value <= t:
            return i
    return len(thresholds)

def compute_class_weights(subjects, subjects_data, thresholds, label, zero):
    # 1) Build a list of class indices for every sample
    bins = []
    for sid in subjects:
        raw = subjects_data[sid]['target']

        if label:
            bin_idx = int(raw)
        elif zero:
            bin_idx = 0 if raw == 0 else bin_value_quantile(raw, thresholds) + 1
        else:
            bin_idx = bin_value_quantile(raw, thresholds)

        bins.append(bin_idx)

    # 2) Count how many samples in each class
    counts = Counter(bins)
    print(counts)
    num_classes = max(counts.keys()) + 1  # assumes classes are 0..C-1

    # 3) Compute inverse‐frequency weights
    #    weight[c] = total_samples / (num_classes * counts[c])
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


class TdludatasetTorchioFourViewFolds(Dataset):
    """
    Dataset for four-view mammograms: CC/L, MLO/L, MLO/R, CC/R.
    Supports optional 10-fold cross-validation with separate validation and test folds.
    Accepts `cross_val_fold` for test fold and optional `val_fold` for validation fold (0-9).
    Returns a tensor [4,3,H,W] and one-hot label for `target`.
    """
    def __init__(
        self,
        image_dir: str,
        csv_path: str,
        num_bins: int,
        target: str,
        label: bool,
        zero: bool,
        purpose: str = 'train',  # 'train', 'validation', or 'test'
        cross_val_fold: int = None,  # test fold index 0-9
        val_fold: int = None,        # validation fold index 0-9 (if None, uses next fold)
        split_ratio=(0.8, 0.1, 0.1),  # used if cross_val_fold is None
        use_augmentation: bool = True
    ):
        self.image_dir = image_dir
        self.num_bins = num_bins
        self.target = target
        self.label = label
        self.zero = zero
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
        # df = df[df['total_area_final'].notna() & (df['total_area_final'].astype(str).str.strip() != '')]
        df = df[df[['ViewPosition_comp2', 'ImageLaterality_comp2']]
                .apply(lambda r: (r['ViewPosition_comp2'], r['ImageLaterality_comp2']) in self.allowed_combos, axis=1)]
        df['stem'] = df['filename'].str.replace(r"\.[^.]+$", "", regex=True)
        df['subject_id'] = df['subject_id'].astype(str)

        # Compute quantile thresholds
        vals = pd.to_numeric(df[target], errors='coerce')
        if self.zero:
            num_bins = num_bins - 1
            valid = vals[(vals.notna()) & (vals > 0)]
        else:
            valid = vals[(vals.notna())]
        quantiles = [i / num_bins for i in range(1, num_bins)]
        self.thresholds = valid.quantile(quantiles).tolist()

        # Build subject_data map
        all_data = {}
        for sid, group in df.groupby('subject_id'):
            combos = {}
            for _, row in group.iterrows():
                combos[(row['ViewPosition_comp2'], row['ImageLaterality_comp2'])] = row['filename'].replace('.dcm', '.png')
            if set(combos.keys()) == set(self.allowed_combos):
                if not np.isnan(float(group.iloc[0][target])):
                    all_data[sid] = {'combos': combos, 'target': float(group.iloc[0][target])}
        all_subjects = list(all_data.keys())

        # Cross-validation splitting
        assert 0 <= cross_val_fold < 10, "cross_val_fold must be in [0,9]"
        subs = all_subjects.copy()
        random.seed(1234)
        random.shuffle(subs)
        folds = np.array_split(subs, 10)
        test_ids = list(folds[cross_val_fold])
        # determine validation fold
        if val_fold is None:
            val_idx = (cross_val_fold + 1) % 10
        else:
            val_idx = val_fold
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

        # save splits
        ts = time.time()
        split_name = f"fold{cross_val_fold}_"
        if purpose in ['test']:
            with open(f"{purpose}_subjects_{split_name}{ts}.json", 'w') as f:
                json.dump(self.subjects, f)
        if purpose in ['validation']:
            with open(f"{purpose}_subjects_{split_name}{ts}.json", 'w') as f:
                json.dump(self.subjects, f)
        if purpose in ['train']:
            with open(f"{purpose}_subjects_{split_name}{ts}.json", 'w') as f:
                json.dump(self.subjects, f)

        self.transform = self._make_transform()
        
        self.class_weights = compute_class_weights(
            self.subjects,
            self.subject_data,
            self.thresholds,
            self.label,
            self.zero
        )
        print(f"{self.purpose} Class weights: {self.class_weights}")

    def _make_transform(self):
        mean, std = (0.113,) * 3, (0.185,) * 3
        tio_augs = tio.Compose([
            # Intensity / bias
            tio.RandomBiasField(coefficients=(0.1, 0.3), p=0.5),
            tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5),
            tio.RandomNoise(mean=0.0, std=(0, 0.1), p=0.5),
            tio.RandomBlur(std=(0.5, 1.5), p=0.3),

            # Geometric
            tio.RandomAffine(
                scales=(0.9, 1.1),
                degrees=15,
                translation=0,
                isotropic=False,
                p=0.5
            ),
            # tio.RandomElasticDeformation(
            #     num_control_points=7,
            #     max_displacement=(5, 5, 0),
            #     locked_borders=2,
            #     p=0.3
            # ),
            tio.RandomFlip(axes=('LR',), p=0.5),
        ])

        wrapper = TorchIOWrapper(tio_augs)
        base = [transforms.Resize((1024, 1024)), transforms.PILToTensor(), ConvertImageDtype(torch.float32)]
        if self.purpose == 'train' and self.use_augmentation:
            base.append(wrapper)
        base += [Lambda(lambda x: x.repeat(3, 1, 1)), transforms.Normalize(mean, std)]
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

        if self.label:
            bin_idx = raw
        elif self.zero:
            bin_idx = 0 if raw == 0 else bin_value_quantile(raw, self.thresholds) + 1
        else:
            bin_idx = bin_value_quantile(raw, self.thresholds)

        label = torch.tensor(bin_idx, dtype=torch.long)
        return views_tensor, label, file_names

if __name__ == "__main__":
    ds = TdludatasetTorchioFourViewFolds(
        image_dir='...',
        csv_path='...',
        num_bins=2,
        target='tdlu_density',
        purpose='validation',
        cross_val_fold=0,
        val_fold=1
    )
    print(len(ds), ds[0][0].shape, ds)