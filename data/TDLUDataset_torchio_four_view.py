import os
import random
from collections import Counter

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
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


class TorchIOWrapper:
    def __init__(self, tio_transform: tio.Transform):
        self.tio_transform = tio_transform

    def __call__(self, img_tensor):
        vol = img_tensor.unsqueeze(0)
        subject = tio.Subject(image=tio.ScalarImage(tensor=vol))
        transformed = self.tio_transform(subject)
        return transformed.image.data.squeeze(0).float()


class TdludatasetTorchioFourView(Dataset):
    """
    Dataset for four-view mammograms: CC/L, MLO/L, MLO/R, CC/R.
    Groups by `subject_id` column. Only subjects with all four views are kept.
    Returns a tensor [4,3,H,W] and one-hot label for `target`.
    """
    def __init__(
        self,
        image_dir: str,
        csv_path: str,
        num_bins: int,
        target: str,
        split_ratio=(0.8, 0.1, 0.1),
        purpose='train',
        use_augmentation=True
    ):
        self.image_dir = image_dir
        self.num_bins = num_bins
        self.target = target
        self.purpose = purpose
        self.use_augmentation = use_augmentation

        # Allowed view-laterality combinations
        self.allowed_combos = [
            ('CC', 'L'),
            ('MLO', 'L'),
            ('MLO', 'R'),
            ('CC', 'R'),
        ]

        # Load and filter CSV for allowed combos
        df = pd.read_csv(csv_path)
        df = df[df['total_area_final'].notna() & (df['total_area_final'].astype(str).str.strip() != '')]
        df = df[df[['ViewPosition_comp2', 'ImageLaterality_comp2']]
                .apply(lambda r: (r['ViewPosition_comp2'], r['ImageLaterality_comp2']) in self.allowed_combos, axis=1)]

        # Derive stem (filename without extension)
        df['stem'] = df['filename'].str.replace(r"\.[^.]+$", "", regex=True)
        df['subject_id'] = df['subject_id'].astype(str)

        # Compute quantile thresholds on non-zero target values
        nz = num_bins - 1
        vals = pd.to_numeric(df[target], errors='coerce')
        nonzeros = vals[(vals.notna()) & (vals != 0)]
        quantiles = [i/nz for i in range(1, nz)]
        self.thresholds = nonzeros.quantile(quantiles).tolist()

        # Build mapping: subject_id -> dict of combo->filename and target
        self.subject_data = {}
        grouped = df.groupby('subject_id')
        for sid, group in grouped:
            combos = {}
            for _, row in group.iterrows():
                combo = (row['ViewPosition_comp2'], row['ImageLaterality_comp2'])
                png_file = row['filename'].replace('.dcm', '.png')
                combos[combo] = png_file
            # keep only if all combos exist
            if set(combos.keys()) == set(self.allowed_combos):
                # assume target same for all rows of subject
                tgt = float(group.iloc[0][target])
                self.subject_data[sid] = {'combos': combos, 'target': tgt}


        # List of valid subject_ids
        self.subjects = list(self.subject_data.keys())

        # Split subjects
        random.seed(42)
        random.shuffle(self.subjects)
        n = len(self.subjects)
        t = int(split_ratio[0] * n)
        v = int((split_ratio[0] + split_ratio[1]) * n)
        if purpose == 'train':
            sel = self.subjects[:t]
        elif purpose == 'validation':
            sel = self.subjects[t:v]
        else:
            sel = self.subjects[v:]
        self.subjects = sel

        # Compute sample weights
        bins = []
        for sid in self.subjects:
            raw = self.subject_data[sid]['target']
            bin_idx = 0 if raw==0 else bin_value_quantile(raw, self.thresholds)+1
            bins.append(bin_idx)
        counts = Counter(bins)
        self.sample_weights = [1.0/counts[b] for b in bins]

        if self.purpose == "train":
            # Save chosen subject ID to JSON file
            with open(f"train_subjects_{time.time()}.json", "w") as f:
                json.dump(list(sel), f)
        elif self.purpose == "validation":
            # Save chosen subject ID from JSON file
            with open(f"val_subjects_{time.time()}.json", "w") as f:
                json.dump(list(sel), f)
        if self.purpose == "test":
            # Save chosen subject ID from JSON file
            with open(f"test_subjects_{time.time()}.json", "w") as f:
                json.dump(list(sel), f)

        # Prepare transforms
        self.transform = self._make_transform()

    def _make_transform(self):
        mean, std = (0.113,)*3, (0.185,)*3

        # Stronger, more diverse TorchIO augmentations
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
            tio.RandomElasticDeformation(
                num_control_points=7,
                max_displacement=(5, 5, 0),
                locked_borders=2,
                p=0.3
            ),
            tio.RandomFlip(axes=('LR',), p=0.5),
        ])

        wrapper = TorchIOWrapper(tio_augs)

        base = [
            transforms.Resize((1024, 1024)),
            transforms.PILToTensor(),
            ConvertImageDtype(torch.float32),
        ]
        if self.purpose == 'train' and self.use_augmentation:
            base.append(wrapper)

        base += [
            # replicate grayscale to 3 channels
            Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean, std),
        ]
        return transforms.Compose(base)


    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        sid = self.subjects[idx]
        data = self.subject_data[sid]
        combos = data['combos']

        # Load and transform each view in fixed order
        views = []
        for combo in self.allowed_combos:
            fname = combos[combo]
            path = os.path.join(self.image_dir, fname)
            img = Image.open(path)
            img_t = self.transform(img)
            views.append(img_t)
        views_tensor = torch.stack(views, dim=0)  # [4,3,H,W]

        # Compute bin index
        raw = data['target']
        bin_idx = 0 if raw == 0 else bin_value_quantile(raw, self.thresholds) + 1
        label = F.one_hot(torch.tensor(bin_idx), num_classes=self.num_bins).float()

        return views_tensor, label


if __name__ == "__main__":
    ds = TdludatasetTorchioFourView(
        image_dir='/fs/nexus-scratch/tuxunlu/git/tdlu/dataset/WUSTL_png_nomarker_16',
        csv_path='/fs/nexus-scratch/tuxunlu/git/tdlu/dataset/umd_annot_md_TDLU_y2025m04d28.csv',
        num_bins=2,
        target='tdlu_density',
        purpose='train'
    )
    print("Total subjects:", len(ds))
    v, l = ds[2]
    print("Views tensor:", v.shape, "Label:", l)
