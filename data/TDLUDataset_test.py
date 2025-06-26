import os
from collections import Counter

import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from utils.ToTensor16RGB import ToTensor16RGB
from tqdm import tqdm


def bin_value_quantile(value, thresholds):
    """
    Convert a non-zero numeric value into a bin index based on quantile thresholds.
    Returned index is zero-based, so later +1 for bins 1..N.
    """
    for i, t in enumerate(thresholds):
        if value <= t:
            return i
    return len(thresholds)


class TDLUDataset_test(Dataset):
    def __init__(
        self,
        root_dir: str,
        purpose: str,
        csv_path: str,
        num_bins: int,
        target: str,
        use_augmentation: bool = True,
        aug_prob: float = 0.5,
        mamm_only: bool = False,
    ):
        """
        root_dir:   path to folder containing 'train/', 'val/', 'test/'
        purpose:    one of 'train', 'validation', or 'test'
        csv_path:   CSV with ['filename', <target>, 'BreastDensity_avg', 'mamm_age', 'cbmi_donation', ...]
        num_bins:   total bins (0 reserved for zero)
        target:     column name to bin (e.g. 'tdlu_density')
        use_augmentation: apply spatial augment only if purpose=='train' and this is True
        mamm_only:  if True, __getitem__ returns just (img, onehot)
        """
        assert purpose in ("train", "validation", "test"), \
            "purpose must be 'train', 'validation', or 'test'"

        # map 'validation' to folder 'val'
        folder = purpose if purpose != "validation" else "val"
        self.image_dir = os.path.join(root_dir, folder)

        self.purpose = purpose
        self.num_bins = num_bins
        self.target = target
        self.use_augmentation = use_augmentation and (purpose == "train")
        self.aug_prob = aug_prob
        self.mamm_only = mamm_only

        # build transforms
        self.transforms = self._make_transforms()

        # load CSV & compute quantile thresholds
        df = pd.read_csv(csv_path)
        df["filename"] = df["filename"].astype(str)
        df["stem"] = df["filename"].str.replace(r"\.[^.]+$", "", regex=True)
        df_index = df.set_index("stem")

        non_zero_bins = num_bins - 1
        vals = pd.to_numeric(df[target], errors="coerce")
        non_zero = vals[(vals.notna()) & (vals != 0)]
        qs = [i / non_zero_bins for i in range(1, non_zero_bins)]
        self.quantile_thresholds = non_zero.quantile(qs).tolist()

        # gather files & labels
        all_png = sorted(f for f in os.listdir(self.image_dir) if f.lower().endswith(".png"))
        self.image_files = []
        self.label_map = {}
        self.density_map = {}
        self.age_map = {}
        self.bmi_map = {}

        for fn in all_png:
            stem = os.path.splitext(fn)[0]
            if stem not in df_index.index:
                continue

            row = df_index.loc[stem]
            raw = row[target]
            if raw == "N":
                continue

            val = float(raw)
            bin_idx = 0 if val == 0 else bin_value_quantile(val, self.quantile_thresholds) + 1

            self.image_files.append(fn)
            self.label_map[fn] = bin_idx
            self.density_map[fn] = float(row["BreastDensity_avg"])
            self.age_map[fn]     = float(row["mamm_age"])
            self.bmi_map[fn]     = float(row["cbmi_donation"])

        # sample weights for balanced loss
        counts = Counter(self.label_map.values())
        self.sample_weights = [1.0 / counts[self.label_map[f]] for f in self.image_files]

    def _make_transforms(self):
        mean = (0.111, 0.111, 0.111)
        std  = (0.185, 0.185, 0.185)

        ops = [
            transforms.Resize((1024, 1024)),
            ToTensor16RGB()
        ]

        if self.use_augmentation:
            spatial = [
                transforms.RandomRotation(degrees=(-180, 180), fill=0),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1,2.0))
            ]
            ops.append(transforms.RandomApply(spatial, p=self.aug_prob))

        ops.append(transforms.Normalize(mean, std))
        return transforms.Compose(ops)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fn  = self.image_files[idx]
        img = Image.open(os.path.join(self.image_dir, fn))
        img = self.transforms(img)

        bin_idx = self.label_map[fn]
        onehot  = F.one_hot(torch.tensor(bin_idx), num_classes=self.num_bins).float()

        if self.mamm_only:
            return img, onehot

        return (
            img,
            torch.tensor(self.density_map[fn], dtype=torch.float32),
            torch.tensor(self.age_map[fn],     dtype=torch.float32),
            torch.tensor(self.bmi_map[fn],     dtype=torch.float32),
            onehot
        )


def get_dataset_stats(dataset: Dataset, batch_size=32, num_workers=12):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    channel_sum = torch.zeros(3)
    channel_sq  = torch.zeros(3)
    total_pix   = 0

    for batch in tqdm(loader, desc="Recomputing stats"):
        imgs = batch[0] if isinstance(batch, (list,tuple)) else batch
        b, c, h, w = imgs.shape
        pix = h * w

        channel_sum += imgs.view(b, c, pix).sum(dim=(0,2))
        channel_sq  += (imgs.view(b, c, pix) ** 2).sum(dim=(0,2))
        total_pix   += pix * b

    means = channel_sum / total_pix
    vars_ = channel_sq / total_pix - means**2
    stds  = torch.sqrt(vars_)
    return means.tolist(), stds.tolist()


if __name__ == "__main__":
    ROOT = "/fs/nexus-scratch/tuxunlu/git/tdlu/dataset/WUSTL_dataset"
    CSV  = "/fs/nexus-scratch/tuxunlu/git/tdlu/dataset/umd_annot_md_TDLU_y2025m04d11.csv"

    train_dataset = TDLUDataset_test(
        root_dir=ROOT, purpose="train", csv_path=CSV,
        num_bins=4, target="tdlu_density", use_augmentation=True, mamm_only=True
    )

    val_dataset   = TDLUDataset_test(
        root_dir=ROOT, purpose="validation", csv_path=CSV,
        num_bins=4, target="tdlu_density", use_augmentation=False, mamm_only=True
    )

    test_dataset  = TDLUDataset_test(
        root_dir=ROOT, purpose="test", csv_path=CSV,
        num_bins=4, target="tdlu_density", use_augmentation=False, mamm_only=True
    )

    print("Sizes →", len(train_dataset), len(val_dataset), len(test_dataset))

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
