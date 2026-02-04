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
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from torch.nn import ModuleList
from torchvision.transforms import ConvertImageDtype, Lambda
import torchio as tio
import scipy.io as sio
from utils.ToTensor16RGB import ToTensor16RGB

import matplotlib.pyplot as plt

class JointTioAug:
    """
    Comprehensive augmentation strategy for mammogram classification.
    Designed to preserve anatomical structures while adding robustness to:
    - Acquisition variations (intensity, contrast, noise)
    - Patient positioning (small rotations, translations)
    - Equipment differences (bias field, blur)
    """
    def __init__(self, size=(1024, 1024), augmentation_strength='moderate'):
        """
        Args:
            size: Target image size
            augmentation_strength: 'mild', 'moderate', or 'strong'
        """
        self.size = size
        self.augmentation_strength = augmentation_strength
        
        # Spatial transforms - mild to preserve anatomical structures
        if augmentation_strength == 'mild':
            spatial_augs = [
                tio.RandomFlip(axes=(1,), p=0.5),  # Horizontal flip (medically reasonable)
                tio.RandomAffine(
                    scales=(0.95, 1.05),  # Very small scaling
                    degrees=(-5, 5),  # Small rotation
                    translation=(10, 10, 0),  # Small translation
                    image_interpolation='linear',
                    p=0.4,
                ),
            ]
        elif augmentation_strength == 'moderate':
            spatial_augs = [
                tio.RandomFlip(axes=(1,), p=0.5),  # Horizontal flip
                tio.RandomAffine(
                    scales=(0.92, 1.08),  # Moderate scaling
                    degrees=(-10, 10),  # Moderate rotation
                    translation=(15, 15, 0),  # Moderate translation
                    image_interpolation='linear',
                    p=0.5,
                ),
            ]
        else:  # strong
            spatial_augs = [
                tio.RandomFlip(axes=(1,), p=0.5),
                tio.RandomAffine(
                    scales=(0.90, 1.10),  # Larger scaling
                    degrees=(-15, 15),  # Larger rotation
                    translation=(20, 20, 0),
                    image_interpolation='linear',
                    p=0.6,
                ),
            ]
        
        self.spatial = tio.Compose(spatial_augs)
        
        # Intensity transforms - simulate acquisition variations
        if augmentation_strength == 'mild':
            intensity_augs = [
                tio.RandomGamma(log_gamma=(-0.15, 0.15), p=0.4),  # Contrast adjustment
                tio.RandomNoise(std=(0, 0.015), p=0.3),  # Small noise
                tio.RandomBlur(std=(0.25, 0.5), p=0.3),  # Mild blur
            ]
        elif augmentation_strength == 'moderate':
            intensity_augs = [
                tio.RandomBiasField(coefficients=(0.1, 0.3), p=0.3),  # Bias field (common in MRI, mild for mammo)
                tio.RandomGamma(log_gamma=(-0.2, 0.2), p=0.5),  # Contrast adjustment
                tio.RandomNoise(std=(0, 0.02), p=0.4),  # Moderate noise
                tio.RandomBlur(std=(0.25, 0.75), p=0.4),  # Moderate blur
            ]
        else:  # strong
            intensity_augs = [
                tio.RandomBiasField(coefficients=(0.1, 0.4), p=0.4),
                tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.6),
                tio.RandomNoise(std=(0, 0.03), p=0.5),
                tio.RandomBlur(std=(0.25, 1.0), p=0.5),
            ]
        
        self.intensity = tio.Compose(intensity_augs)

    def __call__(self, img_tensor_CHW, dense_mask_tensor_HW, breast_mask_tensor_HW):
        # TorchIO expects [C, H, W, D]; make D=1 for 2D
        img4d  = img_tensor_CHW.unsqueeze(-1)                 # [C,H,W,1]
        dense_mask4d = dense_mask_tensor_HW.unsqueeze(0).unsqueeze(-1)  # [1,H,W,1]
        breast_mask4d = breast_mask_tensor_HW.unsqueeze(0).unsqueeze(-1)  # [1,H,W,1]

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=img4d),
            dense_mask=tio.LabelMap(tensor=dense_mask4d),
            breast_mask=tio.LabelMap(tensor=breast_mask4d)
        )
        subject = self.spatial(subject)        # same spatial params shared
        subject.image = self.intensity(subject.image)      # applied to image only

        img_aug  = subject.image.data.squeeze(-1)             # back to [C,H,W]
        dense_mask_aug = subject.dense_mask.data.squeeze(0).squeeze(-1)   # back to [H,W]
        dense_mask_aug = (dense_mask_aug > 0).to(torch.bool) 
        breast_mask_aug = subject.breast_mask.data.squeeze(0).squeeze(-1)   # back to [H,W]
        breast_mask_aug = (breast_mask_aug > 0).to(torch.bool) 
        return img_aug, dense_mask_aug, breast_mask_aug


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
        elif label_type == "raw":
            bin_idx = bin_value_quantile(raw, thresholds)
        else:
            raise ValueError("label_type must be 'label' or 'raw'")

        bins.append(bin_idx)

    # 2) Count how many samples in each class
    counts = Counter(bins)
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


class TdludatasetTorchioFourViewMaskMetaAvg10FoldsHologic(Dataset):
    """
    Dataset for four-view mammograms: CC/L, MLO/L, MLO/R, CC/R.
    Supports optional 10-fold cross-validation with separate validation and test folds.
    Accepts `cross_val_fold` for test fold and optional `val_fold` for validation fold (0-9).
    Returns views_tensor, meta, label, file_names
    """
    def __init__(
        self,
        image_dir: str,
        dense_mask_dir: str,
        breast_mask_dir: str,
        csv_path: str,
        num_bins: int,
        target: str,
        meta_cols: list,
        label_type: str,
        purpose: str = 'train',  # 'train', 'validation', or 'test'
        cross_val_fold: int = None,  # test fold index 0-9
        split_ratio=(0.8, 0.1, 0.1),  # used if cross_val_fold is None
        use_augmentation: bool = True,
        augmentation_strength: str = 'moderate'  # 'mild', 'moderate', or 'strong'
    ):
        self.image_dir = image_dir
        self.dense_mask_dir = dense_mask_dir
        self.breast_mask_dir = breast_mask_dir
        self.num_bins = num_bins
        self.target = target
        self.meta_cols = meta_cols
        self.label_type = label_type
        self.purpose = purpose
        self.use_augmentation = use_augmentation
        self.augmentation_strength = augmentation_strength

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
        # Include only Hologic rows
        df = df[df['Manufacturer'] == 'HOLOGIC  Inc.']

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

            # 3) extract the 3 shared meta-features
            subj_meta = group.iloc[0][self.meta_cols].astype(float).values  # → shape (3,)

            all_data[sid] = {
                'combos': combos,
                'target': raw_target,
                'subject_meta': subj_meta
            }


        # Cross-validation splitting (stratified by label)
        assert 0 <= cross_val_fold < 10, "cross_val_fold must be in [0,9]"
        subs = list(all_data.keys())
        labels = [int(all_data[sid]['target']) for sid in subs]
        rng = np.random.RandomState(1234)
        subs = np.array(subs)
        labels = np.array(labels)
        folds = [[] for _ in range(10)]
        for cls in np.unique(labels):
            cls_mask = labels == cls
            cls_ids = subs[cls_mask]
            rng.shuffle(cls_ids)
            split = np.array_split(cls_ids, 10)
            for i in range(10):
                folds[i].extend(split[i].tolist())
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
        # Initialize augmentation with specified strength
        self.joint_aug = JointTioAug(size=(1024, 1024), augmentation_strength=self.augmentation_strength)

        train_meta_values = []
        train_race_values = []
        for sid in train_ids:
            subj_meta = all_data[sid]['subject_meta']   # [BreastDensity, mamm_age, cbmi, geanc_Race]
            # first three are continuous
            train_meta_values.append(subj_meta[:3].astype(float))
            train_race_values.append(int(subj_meta[3]))

        train_meta_array = np.stack(train_meta_values)   # [N_train, 3]

        # Calculate mean and std for continuous features only
        self.meta_mean = torch.tensor(np.mean(train_meta_array, axis=0), dtype=torch.float32)  # [3]
        self.meta_std  = torch.tensor(np.std(train_meta_array,  axis=0), dtype=torch.float32)  # [3]

        # Number of race categories (1–7)
        self.num_race_classes = 7
        
        self.class_weights = compute_class_weights(
            self.subjects,
            self.subject_data,
            self.thresholds,
            self.label_type
        )
        print(f"{self.purpose} Class weights: {self.class_weights}")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        sid = self.subjects[idx]
        data = self.subject_data[sid]
        views = []
        dense_masks = []
        breast_masks = []
        file_names = []
        for combo in self.allowed_combos:
            fname = data['combos'][combo]          # get the filename

            img_path = os.path.join(self.image_dir, fname)
            img = Image.open(img_path)
            img = TF.convert_image_dtype(transforms.PILToTensor()(img), torch.float32)

            # # print range of img before normalize
            # print(f"Image min/max before normalize: {img.min().item():.3f}/{img.max().item():.3f}")

            img  = TF.resize(img, (1024, 1024), interpolation=InterpolationMode.BILINEAR)
            if self.purpose == 'train' and self.use_augmentation:
                # Create dummy masks (zeros) because this dataset is mammogram-only
                dummy_mask = torch.zeros(1024, 1024, dtype=torch.bool)
                img, _, _ = self.joint_aug(img, dummy_mask, dummy_mask)
            else:
                # still ensure same resize rule when no random augs:
                pass

            
            img = img.repeat(3, 1, 1)
            # Use ImageNet normalization to stay aligned with pretrained ResNet weights
            img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            # # print range of img after normalize
            # print(f"Image min/max after normalize: {img.min().item():.3f}/{img.max().item():.3f}")

            views.append(img)
            file_names.append(fname)
        
        views_tensor = torch.stack(views, 0)
        masks_tensor = torch.zeros(views_tensor.size(0), 2, views_tensor.size(2), views_tensor.size(3))

        raw = data['target']

        
        if self.label_type == "raw":
            bin_idx = bin_value_quantile(raw, self.thresholds)
        elif self.label_type == "label":
            # Provided raw data are labels already
            bin_idx = raw
        elif self.label_type == "zero":
            # Provided raw data are NOT labels and need to reserve zeros as a separate class
            bin_idx = 0 if raw == 0 else bin_value_quantile(raw, self.thresholds) + 1
        else:
            # Provided raw data are NOT labels
            bin_idx = bin_value_quantile(raw, self.thresholds)

        label = torch.tensor(bin_idx, dtype=torch.long)

                # subject_meta is [BreastDensity_avg, mamm_age, cbmi_donation, geanc_Race]
        subj_meta = data['subject_meta']

        # ---- continuous features (3 dims) ----
        cont_np = np.asarray(subj_meta[:3], dtype=np.float32)   # [3]
        cont = torch.from_numpy(cont_np)                        # [3]

        cont_norm = (cont - self.meta_mean) / self.meta_std     # [3]

        # ---- categorical race (1–7) as one-hot ----
        race_int = int(subj_meta[3])
        # convert {1,…,7} → {0,…,6}
        race_idx = race_int - 1
        if race_idx < 0 or race_idx >= self.num_race_classes:
            raise ValueError(f"geanc_Race out of range: {race_int}")

        race_one_hot = F.one_hot(
            torch.tensor(race_idx, dtype=torch.long),
            num_classes=self.num_race_classes,
        ).to(torch.float32)                                     # [7]

        # ---- final meta vector: [3 continuous (z-scored) + 7 one-hot] ----
        meta = torch.cat([cont_norm, race_one_hot], dim=0)      # [10]


        return views_tensor, masks_tensor, meta, label, file_names

if __name__ == "__main__":
    train_ds = TdludatasetTorchioFourViewMaskMetaAvg10Folds(
        image_dir='/beacon-scratch/tuxunlu/git/tdlu/KOMEN/WUSTL_png_nomarker_16',
        dense_mask_dir = '/beacon-scratch/tuxunlu/git/tdlu/KOMEN/LIBRA_Masks_npy',
        breast_mask_dir = '/beacon-scratch/tuxunlu/git/tdlu/KOMEN/WUSTL_png_nomarker_16_contour',
        csv_path='/beacon-scratch/tuxunlu/git/tdlu/KOMEN/umd_annot_md_TDLU_y2025m07d09.csv',
        num_bins=2,
        target='tdlu_density_extreme',
        label_type='label',
        meta_cols=['mamm_age', 'cbmi_donation', 'geanc_Race'],
        purpose='train',
        cross_val_fold=0,
    )

    val_ds = TdludatasetTorchioFourViewMaskMetaAvg10Folds(
        image_dir='/beacon-scratch/tuxunlu/git/tdlu/KOMEN/WUSTL_png_nomarker_16',
        dense_mask_dir = '/beacon-scratch/tuxunlu/git/tdlu/KOMEN/LIBRA_Masks_npy',
        breast_mask_dir = '/beacon-scratch/tuxunlu/git/tdlu/KOMEN/WUSTL_png_nomarker_16_contour',
        csv_path='/beacon-scratch/tuxunlu/git/tdlu/KOMEN/umd_annot_md_TDLU_y2025m07d09.csv',
        num_bins=3,
        target='BreastDensity_avg_four',
        label_type='raw',
        meta_cols=['mamm_age', 'cbmi_donation', 'geanc_Race'],
        purpose='validation',
        cross_val_fold=0,
    )

    idx=10
    
    views, masks, meta, label, filenames = train_ds[idx]
    print("Train: ", views.shape, masks.shape, meta, label, filenames)
    exit()

    views, masks, meta, label, filenames = val_ds[idx]
    print("Validation: ", views.shape, masks.shape, meta, label, filenames)

 
    # statistics used in your Normalize
    mean = torch.tensor([0.113, 0.113, 0.113])[:, None, None]
    std  = torch.tensor([0.185, 0.185, 0.185])[:, None, None]

    fig, axes = plt.subplots(3, len(views), figsize=(12,8))
    for i, (v, m, fname) in enumerate(zip(views, masks, filenames)):
        # un-normalize
        img = v * std + mean               # still [3,H,W]
        img = img.clamp(0,1)               # ensure in [0,1]
        img = img.permute(1,2,0).cpu().numpy()  # → H×W×3

        # if you want grayscale, average the channels:
        gray = img.mean(-1)

        axes[0, i].imshow(gray, cmap='gray')
        axes[0, i].set_title(fname.split('.')[0])
        axes[0, i].axis('off')

        axes[1, i].imshow(m[0], cmap='gray')
        axes[1, i].set_title(fname.split('.')[0])
        axes[1, i].axis('off')

        axes[2, i].imshow(m[1], cmap='gray')
        axes[2, i].set_title(fname.split('.')[0])
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig('train views.png', dpi=300)


    views, masks, meta, label, filenames = val_ds[idx]
    print("Validation: ", views.shape, masks.shape, meta, label, filenames)
    # statistics used in your Normalize
    mean = torch.tensor([0.113, 0.113, 0.113])[:, None, None]
    std  = torch.tensor([0.185, 0.185, 0.185])[:, None, None]

    fig, axes = plt.subplots(3, len(views), figsize=(16,4))
    for i, (v, m, fname) in enumerate(zip(views, masks, filenames)):
        # un-normalize
        img = v * std + mean               # still [3,H,W]
        img = img.clamp(0,1)               # ensure in [0,1]
        img = img.permute(1,2,0).cpu().numpy()  # → H×W×3

        # if you want grayscale, average the channels:
        gray = img.mean(-1)

        axes[0, i].imshow(gray, cmap='gray')
        axes[0, i].set_title(fname.split('.')[0])
        axes[0, i].axis('off')

        axes[1, i].imshow(m[0], cmap='gray')
        axes[1, i].set_title(fname.split('.')[0])
        axes[1, i].axis('off')

        axes[2, i].imshow(m[1], cmap='gray')
        axes[2, i].set_title(fname.split('.')[0])
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig('val views.png', dpi=300)
