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
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
import torchio as tio
import matplotlib.pyplot as plt

class JointTioAug:
    def __init__(self, size=(1024, 1024), random_crop_size=(896, 1024)):
        """
        Args:
            size: Output spatial size (H, W).
            random_crop_size: (min_crop, max_crop) or single int for random crop.
                Crop a random patch of size in [min_crop, max_crop] then resize to size.
                Use None to disable random crop.
        """
        self.size = size
        self.random_crop_size = random_crop_size
        # spatial transforms shared by image & mask
        self.spatial = tio.Compose([
            tio.RandomFlip(axes=(1,), p=0.5),                 # horizontal flip
            # tio.RandomAffine(
            #     scales=(0.85, 1.15),
            #     degrees=(-30, 30),                             # rotate around z
            #     translation=(15, 15, 0),                          # tweak if needed
            #     image_interpolation='linear',
            #     label_interpolation='nearest',
            #     p=0.3,
            # ),
        ])
        # intensity-only transforms (auto-skip LabelMap)
        # FIX: Uncommented to prevent texture/intensity memorization
        self.intensity = tio.Compose([
            tio.RandomBiasField(p=0.1),
            tio.RandomGamma(log_gamma=(-0.2, 0.2), p=0.3),
            tio.RandomNoise(std=(0, 0.01), p=0.5),
            tio.RandomBlur(std=(0.25, 0.75), p=0.3),
        ])

    def __call__(self, img_tensor_CHW, dense_mask_tensor_HW, breast_mask_tensor_HW=None):
        # TorchIO expects [C, H, W, D]; make D=1 for 2D
        img4d  = img_tensor_CHW.unsqueeze(-1)                 # [C,H,W,1]
        dense_mask4d = dense_mask_tensor_HW.unsqueeze(-1)  # [1,H,W,1]
        
        # Handle optional breast_mask
        if breast_mask_tensor_HW is not None:
            breast_mask4d = breast_mask_tensor_HW.unsqueeze(0).unsqueeze(-1)  # [1,H,W,1]
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=img4d),
                dense_mask=tio.LabelMap(tensor=dense_mask4d),
                breast_mask=tio.LabelMap(tensor=breast_mask4d)
            )
        else:
            # If no breast_mask, only use dense_mask
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=img4d),
                dense_mask=tio.LabelMap(tensor=dense_mask4d)
            )
        
        subject = self.spatial(subject)        # same spatial params shared
        subject.image = self.intensity(subject.image)      # applied to image only

        img_aug  = subject.image.data.squeeze(-1)             # back to [C,H,W]
        dense_mask_aug = subject.dense_mask.data.squeeze(0).squeeze(-1)   # back to [H,W]
        dense_mask_aug = (dense_mask_aug > 0).to(torch.bool) 
        
        if breast_mask_tensor_HW is not None:
            breast_mask_aug = subject.breast_mask.data.squeeze(0).squeeze(-1)   # back to [H,W]
            breast_mask_aug = (breast_mask_aug > 0).to(torch.bool)
        else:
            # Create a dummy breast_mask (all True) if not provided
            breast_mask_aug = torch.ones_like(dense_mask_aug, dtype=torch.bool)

        # Random crop: crop a random patch then resize back (same crop for image & masks)
        if self.random_crop_size is not None:
            if isinstance(self.random_crop_size, (tuple, list)):
                crop_min, crop_max = self.random_crop_size[0], self.random_crop_size[1]
            else:
                crop_min = crop_max = self.random_crop_size
            crop_sz = random.randint(crop_min, crop_max) if crop_min < crop_max else crop_min
            h, w = img_aug.shape[-2:]
            if crop_sz < min(h, w):
                top = random.randint(0, h - crop_sz)
                left = random.randint(0, w - crop_sz)
                img_aug = img_aug[..., top:top + crop_sz, left:left + crop_sz]
                dense_mask_aug = dense_mask_aug[..., top:top + crop_sz, left:left + crop_sz]
                breast_mask_aug = breast_mask_aug[..., top:top + crop_sz, left:left + crop_sz]
                target_h, target_w = self.size
                img_aug = F.interpolate(
                    img_aug.unsqueeze(0), size=(target_h, target_w),
                    mode='bilinear', align_corners=False
                ).squeeze(0)
                dense_mask_aug = F.interpolate(
                    dense_mask_aug.unsqueeze(0).unsqueeze(0).float(),
                    size=(target_h, target_w), mode='nearest'
                ).squeeze(0).squeeze(0).to(torch.bool)
                breast_mask_aug = F.interpolate(
                    breast_mask_aug.unsqueeze(0).unsqueeze(0).float(),
                    size=(target_h, target_w), mode='nearest'
                ).squeeze(0).squeeze(0).to(torch.bool)
        
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

    counts = Counter(bins)
    num_classes = max(counts.keys()) + 1  # assumes classes are 0..C-1

    total = len(bins)
    class_weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 0)
        if cnt > 0:
            w = total / (num_classes * cnt)
        else:
            w = 0.0
        class_weights.append(w)
    
    return class_weights


class TdludatasetTorchioFourViewMaskMetaAvg10FoldsStamp(Dataset):
    """
    Dataset for STAMP mammograms: CC/L or CC/R (MLO views excluded).
    Note: Each studyid only has one CC view (either CC/L or CC/R), not both.
    """
    def __init__(
        self,
        image_dir: str,
        dense_mask_dir: str,
        breast_mask_dir: str = None,  
        csv_path: str = None,
        num_bins: int = None,
        target: str = None,
        meta_cols: list = None,
        label_type: str = None,
        purpose: str = 'train', 
        cross_val_fold: int = None, 
        num_folds: int = 5, 
        split_ratio=(0.7, 0.15, 0.15), 
        use_augmentation: bool = True
    ):
        if csv_path is None or num_bins is None or target is None or meta_cols is None or label_type is None:
            raise ValueError("csv_path, num_bins, target, meta_cols, and label_type are required")
        self.image_dir = image_dir
        self.dense_mask_dir = dense_mask_dir
        self.breast_mask_dir = breast_mask_dir 
        self.num_bins = num_bins
        self.target = target
        self.meta_cols = meta_cols
        self.label_type = label_type
        self.purpose = purpose
        self.use_augmentation = use_augmentation
        self.split_ratio = split_ratio

        self.allowed_combos = [
            ('CC', 'L'),
            ('CC', 'R'),
        ]

        df = pd.read_csv(csv_path)
        df = df[df['ViewPosition'] != 'MLO']
        df['ImageLaterality'] = df['breastside_final'].map({'Left': 'L', 'Right': 'R'})
        df['subject_id'] = df['studyid'].astype(str)
        df['filename'] = df['Transformed_file_namenew']
        
        df['geanc_Race'] = df['RACE_COMPOSITE']

        df = df[df[['ViewPosition', 'ImageLaterality']]
                .apply(lambda r: (r['ViewPosition'], r['ImageLaterality']) in self.allowed_combos, axis=1)]

        df['stem'] = df['filename'].str.replace(r"\.[^.]+$", "", regex=True)

        meta_df = df[['subject_id'] + self.meta_cols].drop_duplicates('subject_id')
        meta_map = {row.subject_id: row[self.meta_cols].values.astype(float) for _, row in meta_df.iterrows()}

        vals = pd.to_numeric(df[target], errors='coerce')
        
        if self.label_type == "zero":
            num_bins = num_bins - 1
            valid = vals[(vals.notna()) & (vals > 0)]
        else:
            valid = vals[(vals.notna()) & (vals > 0)]

        quantiles = [i / num_bins for i in range(1, num_bins)]
        self.thresholds = valid.quantile(quantiles).tolist()

        all_data = {}
        for sid, group in df.groupby('subject_id'):
            combos = {}
            for _, row in group.iterrows():
                key = (row['ViewPosition'], row['ImageLaterality'])
                fname = row['filename']
                if not fname.endswith('.png'):
                    fname = fname + '.png'
                combos[key] = fname

            if len(combos) == 0:
                continue
            if not any(k in self.allowed_combos for k in combos.keys()):
                continue
            
            raw_target = float(group.iloc[0][target])
            if np.isnan(raw_target):
                continue

            subj_meta_series = pd.to_numeric(group.iloc[0][self.meta_cols], errors='coerce')
            subj_meta = subj_meta_series.values.astype(float)
            if not np.isfinite(subj_meta).all():
                continue

            all_data[sid] = {
                'combos': combos,
                'target': raw_target,
                'subject_meta': subj_meta
            }

        subs = list(all_data.keys())
        random.seed(1234)
        random.shuffle(subs)
        n = len(subs)

        # Fold-based split:
        # - `cross_val_fold` selects the test fold.
        # - validation is the next fold (cyclic).
        # - train is everything else.
        #
        # If `cross_val_fold` is not provided, fall back to `split_ratio`.
        if cross_val_fold is not None:
            if not (0 <= cross_val_fold < num_folds):
                raise ValueError(
                    f"cross_val_fold must be in [0, {num_folds - 1}], got {cross_val_fold}"
                )

            folds = np.array_split(subs, num_folds)
            test_fold_idx = int(cross_val_fold)
            val_fold_idx = (test_fold_idx + 1) % num_folds

            test_ids = list(folds[test_fold_idx])
            val_ids = list(folds[val_fold_idx])

            train_ids: list = []
            for i in range(num_folds):
                if i in (test_fold_idx, val_fold_idx):
                    continue
                train_ids.extend(list(folds[i]))

            if len(train_ids) == 0:
                raise ValueError(
                    "Train split is empty. Check that `num_folds` is large enough for the "
                    f"requested `cross_val_fold`={cross_val_fold}."
                )

            if purpose == 'train':
                selected = train_ids
            elif purpose == 'validation':
                selected = val_ids
            elif purpose == 'test':
                selected = test_ids
            else:
                raise ValueError("purpose must be 'train', 'validation', or 'test'")
        else:
            train_ratio, val_ratio, test_ratio = self.split_ratio
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            n_test = n - n_train - n_val

            train_ids = list(subs[:n_train])
            val_ids = list(subs[n_train:n_train + n_val])
            test_ids = list(subs[n_train + n_val:])

            if purpose == 'train':
                selected = train_ids
            elif purpose == 'validation':
                selected = val_ids
            elif purpose == 'test':
                selected = test_ids
            else:
                raise ValueError("purpose must be 'train', 'validation', or 'test'")

        self.subjects = selected
        self.subject_data = {sid: all_data[sid] for sid in self.subjects}
        self.joint_aug = JointTioAug(size=(1024, 1024))

        train_meta_values = []
        train_race_values = []
        for sid in train_ids:
            subj_meta = all_data[sid]['subject_meta'] 
            # FIX: Dynamically handle N continuous columns + 1 categorical column
            n_cont = len(subj_meta) - 1
            train_meta_values.append(subj_meta[:n_cont].astype(float))
            train_race_values.append(int(subj_meta[-1]))

        train_meta_array = np.stack(train_meta_values)

        self.meta_mean = torch.tensor(np.mean(train_meta_array, axis=0), dtype=torch.float32)
        self.meta_std  = torch.tensor(np.std(train_meta_array,  axis=0), dtype=torch.float32)
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
        
        available_combos = list(data['combos'].keys())
        
        for combo in available_combos:
            fname = data['combos'][combo] 

            fname_base = fname.replace('.png', '')
            dense_mask_path = os.path.join(self.dense_mask_dir, f"{fname_base}_mask.png")
            dense_mask = Image.open(dense_mask_path)
            dense_mask = TF.convert_image_dtype(transforms.PILToTensor()(dense_mask), torch.bool)
            
            if self.breast_mask_dir is not None:
                breast_masks_path = os.path.join(self.breast_mask_dir, f"{fname_base}_BreastMask.png")
                breast_mask = Image.open(breast_masks_path)
                breast_mask = TF.convert_image_dtype(transforms.PILToTensor()(breast_mask), torch.bool)
            else:
                breast_mask = None
            
            img_path = os.path.join(self.image_dir, fname)
            img = Image.open(img_path)
            img = TF.convert_image_dtype(transforms.PILToTensor()(img), torch.float32)

            img  = TF.resize(img, (1024, 1024), interpolation=InterpolationMode.BILINEAR)
            dense_mask = TF.resize(dense_mask.unsqueeze(0), (1024, 1024), interpolation=InterpolationMode.NEAREST).squeeze(0)
            if breast_mask is not None:
                breast_mask = TF.resize(breast_mask.unsqueeze(0), (1024, 1024), interpolation=InterpolationMode.NEAREST).squeeze(0)
            
            if self.purpose == 'train' and self.use_augmentation:
                if breast_mask is not None:
                    img, dense_mask, breast_mask = self.joint_aug(img, dense_mask, breast_mask)
                else:
                    img, dense_mask, _ = self.joint_aug(img, dense_mask, None)

            dense_mask = dense_mask.squeeze(0)
            if breast_mask is not None:
                breast_mask = breast_mask.squeeze(0)
            
            img = img.repeat(3, 1, 1)
            # FIX: Normalize with dataset's grayscale statistics, NOT ImageNet color tints
            img = TF.normalize(img, mean=[0.128, 0.128, 0.128], std=[0.235, 0.235, 0.235])

            views.append(img)
            dense_masks.append(dense_mask)
            breast_masks.append(breast_mask)
            file_names.append(fname)
        
        views_tensor = torch.stack(views, 0)
        dense_masks_tensor = torch.stack(dense_masks, 0)

        has_breast_mask = any(mask is not None for mask in breast_masks)
        if has_breast_mask:
            breast_masks_filled = [
                mask if mask is not None else torch.zeros_like(dense_masks[i], dtype=torch.bool)
                for i, mask in enumerate(breast_masks)
            ]
            breast_masks_tensor = torch.stack(breast_masks_filled, 0)
        else:
            breast_masks_tensor = None
            
        if breast_masks_tensor is not None:
            masks_tensor = torch.stack([dense_masks_tensor, breast_masks_tensor], dim=1).float()
        else:
            masks_tensor = dense_masks_tensor.unsqueeze(1).float()

        raw = data['target']

        if self.label_type == "raw":
            bin_idx = bin_value_quantile(raw, self.thresholds)
        elif self.label_type == "label":
            bin_idx = int(raw)
        elif self.label_type == "zero":
            bin_idx = 0 if raw == 0 else bin_value_quantile(raw, self.thresholds) + 1
        else:
            bin_idx = bin_value_quantile(raw, self.thresholds)

        label = torch.tensor(bin_idx, dtype=torch.long)

        subj_meta = data['subject_meta']

        # FIX: Dynamic continuous feature sizing
        n_cont = len(subj_meta) - 1
        cont_np = np.asarray(subj_meta[:n_cont], dtype=np.float32)
        cont = torch.from_numpy(cont_np)

        cont_norm = (cont - self.meta_mean) / self.meta_std

        race_int = int(subj_meta[-1])
        race_idx = race_int - 1
        if race_idx < 0 or race_idx >= self.num_race_classes:
            raise ValueError(f"geanc_Race out of range: {race_int}")

        race_one_hot = F.one_hot(
            torch.tensor(race_idx, dtype=torch.long),
            num_classes=self.num_race_classes,
        ).to(torch.float32)

        meta = torch.cat([cont_norm, race_one_hot], dim=0)

        return views_tensor, masks_tensor, meta, label, file_names
