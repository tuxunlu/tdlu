import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image


ALLOWED_COMBOS = {
    ("CC", "L"),
    ("MLO", "L"),
    ("MLO", "R"),
    ("CC", "R"),
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_subject_table(csv_path: str) -> pd.DataFrame:
    """
    Apply the same filtering logic as
    TdludatasetTorchioFourViewMaskMetaAvg10FoldsHologic to build a
    subject-level table with BreastDensity_avg and tdlu_density_extreme_zero.
    """
    df = pd.read_csv(csv_path)

    # Match dataset filters
    df = df[df["geanc_Race"] != "N"]
    df = df[df["Manufacturer"] == "HOLOGIC  Inc."]

    df = df[
        df[["ViewPosition_comp2", "ImageLaterality_comp2"]]
        .apply(
            lambda r: (r["ViewPosition_comp2"], r["ImageLaterality_comp2"])
            in ALLOWED_COMBOS,
            axis=1,
        )
    ]

    df["subject_id"] = df["subject_id"].astype(str)

    subjects = []
    for sid, group in df.groupby("subject_id"):
        combos = {
            (row["ViewPosition_comp2"], row["ImageLaterality_comp2"])
            for _, row in group.iterrows()
        }
        if combos != ALLOWED_COMBOS:
            continue

        row0 = group.iloc[0]
        try:
            bd = float(row0["BreastDensity_avg"])
            tdlu = float(row0["tdlu_density_extreme_zero"])
        except (KeyError, ValueError):
            continue

        if np.isnan(bd) or np.isnan(tdlu):
            continue

        subjects.append(
            {
                "subject_id": sid,
                "BreastDensity_avg": bd,
                "tdlu_density_extreme_zero": tdlu,
            }
        )

    subj_df = pd.DataFrame(subjects)
    return subj_df


def summarize_and_correlation(subj_df: pd.DataFrame) -> None:
    print("=== Subject-level counts ===")
    print(f"N subjects: {len(subj_df)}")
    print()

    print("=== BreastDensity_avg summary ===")
    print(subj_df["BreastDensity_avg"].describe())
    print()

    print("=== tdlu_density_extreme_zero summary ===")
    print(subj_df["tdlu_density_extreme_zero"].describe())
    print("Value counts:")
    print(subj_df["tdlu_density_extreme_zero"].value_counts().sort_index())
    print()

    corr_pearson = subj_df[["BreastDensity_avg", "tdlu_density_extreme_zero"]].corr(
        method="pearson"
    ).iloc[0, 1]
    corr_spearman = subj_df[["BreastDensity_avg", "tdlu_density_extreme_zero"]].corr(
        method="spearman"
    ).iloc[0, 1]

    print("=== Correlation between BreastDensity_avg and tdlu_density_extreme_zero ===")
    print(f"Pearson  r = {corr_pearson:.4f}")
    print(f"Spearman ρ = {corr_spearman:.4f}")
    print()


class SubjectImageDataset(torch.utils.data.Dataset):
    """
    Builds per-subject image tensors for shallow MLP baseline.
    Uses the same filtering / subject construction as the TorchIO dataset.
    For each subject it:
      - loads 4 PNG views,
      - resizes to 1024×1024,
      - converts to 3‑channel and ImageNet-normalized tensor.

    Returns:
      views: [4, 3, H, W]
      bd_bin: int in [0, num_bd_bins-1]
      tdlu_label: int (assumes original tdlu_density_extreme_zero is categorical/label‑like)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        full_csv: pd.DataFrame,
        image_dir: str,
        num_bd_bins: int = 3,
    ):
        self.image_dir = image_dir

        # Build mapping subject -> filenames for 4 views
        by_subject: Dict[str, Dict[Tuple[str, str], str]] = {}
        for sid, group in full_csv.groupby("subject_id"):
            combos = {}
            for _, row in group.iterrows():
                key = (row["ViewPosition_comp2"], row["ImageLaterality_comp2"])
                if key in ALLOWED_COMBOS:
                    combos[key] = row["filename"].replace(".dcm", ".png")
            if set(combos.keys()) == ALLOWED_COMBOS:
                by_subject[str(sid)] = combos

        self.subjects = []
        for _, row in df.iterrows():
            sid = row["subject_id"]
            if sid not in by_subject:
                continue
            self.subjects.append(
                {
                    "subject_id": sid,
                    "BreastDensity_avg": float(row["BreastDensity_avg"]),
                    "tdlu_density_extreme_zero": int(row["tdlu_density_extreme_zero"]),
                    "combos": by_subject[sid],
                }
            )

        # Build BD bins based on training distribution (using all subjects here)
        bds = np.array([s["BreastDensity_avg"] for s in self.subjects], dtype=np.float32)
        qs = [i / num_bd_bins for i in range(1, num_bd_bins)]
        self.bd_thresholds = np.quantile(bds, qs).tolist()

        self.num_bd_bins = num_bd_bins

        self.to_tensor = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )

    def __len__(self) -> int:
        return len(self.subjects)

    def _bin_bd(self, val: float) -> int:
        for i, t in enumerate(self.bd_thresholds):
            if val <= t:
                return i
        return self.num_bd_bins - 1

    def __getitem__(self, idx: int):
        entry = self.subjects[idx]
        combos = entry["combos"]

        imgs = []
        # fixed order to be consistent
        for combo in [("CC", "L"), ("MLO", "L"), ("MLO", "R"), ("CC", "R")]:
            fname = combos[combo]
            img_path = os.path.join(self.image_dir, fname)
            img = Image.open(img_path).convert("L")
            img = self.to_tensor(img)  # [1, H, W]
            img = img.repeat(3, 1, 1)  # [3, H, W]
            img = transforms.functional.normalize(
                img, mean=IMAGENET_MEAN, std=IMAGENET_STD
            )
            imgs.append(img)

        views = torch.stack(imgs, dim=0)  # [4, 3, H, W]
        bd_bin = self._bin_bd(entry["BreastDensity_avg"])
        tdlu_label = entry["tdlu_density_extreme_zero"]
        return views, bd_bin, tdlu_label


class ShallowMLP(nn.Module):
    def __init__(self, in_dim: int = 512, num_classes: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def extract_features_and_train(
    csv_path: str,
    image_dir: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_bd_bins: int = 3,
) -> None:
    full_df = pd.read_csv(csv_path)
    full_df = full_df[full_df["geanc_Race"] != "N"]
    full_df = full_df[full_df["Manufacturer"] == "HOLOGIC  Inc."]
    full_df = full_df[
        full_df[["ViewPosition_comp2", "ImageLaterality_comp2"]]
        .apply(
            lambda r: (r["ViewPosition_comp2"], r["ImageLaterality_comp2"])
            in ALLOWED_COMBOS,
            axis=1,
        )
    ]
    full_df["subject_id"] = full_df["subject_id"].astype(str)

    subj_df = build_subject_table(csv_path)

    dataset = SubjectImageDataset(subj_df, full_df, image_dir, num_bd_bins=num_bd_bins)

    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)), test_size=0.2, random_state=42, stratify=subj_df["tdlu_density_extreme_zero"]
    )

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True
    )

    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.eval()
    backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad = False

    mlp_bd = ShallowMLP(in_dim=512, num_classes=num_bd_bins).to(device)
    mlp_tdlu = ShallowMLP(in_dim=512, num_classes=3).to(device)

    crit = nn.CrossEntropyLoss()
    # Slightly smaller LR and cosine schedule for smoother convergence
    opt_bd = torch.optim.Adam(mlp_bd.parameters(), lr=5e-4)
    opt_tdlu = torch.optim.Adam(mlp_tdlu.parameters(), lr=5e-4)
    sched_bd = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_bd, T_max=20, eta_min=1e-5
    )
    sched_tdlu = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_tdlu, T_max=20, eta_min=1e-5
    )

    def run_epoch(loader, train: bool = True):
        if train:
            mlp_bd.train()
            mlp_tdlu.train()
        else:
            mlp_bd.eval()
            mlp_tdlu.eval()

        all_bd_true, all_bd_pred = [], []
        all_tdlu_true, all_tdlu_pred = [], []

        for views, bd_bin, tdlu_label in loader:
            B, V, C, H, W = views.shape
            views = views.to(device)
            bd_bin = bd_bin.to(device)
            tdlu_label = tdlu_label.to(device)

            with torch.no_grad():
                feats = backbone(views.view(B * V, C, H, W))
                feats = feats.view(B, V, -1).mean(dim=1)

            if train:
                opt_bd.zero_grad()
                opt_tdlu.zero_grad()

            logits_bd = mlp_bd(feats)
            logits_tdlu = mlp_tdlu(feats)

            loss_bd = crit(logits_bd, bd_bin)
            loss_tdlu = crit(logits_tdlu, tdlu_label)
            loss = loss_bd + loss_tdlu

            if train:
                loss.backward()
                opt_bd.step()
                opt_tdlu.step()

            all_bd_true.append(bd_bin.detach().cpu().numpy())
            all_bd_pred.append(logits_bd.argmax(dim=1).detach().cpu().numpy())
            all_tdlu_true.append(tdlu_label.detach().cpu().numpy())
            all_tdlu_pred.append(logits_tdlu.argmax(dim=1).detach().cpu().numpy())

        all_bd_true = np.concatenate(all_bd_true)
        all_bd_pred = np.concatenate(all_bd_pred)
        all_tdlu_true = np.concatenate(all_tdlu_true)
        all_tdlu_pred = np.concatenate(all_tdlu_pred)

        acc_bd = accuracy_score(all_bd_true, all_bd_pred)
        acc_tdlu = accuracy_score(all_tdlu_true, all_tdlu_pred)
        return acc_bd, acc_tdlu

    print("=== Training shallow MLP baseline on frozen ResNet18 features ===")
    for epoch in range(1, 21):
        train_acc_bd, train_acc_tdlu = run_epoch(train_loader, train=True)
        val_acc_bd, val_acc_tdlu = run_epoch(val_loader, train=False)
        sched_bd.step()
        sched_tdlu.step()
        print(
            f"Epoch {epoch:02d} | "
            f"train_acc_bd={train_acc_bd:.3f} val_acc_bd={val_acc_bd:.3f} | "
            f"train_acc_tdlu={train_acc_tdlu:.3f} val_acc_tdlu={val_acc_tdlu:.3f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze BreastDensity_avg vs tdlu_density_extreme_zero and train shallow MLP baselines."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/beacon-scratch/tuxunlu/git/tdlu/KOMEN/umd_annot_md_TDLU_y2025m07d09.csv",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/beacon-scratch/tuxunlu/git/tdlu/KOMEN/WUSTL_png_nomarker_16",
    )
    parser.add_argument(
        "--no_mlp",
        action="store_true",
        help="If set, only compute distributions and correlations, skip MLP training.",
    )
    args = parser.parse_args()

    subj_df = build_subject_table(args.csv_path)
    summarize_and_correlation(subj_df)

    if not args.no_mlp:
        extract_features_and_train(args.csv_path, args.image_dir)


if __name__ == "__main__":
    main()

