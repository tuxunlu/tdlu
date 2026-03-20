import argparse
import os

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tdlu.data.TDLUDataset_torchio_four_view_mask_meta_avg_10_folds_STAMP import (
    TdludatasetTorchioFourViewMaskMetaAvg10FoldsStamp,
)


def build_dataset(cfg: dict, purpose: str):
    """Instantiate the STAMP TDLU dataset, using only its metadata and label outputs."""
    split_ratio = cfg.get("split_ratio", [0.8, 0.1, 0.1])

    dataset = TdludatasetTorchioFourViewMaskMetaAvg10FoldsStamp(
        image_dir=cfg["image_dir"],
        dense_mask_dir=cfg["dense_mask_dir"],
        breast_mask_dir=cfg.get("breast_mask_dir"),
        csv_path=cfg["csv_path"],
        num_bins=cfg["num_bins"],
        target=cfg["target"],
        meta_cols=cfg["meta_cols"],
        label_type=cfg["label_type"],
        purpose=purpose,
        cross_val_fold=cfg.get("cross_val_fold", None),
        num_folds=cfg.get("num_folds", 5),
        split_ratio=tuple(split_ratio),
        use_augmentation=False,
    )
    return dataset


def dataset_to_numpy(ds):
    """Extract metadata features and labels from a dataset as numpy arrays."""
    metas = []
    labels = []
    for idx in range(len(ds)):
        _, _, meta, label, _ = ds[idx]
        metas.append(meta.numpy())
        labels.append(int(label.item()))
    X = np.stack(metas, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


def run_logistic_regression(cfg: dict):
    """Fit and evaluate a multinomial logistic regression on metadata only."""
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Building datasets...")
    train_ds = build_dataset(cfg, purpose="train")
    val_ds = build_dataset(cfg, purpose="validation")
    test_ds = build_dataset(cfg, purpose="test")

    print("Extracting metadata features and labels...")
    X_train, y_train = dataset_to_numpy(train_ds)
    X_val, y_val = dataset_to_numpy(val_ds)
    X_test, y_test = dataset_to_numpy(test_ds)

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )

    print("Fitting logistic regression on train metadata...")
    clf.fit(X_train, y_train)

    def evaluate(split_name, X, y):
        y_pred = clf.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"\n=== {split_name} ===")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion matrix:")
        print(confusion_matrix(y, y_pred))
        print("\nClassification report:")
        print(classification_report(y, y_pred, digits=4))

    evaluate("Train", X_train, y_train)
    evaluate("Validation", X_val, y_val)
    evaluate("Test", X_test, y_test)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run logistic regression to predict TDLU class from metadata only."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="tdlu/config/config_torchio_3bins_tdlu_stamp_one_view_10folds_zero_extreme_meta_avg_cross_attention.yaml",
        help="Path to YAML config used for the STAMP TDLU dataset.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    required_keys = [
        "image_dir",
        "dense_mask_dir",
        "csv_path",
        "num_bins",
        "target",
        "meta_cols",
        "label_type",
    ]
    for k in required_keys:
        if k not in cfg:
            raise KeyError(f"Required key '{k}' missing from config {path}")
    return cfg


def main():
    args = parse_args()
    config_path = args.config

    if not os.path.isabs(config_path):
        # Interpret relative paths from repository root where this script lives.
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, config_path)

    print(f"Loading config from: {config_path}")
    cfg = load_config(config_path)
    run_logistic_regression(cfg)


if __name__ == "__main__":
    main()

