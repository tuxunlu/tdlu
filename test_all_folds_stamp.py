#!/usr/bin/env python3
"""
Evaluate STAMP model checkpoints across folds.
STAMP batch format: (views, masks, meta, label, file_names)
Model forward: model(views, masks, meta)
"""
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from data import DataInterface
from models import ModelInterface
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to base config YAML (e.g. config/config_torchio_3bins_tdlu_stamp_one_view_10folds_zero_extreme_meta_avg_concat.yaml)')
    for i in range(10):
        parser.add_argument(f'--checkpoint_{i}', type=str, default=None, help=f'Path to checkpoint for fold {i}')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load base config
    if not os.path.exists(args.config):
        print(f"Error: Config not found: {args.config}")
        return
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = {k.lower(): v for k, v in config.items()}

    # Collect folds with checkpoints
    run_configs = []
    for i in range(10):
        ckpt_path = getattr(args, f'checkpoint_{i}')
        if ckpt_path and os.path.exists(ckpt_path):
            run_configs.append((i, ckpt_path))

    if not run_configs:
        print("No fold checkpoints provided. Exiting.")
        return

    print(f"Found {len(run_configs)} folds to evaluate.")

    global_preds = []
    global_targets = []
    global_probs = []
    global_breast_density = []

    for fold_idx, ckpt_path in run_configs:
        print(f"\n--- Processing Fold {fold_idx} ---")
        print(f"Ckpt: {ckpt_path}")

        # Config for this fold (cross_val_fold for compatibility)
        fold_config = {**config, 'cross_val_fold': fold_idx}

        try:
            data_module = DataInterface(**fold_config)
            if hasattr(data_module, 'setup'):
                data_module.setup(stage='test')
            test_loader = data_module.test_dataloader()
        except Exception as e:
            print(f"Error setting up data for fold {fold_idx}: {e}")
            continue

        try:
            model_module = ModelInterface.load_from_checkpoint(
                ckpt_path, map_location=device, strict=False, **fold_config
            )
            model_module.to(device)
            model_module.eval()
        except Exception as e:
            print(f"Error loading model for fold {fold_idx}: {e}")
            continue

        fold_preds = []
        fold_targets = []
        fold_probs = []
        fold_densities = []

        print(f"Samples in test loader: {len(test_loader.dataset)}")

        with torch.no_grad():
            for batch in test_loader:
                # STAMP batch: (views, masks, meta, label, file_names)
                *test_input, label, file_names = batch
                test_input = [t.to(device) for t in test_input]

                # Forward: model(views, masks, meta)
                logits = model_module._extract_logits(model_module(*test_input))
                preds = logits.argmax(dim=1)
                probs = torch.softmax(logits, dim=1)

                # BreastDensity is typically 3rd meta col (index 2); use zeros if unavailable
                try:
                    meta = test_input[2]
                    if meta.shape[1] >= 3:
                        breast_density = meta[:, 2].cpu().numpy()
                    else:
                        breast_density = np.zeros(preds.shape[0])
                except Exception:
                    breast_density = np.zeros(preds.shape[0])

                fold_preds.append(preds.cpu().numpy())
                fold_targets.append(label.cpu().numpy())
                fold_probs.append(probs.cpu().numpy())
                fold_densities.append(breast_density)

        if len(fold_preds) > 0:
            global_preds.append(np.concatenate(fold_preds))
            global_targets.append(np.concatenate(fold_targets))
            global_probs.append(np.concatenate(fold_probs, axis=0))
            global_breast_density.append(np.concatenate(fold_densities))

    if not global_preds:
        print("No predictions generated. Exiting.")
        return

    all_preds = np.concatenate(global_preds)
    all_targets = np.concatenate(global_targets)
    all_probs = np.concatenate(global_probs, axis=0)
    all_breast_density = np.concatenate(global_breast_density)

    print("\n===========================================")
    print("--- Overall Evaluation (All Folds) ---")
    print("===========================================")
    print(f"Total samples evaluated: {len(all_targets)}")

    if all_targets.ndim > 1:
        all_targets = np.argmax(all_targets, axis=1)

    accuracy = (all_preds == all_targets).mean()
    print(f"Overall test accuracy: {accuracy:.4f}")
    f1 = f1_score(all_targets, all_preds, average='macro')
    print(f"Overall test F1 score: {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    num_bins = cm.shape[0]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_norm, cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.title(f"STAMP Overall Confusion Matrix ({len(run_configs)} Folds)")
    plt.colorbar()
    ticks = np.arange(num_bins)
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    thresh = cm_norm.max() / 2

    for i in range(num_bins):
        for j in range(num_bins):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            plt.text(
                j, i,
                f"{count}\n({pct:.1f}%)",
                ha='center', va='center',
                color='white' if cm_norm[i, j] > thresh else 'black'
            )

    plt.tight_layout()
    out_path = "overall_confusion_matrix_stamp.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

    # ROC curves (for binary or multiclass)
    num_classes = all_probs.shape[1]
    if num_classes == 2:
        y_true = all_targets
        y_score = all_probs[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"STAMP ROC (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], 'k--', label="Chance (AUC = 0.5)")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("STAMP Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("overall_roc_curve_stamp.png", dpi=300)
        plt.close()
        print(f"Saved overall_roc_curve_stamp.png (AUC: {roc_auc:.3f})")


if __name__ == '__main__':
    main()
