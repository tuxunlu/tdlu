#!/usr/bin/env python3
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from data import DataInterface
from models import ModelInterface
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score

def main():
    parser = ArgumentParser()
    
    # We define arguments for fold_0 to fold_9
    # NOTE: In your command, 'fold_X' is carrying the PATH to the config file.
    # 'checkpoint_X' is carrying the PATH to the checkpoint.
    for i in range(10):
        parser.add_argument(f'--fold_{i}', type=str, default=None, help=f'Path to hparams.yaml for fold {i}')
        parser.add_argument(f'--checkpoint_{i}', type=str, default=None, help=f'Path to checkpoint for fold {i}')
    
    args = parser.parse_args()

    # — Device selection —
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # — Collect Config/Checkpoint Pairs —
    # List of tuples: (fold_index, config_path, checkpoint_path)
    run_configs = []
    for i in range(10):
        config_path = getattr(args, f'fold_{i}')
        ckpt_path = getattr(args, f'checkpoint_{i}')
        
        if config_path is not None and ckpt_path is not None:
            run_configs.append((i, config_path, ckpt_path))
    
    if not run_configs:
        print("No fold/checkpoint pairs provided. Exiting.")
        return

    print(f"Found {len(run_configs)} folds to evaluate.")

    # — Global Accumulators —
    global_preds = []
    global_targets = []
    global_probs = []
    global_breast_density = []

    # — Loop over provided folds —
    for fold_idx, config_path, ckpt_path in run_configs:
        print(f"\n--- Processing Fold {fold_idx} ---")
        print(f"Config: {config_path}")
        print(f"Ckpt:   {ckpt_path}")

        # 1. Load the specific config for this fold
        if not os.path.exists(config_path):
            print(f"Error: Config file not found {config_path}")
            continue
            
        with open(config_path) as f:
            # hparams.yaml usually has a structure, we assume simple key-values or keys at root
            config = yaml.safe_load(f)
        
        # Normalize config keys
        config = {k.lower(): v for k,v in config.items()}
        
        # **CRITICAL**: Enforce the fold index so DataInterface loads the correct test split.
        # We assume 'fold_0' argument corresponds to the data split 0.
        config['fold'] = fold_idx 

        # 2. Setup Data Module
        try:
            data_module = DataInterface(**config)
            # Some Lightning data modules need setup() called manually if not using Trainer.fit
            if hasattr(data_module, 'setup'):
                data_module.setup(stage='test')
            
            test_loader = data_module.test_dataloader()
        except Exception as e:
            print(f"Error setting up data for fold {fold_idx}: {e}")
            continue
        
        # 3. Setup Model
        # We load from checkpoint, which handles architecture init usually
        try:
            if ckpt_path.endswith(('.pth', '.pt')):
                # Manual state dict load
                model_module = ModelInterface(**config)
                state = torch.load(ckpt_path, map_location=device, weights_only=False)
                if 'state_dict' in state:
                    model_module.load_state_dict(state['state_dict'], strict=False)
                else:
                    model_module.load_state_dict(state, strict=False)
            else:
                # Lightning .ckpt load
                model_module = ModelInterface.load_from_checkpoint(
                    ckpt_path, map_location=device, strict=False, **config
                )
            
            model_module.to(device)
            model_module.eval()
        except Exception as e:
            print(f"Error loading model for fold {fold_idx}: {e}")
            continue

        # 4. Inference Loop
        fold_preds = []
        fold_targets = []
        fold_probs = []
        fold_densities = []

        print(f"Samples in test loader: {len(test_loader.dataset)}")
        
        with torch.no_grad():
            for batch in test_loader:
                # Unpack batch - robust unpacking
                # Assuming standard: inputs (maybe list), label, filenames
                if len(batch) == 3:
                    inputs, label, filenames = batch
                    if isinstance(inputs, list):
                        test_input = [t.to(device) for t in inputs]
                    else:
                        test_input = [inputs.to(device)]
                else:
                    # Fallback for complex tuples
                    *test_input, label, file_names = batch
                    test_input = [t.to(device) for t in test_input]
                
                # Forward pass
                test_logits_target = model_module(*test_input)   # [N, C]
                preds = test_logits_target.argmax(dim=1)         # [N]
                probs = torch.softmax(test_logits_target, dim=1) # [N, C]

                # Extract breast density for visualization
                # Assuming density is at input[1][:, 0] based on your previous code
                # We need to be careful if input structure varies
                try:
                    # Try accessing the second input element (metadata vector)
                    breast_density = test_input[1][:, 0].cpu().numpy()
                except:
                    # Fallback if structure is different
                    breast_density = np.zeros(preds.shape[0])

                fold_preds.append(preds.cpu().numpy())
                fold_targets.append(label.cpu().numpy())
                fold_probs.append(probs.cpu().numpy())
                fold_densities.append(breast_density)

        # Concatenate fold results
        if len(fold_preds) > 0:
            global_preds.append(np.concatenate(fold_preds))
            global_targets.append(np.concatenate(fold_targets))
            global_probs.append(np.concatenate(fold_probs, axis=0))
            global_breast_density.append(np.concatenate(fold_densities))

    # — Aggregate All Results —
    if not global_preds:
        print("No predictions generated. Exiting.")
        return

    all_preds   = np.concatenate(global_preds)
    all_targets = np.concatenate(global_targets)
    all_probs   = np.concatenate(global_probs, axis=0)
    all_breast_density = np.concatenate(global_breast_density)

    print("\n===========================================")
    print("--- Overall Evaluation (All Folds) ---")
    print("===========================================")
    print(f"Total samples evaluated: {len(all_targets)}")

    # Convert one-hot back to integer labels if needed
    if all_targets.ndim > 1:
        all_targets = np.argmax(all_targets, axis=1)

    # — Accuracy and F1 score —
    accuracy = (all_preds == all_targets).mean()
    print(f"Overall test accuracy: {accuracy:.4f}")
    f1 = f1_score(all_targets, all_preds, average='macro')
    print(f"Overall test F1 score: {f1:.4f}")

    # — Confusion matrix —
    cm = confusion_matrix(all_targets, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    num_bins = cm.shape[0]
    avg_density = np.full_like(cm, np.nan, dtype=float)

    for i in range(num_bins):
        for j in range(num_bins):
            mask = (all_targets == i) & (all_preds == j)
            if np.any(mask):
                avg_density[i, j] = all_breast_density[mask].mean()

    # Plotting CM
    plt.figure(figsize=(8,6))
    plt.imshow(cm_norm, cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.title(f"Overall Confusion Matrix ({len(run_configs)} Folds)")
    plt.colorbar()
    ticks = np.arange(num_bins)
    plt.xticks(ticks, ticks); plt.yticks(ticks, ticks)
    plt.xlabel("Predicted"); plt.ylabel("True")
    thresh = cm_norm.max() / 2

    for i in range(num_bins):
        for j in range(num_bins):
            count = cm[i, j]
            pct   = cm_norm[i, j] * 100
            plt.text(
                j, i,
                f"{count}\n({pct:.1f}%)",
                ha='center', va='center',
                color='white' if cm_norm[i, j] > thresh else 'black'
            )

    plt.tight_layout()
    plt.savefig("overall_confusion_matrix_3class.png", dpi=300)
    print("Saved overall_confusion_matrix_3class.png")

    # — ROC curves —
    num_classes = all_probs.shape[1]

    if num_classes == 2:
        # scores for positive class (class 1)
        y_true = all_targets
        y_score = all_probs[:, 1]

        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f"Overall ROC (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], 'k--', label="Chance (AUC = 0.5)")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Overall Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("overall_roc_curve_3class.png", dpi=300)
        plt.close()
        print(f"Saved overall_roc_curve_3class.png (AUC: {roc_auc:.3f})")


if __name__ == '__main__':
    main()