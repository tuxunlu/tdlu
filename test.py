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
from models import ModelInterface, ModelInterfaceAux, ModelInterfaceAuxSaliency
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

def grad_cam(model, image, target_class, target_layer):
    """Compute Grad‑CAM heatmap for a single image."""
    activations, gradients = [], []

    def forward_hook(module, inp, outp):
        activations.append(outp.detach())
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    logits, _ = model(image)
    score = logits[0, target_class]
    model.zero_grad()
    score.backward(retain_graph=True)

    fh.remove(); bh.remove()

    act  = activations[-1]  # [1,C,H,W]
    grad = gradients[-1]    # [1,C,H,W]
    weights = grad.mean(dim=(2,3), keepdim=True)  # [1,C,1,1]
    cam = (weights * act).sum(dim=1).squeeze()     # [H,W]
    cam = torch.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.cpu().numpy()

def main():
    parser = ArgumentParser()
    parser.add_argument('--config_path', default='config/config.yaml',
                        help='YAML config file')
    parser.add_argument('--test_checkpoint', required=True,
                        help='Path to .pth or .ckpt checkpoint')
    args = parser.parse_args()

    # — Load config —
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config not found: {args.config_path}")
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    config = {k.lower(): v for k,v in config.items()}

    # — Data & Model setup —
    data_module  = DataInterface(**config)
    model_module = ModelInterface(**config)

    # — Device selection —
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # — Load checkpoint onto device —
    ckpt = args.test_checkpoint
    if ckpt.endswith(('.pth', '.pt')):
        state = torch.load(ckpt, map_location=device)
        model_module.load_state_dict(state)
    else:
        # Lightning .ckpt load
        model_module = ModelInterface.load_from_checkpoint(
            ckpt, map_location=device, **config
        )
    # **Ensure model is on GPU** (or chosen device)
    model_module.to(device)
    model_module.eval()

    # — Lightning’s built‑in test —
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=False,
        enable_progress_bar=False
    )
    results = trainer.test(model=model_module,
                           datamodule=data_module,
                           verbose=True)
    print("Test results:", results)

    # 🔄 **Re‑move model back to GPU** after test (Lightning may have moved it to CPU) :contentReference[oaicite:12]{index=12}
    model_module.to(device)
    model_module.eval()

    # — Manual metrics & visualizations —
    test_loader = data_module.test_dataloader()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    print(len(test_loader.dataset), "test samples")
    print(len(train_loader.dataset), "train samples")
    print(len(val_loader.dataset), "validation samples")
    all_preds, all_targets, all_probs, all_breast_density= [], [], [], []
    
    loader = test_loader

    with torch.no_grad():
        for *test_input, label, file_names in loader:
            test_input = [t.to(device) for t in test_input]
            
            test_logits_target = model_module(*test_input)   # [N, C]
            preds = test_logits_target.argmax(dim=1)         # [N]

            # --- NEW: probabilities ---
            probs = torch.softmax(test_logits_target, dim=1) # [N, C]
            all_probs.append(probs.cpu().numpy())

            # breast density just for your confusion-matrix overlay
            breast_density = test_input[1][:, 0].cpu().numpy()
            all_breast_density.append(breast_density)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(label.cpu().numpy())


    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_probs   = np.concatenate(all_probs, axis=0)   # [N, C]
    all_breast_density = np.concatenate(all_breast_density, axis=0)


    # Convert one-hot back to integer labels if needed
    if all_targets.ndim > 1:
        all_targets = np.argmax(all_targets, axis=1)

    # — Accuracy and F1 score —
    accuracy = (all_preds == all_targets).mean()
    print(f"Overall test accuracy: {accuracy:.4f}")
    from sklearn.metrics import f1_score
    f1 = f1_score(all_targets, all_preds, average='macro')
    print(f"Overall test F1 score: {f1:.4f}")

    # — Confusion matrix —
    cm = confusion_matrix(all_targets, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    # --- new: compute average density per cell ---
    num_bins = cm.shape[0]
    avg_density = np.full_like(cm, np.nan, dtype=float)
    for i in range(num_bins):
        for j in range(num_bins):
            mask = (all_targets == i) & (all_preds == j)
            if np.any(mask):
                avg_density[i, j] = all_breast_density[mask].mean()

    std_density = np.full_like(cm, np.nan, dtype=float)
    for i in range(num_bins):
        for j in range(num_bins):
            mask = (all_targets == i) & (all_preds == j)
            if np.any(mask):
                std_density[i, j] = all_breast_density[mask].std()

    # plotting
    plt.figure(figsize=(8,6))
    plt.imshow(cm_norm, cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.title("Normalized Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(num_bins)
    plt.xticks(ticks, ticks); plt.yticks(ticks, ticks)
    plt.xlabel("Predicted"); plt.ylabel("True")
    thresh = cm_norm.max() / 2

    for i in range(num_bins):
        for j in range(num_bins):
            count = cm[i, j]
            pct   = cm_norm[i, j] * 100
            mean_den = avg_density[i, j]
            if np.isnan(mean_den):
                den_str = "–"
                std_str = "–"
            else:
                den_str = f"mean: {mean_den:.2f}"
                std_str = f"std: {std_density[i, j]:.2f}" if not np.isnan(std_density[i, j]) else "–"
            plt.text(
                j, i,
                f"{count}\n({pct:.1f}%)",
                ha='center', va='center',
                color='white' if cm_norm[i, j] > thresh else 'black'
            )

    plt.tight_layout()
    plt.savefig("confusion_matrix_with_density.png", dpi=300)
    plt.close()

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
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], 'k--', label="Chance (AUC = 0.5)")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("roc_curve_binary.png", dpi=300)
        plt.close()

    # # — Saliency & Grad-CAM for multi-view inputs —
    # dataset = loader.dataset
    # num_vis = config.get('num_visualizations', 5)
    # indices = [i for i in range(num_vis)]
    # for idx in indices:
    #     # 1) Load one sample: images (V, C, H, W), target_unused
    #     images, masks_tensor, meta, _, filename = dataset[idx]                # torch.Tensor (4, C, H, W)
    #     # images, meta, label, filename = dataset[idx]
    #     V, C, H, W = images.shape

    #     # 2) Add batch dim and send to device
    #     imgs = images.unsqueeze(0).to(device)   # (1, 4, C, H, W)
    #     masks_tensor = masks_tensor.unsqueeze(0).to(device)
    #     meta = meta.unsqueeze(0).to(device)
    #     imgs.requires_grad_()

    #     # 3) Forward to get logits
    #     out, _ = model_module(imgs, masks_tensor, meta)               # (1, 2)
    #     # out, _ = model_module(imgs, meta) 
    #     pred = out.argmax(dim=1).item()         # scalar 0 or 1

    #     # — Saliency (per view) —
    #     model_module.zero_grad()

        
    #     out[0, pred].backward(retain_graph=True)
    #     # grads: (1,4,C,H,W)
    #     grads = imgs.grad.abs().cpu().squeeze(0)    # (4,C,H,W)
    #     sals  = []
    #     for v in range(V):
    #         # max over channel dim → (H,W)
    #         sal = grads[v].max(dim=0)[0]
    #         sal = (sal - sal.min())/(sal.max()-sal.min()+1e-8)
    #         sals.append(sal.numpy())

    #     # — Plot Saliency grid —
    #     fig, axes = plt.subplots(2,4,figsize=(16,12))
    #     for i in range(4):
    #         axes[0, i].imshow(sals[i], cmap='hot')
    #         # sal_im = ax.imshow(sals[v], cmap='hot')
    #         axes[0, i].set_title(f"Saliency View {i}")
    #         axes[0, i].axis('off')
    #         axes[1, i].imshow(imgs[0, i, 0].detach().cpu().numpy(), cmap='gray')
    #         # axes[2, i].imshow(masks_tensor[0, i, 0].cpu().numpy(), cmap='gray')
    #         # fig.colorbar(sal_im, ax=axes.ravel().tolist(),
    #         #             orientation='vertical',
    #         #             fraction=0.02, pad=0.01,
    #         #             label='Saliency intensity')
        
    #     plt.subplots_adjust(hspace=0)
    #     plt.subplots_adjust(wspace=0)
    #     plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    #     plt.savefig(f"saliency_{filename}.png", dpi=300)
    #     plt.close()


if __name__ == '__main__':
    main()
