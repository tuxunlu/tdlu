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
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

def grad_cam(model, image, density, age, bmi, target_class, target_layer):
    """Compute Grad‑CAM heatmap for a single image."""
    activations, gradients = [], []

    def forward_hook(module, inp, outp):
        activations.append(outp.detach())
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    logits, _ = model(image, density, age, bmi)
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

    # — Reproducibility —
    # seed = config.get('seed', 42)
    # pl.seed_everything(41)
    # random.seed(41)

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
    model_module.to(device)  # maps all weights & buffers to GPU :contentReference[oaicite:11]{index=11}
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
                           verbose=False)
    print("Test results:", results)

    # 🔄 **Re‑move model back to GPU** after test (Lightning may have moved it to CPU) :contentReference[oaicite:12]{index=12}
    model_module.to(device)
    model_module.eval()

    # — Manual metrics & visualizations —
    test_loader = data_module.test_dataloader()
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for images, density, age, bmi, targets in test_loader:
            # **Send inputs to same device** :contentReference[oaicite:13]{index=13}
            images  = images.to(device)
            density = density.float().to(device)
            age     = age.float().to(device)
            bmi     = bmi.float().to(device)
            targets = targets.to(device)

            logits, _ = model_module(images, density, age, bmi)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_probs   = np.concatenate(all_probs, axis=0)

    # Convert one-hot back to integer labels if needed
    if all_targets.ndim > 1:
        all_targets = np.argmax(all_targets, axis=1)

    # — Accuracy —
    accuracy = (all_preds == all_targets).mean()
    print(f"Overall test accuracy: {accuracy:.4f}")

    # — Confusion matrix —
    cm = confusion_matrix(all_targets, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(8,6))
    plt.imshow(cm_norm, cmap=plt.cm.Blues)
    plt.title("Normalized Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(config['num_bins'])
    plt.xticks(ticks, ticks); plt.yticks(ticks, ticks)
    plt.xlabel("Predicted"); plt.ylabel("True")
    thresh = cm_norm.max()/2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i,
                f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)",
                ha='center', va='center',
                color='white' if cm_norm[i,j]>thresh else 'black'
            )
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()

    # — ROC curves —
    targets_bin = label_binarize(all_targets, classes=list(range(config['num_bins'])))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(config['num_bins']):
        fpr[i], tpr[i], _ = roc_curve(targets_bin[:,i], all_probs[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr['micro'], tpr['micro'], _ = roc_curve(targets_bin.ravel(), all_probs.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    plt.figure(figsize=(8,6))
    plt.plot(fpr['micro'], tpr['micro'], ':', lw=4,
             label=f"micro (AUC={roc_auc['micro']:.2f})")
    for i in range(config['num_bins']):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f"Class {i} (AUC={roc_auc[i]:.2f})")
    plt.plot([0,1],[0,1],'k--', lw=2)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves"); plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=300)
    plt.close()

    # — Saliency & Grad‑CAM —
    dataset = test_loader.dataset
    num_vis = config.get('num_visualizations', 5)
    indices = random.sample(range(len(dataset)), num_vis)
    for idx in indices:
        print(len(dataset[idx]))
        image, dens, age, bmi, targets = dataset[idx]
        subject_id = getattr(dataset, 'image_files', [str(idx)])[idx].split('.')[0]

        img = image.unsqueeze(0).to(device); img.requires_grad_()
        dens_t = torch.tensor([dens]).unsqueeze(0).float().to(device)
        age_t = torch.tensor([age]).unsqueeze(0).float().to(device)
        bmi_t = torch.tensor([bmi]).unsqueeze(0).float().to(device)


        out, _ = model_module(img, dens_t, age_t, bmi_t)
        pred = out.argmax(dim=1).item()

        # Saliency
        score = out[0, pred]
        model_module.zero_grad(); score.backward(retain_graph=True)
        grad = img.grad.abs()
        sal, _ = grad.max(dim=1)
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        plt.figure(figsize=(6,5))
        plt.imshow(sal.cpu().squeeze(), cmap='hot')
        plt.title(f"Saliency Map {subject_id}")
        plt.colorbar(); plt.savefig(f"saliency_{subject_id}.png", dpi=300); plt.close()

        # Grad‑CAM
        heatmap = grad_cam(model_module, img, dens_t, age_t, bmi_t, pred, model_module.model.backbone._modules["7"])
        img_np = image.permute(1,2,0).cpu().numpy()
        plt.figure(figsize=(6,5))
        plt.imshow(img_np, cmap='gray')
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.title(f"Grad‑CAM {subject_id}")
        plt.colorbar(); plt.savefig(f"gradcam_{subject_id}.png", dpi=300); plt.close()

if __name__ == '__main__':
    main()
