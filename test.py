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
    all_preds, all_targets, all_probs = [], [], []
    
    loader = test_loader

    with torch.no_grad():
        for images, targets, _ in loader:
            # **Send inputs to same device** :contentReference[oaicite:13]{index=13}
            images  = images.to(device)
            targets = targets.to(device)

            logits, _ = model_module(images)
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
    plt.imshow(cm_norm, cmap=plt.cm.Blues, vmin=0, vmax=1)
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

    # # — ROC curves —
    # targets_bin = label_binarize(all_targets, classes=list(range(config['num_bins'])))
    # fpr, tpr, roc_auc = {}, {}, {}
    # for i in range(config['num_bins']):
    #     fpr[i], tpr[i], _ = roc_curve(targets_bin[:,i], all_probs[:,i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    # fpr['micro'], tpr['micro'], _ = roc_curve(targets_bin.ravel(), all_probs.ravel())
    # roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # plt.figure(figsize=(8,6))
    # plt.plot(fpr['micro'], tpr['micro'], ':', lw=4,
    #          label=f"micro (AUC={roc_auc['micro']:.2f})")
    # for i in range(config['num_bins']):
    #     plt.plot(fpr[i], tpr[i], lw=2,
    #              label=f"Class {i} (AUC={roc_auc[i]:.2f})")
    # plt.plot([0,1],[0,1],'k--', lw=2)
    # plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    # plt.title("ROC Curves"); plt.legend(loc="lower right")
    # plt.tight_layout()
    # plt.savefig("roc_curves.png", dpi=300)
    # plt.close()

    # — Saliency & Grad-CAM for multi-view inputs —
    dataset = loader.dataset
    num_vis = config.get('num_visualizations', 5)
    indices = random.sample(range(len(dataset)), num_vis)
    for idx in indices:
        # 1) Load one sample: images (V, C, H, W), target_unused
        images, _, filename = dataset[idx]                # torch.Tensor (4, C, H, W)
        V, C, H, W = images.shape

        # 2) Add batch dim and send to device
        imgs = images.unsqueeze(0).to(device)   # (1, 4, C, H, W)
        imgs.requires_grad_()

        # 3) Forward to get logits
        out, _ = model_module(imgs)               # (1, 2)
        pred = out.argmax(dim=1).item()         # scalar 0 or 1

        # # — Grad-CAM (single heatmap) —
        # cam = grad_cam_multi(
        #     model_module,
        #     imgs,
        #     pred,
        #     model_module.model.backbone._modules["7"]
        # )
        # # resize to input H×W
        # cam_resized = cv2.resize(cam, (W, H))

        # — Saliency (per view) —
        model_module.zero_grad()
        imgs.grad = None
        out[0, pred].backward(retain_graph=True)
        # grads: (1,4,C,H,W)
        grads = imgs.grad.abs().cpu().squeeze(0)    # (4,C,H,W)
        sals  = []
        for v in range(V):
            # max over channel dim → (H,W)
            sal = grads[v].max(dim=0)[0]
            sal = (sal - sal.min())/(sal.max()-sal.min()+1e-8)
            sals.append(sal.numpy())

        # # — Plot Grad-CAM grid —
        # fig, axes = plt.subplots(2,2,figsize=(8,8))
        # for v, ax in enumerate(axes.flatten()):
        #     # extract view v image
        #     img_np = imgs.cpu().squeeze(0)[v].permute(1,2,0).numpy()
        #     if C==1:
        #         ax.imshow(img_np.squeeze(), cmap='gray')
        #     else:
        #         ax.imshow(img_np)
        #     ax.imshow(cam_resized, cmap='jet', alpha=0.5)
        #     ax.set_title(f"Grad-CAM View {v}")
        #     ax.axis('off')
        # plt.tight_layout()
        # plt.savefig(f"gradcam_{idx}.png", dpi=300)
        # plt.close()

        # — Plot Saliency grid —
        fig, axes = plt.subplots(2,2,figsize=(8,8))
        for v, ax in enumerate(axes.flatten()):
            sal_im = ax.imshow(sals[v], cmap='hot')
            ax.set_title(f"Saliency View {v}")
            ax.axis('off')
        fig.colorbar(sal_im, ax=axes.ravel().tolist(),
                 orientation='vertical',
                 fraction=0.02, pad=0.01,
                 label='Saliency intensity')
        plt.tight_layout()
        # plt.savefig(f"saliency_{filename}.png", dpi=300)
        plt.close()


if __name__ == '__main__':
    main()
