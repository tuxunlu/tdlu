# Plan to Fix TDLU Model Overfitting

**Problem:** Train acc/F1 improve (0.35→0.6+), val acc/F1 oscillate (0.35–0.55) with no improvement. Model memorizes training data.

**Root cause:** High model capacity + small dataset (~320 train) + weak/noisy TDLU signal.

---

## Phase 1: Quick Wins (Config-Only)

| # | Change | File | Rationale |
|---|--------|------|-----------|
| 1.1 | **`freeze_backbone: True`** | config | Already done. Train only head + fusion first. |
| 1.2 | **`fusion_method: "concat"`** | config | Cross-attention with 1 view + 1 meta token is degenerate. Concat is simpler, fewer params, less overfitting. |
| 1.3 | **`dropout_rate: 0.4`** | config | Already 0.35; bump to 0.4 for more regularization. |
| 1.4 | **`weight_decay: 5e-4`** | config | Stronger L2 regularization (currently 1e-4). |
| 1.5 | **`lr: 1e-4`** | config | Lower LR further (currently 2e-4) for gentler updates. |
| 1.6 | **`batch_size: 16`** | config | Larger batches → smoother gradients. Only if GPU memory allows. |

---

## Phase 2: Architecture Simplification

| # | Change | File | Rationale |
|---|--------|------|-----------|
| 2.1 | **Smaller classification head** | `MGModule_one_view_meta_avg_cross_attention.py` | Change `ClassifierHead` from `512→512→num_classes` to `512→256→num_classes`. Fewer params. |
| 2.2 | **Smaller meta encoder** | model | Change `meta_proj` from `256→512` to `128→512`. |
| 2.3 | **Remove view_proj** | model | Optional. View features go directly to fusion. Reduces params. |
| 2.4 | **Reduce transformer_heads** | config | Add `transformer_heads: 4` (if keeping cross_attn). Fewer params. |

---

## Phase 3: Data & Augmentation

| # | Change | File | Rationale |
|---|--------|------|-----------|
| 3.1 | **Enable intensity augmentation** | `TDLUDataset_torchio_four_view_mask_meta_avg_10_folds_STAMP.py` | Uncomment `RandomGamma`, `RandomNoise`, `RandomBlur` in `JointTioAug.intensity`. More data diversity. |
| 3.2 | **Stronger spatial augmentation** | dataset | Increase `RandomAffine` p from 0.3 to 0.5; wider scales (0.8, 1.2). |
| 3.3 | **Mixup or CutMix** | new | Add mixup (alpha=0.2) in training. Helps generalization. Requires ModelInterface change. |
| 3.4 | **Label smoothing** | config + loss | Add `label_smoothing: 0.1` to CrossEntropy/Focal. Reduces overconfident predictions. |

---

## Phase 4: Training Strategy

| # | Change | File | Rationale |
|---|--------|------|-----------|
| 4.1 | **Two-stage training** | new script or config | Stage 1: freeze backbone, train 100 epochs. Stage 2: unfreeze backbone, lr=1e-5, train 50 epochs. |
| 4.2 | **Progressive unfreezing** | model | Unfreeze layer4 first, then layer3, etc. Gradual fine-tuning. |
| 4.3 | **Reduce max_epochs** | config | 500 might be too long. Try 200–300 with early stopping (or manual checkpoint selection). |

---

## Phase 5: Diagnostics & Baselines

| # | Change | Purpose |
|---|--------|---------|
| 5.1 | **Metadata-only baseline** | Train logistic regression on meta only. If it matches CNN, image is not contributing. |
| 5.2 | **Image-only baseline** | Remove metadata from model. See if image alone can learn. |
| 5.3 | **Per-class val metrics** | Log val_acc, val_f1 per class. Check if one class dominates or is ignored. |
| 5.4 | **Confusion matrix** | Log at end of epoch. Diagnose systematic errors. |

---

## Recommended Implementation Order

### Step 1 (Immediate)
1. Set `fusion_method: "concat"` in config.
2. Set `dropout_rate: 0.4`, `weight_decay: 5e-4`, `lr: 1e-4`.
3. Retrain. If val improves → continue. If not → Step 2.

### Step 2 (Architecture)
1. Shrink classification head: `512→256→num_classes`.
2. Shrink meta encoder: `128→512`.
3. Retrain.

### Step 3 (Data)
1. Enable intensity augmentation (RandomGamma, RandomNoise).
2. Add label smoothing (if supported by FocalLoss).
3. Retrain.

### Step 4 (Two-stage)
1. Train 100 epochs with frozen backbone.
2. Load best checkpoint, unfreeze backbone, set lr=1e-5, train 50 more epochs.
3. Compare val_f1 to single-stage.

### Step 5 (Diagnostics)
1. Run metadata-only logistic regression.
2. Compare to full model. If similar → image pathway may not be learning.

---

## Config Summary (Step 1)

```yaml
# Recommended config changes for Step 1
fusion_method: "concat"
dropout_rate: 0.4
weight_decay: 5.0e-4
lr: 1.0e-04
freeze_backbone: True
```

---

## Success Criteria

- **Val F1** shows upward trend (not just oscillation).
- **Val acc** ≥ 0.45 and improving over epochs.
- **Gap** between train and val metrics < 0.15 (e.g., train_acc 0.55, val_acc 0.45).

---

## Files to Modify

| Phase | Files |
|-------|-------|
| 1 | `config/config_torchio_3bins_tdlu_stamp_one_view_10folds_zero_extreme_meta_avg_cross_attention.yaml` |
| 2 | `models/MGModule_one_view_meta_avg_cross_attention.py` |
| 3 | `data/TDLUDataset_torchio_four_view_mask_meta_avg_10_folds_STAMP.py`, `models/loss/FocalLoss.py` |
| 4 | `main.py` or new training script |
| 5 | New analysis script, `models/ModelInterface.py` (for per-class logging) |
