# TDLU STAMP Model Analysis: Why Performance is Near Random

## Executive Summary

The model achieves **val_acc ≈ 41.7%** and **val_f1 ≈ 0.26** on 3-class TDLU density prediction (random baseline ≈ 33%). Several architectural, training, and data issues explain this poor performance. Below are the root causes and concrete fixes.

---

## 1. CRITICAL: Dense Mask is Loaded but Never Used

**Location:** `MGModule_one_view_meta_avg_cross_attention.py` line 124

The model receives `mask` (dense tissue mask) in `forward(views, mask, meta)` but **never uses it**. TDLU structures are located in dense tissue regions; the mask indicates where to look.

**Evidence:** Other models in the codebase use the mask:
- `MGModule_four_view_mask_meta_avg_cross_attention.py`: concatenates mask as extra channels to the image
- `MGModule_four_view_meta_avg_cross_attention_GAP.py`: uses mask for ROI pooling (dense vs background)

**Fix:** Incorporate the dense mask, e.g.:
- **Option A:** Concatenate dense mask as 4th channel (like `MGModule_four_view_mask_meta_avg_cross_attention`)
- **Option B:** Use mask for attention-weighted pooling over the feature map (like GAP variant)
- **Option C:** Apply mask to zero-out non-dense regions before backbone

---

## 2. Focal Loss Gamma = 0 Disables Focal Effect

**Location:** `config_torchio_3bins_tdlu_stamp_one_view_10folds_zero_extreme_meta_avg_cross_attention.yaml` line 17

```yaml
focal_loss_gamma: 0
```

With γ=0, `(1 - p_t)^0 = 1`, so Focal Loss reduces to **alpha-weighted CrossEntropy**. The focal down-weighting of easy examples is completely disabled. For class imbalance and hard examples, γ=1–2 is standard.

**Fix:** Set `focal_loss_gamma: 1` or `2` (other configs in the repo use 1–2).

---

## 3. Cross-Attention with Single View/Meta Token is Degenerate

**Location:** `MGModule_one_view_meta_avg_cross_attention.py` lines 143–154

With **1 view token** and **1 meta token**, cross-attention has:
- Query: 1 token
- Key/Value: 1 token each

Attention over a single key/value is trivial (no real selection). The model effectively learns a fixed combination, not attention over multiple regions.

**Fix:**
- Use **spatial tokens** from the backbone (e.g., from `layer4` feature map) instead of a single global vector, so attention can focus on relevant regions.
- Or simplify to **concat fusion** and drop the cross-attention overhead.

---

## 4. Image Normalization May Be Suboptimal for Mammograms

**Location:** `TDLUDataset_torchio_four_view_mask_meta_avg_10_folds_STAMP.py` line 298

```python
img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

ImageNet normalization is used for grayscale mammograms (repeated to 3 channels). Mammograms have very different intensity distributions. This can hurt or help depending on pretraining.

**Fix:** Consider:
- Mammogram-specific normalization (e.g., from dataset statistics)
- Or keep ImageNet stats if using ImageNet-pretrained backbone, but verify that grayscale→RGB replication is appropriate

---

## 5. Breast Density Not in Metadata (STAMP)

**Location:** Config `meta_cols: ["mamm_age", "PATIENTI_BMI", "RACE_COMPOSITE"]`

The STAMP CSV has `BreastDensity`, `BreastDensityPercent_Image`, `DenseArea_sqcm`, etc. TDLU density correlates with breast density. Other configs use `BreastDensity_avg` or similar.

**Fix:** Add `BreastDensity` or `BreastDensityPercent_Image` to `meta_cols` if the goal is maximum predictive performance. If the goal is image-only prediction, keep metadata minimal and ensure the model is not over-relying on metadata. Note: Some CSV rows may have missing BreastDensity; those subjects will be excluded. If the dataset shrinks significantly, revert to 3 meta_cols.

---

## 6. Trainer Strategy with Single GPU

**Location:** `main.py` line 134

```python
trainer = Trainer(accelerator="gpu", devices=1, strategy="ddp", ...)
```

Using `strategy="ddp"` with `devices=1` can cause unnecessary overhead or odd behavior. For single-GPU training, `strategy="auto"` or omitting it is typical.

**Fix:** Use `strategy="auto"` or remove the strategy when `devices=1`.

---

## 7. Potential Metadata Leakage

If TDLU correlates strongly with age, BMI, or race, the model may learn to predict mainly from metadata and ignore the image. That would give modest gains over random but would not generalize to deployment where image-based prediction is desired.

**Fix:** Run a **metadata-only baseline** (e.g., logistic regression on meta) to compare. If meta-only performs similarly to the full model, the image pathway may not be contributing.

---

## 8. Dataset Size and Class Balance

STAMP has ~450–500 unique subjects after filtering. With 10-fold CV, train set ≈ 320–400, val ≈ 40–50. Small val set → noisy metrics. Class distribution (0, 1, 2) may be imbalanced; focal alpha helps but gamma=0 limits its effect.

**Fix:** Consider stratified sampling, more aggressive augmentation, or combining folds for larger train/val sets.

---

## 9. ResNet18 and Input Resolution

ResNet18 on 1024×1024 produces ~32×32 feature maps. For fine structures like TDLUs, this may lose spatial detail. A larger backbone (e.g., ResNet50) or higher-resolution processing could help.

**Fix:** Experiment with ResNet50 or EfficientNet, or use a backbone that preserves more spatial resolution (e.g., feature pyramid).

---

## 10. Possible NaN/Inf in Metadata

**Location:** Dataset lines 271–272

```python
if not np.isfinite(subj_meta).all():
    continue
```

Some CSV rows have empty `RACE_COMPOSITE` or `PATIENTI_BMI`. The dataset skips these, but it's worth confirming no silent NaN propagation.

---

## Recommended Action Plan (Priority Order)

| Priority | Issue | Action |
|----------|-------|--------|
| 1 | Mask not used | Add mask as 4th channel or use for ROI pooling |
| 2 | Focal gamma=0 | Set `focal_loss_gamma: 1` or `2` |
| 3 | Add BreastDensity to meta | Add `BreastDensity` or `BreastDensityPercent_Image` to meta_cols |
| 4 | Trainer strategy | Use `strategy="auto"` for single GPU |
| 5 | Metadata-only baseline | Run logistic regression on meta to check leakage |
| 6 | Cross-attention design | Consider spatial tokens or simpler concat fusion |
| 7 | Backbone/architecture | Try ResNet50 or higher-res processing |

---

## Files to Modify

1. **`MGModule_one_view_meta_avg_cross_attention.py`** – Use mask (channel concat or ROI pooling)
2. **`config_torchio_3bins_tdlu_stamp_one_view_10folds_zero_extreme_meta_avg_cross_attention.yaml`** – focal_loss_gamma, meta_cols, etc.
3. **`main.py`** – Trainer strategy for single GPU
4. **`TDLUDataset_torchio_four_view_mask_meta_avg_10_folds_STAMP.py`** – Optional: add BreastDensity to meta_cols handling

---

## Training Instability: Val/Train Acc & F1 Oscillation

**Symptom:** val_acc and val_f1 (and train_acc, train_f1) oscillate drastically (e.g., 0.35↔0.55) with no clear convergence. "Overfitting at start." Observed on both STAMP (one-view) and KOMEN (four-view).

### Root Causes

1. **Small validation set** (~40–50 samples with 10-fold CV) → high metric variance; a few predictions flipping = 10–20% swing.
2. **No LR warmup** → full LR from epoch 0 causes aggressive early updates.
3. **LR too high** (5e-4) for transfer learning → overshooting.
4. **No gradient clipping** → occasional large updates.

### Fixes Applied

- **`ModelInterface.py`**: Added cosine + warmup when `warmup_epochs > 0` (LinearLR warmup → CosineAnnealingLR).
- **STAMP config**: `lr: 2e-4`, `warmup_epochs: 15`, `warmup_lr: 1e-6`, `gradient_clip_val: 1.0`.

### For KOMEN Four-View

Add the same to your four-view config:

```yaml
lr: 2.0e-04
warmup_epochs: 15
warmup_lr: 1.0e-06
gradient_clip_val: 1.0
```

---

## Validation Not Improving (Overfitting)

**Symptom:** Train acc/F1 improve (0.35→0.6+), val acc/F1 oscillate (0.35–0.55) with no upward trend.

**Cause:** Model memorizes training data; weak generalization.

**Fixes applied:**
- `freeze_backbone: True` – train head + fusion first, reduce capacity
- `weight_decay: 1.0e-4` – stronger L2 regularization
- `dropout_rate: 0.35` – higher dropout in head/fusion
- `early_stopping_patience: 30` – stop when val_f1 plateaus
