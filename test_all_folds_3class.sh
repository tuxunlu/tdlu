#!/bin/bash
# Test script for image-only model experiments (avg_image_only_resnet_transformer_optimized)
# Note: Fold 9 is missing from the experiments
python test_all_folds.py \
--fold_0  /beacon-scratch/tuxunlu/git/tdlu/runs/20260109-04-42-28-config_torchio_3bins_tdlu_extreme_zero_stacked_10fold0_avg_image_only_resnet_transformer_optimized/version_0/hparams.yaml \
--checkpoint_0  /beacon-scratch/tuxunlu/git/tdlu/runs/20260109-04-42-28-config_torchio_3bins_tdlu_extreme_zero_stacked_10fold0_avg_image_only_resnet_transformer_optimized/version_0/checkpoints/best-epoch=016-val_f1=0.47188-val_acc=0.49500.ckpt \
--fold_1  /beacon-scratch/tuxunlu/git/tdlu/runs/20260109-07-20-00-config_torchio_3bins_tdlu_extreme_zero_stacked_10fold1_avg_image_only_resnet_transformer_optimized/version_0/hparams.yaml \
--checkpoint_1  /beacon-scratch/tuxunlu/git/tdlu/runs/20260109-07-20-00-config_torchio_3bins_tdlu_extreme_zero_stacked_10fold1_avg_image_only_resnet_transformer_optimized/version_0/checkpoints/best-epoch=077-val_f1=0.46061-val_acc=0.48889.ckpt \
--fold_2  /beacon-scratch/tuxunlu/git/tdlu/runs/20260109-09-57-27-config_torchio_3bins_tdlu_extreme_zero_stacked_10fold2_avg_image_only_resnet_transformer_optimized/version_0/hparams.yaml \
--checkpoint_2  /beacon-scratch/tuxunlu/git/tdlu/runs/20260109-09-57-27-config_torchio_3bins_tdlu_extreme_zero_stacked_10fold2_avg_image_only_resnet_transformer_optimized/version_0/checkpoints/best-epoch=011-val_f1=0.59709-val_acc=0.60825.ckpt \
--fold_3  /beacon-scratch/tuxunlu/git/tdlu/runs/20260109-04-42-28-config_torchio_3bins_tdlu_extreme_zero_stacked_10fold3_avg_image_only_resnet_transformer_optimized/version_0/hparams.yaml \
--checkpoint_3  /beacon-scratch/tuxunlu/git/tdlu/runs/20260109-04-42-28-config_torchio_3bins_tdlu_extreme_zero_stacked_10fold3_avg_image_only_resnet_transformer_optimized/version_0/checkpoints/best-epoch=024-val_f1=0.40769-val_acc=0.41563.ckpt \
--fold_4  /beacon-scratch/tuxunlu/git/tdlu/runs/20260109-07-22-05-config_torchio_3bins_tdlu_extreme_zero_stacked_10fold4_avg_image_only_resnet_transformer_optimized/version_0/hparams.yaml \
--checkpoint_4  /beacon-scratch/tuxunlu/git/tdlu/runs/20260109-07-22-05-config_torchio_3bins_tdlu_extreme_zero_stacked_10fold4_avg_image_only_resnet_transformer_optimized/version_0/checkpoints/best-epoch=060-val_f1=0.47833-val_acc=0.48413.ckpt \
--fold_5  /beacon-scratch/tuxunlu/git/tdlu/runs/20260109-10-00-13-config_torchio_3bins_tdlu_extreme_zero_stacked_10fold5_avg_image_only_resnet_transformer_optimized/version_0/hparams.yaml \
--checkpoint_5  /beacon-scratch/tuxunlu/git/tdlu/runs/20260109-10-00-13-config_torchio_3bins_tdlu_extreme_zero_stacked_10fold5_avg_image_only_resnet_transformer_optimized/version_0/checkpoints/best-epoch=010-val_f1=0.42717-val_acc=0.45833.ckpt \
--fold_6  /beacon-scratch/tuxunlu/git/tdlu/runs/20260109-04-42-56-config_torchio_3bins_tdlu_extreme_zero_stacked_10fold6_avg_image_only_resnet_transformer_optimized/version_0/hparams.yaml \
--checkpoint_6  /beacon-scratch/tuxunlu/git/tdlu/runs/20260109-04-42-56-config_torchio_3bins_tdlu_extreme_zero_stacked_10fold6_avg_image_only_resnet_transformer_optimized/version_0/checkpoints/best-epoch=027-val_f1=0.54165-val_acc=0.58532.ckpt \
--fold_7  /beacon-scratch/tuxunlu/git/tdlu/runs/20260109-07-26-25-config_torchio_3bins_tdlu_extreme_zero_stacked_10fold7_avg_image_only_resnet_transformer_optimized/version_0/hparams.yaml \
--checkpoint_7  /beacon-scratch/tuxunlu/git/tdlu/runs/20260109-07-26-25-config_torchio_3bins_tdlu_extreme_zero_stacked_10fold7_avg_image_only_resnet_transformer_optimized/version_0/checkpoints/best-epoch=007-val_f1=0.46114-val_acc=0.58816.ckpt \
--fold_8  /beacon-scratch/tuxunlu/git/tdlu/runs/20260109-10-09-49-config_torchio_3bins_tdlu_extreme_zero_stacked_10fold8_avg_image_only_resnet_transformer_optimized/version_0/hparams.yaml \
--checkpoint_8  /beacon-scratch/tuxunlu/git/tdlu/runs/20260109-10-09-49-config_torchio_3bins_tdlu_extreme_zero_stacked_10fold8_avg_image_only_resnet_transformer_optimized/version_0/checkpoints/best-epoch=028-val_f1=0.46935-val_acc=0.46006.ckpt