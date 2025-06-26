#!/bin/bash
python test.py \
--config_path /fs/nexus-scratch/tuxunlu/git/tdlu/runs/20250626-14-11-43-config_torchio_2bins_tdlu_extreme_four_views_fold_0_focalloss_full_aug_gamma_1/version_0/hparams.yaml \
--test_checkpoint /fs/nexus-scratch/tuxunlu/git/tdlu/runs/20250626-14-11-43-config_torchio_2bins_tdlu_extreme_four_views_fold_0_focalloss_full_aug_gamma_1/version_0/checkpoints/best-epoch=014-val_f1=0.96330-val_acc=0.96875.ckpt \