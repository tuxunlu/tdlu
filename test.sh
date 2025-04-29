#!/bin/bash

python test.py \
--config_path /fs/nexus-scratch/tuxunlu/git/tdlu/config/config.yaml \
--test_checkpoint /fs/nexus-scratch/tuxunlu/git/tdlu/runs/20250424-11-34-25-smoothed_metafuse_mirai_freeze/version_0/checkpoints/best-epoch=474-val_acc=0.73529.ckpt
