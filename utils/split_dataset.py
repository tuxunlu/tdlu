import os
import shutil
import random
from collections import defaultdict

# 1. PARAMETERS
src_dir = "/fs/nexus-scratch/tuxunlu/git/tdlu/dataset/WUSTL_png_nomarker_16"  # your source directory
out_base = "/fs/nexus-scratch/tuxunlu/git/tdlu/dataset/WUSTL_dataset"                # where new train/val/test folders will go
ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
seed = 42

# 2. COLLECT & GROUP
groups = defaultdict(list)
for fn in os.listdir(src_dir):
    if not fn.lower().endswith(".png"):
        continue
    subject_id = fn.split("-", 1)[0]
    groups[subject_id].append(os.path.join(src_dir, fn))

# 3. SPLIT SUBJECTS
subject_ids = list(groups.keys())
random.seed(seed)
random.shuffle(subject_ids)

n = len(subject_ids)
n_train = int(ratios["train"] * n)
n_val   = int(ratios["val"]   * n)

train_ids = subject_ids[:n_train]
val_ids   = subject_ids[n_train:n_train+n_val]
test_ids  = subject_ids[n_train+n_val:]

splits = {
    "train": train_ids,
    "val":   val_ids,
    "test":  test_ids
}

# 4. MAKE OUTPUT DIRS
for split in splits:
    os.makedirs(os.path.join(out_base, split), exist_ok=True)

# 5. COPY FILES
for split, subjs in splits.items():
    for sid in subjs:
        for src_fp in groups[sid]:
            dst_fp = os.path.join(out_base, split, os.path.basename(src_fp))
            shutil.copy(src_fp, dst_fp)

print("Done!")
print(f"  Subjects → train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")
