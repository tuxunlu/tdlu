#!/usr/bin/env python3
"""
Count the number of samples per class for the STAMP TDLU dataset.
Uses the same filtering and binning logic as TDLUDataset_torchio_four_view_mask_meta_avg_10_folds_STAMP.
"""
import argparse
import random
import sys
from collections import Counter
import numpy as np
import pandas as pd
import yaml


def bin_value_quantile(value, thresholds):
    """Return which bin value belongs to based on thresholds."""
    for i, t in enumerate(thresholds):
        if value <= t:
            return i
    return len(thresholds)


def get_class_label(raw_target, label_type, thresholds):
    """Assign class label based on label_type."""
    if label_type == "raw":
        return bin_value_quantile(raw_target, thresholds)
    elif label_type == "label":
        return int(raw_target)
    elif label_type == "zero":
        if raw_target == 0:
            return 0
        return bin_value_quantile(raw_target, thresholds) + 1
    else:
        return bin_value_quantile(raw_target, thresholds)


def count_class_samples(
    csv_path: str,
    target: str = "tdlu_density_extreme_zero",
    meta_cols: list = None,
    num_bins: int = 3,
    label_type: str = "label",
    split_ratio: tuple = (0.8, 0.2, 0.0),
    seed: int = 1234,
):
    """
    Count samples per class for train/val/test splits.
    Applies the same filtering logic as TDLUDataset_torchio_four_view_mask_meta_avg_10_folds_STAMP.
    """
    if meta_cols is None:
        meta_cols = ["mamm_age", "PATIENTI_BMI", "BreastDensity", "RACE_COMPOSITE"]

    allowed_combos = [("CC", "L"), ("CC", "R")]
    exclusions = {}  # track excluded counts and reasons

    df = pd.read_csv(csv_path)
    n_total_rows = len(df)
    n_unique_subjects_raw = df["studyid"].nunique()

    # Exclude MLO views
    n_mlo = (df["ViewPosition"] == "MLO").sum()
    df = df[df["ViewPosition"] != "MLO"]
    exclusions["MLO views (rows)"] = n_mlo

    df["ImageLaterality"] = df["breastside_final"].map({"Left": "L", "Right": "R"})
    df["subject_id"] = df["studyid"].astype(str)
    df["filename"] = df["Transformed_file_namenew"]

    # Exclude geanc_Race == 'N' or na
    df["geanc_Race"] = df["RACE_COMPOSITE"]

    # Exclude view/laterality not in allowed combos (CC/L, CC/R)
    in_allowed = df[["ViewPosition", "ImageLaterality"]].apply(
        lambda r: (r["ViewPosition"], r["ImageLaterality"]) in allowed_combos,
        axis=1,
    )
    n_bad_combo = (~in_allowed).sum()
    df = df[in_allowed]
    exclusions["ViewPosition/ImageLaterality not in (CC/L, CC/R) (rows)"] = n_bad_combo

    n_rows_after_row_filters = len(df)
    n_subjects_after_row_filters = df["subject_id"].nunique()

    vals = pd.to_numeric(df[target], errors="coerce")

    if label_type == "zero":
        num_bins_for_thresh = num_bins - 1
        valid = vals[(vals.notna()) & (vals > 0)]
    else:
        num_bins_for_thresh = num_bins
        valid = vals[(vals.notna()) & (vals > 0)]

    quantiles = [i / num_bins_for_thresh for i in range(1, num_bins_for_thresh)]
    thresholds = valid.quantile(quantiles).tolist()

    # Subject-level exclusions
    n_excl_no_combos = 0
    n_excl_nan_target = 0
    excl_nonfinite_meta = {}  # sid -> list of non-finite meta column names

    all_data = {}
    for sid, group in df.groupby("subject_id"):
        combos = {}
        for _, row in group.iterrows():
            key = (row["ViewPosition"], row["ImageLaterality"])
            fname = row["filename"]
            if not fname.endswith(".png"):
                fname = fname + ".png"
            combos[key] = fname

        if len(combos) == 0:
            n_excl_no_combos += 1
            continue
        if not any(k in allowed_combos for k in combos.keys()):
            n_excl_no_combos += 1
            continue

        raw_target = float(group.iloc[0][target])
        if np.isnan(raw_target):
            n_excl_nan_target += 1
            continue

        subj_meta_series = pd.to_numeric(group.iloc[0][meta_cols], errors="coerce")
        subj_meta = subj_meta_series.values.astype(float)
        if not np.isfinite(subj_meta).all():
            bad_cols = [meta_cols[i] for i in range(len(meta_cols)) if not np.isfinite(subj_meta[i])]
            excl_nonfinite_meta[sid] = bad_cols
            continue

        all_data[sid] = {
            "combos": combos,
            "target": raw_target,
            "subject_meta": subj_meta,
        }

    exclusions["No valid view combos (subjects)"] = n_excl_no_combos
    exclusions["NaN target (subjects)"] = n_excl_nan_target
    exclusions["Non-finite meta (subjects)"] = len(excl_nonfinite_meta)
    exclusions["Non-finite meta details"] = excl_nonfinite_meta

    subs = list(all_data.keys())
    random.seed(seed)
    random.shuffle(subs)
    n = len(subs)

    train_ratio, val_ratio, test_ratio = split_ratio
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_ids = list(subs[:n_train])
    val_ids = list(subs[n_train : n_train + n_val])
    test_ids = list(subs[n_train + n_val :])

    def count_for_subjects(subject_ids):
        labels = []
        for sid in subject_ids:
            raw = all_data[sid]["target"]
            lbl = get_class_label(raw, label_type, thresholds)
            labels.append(lbl)
        return Counter(labels)

    overall_counts = count_for_subjects(subs)
    train_counts = count_for_subjects(train_ids)
    val_counts = count_for_subjects(val_ids)
    test_counts = count_for_subjects(test_ids)

    num_classes = max(overall_counts.keys()) + 1 if overall_counts else 0

    return {
        "overall": overall_counts,
        "train": train_counts,
        "validation": val_counts,
        "test": test_counts,
        "num_classes": num_classes,
        "total": n,
        "thresholds": thresholds,
        "exclusions": exclusions,
        "n_total_rows": n_total_rows,
        "n_unique_subjects_raw": n_unique_subjects_raw,
        "n_rows_after_row_filters": n_rows_after_row_filters,
        "n_subjects_after_row_filters": n_subjects_after_row_filters,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Count samples per class for STAMP TDLU dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides other args)",
    )
    parser.add_argument("--csv_path", type=str, help="Path to annotation CSV")
    parser.add_argument(
        "--target",
        type=str,
        default="tdlu_density_extreme_zero",
        help="Target column name",
    )
    parser.add_argument(
        "--meta_cols",
        type=str,
        nargs="+",
        default=["mamm_age", "PATIENTI_BMI", "BreastDensity", "RACE_COMPOSITE"],
        help="Meta columns for filtering",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=3,
        help="Number of bins (for raw/zero label types)",
    )
    parser.add_argument(
        "--label_type",
        type=str,
        choices=["label", "raw", "zero"],
        default="label",
        help="Label type: label (categorical), raw (quantile-binned), zero (0 vs quantile-binned)",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        nargs=3,
        default=[0.8, 0.2, 0.0],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test split ratio",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for split")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        csv_path = cfg.get("csv_path")
        if not csv_path:
            print("Error: csv_path not found in config", file=sys.stderr)
            sys.exit(1)
        meta_cols = cfg.get("meta_cols", ["mamm_age", "PATIENTI_BMI", "BreastDensity", "RACE_COMPOSITE"])
        num_bins = cfg.get("num_bins", 3)
        target = cfg.get("target", "tdlu_density_extreme_zero")
        label_type = cfg.get("label_type", "label")
        split_ratio = tuple(cfg.get("split_ratio", [0.8, 0.2, 0.0]))
        seed = cfg.get("seed", 1234)
    else:
        if not args.csv_path:
            print("Error: --csv_path required when not using --config", file=sys.stderr)
            sys.exit(1)
        csv_path = args.csv_path
        meta_cols = args.meta_cols
        num_bins = args.num_bins
        target = args.target
        label_type = args.label_type
        split_ratio = tuple(args.split_ratio)
        seed = args.seed

    result = count_class_samples(
        csv_path=csv_path,
        target=target,
        meta_cols=meta_cols,
        num_bins=num_bins,
        label_type=label_type,
        split_ratio=split_ratio,
        seed=seed,
    )

    print("=" * 60)
    print("STAMP TDLU Dataset - Class Sample Counts")
    print("=" * 60)
    print(f"CSV: {csv_path}")
    print(f"Target: {target}, label_type: {label_type}, num_bins: {num_bins}")
    print(f"Split ratio: {split_ratio}")
    print()
    print("--- Excluded Samples ---")
    print(f"  Total rows in CSV: {result['n_total_rows']}")
    print(f"  Unique subjects in CSV: {result['n_unique_subjects_raw']}")
    for reason, val in result["exclusions"].items():
        if reason == "Non-finite meta details":
            continue  # printed separately below
        if val > 0:
            print(f"  Excluded ({reason}): {val}")
    nonfinite_details = result["exclusions"].get("Non-finite meta details", {})
    if nonfinite_details:
        print("  Non-finite meta (subject_id -> meta columns):")
        for sid, bad_cols in sorted(nonfinite_details.items()):
            print(f"    {sid}: {bad_cols}")
    n_excl_total_rows = sum(
        v for k, v in result["exclusions"].items() if "(rows)" in k
    )
    n_excl_total_subjects = sum(
        v for k, v in result["exclusions"].items() if "(subjects)" in k
    )
    print(f"  Rows after row-level filters: {result['n_rows_after_row_filters']} "
          f"(excluded {n_excl_total_rows} rows)")
    print(f"  Subjects after row-level filters: {result['n_subjects_after_row_filters']}")
    print(f"  Subjects excluded at subject-level: {n_excl_total_subjects}")
    print(f"  Included subjects: {result['total']}")
    print()
    print(f"Number of classes: {result['num_classes']}")
    if result["thresholds"]:
        print(f"Quantile thresholds (for raw/zero): {result['thresholds']}")
    print()

    for split_name, counts in [
        ("Overall", result["overall"]),
        ("Train", result["train"]),
        ("Validation", result["validation"]),
        ("Test", result["test"]),
    ]:
        total = sum(counts.values())
        print(f"--- {split_name} (n={total}) ---")
        for c in range(result["num_classes"]):
            cnt = counts.get(c, 0)
            pct = 100 * cnt / total if total > 0 else 0
            print(f"  Class {c}: {cnt:5d} ({pct:5.1f}%)")
        print()

    return result


if __name__ == "__main__":
    main()
