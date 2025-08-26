from copyreg import pickle
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from data.TDLUDataset_torchio_four_view_meta_avg_folds import TdludatasetTorchioFourViewMetaAvgFolds
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm
import torch
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


def dataset_to_xy(torch_dataset, batch_size=512, num_workers=12):
    def collate_fn(batch):
        # each item: (_, meta, label, _)
        metas = []
        labels = []
        for _, meta, label, _ in batch:
            # assume meta is tensor-like of shape (4,) and label is scalar tensor or number
            if isinstance(meta, torch.Tensor):
                metas.append(meta)
            else:
                metas.append(torch.tensor(meta))
            if isinstance(label, torch.Tensor):
                labels.append(label.squeeze())
            else:
                labels.append(torch.tensor(label))
        metas = torch.stack(metas)  # (B, 4)
        labels = torch.stack(labels)  # (B,) 
        return metas, labels

    loader = DataLoader(
        torch_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    metas_list = []
    labels_list = []
    for metas, labels in tqdm(loader, desc="Extracting samples (batched)", unit="batch"):
        metas_list.append(metas.detach().cpu().numpy())
        labels_list.append(labels.detach().cpu().numpy())

    X_np = np.concatenate(metas_list, axis=0)  # (N,4)
    y = np.concatenate(labels_list, axis=0).astype(int)

    df = pd.DataFrame(
        X_np,
        columns=['breast_density', 'bmi', 'age', 'race']
    )

    return df, y

if __name__ == "__main__":
    test = True
    fold = 5
    
    train_dataset = TdludatasetTorchioFourViewMetaAvgFolds(
        image_dir='/beacon-scratch/tuxunlu/git/tdlu/dataset/WUSTL_png_nomarker_16',
        csv_path='/beacon-scratch/tuxunlu/git/tdlu/dataset/umd_annot_md_TDLU_y2025m07d09.csv',
        num_bins=3,
        target='tdlu_density_extreme_zero',
        label_type="label",
        meta_cols=['BreastDensity', 'mamm_age', 'cbmi_donation', 'geanc_Race'],
        purpose='train',
        cross_val_fold=fold,
    )
    val_dataset = TdludatasetTorchioFourViewMetaAvgFolds(
        image_dir='/beacon-scratch/tuxunlu/git/tdlu/dataset/WUSTL_png_nomarker_16',
        csv_path='/beacon-scratch/tuxunlu/git/tdlu/dataset/umd_annot_md_TDLU_y2025m07d09.csv',
        num_bins=3,
        target='tdlu_density_extreme_zero',
        label_type="label",
        meta_cols=['BreastDensity', 'mamm_age', 'cbmi_donation', 'geanc_Race'],
        purpose='validation',
        cross_val_fold=fold,
    )
    test_dataset = TdludatasetTorchioFourViewMetaAvgFolds(
        image_dir='/beacon-scratch/tuxunlu/git/tdlu/dataset/WUSTL_png_nomarker_16',
        csv_path='/beacon-scratch/tuxunlu/git/tdlu/dataset/umd_annot_md_TDLU_y2025m07d09.csv',
        num_bins=3,
        target='tdlu_density_extreme_zero',
        label_type="label",
        meta_cols=['BreastDensity', 'mamm_age', 'cbmi_donation', 'geanc_Race'],
        purpose='test',
        cross_val_fold=fold,
    )

    if test is False:
        numeric_features = ["breast_density", "bmi", "age", "race"]
        numeric_transformer = Pipeline([
            ("scaler", StandardScaler())
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_features),
        ])

        clf = Pipeline([
            ("preprocessor", preprocessor),
            ("logreg", LogisticRegression(
                solver="saga",              # supports l1/l2/elasticnet; robust for regularization
                penalty="l2",               # default; change as needed
                class_weight="balanced",    # automatically inverse-weights by frequency
                max_iter=2000,
                random_state=42,
                verbose=1,
            )),
        ])

        X_train, y_train = dataset_to_xy(train_dataset)

        print("Created datasets:")

        clf.fit(X_train, y_train)

        # save
        with open('model.pkl','wb') as f:
            pickle.dump(clf,f)

    else:
        with open('model.pkl', 'rb') as f:
            clf = pickle.load(f)

        # val_X, val_y = dataset_to_xy(val_dataset)

        # y_pred = clf.predict(val_X)
        # y_score = clf.predict_proba(val_X)

        # print(classification_report(val_y, y_pred))
        # roc_auc = roc_auc_score(
        #     val_y,
        #     y_score,
        #     multi_class="ovr",    # one-vs-rest scheme
        #     average="macro"       # macro to treat classes equally
        # )
        # print(f"Validation ROC AUC: {roc_auc:.4f}")

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            num_workers=12,
            shuffle=False,
            persistent_workers=True
        )

        all_preds, all_targets, all_probs, all_breast_density= [], [], [], []
        with torch.no_grad():
            for _, meta, label, _ in test_dataloader:
                meta = meta.detach().cpu().numpy()  # (1, 4)
                label = label.detach().cpu().numpy()  # (1,)
                
                df = pd.DataFrame(
                    meta,
                    columns=['breast_density', 'bmi', 'age', 'race']
                )

                preds = clf.predict(df)
                breast_density = meta[0][0]
                all_breast_density.append(breast_density)

                # all_probs.append(probs.cpu().numpy())
                all_preds.append(preds)
                all_targets.append(label)

        all_preds   = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        # all_probs   = np.concatenate(all_probs, axis=0)
        all_breast_density = np.array(all_breast_density)

        # Convert one-hot back to integer labels if needed
        if all_targets.ndim > 1:
            all_targets = np.argmax(all_targets, axis=1)

        # — Accuracy —
        accuracy = (all_preds == all_targets).mean()
        f1_macro = f1_score(all_targets, all_preds, average='macro')
        print(f"Overall test accuracy: {accuracy:.4f}")
        print(f"F1-score (macro): {f1_macro:.4f}")

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
        plt.title("Normalized Confusion Matrix\n(with avg. breast density)")
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
                    f"{count}\n({pct:.1f}%)\n{den_str}\n{std_str}",
                    ha='center', va='center',
                    color='white' if cm_norm[i, j] > thresh else 'black'
                )

        plt.tight_layout()
        plt.savefig("confusion_matrix_with_density_logistic.png", dpi=300)
        plt.close()
