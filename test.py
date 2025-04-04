import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from dataset.TDLUDataset import TDLUDataset
from models import MGModule

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the dataset.
    dataset = TDLUDataset(
        image_dir='/fs/nexus-scratch/tuxunlu/git/tdlu/WUSTL. Unmodified mammograms-selected/WUSTL_png',
        csv_path='/fs/nexus-scratch/tuxunlu/git/tdlu/WUSTL. Unmodified mammograms-selected/umd_annot_md_TDLU_y2025m03d13.csv',
        augment=True,
        weights_json_path='/fs/nexus-scratch/tuxunlu/git/tdlu/dataset/weights.json',
        target="BreastDensity_avg",
        num_bins=10,
    )
    
    # Use a DataLoader with batch_size=1 for evaluation.
    _, test_dataloader = dataset.get_dataloaders(batch_size=1, train_split=0, num_workers=4, pin_memory=True)
    print(len(test_dataloader))
    
    # Initialize the model and load the trained weights.
    model = MGModule()  # Make sure MGModule doesn't attempt to load pretrained weights if not needed.
    model.load_state_dict(torch.load("trained_model_10bins.pth", map_location=device))
    model.to(device)
    model.eval()  # Set model to evaluation mode

    all_preds = []
    all_targets = []

    # Evaluate the model on the dataset.
    print("Evaluating on the dataset:")
    with torch.no_grad():
        pbar = tqdm(test_dataloader, desc="Evaluating")
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass.
            outputs, _ = model(images)
            # Obtain predicted class.
            _, preds = torch.max(outputs, dim=1)
            
            # Store predictions and targets.
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate results from each batch.
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate overall accuracy.
    accuracy = np.mean(all_preds == all_targets)
    print(f"Overall Accuracy: {accuracy:.4f}")

    # Compute the confusion matrix (raw counts).
    cm = confusion_matrix(all_targets, all_preds)
    
    # Normalize row-wise (each row sums to 1) to obtain percentages.
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = cm.astype('float') / row_sums

    # Plot the confusion matrix.
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix with Count and Row-wise Percentage")
    cbar = plt.colorbar(label="Normalized Percentage")
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    # Annotate each cell with the raw count and normalized percentage.
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = cm_normalized[i, j] * 100
            count = cm[i, j]
            plt.text(j, i, f"{count}\n({percentage:.1f}%)",
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm_normalized[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
