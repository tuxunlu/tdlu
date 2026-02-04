import numpy as np
import matplotlib.pyplot as plt

def plot_cm():
    # 1. Define the specific data
    cm = np.array([[672, 125],
                   [87, 43]])

    # 2. Compute Normalization (matching your snippet's logic)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    num_bins = cm.shape[0]

    # 3. Plotting (Exact format from your provided code)
    plt.figure(figsize=(8,6))
    plt.imshow(cm_norm, cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.title("Normalized Confusion Matrix")
    plt.colorbar()
    
    ticks = np.arange(num_bins)
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    thresh = cm_norm.max() / 2

    for i in range(num_bins):
        for j in range(num_bins):
            count = cm[i, j]
            pct   = cm_norm[i, j] * 100
            
            # Text annotation matching your style: Count + (Percentage)
            plt.text(
                j, i,
                f"{count}\n({pct:.1f}%)",
                ha='center', va='center',
                color='white' if cm_norm[i, j] > thresh else 'black'
            )

    plt.tight_layout()
    # Saving with the same DPI setting
    plt.savefig("confusion_matrix_style_match.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_cm()