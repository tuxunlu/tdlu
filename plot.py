import matplotlib.pyplot as plt
import numpy as np

cm = np.array([[133, 94, 26],
          [95, 93, 39],
          [31, 32, 15]])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(8,6))
plt.imshow(cm_norm, cmap=plt.cm.Blues, vmin=0, vmax=1)
plt.title("Normalized Confusion Matrix\nImage Only")
plt.colorbar()
ticks = np.arange(3)
plt.xticks(ticks, ticks); plt.yticks(ticks, ticks)
plt.xlabel("Predicted"); plt.ylabel("True")
thresh = cm_norm.max() / 2


for i in range(3):
        for j in range(3):
            count = cm[i, j]
            pct   = cm_norm[i, j] * 100
            plt.text(
                j, i,
                f"{count}\n({pct:.1f}%)",
                ha='center', va='center',
                color='white' if cm_norm[i, j] > thresh else 'black'
            )

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()