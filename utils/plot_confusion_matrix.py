"""
Utility to plot confusion matrices in the same style as ModelInterface.
Supports both integer counts and float values (e.g. averaged across folds).
"""
import os
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Optional[str] = None,
    class_names: Optional[list] = None,
    title: str = "Confusion Matrix",
    xlabel: str = "Predicted",
    ylabel: str = "True",
    figsize: tuple = (8, 6),
    dpi: int = 150,
) -> plt.Figure:
    """
    Plot a confusion matrix using the same style as ModelInterface.

    Args:
        cm: 2D array of shape (n_classes, n_classes). Rows = True, Cols = Predicted.
        save_path: If provided, save the figure to this path.
        class_names: Labels for classes (default: "0", "1", "2", ...).
        title: Figure title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size (width, height).
        dpi: DPI for saved figure.

    Returns:
        The matplotlib Figure.
    """
    cm = np.asarray(cm)
    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # Use integer format for int arrays, float format otherwise
    is_int = np.issubdtype(cm.dtype, np.integer) or np.all(np.equal(np.mod(cm, 1), 0))
    fmt = "d" if is_int else ".2f"

    fig, ax = plt.subplots(figsize=figsize)
    if _HAS_SEABORN:
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={"label": "Count" if is_int else "Value"},
        )
    else:
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax, label="Count" if is_int else "Value")
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:{fmt}}",
                    ha="center",
                    va="center",
                    color="black",
                )
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if save_path:
        dirname = os.path.dirname(save_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig


if __name__ == "__main__":
    # Example: plot the user's provided confusion matrix (averaged across folds)
    cm = np.array([
        [13.5, 16.75, 14.75],
        [12.75, 53.0, 23.25],
        [7.0, 14.5, 18.5],
    ])
    plot_confusion_matrix(
        cm,
        save_path="confusion_matrix_avg.png",
        title="Confusion Matrix (Averaged)",
    )
    print("Saved confusion_matrix_avg.png")
