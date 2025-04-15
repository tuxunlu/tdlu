import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from dataset.TDLUDataset import TDLUDataset
from models import MGModule, MGModule_SingleHead, MGModuleViT
import random

def grad_cam(model, image, density, target_class, target_layer):
    """
    Compute the Grad-CAM heatmap for a single image given a target class.
    
    Parameters:
        model (nn.Module): The classification model.
        image (torch.Tensor): A single image tensor with batch dimension, e.g., shape [1, C, H, W].
        density (torch.Tensor): The corresponding density input, shape [1, 1].
        target_class (int): The predicted or specified class index.
        target_layer (nn.Module): The convolutional layer to hook (e.g., model.backbone[7] for ResNet18 layer4).
    
    Returns:
        np.array: The Grad-CAM heatmap in numpy format (normalized between 0 and 1).
    """
    activations = []
    gradients = []

    # Forward hook: capture activations from the target layer.
    def forward_hook(module, input, output):
        activations.append(output.detach())

    # Backward hook: capture gradients from the target layer.
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    # Register hooks on the target layer.
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    # Forward pass: obtain model output.
    logits, _ = model(image, density)
    # Select the score corresponding to the target class.
    score = logits[0, target_class]
    
    # Clear any existing gradients and backpropagate.
    model.zero_grad()
    score.backward()

    # Remove hooks to avoid modifying the model state.
    forward_handle.remove()
    backward_handle.remove()

    # Retrieve the last stored activations and gradients.
    activation = activations[-1]   # Shape: [1, C, H, W]
    gradient = gradients[-1]       # Shape: [1, C, H, W]

    # Compute the weights: global average pooling of the gradients over the spatial dimensions.
    weights = torch.mean(gradient, dim=(2, 3), keepdim=True)  # Shape: [1, C, 1, 1]

    # Weighted combination of the activations.
    grad_cam_map = torch.sum(weights * activation, dim=1).squeeze()  # Shape: [H, W]

    # Apply ReLU and normalize the heatmap.
    grad_cam_map = torch.relu(grad_cam_map)
    grad_cam_map = (grad_cam_map - grad_cam_map.min()) / (grad_cam_map.max() - grad_cam_map.min() + 1e-8)
    
    return grad_cam_map.cpu().numpy()


def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_bins = 4

    random.seed(42)  # For reproducibility
    
    # Instantiate the dataset.
    dataset = TDLUDataset(
        image_dir='/fs/nexus-scratch/tuxunlu/git/tdlu/WUSTL_Unmodified_mammograms_selected/WUSTL_png_GE+minmax',
        csv_path='/fs/nexus-scratch/tuxunlu/git/tdlu/WUSTL_Unmodified_mammograms_selected/umd_annot_md_TDLU_y2025m03d13.csv',
        augment=True,
        weights_json_path='/fs/nexus-scratch/tuxunlu/git/tdlu/dataset/weights.json',
        target="tdlu_density",
        num_bins=num_bins,
    )
    
    # Use a DataLoader with batch_size=20 for evaluation.
    # Note: Ensure that the dataloader returns three items: images, breast_density, and targets.
    _, test_dataloader = dataset.get_dataloaders(batch_size=20, train_split=0.7, num_workers=4, pin_memory=True)
    print(f"Number of test batches: {len(test_dataloader)}")
    
    # Initialize the model and load the trained weights.
    model = MGModule_SingleHead(num_bins=num_bins)  # Ensure the model does not automatically load pretrained weights.
    model.load_state_dict(torch.load("/fs/nexus-scratch/tuxunlu/git/tdlu/runs/mg_experiment_20_fused+single_head+minmax+GE+multilevel_tdlu_density_4/checkpoint_epoch_180.pth", map_location=device))
    model.to(device)
    model.eval()  # Set model to evaluation mode

    all_preds = []
    all_targets = []
    all_probs = []  # To store the softmax probabilities for ROC computation

    # # Evaluate the model on the dataset.
    # print("Evaluating on the dataset:")
    # with torch.no_grad():
    #     pbar = tqdm(test_dataloader, desc="Evaluating")
    #     for images, breast_density, targets in pbar:
    #         images = images.to(device)
    #         breast_density = breast_density.float().to(device)  # Ensure density is in floating point.
    #         targets = targets.to(device)
            
    #         # Forward pass: pass both images and breast_density.
    #         outputs, _ = model(images, breast_density)
            
    #         # Compute softmax probabilities.
    #         probs = torch.softmax(outputs, dim=1)
    #         all_probs.append(probs.cpu().numpy())
            
    #         # Obtain predicted class.
    #         _, preds = torch.max(outputs, dim=1)
            
    #         # Store predictions and targets.
    #         all_preds.append(preds.cpu().numpy())
    #         all_targets.append(targets.cpu().numpy())

    # # Concatenate results from each batch.
    # all_preds = np.concatenate(all_preds)
    # all_targets = np.concatenate(all_targets)
    # all_probs = np.concatenate(all_probs, axis=0)

    # # Calculate overall accuracy.
    # accuracy = np.mean(all_preds == all_targets)
    # print(f"Overall Accuracy: {accuracy:.4f}")

    # # Compute the confusion matrix (raw counts).
    # cm = confusion_matrix(all_targets, all_preds)
    
    # # Normalize row-wise (each row sums to 1) to obtain percentages.
    # row_sums = cm.sum(axis=1)[:, np.newaxis]
    # cm_normalized = cm.astype('float') / row_sums

    # # Plot the confusion matrix.
    # plt.figure(figsize=(10, 8))
    # plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title("Confusion Matrix with Count and Row-wise Percentage")
    # cbar = plt.colorbar(label="Normalized Percentage")
    
    # tick_marks = np.arange(num_bins)
    # plt.xticks(tick_marks, tick_marks)
    # plt.yticks(tick_marks, tick_marks)
    # plt.xlabel("Predicted Label")
    # plt.ylabel("True Label")
    
    # # Annotate each cell with the raw count and normalized percentage.
    # thresh = cm_normalized.max() / 2.
    # for i in range(cm.shape[0]):
    #     for j in range(cm.shape[1]):
    #         percentage = cm_normalized[i, j] * 100
    #         count = cm[i, j]
    #         plt.text(j, i, f"{count}\n({percentage:.1f}%)",
    #                  horizontalalignment="center",
    #                  verticalalignment="center",
    #                  color="white" if cm_normalized[i, j] > thresh else "black")

    # plt.tight_layout()
    # plt.savefig("confusion_matrix.png", dpi=300)
    # plt.show()

    # # ----- ROC Curve Computation and Plotting -----
    # # Convert ground truth into one-hot encoded format for multiclass ROC computation.
    # all_targets_bin = label_binarize(all_targets, classes=range(num_bins))
    
    # # Compute ROC curve and ROC area for each class.
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(num_bins):
    #     fpr[i], tpr[i], _ = roc_curve(all_targets_bin[:, i], all_probs[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    
    # # Compute micro-average ROC curve and ROC area.
    # fpr["micro"], tpr["micro"], _ = roc_curve(all_targets_bin.ravel(), all_probs.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # # Plot ROC curves.
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
    #          linestyle=':', linewidth=4)
    
    # # Define a list of colors for each class.
    # colors = ['aqua', 'darkorange', 'cornflowerblue', 'green']
    # for i, color in zip(range(num_bins), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=2,
    #              label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')

    # plt.plot([0, 1], [0, 1], 'k--', lw=2)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curves')
    # plt.legend(loc="lower right")
    # plt.savefig("roc_curve.png", dpi=300)
    # plt.show()

    # ----- Single Input Prediction with Saliency Map -----
    print("\n--- Single Input Prediction with Saliency Map ---")
    # Choose 10 random samples from the dataset
    num_samples = 1
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i in indices:
        # Get single sample.
        single_image, single_breast_density, single_target = dataset[i]
        # Retrieve subject_id from the corresponding image file.
        subject_id = dataset.image_files[i].split(".")[0]
        
        # Add batch dimension and enable gradient tracking for saliency computation.
        single_image_batch = single_image.unsqueeze(0).to(device)
        single_image_batch.requires_grad_()  # Enable gradients with respect to the image.
        
        single_breast_density_batch = torch.tensor([single_breast_density]).float().to(device)
        
        # Clear previous gradients.
        model.zero_grad()
        
        # Forward pass to get predictions.
        output, _ = model(single_image_batch, single_breast_density_batch)
        _, single_pred = torch.max(output, dim=1)
        
        print(f"Single Input Prediction for subject {subject_id}: {single_pred.item()}, Ground Truth: {single_target}")
        
        # ----- Saliency Map Computation -----
        # Select the score for the predicted class.
        score = output[0, single_pred]
        # Backpropagate to get the gradient with respect to the input image.
        score.backward()
        
        # Get the gradients on the input image. For multi-channel images, take maximum across channels.
        grad_saliency = single_image_batch.grad.data.abs()
        saliency, _ = torch.max(grad_saliency, dim=1)  # shape: [1, H, W]
        saliency = saliency.squeeze().cpu().numpy()
        
        # Normalize the saliency map to [0, 1] for visualization.
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        # Plot and save the saliency map.
        plt.figure(figsize=(6, 5))
        plt.imshow(saliency, cmap='hot')
        plt.title(f"Saliency Map for subject {subject_id}")
        plt.colorbar()
        plt.savefig(f"saliency_map_{subject_id}.png", dpi=300)
        plt.show()


    # ----- Single Input Grad-CAM Visualization -----
    print("\n--- Single Input Grad-CAM Visualization ---")
    # Choose one random sample from your dataset. (Adjust num_samples if desired.)
    sample_index = indices[0]

    # Retrieve a single sample: image, density, and target.
    single_image, single_breast_density, single_target = dataset[sample_index]
    subject_id = dataset.image_files[sample_index].split(".")[0]

    # Prepare the inputs: add batch dimensions and set appropriate device.
    single_image_batch = single_image.unsqueeze(0).to(device)
    # Ensure density tensor has shape [1, 1]
    single_breast_density = torch.tensor([single_breast_density]).unsqueeze(0).float().to(device)

    # Forward pass to obtain predictions.
    logits, _ = model(single_image_batch, single_breast_density)
    _, predicted_class = torch.max(logits, dim=1)
    print(f"Grad-CAM Prediction for subject {subject_id}: {predicted_class.item()}, Ground Truth: {single_target}")

    # ----- Grad-CAM Computation -----
    # Here we choose the last convolutional layer of the backbone.
    # For ResNet18, model.backbone is a Sequential of:
    # [0] conv1, [1] bn1, [2] relu, [3] maxpool, [4] layer1, [5] layer2, [6] layer3, [7] layer4, [8] avgpool.
    # We use index [7] as the target for Grad-CAM.
    target_layer = model.backbone[7]

    # Compute the heatmap for the predicted class.
    heatmap = grad_cam(model, single_image_batch, single_breast_density, predicted_class.item(), target_layer)

    # ----- Visualization -----
    plt.figure(figsize=(6, 5))
    # Assume the input image is a 3-channel image; convert to numpy.
    # If your images are normalized, you might need to denormalize them before visualizing.
    img_np = single_image.squeeze().permute(1, 2, 0).cpu().numpy()

    plt.imshow(img_np, cmap='gray')
    # Overlay the heatmap with an alpha (transparency) value.
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title(f"Grad-CAM for subject {subject_id}")
    plt.colorbar(label="Activation")
    plt.savefig(f"gradcam_{subject_id}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
