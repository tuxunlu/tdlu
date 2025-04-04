import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from PIL import Image
from dataset.TDLUDataset import TDLUDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import get_run_number
from models import MGModule


if __name__ == "__main__":
    # Initialize TensorBoard writer.
    writer = SummaryWriter(os.path.join("runs", f"mg_experiment_{get_run_number()}"))

    # Instantiate the dataset and specify a path for storing weights.
    dataset = TDLUDataset(
        image_dir='/fs/nexus-scratch/tuxunlu/git/tdlu/WUSTL. Unmodified mammograms-selected/WUSTL_png',
        csv_path='/fs/nexus-scratch/tuxunlu/git/tdlu/WUSTL. Unmodified mammograms-selected/umd_annot_md_TDLU_y2025m03d13.csv',
        augment=True,
        weights_json_path='/fs/nexus-scratch/tuxunlu/git/tdlu/dataset/weights.json',  # JSON file for caching weights.
        target="BreastDensity_avg",  # Column name in CSV for target labels.
        num_bins=10,  # Number of bins for classification.
    )

    # Get a DataLoader with balanced sampling.
    train_dataloader, test_dataloader = dataset.get_dataloaders(batch_size=20, num_workers=4, pin_memory=True)

    # Initialize the model and load pretrained weights.
    model = MGModule(pretrained_path="/fs/nexus-scratch/tuxunlu/git/tdlu/mgh_mammo_MIRAI_Base_May20_2019.p")
    
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define cross entropy loss.
    # CrossEntropyLoss expects raw logits (shape [B, 40]) and target class indices (type torch.long).
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    num_epochs = 500
    global_step = 0

    # Training loop.
    model.train()  # set model to training mode.
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        # Log current learning rate.
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("LearningRate/train_epoch", current_lr, epoch)
        
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, targets) in pbar:
            images = images.to(device)
            targets = targets.long().to(device)  # targets as class indices.

            optimizer.zero_grad()
            outputs, features = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1

            # Compute training accuracy.
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
            writer.add_scalar("Loss/train_batch", loss.item(), global_step)
            pbar.set_postfix(loss=loss.item())

        # Scheduler step at end of epoch.
        scheduler.step()
        avg_loss = epoch_loss / len(train_dataloader)
        epoch_accuracy = correct / total if total > 0 else 0.0
        
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, LR: {current_lr:.6f}")
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        writer.add_scalar("Accuracy/train_epoch", epoch_accuracy, epoch)

    torch.save(model.state_dict(), "trained_model_10bins.pth")
    writer.close()
