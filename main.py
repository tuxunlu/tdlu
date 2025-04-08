import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from models import MGModule  # assuming the hierarchical model is defined in models.py
from dataset.TDLUDataset import TDLUDataset
from utils import get_run_number

if __name__ == "__main__":
    # Configuration parameters.
    target = "tdlu_density"
    description = "fused+hierarchical+minmax"
    num_bins = 5  # Total number of bins/classes. Class 0 is handled separately.
    batch_size = 32
    num_save = 20

    run_dir = os.path.join("runs", f"mg_experiment_{get_run_number()}_{description}_{target}_{num_bins}")
    os.makedirs(run_dir, exist_ok=True)
    
    writer = SummaryWriter(run_dir)

    # Instantiate the dataset.
    dataset = TDLUDataset(
        image_dir='/fs/nexus-scratch/tuxunlu/git/tdlu/WUSTL_Unmodified_mammograms_selected/WUSTL_png_minmax',
        csv_path='/fs/nexus-scratch/tuxunlu/git/tdlu/WUSTL_Unmodified_mammograms_selected/umd_annot_md_TDLU_y2025m03d13.csv',
        augment=True,
        weights_json_path=None,
        target=target,
        num_bins=num_bins,
    )

    train_dataloader, test_dataloader = dataset.get_dataloaders(
        batch_size=batch_size, train_split=0.7, num_workers=4, pin_memory=True
    )

    # Initialize the hierarchical model.
    model = MGModule(pretrained_path="/fs/nexus-scratch/tuxunlu/git/tdlu/mgh_mammo_MIRAI_Base_May20_2019.p",
                     num_bins=num_bins)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Criterion and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    num_epochs = 500
    global_step = 0

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        # For visualization: count final class predictions.
        class_counts = torch.zeros(num_bins, dtype=torch.int32)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("LearningRate/train_epoch", current_lr, epoch)
        
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                    desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, breast_density, targets) in pbar:
            images = images.to(device)
            breast_density = breast_density.float().to(device)  # Ensure shape is [batch_size]
            targets = targets.long().to(device)  # Ground truth labels in [0, num_bins-1]

            # Update class counts for logging.
            class_counts += torch.bincount(targets.cpu(), minlength=num_bins)

            optimizer.zero_grad()
            # Forward pass returns binary and multi logits.
            binary_logits, multi_logits, features = model(images, breast_density)
            
            # Prepare binary targets: 0 if ground truth is 0, otherwise 1.
            binary_targets = torch.where(targets == 0, torch.zeros_like(targets), torch.ones_like(targets))
            binary_loss = criterion(binary_logits, binary_targets)
            
            # For non-zero targets, compute the multi-head loss.
            nonzero_mask = (targets != 0)
            if nonzero_mask.sum() > 0:
                # Adjust targets: subtract 1 so that classes 1..(num_bins-1) become 0..(num_bins-2)
                multi_targets = targets[nonzero_mask] - 1
                multi_loss = criterion(multi_logits[nonzero_mask], multi_targets)
            else:
                multi_loss = 0.0

            loss = binary_loss + multi_loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1

            # Compute predictions:
            # For the binary head: if predicted 0 then final class is 0, else use multi head.
            binary_pred = torch.argmax(binary_logits, dim=1)
            multi_pred = torch.argmax(multi_logits, dim=1) + 1  # shift back by adding 1
            final_pred = torch.where(binary_pred == 0, torch.zeros_like(binary_pred), multi_pred)

            correct += (final_pred == targets).sum().item()
            total += targets.size(0)
            
            writer.add_scalar("Loss/train_batch", loss.item(), global_step)
            pbar.set_postfix(loss=loss.item())

        scheduler.step()
        avg_loss = epoch_loss / len(train_dataloader)
        epoch_accuracy = correct / total if total > 0 else 0.0
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | "
              f"Train Acc: {epoch_accuracy:.4f} | LR: {current_lr:.6f}")
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        writer.add_scalar("Accuracy/train_epoch", epoch_accuracy, epoch)

        # Log class distribution histogram.
        fig, ax = plt.subplots()
        ax.bar(range(num_bins), class_counts.numpy())
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title(f"Class Distribution in Epoch {epoch+1}")
        writer.add_figure("Class_Distribution", fig, epoch)
        plt.close(fig)

        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, densities, targets in test_dataloader:
                images = images.to(device)
                densities = densities.float().to(device)
                targets = targets.long().to(device)
                binary_logits, multi_logits, _ = model(images, densities)
                
                binary_targets = torch.where(targets == 0, torch.zeros_like(targets), torch.ones_like(targets))
                binary_loss = criterion(binary_logits, binary_targets)
                nonzero_mask = (targets != 0)
                if nonzero_mask.sum() > 0:
                    multi_targets = targets[nonzero_mask] - 1
                    multi_loss = criterion(multi_logits[nonzero_mask], multi_targets)
                else:
                    multi_loss = 0.0
                loss = binary_loss + multi_loss
                test_loss += loss.item() * images.size(0)
                
                binary_pred = torch.argmax(binary_logits, dim=1)
                multi_pred = torch.argmax(multi_logits, dim=1) + 1
                final_pred = torch.where(binary_pred == 0, torch.zeros_like(binary_pred), multi_pred)
                test_correct += (final_pred == targets).sum().item()
                test_total += targets.size(0)
        avg_test_loss = test_loss / test_total if test_total > 0 else 0.0
        test_accuracy = test_correct / test_total if test_total > 0 else 0.0
        print(f"Epoch {epoch+1} | Test Loss: {avg_test_loss:.4f} | Test Acc: {test_accuracy:.4f}")
        writer.add_scalar("Loss/test_epoch", avg_test_loss, epoch)
        writer.add_scalar("Accuracy/test_epoch", test_accuracy, epoch)
        model.train()

        if (epoch + 1) % num_save == 0:
            checkpoint_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)

    final_model_path = os.path.join(run_dir, "trained_model.pth")
    torch.save(model.state_dict(), final_model_path)
    writer.close()
