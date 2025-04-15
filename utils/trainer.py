import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_and_evaluate(
    model, train_dataloader, test_dataloader, criterion, writer, run_dir,
    num_epochs, learning_rate, num_save, num_bins
):
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    global_step = 0

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        class_counts = torch.zeros(num_bins, dtype=torch.int32)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("LearningRate/train_epoch", current_lr, epoch)
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                    desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, breast_density, targets) in pbar:
            images = images.to(device)
            breast_density = breast_density.float().to(device)
            targets = targets.long().to(device)
            class_counts += torch.bincount(targets.cpu(), minlength=num_bins)
            optimizer.zero_grad()
            
            # Assume model returns logits for all classes and any additional features as needed.
            logits, features = model(images, breast_density)
            
            # Use a single cross entropy loss over all classes
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            global_step += 1
            
            # Prediction: choose the class with the highest logit value
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
            
            writer.add_scalar("Loss/train_batch", loss.item(), global_step)
            pbar.set_postfix(loss=loss.item())
        scheduler.step()
        avg_loss = epoch_loss / len(train_dataloader)
        epoch_accuracy = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | Train Acc: {epoch_accuracy:.4f} | LR: {current_lr:.6f}")
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        writer.add_scalar("Accuracy/train_epoch", epoch_accuracy, epoch)
        
        # Log the class distribution for the current epoch
        fig, ax = plt.subplots()
        ax.bar(range(num_bins), class_counts.numpy())
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title(f"Class Distribution in Epoch {epoch+1}")
        writer.add_figure("Class_Distribution", fig, epoch)
        plt.close(fig)
        
        # --- Evaluation ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, densities, targets in test_dataloader:
                images = images.to(device)
                densities = densities.float().to(device)
                targets = targets.long().to(device)
                
                # Get logits and compute loss as in training
                logits, _ = model(images, densities)
                loss = criterion(logits, targets)
                test_loss += loss.item() * images.size(0)
                
                predictions = torch.argmax(logits, dim=1)
                test_correct += (predictions == targets).sum().item()
                test_total += targets.size(0)
                
        avg_test_loss = test_loss / test_total if test_total > 0 else 0.0
        test_accuracy = test_correct / test_total if test_total > 0 else 0.0
        print(f"Epoch {epoch+1} | Test Loss: {avg_test_loss:.4f} | Test Acc: {test_accuracy:.4f}")
        writer.add_scalar("Loss/test_epoch", avg_test_loss, epoch)
        writer.add_scalar("Accuracy/test_epoch", test_accuracy, epoch)
        model.train()
        
        # Save checkpoint every 'num_save' epochs
        if (epoch + 1) % num_save == 0:
            checkpoint_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch+1}.pth")
            # Handle models wrapped with DataParallel if applicable
            if hasattr(model, "module"):
                torch.save(model.module.state_dict(), checkpoint_path)
            else:
                torch.save(model.state_dict(), checkpoint_path)
                
    final_model_path = os.path.join(run_dir, "trained_model.pth")
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), final_model_path)
    else:
        torch.save(model.state_dict(), final_model_path)
    writer.close()
