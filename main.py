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



class ClassificationHead(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc_out = nn.Linear(64, 40)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        out = self.fc_out(x)
        return out

class MGModule(torch.nn.Module):
    def __init__(self, pretrained_path=None):
        super(MGModule, self).__init__()
        resnet = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # Replace the regression head with a classification head that outputs 40 logits.
        self.head = ClassificationHead(input_dim=resnet.fc.in_features, dropout_rate=0.5)
        self.mg_load_pretrained_model(pretrained_path)

    # Load mirai model as pretrained model
    def mg_load_pretrained_model(self, mirai_path: str):
        mirai_weight = torch.load(mirai_path, map_location="cpu", weights_only=False)
        mirai_weight = mirai_weight.module._model
        model_modules = self.backbone._modules
        model_modules["0"].load_state_dict(mirai_weight.downsampler.conv1.state_dict())
        model_modules["1"].load_state_dict(mirai_weight.downsampler.bn1.state_dict())
        model_modules["2"].load_state_dict(mirai_weight.downsampler.relu.state_dict())
        model_modules["3"].load_state_dict(mirai_weight.downsampler.maxpool.state_dict())

        model_modules["4"]._modules["0"].load_state_dict(mirai_weight.layer1_0.state_dict())
        model_modules["4"]._modules["1"].load_state_dict(mirai_weight.layer1_1.state_dict())
        model_modules["5"]._modules["0"].load_state_dict(mirai_weight.layer2_0.state_dict())
        model_modules["5"]._modules["1"].load_state_dict(mirai_weight.layer2_1.state_dict())
        model_modules["6"]._modules["0"].load_state_dict(mirai_weight.layer3_0.state_dict())
        model_modules["6"]._modules["1"].load_state_dict(mirai_weight.layer3_1.state_dict())
        model_modules["7"]._modules["0"].load_state_dict(mirai_weight.layer4_0.state_dict())
        model_modules["7"]._modules["1"].load_state_dict(mirai_weight.layer4_1.state_dict())

    def forward(self, mg):
        mg_feature = self.backbone(mg)
        mg_feature_flatten = torch.flatten(mg_feature, 1, -1)
        out = self.head(mg_feature_flatten)
        return out, mg_feature_flatten

if __name__ == "__main__":
    # Initialize TensorBoard writer.
    writer = SummaryWriter(os.path.join("runs", f"mg_experiment_{get_run_number()}"))

    # Instantiate the dataset and specify a path for storing weights.
    dataset = TDLUDataset(
        image_dir='/fs/nexus-scratch/tuxunlu/git/tdlu/WUSTL. Unmodified mammograms-selected/WUSTL_png',
        csv_path='/fs/nexus-scratch/tuxunlu/git/tdlu/WUSTL. Unmodified mammograms-selected/umd_annot_md_TDLU_y2025m03d13.csv',
        augment=True,
        weights_json_path=None,  # JSON file for caching weights.
        target="BreastDensity_avg",  # Column name in CSV for target labels.
        num_bins=40,  # Number of bins for classification.
    )

    # Get a DataLoader with balanced sampling.
    dataloader = dataset.get_dataloader(batch_size=10, num_workers=4, pin_memory=True)

    # Initialize the model and load pretrained weights.
    model = MGModule(pretrained_path="/fs/nexus-scratch/tuxunlu/git/tdlu/mgh_mammo_MIRAI_Base_May20_2019.p")
    
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define cross entropy loss.
    # CrossEntropyLoss expects raw logits (shape [B, 40]) and target class indices (type torch.long).
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    num_epochs = 50
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
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
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
        avg_loss = epoch_loss / len(dataloader)
        epoch_accuracy = correct / total if total > 0 else 0.0
        
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, LR: {current_lr:.6f}")
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        writer.add_scalar("Accuracy/train_epoch", epoch_accuracy, epoch)

    torch.save(model.state_dict(), "trained_model.pth")
    writer.close()
