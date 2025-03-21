import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from PIL import Image
from dataset.TDLUDataset import TDLUDataset
from tqdm import tqdm


class MGModule(torch.nn.Module):
    def __init__(self):
        super(MGModule, self).__init__()
        resnet = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # Regression
        self.head = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1),
        )

    def forward(self, mg):
        mg_feature = self.backbone(mg)
        mg_feature_flatten = torch.flatten(mg_feature, 1, -1)
        out = self.head(mg_feature_flatten)
        return out, mg_feature_flatten


# Load mirai model as pretrained model
def mg_load_pretrained_model(model: MGModule, mirai_path: str):
    mirai_weight = torch.load(mirai_path, map_location="cpu", weights_only=False)
    mirai_weight = mirai_weight.module._model
    model_modules = model.backbone._modules
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

    torch.nn.parallel.data_parallel
    return model


if __name__ == "__main__":
    dataset = TDLUDataset(
        image_dir='/Volumes/PRO-G40/WUSTL. Unmodified mammograms-selected/Batch 1_png',
        csv_path='/Volumes/PRO-G40/WUSTL. Unmodified mammograms-selected/umd_annot_md_TDLU_y2025m03d13.csv',
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    model = MGModule()
    model = mg_load_pretrained_model(model, "/Users/nicklu/Documents/Git/tdlu/mgh_mammo_MIRAI_Base_May20_2019.p")
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Set number of epochs
    num_epochs = 10

    # Training loop
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        # Wrap the dataloader with tqdm for a progress bar
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, targets) in pbar:
            # Move data to the proper device
            images = images.to(device)
            targets = targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass: compute predictions and extract features if needed
            outputs, features = model(images)
            
            # Compute the loss; squeeze outputs if needed to match the target shape
            loss = criterion(outputs.squeeze(), targets.float())
            
            # Backward pass: compute gradients and update weights
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            # Update tqdm description with current loss
            pbar.set_postfix(loss=loss.item())

        # Print average loss for the epoch
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    