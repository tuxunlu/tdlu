import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from PIL import Image
from dataset.TDLUDataset import TDLUDataset
from tqdm import tqdm

# Define the model architecture (same as used in training)
class MGModule(torch.nn.Module):
    def __init__(self):
        super(MGModule, self).__init__()
        resnet = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # Classification head for 64 classes
        self.head = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 64)
            # No activation function; raw logits are expected by CrossEntropyLoss
        )

    def forward(self, mg):
        mg_feature = self.backbone(mg)
        mg_feature_flatten = torch.flatten(mg_feature, 1, -1)
        out = self.head(mg_feature_flatten)
        return out, mg_feature_flatten

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the dataset and dataloader.
    # Update the image_dir and csv_path as needed.
    dataset = TDLUDataset(
        image_dir='/fs/nexus-scratch/tuxunlu/git/tdlu/WUSTL. Unmodified mammograms-selected/Batch 1_png',
        csv_path='/fs/nexus-scratch/tuxunlu/git/tdlu/WUSTL. Unmodified mammograms-selected/umd_annot_md_TDLU_y2025m03d13.csv'
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Initialize the model and load the trained weights
    model = MGModule()
    model.load_state_dict(torch.load("trained_model.pth", map_location=device))
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Run evaluation on each image in the dataloader
    print("Evaluating on the dataset:")
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            preds, _ = model(images)
            
            # Since batch_size is 1 by default, loop over batch elements in case of a different batch size.
            for pred, true in zip(preds.cpu().numpy(), targets.cpu().numpy()):
                print(f"Predicted: {pred}, True: {true}")

if __name__ == "__main__":
    main()
