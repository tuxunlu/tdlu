import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import ResNet18_Weights

class ClassificationHead(nn.Module):
    """
    This head performs classification into num_classes categories.
    """
    def __init__(self, input_dim, num_classes, dropout_rate=0.5):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc_out(x)
        return out

class MGModule_SingleHead(nn.Module):
    def __init__(self, num_bins, pretrained_path=None):
        super(MGModule_SingleHead, self).__init__()
        # Load a pre-trained ResNet18 backbone.
        resnet = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove the final fully connected layer.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.image_feature_dim = resnet.fc.in_features  # e.g., typically 512

        # Density Embedding: embed density (1D) into a 64-dimensional vector.
        self.density_emb = nn.Sequential(
            nn.Linear(1, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        
        # First-Level Fusion: fuse the backbone's image features and the density embedding.
        # Combined dimension: image_feature_dim + 64 (e.g., 512 + 64 = 576).
        self.early_fusion = nn.Sequential(
            nn.Linear(self.image_feature_dim + 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Second-Level Fusion: reintroduce the density embedding to the early fused features.
        # Concatenated dimension: 512 + 64 = 576, projected to 256.
        self.fusion_fc = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Single classification head to output predictions over num_bins classes.
        self.classification_head = ClassificationHead(input_dim=256, num_classes=num_bins, dropout_rate=0.5)
        
        if pretrained_path is not None:
            self.mg_load_pretrained_model(pretrained_path)
    
    def mg_load_pretrained_model(self, mirai_path: str):
        # Optionally load pretrained weights for the backbone.
        mirai_weight = torch.load(mirai_path, map_location="cpu", weights_only=False)
        mirai_weight = mirai_weight.module._model
        self.backbone._modules["0"].load_state_dict(mirai_weight.downsampler.conv1.state_dict())
        self.backbone._modules["1"].load_state_dict(mirai_weight.downsampler.bn1.state_dict())
        self.backbone._modules["2"].load_state_dict(mirai_weight.downsampler.relu.state_dict())
        self.backbone._modules["3"].load_state_dict(mirai_weight.downsampler.maxpool.state_dict())
        self.backbone._modules["4"]._modules["0"].load_state_dict(mirai_weight.layer1_0.state_dict())
        self.backbone._modules["4"]._modules["1"].load_state_dict(mirai_weight.layer1_1.state_dict())
        self.backbone._modules["5"]._modules["0"].load_state_dict(mirai_weight.layer2_0.state_dict())
        self.backbone._modules["5"]._modules["1"].load_state_dict(mirai_weight.layer2_1.state_dict())
        self.backbone._modules["6"]._modules["0"].load_state_dict(mirai_weight.layer3_0.state_dict())
        self.backbone._modules["6"]._modules["1"].load_state_dict(mirai_weight.layer3_1.state_dict())
        self.backbone._modules["7"]._modules["0"].load_state_dict(mirai_weight.layer4_0.state_dict())
        self.backbone._modules["7"]._modules["1"].load_state_dict(mirai_weight.layer4_1.state_dict())
    
    def forward(self, mg, density):
        # Extract image features from the backbone.
        mg_feature = self.backbone(mg)
        mg_feature_flatten = torch.flatten(mg_feature, 1)  # shape: [B, image_feature_dim]
        
        # Ensure density is of shape [B, 1].
        if density.dim() == 1:
            density = density.unsqueeze(1)
        
        # Compute density embedding: shape [B, 64].
        density_embedded = self.density_emb(density)
        
        # ----- First-Level Fusion -----
        # Concatenate image features and density embedding.
        early_fused = torch.cat([mg_feature_flatten, density_embedded], dim=1)  # shape: [B, image_feature_dim+64]
        early_fused = self.early_fusion(early_fused)  # shape: [B, 512]
        
        # ----- Second-Level Fusion -----
        # Reintroduce the density embedding and fuse again.
        second_fusion_input = torch.cat([early_fused, density_embedded], dim=1)  # shape: [B, 512+64]
        multi_fused_feature = self.fusion_fc(second_fusion_input)  # shape: [B, 256]
        
        # Single classification head.
        logits = self.classification_head(multi_fused_feature)
        
        return logits, multi_fused_feature

# Example usage:
if __name__ == "__main__":
    model = MGModule_SingleHead(num_bins=10)
    mg_input = torch.randn(8, 3, 224, 224)    # e.g., a batch of 8 images.
    density_input = torch.randn(8)             # one density per image.
    logits, fused_features = model(mg_input, density_input)
    print("Logits shape:", logits.shape)
    print("Fused features shape:", fused_features.shape)
