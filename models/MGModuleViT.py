import torch
import torch.nn as nn
import torchvision
from torchvision.models import ViT_H_14_Weights

# The binary and multi-class heads remain the same.
class BinaryClassificationHead(nn.Module):
    """
    This head predicts whether the sample belongs to class 0 or non-zero.
    It outputs 2 logits.
    """
    def __init__(self, input_dim, dropout_rate=0.5):
        super(BinaryClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc_out(x)
        return out

class MultiClassificationHead(nn.Module):
    """
    This head predicts among the non-zero classes.
    Its output dimension is (num_bins - 1).
    """
    def __init__(self, input_dim, num_classes, dropout_rate=0.5):
        super(MultiClassificationHead, self).__init__()
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
        self.fc_out = nn.Linear(64, num_classes)
        
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

class MGModuleViT(nn.Module):
    """
    MGModuleViT uses a Vision Transformer backbone for feature extraction.
    The density is embedded and fused with the ViT feature (i.e., the [CLS] token embedding),
    and then passed to two classification heads similar to the original design.

    For ViT_h_14:
      - The backbone's feature dimension (hidden_dim) is 1280.
      - When concatenating with the 64-dimensional density embedding:
            1280 + 64 = 1344.
      - The early fusion layer uses an input dimension of 1344.
    """
    def __init__(self, num_bins, pretrained=True, pretrained_path=None):
        super(MGModuleViT, self).__init__()
        # Load a pre-trained ViT_h_14 backbone with weights.
        weights = ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1 if pretrained else None
        vit = torchvision.models.vit_h_14(weights=weights)
        # Remove the classification head so that the output is the [CLS] token embedding.
        vit.head = nn.Identity()
        self.backbone = vit
        # For ViT_h_14, the hidden dimension is 1280.
        self.image_feature_dim = vit.hidden_dim  # Expected to be 1280

        # Density Embedding: embed density (1D) into a 64-dimensional vector.
        self.density_emb = nn.Sequential(
            nn.Linear(1, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        
        # First-Level Fusion: fuse the ViT backbone's features and the density embedding.
        # Combined dimension: image_feature_dim (1280) + 64 = 1344.
        self.early_fusion = nn.Sequential(
            nn.Linear(1000 + 64, 512),  # 1344 -> 512
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Second-Level Fusion: reintroduce the density embedding to the early fused features.
        # Concatenated dimension: 512 + 64 = 576, then projected to 256.
        self.fusion_fc = nn.Sequential(
            nn.Linear(512 + 64, 256),  # 576 -> 256
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Classification heads for binary and multi-class tasks.
        self.binary_head = BinaryClassificationHead(input_dim=256, dropout_rate=0.5)
        self.multi_head = MultiClassificationHead(input_dim=256, num_classes=num_bins - 1, dropout_rate=0.5)
        
        if pretrained_path is not None:
            self.mg_load_pretrained_model(pretrained_path)
    
    def mg_load_pretrained_model(self, pretrained_path: str):
        # Optionally load pretrained weights for the backbone from a file.
        # Adapt the weight loading logic as necessary.
        state_dict = torch.load(pretrained_path, map_location="cpu")
        self.backbone.load_state_dict(state_dict)

    def forward(self, x, density):
        """
        Forward pass:
          - x: input images, shape [B, 3, 224, 224]
          - density: density values, shape [B] or [B, 1]
        """
        # Extract features from the ViT backbone.
        # The backbone returns the [CLS] token embedding of shape [B, image_feature_dim] (i.e., [B, 1280]).
        vit_features = self.backbone(x)

        # Ensure density is [B, 1].
        if density.dim() == 1:
            density = density.unsqueeze(1)
        
        # Compute density embedding: shape [B, 64].
        density_embedded = self.density_emb(density)
        
        # ----- First-Level Fusion -----
        # Concatenate ViT features (shape: [B, 1280]) and density embedding (shape: [B, 64]).
        # Resulting shape: [B, 1280 + 64] = [B, 1344].
        early_fused = torch.cat([vit_features, density_embedded], dim=1)
        early_fused = self.early_fusion(early_fused)  # Output shape: [B, 512]
        
        # ----- Second-Level Fusion -----
        # Reintroduce the density embedding (shape: [B, 64]) and fuse with early features.
        # Resulting shape: [B, 512 + 64] = [B, 576] => projected to [B, 256].
        second_fusion = torch.cat([early_fused, density_embedded], dim=1)
        multi_fused_feature = self.fusion_fc(second_fusion)  # Output shape: [B, 256]
        
        # Get logits from classification heads.
        binary_logits = self.binary_head(multi_fused_feature)
        multi_logits = self.multi_head(multi_fused_feature)
        
        return binary_logits, multi_logits, multi_fused_feature

# Example usage:
if __name__ == "__main__":
    # Create the model. The pretrained ViT backbone is loaded automatically.
    model = MGModuleViT(num_bins=10, pretrained=True)
    
    # Create dummy data: a batch of 8 images of shape 3 x 224 x 224
    # and a corresponding density value per image.
    images = torch.randn(8, 3, 224, 224)
    density = torch.randn(8)
    
    binary_logits, multi_logits, fused_features = model(images, density)
    print("Binary logits shape:", binary_logits.shape)
    print("Multi logits shape:", multi_logits.shape)
    print("Fused features shape:", fused_features.shape)
