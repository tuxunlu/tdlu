import torch
import torch.nn as nn
import torchvision
import torch.nn.init as init

class ClassificationHead(nn.Module):
    """
    Classification head: projects transformer output to num_classes.
    """
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc_out(x)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        init.ones_(m.weight)
        init.zeros_(m.bias)

class MgmoduleFourViewConcat(nn.Module):
    """
    Model that encodes 4 mammogram views with shared backbone, fuses via transformer,
    and outputs logits for classification.
    """
    def __init__(
        self,
        num_bins: int,
        num_views: int = 4,
        embed_dim: int = 512,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False,
        pretrained_path: str = None,
        model_weight_path: str = None
    ):
        super().__init__()
        # Load ResNet18 backbone without final FC
        resnet = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = resnet.fc.in_features  # typically 512

        # Optional pretrained loading
        if pretrained_path:
            self._load_pretrained_backbone(pretrained_path)
        
        if model_weight_path:
            self._load_model_weight_backbone(model_weight_path)

        if freeze_backbone:
            for name, p in self.backbone.named_parameters():
                if name.startswith(("4", "5", "6", "7")):
                    p.requires_grad = True 
                else:
                    p.requires_grad = False  # conv1, bn1, maxpool

        # Classification head after fusion
        self.classification_head = ClassificationHead(
            input_dim=self.feature_dim * num_views,
            num_classes=num_bins,
            dropout_rate=dropout_rate
        )

    def _load_model_weight_backbone(self, model_weight_path: str):
        ckpt = torch.load(model_weight_path)
        state_dict = ckpt.get("state_dict", ckpt)

        backbone_ckpt = {}
        prefix = "model.backbone."
        for full_key, tensor in state_dict.items():
            if full_key.startswith(prefix):
                # strip off the prefix so it matches your self.backbone keys
                new_key = full_key[len(prefix):]
                backbone_ckpt[new_key] = tensor
        
        load_info = self.backbone.load_state_dict(backbone_ckpt, strict=False)
        print(f"Backbone loaded with missing keys: {load_info.missing_keys}  unexpected keys: {load_info.unexpected_keys}")


    def _load_pretrained_backbone(self, mirai_path: str):
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

    def forward(self, views: torch.Tensor):
        """
        views: Tensor of shape [B, V, C, H, W], where V=4 views.
        Returns:
          logits: [B, num_bins]
          fused_features: [B, transformer_embed_dim]
        """
        B, V, C, H, W = views.shape
        assert V == 4, f"Expected 4 views, got {V}"

        # Flatten batch and view dims to encode all views
        x = views.view(B * V, C, H, W)
        feats = self.backbone(x)                    # [B*V, D, 1, 1]
        feats_flat = feats.view(B, -1)  # [B, V*D]

        # Classification
        logits = self.classification_head(feats_flat)  # [B, num_bins]
        return logits, feats_flat

# Example usage
def example():
    model = MgmoduleFourViewConcat(num_bins=2, freeze_backbone=False)
    # Batch of 4-view images: [B,4,3,224,224]
    dummy = torch.randn(8, 4, 3, 224, 224)
    logits, fused = model(dummy)
    print(logits.shape, fused.shape)

if __name__ == "__main__":
    example()
