import torch
import torch.nn as nn
import torchvision
import torch.nn.init as init
from torchvision.models import ResNet18_Weights

class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.0):
        super().__init__()
        # Reduced hidden dim and set default dropout to 0.0 to prevent thrashing
        self.fc1 = nn.Linear(input_dim, 256) 
        self.bn1 = nn.LayerNorm(256)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc_out = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return self.fc_out(x)


class MgmoduleOneViewMetaConcat(nn.Module):
    """
    STAMP model variant for single-view mammograms with metadata fusion.
    Refactored to fix 4-channel conv1 freezing, remove sequence=1 cross-attention,
    and prevent metadata over-parameterization.
    """

    def __init__(
        self,
        num_bins: int,
        num_meta_features: int = 10,
        dropout_rate: float = 0.1,  # Lowered default global dropout
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # ResNet18 image encoder (4 channels: RGB + dense mask)
        resnet = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Inflate first conv from 3->4 to accept dense mask as 4th channel
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
        resnet.conv1 = new_conv
        resnet.fc = nn.Identity()
        self.backbone = resnet
        
        self.image_feature_dim = 512
        self.meta_feature_dim = 64  # Drastically reduced to prevent memorization

        # Leaner Meta encoder
        self.meta_proj = nn.Sequential(
            nn.Linear(num_meta_features, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, self.meta_feature_dim),
            nn.LayerNorm(self.meta_feature_dim),
        )

        # View feature post-projection
        self.view_proj = nn.Sequential(
            nn.LayerNorm(self.image_feature_dim),
            nn.Linear(self.image_feature_dim, self.image_feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Fusion Projection (maps concatenated features to classification head input)
        fused_dim = self.image_feature_dim + self.meta_feature_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )

        self.classification_head = ClassificationHead(
            input_dim=256,
            num_classes=num_bins,
            dropout_rate=0.0,  # Trust image augmentations over linear dropout here
        )

        self._init_weights()
        if freeze_backbone:
            self._freeze_backbone()

    def _init_weights(self):
        modules_to_init = [
            self.meta_proj,
            self.view_proj,
            self.fusion_proj,
            self.classification_head,
        ]
        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    init.ones_(m.weight)
                    init.zeros_(m.bias)

    def _freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
            # FIX: explicitly unfreeze the newly initialized conv1 as well as layer4
            if name.startswith("layer4") or name == "conv1.weight":
                param.requires_grad = True

    def forward(self, views: torch.Tensor, mask: torch.Tensor, meta: torch.Tensor):
        if views.dim() == 4:
            views = views.unsqueeze(1)
        if views.dim() != 5:
            raise ValueError(f"Expected views to have 4 or 5 dims, got shape {tuple(views.shape)}")

        B, V, C, H, W = views.shape
        if V != 1:
            raise ValueError(f"Expected a single view (V=1) for STAMP, got V={V}")

        # Extract single view and dense mask; concatenate mask as 4th channel
        single_view = views[:, 0, :, :, :]  # [B, 3, H, W]
        
        if mask.dim() == 5:
            dense_mask = mask[:, 0, 0, :, :].unsqueeze(1).float()  # [B, 1, H, W]
        else:
            dense_mask = mask[:, 0, :, :].unsqueeze(1).float()  # [B, 1, H, W]
            
        if dense_mask.shape[-2:] != single_view.shape[-2:]:
            dense_mask = torch.nn.functional.interpolate(
                dense_mask, size=single_view.shape[-2:], mode="nearest"
            )
            
        x = torch.cat([single_view, dense_mask], dim=1)  # [B, 4, H, W]
        
        # Image Features
        view_features = self.backbone(x).flatten(1)  # [B, 512]
        view_features = self.view_proj(view_features)  # [B, 512]

        # Meta Features
        meta_features = self.meta_proj(meta)  # [B, 64]

        # Concat Fusion
        fused = torch.cat([view_features, meta_features], dim=1)  # [B, 576]
        fused = self.fusion_proj(fused)  # [B, 256]

        logits = self.classification_head(fused)
        return logits