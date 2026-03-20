import torch
import torch.nn as nn
import torchvision
import torch.nn.init as init
from torchvision.models import ResNet18_Weights


class ResNetSpatialBackbone(nn.Module):
    """ResNet backbone that returns spatial features [B, C, H, W] instead of flattening."""

    def __init__(self, resnet: nn.Module):
        super().__init__()
        self.resnet = resnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        return x  # [B, 512, H, W]


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.LayerNorm(512)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return self.fc_out(x)


class MgmoduleOneViewMetaAvgCrossAttention(nn.Module):
    """
    STAMP model variant for single-view mammograms with metadata fusion.
    For cross_attention: metadata (age, BMI, etc.) guides where the model looks in the
    mammogram. Spatial features [B, 512, 32, 32] are flattened to [B, 1024, 512];
    metadata [B, 1, 512] is the Query, image sequence is Key/Value.
    """

    def __init__(
        self,
        num_bins: int,
        transformer_heads: int = 8,
        num_meta_features: int = 10,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = True,
        unfreeze_backbone_epoch: int = None,
        fusion_method: str = "cross_attention",  # "cross_attention", "concat", "weighted_sum"
    ):
        super().__init__()
        self.fusion_method = fusion_method
        self.unfreeze_backbone_epoch = unfreeze_backbone_epoch

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
            new_conv.weight[:, 3:4, :, :] = 0  # zero-init: preserves pretrained output on step 1
        resnet.conv1 = new_conv
        resnet.avgpool = nn.Identity()  # keep spatial map [B, 512, 32, 32] for 1024x1024 input
        resnet.fc = nn.Identity()
        self.backbone = ResNetSpatialBackbone(resnet)  # custom forward stops before flatten
        self.feature_dim = 512

        # Meta encoder: output 512 to match backbone for cross-attention query
        self.meta_proj = nn.Sequential(
            nn.Linear(num_meta_features, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
        )

        if fusion_method == "cross_attention":
            # Metadata as Query, image spatial sequence as Key/Value
            self.cross_attn_meta_to_image = nn.MultiheadAttention(
                embed_dim=self.feature_dim,
                num_heads=transformer_heads,
                batch_first=True,
                dropout=dropout_rate,
            )
        elif fusion_method == "concat":
            self.fusion_proj = nn.Linear(self.feature_dim * 2, self.feature_dim)
        elif fusion_method == "weighted_sum":
            self.meta_weight = nn.Parameter(torch.tensor(1.0))
            self.fusion_proj = nn.Linear(self.feature_dim, self.feature_dim)
        else:
            raise ValueError(f"Unsupported fusion_method: {fusion_method}")

        self.classification_head = ClassificationHead(
            input_dim=self.feature_dim,
            num_classes=num_bins,
            dropout_rate=dropout_rate,
        )

        self._init_weights()
        self._backbone_unfrozen = False
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze all backbone params."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning. Idempotent."""
        if self._backbone_unfrozen:
            return
        self._backbone_unfrozen = True
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen for fine-tuning.")

    def _init_weights(self):
        modules_to_init = [self.meta_proj, self.classification_head]
        if self.fusion_method == "cross_attention":
            modules_to_init.append(self.cross_attn_meta_to_image)
        else:
            modules_to_init.append(self.fusion_proj)
        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    init.ones_(m.weight)
                    init.zeros_(m.bias)

    def forward(self, views: torch.Tensor, mask: torch.Tensor, meta: torch.Tensor):
        # Accept either [B, C, H, W] or [B, 1, C, H, W].
        if views.dim() == 4:
            views = views.unsqueeze(1)
        if views.dim() != 5:
            raise ValueError(f"Expected views to have 4 or 5 dims, got shape {tuple(views.shape)}")

        B, V, C, H, W = views.shape
        if V != 1:
            raise ValueError(f"Expected a single view (V=1) for STAMP, got V={V}")

        # Extract single view and dense mask; concatenate mask as 4th channel
        single_view = views[:, 0, :, :, :]  # [B, 3, H, W]
        # mask: [B, V, mask_ch, H, W] or [B, V, H, W]; take dense (first channel)
        if mask.dim() == 5:
            dense_mask = mask[:, 0, 0, :, :].unsqueeze(1).float()  # [B, 1, H, W]
        else:
            dense_mask = mask[:, 0, :, :].unsqueeze(1).float()  # [B, 1, H, W]
        # Resize mask if needed (views are 1024x1024)
        if dense_mask.shape[-2:] != single_view.shape[-2:]:
            dense_mask = torch.nn.functional.interpolate(
                dense_mask, size=single_view.shape[-2:], mode="nearest"
            )
        x = torch.cat([single_view, dense_mask], dim=1)  # [B, 4, H, W]
        spatial_features = self.backbone(x)  # [B, 512, 32, 32] (avgpool removed)

        # Process metadata -> [B, 1, 512]
        meta_features = self.meta_proj(meta).unsqueeze(1)

        if self.fusion_method == "cross_attention":
            # Flatten spatial map to sequence: [B, 1024, 512]
            B, D, h, w = spatial_features.shape
            image_seq = spatial_features.flatten(2).permute(0, 2, 1)  # [B, h*w, D]
            # Metadata as Query, image as Key/Value -> model attends to image based on metadata
            attended, _ = self.cross_attn_meta_to_image(
                query=meta_features,
                key=image_seq,
                value=image_seq,
            )
            fused = attended.squeeze(1)  # [B, 512]
        elif self.fusion_method == "concat":
            view_features = spatial_features.flatten(2).mean(dim=2)  # [B, 512] global avg pool
            fused = torch.cat([view_features, meta_features.squeeze(1)], dim=1)
            fused = self.fusion_proj(fused)
        else:  # weighted_sum
            view_features = spatial_features.flatten(2).mean(dim=2)  # [B, 512]
            fused = view_features + meta_features.squeeze(1) * self.meta_weight
            fused = self.fusion_proj(fused)

        logits = self.classification_head(fused)
        return logits
