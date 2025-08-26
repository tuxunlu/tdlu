import torch
import torch.nn as nn
import torchvision
import torch.nn.init as init
from torchvision.models import ResNet101_Weights

class ClassificationHead(nn.Module):
    """
    Classification head: projects transformer output to num_classes.
    """
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.LayerNorm(256)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return self.fc_out(x)

def freeze_stages(backbone, train_from="layer4"):
    # Your backbone is a Sequential of resnet children[:-1]:
    # 0:conv1, 1:bn1, 2:relu, 3:maxpool, 4:layer1, 5:layer2, 6:layer3, 7:layer4
    stage_idx = {"conv1":0,"bn1":1,"relu":2,"maxpool":3,"layer1":4,"layer2":5,"layer3":6,"layer4":7}
    start = stage_idx[train_from]
    for i, (name, module) in enumerate(backbone.named_children()):
        requires = (i >= start)
        for p in module.parameters():
            p.requires_grad = requires

def init_weights(m):
    """
    Initialize layer parameters properly
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
        init.ones_(m.weight)
        init.zeros_(m.bias)

class MgmoduleFourViewMaskMetaAvgCrossAttention(nn.Module):
    """
    Model that encodes 4 mammogram views with shared backbone, fuses via transformer,
    and outputs logits for classification.
    """
    def __init__(
        self,
        num_bins: int,
        num_views: int = 4,
        transformer_embed_dim: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        num_meta_features: int = 3,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = True,
        pretrained_path: str = None,
        model_weight_path: str = None,
        meta_only: bool = False
    ):
        super().__init__()
        # Load ResNet18 backbone without final FC
        resnet = torchvision.models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        # Inflate first conv from 3->4 in-ch using a standard trick:
        # copy RGB weights and set the extra channel to the mean of RGB weights.
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(
            in_channels=4, out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=False
        )
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
        resnet.conv1 = new_conv
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = resnet.fc.in_features  # typically 512

        # Use only metadata for classification
        self.meta_only = meta_only
        
        self.global_meta = nn.Sequential(
            nn.Linear(num_meta_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048)
        )


        # Transformer for fusion
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.feature_dim,
            nhead=transformer_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=transformer_layers)

        # Classification head after fusion
        self.classification_head = ClassificationHead(
            input_dim=self.feature_dim,
            num_classes=num_bins,
            dropout_rate=dropout_rate
        )

        # Optionally freeze backbone
        if freeze_backbone:
            freeze_stages(self.backbone)

        for name, param in self.backbone.named_parameters():
            print(name, param.requires_grad)

        self.global_meta.apply(init_weights)
        self.decoder.apply(init_weights)
        self.classification_head.apply(init_weights)

    def forward(self, views: torch.Tensor, mask: torch.Tensor, meta: torch.Tensor):
        """
        views: Tensor of shape [B, C, H, W], where C=3 views.
        meta:  Tensor [B, 3] → the tabular [age, BMI, ancestry]
        Returns:
          logits: [B, num_bins]
          fused_features: [B, transformer_embed_dim]
        """
        if self.meta_only:
            assert meta is not None, "Metadata must be provided for meta-only mode"
            meta_feats = self.global_meta(meta)              # [B, D]
            logits = self.classification_head(meta_feats)   # [B, num_bins]
            return logits, meta_feats

        B, V, C, H, W = views.shape
        assert C == 3, f"Expected 3 channels, got {C}"
        assert V == 4, f"Expected 4 views, got {V}"

        # Add mask as the 4th channel
        views = torch.cat([views, mask], dim=2)

        views = views.view(B * V, C + 1, H, W)  # [B*V, 4, H, W]

        # Flatten batch and view dims to encode all views
        view_tokens = self.backbone(views).flatten(1)           # [B*V, D]
        view_tokens = view_tokens.view(B, V, self.feature_dim)        # [B, V, D]

        # 3) global meta-token
        meta = meta[:, 0:3]
        meta_token = self.global_meta(meta).unsqueeze(1)           # [B, 1, D]

        # 4) transformer fusion
        fused_seq   = self.decoder(tgt=meta_token, memory=view_tokens)

        # Aggregate transformer outputs (e.g., mean pooling)
        fused = fused_seq.squeeze(1)             # [B, D]

        # Classification
        logits = self.classification_head(fused)  # [B, num_bins]
        return logits, fused

# Example usage:
if __name__ == "__main__":
    model = MgmoduleFourViewStackedMaskMetaAvgCrossAttention(num_bins=2)
    # dummy 4-view batch + dummy meta
    dummy_views = torch.randn(8, 4, 3, 1024, 1024)
    dummy_mask = torch.randn(8, 4, 1, 1024, 1024)
    dummy_meta  = torch.randn(8, 3)  # e.g. [Breast density, age, BMI, ancestry]
    logits, fused = model(dummy_views, dummy_mask, dummy_meta)
    print("logits:", logits.shape)   # (8, 2)
    print("fused: ", fused.shape)    # (8, 2048)
