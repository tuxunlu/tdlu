import torch
import torch.nn as nn
import torchvision
import torch.nn.init as init
from torchvision.models import ResNet101_Weights

class UpBlock(nn.Module):
    """Conv → (optional BN) → ReLU → ConvTranspose2d(x2)"""
    def __init__(self, in_ch, out_ch, use_bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=2, stride=2)
        self.apply(init_weights)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.up(x)
        return x

class SegmentationHead(nn.Module):
    """
    Simple decoder from stride-32 features (ResNet layer4) back to HxW.
    For 1024x1024 input, layer4 is ~32x32; 5 up steps → 1024x1024.
    """
    def __init__(self, in_ch=2048, mid_ch=512, out_ch=1):
        super().__init__()
        self.block1 = UpBlock(in_ch,   mid_ch)      # 32→64
        self.block2 = UpBlock(mid_ch,  mid_ch//2)   # 64→128
        self.block3 = UpBlock(mid_ch//2, mid_ch//4) # 128→256
        self.block4 = UpBlock(mid_ch//4, mid_ch//8) # 256→512
        self.block5 = UpBlock(mid_ch//8, mid_ch//8) # 512→1024
        self.head   = nn.Conv2d(mid_ch//8, out_ch, kernel_size=1)
        self.apply(init_weights)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.head(x)       # logits; use BCEWithLogitsLoss
        return x

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

class MgmoduleFourViewMetaAvgCrossAttentionSaliency(nn.Module):
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
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.feature_dim = 2048

        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # returns [B*V, C, 1, 1]

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

        # Segmentation head
        self.seg_head = SegmentationHead(in_ch=self.feature_dim, mid_ch=512, out_ch=1)

        # Optionally freeze backbone
        if freeze_backbone:
            freeze_stages(self.backbone)

        self.global_meta.apply(init_weights)
        self.decoder.apply(init_weights)
        self.classification_head.apply(init_weights)

    def forward(self, views: torch.Tensor, mask: torch.Tensor, meta: torch.Tensor):
        """
        views: Tensor of shape [B, V, C, H, W]
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

        x = views.view(B * V, C, H, W)  # [B*V, 3, H, W]

        # Flatten batch and view dims to encode all views
        feat_maps = self.backbone(x)
        view_tokens = self.gap(feat_maps).flatten(1)              # [B*V, 2048]
        view_tokens = view_tokens.view(B, V, self.feature_dim)    # [B, V, 2048]

        # 3) global meta-token
        meta = meta[:, 1:4]
        meta_token = self.global_meta(meta).unsqueeze(1)           # [B, 1, D]

        # 4) transformer fusion
        fused_seq   = self.decoder(tgt=meta_token, memory=view_tokens)

        # Aggregate transformer outputs (e.g., mean pooling)
        fused = fused_seq.squeeze(1)             # [B, D]

        # Classification
        logits = self.classification_head(fused)  # [B, num_bins]
        seg = self.seg_head(feat_maps)                       # [B*V, 1, H, W]
        seg_logits = seg.view(B, V, 1, H, W)                 # [B, V, 1, H, W]
        return logits, seg_logits, fused

# Example usage:
if __name__ == "__main__":
    model = MgmoduleFourViewMetaAvgCrossAttentionSaliency(num_bins=2)
    # dummy 4-view batch + dummy meta
    dummy_views = torch.randn(8, 4, 3, 1024, 1024)
    dummy_mask = torch.randn(8, 4, 1, 1024, 1024)
    dummy_meta  = torch.randn(8, 3)  # e.g. [Breast density, age, BMI, ancestry]
    logits, seg_logits, fused = model(dummy_views, dummy_mask, dummy_meta)
    print("logits: ", logits.shape)   # (8, 2)
    print("saliency: ", seg_logits.shape)
    print("fused: ", fused.shape)    # (8, 2048)
