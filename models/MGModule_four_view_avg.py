import torch
import torch.nn as nn
import torchvision
import torch.nn.init as init
from typing import Optional
from torchvision.models import ResNet18_Weights


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


class MgmoduleFourViewAvg(nn.Module):
    """
    Standard multi-view feature fusion architecture for mammography.
    Uses a shared ResNet18 backbone to extract features from each view,
    then fuses them using self-attention transformer without positional embeddings
    (since view order is always fixed). Follows standard academic design patterns.
    Mask/meta inputs are ignored (kept for interface compatibility).
    """

    def __init__(
        self,
        num_bins: int,
        num_views: int = 4,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        transformer_dim_feedforward: int = 2048,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = True,
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()
        self.num_views = num_views
        self.freeze_backbone = freeze_backbone

        # Shared ResNet18 feature extractor (ImageNet weights by default)
        resnet = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Identity()
        self.backbone = resnet
        self.feature_dim = 512

        # View feature projection (standard normalization + projection)
        self.view_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Self-attention transformer encoder for view-to-view interaction
        # No positional embeddings needed since view order is fixed
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_dim_feedforward,
            dropout=dropout_rate,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Multi-head attention pooling for robust feature aggregation
        # This is a standard approach in vision transformers for aggregating tokens
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=transformer_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        
        # Learnable query for attention pooling (standard design)
        self.pool_query = nn.Parameter(torch.randn(1, 1, self.feature_dim) * 0.02)

        # Final fusion projection
        self.fusion_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.classification_head = ClassificationHead(
            input_dim=self.feature_dim,
            num_classes=num_bins,
            dropout_rate=dropout_rate,
        )

        self._init_weights()
        if freeze_backbone:
            self._freeze_backbone()

        # Optional: load external checkpoint (best-effort)
        if pretrained_path:
            try:
                state = torch.load(pretrained_path, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                self.load_state_dict(state, strict=False)
            except Exception as exc:
                print(f"Warning: failed to load {pretrained_path}: {exc}")

    def _init_weights(self):
        modules = [self.view_proj, self.transformer, self.attention_pool, 
                   self.fusion_proj, self.classification_head]
        for module in modules:
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
            if name.startswith("layer4"):
                param.requires_grad = True

        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                module.eval()
                module.track_running_stats = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone is not None:
            self.backbone.train(mode)
            if self.freeze_backbone:
                for module in self.backbone.modules():
                    if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                        module.eval()
                        module.track_running_stats = False
        return self

    def forward(self, views: torch.Tensor, mask: torch.Tensor = None, meta: torch.Tensor = None):
        """
        Standard multi-view fusion forward pass.
        
        Args:
            views: [B, 4, 3, H, W] - four mammogram views in fixed order
            mask: unused (kept for interface compatibility)
            meta: unused (kept for interface compatibility)
        
        Returns:
            logits: [B, num_bins] - classification logits
        """
        B, V, C, H, W = views.shape
        assert V == self.num_views, f"Expected {self.num_views} views, got {V}"

        # Step 1: Extract features from each view using shared backbone
        view_features = self.backbone(views.view(B * V, C, H, W)).flatten(1)  # [B*V, D]
        view_features = view_features.view(B, V, self.feature_dim)  # [B, V, D]
        
        # Step 2: Project view features (normalization + linear projection)
        view_features = self.view_proj(view_features)  # [B, V, D]
        
        # Step 3: Self-attention transformer for view-to-view interaction
        # No positional embeddings needed since view order is always fixed
        attended_features = self.transformer(view_features)  # [B, V, D]
        
        # Step 4: Multi-head attention pooling to aggregate views
        # This is a standard approach: use a learnable query to attend over all views
        pool_query = self.pool_query.expand(B, -1, -1)  # [B, 1, D]
        fused_features, _ = self.attention_pool(
            query=pool_query,  # [B, 1, D]
            key=attended_features,  # [B, V, D]
            value=attended_features  # [B, V, D]
        )  # [B, 1, D]
        fused_features = fused_features.squeeze(1)  # [B, D]
        
        # Step 5: Final fusion projection
        fused_features = self.fusion_proj(fused_features)  # [B, D]
        
        # Step 6: Classification
        logits = self.classification_head(fused_features)  # [B, num_bins]
        return logits
