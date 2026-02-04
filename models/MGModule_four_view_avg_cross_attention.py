import torch
import torch.nn as nn
import torchvision
import torch.nn.init as init
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

class MgmoduleFourViewAvgCrossAttention(nn.Module):
    """
    Image-only version of cross-attention model.
    Processes four mammogram views with a shared ResNet18 backbone and
    fuses them using self-attention transformer. No metadata is used.
    Mask/meta inputs are ignored (kept for interface compatibility).
    """
    def __init__(
        self,
        num_bins: int,
        num_views: int = 4,
        transformer_embed_dim: int = 2048,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = True,
        pretrained_path: str = None,
        fusion_method: str = "self_attention",  # Options: "self_attention", "concat", "weighted_sum"
    ):
        super().__init__()
        self.num_views = num_views
        self.fusion_method = fusion_method
        self.freeze_backbone = freeze_backbone

        # Load ResNet18 backbone
        resnet = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Identity()  # Remove final classification layer
        self.backbone = resnet
        self.feature_dim = 512

        # View feature processing
        self.view_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # Learnable positional embedding to retain view identity (CC/MLO, L/R)
        self.view_pos_embed = nn.Parameter(torch.randn(1, num_views, self.feature_dim) * 0.02)
        
        # Different fusion strategies
        if fusion_method == "self_attention":
            # Self-attention transformer to fuse views
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.feature_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_embed_dim,
                dropout=dropout_rate,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
            
            # Learnable global token (CLS token) that aggregates information from all views
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.feature_dim) * 0.02)
            
            # Optional attention pooling for additional context
            self.attention_pool = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, self.feature_dim // 2),
                nn.Tanh(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.feature_dim // 2, 1)
            )
            self.fusion_proj = None  # Not needed for self-attention
            
        elif fusion_method == "concat":
            # Simple concatenation
            self.fusion_proj = nn.Linear(self.feature_dim * num_views, self.feature_dim)
            
        elif fusion_method == "weighted_sum":
            # Learnable weighted sum
            self.view_weights = nn.Parameter(torch.ones(num_views) / num_views)
            self.fusion_proj = nn.Linear(self.feature_dim, self.feature_dim)
        
        # Classification head
        self.classification_head = ClassificationHead(
            input_dim=self.feature_dim,
            num_classes=num_bins,
            dropout_rate=dropout_rate
        )
        
        # Apply weight initialization
        self._init_weights()
        
        # Freeze backbone if needed
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
        modules_to_init = [
            self.view_proj,
            self.classification_head,
        ]
        if self.fusion_method == "self_attention":
            modules_to_init.append(self.transformer)
            if self.attention_pool is not None:
                modules_to_init.append(self.attention_pool)
        elif self.fusion_proj is not None:
            modules_to_init.append(self.fusion_proj)
            
        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
        Args:
            views: [B, 4, 3, H, W] - four mammogram views
            mask: unused (kept for interface compatibility)
            meta: unused (kept for interface compatibility)
        Returns:
            logits: [B, num_bins]
        """
        B, V, C, H, W = views.shape
        assert V == self.num_views, f"Expected {self.num_views} views, got {V}"
        
        # Flatten batch and view dims
        views_flat = views.view(B * V, C, H, W)
        
        # Extract view features with shared backbone
        view_features = self.backbone(views_flat).flatten(1)  # [B*V, D]
        view_features = view_features.view(B, V, self.feature_dim)  # [B, V, D]
        
        # Add positional embeddings and project
        view_features = self.view_proj(view_features + self.view_pos_embed[:, :V])  # [B, V, D]
        
        # Fusion strategies
        if self.fusion_method == "self_attention":
            # Prepend CLS token
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
            tokens = torch.cat([cls_tokens, view_features], dim=1)  # [B, 1+V, D]
            
            # Apply transformer
            fused_tokens = self.transformer(tokens)  # [B, 1+V, D]
            
            # Extract CLS token as global summary
            cls_fused = fused_tokens[:, 0]  # [B, D]
            
            # Optional: gated attention over view-specific tokens for additional context
            view_tokens_only = fused_tokens[:, 1:]  # [B, V, D]
            attn_scores = self.attention_pool(view_tokens_only)  # [B, V, 1]
            attn_weights = torch.softmax(attn_scores, dim=1)
            attn_fused = (attn_weights * view_tokens_only).sum(dim=1)  # [B, D]
            
            # Combine CLS token and attention-weighted views
            fused = 0.5 * cls_fused + 0.5 * attn_fused  # [B, D]
            
        elif self.fusion_method == "concat":
            # Simple concatenation
            views_flat = view_features.view(B, -1)  # [B, V*D]
            fused = self.fusion_proj(views_flat)  # [B, D]
            
        elif self.fusion_method == "weighted_sum":
            # Learnable weighted combination
            view_weights_norm = torch.softmax(self.view_weights, dim=0)  # [V]
            fused = (view_features * view_weights_norm.view(1, -1, 1)).sum(dim=1)  # [B, D]
            fused = self.fusion_proj(fused)  # [B, D]
        
        # Classification
        logits = self.classification_head(fused)  # [B, num_bins]
        return logits
