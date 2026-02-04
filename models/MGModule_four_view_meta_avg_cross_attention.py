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

class MgmoduleFourViewMetaAvgCrossAttention(nn.Module):
    """
    Improved version with multiple fusion strategies
    """
    def __init__(
        self,
        num_bins: int,
        num_views: int = 4,
        transformer_embed_dim: int = 2048,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        num_meta_features: int = 10,
        dropout_rate: float = 0.2,
        freeze_backbone: bool = True,
        pretrained_path: str = None,
        fusion_method: str = "cross_attention",  # Options: "cross_attention", "concat", "weighted_sum"
    ):
        super().__init__()
        self.num_views = num_views
        self.fusion_method = fusion_method

        # Load ResNet18 backbone
        resnet = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Identity()  # Remove final classification layer
        self.backbone = resnet
        self.feature_dim = 512

        # Meta feature processing
        self.meta_proj = nn.Sequential(
            nn.Linear(num_meta_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )
        
        # View feature processing
        self.view_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Different fusion strategies
        if fusion_method == "cross_attention":
            # Cross-attention: views attend to meta and vice versa
            self.cross_attn_meta_to_views = nn.MultiheadAttention(
                embed_dim=self.feature_dim, 
                num_heads=transformer_heads, 
                batch_first=True,
                dropout=dropout_rate
            )
            self.cross_attn_views_to_meta = nn.MultiheadAttention(
                embed_dim=self.feature_dim, 
                num_heads=transformer_heads, 
                batch_first=True,
                dropout=dropout_rate
            )
            self.fusion_proj = nn.Linear(self.feature_dim * 2, self.feature_dim)
            
        elif fusion_method == "concat":
            # Simple concatenation
            self.fusion_proj = nn.Linear(self.feature_dim * 2, self.feature_dim)
            
        elif fusion_method == "weighted_sum":
            # Learnable weighted sum
            self.view_weights = nn.Parameter(torch.ones(num_views) / num_views)
            self.meta_weight = nn.Parameter(torch.tensor(1.0))
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

        for n, p in self.backbone.named_parameters():
            print(n, p.requires_grad)

    
    def forward(self, views: torch.Tensor, mask: torch.Tensor, meta: torch.Tensor):
        B, V, C, H, W = views.shape
        
        # Flatten batch and view dims
        views_flat = views.view(B * V, C, H, W)
        
        # Extract view features
        view_features = self.backbone(views_flat).flatten(1)  # [B*V, D]
        view_features = view_features.view(B, V, self.feature_dim)  # [B, V, D]
        view_features = self.view_proj(view_features)  # [B, V, D]
        
        # Process meta features
        meta_features = self.meta_proj(meta).unsqueeze(1)  # [B, 1, D]
        
        # Fusion strategies
        if self.fusion_method == "cross_attention":
            # Meta attends to views
            meta_attended, _ = self.cross_attn_meta_to_views(
                query=meta_features,  # [B, 1, D]
                key=view_features,    # [B, V, D]
                value=view_features   # [B, V, D]
            )
            
            # Views attend to meta
            views_attended, _ = self.cross_attn_views_to_meta(
                query=view_features,  # [B, V, D]
                key=meta_features,    # [B, 1, D]
                value=meta_features   # [B, 1, D]
            )
            
            # Pool view features
            views_pooled = views_attended.mean(dim=1)  # [B, D]
            
            # Combine
            fused = torch.cat([meta_attended.squeeze(1), views_pooled], dim=1)  # [B, 2*D]
            fused = self.fusion_proj(fused)  # [B, D]
            
        elif self.fusion_method == "concat":
            # Simple concatenation
            views_pooled = view_features.mean(dim=1)  # [B, D]
            fused = torch.cat([views_pooled, meta_features.squeeze(1)], dim=1)  # [B, 2*D]
            fused = self.fusion_proj(fused)
            
        elif self.fusion_method == "weighted_sum":
            # Learnable weighted combination
            views_weighted = (view_features * self.view_weights.view(1, -1, 1)).sum(dim=1)  # [B, D]
            meta_weighted = meta_features.squeeze(1) * self.meta_weight
            fused = views_weighted + meta_weighted
            fused = self.fusion_proj(fused)
        
        # Classification
        logits = self.classification_head(fused)  # [B, num_bins]
        return logits