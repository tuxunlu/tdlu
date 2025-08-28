import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# ---------------- utils ----------------
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.LayerNorm(256)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(256, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self.dropout(self.gelu(self.bn1(self.fc1(x)))))

# --------------- model -----------------
class MgmoduleFourViewMetaAvgCrossAttentionGap(nn.Module):
    """
    Choice B: encode 4 views separately (shared backbone), mask-weighted pooling per view,
    fuse {views + meta} with a TransformerEncoder, then classify.
    Inputs:
      views: [B, V, C, H, W]
      masks: [B, V, H, W]   (binary/float in {0,1})
      meta:  [B, M]
    """
    def __init__(self,
                 num_bins: int,
                 num_views: int = 4,
                 in_channels: int = 3,
                 transformer_embed_dim: int = 512,
                 transformer_heads: int = 8,
                 transformer_layers: int = 2,
                 num_meta_features: int = 3,
                 dropout_rate: float = 0.5,
                 freeze_backbone: bool = False,
                 pretrained_path: str = None,
                 model_weight_path: str = None,
                 meta_only: bool = False):
        super().__init__()
        self.num_views = num_views
        self.meta_only = meta_only

        # ----- ResNet18 backbone (keep SPATIAL map): conv1..layer4 -> [*, 512, h, w]
        resnet = torchvision.models.resnet18(weights=None)
        # adapt conv1 to the requested input channels
        old_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(in_channels, old_conv.out_channels,
                                 kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride, padding=old_conv.padding, bias=False)
        with torch.no_grad():
            if old_conv.weight.shape[1] == 3 and in_channels == 1:
                resnet.conv1.weight.copy_(old_conv.weight.mean(1, keepdim=True))
            elif old_conv.weight.shape[1] == 3 and in_channels > 1:
                w = old_conv.weight.mean(1, keepdim=True)
                resnet.conv1.weight.copy_(w.repeat(1, in_channels, 1, 1))
        # drop avgpool+fc to keep spatial features
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # -> [B*, 512, h, w]
        self.feat_dim = 512
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # per-view pooling: ROI + BG + Global -> project to token dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.view_proj = nn.Linear(self.feat_dim * 3, transformer_embed_dim)

        # meta token
        self.global_meta = nn.Sequential(
            nn.Linear(num_meta_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, transformer_embed_dim),
        )

        # transformer encoder for fusion (batch_first)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=transformer_embed_dim, nhead=transformer_heads,
            dropout=dropout_rate, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)

        # classifier on fused token
        self.classification_head = ClassificationHead(
            input_dim=transformer_embed_dim, num_classes=num_bins, dropout_rate=dropout_rate
        )

        # optional: load weights
        if pretrained_path: self._load_pretrained_backbone(pretrained_path)
        if model_weight_path: self._load_model_weight_backbone(model_weight_path)

        # init heads
        self.view_proj.apply(init_weights)
        self.global_meta.apply(init_weights)
        self.classification_head.apply(init_weights)

    # --- OPTIONAL checkpoint loaders (left as in your version) ---
    def _load_model_weight_backbone(self, model_weight_path: str):
        ckpt = torch.load(model_weight_path)
        state_dict = ckpt.get("state_dict", ckpt)
        backbone_ckpt, prefix = {}, "model.backbone."
        for k, v in state_dict.items():
            if k.startswith(prefix): backbone_ckpt[k[len(prefix):]] = v
        info = self.backbone.load_state_dict(backbone_ckpt, strict=False)
        print("Backbone loaded. missing:", info.missing_keys, "unexpected:", info.unexpected_keys)

    def _load_pretrained_backbone(self, mirai_path: str):
        # Your custom MIRAI loader (unchanged)
        pass

    # --- mask-aware pooling helper ---
    @torch.no_grad()
    def _downsample_mask(self, mask, size_hw):
        return F.interpolate(mask.float(), size=size_hw, mode='nearest')

    def forward(self, views: torch.Tensor, masks: torch.Tensor, meta: torch.Tensor):
        """
        views: [B,V,C,H,W], masks: [B,V,H,W], meta: [B,M]
        """
        if self.meta_only:
            meta_feats = self.global_meta(meta)
            logits = self.classification_head(meta_feats)
            return logits, meta_feats

        B, V, C, H, W = views.shape
        assert V == self.num_views, f"Expected {self.num_views} views, got {V}"

        # encode each view with shared backbone (vectorized as big batch)
        x = views.view(B * V, C, H, W)
        fmaps = self.backbone(x)                        # [B*V, 512, h, w]
        _, D, h, w = fmaps.shape

        # prepare masks
        m = masks.view(B * V, 1, H, W)
        m_d = self._downsample_mask(m, (h, w))         # [B*V,1,h,w]

        eps = 1e-6
        msum = m_d.sum((-2, -1)).clamp_min(eps)        # [B*V,1]
        roi  = (fmaps * m_d).sum((-2, -1)) / msum      # [B*V, D]
        bg_denom = (1 - m_d).sum((-2, -1)).clamp_min(eps)
        bg   = (fmaps * (1 - m_d)).sum((-2, -1)) / bg_denom  # [B*V, D]
        glob = self.gap(fmaps).flatten(1)              # [B*V, D]

        per_view_vec  = torch.cat([roi, bg, glob], dim=1)     # [B*V, 3D]
        per_view_tok  = self.view_proj(per_view_vec)           # [B*V, E]
        per_view_tok  = per_view_tok.view(B, V, -1)            # [B, V, E]

        # meta token (acts like CLS)
        meta = meta[:, 1:4]
        meta_tok = self.global_meta(meta).unsqueeze(1)         # [B, 1, E]

        # fuse and classify
        seq   = torch.cat([meta_tok, per_view_tok], dim=1)     # [B, V+1, E]
        fused = self.encoder(seq)[:, 0, :]                     # [B, E]
        logits = self.classification_head(fused)               # [B, num_bins]
        return logits, fused

# ---------------- example ----------------
if __name__ == "__main__":
    B, V, C, H, W = 8, 4, 3, 1024, 1024
    model = MgmoduleFourViewMetaAvgCrossAttention(num_bins=2, num_views=V, in_channels=C)
    views = torch.randn(B, V, C, H, W)
    masks = (torch.rand(B, V, H, W) > 0.5).float()
    meta  = torch.randn(B, 3)
    logits, fused = model(views, masks, meta)
    print("logits:", logits.shape, "fused:", fused.shape)  # [2,2], [2,512] if E=512
