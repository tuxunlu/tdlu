import torch
import torch.nn as nn
import torchvision
import torch.nn.init as init

class RegressionHead(nn.Module):
    """
    Simple MLP head for regression. Set out_dim>1 for multi-target regression.
    If you want to bound the range, pass range_min/max to enable a sigmoid+affine.
    """
    def __init__(self, input_dim: int, out_dim: int = 1, dropout_rate: float = 0.2,
                 range_min: float | None = None, range_max: float | None = None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(256, out_dim)
        self.range_min = range_min
        self.range_max = range_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc_out(x)  # [B, out_dim]
        if self.range_min is not None and self.range_max is not None:
            # bound into [range_min, range_max]
            x = torch.sigmoid(x) * (self.range_max - self.range_min) + self.range_min
        return x

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        if hasattr(m, "weight") and m.weight is not None:
            init.ones_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            init.zeros_(m.bias)

class MgmoduleFourViewMetaAvgCrossAttentionRegression(nn.Module):
    """
    4-view backbone -> project to transformer_embed_dim -> meta as query token -> decoder -> regression.
    """
    def __init__(
        self,
        out_dim: int = 1,                # number of regression targets
        num_views: int = 4,
        transformer_embed_dim: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        num_meta_features: int = 4,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False,
        pretrained_path: str | None = None,
        model_weight_path: str | None = None,
        meta_only: bool = False,
        range_min: float | None = None,  # optional bounding
        range_max: float | None = None
    ):
        super().__init__()
        self.meta_only = meta_only
        self.num_views = num_views

        # Backbone
        resnet = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # [B, 512, 1, 1]
        self.feature_dim = resnet.fc.in_features  # typically 512

        if pretrained_path:
            self._load_pretrained_backbone(pretrained_path)
        if model_weight_path:
            self._load_model_weight_backbone(model_weight_path)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Project backbone features -> transformer_embed_dim (so decoder dims match)
        self.proj = nn.Linear(self.feature_dim, transformer_embed_dim)

        # Meta encoder -> transformer_embed_dim (so query dims match)
        self.global_meta = nn.Sequential(
            nn.Linear(num_meta_features, transformer_embed_dim),
            nn.LayerNorm(transformer_embed_dim),
            nn.GELU()
        )

        # Transformer decoder (meta token attends to view tokens)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=transformer_embed_dim,
            nhead=transformer_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=transformer_layers)

        # Regression head
        self.regression_head = RegressionHead(
            input_dim=transformer_embed_dim,
            out_dim=out_dim,
            dropout_rate=dropout_rate,
            range_min=range_min,
            range_max=range_max
        )

        # Inits
        self.global_meta.apply(init_weights)
        self.decoder.apply(init_weights)
        self.regression_head.apply(init_weights)
        self.proj.apply(init_weights)

    def _load_model_weight_backbone(self, model_weight_path: str):
        ckpt = torch.load(model_weight_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        backbone_ckpt = {}
        prefix = "model.backbone."
        for full_key, tensor in state_dict.items():
            if full_key.startswith(prefix):
                new_key = full_key[len(prefix):]
                backbone_ckpt[new_key] = tensor
        load_info = self.backbone.load_state_dict(backbone_ckpt, strict=False)
        print(f"Backbone loaded with missing keys: {load_info.missing_keys}  unexpected keys: {load_info.unexpected_keys}")

    def _load_pretrained_backbone(self, mirai_path: str):
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

    def forward(self, views: torch.Tensor, meta: torch.Tensor):
        """
        views: [B, V, C, H, W]
        meta:  [B, num_meta_features]
        returns:
          preds: [B, out_dim] (regression)
          fused: [B, transformer_embed_dim]
        """
        B, V, C, H, W = views.shape
        assert V == self.num_views, f"Expected {self.num_views} views, got {V}"

        if self.meta_only:
            meta_feats = self.global_meta(meta)           # [B, D]
            preds = self.regression_head(meta_feats)      # [B, out_dim]
            return preds, meta_feats

        x = views.view(B * V, C, H, W)
        feats = self.backbone(x)                          # [B*V, D0, 1, 1]
        feats = feats.view(B, V, self.feature_dim)        # [B, V, D0]
        feats = self.proj(feats)                          # [B, V, D]

        query = self.global_meta(meta).unsqueeze(1)       # [B, 1, D]
        fused_seq = self.decoder(tgt=query, memory=feats) # [B, 1, D]
        fused = fused_seq.squeeze(1)                      # [B, D]

        preds = self.regression_head(fused)               # [B, out_dim]
        return preds, fused

# Example usage
if __name__ == "__main__":
    model = MgmoduleFourViewMetaAvgCrossAttentionRegression(out_dim=1, transformer_embed_dim=512, range_min=0, range_max=600)
    dummy_views = torch.randn(8, 4, 3, 224, 224)
    dummy_meta  = torch.randn(8, 4)  # e.g., [density, age, BMI, ancestry]
    preds, fused = model(dummy_views, dummy_meta)
    print("preds:", preds.shape)  # (8, 1)
    print("fused:", fused.shape)  # (8, 512)
    print(preds)
