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


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        if hasattr(m, 'weight') and m.weight is not None:
            init.ones_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init.zeros_(m.bias)

class MgmoduleFourViewMetaAvgCrossAttentionAux(nn.Module):
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
        freeze_backbone: bool = False,
        pretrained_path: str = None,
        model_weight_path: str = None,
        meta_only: bool = False
    ):
        super().__init__()
        # Load ResNet18 backbone without final FC
        resnet = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = resnet.fc.in_features  # typically 512
        self.num_views = num_views

        self.meta_only = meta_only

        # Optional pretrained loading
        if pretrained_path:
            self._load_pretrained_backbone(pretrained_path)
        
        if model_weight_path:
            self._load_model_weight_backbone(model_weight_path)

        if freeze_backbone:
            for name, p in self.backbone.named_parameters():
                if name.startswith(("4", "5", "6", "7")):
                    p.requires_grad = False 
                else:
                    p.requires_grad = True  # conv1, bn1, maxpool

        self.global_meta = nn.Sequential(
            nn.Linear(num_meta_features, transformer_embed_dim),
            nn.LayerNorm(transformer_embed_dim),
            nn.GELU()
        )

        # Transformer for fusion
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=transformer_embed_dim,
            nhead=transformer_heads,
            dropout=dropout_rate,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=transformer_layers)

        # Classification head after fusion
        self.tdlu_density_head = ClassificationHead(
            input_dim=transformer_embed_dim,
            num_classes=num_bins,
            dropout_rate=dropout_rate
        )

        self.breast_density_head = ClassificationHead(
            input_dim=transformer_embed_dim,
            num_classes=num_bins,
            dropout_rate=dropout_rate
        )

        # Init
        self.global_meta.apply(init_weights)
        self.decoder.apply(init_weights)
        self.tdlu_density_head.apply(init_weights)
        self.breast_density_head.apply(init_weights)

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

    def forward(self, views: torch.Tensor, meta: torch.Tensor):
        """
        views: Tensor of shape [B, V, C, H, W], where V=4 views.
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
        
        # Discard Breast Density
        meta = meta[:, 0:3]

        B, V, C, H, W = views.shape
        assert V == self.num_views, f"Expected {self.num_views} views, got {V}"

        # Flatten batch and view dims to encode all views
        x = views.view(B * V, C, H, W)
        mamm_feats = self.backbone(x)                    # [B*V, D, 1, 1]
        mamm_feats = mamm_feats.view(B, V, self.feature_dim)  # [B, V, D]

        # 3) global meta-token
        meta_token = self.global_meta(meta)
        meta_token = meta_token.unsqueeze(0)                # [1,B,D]

        # 4) transformer fusion
        view_tokens = mamm_feats.permute(1,0,2)         # [V,B,D]
        seq         = torch.cat([view_tokens, meta_token], dim=0)
        fused_seq   = self.decoder(tgt=meta_token, memory=view_tokens)

        # Aggregate transformer outputs (e.g., mean pooling)
        fused = fused_seq.squeeze(0)             # [B, D]

        # Heads
        tdlu_logits = self.tdlu_density_head(fused)             # [B, num_bins]
        bd_logits = self.breast_density_head(fused)               # [B, num_bins]
        return tdlu_logits, bd_logits, fused

# Example usage:
if __name__ == "__main__":
    model = MgmoduleFourViewMeta(num_bins=2)
    # dummy 4-view batch + dummy meta
    dummy_views = torch.randn(8, 4, 3, 224, 224)
    dummy_density = torch.randn(8, 4)
    dummy_meta  = torch.randn(8, 3)  # e.g. [age, BMI, ancestry]
    logits, fused = model(dummy_views, dummy_density, dummy_meta)
    print("logits:", logits.shape)   # (8, 2)
    print("fused: ", fused.shape)    # (8, 512)
