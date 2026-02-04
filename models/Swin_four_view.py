import torch
import torch.nn as nn
import torchvision
import torch.nn.init as init
from transformers import SwinModel, AutoConfig, SwinForImageClassification

class ClassificationHead(nn.Module):
    """
    Classification head: projects transformer output to num_classes.
    """
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc_out(x)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        init.ones_(m.weight)
        init.zeros_(m.bias)

class SwinFourView(nn.Module):
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
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False,
        swin_model_name: str = "Koushim/breast-cancer-swin-classifier",
    ):
        super().__init__()
        # Load Swin backbone with hidden states (to get reshaped_hidden_states)
        config = AutoConfig.from_pretrained(
            swin_model_name,
            output_hidden_states=True,
            return_dict=True,
        )
        self.backbone = SwinForImageClassification.from_pretrained(
            swin_model_name,
        )

        self.swin_feature_dim = config.hidden_size

        # Transformer for fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.swin_feature_dim,
            nhead=transformer_heads,
            dropout=dropout_rate,
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )

        # Classification head after fusion
        self.classification_head = ClassificationHead(
            input_dim=self.swin_feature_dim,
            num_classes=num_bins,
            dropout_rate=dropout_rate
        )

        self.classification_head.apply(init_weights)

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

    def forward(self, views: torch.Tensor, meta: torch.Tensor):
        """
        views: Tensor of shape [B, V, C, H, W], where V=4 views.
        Returns:
          logits: [B, num_bins]
          fused_features: [B, transformer_embed_dim]
        """
        B, V, C, H, W = views.shape
        assert V == 4, f"Expected 4 views, got {V}"

        # Flatten batch and view dims to encode all views
        x = views.view(B * V, C, H, W)
        outputs = self.backbone(
            x,
            interpolate_pos_encoding=True,
            output_hidden_states=True
        )

        hidden_states = outputs.reshaped_hidden_states  # tuple/list
        if hidden_states is None:
            raise RuntimeError("hidden_states not returned; ensure output_hidden_states=True in config/forward.")

        last_hidden = hidden_states[-1]  # [B*V, H', W', C]
        view_feats = last_hidden.mean(dim=(2,3))

        view_embeds = view_feats.view(B, V, self.swin_feature_dim)               # → [B, V, C]
        seq        = view_embeds.permute(1, 0, 2)            # → [V, B, C]
        fused_seq  = self.transformer(seq)
        fused      = fused_seq.mean(dim=0)                   # → [B, C]
        logits     = self.classification_head(fused)       

        return logits, fused

# Example usage
def example():
    model = SwinFourView(num_bins=2, freeze_backbone=False)
    # Batch of 4-view images: [B,4,3,224,224]
    dummy = torch.randn(8, 4, 3, 224, 224)
    meta = torch.randn(8, 10)  # Example meta-data with 10 features
    logits, fused = model(dummy, meta)
    print(logits.shape, fused.shape)

if __name__ == "__main__":
    example()
