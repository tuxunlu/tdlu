import torch
import torch.nn as nn
import torch.nn.init as init


class MetadataOnlyMLP(nn.Module):
    def __init__(
        self,
        num_meta_features: int,
        hidden_dims: tuple[int, ...] = (64, 128, 128),
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = num_meta_features
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            in_dim = h

        self.net = nn.Sequential(*layers)
        self.out_dim = in_dim

    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        return self.net(meta)


class MgmoduleMetadataOnly(nn.Module):
    """
    Metadata-only variant.

    This module intentionally ignores image inputs and predicts using metadata only.
    It keeps a compatible `forward(views, mask, meta)` signature so it can be swapped
    into existing training/eval pipelines without dataset refactors.
    """

    def __init__(
        self,
        num_bins: int,
        num_meta_features: int = 10,
        hidden_dims: tuple[int, ...] = (64, 128, 128),
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.meta_encoder = MetadataOnlyMLP(
            num_meta_features=num_meta_features,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
        )
        self.classifier = nn.Linear(self.meta_encoder.out_dim, num_bins)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                init.ones_(m.weight)
                init.zeros_(m.bias)

    def forward(
        self,
        views: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        meta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if meta is None:
            raise ValueError("meta must be provided for metadata-only model.")
        feats = self.meta_encoder(meta)
        return self.classifier(feats)
