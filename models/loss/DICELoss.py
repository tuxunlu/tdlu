# ./loss/DICELoss.py
import torch
import torch.nn as nn

class DICELoss(nn.Module):
    """
    Binary Dice loss that accepts logits and boolean/0-1 targets.
    Computes 1 - Dice over the last three dims (C,H,W) and then reduces over batch.
    """
    def __init__(self, eps: float = 1e-6, reduction: str = "mean", from_logits: bool = True):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.from_logits = from_logits

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # targets may be bool; cast to float in {0,1}
        targets = targets.to(dtype=inputs.dtype)

        # logits -> probabilities if requested
        if self.from_logits:
            inputs = torch.sigmoid(inputs)

        # sum over (C,H,W); keep batch/view dims if present
        # supports shapes like [B,V,1,H,W] or [B,1,H,W]
        dims = tuple(range(inputs.ndim - 3, inputs.ndim))
        intersection = torch.sum(inputs * targets, dim=dims)
        union = torch.sum(inputs, dim=dims) + torch.sum(targets, dim=dims)

        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        loss = 1.0 - dice

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # "none"
