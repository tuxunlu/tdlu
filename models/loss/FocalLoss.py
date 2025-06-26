# loss/focal_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al. 2017):
      FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
      gamma: focusing parameter γ ≥ 0
      alpha: None (no class‐weighting), 
             scalar float (same weight for all classes), 
             or list/tuple/tensor of shape [num_classes].
      reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self,
                 gamma: float = 2.0,
                 alpha: Optional[Union[float, List[float], torch.Tensor]] = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        # process alpha
        if alpha is None:
            self.register_buffer('alpha', None)
        else:
            # convert list/tuple to tensor
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float)
            elif isinstance(alpha, float) or isinstance(alpha, int):
                alpha = torch.tensor([alpha,] if alpha.__class__ is float else [alpha,], dtype=torch.float)
            elif not isinstance(alpha, torch.Tensor):
                raise ValueError(f"Unsupported alpha type {type(alpha)}")
            # register as buffer so it moves with .to(device)
            self.register_buffer('alpha', alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [N, C] where C = num_classes
        targets: [N] with values in {0, …, C-1}
        """
        # 1) standard per‐sample CE
        ce = F.cross_entropy(logits, targets, reduction='none')  # [N]
        pt = torch.exp(-ce)                                     # [N]

        # 2) focal term
        focal_term = (1 - pt) ** self.gamma                     # [N]
        loss = focal_term * ce                                  # [N]

        # 3) optional alpha per‐class weighting
        if self.alpha is not None:
            # make sure alpha is on same device/dtype
            alpha = self.alpha.to(logits.device).type_as(logits)    # [C] or [1]
            # pick the weight for each sample’s true class
            at = alpha[targets]                                     # [N]
            loss = at * loss                                        # [N]

        # 4) reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
