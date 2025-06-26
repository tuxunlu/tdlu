import torch
import torch.nn as nn
import torch.nn.functional as F

class OhemCELoss(nn.Module):
    """
    Online Hard Example Mining with CrossEntropy.
    Keeps only the top `keep_ratio` fraction of hardest samples per batch.
    """
    def __init__(self, keep_ratio: float = 0.7, ignore_index: int = -100):
        """
        Args:
          keep_ratio: fraction of samples to keep (e.g. 0.7 keeps the hardest 70%).
          ignore_index: target value to ignore in loss (passed to F.cross_entropy).
        """
        super().__init__()
        assert 0.0 < keep_ratio <= 1.0
        self.keep_ratio = keep_ratio
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
          logits: [B, C] raw, un‐normalized scores
          targets: [B] integer class labels in [0..C-1]
        Returns:
          scalar: mean loss over the hardest samples
        """
        # compute per‐sample CE loss (no reduction)
        losses = F.cross_entropy(logits, targets,
                                 reduction='none',
                                 ignore_index=self.ignore_index)

        # how many to keep
        B = losses.numel()
        keep_num = max(1, int(self.keep_ratio * B))

        # get the threshold loss to keep
        topk_losses, _ = torch.topk(losses, k=keep_num, sorted=False)
        thresh = topk_losses.min()   # smallest loss among the top‐k, so >= this are hard

        # mask only those losses >= thresh
        hard_mask = losses >= thresh
        hard_losses = losses[hard_mask]

        return hard_losses.mean()