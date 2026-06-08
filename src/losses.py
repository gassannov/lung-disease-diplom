"""Loss functions for imbalanced classification."""

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced classification.

    Example:
        criterion = FocalLoss(gamma=2.0)
        loss = criterion(logits, labels)
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        """Initialize focal loss.

        Args:
            gamma: Focusing parameter.
            weight: Optional class weights.

        Returns:
            None.
        """
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Model logits.
            targets: Ground-truth class ids.

        Returns:
            Scalar loss tensor.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.weight)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

