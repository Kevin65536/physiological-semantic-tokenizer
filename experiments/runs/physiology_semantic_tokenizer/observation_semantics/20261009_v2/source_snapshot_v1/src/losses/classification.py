"""Classification losses used by downstream training scripts."""

import torch
import torch.nn as nn


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with optional label smoothing and soft labels."""

    def __init__(self, smoothing: float = 0.0, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(1)

        if target.dim() == 2:
            log_pred = torch.log_softmax(pred, dim=-1)
            loss = -torch.sum(target * log_pred, dim=-1)
        else:
            log_pred = torch.log_softmax(pred, dim=-1)
            with torch.no_grad():
                smooth_labels = torch.zeros_like(pred).fill_(self.smoothing / n_classes)
                smooth_labels.scatter_(
                    1,
                    target.unsqueeze(1),
                    1.0 - self.smoothing + self.smoothing / n_classes,
                )
            loss = -torch.sum(smooth_labels * log_pred, dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


__all__ = ['LabelSmoothingCrossEntropy']