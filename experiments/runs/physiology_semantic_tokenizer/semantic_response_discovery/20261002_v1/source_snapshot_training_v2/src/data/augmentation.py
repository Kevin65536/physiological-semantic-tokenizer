"""
Data augmentation transforms for EEG/fNIRS signals.

These transforms are designed to improve cross-subject generalization
in motor imagery classification tasks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Any

from ..losses import LabelSmoothingCrossEntropy


class SignalAugmentor(nn.Module):
    """
    Augmentation module for EEG/fNIRS signals.
    
    Applies various augmentations to improve robustness and generalization.
    """
    
    def __init__(
        self,
        time_shift_max: int = 0,
        channel_dropout_prob: float = 0.0,
        gaussian_noise_std: float = 0.0,
        scaling_range: Tuple[float, float] = (1.0, 1.0),
        mixup_alpha: float = 0.0,
    ):
        """
        Args:
            time_shift_max: Maximum time shift in samples (bidirectional)
            channel_dropout_prob: Probability of zeroing out a channel
            gaussian_noise_std: Std of Gaussian noise to add
            scaling_range: Min and max amplitude scaling factors
            mixup_alpha: Mixup alpha parameter (0 = disabled)
        """
        super().__init__()
        
        self.time_shift_max = time_shift_max
        self.channel_dropout_prob = channel_dropout_prob
        self.gaussian_noise_std = gaussian_noise_std
        self.scaling_range = scaling_range
        self.mixup_alpha = mixup_alpha
    
    def forward(
        self, 
        x: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply augmentations.
        
        Args:
            x: Input tensor [B, C, T]
            labels: Optional labels tensor [B] for mixup
            
        Returns:
            Augmented tensor [B, C, T], optionally soft labels [B, num_classes]
        """
        if not self.training:
            return x, labels
        
        x = x.clone()
        soft_labels = None
        
        # 1. Time shift
        if self.time_shift_max > 0:
            x = self._time_shift(x)
        
        # 2. Channel dropout
        if self.channel_dropout_prob > 0:
            x = self._channel_dropout(x)
        
        # 3. Gaussian noise
        if self.gaussian_noise_std > 0:
            x = self._add_noise(x)
        
        # 4. Amplitude scaling
        if self.scaling_range != (1.0, 1.0):
            x = self._scale(x)
        
        # 5. Mixup (requires labels)
        if self.mixup_alpha > 0 and labels is not None:
            x, soft_labels = self._mixup(x, labels)
            return x, soft_labels
        
        return x, labels
    
    def _time_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Random time shift within limits."""
        B, C, T = x.shape
        shifts = torch.randint(-self.time_shift_max, self.time_shift_max + 1, (B,))
        
        shifted = []
        for i in range(B):
            s = shifts[i].item()
            if s > 0:
                # Shift right - pad left with edge value
                shifted_i = torch.cat([x[i, :, :1].expand(-1, s), x[i, :, :-s]], dim=1)
            elif s < 0:
                # Shift left - pad right with edge value
                shifted_i = torch.cat([x[i, :, -s:], x[i, :, -1:].expand(-1, -s)], dim=1)
            else:
                shifted_i = x[i]
            shifted.append(shifted_i)
        
        return torch.stack(shifted, dim=0)
    
    def _channel_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly zero out channels."""
        B, C, T = x.shape
        mask = torch.rand(B, C, 1, device=x.device) > self.channel_dropout_prob
        return x * mask.float()
    
    def _add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(x) * self.gaussian_noise_std
        return x + noise
    
    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        """Random amplitude scaling per sample."""
        B = x.shape[0]
        scales = torch.empty(B, 1, 1, device=x.device).uniform_(*self.scaling_range)
        return x * scales
    
    def _mixup(
        self, 
        x: torch.Tensor, 
        labels: torch.Tensor,
        num_classes: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup augmentation."""
        B = x.shape[0]
        
        # Sample mixup coefficient
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Random permutation for mixing
        indices = torch.randperm(B, device=x.device)
        
        # Mix features
        x_mixed = lam * x + (1 - lam) * x[indices]
        
        # Create soft labels
        labels_one_hot = torch.zeros(B, num_classes, device=x.device)
        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
        
        labels_shuffled = torch.zeros(B, num_classes, device=x.device)
        labels_shuffled.scatter_(1, labels[indices].unsqueeze(1), 1)
        
        soft_labels = lam * labels_one_hot + (1 - lam) * labels_shuffled
        
        return x_mixed, soft_labels


class DualModalityAugmentor(nn.Module):
    """Augmentor for dual-modality data."""
    
    def __init__(
        self,
        eeg_augmentor: Optional[SignalAugmentor] = None,
        fnirs_augmentor: Optional[SignalAugmentor] = None,
        cross_modal_mixup: bool = False,
    ):
        super().__init__()
        self.eeg_augmentor = eeg_augmentor or SignalAugmentor()
        self.fnirs_augmentor = fnirs_augmentor or SignalAugmentor()
        self.cross_modal_mixup = cross_modal_mixup
    
    def forward(
        self,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Apply augmentations to both modalities."""
        
        eeg_aug, soft_labels = self.eeg_augmentor(eeg, labels)
        fnirs_aug, _ = self.fnirs_augmentor(fnirs, labels)  # Use same labels
        
        return eeg_aug, fnirs_aug, soft_labels if soft_labels is not None else labels


def create_augmentor_from_config(config: Dict[str, Any]) -> Optional[SignalAugmentor]:
    """Create augmentor from config dict."""
    aug_cfg = config.get('augmentation', {})
    
    if not aug_cfg.get('enabled', False):
        return None
    
    return SignalAugmentor(
        time_shift_max=aug_cfg.get('time_shift', {}).get('max_shift', 0) if aug_cfg.get('time_shift', {}).get('enabled', False) else 0,
        channel_dropout_prob=aug_cfg.get('channel_dropout', {}).get('prob', 0.0) if aug_cfg.get('channel_dropout', {}).get('enabled', False) else 0.0,
        gaussian_noise_std=aug_cfg.get('gaussian_noise', {}).get('std', 0.0) if aug_cfg.get('gaussian_noise', {}).get('enabled', False) else 0.0,
        scaling_range=tuple(aug_cfg.get('scaling', {}).get('range', [1.0, 1.0])) if aug_cfg.get('scaling', {}).get('enabled', False) else (1.0, 1.0),
        mixup_alpha=aug_cfg.get('mixup', {}).get('alpha', 0.0) if aug_cfg.get('mixup', {}).get('enabled', False) else 0.0,
    )


if __name__ == '__main__':
    # Quick test
    print("Testing SignalAugmentor...")
    
    aug = SignalAugmentor(
        time_shift_max=20,
        channel_dropout_prob=0.1,
        gaussian_noise_std=0.05,
        scaling_range=(0.9, 1.1),
        mixup_alpha=0.2,
    )
    aug.train()
    
    x = torch.randn(8, 30, 800)
    labels = torch.randint(0, 2, (8,))
    
    x_aug, soft_labels = aug(x, labels)
    
    print(f"Input shape: {x.shape}")
    print(f"Augmented shape: {x_aug.shape}")
    print(f"Soft labels shape: {soft_labels.shape}")
    print(f"Soft labels sample: {soft_labels[0]}")
    
    print("\nTesting LabelSmoothingCrossEntropy...")
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    logits = torch.randn(8, 2)
    loss = criterion(logits, soft_labels)
    print(f"Loss with soft labels: {loss.item():.4f}")
    
    loss = criterion(logits, labels)
    print(f"Loss with hard labels: {loss.item():.4f}")
    
    print("\n✓ All tests passed!")
