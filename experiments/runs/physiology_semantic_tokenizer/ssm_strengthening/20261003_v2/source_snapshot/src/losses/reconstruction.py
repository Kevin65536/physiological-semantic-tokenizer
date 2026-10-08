"""Reconstruction losses shared by tokenizer training pipelines."""

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def compute_stft_loss(
    x: torch.Tensor,
    x_rec: torch.Tensor,
    n_fft: int = 256,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
) -> torch.Tensor:
    """Compute STFT-based spectral loss."""
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft

    if x.dim() == 3:
        x = x.squeeze(1)
        x_rec = x_rec.squeeze(1)

    if x.shape[-1] < n_fft:
        pad_size = n_fft - x.shape[-1]
        x = F.pad(x, (0, pad_size))
        x_rec = F.pad(x_rec, (0, pad_size))

    window = torch.hann_window(win_length, device=x.device)

    stft_x = torch.stft(x, n_fft, hop_length, win_length, window, return_complex=True)
    stft_rec = torch.stft(x_rec, n_fft, hop_length, win_length, window, return_complex=True)

    mag_x = stft_x.abs()
    mag_rec = stft_rec.abs()

    mag_loss = F.l1_loss(mag_rec, mag_x)

    log_mag_x = torch.log(mag_x.clamp(min=1e-7))
    log_mag_rec = torch.log(mag_rec.clamp(min=1e-7))
    log_mag_loss = F.l1_loss(log_mag_rec, log_mag_x)

    return mag_loss + log_mag_loss


def compute_multi_stft_loss(
    x: torch.Tensor,
    x_rec: torch.Tensor,
    fft_sizes: Sequence[int] = (64, 128, 256, 512),
    hop_sizes: Optional[Sequence[int]] = None,
    win_sizes: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Compute multi-scale STFT loss."""
    if hop_sizes is None:
        hop_sizes = [n // 4 for n in fft_sizes]
    if win_sizes is None:
        win_sizes = fft_sizes

    total_loss = torch.tensor(0.0, device=x.device)

    for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_sizes):
        if x.shape[-1] < n_fft:
            continue
        total_loss = total_loss + compute_stft_loss(x, x_rec, n_fft, hop, win)

    return total_loss / len(fft_sizes)


def compute_band_power_loss(
    x: torch.Tensor,
    x_rec: torch.Tensor,
    fs: float = 200.0,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute band-power preservation losses for EEG reconstruction."""
    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45),
        }

    if x.dim() == 3:
        x = x.squeeze(1)
        x_rec = x_rec.squeeze(1)

    n_fft = x.shape[-1]
    freqs = torch.fft.rfftfreq(n_fft, d=1 / fs).to(x.device)

    fft_x = torch.fft.rfft(x, n=n_fft)
    fft_rec = torch.fft.rfft(x_rec, n=n_fft)

    losses = {}
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        if mask.sum() == 0:
            continue

        power_x = (fft_x.abs()[:, mask] ** 2).mean(dim=-1)
        power_rec = (fft_rec.abs()[:, mask] ** 2).mean(dim=-1)
        losses[f'{band_name}_power_loss'] = F.mse_loss(power_rec, power_x)

    return losses


def compute_smoothness_loss(x_rec: torch.Tensor) -> torch.Tensor:
    """Compute a smoothness penalty for fNIRS reconstruction."""
    diff1 = x_rec[..., 1:] - x_rec[..., :-1]
    diff2 = diff1[..., 1:] - diff1[..., :-1]
    return (diff2 ** 2).mean()


__all__ = [
    'compute_stft_loss',
    'compute_multi_stft_loss',
    'compute_band_power_loss',
    'compute_smoothness_loss',
]