"""Checkpoint helpers shared by training scripts."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

PathLike = Union[str, Path]


def load_checkpoint_file(checkpoint_path: PathLike, device: Any = 'cpu') -> Dict[str, Any]:
    """Load a checkpoint file with compatibility for different torch versions."""
    path = Path(checkpoint_path)
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def save_checkpoint_payload(checkpoint: Dict[str, Any], save_path: PathLike) -> Path:
    """Persist a prebuilt checkpoint payload to disk."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


def save_training_checkpoint(
    model,
    optimizer,
    epoch: int,
    save_path: PathLike,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save a standard training checkpoint to disk."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if extra:
        checkpoint.update(extra)

    return save_checkpoint_payload(checkpoint, path)


def load_training_checkpoint(
    checkpoint_path: PathLike,
    model,
    optimizer=None,
    device: Any = 'cpu',
    strict: bool = True,
) -> Dict[str, Any]:
    """Load model and optional optimizer state from a standard training checkpoint."""
    checkpoint = load_checkpoint_file(checkpoint_path, device=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


__all__ = [
    'load_checkpoint_file',
    'save_checkpoint_payload',
    'save_training_checkpoint',
    'load_training_checkpoint',
]