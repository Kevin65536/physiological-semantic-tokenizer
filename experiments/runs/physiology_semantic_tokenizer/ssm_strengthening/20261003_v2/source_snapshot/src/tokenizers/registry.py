"""Registry for the active physiology-semantic tokenizer."""

from typing import Any

import torch.nn as nn


_TOKENIZER_REGISTRY: dict[str, type[nn.Module]] = {}


def register_tokenizer(name: str):
    """Register a tokenizer class under one configuration name."""

    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        _TOKENIZER_REGISTRY[name] = cls
        return cls

    return decorator


def get_tokenizer_class(name: str) -> type[nn.Module]:
    """Return one registered tokenizer class."""
    try:
        return _TOKENIZER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown tokenizer: {name}. Available: {list_tokenizers()}"
        ) from exc


def create_tokenizer(config: dict[str, Any]) -> nn.Module:
    """Create the configured active tokenizer through its owned constructor."""
    tokenizer_type = config.get("model", {}).get("type")
    if not tokenizer_type:
        raise ValueError("model.type is required")
    cls = get_tokenizer_class(str(tokenizer_type))
    from_config = getattr(cls, "from_config", None)
    if not callable(from_config):
        raise TypeError(f"Tokenizer {tokenizer_type} does not implement from_config")
    return from_config(config)


def list_tokenizers() -> list[str]:
    """List registered tokenizer names in deterministic order."""
    return sorted(_TOKENIZER_REGISTRY)
