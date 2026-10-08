"""Filesystem serialization helpers for experiment artifacts."""

import json
from pathlib import Path
from typing import Any, Union

import numpy as np
import yaml

PathLike = Union[str, Path]


def write_json(
    path: PathLike,
    payload: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = True,
) -> Path:
    """Write a JSON artifact and ensure its parent directory exists."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=indent, ensure_ascii=ensure_ascii),
        encoding='utf-8',
    )
    return output_path


def write_yaml(
    path: PathLike,
    payload: Any,
    *,
    default_flow_style: bool = False,
    allow_unicode: bool = True,
) -> Path:
    """Write a YAML artifact and ensure its parent directory exists."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as handle:
        yaml.dump(
            payload,
            handle,
            default_flow_style=default_flow_style,
            allow_unicode=allow_unicode,
            sort_keys=False,
        )
    return output_path


def save_npz(path: PathLike, **payload: Any) -> Path:
    """Write a compressed NumPy artifact and ensure its parent directory exists."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return output_path


__all__ = ['write_json', 'write_yaml', 'save_npz']