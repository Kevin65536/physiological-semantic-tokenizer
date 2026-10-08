"""Versioned auxiliary-target sidecars for physiology-semantic tokenizers.

The sidecar is deliberately keyed by the measured unified-window identity.  It
may select the local measurement channels used to construct an admitted target,
but it never supplies EEG or fNIRS observations itself.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


TARGET_SIDECAR_SCHEMA = "physiology_semantic_target_sidecar_v1"
TARGET_ARRAY_SCHEMA = "physiology_semantic_patch_targets_v1"


def target_sample_key(
    dataset_id: str,
    subject: str,
    record_id: str,
    event_index: int,
) -> str:
    """Return the anchor-independent identity used for target joins."""

    return f"{dataset_id}|{subject}|{record_id}|event={int(event_index)}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PhysiologySemanticTargetSidecar:
    """Read a finite, immutable patch-target table into a sample-key index."""

    _target_fields = (
        "eeg_target",
        "eeg_uncertainty",
        "fnirs_target",
        "fnirs_uncertainty",
    )
    _mask_fields = (
        "eeg_local_valid_mask",
        "eeg_prototype_valid_mask",
        "eeg_context_valid_mask",
        "eeg_coupling_valid_mask",
        "fnirs_local_valid_mask",
        "fnirs_prototype_valid_mask",
        "fnirs_context_valid_mask",
        "fnirs_coupling_valid_mask",
    )

    def __init__(
        self,
        root: str | Path,
        *,
        expected_family: str | None = None,
        expected_version: str | None = None,
    ) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Auxiliary-target manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != TARGET_SIDECAR_SCHEMA:
            raise ValueError(
                f"Unsupported auxiliary-target schema: {self.manifest.get('schema')!r}"
            )
        self.target_family = str(self.manifest.get("target_family", ""))
        self.target_version = str(self.manifest.get("target_version", ""))
        if expected_family is not None and self.target_family != str(expected_family):
            raise ValueError(
                f"Auxiliary-target family mismatch: {self.target_family!r} != {expected_family!r}"
            )
        if expected_version is not None and self.target_version != str(expected_version):
            raise ValueError(
                f"Auxiliary-target version mismatch: {self.target_version!r} != {expected_version!r}"
            )
        if bool(self.manifest.get("protected_test_included", False)):
            raise ValueError("Development target sidecars must not contain protected-test samples")

        arrays_path = self.root / str(self.manifest.get("arrays_file", "targets.npz"))
        if not arrays_path.is_file():
            raise FileNotFoundError(f"Auxiliary-target arrays not found: {arrays_path}")
        expected_sha = self.manifest.get("arrays_sha256")
        if expected_sha and _sha256(arrays_path) != expected_sha:
            raise RuntimeError(f"Auxiliary-target array hash mismatch: {arrays_path}")
        with np.load(arrays_path, allow_pickle=False) as payload:
            self.arrays = {key: np.asarray(payload[key]) for key in payload.files}
        if str(np.asarray(self.arrays.get("schema", "")).item()) != TARGET_ARRAY_SCHEMA:
            raise ValueError("Auxiliary-target NPZ schema mismatch")

        required = {
            "sample_key",
            "selected_eeg_channels",
            "selected_fnirs_channels",
            *self._target_fields,
            *self._mask_fields,
        }
        missing = required.difference(self.arrays)
        if missing:
            raise KeyError(f"Auxiliary-target arrays missing fields: {sorted(missing)}")
        keys = [str(value) for value in np.asarray(self.arrays["sample_key"]).tolist()]
        if len(keys) != len(set(keys)):
            raise ValueError("Auxiliary-target sample keys must be unique")
        sample_count = int(self.manifest.get("sample_count", -1))
        if sample_count != len(keys):
            raise ValueError(
                f"Auxiliary-target sample count mismatch: manifest={sample_count}, arrays={len(keys)}"
            )
        for field in required - {"sample_key"}:
            if int(np.asarray(self.arrays[field]).shape[0]) != len(keys):
                raise ValueError(f"Auxiliary-target field {field!r} has inconsistent sample count")
        observed_order_sha = hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()
        if observed_order_sha != self.manifest.get("sample_order_sha256"):
            raise RuntimeError("Auxiliary-target sample-order hash mismatch")
        self._index = {key: index for index, key in enumerate(keys)}

        target_shapes = {
            "eeg": tuple(np.asarray(self.arrays["eeg_target"]).shape[1:]),
            "fnirs": tuple(np.asarray(self.arrays["fnirs_target"]).shape[1:]),
        }
        if target_shapes["eeg"] != (10, 6) or target_shapes["fnirs"] != (10, 9):
            raise ValueError(
                "The current tokenizer sidecar requires EEG [10,6] and fNIRS [10,9] targets"
            )

    def __len__(self) -> int:
        return len(self._index)

    def contains(self, sample_key: str) -> bool:
        return str(sample_key) in self._index

    def lookup(self, sample_key: str) -> dict[str, Any] | None:
        index = self._index.get(str(sample_key))
        if index is None:
            return None
        output: dict[str, Any] = {
            "schema": TARGET_ARRAY_SCHEMA,
            "target_family": self.target_family,
            "target_version": self.target_version,
            "sample_key": str(sample_key),
            "selected_eeg_channels": tuple(
                str(value) for value in np.asarray(self.arrays["selected_eeg_channels"][index]).tolist()
            ),
            "selected_fnirs_channels": tuple(
                str(value) for value in np.asarray(self.arrays["selected_fnirs_channels"][index]).tolist()
            ),
        }
        for field in self._target_fields:
            output[field] = torch.from_numpy(
                np.asarray(self.arrays[field][index], dtype=np.float32).copy()
            )
        for field in self._mask_fields:
            output[field] = torch.from_numpy(
                np.asarray(self.arrays[field][index], dtype=bool).copy()
            )
        return output

    @staticmethod
    def empty_target(*, tokens: int = 10) -> dict[str, torch.Tensor]:
        """Return a collatable target whose masks authorize no supervision."""

        output: dict[str, torch.Tensor] = {
            "eeg_target": torch.zeros(tokens, 6, dtype=torch.float32),
            "eeg_uncertainty": torch.ones(tokens, 6, dtype=torch.float32),
            "fnirs_target": torch.zeros(tokens, 9, dtype=torch.float32),
            "fnirs_uncertainty": torch.ones(tokens, 9, dtype=torch.float32),
        }
        for modality in ("eeg", "fnirs"):
            for entry in ("local", "prototype", "context", "coupling"):
                output[f"{modality}_{entry}_valid_mask"] = torch.zeros(
                    tokens, dtype=torch.bool
                )
        return output


def sidecar_manifest_sha256(sidecar: PhysiologySemanticTargetSidecar) -> str:
    return _sha256(sidecar.root / "manifest.json")


__all__ = [
    "PhysiologySemanticTargetSidecar",
    "TARGET_ARRAY_SCHEMA",
    "TARGET_SIDECAR_SCHEMA",
    "sidecar_manifest_sha256",
    "target_sample_key",
]
