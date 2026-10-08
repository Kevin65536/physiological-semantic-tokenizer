"""Versioned R-series shared-driver trajectory and raw-view registries.

The target sidecar and raw-view registry are deliberately separate artifacts.
The former supplies privileged training targets only; the latter supplies a
measurement-derived, frozen channel/anchor choice.  This prevents target
presence or target metadata from changing the tensors seen by a tokenizer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SHARED_DRIVER_SIDECAR_SCHEMA = "shared_driver_trajectory_sidecar_v1"
SHARED_DRIVER_ARRAY_SCHEMA = "shared_driver_trajectory_arrays_v1"
RAW_VIEW_REGISTRY_SCHEMA = "physiology_raw_view_registry_v1"
RAW_VIEW_ARRAY_SCHEMA = "physiology_raw_view_arrays_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(root: Path, expected_schema: str) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != expected_schema:
        raise ValueError(
            f"Unsupported schema at {path}: {manifest.get('schema')!r}; "
            f"expected {expected_schema!r}"
        )
    if bool(manifest.get("protected_test_included", False)):
        raise ValueError("Development artifacts must not contain protected-test samples")
    return manifest


def _load_arrays(root: Path, manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = root / str(manifest.get("arrays_file", "arrays.npz"))
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_sha = str(manifest.get("arrays_sha256", ""))
    if expected_sha and _sha256(path) != expected_sha:
        raise RuntimeError(f"Array hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _validate_index(
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
) -> tuple[list[str], dict[str, int]]:
    if "sample_key" not in arrays:
        raise KeyError("Artifact arrays require sample_key")
    keys = [str(value) for value in np.asarray(arrays["sample_key"]).tolist()]
    if len(keys) != len(set(keys)):
        raise ValueError("sample_key values must be unique")
    expected_count = int(manifest.get("sample_count", -1))
    if expected_count != len(keys):
        raise ValueError(
            f"Sample count mismatch: manifest={expected_count}, arrays={len(keys)}"
        )
    observed_order_sha = hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()
    if observed_order_sha != manifest.get("sample_order_sha256"):
        raise RuntimeError("Sample-order hash mismatch")
    for name, values in arrays.items():
        if name == "schema":
            continue
        if int(np.asarray(values).shape[0]) != len(keys):
            raise ValueError(f"Field {name!r} has inconsistent sample count")
    return keys, {key: index for index, key in enumerate(keys)}


class SharedDriverTrajectorySidecar:
    """Immutable full-trajectory R1-D/R1-P targets keyed by measured sample."""

    _required_arrays = {
        "schema",
        "sample_key",
        "target_shared_driver",
        "target_point_valid_mask",
        "target_eeg_only_driver",
        "eeg_only_point_valid_mask",
        "teacher_scope",
        "teacher_parameter_fold",
        "teacher_gauge_hash",
        "teacher_source_hash",
    }

    def __init__(
        self,
        root: str | Path,
        *,
        expected_scope: str | None = None,
        expected_family: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.manifest = _load_manifest(self.root, SHARED_DRIVER_SIDECAR_SCHEMA)
        self.target_family = str(self.manifest.get("target_family", ""))
        self.teacher_scope = str(self.manifest.get("teacher_scope", ""))
        if expected_scope is not None and self.teacher_scope != str(expected_scope):
            raise ValueError(
                f"Teacher scope mismatch: {self.teacher_scope!r} != {expected_scope!r}"
            )
        if expected_family is not None and self.target_family != str(expected_family):
            raise ValueError(
                f"Target family mismatch: {self.target_family!r} != {expected_family!r}"
            )
        self.arrays = _load_arrays(self.root, self.manifest)
        missing = self._required_arrays.difference(self.arrays)
        if missing:
            raise KeyError(f"Shared-driver arrays missing fields: {sorted(missing)}")
        schema = str(np.asarray(self.arrays["schema"]).item())
        if schema != SHARED_DRIVER_ARRAY_SCHEMA:
            raise ValueError(f"Shared-driver array schema mismatch: {schema!r}")
        self.sample_keys, self._index = _validate_index(self.arrays, self.manifest)

        joint = np.asarray(self.arrays["target_shared_driver"])
        joint_mask = np.asarray(self.arrays["target_point_valid_mask"])
        eeg_only = np.asarray(self.arrays["target_eeg_only_driver"])
        eeg_only_mask = np.asarray(self.arrays["eeg_only_point_valid_mask"])
        if joint.shape[1:] != (10, 20) or eeg_only.shape != joint.shape:
            raise ValueError("Shared-driver targets must both have shape [N,10,20]")
        if joint_mask.shape != joint.shape or eeg_only_mask.shape != joint.shape:
            raise ValueError("Point-valid masks must match [N,10,20] targets")
        if not np.isfinite(joint[joint_mask.astype(bool)]).all():
            raise ValueError("Joint target contains non-finite supported points")
        if not np.isfinite(eeg_only[eeg_only_mask.astype(bool)]).all():
            raise ValueError("EEG-only target contains non-finite supported points")
        row_scopes = {str(value) for value in self.arrays["teacher_scope"].tolist()}
        if row_scopes != {self.teacher_scope}:
            raise ValueError("Per-row teacher_scope differs from manifest scope")

    def __len__(self) -> int:
        return len(self.sample_keys)

    def contains(self, sample_key: str) -> bool:
        return str(sample_key) in self._index

    def lookup(self, sample_key: str) -> dict[str, Any] | None:
        index = self._index.get(str(sample_key))
        if index is None:
            return None
        joint = torch.from_numpy(
            np.asarray(self.arrays["target_shared_driver"][index], dtype=np.float32).copy()
        )
        joint_points = torch.from_numpy(
            np.asarray(self.arrays["target_point_valid_mask"][index], dtype=bool).copy()
        )
        eeg_only = torch.from_numpy(
            np.asarray(self.arrays["target_eeg_only_driver"][index], dtype=np.float32).copy()
        )
        eeg_only_points = torch.from_numpy(
            np.asarray(self.arrays["eeg_only_point_valid_mask"][index], dtype=bool).copy()
        )
        return {
            "schema": SHARED_DRIVER_ARRAY_SCHEMA,
            "sample_key": str(sample_key),
            "target_shared_driver": joint,
            "target_point_valid_mask": joint_points,
            "target_eeg_only_driver": eeg_only,
            "eeg_only_point_valid_mask": eeg_only_points,
            "teacher_mask": joint_points.any(dim=-1),
            "eeg_only_teacher_mask": eeg_only_points.any(dim=-1),
            "joint_correction": joint - eeg_only,
            "joint_correction_point_valid_mask": joint_points & eeg_only_points,
            "teacher_scope": str(self.arrays["teacher_scope"][index]),
            "teacher_parameter_fold": str(
                self.arrays["teacher_parameter_fold"][index]
            ),
            "teacher_gauge_hash": str(self.arrays["teacher_gauge_hash"][index]),
            "teacher_source_hash": str(self.arrays["teacher_source_hash"][index]),
        }

    @staticmethod
    def point_loss_mask(
        token_valid_mask: torch.Tensor,
        teacher_mask: torch.Tensor,
        target_point_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return measurement ∩ teacher ∩ point-support without changing denominator."""

        token_valid_mask = token_valid_mask.to(dtype=torch.bool)
        teacher_mask = teacher_mask.to(dtype=torch.bool)
        target_point_valid_mask = target_point_valid_mask.to(dtype=torch.bool)
        if token_valid_mask.shape != teacher_mask.shape:
            raise ValueError("token_valid_mask and teacher_mask must share [B,N]")
        if target_point_valid_mask.shape[:2] != token_valid_mask.shape:
            raise ValueError("target_point_valid_mask must have shape [B,N,U]")
        return (
            token_valid_mask.unsqueeze(-1)
            & teacher_mask.unsqueeze(-1)
            & target_point_valid_mask
        )


class PhysiologyRawViewRegistry:
    """Frozen measured channel choices, stored independently of teacher targets."""

    _required_arrays = {
        "schema",
        "sample_key",
        "selected_eeg_channels",
        "selected_fnirs_channels",
        "anchor_id",
        "selection_fold",
        "selection_source_hash",
    }

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.manifest = _load_manifest(self.root, RAW_VIEW_REGISTRY_SCHEMA)
        self.arrays = _load_arrays(self.root, self.manifest)
        missing = self._required_arrays.difference(self.arrays)
        if missing:
            raise KeyError(f"Raw-view arrays missing fields: {sorted(missing)}")
        schema = str(np.asarray(self.arrays["schema"]).item())
        if schema != RAW_VIEW_ARRAY_SCHEMA:
            raise ValueError(f"Raw-view array schema mismatch: {schema!r}")
        self.sample_keys, self._index = _validate_index(self.arrays, self.manifest)
        eeg = np.asarray(self.arrays["selected_eeg_channels"])
        fnirs = np.asarray(self.arrays["selected_fnirs_channels"])
        if eeg.shape[1:] != (6,) or fnirs.shape[1:] != (2,):
            raise ValueError("Raw-view registry requires six EEG and two fNIRS channels")
        if any(len(set(map(str, row))) != 6 for row in eeg.tolist()):
            raise ValueError("Each raw-view EEG selection must contain six unique channels")
        if any(len(set(map(str, row))) != 2 for row in fnirs.tolist()):
            raise ValueError("Each raw-view fNIRS selection must contain two unique channels")

    def __len__(self) -> int:
        return len(self.sample_keys)

    def contains(self, sample_key: str) -> bool:
        return str(sample_key) in self._index

    def lookup(self, sample_key: str) -> dict[str, Any] | None:
        index = self._index.get(str(sample_key))
        if index is None:
            return None
        return {
            "schema": RAW_VIEW_ARRAY_SCHEMA,
            "sample_key": str(sample_key),
            "selected_eeg_channels": tuple(
                str(value)
                for value in np.asarray(
                    self.arrays["selected_eeg_channels"][index]
                ).tolist()
            ),
            "selected_fnirs_channels": tuple(
                str(value)
                for value in np.asarray(
                    self.arrays["selected_fnirs_channels"][index]
                ).tolist()
            ),
            "anchor_id": str(self.arrays["anchor_id"][index]),
            "selection_fold": str(self.arrays["selection_fold"][index]),
            "selection_source_hash": str(
                self.arrays["selection_source_hash"][index]
            ),
        }


def string_array(values: Sequence[str]) -> np.ndarray:
    """Create a non-object NumPy string array suitable for allow_pickle=False."""

    return np.asarray([str(value) for value in values], dtype=np.str_)


__all__ = [
    "PhysiologyRawViewRegistry",
    "RAW_VIEW_ARRAY_SCHEMA",
    "RAW_VIEW_REGISTRY_SCHEMA",
    "SHARED_DRIVER_ARRAY_SCHEMA",
    "SHARED_DRIVER_SIDECAR_SCHEMA",
    "SharedDriverTrajectorySidecar",
    "string_array",
]
