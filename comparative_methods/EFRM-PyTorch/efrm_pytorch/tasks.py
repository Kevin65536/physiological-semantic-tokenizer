"""Seven-task downstream contract shared with the STA-Net comparison matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.clean_physiology_cache import DEFAULT_CLEAN_CACHE_ROOT
from src.data.unified_physiology import (
    REFEDContinuousSequenceDataset,
    UnifiedPhysiologyWindowDataset,
    canonical_label,
    refed_continuous_target_window,
)

from .data import EFRMPairedWindowAdapter


@dataclass(frozen=True)
class EFRMTaskSpec:
    key: str
    dataset_id: str
    namespace: str
    task_type: str
    class_names: tuple[str, ...] = ()
    target_names: tuple[str, ...] = ()
    target_length: int = 1
    input_duration_s: float = 8.0
    scientific_scope: str = "efrm_sync_classification"

    @property
    def output_dim(self) -> int:
        return len(self.class_names) if self.task_type == "classification" else len(self.target_names)


TASK_SPECS: dict[str, EFRMTaskSpec] = {
    "motor_imagery": EFRMTaskSpec(
        key="motor_imagery", dataset_id="eeg_fnirs_single_trial",
        namespace="eeg_fnirs_single_trial:motor_imagery", task_type="classification",
        class_names=("LMI", "RMI"),
    ),
    "mental_arithmetic": EFRMTaskSpec(
        key="mental_arithmetic", dataset_id="eeg_fnirs_single_trial",
        namespace="eeg_fnirs_single_trial:mental_arithmetic", task_type="classification",
        class_names=("MA", "BL"),
    ),
    "wg": EFRMTaskSpec(
        key="wg", dataset_id="simultaneous_eeg_nirs",
        namespace="simultaneous_eeg_nirs:wg", task_type="classification",
        class_names=("WG", "BL"),
    ),
    "nback": EFRMTaskSpec(
        key="nback", dataset_id="simultaneous_eeg_nirs",
        namespace="simultaneous_eeg_nirs:nback", task_type="classification",
        class_names=("0-back session", "2-back session", "3-back session"),
    ),
    "dsr": EFRMTaskSpec(
        key="dsr", dataset_id="simultaneous_eeg_nirs",
        namespace="simultaneous_eeg_nirs:dsr", task_type="classification",
        class_names=("Go", "No-go"), input_duration_s=2.0,
        scientific_scope="efrm_sync_dsr_context_adapter",
    ),
    "visual": EFRMTaskSpec(
        key="visual", dataset_id="visual_cognitive_motivation",
        namespace="visual_cognitive_motivation:visual_cognitive_motivation",
        task_type="classification", class_names=("RR", "RF", "FF", "FR"),
    ),
    "refed_regression": EFRMTaskSpec(
        key="refed_regression", dataset_id="refed", namespace="refed:emotion_video",
        task_type="regression", target_names=("valence", "arousal"), target_length=20,
        input_duration_s=20.0, scientific_scope="efrm_sync_regression_adapter",
    ),
}


def get_task_spec(key: str) -> EFRMTaskSpec:
    try:
        return TASK_SPECS[key]
    except KeyError as exc:
        raise KeyError(f"unknown EFRM task {key!r}; expected one of {sorted(TASK_SPECS)}") from exc


class EFRMUnifiedTaskDataset(Dataset):
    """Task-filtered paired windows with exactly the shared task ordering."""

    def __init__(
        self,
        spec: EFRMTaskSpec,
        cache_root: str = DEFAULT_CLEAN_CACHE_ROOT,
        *,
        require_full_analysis_support: bool = False,
    ) -> None:
        self.spec = spec
        self.adapter = EFRMPairedWindowAdapter(
            duration_s=spec.input_duration_s,
            require_full_analysis_support=require_full_analysis_support,
        )
        self.class_to_index = {name: index for index, name in enumerate(spec.class_names)}
        self.target_center: np.ndarray | None = None
        self.target_scale: np.ndarray | None = None
        if spec.task_type == "regression":
            self.base = REFEDContinuousSequenceDataset(cache_root=cache_root)
            self.indices = list(range(len(self.base)))
            self.excluded_label_counts: dict[str, int] = {}
        else:
            self.base = UnifiedPhysiologyWindowDataset(cache_root=cache_root, dataset_ids=(spec.dataset_id,))
            allowed = set(spec.class_names)
            self.indices = []
            excluded: dict[str, int] = {}
            for index, window in enumerate(self.base.windows):
                label = canonical_label(window.event, window.record.dataset_id)
                if label["namespace"] != spec.namespace:
                    continue
                condition = str(label["condition"])
                if condition not in allowed:
                    excluded[condition] = excluded.get(condition, 0) + 1
                    continue
                self.indices.append(index)
            self.excluded_label_counts = excluded
        if not self.indices:
            raise RuntimeError(f"no admitted samples found for {spec.namespace}")

    def __len__(self) -> int:
        return len(self.indices)

    def set_target_scaler(self, center: Sequence[float], scale: Sequence[float]) -> None:
        center_array = np.asarray(center, dtype=np.float32).reshape(-1, 1)
        scale_array = np.asarray(scale, dtype=np.float32).reshape(-1, 1)
        if center_array.shape != (self.spec.output_dim, 1) or scale_array.shape != center_array.shape:
            raise ValueError("target scaler dimensions do not match task output")
        if not np.isfinite(center_array).all() or not np.isfinite(scale_array).all() or (scale_array <= 0).any():
            raise ValueError("target scaler requires finite centers and positive scales")
        self.target_center, self.target_scale = center_array, scale_array

    def fit_target_scaler(self, indices: Sequence[int]) -> dict[str, Any]:
        if self.spec.task_type != "regression":
            raise RuntimeError("target scaling is defined only for regression")
        values: list[list[np.ndarray]] = [[] for _ in self.spec.target_names]
        for index in indices:
            window = self.base.windows[self.indices[int(index)]]
            extracted = refed_continuous_target_window(
                window.event,
                window_start_s=window.window_offset_s,
                window_duration_s=self.base.window_duration_s,
                target_sample_rate_hz=self.base.target_sample_rate_hz,
            )
            target = np.asarray(extracted["values"], dtype=np.float64)
            valid = np.asarray(extracted["valid_mask"], dtype=bool) & np.isfinite(target)
            for coordinate in range(len(values)):
                values[coordinate].append(target[coordinate, valid[coordinate]])
        flattened = [np.concatenate(parts) for parts in values]
        center = np.asarray([part.mean() for part in flattened], dtype=np.float32)
        scale = np.asarray([part.std() for part in flattened], dtype=np.float32)
        scale = np.where(scale > 1e-6, scale, 1.0)
        self.set_target_scaler(center, scale)
        return {
            "center": center.tolist(),
            "scale": scale.tolist(),
            "valid_value_count": [int(part.size) for part in flattened],
            "fit_scope": "training split only",
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        source = self.base[self.indices[int(index)]]
        adapted = self.adapter.adapt(source, crop_start_s=0.0)
        if self.spec.task_type == "classification":
            condition = str(source["label"]["condition"])
            adapted["target"] = torch.tensor(self.class_to_index[condition], dtype=torch.long)
            adapted["target_valid_mask"] = torch.tensor(True)
        else:
            target = np.asarray(source["target"], dtype=np.float32)
            valid = np.asarray(source["target_valid_mask"], dtype=bool)
            adapted["target_native"] = torch.from_numpy(target.copy())
            if self.target_center is not None and self.target_scale is not None:
                target = (target - self.target_center) / self.target_scale
            adapted["target"] = torch.from_numpy(target)
            adapted["target_valid_mask"] = torch.from_numpy(valid)
        return adapted

    def lightweight_metadata(self, index: int) -> dict[str, Any]:
        window = self.base.windows[self.indices[int(index)]]
        label = canonical_label(window.event, window.record.dataset_id)
        return {
            "dataset_index": int(index),
            "subject": str(window.record.canonical_subject_id),
            "record_id": str(window.record.base_record_id),
            "join_key": str(window.record.join_key),
            "condition": str(label["condition"]),
            "class_index": label.get("class_index"),
            "window_offset_s": float(window.window_offset_s),
            "event_index": int(window.event.get("event_index", -1)),
            "trial_group": f"{window.record.join_key}|event={int(window.event.get('event_index', -1))}",
        }

    def metadata_fingerprint(self) -> str:
        stable = [
            {
                "dataset_index": row["dataset_index"],
                "subject": row["subject"],
                "record_id": row["record_id"],
                "trial_group": row["trial_group"],
                "condition": row["condition"],
                "window_offset_s": row["window_offset_s"],
            }
            for row in (self.lightweight_metadata(index) for index in range(len(self)))
        ]
        payload = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def validate_shared_public_split(self, manifest_path: str | Path) -> tuple[list[int], list[int]]:
        path = Path(manifest_path).resolve()
        if "protected" in {part.lower() for part in path.parts}:
            raise PermissionError(f"refusing to read a protected split manifest: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        schema = str(manifest.get("schema"))
        if schema not in {
            "sta_net_split_registry_v2",
            "sta_net_subject_split_v1",
            "efrm_target_public_fold_v1",
        }:
            raise ValueError(f"unsupported public split schema: {schema}")
        if manifest.get("protected_test_opened", manifest.get("reserved_test_opened", False)):
            raise PermissionError("public split reports an opened protected test")
        if manifest.get("task") not in {None, self.spec.key}:
            raise ValueError("split task does not match EFRM task")
        forbidden = {"test_indices", "reserved_test_indices", "protected_indices"}.intersection(manifest)
        if forbidden:
            raise ValueError(f"public split exposes protected indices: {sorted(forbidden)}")
        if schema in {"sta_net_split_registry_v2", "efrm_target_public_fold_v1"}:
            if manifest.get("metadata_sha256") and manifest["metadata_sha256"] != self.metadata_fingerprint():
                raise RuntimeError("shared split metadata fingerprint drifted from EFRM task ordering")
            train = [int(value) for value in manifest["train_indices"]]
            validation = [int(value) for value in manifest["validation_indices"]]
        else:
            train_subjects = {str(value) for value in manifest["train_subjects"]}
            validation_subjects = {str(value) for value in manifest["validation_subjects"]}
            reserved_subjects = {
                str(value) for value in manifest.get("reserved_test_subjects", ())
            }
            if not train_subjects or not validation_subjects:
                raise RuntimeError("subject split has an empty train or validation partition")
            if train_subjects & validation_subjects:
                raise RuntimeError("subject split train/validation subjects overlap")
            if (train_subjects | validation_subjects) & reserved_subjects:
                raise RuntimeError("public and reserved subjects overlap")
            rows = [self.lightweight_metadata(index) for index in range(len(self))]
            available_subjects = {str(row["subject"]) for row in rows}
            unknown = (train_subjects | validation_subjects | reserved_subjects) - available_subjects
            if unknown:
                raise RuntimeError(f"split contains subjects absent from task data: {sorted(unknown)}")
            train = [
                index for index, row in enumerate(rows)
                if str(row["subject"]) in train_subjects
            ]
            validation = [
                index for index, row in enumerate(rows)
                if str(row["subject"]) in validation_subjects
            ]
        if not train or not validation or set(train).intersection(validation):
            raise RuntimeError("split has empty or overlapping train/validation indices")
        if min(train + validation) < 0 or max(train + validation) >= len(self):
            raise IndexError("split contains an out-of-range dataset index")
        return train, validation

    def contract_summary(self) -> dict[str, Any]:
        subjects = sorted({self.lightweight_metadata(index)["subject"] for index in range(len(self))})
        return {
            "schema": "efrm_sync_task_dataset_v1",
            "task": asdict(self.spec),
            "sample_count": len(self),
            "subject_count": len(subjects),
            "subjects": subjects,
            "excluded_label_counts": self.excluded_label_counts,
            "metadata_sha256": self.metadata_fingerprint(),
            "adapter": self.adapter.manifest(),
            "source_loader": self.base.contract_summary(),
        }


def collate_efrm_task(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty EFRM task batch")
    eeg_shapes = {tuple(sample["eeg"].shape) for sample in samples}
    fnirs_shapes = {tuple(sample["fnirs"].shape) for sample in samples}
    if len(eeg_shapes) != 1 or len(fnirs_shapes) != 1:
        raise ValueError("downstream EFRM batches must be record-grouped")
    result: dict[str, Any] = {
        "eeg": torch.stack([sample["eeg"] for sample in samples]),
        "fnirs": torch.stack([sample["fnirs"] for sample in samples]),
        "eeg_patch_valid": torch.stack([sample["eeg_patch_valid"] for sample in samples]),
        "fnirs_patch_valid": torch.stack([sample["fnirs_patch_valid"] for sample in samples]),
        "target": torch.stack([sample["target"] for sample in samples]),
        "target_valid_mask": torch.stack([sample["target_valid_mask"] for sample in samples]),
    }
    if "target_native" in samples[0]:
        result["target_native"] = torch.stack([sample["target_native"] for sample in samples])
    for key in ("sample_id", "dataset_id", "subject", "record_id", "join_key", "condition"):
        result[key] = [sample[key] for sample in samples]
    return result


def task_contract_sha256(spec: EFRMTaskSpec) -> str:
    payload = json.dumps(asdict(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
