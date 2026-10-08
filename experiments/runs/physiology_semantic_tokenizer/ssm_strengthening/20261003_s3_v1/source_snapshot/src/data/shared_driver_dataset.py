"""R-series measured local views with independently joined trajectory targets."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .physiology_semantic_local import (
    LOCAL_VIEW_SCHEMA,
    UnifiedPhysiologyLocalViewDataset,
)
from .physiology_semantic_targets import target_sample_key
from .shared_driver_targets import (
    PhysiologyRawViewRegistry,
    SharedDriverTrajectorySidecar,
)


SHARED_DRIVER_VIEW_SCHEMA = "shared_driver_measurement_view_v1"
SINGLE_TRIAL_PROTECTED_SUBJECT_KEYS = frozenset(
    f"eeg_fnirs_single_trial|subject_{index:02d}" for index in range(24, 30)
)


class SharedDriverWindowDataset(UnifiedPhysiologyLocalViewDataset):
    """Build the raw view first, then attach an R1 trajectory sidecar by identity."""

    def __init__(
        self,
        cache_root: str = "data/cache/physiology_semantic_clean_v1",
        *,
        raw_view_registry_root: str,
        trajectory_sidecar_root: str | None = None,
        expected_teacher_scope: str | None = None,
        expected_target_family: str | None = None,
        require_trajectory_target: bool = True,
        restrict_to_registered_views: bool = True,
        dataset_ids: Sequence[str] = ("eeg_fnirs_single_trial",),
        subject_keys: Iterable[str] | None = None,
        task_namespaces: Iterable[str] | None = None,
        window_duration_s: float = 20.0,
        reject_unknown_labels: bool = True,
        allow_cross_coordinate_systems: bool = False,
        window_offset_s: float = 0.0,
        eeg_signal_branch: str = "single_trial_eeg_artifact_clean_v4",
        base_dataset: Any | None = None,
    ) -> None:
        if subject_keys is None:
            raise ValueError(
                "R-series datasets require an explicit subject_keys allowlist "
                "so the protected boundary fails closed"
            )
        subject_keys = tuple(str(value) for value in subject_keys)
        protected_requested = sorted(
            set(subject_keys).intersection(SINGLE_TRIAL_PROTECTED_SUBJECT_KEYS)
        )
        if protected_requested:
            raise PermissionError(
                "Development R-series loader refuses protected-test subjects: "
                f"{protected_requested}"
            )
        super().__init__(
            cache_root=cache_root,
            dataset_ids=dataset_ids,
            subject_keys=subject_keys,
            task_namespaces=task_namespaces,
            window_duration_s=window_duration_s,
            local_eeg_channels=6,
            reject_unknown_labels=reject_unknown_labels,
            allow_cross_coordinate_systems=allow_cross_coordinate_systems,
            window_offset_s=window_offset_s,
            eeg_signal_branch=eeg_signal_branch,
            auxiliary_target_root=None,
            require_auxiliary_target=False,
            base_dataset=base_dataset,
        )
        self.raw_view_registry = PhysiologyRawViewRegistry(raw_view_registry_root)
        self.trajectory_targets = (
            None
            if trajectory_sidecar_root is None
            else SharedDriverTrajectorySidecar(
                trajectory_sidecar_root,
                expected_scope=expected_teacher_scope,
                expected_family=expected_target_family,
            )
        )
        self.require_trajectory_target = bool(require_trajectory_target)
        if self.require_trajectory_target and self.trajectory_targets is None:
            raise ValueError(
                "require_trajectory_target=True requires trajectory_sidecar_root"
            )

        sample_key_by_base_index: dict[int, str] = {}
        admitted = []
        for entry in self.entries:
            ref = self.base.windows[entry.base_index]
            event_index = int(ref.event.get("event_index", entry.base_index))
            key = target_sample_key(
                str(ref.record.dataset_id),
                str(ref.record.canonical_subject_id),
                str(ref.record.base_record_id),
                event_index,
            )
            sample_key_by_base_index[entry.base_index] = key
            view_present = self.raw_view_registry.contains(key)
            target_present = (
                self.trajectory_targets is not None
                and self.trajectory_targets.contains(key)
            )
            if restrict_to_registered_views and not view_present:
                continue
            if self.require_trajectory_target and not target_present:
                continue
            admitted.append(entry)
        if not admitted:
            raise ValueError("Shared-driver dataset has no admitted registered windows")
        self.entries = admitted
        self._sample_key_by_base_index = sample_key_by_base_index

    @staticmethod
    def _tensor_sha256(eeg: np.ndarray, fnirs: np.ndarray) -> str:
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(eeg).view(np.uint8))
        digest.update(np.ascontiguousarray(fnirs).view(np.uint8))
        return digest.hexdigest()

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        sample = self.base[entry.base_index]
        key = self._sample_key_by_base_index[entry.base_index]
        view = self.raw_view_registry.lookup(key)
        if view is None:
            raise KeyError(f"No frozen raw view for measured sample {key}")

        eeg_indices = self._indices_by_name(
            sample["channel_names"]["eeg"],
            view["selected_eeg_channels"],
            modality="EEG",
        )
        fnirs_indices = self._indices_by_name(
            sample["channel_names"]["fnirs"],
            view["selected_fnirs_channels"],
            modality="fNIRS",
        )
        if len(eeg_indices) != 6 or len(fnirs_indices) != 2:
            raise ValueError("Frozen raw view must select six EEG and two fNIRS channels")
        roles = [
            str(sample["component_roles"]["fnirs"][int(i)])
            for i in fnirs_indices
        ]
        if roles != ["HbO", "HbR"]:
            raise ValueError(
                f"Frozen fNIRS channel order must be [HbO,HbR], got {roles}"
            )
        bad_eeg = np.asarray(sample["bad_channel_mask"]["eeg"], dtype=bool)[
            eeg_indices
        ]
        bad_fnirs = np.asarray(sample["bad_channel_mask"]["fnirs"], dtype=bool)[
            fnirs_indices
        ]
        if bad_eeg.any() or bad_fnirs.any():
            raise ValueError(
                "Frozen raw-view registry selects a channel rejected by the "
                "current measured-data channel contract"
            )

        eeg_valid = np.asarray(sample["valid_mask"]["eeg"], dtype=bool)
        fnirs_valid = np.asarray(sample["valid_mask"]["fnirs"], dtype=bool)
        eeg = np.asarray(sample["eeg"], dtype=np.float32)[eeg_indices].copy()
        fnirs = np.asarray(sample["fnirs"], dtype=np.float32)[fnirs_indices].copy()
        eeg[:, ~eeg_valid] = 0.0
        fnirs[:, ~fnirs_valid] = 0.0

        # The privileged target is looked up only after the complete raw tensor
        # and its hash have been fixed.
        raw_view_sha256 = self._tensor_sha256(eeg, fnirs)
        target = (
            None
            if self.trajectory_targets is None
            else self.trajectory_targets.lookup(key)
        )
        if target is None and self.require_trajectory_target:
            raise KeyError(f"No shared-driver trajectory for measured sample {key}")

        label = sample["label"]
        event = sample["event"]
        anchor_id = str(view["anchor_id"])
        output: dict[str, Any] = {
            "schema": SHARED_DRIVER_VIEW_SCHEMA,
            "base_schema": LOCAL_VIEW_SCHEMA,
            "eeg": torch.from_numpy(np.ascontiguousarray(eeg)),
            "fnirs": torch.from_numpy(np.ascontiguousarray(fnirs)),
            "token_valid_mask": {
                "eeg": self._token_mask(eeg_valid, 400),
                "fnirs": self._token_mask(fnirs_valid, 20),
            },
            "subject_key": entry.subject_key,
            "dataset_id": str(sample["dataset_id"]),
            "subject": str(sample["subject"]),
            "record_id": str(sample["record_id"]),
            "task_namespace": str(label["namespace"]),
            "condition": str(label.get("condition", event.get("label", ""))),
            "label": torch.tensor(int(label["class_index"]), dtype=torch.long),
            "anchor": anchor_id,
            "selected_eeg_channels": list(view["selected_eeg_channels"]),
            "selected_fnirs_channels": list(view["selected_fnirs_channels"]),
            "dependency_group_id": entry.dependency_group_id,
            "sample_id": f"{key}|anchor={anchor_id}",
            "target_sample_key": key,
            "raw_view_sha256": raw_view_sha256,
            "raw_view_selection_fold": str(view["selection_fold"]),
            "raw_view_selection_source_hash": str(
                view["selection_source_hash"]
            ),
            "event_onset_ms": float(event.get("onset_ms", np.nan)),
            "window_offset_s": float(self.base.window_offset_s),
            "has_trajectory_target": torch.tensor(
                target is not None, dtype=torch.bool
            ),
        }
        if target is not None:
            output["teacher"] = {
                key: value
                for key, value in target.items()
                if isinstance(value, torch.Tensor)
            }
            output["teacher_provenance"] = {
                key: value
                for key, value in target.items()
                if not isinstance(value, torch.Tensor)
            }
        return output


__all__ = [
    "SHARED_DRIVER_VIEW_SCHEMA",
    "SINGLE_TRIAL_PROTECTED_SUBJECT_KEYS",
    "SharedDriverWindowDataset",
]
