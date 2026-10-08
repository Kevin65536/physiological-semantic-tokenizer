"""Measurement-first local views for the physiology-semantic tokenizer.

The adapter consumes :class:`UnifiedPhysiologyWindowDataset` samples and only
then constructs the existing six-EEG/two-chromophore model view.  It never
reads a Croce cache or substitutes a teacher target for a measured signal.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .unified_physiology import UnifiedPhysiologyWindowDataset, canonical_label
from .clean_physiology_cache import DEFAULT_CLEAN_CACHE_ROOT
from .physiology_semantic_targets import (
    PhysiologySemanticTargetSidecar,
    target_sample_key,
)


LOCAL_VIEW_SCHEMA = "physiology_semantic_measurement_local_v1"
LOCAL_MEASUREMENT_VIEW_SCHEMA = "physiology_semantic_measurement_local_v2"


@dataclass(frozen=True)
class LocalWindowEntry:
    base_index: int
    dataset_id: str
    subject_key: str
    dependency_group_id: str


def _position(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[row.get(axis) for axis in ("x", "y", "z")] for row in rows],
        dtype=np.float64,
    )


def _visual_dependency_group(record_id: str, event: Mapping[str, Any]) -> str:
    record = str(record_id).replace("_Probe1", "").replace("_Probe2", "")
    event_index = event.get("event_index", event.get("epoch_id", -1))
    return f"{record}|event={event_index}"


class UnifiedPhysiologyLocalViewDataset(Dataset):
    """Select one deterministic measured local view from every unified window."""

    def __init__(
        self,
        cache_root: str = DEFAULT_CLEAN_CACHE_ROOT,
        *,
        dataset_ids: Sequence[str] = ("eeg_fnirs_single_trial",),
        subject_keys: Iterable[str] | None = None,
        task_namespaces: Iterable[str] | None = None,
        window_duration_s: float = 20.0,
        local_eeg_channels: int = 6,
        reject_unknown_labels: bool = True,
        allow_cross_coordinate_systems: bool = False,
        window_offset_s: float = 0.0,
        eeg_signal_branch: str = "single_trial_eeg_artifact_clean_v4",
        output_coordinate: str = "measurement",
        auxiliary_target_root: str | None = None,
        auxiliary_target_family: str | None = None,
        auxiliary_target_version: str | None = None,
        require_auxiliary_target: bool = False,
        base_dataset: UnifiedPhysiologyWindowDataset | None = None,
    ) -> None:
        self.base = base_dataset or UnifiedPhysiologyWindowDataset(
            cache_root=cache_root,
            dataset_ids=dataset_ids,
            window_duration_s=window_duration_s,
            window_offset_s=window_offset_s,
            eeg_signal_branch=eeg_signal_branch,
            output_coordinate=output_coordinate,
        )
        self.local_eeg_channels = int(local_eeg_channels)
        if self.local_eeg_channels != 6:
            raise ValueError("The current local tokenizer contract requires six EEG channels")
        self.allow_cross_coordinate_systems = bool(allow_cross_coordinate_systems)
        self.require_auxiliary_target = bool(require_auxiliary_target)
        if (getattr(self.base, 'output_coordinate', output_coordinate) == 'measurement'
                and auxiliary_target_root is not None):
            raise ValueError('measurement coordinates have no qualified compatible teacher sidecar')
        self.auxiliary_targets = (
            None
            if auxiliary_target_root is None
            else PhysiologySemanticTargetSidecar(
                auxiliary_target_root,
                expected_family=auxiliary_target_family,
                expected_version=auxiliary_target_version,
            )
        )
        if self.require_auxiliary_target and self.auxiliary_targets is None:
            raise ValueError("require_auxiliary_target=True requires an auxiliary target sidecar")
        requested_subjects = None if subject_keys is None else {str(value) for value in subject_keys}
        requested_tasks = None if task_namespaces is None else {str(value) for value in task_namespaces}

        entries: list[LocalWindowEntry] = []
        for index, ref in enumerate(self.base.windows):
            subject_key = f"{ref.record.dataset_id}|{ref.record.canonical_subject_id}"
            label = canonical_label(ref.event, ref.record.dataset_id)
            if requested_subjects is not None and subject_key not in requested_subjects:
                continue
            if requested_tasks is not None and label["namespace"] not in requested_tasks:
                continue
            if reject_unknown_labels and (
                int(label.get("class_index", -1)) < 0
                or str(label.get("condition", "unknown")).lower() == "unknown"
            ):
                continue
            dependency = (
                _visual_dependency_group(ref.record.base_record_id, ref.event)
                if ref.record.dataset_id == "visual_cognitive_motivation"
                else f"{ref.record.join_key}|event={ref.event.get('event_index', index)}"
            )
            entries.append(
                LocalWindowEntry(
                    base_index=index,
                    dataset_id=ref.record.dataset_id,
                    subject_key=subject_key,
                    dependency_group_id=dependency,
                )
            )
        if not entries:
            raise ValueError("Unified local view produced no admitted windows")
        self.entries = entries
        self.subject_keys = {entry.subject_key for entry in entries}

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _paired_indices(sample: Mapping[str, Any]) -> list[tuple[int, int]]:
        roles = list(sample["component_roles"]["fnirs"])
        rows = list(sample["channel_geometry"]["fnirs"])
        bad = np.asarray(sample["bad_channel_mask"]["fnirs"], dtype=bool)
        support = sample.get('channel_valid_mask', {}).get('fnirs')
        if support is not None:
            bad = bad | ~np.asarray(support, dtype=bool).any(axis=1)
        hbr_by_base = {
            str(row.get("base_channel_name")): index
            for index, (row, role) in enumerate(zip(rows, roles))
            if role == "HbR" and not bad[index]
        }
        pairs = []
        for index, (row, role) in enumerate(zip(rows, roles)):
            if role != "HbO" or bad[index]:
                continue
            pair = hbr_by_base.get(str(row.get("base_channel_name")))
            if pair is not None:
                pairs.append((index, pair))
        if not pairs:
            raise ValueError("Measured fNIRS window has no paired HbO/HbR anchors")
        return pairs

    def _select_eeg(
        self,
        sample: Mapping[str, Any],
        fnirs_index: int,
    ) -> np.ndarray:
        eeg_rows = list(sample["channel_geometry"]["eeg"])
        fnirs_rows = list(sample["channel_geometry"]["fnirs"])
        eeg_positions = _position(eeg_rows)
        anchor = _position([fnirs_rows[fnirs_index]])[0]
        eeg_units = {str(row.get("coordinate_units")) for row in eeg_rows}
        fnirs_units = str(fnirs_rows[fnirs_index].get("coordinate_units"))
        if not self.allow_cross_coordinate_systems and (
            len(eeg_units) != 1 or fnirs_units not in eeg_units
        ):
            raise ValueError(
                "Cross-modal local selection requires matching coordinate units; "
                "use an explicitly admitted adapter before enabling cross-system geometry"
            )
        finite = np.all(np.isfinite(eeg_positions), axis=1) & np.isfinite(anchor).all()
        bad = np.asarray(sample["bad_channel_mask"]["eeg"], dtype=bool)
        support = sample.get('channel_valid_mask', {}).get('eeg')
        if support is not None:
            bad = bad | ~np.asarray(support, dtype=bool).any(axis=1)
        eligible = finite & ~bad
        distances = np.linalg.norm(eeg_positions - anchor[None, :], axis=1)
        distances[~eligible] = np.inf
        selected = np.argsort(distances)[: self.local_eeg_channels]
        if len(selected) != self.local_eeg_channels or np.any(~np.isfinite(distances[selected])):
            raise ValueError("Fewer than six finite, non-bad EEG channels surround the fNIRS anchor")
        return np.asarray(selected, dtype=np.int64)

    @staticmethod
    def _indices_by_name(
        available: Sequence[str],
        requested: Sequence[str],
        *,
        modality: str,
    ) -> np.ndarray:
        lookup = {str(name): index for index, name in enumerate(available)}
        missing = [str(name) for name in requested if str(name) not in lookup]
        if missing:
            raise ValueError(f"Sidecar-selected {modality} channels are absent: {missing}")
        indices = np.asarray([lookup[str(name)] for name in requested], dtype=np.int64)
        if len(set(indices.tolist())) != len(indices):
            raise ValueError(f"Sidecar-selected {modality} channels must be unique")
        return indices

    @staticmethod
    def _token_mask(mask: np.ndarray, patch_samples: int) -> torch.Tensor:
        value = np.asarray(mask, dtype=bool)
        if value.size % patch_samples:
            raise ValueError("Validity mask does not align to the tokenizer patch grid")
        return torch.from_numpy(value.reshape(-1, patch_samples).all(axis=1))

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        sample = self.base[entry.base_index]
        measurement = sample.get('coordinate_layer') == 'cleaned_measurement'
        if measurement and self.auxiliary_targets is not None:
            raise ValueError('measurement coordinates have no qualified compatible teacher sidecar')
        event_index = int(sample["event"].get("event_index", entry.base_index))
        target_key = target_sample_key(
            str(sample["dataset_id"]),
            str(sample["subject"]),
            str(sample["record_id"]),
            event_index,
        )
        target = None if self.auxiliary_targets is None else self.auxiliary_targets.lookup(target_key)
        target_rejection_reason = ""
        if target is not None:
            eeg_indices = self._indices_by_name(
                sample["channel_names"]["eeg"],
                target["selected_eeg_channels"],
                modality="EEG",
            )
            fnirs_indices = self._indices_by_name(
                sample["channel_names"]["fnirs"],
                target["selected_fnirs_channels"],
                modality="fNIRS",
            )
            if len(eeg_indices) != self.local_eeg_channels or len(fnirs_indices) != 2:
                raise ValueError("Sidecar local view must select six EEG and two fNIRS channels")
            selected_bad_eeg = bool(
                np.asarray(sample["bad_channel_mask"]["eeg"], dtype=bool)[eeg_indices].any()
            )
            selected_bad_fnirs = bool(
                np.asarray(sample["bad_channel_mask"]["fnirs"], dtype=bool)[fnirs_indices].any()
            )
            if selected_bad_eeg or selected_bad_fnirs:
                if self.require_auxiliary_target:
                    raise ValueError(
                        "Sidecar selected a channel marked bad by the measured-data contract"
                    )
                target = None
                target_rejection_reason = "sidecar_selected_bad_measured_channel"
        if target is not None:
            roles = [sample["component_roles"]["fnirs"][int(i)] for i in fnirs_indices]
            if roles != ["HbO", "HbR"]:
                raise ValueError(f"Sidecar fNIRS channel order must be [HbO,HbR], got {roles}")
            hbo_index, hbr_index = (int(fnirs_indices[0]), int(fnirs_indices[1]))
        else:
            if self.require_auxiliary_target and not target_rejection_reason:
                raise KeyError(f"No auxiliary target for measured sample {target_key}")
            pairs = self._paired_indices(sample)
            digest = hashlib.sha256(target_key.encode("utf-8")).digest()
            pair_index = int.from_bytes(digest[:8], "little") % len(pairs)
            hbo_index, hbr_index = pairs[pair_index]
            eeg_indices = self._select_eeg(sample, hbo_index)

        # Boundary/data-presence validity is authoritative. Historical
        # analysis masks may contain retired artifact exclusions and must not
        # silently remove measured samples or entire tokenizer patches.
        eeg_valid = np.asarray(sample["valid_mask"]["eeg"], dtype=bool)
        fnirs_valid = np.asarray(sample["valid_mask"]["fnirs"], dtype=bool)
        dtype = np.float64 if measurement else np.float32
        eeg = np.asarray(sample["eeg"], dtype=dtype)[eeg_indices].copy()
        fnirs = np.asarray(sample["fnirs"], dtype=dtype)[[hbo_index, hbr_index]].copy()
        eeg_support = np.broadcast_to(eeg_valid, eeg.shape).copy()
        fnirs_support = np.broadcast_to(fnirs_valid, fnirs.shape).copy()
        if measurement:
            eeg_support &= np.asarray(sample['channel_valid_mask']['eeg'], dtype=bool)[eeg_indices]
            fnirs_support &= np.asarray(sample['channel_valid_mask']['fnirs'], dtype=bool)[[hbo_index, hbr_index]]
        eeg[~eeg_support] = 0.0
        fnirs[~fnirs_support] = 0.0
        eeg_valid, fnirs_valid = eeg_support.all(axis=0), fnirs_support.all(axis=0)
        label = sample["label"]
        anchor_name = str(sample["channel_geometry"]["fnirs"][hbo_index].get("base_channel_name"))
        sample_id = f"{target_key}|anchor={anchor_name}"
        output = {
            "schema": LOCAL_MEASUREMENT_VIEW_SCHEMA if measurement else LOCAL_VIEW_SCHEMA,
            "eeg": torch.from_numpy(np.ascontiguousarray(eeg)),
            "fnirs": torch.from_numpy(np.ascontiguousarray(fnirs)),
            "token_valid_mask": {
                "eeg": self._token_mask(eeg_valid, 400),
                "fnirs": self._token_mask(fnirs_valid, 20),
            },
            "subject_key": entry.subject_key,
            "dataset_id": sample["dataset_id"],
            "subject": str(sample["subject"]),
            "record_id": str(sample["record_id"]),
            "task_namespace": str(label["namespace"]),
            "label": torch.tensor(int(label["class_index"]), dtype=torch.long),
            "anchor": anchor_name,
            "selected_eeg_channels": [sample["channel_names"]["eeg"][int(i)] for i in eeg_indices],
            "dependency_group_id": entry.dependency_group_id,
            "sample_id": sample_id,
            "target_sample_key": target_key,
            "has_auxiliary_target": torch.tensor(target is not None, dtype=torch.bool),
            "auxiliary_target_rejection_reason": target_rejection_reason,
        }
        if measurement:
            output.update(unit=dict(sample['unit']), coordinate_layer='cleaned_measurement',
                          teacher_target_status='not_generated_no_qualification',
                          channel_valid_mask={'eeg': torch.from_numpy(eeg_support),
                                              'fnirs': torch.from_numpy(fnirs_support)},
                          selected_fnirs_channels=[sample['channel_names']['fnirs'][i]
                                                   for i in (hbo_index, hbr_index)],
                          measurement_metadata_json=json.dumps({
                              'preprocessing_state': sample['preprocessing_state'],
                              'channel_geometry': {
                                  'eeg': [sample['channel_geometry']['eeg'][int(i)] for i in eeg_indices],
                                  'fnirs': [sample['channel_geometry']['fnirs'][i] for i in (hbo_index, hbr_index)]}}))
        if self.auxiliary_targets is not None:
            if target is None:
                output["teacher"] = self.auxiliary_targets.empty_target(
                    tokens=int(round(self.base.window_duration_s / 2.0))
                )
            else:
                output["teacher"] = {
                    key: value
                    for key, value in target.items()
                    if isinstance(value, torch.Tensor)
                }
        return output


__all__ = [
    "LOCAL_VIEW_SCHEMA",
    "LOCAL_MEASUREMENT_VIEW_SCHEMA",
    "LocalWindowEntry",
    "UnifiedPhysiologyLocalViewDataset",
]
