"""Synchronized unified-loader adapters for the EFRM comparison project."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from src.data.clean_physiology_cache import DEFAULT_CLEAN_CACHE_ROOT
from src.data.unified_physiology import (
    RAW_DATASET_IDS,
    UnifiedPhysiologyWindowDataset,
    canonical_label,
)


ADAPTER_SCHEMA = "efrm_sync_unified_adapter_v1"
SAMPLER_SCHEMA = "balanced_dataset_inventory_diverse_epoch_crop_v1"
RECORD_GROUPED_SAMPLER_SCHEMA = "balanced_dataset_record_epoch_crop_v1"


def _stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _component_base(name: str) -> str:
    return re.sub(r"_(?:HbO|HbR)$", "", str(name), flags=re.IGNORECASE)


def _sample_id(sample: Mapping[str, Any], crop_start_s: float) -> str:
    event = sample.get("event", {})
    raw = {
        "dataset": str(sample["dataset_id"]),
        "subject": str(sample["subject"]),
        "record": str(sample["record_id"]),
        # Downstream sequence datasets can expose several windows from the
        # same source event.  Their canonical sample_id carries the window
        # coordinate even when the sanitized event payload intentionally does
        # not, so it must participate in the EFRM-derived identity.
        "source_sample_id": str(sample.get("sample_id", "")),
        "event": int(event.get("event_index", -1)),
        "window_offset_s": float(event.get("window_offset_s", 0.0)),
        "crop_start_s": float(crop_start_s),
    }
    digest = hashlib.sha256(repr(sorted(raw.items())).encode("utf-8")).hexdigest()[:16]
    return f"{raw['dataset']}|{raw['subject']}|{raw['record']}|{digest}"


def _crop(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    rate_hz: float,
    duration_s: float,
    start_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    target = int(round(float(rate_hz) * float(duration_s)))
    start = int(round(float(rate_hz) * float(start_s)))
    stop = start + target
    array = np.asarray(values, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool).reshape(-1)
    output = np.zeros((array.shape[0], target), dtype=np.float32)
    output_mask = np.zeros(target, dtype=bool)
    source_stop = min(stop, array.shape[1], valid.size)
    usable = max(0, source_stop - start)
    if usable:
        output[:, :usable] = array[:, start:source_stop]
        output_mask[:usable] = valid[start:source_stop]
    output[:, ~output_mask] = 0.0
    return output, output_mask


def _time_patch_mask(time_mask: np.ndarray, patch_samples: int, channels: int) -> np.ndarray:
    mask = np.asarray(time_mask, dtype=bool)
    usable = (mask.size // int(patch_samples)) * int(patch_samples)
    if usable == 0:
        return np.zeros((channels, 0), dtype=bool)
    patch_valid = mask[:usable].reshape(-1, int(patch_samples)).all(axis=1)
    return np.broadcast_to(patch_valid[None, :], (channels, patch_valid.size)).copy()


class EFRMPairedWindowAdapter:
    """Convert one unified synchronized window to variable-channel EFRM tensors."""

    def __init__(
        self,
        *,
        duration_s: float = 8.0,
        eeg_rate_hz: float = 200.0,
        fnirs_rate_hz: float = 10.0,
        eeg_patch_samples: int = 50,
        fnirs_patch_samples: int = 20,
        require_full_analysis_support: bool = True,
    ) -> None:
        self.duration_s = float(duration_s)
        self.eeg_rate_hz = float(eeg_rate_hz)
        self.fnirs_rate_hz = float(fnirs_rate_hz)
        self.eeg_patch_samples = int(eeg_patch_samples)
        self.fnirs_patch_samples = int(fnirs_patch_samples)
        self.require_full_analysis_support = bool(require_full_analysis_support)

    def _validate_rates(self, sample: Mapping[str, Any]) -> None:
        eeg_rate = float(sample["sample_rate_hz"]["eeg"])
        fnirs_rate = float(sample["sample_rate_hz"]["fnirs"])
        if abs(eeg_rate - self.eeg_rate_hz) > 1e-6 or abs(fnirs_rate - self.fnirs_rate_hz) > 1e-6:
            raise ValueError(
                f"EFRM adapter requires EEG@{self.eeg_rate_hz:g} and fNIRS@{self.fnirs_rate_hz:g}; "
                f"received {eeg_rate:g}/{fnirs_rate:g}"
            )

    @staticmethod
    def _paired_fnirs_indices(
        sample: Mapping[str, Any],
    ) -> tuple[list[int], list[int], list[str], np.ndarray]:
        names = [str(value) for value in sample["channel_names"]["fnirs"]]
        roles = [str(value) for value in sample["component_roles"]["fnirs"]]
        bad = np.asarray(sample["bad_channel_mask"]["fnirs"], dtype=bool)
        by_role: dict[str, dict[str, int]] = {"HbO": {}, "HbR": {}}
        order: list[str] = []
        for index, (name, role) in enumerate(zip(names, roles, strict=True)):
            if role not in by_role:
                continue
            base = _component_base(name)
            by_role[role][base] = index
            if role == "HbO":
                order.append(base)
        paired = [base for base in order if base in by_role["HbR"]]
        good = np.asarray(
            [not bad[by_role["HbO"][base]] and not bad[by_role["HbR"][base]] for base in paired],
            dtype=bool,
        )
        if not paired or not good.any():
            raise ValueError("No valid paired HbO/HbR spatial locations remain")
        return (
            [by_role["HbO"][base] for base in paired],
            [by_role["HbR"][base] for base in paired],
            paired,
            good,
        )

    def adapt(self, sample: Mapping[str, Any], *, crop_start_s: float = 0.0) -> dict[str, Any]:
        self._validate_rates(sample)
        eeg_bad = np.asarray(sample["bad_channel_mask"]["eeg"], dtype=bool)
        if eeg_bad.all():
            raise ValueError("No valid EEG channels remain")

        eeg, eeg_time_valid = _crop(
            np.asarray(sample["eeg"]),
            sample["analysis_valid_mask"]["eeg"],
            rate_hz=self.eeg_rate_hz,
            duration_s=self.duration_s,
            start_s=crop_start_s,
        )
        eeg[eeg_bad] = 0.0
        hbo, hbr, fnirs_locations, fnirs_location_good = self._paired_fnirs_indices(sample)
        fnirs_indices = hbo + hbr
        fnirs_flat, fnirs_time_valid = _crop(
            np.asarray(sample["fnirs"])[fnirs_indices],
            sample["analysis_valid_mask"]["fnirs"],
            rate_hz=self.fnirs_rate_hz,
            duration_s=self.duration_s,
            start_s=crop_start_s,
        )
        locations = len(fnirs_locations)
        fnirs = np.stack((fnirs_flat[:locations], fnirs_flat[locations:]), axis=0)
        fnirs[:, ~fnirs_location_good] = 0.0

        eeg_patch_valid = _time_patch_mask(eeg_time_valid, self.eeg_patch_samples, len(eeg_bad))
        eeg_patch_valid &= ~eeg_bad[:, None]
        fnirs_patch_valid = _time_patch_mask(fnirs_time_valid, self.fnirs_patch_samples, locations)
        fnirs_patch_valid &= fnirs_location_good[:, None]
        full_support = bool(eeg_time_valid.all() and fnirs_time_valid.all())
        admitted = full_support or not self.require_full_analysis_support

        return {
            "eeg": torch.from_numpy(eeg[None]),
            "fnirs": torch.from_numpy(fnirs),
            "eeg_patch_valid": torch.from_numpy(eeg_patch_valid),
            "fnirs_patch_valid": torch.from_numpy(fnirs_patch_valid),
            "eeg_time_valid": torch.from_numpy(eeg_time_valid),
            "fnirs_time_valid": torch.from_numpy(fnirs_time_valid),
            "admitted": admitted,
            "sample_id": _sample_id(sample, crop_start_s),
            "dataset_id": str(sample["dataset_id"]),
            "subject": str(sample["subject"]),
            "record_id": str(sample["record_id"]),
            "join_key": str(sample["join_key"]),
            "task_namespace": str(sample.get("label", {}).get("namespace", "")),
            "condition": str(sample.get("label", {}).get("condition", "")),
            "crop_start_s": float(crop_start_s),
            "duration_s": self.duration_s,
            "eeg_channel_names": [str(value) for value in sample["channel_names"]["eeg"]],
            "fnirs_location_names": fnirs_locations,
            "adapter_state": {
                "schema": ADAPTER_SCHEMA,
                "eeg_rate_hz": self.eeg_rate_hz,
                "fnirs_rate_hz": self.fnirs_rate_hz,
                "eeg_patch_samples": self.eeg_patch_samples,
                "fnirs_patch_samples": self.fnirs_patch_samples,
                "channel_policy": "variable_measured_channels_no_duplication_v1",
                "full_analysis_support": full_support,
            },
        }

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": ADAPTER_SCHEMA,
            "duration_s": self.duration_s,
            "sample_rates_hz": {"eeg": self.eeg_rate_hz, "fnirs": self.fnirs_rate_hz},
            "patch_samples": {"eeg": self.eeg_patch_samples, "fnirs": self.fnirs_patch_samples},
            "patch_duration_s": {
                "eeg": self.eeg_patch_samples / self.eeg_rate_hz,
                "fnirs": self.fnirs_patch_samples / self.fnirs_rate_hz,
            },
            "channel_policy": "variable_measured_channels_no_duplication_v1",
            "mask_policy": "exclude bad channels; zero invalid time; mask reconstruction and pooling",
            "require_full_analysis_support": self.require_full_analysis_support,
        }


class EFRMSyncPretrainDataset(Dataset):
    """All admitted synchronized unified windows with deterministic epoch crops."""

    def __init__(
        self,
        cache_root: str = DEFAULT_CLEAN_CACHE_ROOT,
        *,
        dataset_ids: Sequence[str] = RAW_DATASET_IDS,
        task_namespaces: Sequence[str] | None = None,
        seed: int = 42,
        adapter: EFRMPairedWindowAdapter | None = None,
        base: UnifiedPhysiologyWindowDataset | None = None,
    ) -> None:
        self.base = base or UnifiedPhysiologyWindowDataset(
            cache_root=cache_root,
            dataset_ids=tuple(dataset_ids),
        )
        self.task_namespaces = tuple(str(value) for value in (task_namespaces or ()))
        allowed_namespaces = set(self.task_namespaces)
        self.indices = [
            index
            for index, window in enumerate(self.base.windows)
            if not allowed_namespaces
            or canonical_label(window.event, window.record.dataset_id)["namespace"]
            in allowed_namespaces
        ]
        if not self.indices:
            raise RuntimeError(
                "no synchronized pretraining windows survive task namespace filtering: "
                f"{sorted(allowed_namespaces)}"
            )
        self.seed = int(seed)
        self.epoch = 0
        self.adapter = adapter or EFRMPairedWindowAdapter()

    def __len__(self) -> int:
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def lightweight_metadata(self, index: int) -> dict[str, Any]:
        ref = self.base.windows[self.indices[int(index)]]
        label = canonical_label(ref.event, ref.record.dataset_id)
        return {
            "dataset_id": str(ref.record.dataset_id),
            "subject": str(ref.record.canonical_subject_id),
            "record_id": str(ref.record.base_record_id),
            "join_key": str(ref.record.join_key),
            "event_index": int(ref.event.get("event_index", -1)),
            "window_offset_s": float(ref.window_offset_s),
            "task_namespace": str(label["namespace"]),
            "condition": str(label["condition"]),
        }


    def _crop_start(self, sample: Mapping[str, Any], index: int) -> float:
        eeg_seconds = np.asarray(sample["eeg"]).shape[-1] / float(sample["sample_rate_hz"]["eeg"])
        fnirs_seconds = np.asarray(sample["fnirs"]).shape[-1] / float(sample["sample_rate_hz"]["fnirs"])
        maximum = max(0.0, min(eeg_seconds, fnirs_seconds) - self.adapter.duration_s)
        if maximum <= 0:
            return 0.0
        steps = int(math.floor(maximum * self.adapter.fnirs_rate_hz)) + 1
        eeg_mask = np.asarray(sample["analysis_valid_mask"]["eeg"], dtype=bool)
        fnirs_mask = np.asarray(sample["analysis_valid_mask"]["fnirs"], dtype=bool)
        eeg_length = int(round(self.adapter.duration_s * self.adapter.eeg_rate_hz))
        fnirs_length = int(round(self.adapter.duration_s * self.adapter.fnirs_rate_hz))
        valid_starts: list[int] = []
        for candidate in range(steps):
            start_s = candidate / self.adapter.fnirs_rate_hz
            eeg_start = int(round(start_s * self.adapter.eeg_rate_hz))
            fnirs_start = candidate
            if (
                eeg_start + eeg_length <= eeg_mask.size
                and fnirs_start + fnirs_length <= fnirs_mask.size
                and eeg_mask[eeg_start : eeg_start + eeg_length].all()
                and fnirs_mask[fnirs_start : fnirs_start + fnirs_length].all()
            ):
                valid_starts.append(candidate)
        if not valid_starts:
            return 0.0
        selected = _stable_int(self.seed, self.epoch, index, sample["join_key"]) % len(valid_starts)
        return valid_starts[selected] / self.adapter.fnirs_rate_hz

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[self.indices[int(index)]]
        return self.adapter.adapt(sample, crop_start_s=self._crop_start(sample, int(index)))

    def contract_summary(self) -> dict[str, Any]:
        datasets = defaultdict(int)
        task_namespaces = defaultdict(int)
        records: dict[str, set[str]] = defaultdict(set)
        subjects: dict[str, set[str]] = defaultdict(set)
        for index in range(len(self)):
            row = self.lightweight_metadata(index)
            dataset_id = row["dataset_id"]
            datasets[dataset_id] += 1
            task_namespaces[row["task_namespace"]] += 1
            records[dataset_id].add(row["join_key"])
            subjects[dataset_id].add(row["subject"])
        return {
            "schema": "efrm_sync_pretrain_dataset_v1",
            "sample_count": len(self),
            "sample_count_by_dataset": dict(datasets),
            "sample_count_by_task_namespace": dict(task_namespaces),
            "task_namespace_filter": list(self.task_namespaces),
            "record_count_by_dataset": {key: len(value) for key, value in records.items()},
            "subject_count_by_dataset": {key: len(value) for key, value in subjects.items()},
            "adapter": self.adapter.manifest(),
            "source_loader": self.base.contract_summary(),
        }


class CachedEFRMPretrainDataset(Dataset):
    """Read fixed, split-approved 8-second tensors without reopening raw files."""

    def __init__(self, source: EFRMSyncPretrainDataset, selected_indices: Sequence[int],
                 cache_root: str | Path, *, build: bool = True) -> None:
        self.source, self.adapter = source, source.adapter
        self.cache_root = Path(cache_root)
        self.samples_dir = self.cache_root / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.selected_indices = [int(value) for value in selected_indices]
        self.indices = list(range(len(self.selected_indices)))
        self.epoch = 0
        if build:
            self._build_missing()
        self.entries = [self._entry(index) for index in self.selected_indices]
        missing = [row["path"] for row in self.entries if not Path(row["path"]).is_file()]
        if missing:
            raise FileNotFoundError(f"EFRM tensor cache is incomplete; first missing={missing[0]}")

    def _path(self, source_index: int) -> Path:
        return self.samples_dir / f"sample_{self.source.indices[int(source_index)]:06d}.pt"

    def _entry(self, source_index: int) -> dict[str, Any]:
        return {**self.source.lightweight_metadata(source_index), "source_index": source_index,
                "path": str(self._path(source_index))}

    def _build_missing(self) -> None:
        missing = [index for index in self.selected_indices if not self._path(index).is_file()]
        if not missing:
            return
        print(f"EFRM tensor cache: building {len(missing)} fixed synchronized windows", flush=True)
        for position, source_index in enumerate(missing, start=1):
            destination = self._path(source_index)
            temporary = destination.with_suffix(".tmp")
            torch.save(self.source[source_index], temporary)
            temporary.replace(destination)
            if position % 500 == 0 or position == len(missing):
                print(f"EFRM tensor cache: {position}/{len(missing)}", flush=True)
        manifest = {
            "schema": "efrm_fixed_window_tensor_cache_v1",
            "crop_policy": "deterministic_epoch0_common_valid_start",
            "protected_test_opened": False,
            "adapter": self.adapter.manifest(),
            "sample_count": len(list(self.samples_dir.glob("sample_*.pt"))),
        }
        (self.cache_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

    def __len__(self) -> int:
        return len(self.entries)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def lightweight_metadata(self, index: int) -> dict[str, Any]:
        row = self.entries[self.indices[int(index)]]
        return {key: row[key] for key in (
            "dataset_id", "subject", "record_id", "join_key", "event_index", "window_offset_s",
            "task_namespace", "condition",
        )}

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.entries[self.indices[int(index)]]
        return torch.load(row["path"], map_location="cpu", weights_only=False)

    def contract_summary(self) -> dict[str, Any]:
        return {"schema": "efrm_fixed_window_tensor_cache_v1", "sample_count": len(self),
                "cache_root": str(self.cache_root.resolve()),
                "crop_policy": "deterministic_epoch0_common_valid_start",
                "adapter": self.adapter.manifest()}


class RecordGroupedBatchSampler(Sampler[list[int]]):
    """Balance datasets while keeping every batch inside one record inventory."""

    def __init__(
        self,
        dataset: EFRMSyncPretrainDataset,
        batch_size: int,
        *,
        seed: int = 42,
        drop_last: bool = False,
        minimum_batch_size: int = 2,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.minimum_batch_size = int(minimum_batch_size)
        self.epoch = 0
        groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index in range(len(dataset)):
            row = dataset.lightweight_metadata(index)
            groups[row["dataset_id"]][row["join_key"]].append(index)
        self.groups = {dataset_id: dict(records) for dataset_id, records in groups.items()}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.dataset.set_epoch(epoch)

    def _batches(self) -> dict[str, list[list[int]]]:
        output: dict[str, list[list[int]]] = {}
        for dataset_position, dataset_id in enumerate(sorted(self.groups)):
            rng = random.Random(self.seed + self.epoch * 10_007 + dataset_position)
            records = list(self.groups[dataset_id].items())
            rng.shuffle(records)
            batches: list[list[int]] = []
            for _, indices in records:
                shuffled = list(indices)
                rng.shuffle(shuffled)
                for start in range(0, len(shuffled), self.batch_size):
                    batch = shuffled[start : start + self.batch_size]
                    if self.drop_last and len(batch) < self.batch_size:
                        continue
                    if len(batch) >= self.minimum_batch_size:
                        batches.append(batch)
            rng.shuffle(batches)
            if batches:
                output[dataset_id] = batches
        return output

    def __iter__(self) -> Iterator[list[int]]:
        by_dataset = self._batches()
        if not by_dataset:
            return
        dataset_ids = sorted(by_dataset)
        maximum = max(len(value) for value in by_dataset.values())
        for position in range(maximum):
            for dataset_id in dataset_ids:
                batches = by_dataset[dataset_id]
                yield batches[position % len(batches)]

    def __len__(self) -> int:
        by_dataset = self._batches()
        return 0 if not by_dataset else max(len(value) for value in by_dataset.values()) * len(by_dataset)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": RECORD_GROUPED_SAMPLER_SCHEMA,
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
            "minimum_batch_size": self.minimum_batch_size,
            "dataset_record_counts": {
                dataset_id: len(records) for dataset_id, records in self.groups.items()
            },
        }


class InventoryDiverseBatchSampler(Sampler[list[int]]):
    """Balance datasets while making CLIP negatives record-diverse.

    Samples are stackable because each batch has one measured channel
    inventory. Within that inventory, round-robin drawing takes at most one
    window per record per pass. This avoids defining every off-diagonal CLIP
    cell from adjacent windows in the same recording.
    """

    def __init__(
        self,
        dataset: EFRMSyncPretrainDataset,
        batch_size: int,
        *,
        seed: int = 42,
        drop_last: bool = False,
        minimum_batch_size: int = 2,
        inventory_cache_path: str | Path | None = None,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.minimum_batch_size = int(minimum_batch_size)
        self.inventory_cache_path = None if inventory_cache_path is None else Path(inventory_cache_path)
        self.epoch = 0
        records: dict[str, list[int]] = defaultdict(list)
        record_dataset: dict[str, str] = {}
        for index in range(len(dataset)):
            row = dataset.lightweight_metadata(index)
            records[row["join_key"]].append(index)
            record_dataset[row["join_key"]] = row["dataset_id"]

        cached_inventories: dict[str, Any] = {}
        if self.inventory_cache_path is not None and self.inventory_cache_path.exists():
            payload = json.loads(self.inventory_cache_path.read_text(encoding="utf-8"))
            if payload.get("schema") != "efrm_measured_channel_inventory_v1":
                raise ValueError(f"unsupported inventory cache: {self.inventory_cache_path}")
            cached_inventories = dict(payload.get("datasets", {}))

        groups: dict[str, dict[tuple[tuple[str, ...], tuple[str, ...]], dict[str, list[int]]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for record_id, indices in records.items():
            dataset_id = record_dataset[record_id]
            cached = cached_inventories.get(dataset_id)
            if cached is None:
                representative = dataset[indices[0]]
                cached = {
                    "dataset_id": dataset_id,
                    "eeg_channel_names": list(representative["eeg_channel_names"]),
                    "fnirs_location_names": list(representative["fnirs_location_names"]),
                }
                cached_inventories[dataset_id] = cached
            inventory = (
                tuple(str(value) for value in cached["eeg_channel_names"]),
                tuple(str(value) for value in cached["fnirs_location_names"]),
            )
            groups[dataset_id][inventory][record_id] = indices
        if self.inventory_cache_path is not None:
            self.inventory_cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": "efrm_measured_channel_inventory_v1",
                "adapter": dataset.adapter.manifest(),
                "datasets": cached_inventories,
            }
            temporary = self.inventory_cache_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            temporary.replace(self.inventory_cache_path)
        self.groups = {
            dataset_id: {inventory: dict(rows) for inventory, rows in inventories.items()}
            for dataset_id, inventories in groups.items()
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.dataset.set_epoch(epoch)

    def _inventory_batches(
        self,
        records: Mapping[str, Sequence[int]],
        rng: random.Random,
    ) -> list[list[int]]:
        queues = {key: list(values) for key, values in records.items()}
        for values in queues.values():
            rng.shuffle(values)
        active = list(queues)
        rng.shuffle(active)
        buffer: list[int] = []
        batches: list[list[int]] = []
        while active:
            next_active: list[str] = []
            for record_id in active:
                values = queues[record_id]
                if values:
                    buffer.append(values.pop())
                if values:
                    next_active.append(record_id)
                if len(buffer) == self.batch_size:
                    batches.append(buffer)
                    buffer = []
            active = next_active
            rng.shuffle(active)
        if buffer and not self.drop_last and len(buffer) >= self.minimum_batch_size:
            batches.append(buffer)
        return batches

    def _batches(self) -> dict[str, list[list[int]]]:
        output: dict[str, list[list[int]]] = {}
        for dataset_position, dataset_id in enumerate(sorted(self.groups)):
            rng = random.Random(self.seed + self.epoch * 10_007 + dataset_position)
            batches: list[list[int]] = []
            inventories = list(self.groups[dataset_id].values())
            rng.shuffle(inventories)
            for records in inventories:
                batches.extend(self._inventory_batches(records, rng))
            rng.shuffle(batches)
            if batches:
                output[dataset_id] = batches
        return output

    def __iter__(self) -> Iterator[list[int]]:
        by_dataset = self._batches()
        if not by_dataset:
            return
        dataset_ids = sorted(by_dataset)
        maximum = max(len(value) for value in by_dataset.values())
        for position in range(maximum):
            for dataset_id in dataset_ids:
                batches = by_dataset[dataset_id]
                yield batches[position % len(batches)]

    def __len__(self) -> int:
        by_dataset = self._batches()
        return 0 if not by_dataset else max(len(value) for value in by_dataset.values()) * len(by_dataset)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": SAMPLER_SCHEMA,
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
            "minimum_batch_size": self.minimum_batch_size,
            "negative_sampling": "record_diverse_within_measured_channel_inventory",
            "inventory_count_by_dataset": {
                dataset_id: len(inventories) for dataset_id, inventories in self.groups.items()
            },
            "record_count_by_dataset": {
                dataset_id: sum(len(records) for records in inventories.values())
                for dataset_id, inventories in self.groups.items()
            },
            "inventory_cache_path": (
                None if self.inventory_cache_path is None else str(self.inventory_cache_path.resolve())
            ),
        }


def collate_efrm_pairs(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(samples) < 2:
        raise ValueError("EFRM contrastive batches require at least two synchronized pairs")
    if not all(bool(sample["admitted"]) for sample in samples):
        rejected = [sample["sample_id"] for sample in samples if not sample["admitted"]]
        raise ValueError(f"Batch contains samples without full analysis support: {rejected[:3]}")
    eeg_shapes = {tuple(sample["eeg"].shape) for sample in samples}
    fnirs_shapes = {tuple(sample["fnirs"].shape) for sample in samples}
    if len(eeg_shapes) != 1 or len(fnirs_shapes) != 1:
        raise ValueError("Record-grouped EFRM batches must have one channel inventory")
    metadata_keys = (
        "sample_id", "dataset_id", "subject", "record_id", "join_key",
        "task_namespace", "condition", "crop_start_s", "duration_s",
    )
    result: dict[str, Any] = {
        "eeg": torch.stack([sample["eeg"] for sample in samples]),
        "fnirs": torch.stack([sample["fnirs"] for sample in samples]),
        "eeg_patch_valid": torch.stack([sample["eeg_patch_valid"] for sample in samples]),
        "fnirs_patch_valid": torch.stack([sample["fnirs_patch_valid"] for sample in samples]),
        "positive_pair_mask": torch.eye(len(samples), dtype=torch.bool),
        "adapter_state": [dict(sample["adapter_state"]) for sample in samples],
    }
    for key in metadata_keys:
        result[key] = [sample[key] for sample in samples]
    return result
