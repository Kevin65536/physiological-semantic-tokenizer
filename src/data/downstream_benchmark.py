"""Public v5 measurement inventory and subject folds for DOWNSTREAM-REBENCH-v1.

Inventory construction reads the central manifest and event index only. Signal
arrays are opened lazily by the existing unified loader, without normalization
or fitted state. Global inventory indices are stable across modality profiles.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from urllib.parse import quote

import numpy as np

from .clean_physiology_cache import (
    CleanPhysiologyCacheIndex,
    DEFAULT_CLEAN_CACHE_ROOT,
    MEASUREMENT_CACHE_STORAGE,
)
from .event_alignment import EVENT_ALIGNMENT_SCHEMA, window_within_alignment_support
from .unified_physiology import (
    DEFAULT_ADMISSIBLE_ALIGNMENT_CASES,
    ChannelGeometryIndex,
    SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V4,
    UnifiedPhysiologyWindowDataset,
    UnifiedWindowRef,
    canonical_label,
    refed_continuous_target_window,
)


INVENTORY_SCHEMA = "downstream_rebench_inventory_v1"
SPLIT_SCHEMA = "downstream_rebench_subject_folds_v1"
PROCESSING_SCHEMA = "physiology_measurement_alignment_v5"
SPLIT_SEED = 20261016
TASKS: dict[str, dict[str, Any]] = {
    "motor_imagery": dict(dataset_id="eeg_fnirs_single_trial", classes=["LMI", "RMI"], duration_s=8.0),
    "mental_arithmetic": dict(dataset_id="eeg_fnirs_single_trial", classes=["MA", "BL"], duration_s=8.0),
    "wg": dict(dataset_id="simultaneous_eeg_nirs", classes=["WG", "BL"], duration_s=8.0),
    "nback": dict(dataset_id="simultaneous_eeg_nirs", classes=["0-back session", "2-back session", "3-back session"], duration_s=8.0),
    "dsr": dict(dataset_id="simultaneous_eeg_nirs", classes=["Go", "No-go"], duration_s=2.0),
    "visual": dict(dataset_id="visual_cognitive_motivation", classes=["RR", "RF", "FF", "FR"], duration_s=8.0),
    "refed_regression": dict(dataset_id="refed", classes=[], duration_s=20.0),
}

# These broad scalp panels are selected from native channel metadata, before any
# downstream score is computed. No channel substitution or duplication is used.
EEG_ONLY_CHANNELS = {
    "eeg_fnirs_single_trial": ["AFp1", "AFp2", "F7", "F3", "F4", "F8", "FCC3h", "FCC4h", "CCP3h", "CCP4h", "Cz", "Pz", "P3", "P4", "POO1", "POO2"],
    "simultaneous_eeg_nirs": ["Fp1", "Fp2", "F1", "F2", "FC1", "FC2", "C3", "C4", "Cz", "CP1", "CP2", "P3", "P4", "Pz", "O1", "O2"],
    "visual_cognitive_motivation": ["Fp1", "Fp2", "F3", "F4", "FC3", "FC4", "C3", "C4", "CP3", "CP4", "P3", "P4", "O1", "O2", "Fz", "Cz"],
    "refed": ["FP1", "FP2", "F3", "F4", "FC3", "FC4", "C3", "C4", "CP3", "CP4", "P3", "P4", "O1", "O2", "FZ", "CZ"],
}


def _load_object(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else json.loads(Path(value).read_text(encoding="utf-8"))


def _validate_cache(index: CleanPhysiologyCacheIndex) -> None:
    manifest = index.cache_manifest
    if (manifest.get("processing_schema") != PROCESSING_SCHEMA
            or manifest.get("motion_method") != "none"
            or manifest.get("storage") != MEASUREMENT_CACHE_STORAGE):
        raise ValueError("DOWNSTREAM-REBENCH-v1 requires v5 none measurement arrays")
    for record in index.records:
        row = record.manifest
        if (row.get("processing_schema") != PROCESSING_SCHEMA
                or row.get("motion_method") != "none"
                or row.get("storage") != MEASUREMENT_CACHE_STORAGE):
            raise ValueError(f"Incompatible measurement record: {record.join_key}")


def canonical_sample_id(dataset_id: str, subject_id: str, record_id: str,
                        task_id: str, event_id: int | str, start_s: float,
                        duration_s: float) -> str:
    """Encode the native join key and half-open event-relative time support."""
    values = [dataset_id, subject_id, record_id, task_id, str(event_id)]
    identity = "|".join(quote(str(value), safe="_-.") for value in values)
    return f"{identity}|start_us={round(start_s * 1e6)}|stop_us={round((start_s + duration_s) * 1e6)}|measurement"


def _task_for_event(dataset_id: str, event: Mapping[str, Any]) -> str | None:
    if dataset_id == "visual_cognitive_motivation":
        return "visual"
    if dataset_id == "refed":
        return "refed_regression" if event.get("event_type") == "video_segment_with_continuous_labels" else None
    task = str(event.get("metadata", {}).get("task", ""))
    return task if task in TASKS and TASKS[task]["dataset_id"] == dataset_id else None


def _recorded_sample_count(total: int, anchor_ms: float, offset_s: float,
                           duration_s: float, rate_hz: float) -> int:
    start = round((anchor_ms / 1000.0 + offset_s) * rate_hz)
    stop = start + round(duration_s * rate_hz)
    return max(0, min(stop, total) - max(start, 0))


def build_inventory(cache_root: str | Path = DEFAULT_CLEAN_CACHE_ROOT, *,
                    geometry_root: str | Path | None = None) -> dict[str, Any]:
    """Freeze the common seven-task denominator using permitted metadata only.

    Visual probe exports are joined by native epoch into a single event. Missing
    probe support remains visible in masks rather than deleting the event. REFED
    windows have a 20-second stride and retain the final partial input/target.
    """
    root = Path(cache_root).resolve()
    index = CleanPhysiologyCacheIndex(root)
    _validate_cache(index)
    geometry = Path(geometry_root).resolve() if geometry_root is not None else root / "channel_geometry"
    if not (geometry / "channels.jsonl").is_file():
        raise FileNotFoundError(f"Missing explicit geometry sidecar: {geometry / 'channels.jsonl'}")
    geometry_index = ChannelGeometryIndex(root, geometry_root=geometry)
    native_records = {record.join_key: record for record in index.records}
    records: dict[str, Any] = {}
    samples_by_id: dict[str, Any] = {}
    exclusions: list[dict[str, Any]] = []
    task_record_keys: dict[str, set[str]] = defaultdict(set)
    for record in index.records:
        if record.dataset_id == "refed" and record.signal_branch != "hbo_hbr":
            continue
        events = index.events_by_join_key.get(record.join_key, [])
        if not events:
            exclusions.append(dict(join_key=record.join_key, reason="no_indexed_events"))
            continue
        reports = index.reports_by_join_key.get(record.join_key, [])
        if (not reports or any(r.get("alignment_case") not in DEFAULT_ADMISSIBLE_ALIGNMENT_CASES
                               or r.get("label_sequence_match") is False for r in reports)):
            exclusions.append(dict(join_key=record.join_key, reason="unadmitted_alignment",
                                   event_count=len(events), alignment_cases=[r.get("alignment_case") for r in reports]))
            continue
        meta = record.manifest["measurement"]
        eeg_names = list(meta["eeg_channel_names"])
        fnirs_names = list(meta["fnirs_channel_names"])
        if len(eeg_names) != len(set(eeg_names)) or len(fnirs_names) != len(set(fnirs_names)):
            raise ValueError(f"Duplicate real channel identities: {record.join_key}")
        records[record.join_key] = dict(
            dataset_id=record.dataset_id, subject_id=record.canonical_subject_id,
            record_id=record.base_record_id, path=str(record.npz_path.resolve()),
            channel_names=eeg_names, fnirs_channel_names=fnirs_names,
            array_shapes=record.manifest["array_shapes"],
            preprocessing_scope=record.manifest.get("preprocessing_scope"),
            source_files=record.manifest.get("source_files", []),
            eeg_reference=meta["eeg_preprocessing_state"].get("reference", "unreported"),
            eeg_unit=meta["eeg_preprocessing_state"].get("canonical_unit"),
            fnirs_unit=meta["fnirs_preprocessing_state"].get("canonical_unit"),
        )
        for event in events:
            task_id = _task_for_event(record.dataset_id, event)
            if task_id is None:
                continue
            task = TASKS[task_id]
            label = canonical_label(event, record.dataset_id)["condition"]
            if task["classes"] and label not in task["classes"]:
                exclusions.append(dict(join_key=record.join_key, event_index=event["event_index"],
                                       reason="unknown_task_label", label=label))
                continue
            duration = task["duration_s"]
            if task_id == "refed_regression":
                event_duration = float(event["duration_ms"]) / 1000.0
                starts = np.arange(0.0, event_duration, duration).tolist()
            else:
                starts = [0.0]
            for start in starts:
                if not window_within_alignment_support(event, start, duration):
                    exclusions.append(dict(join_key=record.join_key, event_index=event["event_index"],
                                           start_s=start, reason="unverified_concatenation_interval"))
                    continue
                if event.get("eeg_time_ms") is None or event.get("fnirs_time_ms") is None:
                    raise ValueError(f"Event lacks paired clocks: {record.join_key}")
                native_record = record.base_record_id
                event_id = event["event_index"]
                probe = None
                if task_id == "visual":
                    match = re.search(r"_Probe([12])$", native_record, flags=re.IGNORECASE)
                    if match is None or event.get("metadata", {}).get("epoch_id") is None:
                        raise ValueError(f"Visual probe/epoch identity is missing: {record.join_key}")
                    probe = f"Probe{match.group(1)}"
                    native_record = native_record[:match.start()]
                    event_id = event["metadata"]["epoch_id"]
                sample_id = canonical_sample_id(record.dataset_id, record.canonical_subject_id,
                                                native_record, task_id, event_id, start, duration)
                reference = dict(join_key=record.join_key, event_index=int(event["event_index"]), probe=probe,
                                 eeg_time_ms=float(event["eeg_time_ms"]), fnirs_time_ms=float(event["fnirs_time_ms"]))
                reference["recorded_sample_counts"] = {
                    modality: _recorded_sample_count(record.manifest["array_shapes"][modality][0],
                        reference[f"{modality}_time_ms"], start, duration, rate)
                    for modality, rate in (("eeg", 200.0), ("fnirs", 10.0))}
                if sample_id in samples_by_id:
                    sample = samples_by_id[sample_id]
                    if task_id != "visual" or sample["label"] != label:
                        raise ValueError(f"Duplicate/conflicting native sample: {sample_id}")
                    if any(r["probe"] == probe for r in sample["event_refs"]):
                        raise ValueError(f"Duplicate Visual probe: {sample_id}")
                    if abs(sample["event_refs"][0]["eeg_time_ms"] - reference["eeg_time_ms"]) > 5.0:
                        raise ValueError(f"Visual probes disagree on native EEG event: {sample_id}")
                    sample["event_refs"].append(reference)
                else:
                    sample = dict(sample_id=sample_id, dataset_id=record.dataset_id,
                                  task_id=task_id, subject_id=record.canonical_subject_id,
                                  record_id=native_record, event_id=event_id, window_start_s=start,
                                  window_duration_s=duration, label=label,
                                  target=task["classes"].index(label) if task["classes"] else None,
                                  event_refs=[reference])
                    if task_id == "refed_regression":
                        target = refed_continuous_target_window(event, window_start_s=start,
                                                                window_duration_s=duration, target_sample_rate_hz=1.0)
                        sample["target_valid_counts"] = target["valid_mask"].sum(axis=1).tolist()
                        sample["partial_window"] = bool(start + duration > event_duration + 1e-6)
                    samples_by_id[sample_id] = sample
                task_record_keys[task_id].add(record.join_key)
    samples = sorted(samples_by_id.values(), key=lambda row: row["sample_id"])
    tasks: dict[str, Any] = {}
    for task_id, spec in TASKS.items():
        rows = [row for row in samples if row["task_id"] == task_id]
        if not rows:
            tasks[task_id] = dict(spec, sample_count=0, subject_count=0, profiles={})
            continue
        manifests = [records[key] for key in sorted(task_record_keys[task_id])]
        eeg_names = manifests[0]["channel_names"]
        fnirs_names = manifests[0]["fnirs_channel_names"]
        if any(row["channel_names"] != eeg_names or row["fnirs_channel_names"] != fnirs_names for row in manifests):
            raise ValueError(f"Task does not have one common real sensor panel: {task_id}")
        eeg_only = EEG_ONLY_CHANNELS[spec["dataset_id"]]
        if not set(eeg_only).issubset(eeg_names):
            raise ValueError(f"Fixed EEG-only panel is unavailable: {task_id}")
        if task_id == "visual":
            fnirs_names = [f"{probe}:{name}" for probe in ("Probe1", "Probe2") for name in fnirs_names]
        locations = list(dict.fromkeys(re.sub(r"_(HbO|HbR)$", "", name) for name in fnirs_names))
        ordered_fnirs = [f"{location}_{component}" for component in ("HbO", "HbR") for location in locations]
        if set(ordered_fnirs) != set(fnirs_names):
            raise ValueError(f"fNIRS panel lacks paired real HbO/HbR identities: {task_id}")
        fnirs_names = ordered_fnirs
        representative = native_records[sorted(task_record_keys[task_id])[0]]
        eeg_geometry = geometry_index.for_channels(record=representative, modality="eeg", channel_names=eeg_names)
        fnirs_geometry_by_name = {}
        for record_key in sorted(task_record_keys[task_id]):
            source_record = native_records[record_key]
            probe = re.search(r"_Probe([12])$", source_record.base_record_id, flags=re.IGNORECASE)
            source_names = records[record_key]["fnirs_channel_names"]
            source_rows = geometry_index.for_channels(record=source_record, modality="fnirs", channel_names=source_names)
            for name, source_row in zip(source_names, source_rows):
                name = f"Probe{probe.group(1)}:{name}" if task_id == "visual" and probe else name
                fnirs_geometry_by_name.setdefault(name, dict(source_row, channel_name=name))
            if task_id != "visual" or len(fnirs_geometry_by_name) == len(fnirs_names):
                break
        fnirs_geometry = [fnirs_geometry_by_name[name] for name in fnirs_names]
        def geometry_fields(eeg_panel: Sequence[str], hb_panel: Sequence[str]) -> dict[str, Any]:
            eeg_rows = [eeg_geometry[eeg_names.index(name)] for name in eeg_panel]
            hb_rows = [fnirs_geometry[fnirs_names.index(name)] for name in hb_panel]
            return dict(eeg_geometry=[[row["x"], row["y"]] for row in eeg_rows],
                        fnirs_geometry=[[row["x"], row["y"]] for row in hb_rows],
                        geometry_provenance=dict(eeg=eeg_rows, fnirs=hb_rows))
        common = dict(sampling_rate_hz=200.0, fnirs_sampling_rate_hz=10.0,
                      input_samples=round(spec["duration_s"] * 200),
                      fnirs_samples=round(spec["duration_s"] * 10),
                      num_classes=len(spec["classes"]), geometry_root=str(geometry))
        profiles = {
            "eeg_only": dict(common, channel_names=list(eeg_only), fnirs_channel_names=[],
                             fnirs_location_names=[], **geometry_fields(eeg_only, [])),
            "multimodal": dict(common, channel_names=list(eeg_names), fnirs_channel_names=list(fnirs_names),
                               fnirs_location_names=locations, **geometry_fields(eeg_names, fnirs_names)),
        }
        tasks[task_id] = dict(spec, sample_count=len(rows), subject_count=len({r["subject_id"] for r in rows}),
                             class_counts=dict(Counter(r["label"] for r in rows)) if spec["classes"] else {},
                             target_names=["valence", "arousal"] if task_id == "refed_regression" else [],
                             target_shape=[2, 20] if task_id == "refed_regression" else [],
                             target_sample_rate_hz=1.0 if task_id == "refed_regression" else None,
                             support_summary=dict(
                                 full_recorded_eeg_windows=sum(all(ref["recorded_sample_counts"]["eeg"] == common["input_samples"]
                                     for ref in row["event_refs"]) for row in rows),
                                 full_recorded_fnirs_windows=sum(all(ref["recorded_sample_counts"]["fnirs"] == common["fnirs_samples"]
                                     for ref in row["event_refs"]) and (task_id != "visual" or len(row["event_refs"]) == 2) for row in rows),
                                 missing_visual_probe_samples=sum(len(row["event_refs"]) < 2 for row in rows) if task_id == "visual" else 0,
                                 eeg_geometry_missing=sum(any(value is None for value in xy) for xy in profiles["multimodal"]["eeg_geometry"]),
                                 fnirs_geometry_missing=sum(any(value is None for value in xy) for xy in profiles["multimodal"]["fnirs_geometry"]),
                                 channel_supported_flags="not_inspected_metadata_only"),
                             profiles=profiles)
    return dict(schema=INVENTORY_SCHEMA, cache_root=str(root), geometry_root=str(geometry),
                processing_schema=PROCESSING_SCHEMA, motion_method="none", output_coordinate="measurement",
                event_alignment_schema=EVENT_ALIGNMENT_SCHEMA, tasks=tasks, records=records,
                samples=samples, exclusions=exclusions,
                information_support=dict(encoding="event-relative task duration",
                    reference_prefix_s=0.0, preprocessing="offline continuous record, two-sided filtering and resampling",
                    processing_record_scope="record manifest preprocessing_scope and array_shapes",
                    normalization="none; all learned transforms fit the applicable training partition only",
                    visual_probes="concatenate real probe channels; one prediction per native epoch"))


def make_splits(inventory: Mapping[str, Any] | str | Path, *, seed: int = SPLIT_SEED,
                outer_folds: int = 5, inner_folds: int = 3) -> dict[str, Any]:
    """Create deterministic dataset-shared subject folds and global sample indices."""
    inventory = _load_object(inventory)
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise ValueError("Unsupported downstream inventory schema")
    if outer_folds != 5 or inner_folds != 3 or seed != SPLIT_SEED:
        raise ValueError("DOWNSTREAM-REBENCH-v1 freezes 5 outer/3 inner folds and seed 20261016")
    samples = inventory["samples"]
    datasets: dict[str, Any] = {}
    tasks: dict[str, Any] = {}
    for dataset_id in sorted({row["dataset_id"] for row in samples}):
        subjects = sorted({row["subject_id"] for row in samples if row["dataset_id"] == dataset_id})
        if len(subjects) < outer_folds:
            raise ValueError(f"Insufficient subjects for fixed outer folds: {dataset_id}")
        permutation = np.random.default_rng(seed).permutation(subjects)
        outer_groups = [sorted(group.tolist()) for group in np.array_split(permutation, outer_folds)]
        folds = []
        for outer, evaluation in enumerate(outer_groups):
            train = sorted(set(subjects) - set(evaluation))
            if len(train) < inner_folds:
                raise ValueError(f"Insufficient subjects for fixed inner folds: {dataset_id}")
            inner_permutation = np.random.default_rng(seed + outer + 1).permutation(train)
            inner = []
            for number, group in enumerate(np.array_split(inner_permutation, inner_folds)):
                validation = sorted(group.tolist())
                inner.append(dict(fold=number, train_subjects=sorted(set(train) - set(validation)),
                                  validation_subjects=validation))
            folds.append(dict(fold=outer, train_subjects=train, eval_subjects=evaluation, inner_folds=inner))
        datasets[dataset_id] = dict(subjects=subjects, outer_folds=folds)
    for task_id, task in inventory["tasks"].items():
        task_indices = [i for i, row in enumerate(samples) if row["task_id"] == task_id]
        if not task_indices:
            continue
        def indices_for(subjects: Sequence[str]) -> list[int]:
            selected = set(subjects)
            return [i for i in task_indices if samples[i]["subject_id"] in selected]
        def require_coverage(indices: Sequence[int], context: str) -> None:
            if not indices:
                raise ValueError(f"Empty task support for {task_id} {context}")
            expected = set(range(len(task["classes"])))
            observed = {samples[i]["target"] for i in indices}
            if expected and not expected.issubset(observed):
                raise ValueError(f"Fixed folds lack complete class coverage: {task_id} {context}; revise dataset design before evaluation")
        folds = []
        for shared in datasets[task["dataset_id"]]["outer_folds"]:
            train, evaluation = indices_for(shared["train_subjects"]), indices_for(shared["eval_subjects"])
            require_coverage(train, "outer train")
            require_coverage(evaluation, "outer evaluation")
            inner = []
            for group in shared["inner_folds"]:
                fitting = indices_for(group["train_subjects"])
                validation = indices_for(group["validation_subjects"])
                require_coverage(fitting, "inner train")
                require_coverage(validation, "inner validation")
                inner.append(dict(group, train_indices=fitting, eval_indices=validation))
            folds.append(dict(shared, train_indices=train, eval_indices=evaluation, inner_folds=inner))
        tasks[task_id] = dict(dataset_id=task["dataset_id"], folds=folds)
    return dict(schema=SPLIT_SCHEMA, seed=seed, outer_fold_count=outer_folds, inner_fold_count=inner_folds,
                grouping="dataset-native subject; every session/record/event/video/probe follows subject",
                datasets=datasets, tasks=tasks)


def build_pretraining_inventory(inventory: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Derive the declared unlabeled 8 s source subset without reading signals.

    Classification source windows already have 8 s support; DSR is excluded.
    REFED contributes [0,8) and [8,16) within each original 20 s window when
    both measured clocks cover the full crop. Annotation availability is never
    consulted. Source dataset exclusion and subject partitioning belong to the
    pretraining worker and apply to every inherited subject/crop identity.
    """
    source = _load_object(inventory)
    if source.get("schema") != INVENTORY_SCHEMA or source.get("inventory_kind") == "pretraining":
        raise ValueError("Pretraining requires the original downstream inventory")
    result = deepcopy(dict(source))
    result["inventory_kind"] = "pretraining"
    policy = "declared_8s_source_subset_full_paired_recorded_support_v1"
    retained: dict[tuple[Any, ...], dict[str, Any]] = {}
    exclusions = []
    unused_intervals = []
    for parent in source["samples"]:
        if parent["task_id"] == "dsr":
            exclusions.append(dict(parent_sample_id=parent["sample_id"], duration_s=parent["window_duration_s"],
                                   reason="native_dsr_2s_not_expanded"))
            continue
        offsets = (0.0, 8.0) if parent["task_id"] == "refed_regression" else (0.0,)
        used = []
        for crop_offset in offsets:
            if crop_offset + 8.0 > float(parent["window_duration_s"]) + 1e-9:
                raise ValueError("A source crop would exceed its parent encoding support")
            start = float(parent["window_start_s"]) + crop_offset
            refs = deepcopy(parent["event_refs"])
            for reference in refs:
                shapes = source["records"][reference["join_key"]]["array_shapes"]
                reference["recorded_sample_counts"] = {
                    modality: _recorded_sample_count(shapes[modality][0], reference[f"{modality}_time_ms"],
                                                     start, 8.0, rate)
                    for modality, rate in (("eeg", 200.0), ("fnirs", 10.0))}
            if any(ref["recorded_sample_counts"] != {"eeg": 1600, "fnirs": 80} for ref in refs):
                exclusions.append(dict(parent_sample_id=parent["sample_id"], crop_offset_s=crop_offset,
                                       requested_duration_s=8.0,
                                       recorded_sample_counts=[ref["recorded_sample_counts"] for ref in refs],
                                       reason="incomplete_paired_recorded_8s_support"))
                continue
            # The key contains real modality clocks and input identity. Neither
            # label/class values nor task names participate in deduplication.
            signal_key = (parent["dataset_id"], parent["subject_id"], tuple(sorted(
                (ref["join_key"], ref.get("probe") or "",
                 round((ref["eeg_time_ms"] / 1000.0 + start) * 1e6),
                 round((ref["fnirs_time_ms"] / 1000.0 + start) * 1e6)) for ref in refs)), 8_000_000)
            used.append((crop_offset, crop_offset + 8.0))
            if signal_key in retained:
                retained[signal_key]["duplicate_parent_sample_ids"].append(parent["sample_id"])
                exclusions.append(dict(parent_sample_id=parent["sample_id"], crop_offset_s=crop_offset,
                                       reason="duplicate_native_signal_support"))
                continue
            row = deepcopy(parent)
            for key in ("target", "target_valid_counts", "partial_window"):
                row.pop(key, None)
            row.update(sample_id=f"{parent['sample_id']}|pretrain_crop_offset_us={round(crop_offset * 1e6)}|duration_us=8000000",
                       parent_sample_id=parent["sample_id"], crop_offset_s=crop_offset,
                       window_start_s=start, window_duration_s=8.0, event_refs=refs,
                       support_policy=policy, duplicate_parent_sample_ids=[])
            retained[signal_key] = row
        if parent["task_id"] == "refed_regression":
            # Count only available measured input time, independently of labels.
            available_s = min(float(parent["window_duration_s"]), *(
                _recorded_sample_count(source["records"][ref["join_key"]]["array_shapes"][modality][0],
                    ref[f"{modality}_time_ms"], float(parent["window_start_s"]),
                    float(parent["window_duration_s"]), rate) / rate
                for ref in parent["event_refs"] for modality, rate in (("eeg", 200.0), ("fnirs", 10.0))))
            cursor = 0.0
            for left, right in sorted(used):
                if left > cursor:
                    unused_intervals.append(dict(parent_sample_id=parent["sample_id"], dataset_id=parent["dataset_id"],
                        subject_id=parent["subject_id"], record_id=parent["record_id"],
                        parent_relative_start_s=cursor, parent_relative_stop_s=min(left, available_s),
                        duration_s=max(0.0, min(left, available_s) - cursor)))
                cursor = max(cursor, right)
            if available_s > cursor:
                unused_intervals.append(dict(parent_sample_id=parent["sample_id"], dataset_id=parent["dataset_id"],
                    subject_id=parent["subject_id"], record_id=parent["record_id"],
                    parent_relative_start_s=cursor, parent_relative_stop_s=available_s, duration_s=available_s - cursor))
    result["samples"] = sorted(retained.values(), key=lambda row: row["sample_id"])
    result["source_exclusions"] = exclusions
    result["unused_signal_intervals"] = unused_intervals
    result["pretraining_support"] = dict(policy=policy,
        selection="classification 8s; REFED crop offsets 0s/8s within the declared 20s parents; no DSR expansion",
        label_use="no labels or annotation masks used for crop admission, deduplication or model input",
        coverage="declared 8s source subset; not all continuous records",
        subject_partition="inherit dataset-native subject before deriving crops; all tasks/records/probes follow subject",
        dataset_exclusion="worker excludes the full target dataset before pretraining",
        unused_signal_interval_count=len(unused_intervals),
        unused_signal_duration_s=sum(row["duration_s"] for row in unused_intervals),
        excluded_reason_counts=dict(Counter(row["reason"] for row in exclusions)))
    for task_id, task in result["tasks"].items():
        rows = [row for row in result["samples"] if row["task_id"] == task_id]
        task.update(classes=[], duration_s=8.0, sample_count=len(rows),
                    subject_count=len({row["subject_id"] for row in rows}), class_counts={},
                    target_names=[], target_shape=[], target_sample_rate_hz=None)
        for profile in task["profiles"].values():
            profile.update(input_samples=1600, fnirs_samples=80, num_classes=0)
        task["support_summary"] = dict(full_recorded_eeg_windows=len(rows), full_recorded_fnirs_windows=len(rows),
            missing_visual_probe_samples=sum(len(row["event_refs"]) < 2 for row in rows) if task_id == "visual" else 0,
            channel_supported_flags="inherited; exact flags read lazily and retained as input masks")
    result["information_support"]["encoding"] = "8s source crops within frozen parent support"
    return result


class DownstreamBenchmarkDataset:
    """Lazy windows from a frozen inventory; does not fit any transformation.

    ``indices`` address the full inventory, including when ``task_id`` filters
    them. EEG-only samples expose an empty fNIRS tensor and do not return hidden
    Hb samples. The target mask is independent of input validity.
    """

    def __init__(self, inventory: Mapping[str, Any] | str | Path,
                 indices: Sequence[int] | None = None, *, task_id: str | None = None,
                 modality_profile: str = "multimodal") -> None:
        self.inventory = _load_object(inventory)
        if self.inventory.get("schema") != INVENTORY_SCHEMA:
            raise ValueError("Unsupported downstream inventory schema")
        if modality_profile not in ("eeg_only", "multimodal"):
            raise ValueError("modality_profile must be eeg_only or multimodal")
        self.modality_profile = modality_profile
        self.samples = self.inventory["samples"]
        candidates = range(len(self.samples)) if indices is None else indices
        self.indices = [int(i) for i in candidates if task_id is None or self.samples[int(i)]["task_id"] == task_id]
        self._loader = UnifiedPhysiologyWindowDataset(
            self.inventory["cache_root"], geometry_root=self.inventory["geometry_root"],
            eeg_signal_branch=SINGLE_TRIAL_EEG_ARTIFACT_SCHEMA_V4,
            include_event_types=set(), output_coordinate="measurement")
        _validate_cache(self._loader.index)
        self._records = {record.join_key: record for record in self._loader.index.records}
        self._events = {(str(event["join_key"]), int(event["event_index"])): event
                        for event in self._loader.index.events}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        global_index = self.indices[index]
        row = self.samples[global_index]
        profile = self.inventory["tasks"][row["task_id"]]["profiles"][self.modality_profile]
        parts = []
        refs = sorted(row["event_refs"], key=lambda ref: str(ref.get("probe") or ""))
        for reference in refs:
            record = self._records[reference["join_key"]]
            frozen = self.inventory["records"][reference["join_key"]]
            measurement = record.manifest["measurement"]
            if (str(record.npz_path.resolve()) != frozen["path"]
                    or record.manifest["array_shapes"] != frozen["array_shapes"]
                    or measurement["eeg_channel_names"] != frozen["channel_names"]
                    or measurement["fnirs_channel_names"] != frozen["fnirs_channel_names"]):
                raise ValueError("Measurement record identity differs from frozen inventory")
            event = self._events[(reference["join_key"], reference["event_index"])]
            if any(float(event[f"{modality}_time_ms"]) != reference[f"{modality}_time_ms"] for modality in ("eeg", "fnirs")):
                raise ValueError("Event clock differs from frozen inventory")
            if canonical_label(event, record.dataset_id)["condition"] != row["label"]:
                raise ValueError("Task label differs from frozen inventory")
            self._loader.window_duration_s = row["window_duration_s"]
            self._loader.windows = [UnifiedWindowRef(record=record, event=event, window_offset_s=row["window_start_s"])]
            parts.append((reference, self._loader[0]))
            if self.modality_profile == "eeg_only":
                break
        first = parts[0][1]
        eeg_indices = [first["channel_names"]["eeg"].index(name) for name in profile["channel_names"]]
        eeg = first["eeg"][eeg_indices]
        eeg_support = first["channel_valid_mask"]["eeg"][eeg_indices] & first["valid_mask"]["eeg"][None, :]
        fnirs_names = profile["fnirs_channel_names"]
        fnirs = np.zeros((len(fnirs_names), profile["fnirs_samples"]), dtype=np.float64)
        fnirs_support = np.zeros_like(fnirs, dtype=bool)
        fnirs_geometry: list[dict[str, Any]] = [{} for _ in fnirs_names]
        positions = {name: i for i, name in enumerate(fnirs_names)}
        for reference, part in parts:
            for source, name in enumerate(part["channel_names"]["fnirs"]):
                panel_name = f"{reference['probe']}:{name}" if reference["probe"] is not None else name
                if panel_name not in positions:
                    continue
                destination = positions[panel_name]
                fnirs[destination] = part["fnirs"][source]
                fnirs_support[destination] = part["channel_valid_mask"]["fnirs"][source] & part["valid_mask"]["fnirs"]
                fnirs_geometry[destination] = dict(part["channel_geometry"]["fnirs"][source], channel_name=panel_name)
        target_fields = {}
        if self.inventory.get("inventory_kind") == "pretraining":
            pass
        elif row["task_id"] == "refed_regression":
            event = self._events[(refs[0]["join_key"], refs[0]["event_index"])]
            target_data = refed_continuous_target_window(event, window_start_s=row["window_start_s"],
                                                        window_duration_s=row["window_duration_s"], target_sample_rate_hz=1.0)
            target, target_mask = target_data["values"], target_data["valid_mask"]
            target_time_s = target_data["time_s"]
            target_fields = dict(target=target, target_valid_mask=target_mask, target_time_s=target_time_s)
        else:
            target, target_mask, target_time_s = np.int64(row["target"]), np.bool_(True), None
            target_fields = dict(target=target, target_valid_mask=target_mask, target_time_s=target_time_s)
        source_identity = {name: row[name] for name in ("parent_sample_id", "crop_offset_s", "support_policy") if name in row}
        return dict(eeg=eeg, fnirs=fnirs, **target_fields,
                    valid_mask=dict(eeg=first["valid_mask"]["eeg"].copy(),
                        fnirs=fnirs_support.any(axis=0)),
                    channel_valid_mask=dict(eeg=eeg_support, fnirs=fnirs_support),
                    channel_names=dict(eeg=profile["channel_names"], fnirs=fnirs_names),
                    channel_geometry=dict(eeg=[first["channel_geometry"]["eeg"][i] for i in eeg_indices], fnirs=fnirs_geometry),
                    sample_rate_hz=dict(eeg=200.0, fnirs=10.0), unit=first["unit"],
                    global_index=global_index, sample_id=row["sample_id"], subject_id=row["subject_id"],
                    **source_identity, window_start_s=row["window_start_s"], window_duration_s=row["window_duration_s"],
                    subject=row["subject_id"], record_id=row["record_id"], event_id=row["event_id"],
                    task_id=row["task_id"], dataset_id=row["dataset_id"], modality_profile=self.modality_profile,
                    output_coordinate="measurement", preprocessing_state=first["preprocessing_state"],
                    alignment=[part["alignment"] for _, part in parts])
