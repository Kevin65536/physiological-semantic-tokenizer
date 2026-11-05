"""Versioned downstream measurements and task windows, independent of SSM fits.

The continuous record is stored once. Event windows and continuous REFED targets
are views of that record; no teacher output, fitted population scale or codebook
is part of this cache.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np
import yaml

from .clean_physiology_cache import (
    CleanPhysiologyCacheIndex,
    require_measurement_producer,
)
from .event_alignment import ADMISSIBLE_ALIGNMENT_CASES, window_within_alignment_support
from .functional_tokenizer import subject_partitions
from .registry import REGISTERED_DATASETS
from .unified_physiology import (
    RAW_DATASET_IDS, VISUAL_CHANNEL_IDENTITY_SCHEMA, ChannelGeometryIndex,
    _slice_window, _visual_eeg_name_map, canonical_label,
    load_native_eeg_record, preprocess_eeg_record_with_quality,
    refed_continuous_target_window,
)


SCHEMA = "physiology_downstream_input_v1"
EEG_SCHEMA = "native_voltage_measurement_200hz_v1"
VISUAL = "visual_cognitive_motivation"


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(_plain(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if (cfg.get("schema") != SCHEMA or set(cfg["datasets"]) != set(RAW_DATASET_IDS)
            or cfg["source_cache"] != "data/cache/physiology_semantic_clean_v5"
            or cfg["output_cache"] != "data/cache/physiology_downstream_v1"
            or cfg["eeg"]["signal_branch"] != "raw_with_ocular_artifact"
            or cfg["eeg"]["rate_hz"] != 200 or cfg["eeg"]["band_hz"] != [1, 45]
            or cfg["fnirs"] != {"input": "retained_v5_none", "rate_hz": 10, "band_hz": [.01, .2]}
            or cfg["split"]["train_fraction"] != .6 or cfg["split"]["validation_fraction"] != .2
            or cfg["refed"]["duration_s"] != 30 or cfg["refed"]["stride_s"] != 30
            or cfg["refed"]["target_rate_hz"] != 1 or not cfg["refed"]["include_partial"]):
        raise ValueError("Unchanged four-public-dataset downstream measurement contract required")
    return cfg


def record_signature(record):
    m = record.manifest
    return dict(join_key=record.join_key, source_files=m["source_files"],
                array_shapes=m["array_shapes"], processing_schema=m["processing_schema"],
                motion_method=m.get("motion_method"),
                fnirs_channels=m["measurement"]["fnirs_channel_names"],
                fnirs_preprocessing=m["measurement"]["fnirs_preprocessing_state"])


def _native_group(record):
    native = (re.sub(r"_Probe[12]$", "", record.base_record_id)
              if record.dataset_id == VISUAL else record.base_record_id)
    return f"{record.dataset_id}|{record.canonical_subject_id}|{native}"


def _record_directory(root, record):
    parts = [record.dataset_id, record.canonical_subject_id, record.base_record_id]
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", p) for p in parts):
        raise ValueError("Unsafe native record identity")
    return Path(root) / "records" / Path(*parts)


def _partitions(records, cfg, project_root):
    subjects = {ds: sorted({r.canonical_subject_id for r in records if r.dataset_id == ds})
                for ds in cfg["datasets"]}
    assigned = {}
    # Explicit dataset seeds are stable when another dataset is added.
    for ds, members in subjects.items():
        assigned.update(subject_partitions({ds: members}, cfg["split"]["dataset_seeds"][ds],
            cfg["split"]["train_fraction"], cfg["split"]["validation_fraction"]))
    previous = json.loads((project_root / cfg["split"]["retained_two_dataset_inventory"]).read_text())
    for ds, splits in previous["subjects"].items():
        if {s for values in splits.values() for s in values} != set(subjects[ds]):
            raise ValueError("Existing tokenizer cohort changed; preserve its recorded subject assignments")
        for split, members in splits.items():
            for subject in members:
                if assigned[(ds, subject)] != split:
                    raise ValueError("Dataset seed would change a retained subject assignment")
    return {ds: {split: sorted(s for (d, s), value in assigned.items() if d == ds and value == split)
                 for split in ("train", "validation", "test")} for ds in subjects}


def make_plan(project_root, cfg):
    """Freeze native identities, subject groups and all task supports without signals."""
    root = Path(project_root).resolve()
    index = CleanPhysiologyCacheIndex(root / cfg["source_cache"])
    records = [r for r in index.records if r.dataset_id in cfg["datasets"]
               and (r.dataset_id != "refed" or r.signal_branch == "hbo_hbr")]
    partitions = _partitions(records, cfg, root)
    lookup = {(ds, s): split for ds, values in partitions.items()
              for split, members in values.items() for s in members}
    mapping = _visual_eeg_name_map(root)
    visual_names = [mapping[k] for k in sorted(mapping, key=lambda k: int(k[1:]))]
    planned, samples, exclusions, groups = [], [], [], defaultdict(list)
    for r in records:
        require_measurement_producer(r.manifest)
        if (r.manifest["processing_schema"] != "physiology_measurement_alignment_v5"
                or r.manifest.get("motion_method") != "none"):
            raise ValueError("Retained v5/none Hb source required")
        admitted_root = (root / REGISTERED_DATASETS[r.dataset_id].default_root).resolve()
        for source in r.manifest["source_files"]:
            if not (root / source["path"]).resolve().is_relative_to(admitted_root):
                raise ValueError("Source lies outside its registered public dataset")
        split = lookup[(r.dataset_id, r.canonical_subject_id)]
        entry = dict(join_key=r.join_key, dataset=r.dataset_id, subject=r.canonical_subject_id,
                     record=r.base_record_id, split=split, native_group=_native_group(r),
                     source_signature=record_signature(r),
                     eeg_channels=visual_names if r.dataset_id == VISUAL else r.manifest["measurement"]["eeg_channel_names"],
                     fnirs_channels=r.manifest["measurement"]["fnirs_channel_names"])
        planned.append(entry); groups[entry["native_group"]].append(r.join_key)
        reports = index.reports_by_join_key.get(r.join_key, [])
        admitted = bool(reports) and all(x.get("alignment_case") in ADMISSIBLE_ALIGNMENT_CASES
                    and x.get("label_sequence_match") is not False for x in reports)
        for event in index.events_by_join_key.get(r.join_key, []):
            label = canonical_label(event, r.dataset_id)
            identity = dict(join_key=r.join_key, event_index=int(event["event_index"]),
                            dataset=r.dataset_id, subject=r.canonical_subject_id,
                            record=r.base_record_id, task=label["task"], split=split)
            if not admitted:
                exclusions.append(dict(identity, reason="nonadmitted_clock_alignment")); continue
            if event.get("eeg_time_ms") is None or event.get("fnirs_time_ms") is None:
                exclusions.append(dict(identity, reason="missing_paired_clock")); continue
            if r.dataset_id == "refed":
                stream = event["metadata"]["continuous_label_stream"]
                duration = min(r.manifest["array_shapes"]["eeg"][0]/200.,
                               r.manifest["array_shapes"]["fnirs"][0]/10.,
                               event["duration_ms"]/1000., stream["sample_count"]/stream["native_sample_rate_hz"])
                starts = np.arange(0., duration - 1e-9, cfg["refed"]["stride_s"])
                for start in starts:
                    target = refed_continuous_target_window(event, window_start_s=float(start), window_duration_s=30)
                    if not np.asarray(target["valid_mask"]).any():
                        exclusions.append(dict(identity, window_start_s=float(start), reason="no_valid_continuous_target")); continue
                    samples.append(dict(identity,
                        id=f"{r.join_key}|event={event['event_index']:05d}|start={start:.6f}|duration=30",
                        dependency_id=r.join_key, duration_s=30., eeg_start_s=float(start), hb_start_s=float(start),
                        supported_seconds=float(min(30, duration-start)), partial=bool(start+30>duration+1e-9),
                        event_relative_start_s=float(start), event=event,
                        target_kind="continuous_sequence_regression", target_names=["valence", "arousal"]))
                continue
            spec = cfg["tasks"].get(label["task"])
            if spec is None:
                exclusions.append(dict(identity, reason="task_outside_contract")); continue
            if label["condition"] not in spec["classes"]:
                exclusions.append(dict(identity, reason="unknown_or_unsupported_label")); continue
            offset, length = float(spec["offset_s"]), float(spec["duration_s"])
            es, hs = event["eeg_time_ms"]/1000 + offset, event["fnirs_time_ms"]/1000 + offset
            if not window_within_alignment_support(event, offset, length):
                exclusions.append(dict(identity, reason="crosses_unverified_clock_support")); continue
            if (min(es, hs) < 0 or es+length > r.manifest["array_shapes"]["eeg"][0]/200 + 1e-9
                    or hs+length > r.manifest["array_shapes"]["fnirs"][0]/10 + 1e-9):
                exclusions.append(dict(identity, reason="incomplete_recorded_window")); continue
            dependency = f"{entry['native_group']}|eeg_event_ms={event['eeg_time_ms']:.6f}"
            samples.append(dict(identity,
                id=f"{r.join_key}|event={event['event_index']:05d}|offset={offset:g}|duration={length:g}",
                dependency_id=dependency, duration_s=length, eeg_start_s=es, hb_start_s=hs,
                supported_seconds=length, partial=False, event_relative_start_s=offset, event=event,
                target_kind="classification", class_index=spec["classes"].index(label["condition"]),
                condition=label["condition"], target_event_time_s=-offset))
    if len({r["id"] for r in samples}) != len(samples):
        raise ValueError("Duplicate downstream sample identity")
    multiplicity = Counter(s["dependency_id"] for s in samples)
    for s in samples:
        s["dependency_views"] = multiplicity[s["dependency_id"]]
    return dict(schema=SCHEMA, configuration=cfg, source_cache=str(root/cfg["source_cache"]),
                subject_partitions=partitions, records=planned,
                groups=[dict(id=k, records=v) for k, v in sorted(groups.items())],
                samples=samples, exclusions=exclusions,
                input_support="offline_record_preprocessing_then_task_window",
                label_scope="target_event_or_dynamic_sequence_not_a_uniform_state_over_all_input_samples")


@lru_cache(maxsize=1)
def _source_index(root):
    index = CleanPhysiologyCacheIndex(Path(root))
    return {r.join_key: r for r in index.records}


def _save_array(path, values):
    tmp = Path(path).with_suffix(".partial.npy")
    np.save(tmp, values, allow_pickle=False)
    os.replace(tmp, path)


def _link(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        if not os.path.samefile(source, destination):
            raise ValueError("Cannot replace a retained cache array")
        return
    try:
        os.link(source, destination)
    except OSError:
        # A fresh cache may live on a different filesystem.
        tmp = destination.with_suffix(".partial.npy")
        shutil.copyfile(source, tmp); os.replace(tmp, destination)


def prepare_group(project_root, cfg, output, entries):
    """One native EEG read/preprocessing serves both Visual probe records."""
    import resource
    started = time.monotonic(); root, out = Path(project_root), Path(output)
    index = _source_index(str(root/cfg["source_cache"]))
    records = []
    for entry in entries:
        record = index[entry["join_key"]]
        if record_signature(record) != entry["source_signature"]:
            raise ValueError("Registered source changed after the input plan was frozen")
        records.append(record)
    if len({_native_group(r) for r in records}) != 1:
        raise ValueError("Only one native EEG support per producer")
    existing, pending = [], []
    for record in records:
        path = _record_directory(out, record)/"record.json"
        if path.exists():
            m = json.loads(path.read_text())
            if m.get("downstream_input_schema") != SCHEMA or m.get("eeg_input_schema") != EEG_SCHEMA:
                raise ValueError("Cannot reuse a different input version")
            existing.append(m)
        else:
            pending.append(record)
    result = list(existing)
    if pending:
        if existing:
            shared = Path(existing[0]["record_npz"])
            eeg_state = existing[0]["measurement"]["eeg_preprocessing_state"]
            eeg_names = existing[0]["measurement"]["eeg_channel_names"]
            eeg_shape = existing[0]["array_shapes"]["eeg"]
        else:
            native = load_native_eeg_record(root, records[0])
            if list(native.channel_names) != entries[0]["eeg_channels"]:
                raise ValueError("Native channel identities differ from the frozen plan")
            eeg, eeg_state, quality = preprocess_eeg_record_with_quality(native,
                signal_branch=cfg["eeg"]["signal_branch"], output_coordinate="measurement")
            eeg_names, eeg_shape = list(native.channel_names), list(eeg.shape)
            if (eeg.shape[0] != records[0].manifest["array_shapes"]["eeg"][0]
                    or not np.isfinite(eeg).all()):
                raise ValueError("Native EEG preprocessing changed the declared clock support")
            shared = _record_directory(out, pending[0]); shared.mkdir(parents=True, exist_ok=True)
            _save_array(shared/"eeg.npy", eeg)
            _save_array(shared/"eeg_supported_channels.npy", quality["processed_valid_mask"].all(axis=0))
            _save_array(shared/"eeg_bad_channel_mask.npy", quality["bad_channel_mask"])
            del eeg, quality, native
        for record in pending:
            directory = _record_directory(out, record); directory.mkdir(parents=True, exist_ok=True)
            for key in ("eeg", "eeg_supported_channels", "eeg_bad_channel_mask"):
                if directory != shared:
                    _link(shared/f"{key}.npy", directory/f"{key}.npy")
            for key in ("fnirs", "fnirs_supported_channels"):
                _link(record.npz_path/record.manifest["arrays"][key], directory/f"{key}.npy")
            m = deepcopy(record.manifest)
            m.update(record_npz=str(directory.resolve()), downstream_input_schema=SCHEMA,
                     eeg_input_schema=EEG_SCHEMA, source_cache_record=str(record.npz_path),
                     preprocessing_scope="offline_native_EEG_record_1_45Hz_200Hz_and_retained_v5_Hb_no_population_scaling",
                     created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            m["array_shapes"].update(eeg=eeg_shape, eeg_supported_channels=[len(eeg_names)],
                                     eeg_bad_channel_mask=[len(eeg_names)])
            m["measurement"].update(eeg_channel_names=eeg_names, eeg_preprocessing_state=deepcopy(eeg_state))
            m["arrays"] = {k:f"{k}.npy" for k in ("eeg", "fnirs", "eeg_supported_channels", "fnirs_supported_channels", "eeg_bad_channel_mask")}
            m["Hb_reuse"] = dict(source_cache=cfg["source_cache"], source_record=record.join_key,
                                operation="unchanged_array_link_or_copy")
            write_json(directory/"record.json", m); result.append(m)
    return dict(status="completed", group=entries[0]["native_group"], records=result,
                new_records=len(pending), seconds=time.monotonic()-started,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)


def prepare_sidecars(project_root, cfg, output):
    """Retain the event timing; join corrected EEG identities to named geometry."""
    from .channel_geometry import CHANNEL_GEOMETRY_SCHEMA, records_from_refed_paper_layout
    root, out = Path(project_root), Path(output)
    source = root/cfg["source_cache"]
    event_root = out/"event_index"; event_root.mkdir(parents=True, exist_ok=True)
    for name in ("events.jsonl", "alignment_reports.jsonl"):
        target = event_root/name
        original = source/"event_index"/name
        if target.exists():
            if target.read_bytes() != original.read_bytes():
                raise ValueError("Frozen event index changed")
        else:
            shutil.copyfile(original, target)
    event_manifest = json.loads((source/"event_index/event_manifest.json").read_text())
    write_json(event_root/"event_manifest.json", dict(
        schema=event_manifest["schema"], event_alignment_schema=event_manifest["event_alignment_schema"],
        source_event_index=str(source/"event_index"), signal_cache=str(out),
        provenance="event rows and clock admission unchanged; signals have a new EEG identity"))
    rows = [json.loads(line) for line in (source/"channel_geometry/channels.jsonl").read_text().splitlines()
            if line.strip() and json.loads(line)["dataset_id"] != "refed"]
    refed = root/REGISTERED_DATASETS["refed"].default_root
    rows.extend(r.to_dict() for r in records_from_refed_paper_layout(
        root/"src/data/assets/refed_paper_layout_v1.json", refed/"EEG_channels.csv", refed/"fNIRS_coordinates.csv"))
    geometry_root = out/"channel_geometry"; geometry_root.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    target = geometry_root/"channels.jsonl"
    if target.exists() and target.read_text() != text:
        raise ValueError("Frozen geometry changed")
    if not target.exists():
        target.write_text(text)
    write_json(geometry_root/"manifest.json", dict(schema=CHANNEL_GEOMETRY_SCHEMA,
        sources=[str(source/"channel_geometry"), "src/data/assets/refed_paper_layout_v1.json"],
        Visual_identity_source="Raw_EEG_channel_reference.xls; geometry joined by electrode name",
        refed_geometry="refed_paper_layout_2d_v1_not_individual_3d_registration"))


def verify_cache(cache_root):
    """Check every stored record and every actual sample, including partial labels."""
    dataset = DownstreamInputDataset(cache_root)
    source = _source_index(dataset.plan["source_cache"])
    errors = []; record_counts = Counter(); sample_counts = Counter(); partial = Counter()
    hb_same_inode = 0; shapes = defaultdict(set)
    for record in dataset.index.records:
        arrays = dataset.index.load_record_arrays(record)
        old = source[record.join_key]
        for key in ("eeg", "fnirs"):
            values = arrays[key]; support = arrays[key+"_supported_channels"]
            if (values.dtype != np.float64 or list(values.shape) != record.manifest["array_shapes"][key]
                    or support.dtype != bool or support.shape != (values.shape[1],)):
                errors.append(record.join_key+":"+key+":shape_or_dtype"); continue
            for begin in range(0, len(values), 50000):
                if not np.isfinite(values[begin:begin+50000, support]).all():
                    errors.append(record.join_key+":"+key+":nonfinite_supported_signal"); break
        parent = old.npz_path/old.manifest["arrays"]["fnirs"]
        if os.path.samefile(parent, record.npz_path/"fnirs.npy"):
            hb_same_inode += 1
        elif not np.array_equal(np.load(parent, mmap_mode="r"), arrays["fnirs"], equal_nan=True):
            errors.append(record.join_key+":Hb_changed")
        names = record.manifest["measurement"]["eeg_channel_names"]
        if record.dataset_id == VISUAL and (len(names) != 31 or "FCz" not in names):
            errors.append(record.join_key+":Visual_missing_FCz")
        record_counts[record.dataset_id] += 1
    for i, row in enumerate(dataset.samples):
        sample = dataset[i]
        task = sample["task"]
        for key, rate in (("eeg", 200), ("fnirs", 10)):
            if sample[key].shape[1] != round(row["duration_s"]*rate):
                errors.append(row["id"]+":"+key+":window_length")
            if not np.isfinite(sample[key][sample["valid_mask"][key]]).all():
                errors.append(row["id"]+":"+key+":invalid_sample")
            if any(not g["position_available"] for g in sample["channel_geometry"][key]):
                errors.append(row["id"]+":"+key+":missing_geometry")
        if row["target_kind"] == "continuous_sequence_regression":
            label = sample["label"]
            if label["values"].shape != (2, 30) or not label["valid_mask"].any():
                errors.append(row["id"]+":continuous_target")
        sample_counts[(sample["dataset"], task, sample["split"])] += 1
        partial[task] += int(row["partial"])
        shapes[task].add((sample["eeg"].shape, sample["fnirs"].shape))
    result = dict(schema=SCHEMA+"_verification", status="passed" if not errors else "failed",
        checked_records=len(dataset.index.records), checked_samples=len(dataset),
        record_counts=dict(record_counts),
        sample_counts=[dict(dataset=ds, task=t, split=s, samples=n) for (ds,t,s),n in sorted(sample_counts.items())],
        partial_by_task=dict(partial), Hb_byte_identical_links=hb_same_inode,
        shapes={k:sorted(v) for k,v in shapes.items()},
        distinct_dependency_units=len({r["dependency_id"] for r in dataset.samples}),
        exclusions=dict(Counter(r["reason"] for r in dataset.plan["exclusions"])),
        errors=errors, source="input_plan.json_and_cache_manifest.json_and_all_referenced_arrays")
    write_json(Path(cache_root)/"verification.json", result)
    if errors:
        raise ValueError(f"Downstream cache verification failed: {errors[:10]}")
    return result


class DownstreamInputDataset:
    """Read the frozen sample list and mmap measurements; never rebuild or refit."""
    def __init__(self, cache_root, *, split=None, task=None):
        self.root = Path(cache_root)
        self.plan = json.loads((self.root/"input_plan.json").read_text())
        if self.plan.get("schema") != SCHEMA:
            raise ValueError("Expected the versioned downstream input plan")
        self.index = CleanPhysiologyCacheIndex(self.root)
        if self.index.cache_manifest.get("downstream_input_schema") != SCHEMA:
            raise ValueError("Expected a completed downstream input cache")
        self.records = {r.join_key:r for r in self.index.records}
        if set(self.records) != {r["join_key"] for r in self.plan["records"]}:
            raise ValueError("The completed cache differs from its complete planned record population")
        if split is not None and split not in ("train", "validation", "test"):
            raise ValueError("Unknown subject partition")
        self.samples = [s for s in self.plan["samples"] if (split is None or s["split"] == split)
                        and (task is None or s["task"] == task)]
        self.geometry = ChannelGeometryIndex(self.root)
        self._arrays = {}
        self._geometry = {}  # Fixed record population; reuse metadata across random windows.
        assigned = {(ds, s): part for ds, parts in self.plan["subject_partitions"].items()
                    for part, members in parts.items() for s in members}
        for sample in self.samples:
            if assigned[(sample["dataset"], sample["subject"])] != sample["split"]:
                raise ValueError("Sample crosses its frozen subject partition")
            if sample["join_key"] not in self.records:
                raise ValueError("Planned sample lacks a completed measurement record")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        row = self.samples[item]; r = self.records[row["join_key"]]
        m = r.manifest["measurement"]
        if (r.manifest.get("eeg_input_schema") != EEG_SCHEMA
                or m["eeg_preprocessing_state"]["signal_branch"] != "raw_with_ocular_artifact"):
            raise ValueError("Downstream EEG input identity changed")
        if (r.dataset_id == VISUAL and m["eeg_preprocessing_state"].get("channel_identity", {}).get("schema")
                != VISUAL_CHANNEL_IDENTITY_SCHEMA):
            raise ValueError("Corrected Visual channel identity required")
        if r.join_key not in self._arrays:
            if len(self._arrays) >= 2:
                self._arrays.pop(next(iter(self._arrays)))
            self._arrays[r.join_key] = self.index.load_record_arrays(r)
        arrays = self._arrays[r.join_key]
        if r.join_key not in self._geometry:
            self._geometry[r.join_key] = {k:self.geometry.for_channels(record=r, modality=k,
                channel_names=m[k+"_channel_names"]) for k in ("eeg", "fnirs")}
        values, masks, times, alignment = {}, {}, {}, {}
        for modality, start, rate in (("eeg", row["eeg_start_s"], 200.), ("fnirs", row["hb_start_s"], 10.)):
            a = arrays[modality]
            if a.dtype != np.float64 or list(a.shape) != r.manifest["array_shapes"][modality]:
                raise ValueError("Cached array differs from its recorded shape/dtype")
            v, support = _slice_window(a, start*1000, row["duration_s"], rate)
            supported = arrays[modality+"_supported_channels"]
            mask = support[None, :] & supported[:, None]
            # REFED's signal, annotation and published duration share one support.
            valid_steps = min(len(support), int(np.ceil(row["supported_seconds"]*rate-1e-9)))
            mask[:, valid_steps:] = False
            v = np.where(mask, v, 0.)
            values[modality], masks[modality] = v, mask
            actual = round(start*rate)/rate
            times[modality] = actual + np.arange(v.shape[1])/rate
            alignment[modality] = dict(requested_start_s=start, actual_start_s=actual,
                                       rounding_error_s=actual-start)
        if row["target_kind"] == "classification":
            label = dict(kind="classification", namespace=f"{row['dataset']}:{row['task']}",
                         condition=row["condition"], class_index=row["class_index"],
                         target_event_time_s=row["target_event_time_s"])
        else:
            target = refed_continuous_target_window(row["event"], window_start_s=row["event_relative_start_s"],
                                                   window_duration_s=row["duration_s"])
            label = dict(kind=row["target_kind"], values=target["values"], valid_mask=target["valid_mask"],
                         names=target["target_names"], time_s=target["time_s"], rate_hz=1.)
        return dict(schema=SCHEMA, id=row["id"], dependency_id=row["dependency_id"],
                    dependency_views=row["dependency_views"], split=row["split"],
                    dataset=row["dataset"], subject=row["subject"], record=row["record"], task=row["task"],
                    eeg=values["eeg"], fnirs=values["fnirs"], valid_mask=masks, time_s=times,
                    sample_rate_hz={"eeg":200., "fnirs":10.}, alignment=alignment,
                    channel_names={k:m[k+"_channel_names"] for k in ("eeg", "fnirs")},
                    channel_geometry=deepcopy(self._geometry[r.join_key]),
                    unit={k:m[k+"_preprocessing_state"]["canonical_unit"] for k in ("eeg", "fnirs")},
                    label=label, partial=row["partial"], preprocessing_support="complete_native_record_offline",
                    source_identity=r.manifest["source_files"])
