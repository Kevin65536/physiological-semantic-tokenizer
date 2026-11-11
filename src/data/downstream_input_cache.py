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
SCHEMA_V2 = "physiology_downstream_input_v2"
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
    if cfg.get('schema') == SCHEMA_V2:
        owner=Path(__file__).resolve().parents[2]/'experiments/configs/physiology_semantic_tokenizer/downstream_input_v2.yaml'
        if cfg != yaml.safe_load(owner.read_text()):
            raise ValueError('Use the reviewed Visual-only v2 migration contract')
        return cfg
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
    if json.loads((Path(cache_root)/'input_plan.json').read_text()).get('schema') == SCHEMA_V2:
        return verify_visual_migration(cache_root)
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
        self.schema=self.plan.get('schema')
        if self.schema not in (SCHEMA,SCHEMA_V2):
            raise ValueError("Expected the versioned downstream input plan")
        self.index = CleanPhysiologyCacheIndex(self.root)
        if self.index.cache_manifest.get("downstream_input_schema") != self.schema:
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
        return dict(schema=self.schema, id=row["id"], dependency_id=row["dependency_id"],
                    dependency_views=row["dependency_views"], split=row["split"],
                    dataset=row["dataset"], subject=row["subject"], record=row["record"], task=row["task"],
                    eeg=values["eeg"], fnirs=values["fnirs"], valid_mask=masks, time_s=times,
                    sample_rate_hz={"eeg":200., "fnirs":10.}, alignment=alignment,
                    channel_names={k:m[k+"_channel_names"] for k in ("eeg", "fnirs")},
                    channel_geometry=deepcopy(self._geometry[r.join_key]),
                    unit={k:m[k+"_preprocessing_state"]["canonical_unit"] for k in ("eeg", "fnirs")},
                    label=label, partial=row["partial"], preprocessing_support="complete_native_record_offline",
                    source_identity=r.manifest["source_files"],
                    fnirs_normalization=deepcopy(r.manifest.get('fnirs_normalization')))


def _visual_selection(record):
    """Device agreement is checked before the signal array is opened."""
    from .event_alignment import read_visual_fnirs_exclusions
    files=[Path(s['path']) for s in record['source_files']]
    pair=[]
    for suffix in ['_Oxy.csv','_Deoxy.csv']:
        found=[p for p in files if p.name.endswith(suffix)]
        if len(found)!=1:raise ValueError('Exact paired Visual source files required')
        pair.append(read_visual_fnirs_exclusions(found[0]))
    if any(pair[0][k]!=pair[1][k] for k in ['channel_names','excluded_channels']):
        raise ValueError('Visual Oxy/Deoxy exclusion metadata disagree')
    names=record['measurement']['fnirs_channel_names']
    expected=[c+'_'+h for c in pair[0]['channel_names'] for h in ['HbO','HbR']]
    if names!=expected:raise ValueError('Visual cached/native channel identity mismatch')
    mask=np.load(Path(record['record_npz'])/record['arrays']['fnirs_supported_channels'],allow_pickle=False)
    if mask.dtype!=bool or mask.shape!=(48,):raise ValueError('Unexpected Visual support')
    supported=mask.reshape(24,2).all(1)
    good=[i for i,c in enumerate(pair[0]['channel_names']) if c not in pair[0]['excluded_channels'] and supported[i]]
    if not good:raise ValueError('No usable Visual Hb pairs')
    columns=[2*i+j for i in good for j in [0,1]]
    return dict(schema='visual_device_exclusion_v1',native_pair_count=24,
        retained_pair_indices=good,retained_columns=columns,
        retained_names=[names[i] for i in columns],device_excluded_channels=pair[0]['excluded_channels'],
        unsupported_channels=[c for c,v in zip(pair[0]['channel_names'],supported) if not v],
        source_headers=pair,rule='device_zero_intersection_existing_paired_support')


def visual_migration_plan(root,cfg):
    root=Path(root)
    base=json.loads((root/cfg['source_measurement']/'cache_manifest.json').read_text())
    down=json.loads((root/cfg['source_downstream']/'cache_manifest.json').read_text())
    checked=json.loads((root/cfg['source_downstream']/'verification.json').read_text())
    if base['execution']!='completed' or down['execution']!='completed' or checked['status']!='passed' or checked['errors']:
        raise ValueError('Completed verified sources required')
    for m in [base,down]:require_measurement_producer(m)
    if down['downstream_input_schema']!=SCHEMA:raise ValueError('Migration must start from downstream v1')
    sources={m['join_key']:m for m in base['records']};inputs={m['join_key']:m for m in down['records']}
    if set(sources)!=set(inputs):raise ValueError('Input and measurement populations differ')
    plan=json.loads((root/cfg['source_downstream']/'input_plan.json').read_text())
    selection={k:_visual_selection(m) for k,m in sources.items() if m['dataset_id']==VISUAL}
    # Freeze only previously admitted input windows, not SSM predictions or states.
    calibration_root=root/cfg['source_calibration_inputs']
    old_plan=json.loads((calibration_root/'plan.json').read_text())
    old_coords=json.loads((calibration_root/'coordinates.json').read_text())['individual']
    records={r['cache_key']:r for r in old_plan['records'] if r['dataset']==VISUAL}
    subjects=sorted({m['canonical_subject_id'] for m in sources.values() if m['dataset_id']==VISUAL})
    calibrations={}
    for sub in subjects:
        coordinate=old_coords[VISUAL+'__'+sub];windows=[]
        for source in coordinate['sources']:
            r=records[source['record_key']];w=next(w for w in r['cache_windows'] if w['key']==source['window_key'])
            key=r['join_key']
            if key not in selection or r['status']!='available':raise ValueError('Unadmitted calibration record')
            windows.append(dict(join_key=key,window_key=w['key'],start=round(w['hb_start_s']*10),
                steps=round(w['duration_s']*10),original_pairs=source['pairs']))
        calibrations[sub]=dict(windows=windows,old_factor=coordinate['hb_factor'],old_count=coordinate['training_samples'],
            split=coordinate['split'],scope='own_unlabelled_previously_admitted_90s_windows_with_overlap')
    return dict(schema=SCHEMA_V2,configuration=cfg,selection=selection,calibrations=calibrations,
        source_record_keys=sorted(sources),subject_partitions=plan['subject_partitions'],
        original_samples=len(plan['samples']),original_exclusions=len(plan['exclusions']))


def _migration_directory(root,record):
    return Path(root)/'records'/record['dataset_id']/record['canonical_subject_id']/record['base_record_id']


def _visual_migration_subject(payload):
    import resource
    root_s,cfg,sub,selection,calibration,source_records,downstream_records=payload
    root=Path(root_s);lower=root/cfg['output_measurement'];upper=root/cfg['output_cache']
    started=time.monotonic();done=upper/'tasks'/(sub+'.json')
    if done.exists():return json.loads(done.read_text())
    # Fail closed on changed header metadata before any Hb signal read.
    for key,m in source_records.items():
        if _visual_selection(m)!=selection[key]:raise ValueError('Visual source selection changed after planning')
    arrays={k:np.load(Path(m['record_npz'])/m['arrays']['fnirs'],mmap_mode='r',allow_pickle=False).reshape(-1,24,2)
            for k,m in source_records.items()}
    acc={k:[0,np.zeros(2),np.zeros(2)] for k in ['old','usable','excluded']}
    for w in calibration['windows']:
        x=arrays[w['join_key']][w['start']:w['start']+w['steps']]
        if len(x)!=w['steps'] or w['steps']<50:raise ValueError('Changed calibration time support')
        m=source_records[w['join_key']]
        support=np.load(Path(m['record_npz'])/m['arrays']['fnirs_supported_channels']).reshape(24,2).all(1)
        if int(support.sum())!=w['original_pairs']:raise ValueError('Prior calibration pair support changed')
        selected=np.zeros(24,bool);selected[selection[w['join_key']]['retained_pair_indices']]=True
        x=x-x[:50].mean(0)
        for label,mask in [('old',support),('usable',selected),('excluded',support & ~selected)]:
            v=x[:,mask].reshape(-1,2)
            if not np.isfinite(v).all():raise ValueError('Nonfinite admitted measurement')
            a=acc[label];a[0]+=len(v);a[1]+=v.sum(0);a[2]+=(v*v).sum(0)
    moments={}
    mean=acc['old'][1]/acc['old'][0];total=(acc['old'][2]-acc['old'][0]*mean**2).sum()
    for label,(n,s,q) in acc.items():
        if not n:raise ValueError('Empty calibration group')
        variance=q/n-(s/n)**2
        if min(variance)<=1e-24:raise ValueError('Degenerate subject Hb scale')
        common=float(np.sqrt(variance.mean()))
        moments[label]=dict(samples=n,component_sd=np.sqrt(variance).tolist(),common_sd=common,
            factor=cfg['normalization']['target_common_sd']/common,
            original_variance_fraction=float((q-2*mean*s+n*mean**2).sum()/total))
    np.testing.assert_allclose(moments['old']['factor'],calibration['old_factor'],rtol=1e-9)
    if moments['old']['samples']!=calibration['old_count']:raise ValueError('Old calibration count changed')
    factor=moments['usable']['factor'];normalization_path=lower/'normalization'/(sub+'.json')
    norm=dict(schema=cfg['normalization']['schema'],dataset=VISUAL,subject=sub,factor=factor,
        inverse_factor=1/factor,target_common_sd=.025,calibration=calibration,moments=moments,
        baseline='first_5s_reference_for_calibration_windows_only; continuous_cached_record_not_recentered',
        applies_to='both_HbO_HbR_all_records_same_subject',fitted_predictions_used=False)
    write_json(normalization_path,norm)
    lower_records=[];upper_records=[]
    for key,old in source_records.items():
        prior=downstream_records[key];select=selection[key];columns=select['retained_columns']
        lowdir=_migration_directory(lower,old);updir=_migration_directory(upper,old)
        lowdir.mkdir(parents=True,exist_ok=True);updir.mkdir(parents=True,exist_ok=True)
        for name in ['eeg','eeg_supported_channels','eeg_bad_channel_mask']:
            origin=Path(prior['record_npz'])/prior['arrays'][name]
            _link(origin,lowdir/(name+'.npy'));_link(origin,updir/(name+'.npy'))
        physical=arrays[key].reshape(len(arrays[key]),48)[:,columns]
        if not np.isfinite(physical).all():raise ValueError('Invalid compact usable record')
        _save_array(lowdir/'fnirs.npy',physical);_save_array(lowdir/'fnirs_supported_channels.npy',np.ones(len(columns),bool))
        _save_array(updir/'fnirs.npy',physical*factor);_link(lowdir/'fnirs_supported_channels.npy',updir/'fnirs_supported_channels.npy')
        lm=deepcopy(prior);lm.pop('downstream_input_schema',None)
        lm.update(record_npz=str(lowdir.resolve()),measurement_cache_version='physiology_semantic_clean_v6',
            fnirs_selection=select,migration_source=dict(measurement=str(old['record_npz']),EEG=str(prior['record_npz'])),
            preprocessing_scope='retained_v5_measurement; compact_usable_Visual_Hb; verified_native_identity_EEG')
        lm['arrays']={k:k+'.npy' for k in ['eeg','fnirs','eeg_supported_channels','fnirs_supported_channels','eeg_bad_channel_mask']}
        lm['array_shapes'].update(fnirs=list(physical.shape),fnirs_supported_channels=[len(columns)])
        lm['measurement']['fnirs_channel_names']=select['retained_names']
        lm['measurement']['fnirs_component_roles']=[n.rsplit('_',1)[1] for n in select['retained_names']]
        lm['measurement']['fnirs_preprocessing_state'].update(channel_selection='visual_device_exclusion_v1',scaling='none; measurement coordinates retained')
        lm['Hb_reuse']=dict(operation='exact_retained_usable_column_selection',source_record=key)
        um=deepcopy(lm);um.update(record_npz=str(updir.resolve()),downstream_input_schema=SCHEMA_V2,
            source_cache_record=str(lowdir.resolve()),fnirs_normalization=dict(schema=norm['schema'],source=str(normalization_path.resolve())),
            preprocessing_scope='compact_usable_Visual_Hb_times_frozen_subject_factor; no_new_centering')
        um['measurement']['fnirs_preprocessing_state'].update(canonical_unit='subject_normalized_Hb',
            scaling='one_subject_common_Hb_factor',normalization_source=str(normalization_path.resolve()),
            measurement_unit=lm['measurement']['fnirs_preprocessing_state']['canonical_unit'])
        um['Hb_reuse']=dict(operation='usable_measurement_times_frozen_subject_factor',source_record=key)
        write_json(lowdir/'record.json',lm);write_json(updir/'record.json',um)
        lower_records.append(str(lowdir/'record.json'));upper_records.append(str(updir/'record.json'))
    result=dict(status='completed',subject=sub,measurement_records=lower_records,downstream_records=upper_records,
        factor=factor,old_factor=calibration['old_factor'],multiplier=factor/calibration['old_factor'],
        seconds=time.monotonic()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(done,result);return result


def _migration_sidecars(source,destination):
    for folder in ['event_index','channel_geometry']:
        (destination/folder).mkdir(parents=True,exist_ok=True)
        for name in (source/folder).iterdir():
            if name.is_file():
                target=destination/folder/name.name
                if target.exists():
                    if target.read_bytes()!=name.read_bytes():raise ValueError('Sidecar changed')
                else:shutil.copyfile(name,target)


def migrate_visual_caches(root,cfg,stage,workers):
    import fcntl
    from concurrent.futures import ProcessPoolExecutor
    root=Path(root).resolve();low=root/cfg['output_measurement'];out=root/cfg['output_cache']
    if low.resolve() in [(root/cfg['source_measurement']).resolve(),(root/cfg['source_downstream']).resolve()] or out==low:
        raise ValueError('Fresh distinct cache generations required')
    low.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    with (out/'controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=visual_migration_plan(root,cfg)
        planpath=out/'migration_plan.json'
        if planpath.exists():
            if json.loads(planpath.read_text())!=_plain(plan):raise ValueError('Frozen migration plan changed')
        else:
            if (out/'cache_manifest.json').exists() or (low/'cache_manifest.json').exists():raise ValueError('Existing output without this migration plan')
            write_json(planpath,plan)
        if stage=='plan':return dict(stage='plan',visual_records=len(plan['selection']),subjects=len(plan['calibrations']),signal_reads=0)
        if stage=='verify':return verify_visual_migration(out)
        if not os.environ.get('INVOCATION_ID'):raise ValueError('Use durable user systemd for cache migration')
        base=json.loads((root/cfg['source_measurement']/'cache_manifest.json').read_text())
        olddown=json.loads((root/cfg['source_downstream']/'cache_manifest.json').read_text())
        sources={m['join_key']:m for m in base['records']};prior={m['join_key']:m for m in olddown['records']}
        selected=['S01','S06'] if stage=='pilot' else sorted(plan['calibrations'])
        payloads=[]
        for sub in selected:
            keys=[k for k in plan['selection'] if sources[k]['canonical_subject_id']==sub]
            payloads.append((str(root),cfg,sub,{k:plan['selection'][k] for k in keys},plan['calibrations'][sub],
                {k:sources[k] for k in keys},{k:prior[k] for k in keys}))
        started=time.monotonic();results=[]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for r in pool.map(_visual_migration_subject,payloads):
                results.append(r);print(json.dumps({k:r[k] for k in ['subject','multiplier','seconds','peak_rss_mib']}),flush=True)
                write_json(out/'progress.json',dict(stage=stage,completed_subjects=len(results),total_subjects=len(selected)))
        write_json(out/(stage+'_execution.json'),dict(status='completed',subjects=len(results),workers=workers,
            seconds=time.monotonic()-started,peak_worker_mib=max(r['peak_rss_mib'] for r in results),supervisor=os.environ.get('INVOCATION_ID')))
        if stage=='pilot':return dict(status='completed',subjects=len(results))
        # Other datasets retain the same bytes and semantics in both new roots.
        for key,m in sources.items():
            if m['dataset_id']==VISUAL:continue
            for target,original,is_down in [(low,m,False),(out,prior[key],True)]:
                directory=_migration_directory(target,original);directory.mkdir(parents=True,exist_ok=True)
                mm=deepcopy(original);mm['record_npz']=str(directory.resolve())
                for name,relative in original['arrays'].items():
                    _link(Path(original['record_npz'])/relative,directory/(name+'.npy'))
                    mm['arrays'][name]=name+'.npy'
                if is_down:
                    mm['downstream_input_schema']=SCHEMA_V2;mm['source_cache_record']=str(_migration_directory(low,m).resolve())
                else:mm['measurement_cache_version']='physiology_semantic_clean_v6'
                write_json(directory/'record.json',mm)
        _migration_sidecars(root/cfg['source_measurement'],low)
        _migration_sidecars(root/cfg['source_downstream'],out)
        lower_records=[json.loads((_migration_directory(low,m)/'record.json').read_text()) for m in sources.values()]
        upper_records=[json.loads((_migration_directory(out,m)/'record.json').read_text()) for m in sources.values()]
        newplan=json.loads((root/cfg['source_downstream']/'input_plan.json').read_text())
        newplan.update(schema=SCHEMA_V2,configuration=cfg,source_cache=str(low.resolve()),source_downstream=str((root/cfg['source_downstream']).resolve()))
        lowerlookup={r['join_key']:r for r in lower_records}
        for entry in newplan['records']:
            r=lowerlookup[entry['join_key']];entry['fnirs_channels']=r['measurement']['fnirs_channel_names']
            entry['source_signature']=dict(join_key=r['join_key'],source_files=r['source_files'],array_shapes=r['array_shapes'],
                processing_schema=r['processing_schema'],motion_method=r['motion_method'],fnirs_channels=r['measurement']['fnirs_channel_names'],
                fnirs_preprocessing=r['measurement']['fnirs_preprocessing_state'])
        write_json(out/'input_plan.json',newplan)
        for target,original,rows in [(low,base,lower_records),(out,olddown,upper_records)]:
            manifest=deepcopy(original);manifest.update(records=rows,record_count=len(rows),execution='completed',
                migration_contract=cfg,source_cache=str(root/cfg['source_measurement']) if target==low else str(low.resolve()))
            if target==low:manifest.update(measurement_cache_version='physiology_semantic_clean_v6',output_dir=str(low.resolve()))
            else:manifest['downstream_input_schema']=SCHEMA_V2
            write_json(target/'cache_manifest.json',manifest)
        return verify_visual_migration(out)


def verify_visual_migration(cache_root):
    out=Path(cache_root);migration=json.loads((out/'migration_plan.json').read_text());cfg=migration['configuration']
    root=out.resolve().parents[2];low=root/cfg['output_measurement']
    olddown=json.loads((root/cfg['source_downstream']/'cache_manifest.json').read_text())
    oldbase=json.loads((root/cfg['source_measurement']/'cache_manifest.json').read_text())
    original={r['join_key']:r for r in oldbase['records']};previous={r['join_key']:r for r in olddown['records']}
    dataset=DownstreamInputDataset(out);lower=CleanPhysiologyCacheIndex(low);lm={r.join_key:r for r in lower.records}
    oldplan=json.loads((root/cfg['source_downstream']/'input_plan.json').read_text())
    for key in ['samples','exclusions','subject_partitions','groups']:
        if dataset.plan[key]!=oldplan[key]:raise ValueError('Frozen '+key+' changed')
    counts=Counter();visual=[];samples=Counter();max_error=0.
    for record in dataset.index.records:
        m=record.manifest;old=original[record.join_key];prior=previous[record.join_key];bottom=lm[record.join_key]
        for key in ['eeg','eeg_supported_channels','eeg_bad_channel_mask']:
            if not os.path.samefile(record.npz_path/m['arrays'][key],Path(prior['record_npz'])/prior['arrays'][key]):raise ValueError('EEG changed')
        if record.dataset_id!=VISUAL:
            for key in ['fnirs','fnirs_supported_channels']:
                if not os.path.samefile(record.npz_path/m['arrays'][key],Path(prior['record_npz'])/prior['arrays'][key]):raise ValueError('Nonvisual signal changed')
                if not os.path.samefile(bottom.npz_path/bottom.manifest['arrays'][key],Path(old['record_npz'])/old['arrays'][key]):raise ValueError('Nonvisual measurement changed')
        else:
            expected=_visual_selection(old)
            if expected!=m['fnirs_selection'] or expected!=bottom.manifest['fnirs_selection']:raise ValueError('Selection mismatch')
            columns=expected['retained_columns'];raw=np.load(Path(old['record_npz'])/old['arrays']['fnirs'],mmap_mode='r')
            physical=np.load(bottom.npz_path/'fnirs.npy',mmap_mode='r');normalized=np.load(record.npz_path/'fnirs.npy',mmap_mode='r')
            norm=json.loads(Path(m['fnirs_normalization']['source']).read_text());factor=norm['factor']
            np.testing.assert_array_equal(physical,raw[:,columns]);np.testing.assert_allclose(normalized,physical*factor,rtol=1e-12,atol=1e-12)
            if m['measurement']['fnirs_channel_names']!=expected['retained_names'] or not np.isfinite(normalized).all():raise ValueError('Compact identities/values differ')
            np.testing.assert_array_equal(np.load(record.npz_path/'fnirs_supported_channels.npy'),np.ones(len(columns),bool))
            max_error=max(max_error,float(np.max(abs(normalized/factor-physical))))
            visual.append(dict(subject=record.canonical_subject_id,record=record.base_record_id,locations=len(columns)//2,
                channels=[n[:-4] for n in expected['retained_names'][::2]],factor=factor))
        counts[record.dataset_id]+=1
    for i,row in enumerate(dataset.samples):
        sample=dataset[i];samples[sample['dataset']]+=1
        for modality in ['eeg','fnirs']:
            value=sample[modality];mask=sample['valid_mask'][modality]
            if value.shape!=mask.shape or value.shape[0]!=len(sample['channel_names'][modality]) or not np.isfinite(value[mask]).all():raise ValueError('Invalid consumer sample')
            if len(sample['channel_geometry'][modality])!=value.shape[0] or any(not g['position_available'] for g in sample['channel_geometry'][modality]):raise ValueError('Geometry/channel mismatch')
        if sample['dataset']==VISUAL and sample['unit']['fnirs']!='subject_normalized_Hb':raise ValueError('Missing explicit normalized coordinate')
    result=dict(schema=SCHEMA_V2+'_verification',status='passed',checked_records=len(dataset.index.records),checked_samples=len(dataset),
        record_counts=dict(counts),sample_counts=dict(samples),visual_records=visual,inverse_normalization_max_error=max_error,
        errors=[],unchanged_samples_splits_labels=True,other_datasets_byte_identical=True,SSM_refits=0)
    write_json(out/'verification.json',result)
    write_json(low/'verification.json',dict(schema='physiology_semantic_clean_v6_verification',status='passed',checked_records=len(lower.records),
        visual_records=len(visual),exact_usable_measurement_selection=True,corrected_Visual_EEG_reused=True,
        other_datasets_byte_identical=True,verification_source=str((out/'verification.json').resolve()),errors=[]))
    return {k:v for k,v in result.items() if k!='visual_records'}
