"""Small, unprotected v5 fixtures for the public downstream benchmark reader."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.data.downstream_benchmark import (
    EEG_ONLY_CHANNELS,
    INVENTORY_SCHEMA,
    PROCESSING_SCHEMA,
    TASKS,
    DownstreamBenchmarkDataset,
    build_inventory,
    build_pretraining_inventory,
    canonical_sample_id,
    make_splits,
)
from src.data.event_alignment import EVENT_ALIGNMENT_SCHEMA


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def measurement_cache(tmp_path):
    root = tmp_path / "v5"
    records, events, reports, geometry = [], [], [], []
    for task_id, task in TASKS.items():
        dataset_id = task["dataset_id"]
        eeg_names = EEG_ONLY_CHANNELS[dataset_id] + ["extra_real_channel"]
        for subject_number in range(10):
            subject = (f"subject {subject_number + 1:02d}" if dataset_id == "eeg_fnirs_single_trial"
                       else f"VP{subject_number + 1:03d}" if dataset_id == "simultaneous_eeg_nirs"
                       else f"S{subject_number + 1:02d}" if dataset_id == "visual_cognitive_motivation"
                       else str(subject_number + 1))
            record_ids = ([f"{subject}_Part1_Probe1", f"{subject}_Part1_Probe2"] if task_id == "visual"
                          else ["video_1_hbo_hbr"] if task_id == "refed_regression" else [f"cnt_{task_id}"])
            for record_id in record_ids:
                record_path = root / dataset_id / subject.replace(" ", "_") / record_id
                classes = task["classes"] or ["video context"]
                seconds = 21.5 if task_id == "refed_regression" else len(classes) * task["duration_s"]
                branch = ("single_trial_eeg_artifact_clean_v4" if dataset_id == "eeg_fnirs_single_trial"
                          else "simultaneous_eeg_eog_clean_v1" if dataset_id == "simultaneous_eeg_nirs"
                          else "raw_with_ocular_artifact")
                fnirs_names = ["L1_HbO", "L1_HbR", "L2_HbO", "L2_HbR"]
                arrays = {name: f"{name}.npy" for name in ("eeg", "fnirs", "eeg_supported_channels", "fnirs_supported_channels", "eeg_bad_channel_mask")}
                row = dict(schema="clean_eeg_fnirs_cache_v2", storage="measurement_npy_record_v1",
                           processing_schema=PROCESSING_SCHEMA, motion_method="none", execution="completed",
                           dataset_id=dataset_id, subject=subject, record_id=record_id,
                           record_npz=str(record_path), sample_rate_hz=10.0, arrays=arrays,
                           array_shapes=dict(eeg=[round(seconds * 200), len(eeg_names)],
                                             fnirs=[round(seconds * 10), 4], eeg_supported_channels=[len(eeg_names)],
                                             fnirs_supported_channels=[4], eeg_bad_channel_mask=[len(eeg_names)]),
                           measurement=dict(eeg_channel_names=eeg_names, fnirs_channel_names=fnirs_names,
                                            fnirs_component_roles=["HbO", "HbR", "HbO", "HbR"],
                                            eeg_preprocessing_state=dict(signal_branch=branch, canonical_unit="uV"),
                                            fnirs_preprocessing_state=dict(canonical_unit="relative:fixture")),
                           homer2_aligned_contract=dict(alignment_state=dict(schema=PROCESSING_SCHEMA,
                               parameters=dict(motion_method="none"))),
                           preprocessing_scope="offline whole fixture record", source_files=[])
                records.append(row)
                event_record = "video_1" if task_id == "refed_regression" else record_id
                for event_index, label in enumerate(classes):
                    start = event_index * task["duration_s"] * 1000
                    metadata = dict(task=task_id, epoch_id=event_index + 1, epoch_type=label)
                    if task_id == "refed_regression":
                        values = np.stack((np.arange(22), np.arange(22) + 100), axis=1).astype(float).tolist()
                        values[-1][1] = float("nan")
                        metadata.update(continuous_label_stream=dict(names=["valence", "arousal"],
                            values=values, sample_count=22, native_sample_rate_hz=1.0),
                            eeg_samples=21500, fnirs_samples=1024)
                    events.append(dict(schema=EVENT_ALIGNMENT_SCHEMA, dataset_id=dataset_id, subject=subject,
                                       record_id=event_record, event_index=event_index, label=label,
                                       label_index=event_index, eeg_time_ms=start, fnirs_time_ms=start,
                                       event_type="video_segment_with_continuous_labels" if task_id == "refed_regression" else "trial",
                                       duration_ms=seconds * 1000 if task_id == "refed_regression" else None,
                                       metadata=metadata))
                reports.append(dict(schema=EVENT_ALIGNMENT_SCHEMA, dataset_id=dataset_id, subject=subject,
                                    record_id=event_record, alignment_case="stable_fixed_offset", label_sequence_match=True))
        if not any(row["dataset_id"] == dataset_id for row in geometry):
            for modality, names in (("eeg", eeg_names), ("fnirs", fnirs_names)):
                for number, name in enumerate(names):
                    geometry.append(dict(dataset_id=dataset_id, canonical_subject_id="all", modality=modality,
                                         channel_name=name, x=number / 100, y=number / 200, z=1.0,
                                         coordinate_system="fixture", coordinate_units="normalized_head_unit",
                                         source_file="fixture_layout"))
    _write_json(root / "cache_manifest.json", dict(schema="clean_eeg_fnirs_cache_v2",
        storage="measurement_npy_record_v1", processing_schema=PROCESSING_SCHEMA,
        motion_method="none", execution="completed", records=records))
    _write_json(root / "event_index/event_manifest.json", dict(event_alignment_schema=EVENT_ALIGNMENT_SCHEMA))
    _write_jsonl(root / "event_index/events.jsonl", events)
    _write_jsonl(root / "event_index/alignment_reports.jsonl", reports)
    _write_jsonl(root / "channel_geometry/channels.jsonl", geometry)
    return root


def _write_record_arrays(inventory, sample, *, fnirs_bad=False):
    for reference in sample["event_refs"]:
        row = inventory["records"][reference["join_key"]]
        path = Path(row["path"])
        path.mkdir(parents=True, exist_ok=True)
        shapes = row["array_shapes"]
        eeg = np.broadcast_to(np.arange(shapes["eeg"][1], dtype=np.float64)[None], shapes["eeg"])
        fnirs = np.broadcast_to(np.asarray([10., 20., 30., 40.])[None], shapes["fnirs"])
        if reference["probe"] == "Probe2":
            fnirs = fnirs + 100
        for name, values in dict(eeg=eeg, fnirs=fnirs, eeg_supported_channels=np.ones(shapes["eeg"][1], bool),
                                 fnirs_supported_channels=np.full(4, not fnirs_bad, bool),
                                 eeg_bad_channel_mask=np.zeros(shapes["eeg"][1], bool)).items():
            np.save(path / f"{name}.npy", values)


def test_inventory_reads_no_measurement_arrays_and_merges_native_visual_events(measurement_cache, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("inventory must not read signal arrays")
    monkeypatch.setattr(np, "load", forbidden)
    inventory = build_inventory(measurement_cache)
    assert inventory["schema"] == INVENTORY_SCHEMA
    assert set(inventory["tasks"]) == set(TASKS)
    assert inventory["tasks"]["visual"]["sample_count"] == 40
    visual = [row for row in inventory["samples"] if row["task_id"] == "visual"]
    assert all(len(row["event_refs"]) == 2 for row in visual)
    profile = inventory["tasks"]["visual"]["profiles"]["multimodal"]
    assert profile["fnirs_channel_names"] == [
        "Probe1:L1_HbO", "Probe1:L2_HbO", "Probe2:L1_HbO", "Probe2:L2_HbO",
        "Probe1:L1_HbR", "Probe1:L2_HbR", "Probe2:L1_HbR", "Probe2:L2_HbR"]
    assert all(len(t["profiles"]["eeg_only"]["channel_names"]) == 16 for t in inventory["tasks"].values())
    assert len({s["sample_id"] for s in inventory["samples"]}) == len(inventory["samples"])


def test_subject_folds_cover_inventory_once_and_inner_fitting_never_sees_outer_subjects(measurement_cache):
    inventory = build_inventory(measurement_cache)
    splits = make_splits(inventory)
    assert splits == make_splits(inventory)
    for task_id, task in splits["tasks"].items():
        evaluated = []
        for fold in task["folds"]:
            evaluated += fold["eval_indices"]
            train, evaluation = set(fold["train_indices"]), set(fold["eval_indices"])
            assert not train & evaluation
            assert {inventory["samples"][i]["subject_id"] for i in train}.isdisjoint(
                {inventory["samples"][i]["subject_id"] for i in evaluation})
            inner_evaluated = []
            for inner in fold["inner_folds"]:
                fitting, validation = set(inner["train_indices"]), set(inner["eval_indices"])
                assert fitting | validation == train
                assert not fitting & validation
                inner_evaluated += inner["eval_indices"]
            assert sorted(inner_evaluated) == sorted(train)
        assert sorted(evaluated) == [i for i, row in enumerate(inventory["samples"]) if row["task_id"] == task_id]
    mi = splits["tasks"]["motor_imagery"]["folds"]
    ma = splits["tasks"]["mental_arithmetic"]["folds"]
    assert [f["eval_subjects"] for f in mi] == [f["eval_subjects"] for f in ma]
    with pytest.raises(ValueError, match="freezes"):
        make_splits(inventory, seed=1)


def test_reader_keeps_measurement_scale_and_ordered_probe_channels(measurement_cache):
    inventory = build_inventory(measurement_cache)
    position = next(i for i, row in enumerate(inventory["samples"]) if row["task_id"] == "visual")
    _write_record_arrays(inventory, inventory["samples"][position])
    sample = DownstreamBenchmarkDataset(inventory, [position], task_id="visual")[0]
    assert sample["eeg"].shape == (17, 1600)
    assert sample["fnirs"].shape == (8, 80)
    np.testing.assert_array_equal(sample["fnirs"][:, 0], [10, 30, 110, 130, 20, 40, 120, 140])
    np.testing.assert_array_equal(sample["eeg"][:, 0], np.arange(17))
    assert sample["eeg"].dtype == np.float64
    assert sample["global_index"] == position
    assert sample["sample_id"] == inventory["samples"][position]["sample_id"]
    eeg_only = DownstreamBenchmarkDataset(inventory, [position], modality_profile="eeg_only")[0]
    assert eeg_only["eeg"].shape == (16, 1600)
    assert eeg_only["fnirs"].shape == (0, 80)
    assert eeg_only["sample_id"] == sample["sample_id"]


def test_refed_partial_native_targets_have_coordinate_masks_independent_of_inputs(measurement_cache):
    inventory = build_inventory(measurement_cache)
    position = next(i for i, row in enumerate(inventory["samples"])
                    if row["task_id"] == "refed_regression" and row["window_start_s"] == 20)
    row = inventory["samples"][position]
    assert row["partial_window"]
    assert row["target_valid_counts"] == [2, 1]
    _write_record_arrays(inventory, row, fnirs_bad=True)
    sample = DownstreamBenchmarkDataset(inventory, [position])[0]
    assert sample["target"].shape == (2, 20)
    np.testing.assert_array_equal(sample["target_valid_mask"].sum(axis=1), [2, 1])
    np.testing.assert_array_equal(sample["target"][0, :2], [20, 21])
    assert not sample["channel_valid_mask"]["fnirs"].any()
    assert sample["valid_mask"]["eeg"].sum() == 300
    assert not sample["target_valid_mask"][:, 2:].any()


def test_wrong_producer_is_rejected_before_signal_read(measurement_cache, monkeypatch):
    manifest_path = measurement_cache / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["processing_schema"] = "physiology_measurement_alignment_v3"
    _write_json(manifest_path, manifest)
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("wrong producer must fail before array read"))
    with pytest.raises(ValueError, match="v5 none"):
        build_inventory(measurement_cache)


def test_split_rejects_class_gap_without_changing_sample_denominator(measurement_cache):
    inventory = build_inventory(measurement_cache)
    for sample in inventory["samples"]:
        if sample["task_id"] == "motor_imagery":
            sample["target"] = 0
    with pytest.raises(ValueError, match="class coverage"):
        make_splits(inventory)


def test_missing_visual_probe_retains_events_with_explicit_unobserved_channels(measurement_cache):
    manifest_path = measurement_cache / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["records"] = [row for row in manifest["records"] if row["record_id"] != "S01_Part1_Probe2"]
    _write_json(manifest_path, manifest)
    inventory = build_inventory(measurement_cache)
    assert inventory["tasks"]["visual"]["sample_count"] == 40
    assert inventory["tasks"]["visual"]["support_summary"]["missing_visual_probe_samples"] == 4
    position = next(i for i, row in enumerate(inventory["samples"])
                    if row["task_id"] == "visual" and row["subject_id"] == "S01")
    _write_record_arrays(inventory, inventory["samples"][position])
    sample = DownstreamBenchmarkDataset(inventory, [position])[0]
    missing = [i for i, name in enumerate(sample["channel_names"]["fnirs"]) if name.startswith("Probe2:")]
    assert not sample["channel_valid_mask"]["fnirs"][missing].any()
    assert not sample["fnirs"][missing].any()


def test_inventory_binding_rejects_label_changes_before_array_read(measurement_cache, monkeypatch):
    inventory = build_inventory(measurement_cache)
    event_path = measurement_cache / "event_index/events.jsonl"
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    first_index = next(i for i, row in enumerate(inventory["samples"]) if row["task_id"] == "motor_imagery")
    sample = inventory["samples"][first_index]
    event = next(event for event in events if event["dataset_id"] == sample["dataset_id"]
                 and event["subject"].replace(" ", "_") == sample["subject_id"]
                 and event["event_index"] == sample["event_id"]
                 and event["metadata"]["task"] == "motor_imagery")
    event["label"] = "unexpected"
    _write_jsonl(event_path, events)
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("changed native identity must fail before array read"))
    with pytest.raises(ValueError, match="Task label"):
        DownstreamBenchmarkDataset(inventory, [first_index])[0]


def test_sample_identity_encodes_native_fields_and_time_instead_of_array_position():
    identity = canonical_sample_id("d", "s", "part|1", "t", "event/2", 0, 8)
    assert "part%7C1" in identity and "event%2F2" in identity
    assert "start_us=0|stop_us=8000000" in identity
    assert identity != canonical_sample_id("d", "s", "part|1", "t", "event/2", 8, 8)


def test_pretraining_crops_use_only_paired_signal_support_and_record_unused_tails(measurement_cache, monkeypatch):
    inventory = build_inventory(measurement_cache)
    # Target availability does not govern the unlabeled source support.
    for row in inventory["samples"]:
        if row["task_id"] == "refed_regression":
            row["target_valid_counts"] = [0, 0]
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("source inventory must remain metadata only"))
    source = build_pretraining_inventory(inventory)
    assert source["inventory_kind"] == "pretraining"
    assert not any(row["task_id"] == "dsr" for row in source["samples"])
    refed = [row for row in source["samples"] if row["task_id"] == "refed_regression"]
    assert len(refed) == 20
    assert {row["window_start_s"] for row in refed} == {0.0, 8.0}
    assert all(row["window_duration_s"] == 8 for row in source["samples"])
    assert all("parent_sample_id" in row and "target" not in row for row in source["samples"])
    assert source["pretraining_support"]["unused_signal_duration_s"] == 55.0
    unused_full = [row for row in source["unused_signal_intervals"] if row["duration_s"] == 4]
    assert len(unused_full) == 10
    assert all((row["parent_relative_start_s"], row["parent_relative_stop_s"]) == (16, 20) for row in unused_full)
    for row in source["samples"]:
        assert row["subject_id"] == next(parent["subject_id"] for parent in inventory["samples"]
                                         if parent["sample_id"] == row["parent_sample_id"])
    assert all(task["target_shape"] == [] for task in source["tasks"].values())


def test_pretraining_native_deduplication_ignores_label_values(measurement_cache):
    inventory = build_inventory(measurement_cache)
    duplicate = dict(inventory["samples"][0], sample_id="different_parent", label="different_label_metadata")
    inventory["samples"].append(duplicate)
    source = build_pretraining_inventory(inventory)
    assert source["pretraining_support"]["excluded_reason_counts"]["duplicate_native_signal_support"] == 1
    assert sum(len(row["duplicate_parent_sample_ids"]) for row in source["samples"]) == 1


def test_pretraining_reader_returns_no_targets_and_accepts_complete_partial_parent_crop(measurement_cache, monkeypatch):
    import src.data.downstream_benchmark as module

    inventory = build_inventory(measurement_cache)
    parent = next(row for row in inventory["samples"]
                  if row["task_id"] == "refed_regression" and row["window_start_s"] == 20)
    # The 20 s parent is partial, but a full 8 s measured crop exists. Its
    # annotation stream remains only 22 s long and must not affect admission.
    record = inventory["records"][parent["event_refs"][0]["join_key"]]
    record["array_shapes"]["eeg"][0] = 5600
    record["array_shapes"]["fnirs"][0] = 280
    manifest_path = measurement_cache / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for row in manifest["records"]:
        if row["record_npz"] == record["path"]:
            row["array_shapes"] = record["array_shapes"]
    _write_json(manifest_path, manifest)
    source = build_pretraining_inventory(inventory)
    position = next(i for i, row in enumerate(source["samples"])
                    if row["parent_sample_id"] == parent["sample_id"] and row["crop_offset_s"] == 0)
    _write_record_arrays(source, source["samples"][position])
    monkeypatch.setattr(module, "refed_continuous_target_window", lambda *a, **kw: pytest.fail("pretraining must not construct labels"))
    sample = DownstreamBenchmarkDataset(source, [position])[0]
    assert sample["eeg"].shape == (17, 1600)
    assert sample["fnirs"].shape == (4, 80)
    assert sample["valid_mask"]["eeg"].all()
    assert all(key not in sample for key in ("target", "target_valid_mask", "target_time_s", "label"))
