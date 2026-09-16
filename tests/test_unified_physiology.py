import numpy as np
import hashlib
import json
import inspect
from pathlib import Path

from src.data.clean_physiology_cache import CLEAN_CACHE_SCHEMA
from src.data.event_alignment import EVENT_ALIGNMENT_SCHEMA
from types import SimpleNamespace

import pytest

from src.data.eeg_artifact_preprocessing import EEGArtifactCleaningConfig, clean_single_trial_eeg
from src.data.unified_physiology import (
    CANONICAL_EEG_SAMPLE_RATE_HZ,
    CANONICAL_FNIRS_COMPONENTS,
    CANONICAL_FNIRS_SAMPLE_RATE_HZ,
    CANONICAL_UNIT,
    DEFAULT_UNIFIED_WINDOW_DURATION_S,
    EEG_ARTIFACT_MASK_POLICY,
    FORBIDDEN_TASK_NAMESPACES,
    NativeEEGRecord,
    ChannelGeometryIndex,
    REFED_CONTINUOUS_SEQUENCE_SCHEMA,
    REFEDContinuousSequenceDataset,
    UnifiedPhysiologyWindowDataset,
    UNIFIED_PHYSIOLOGY_SCHEMA,
    canonical_fnirs_channel_names,
    canonical_label,
    collate_refed_continuous_sequences,
    fnirs_component_roles,
    preprocess_eeg_record,
    preprocess_eeg_record_with_quality,
    preprocess_fnirs_record,
    refed_continuous_target_window,
)


def test_unified_loader_default_observation_window_is_twenty_seconds():
    default = inspect.signature(UnifiedPhysiologyWindowDataset).parameters["window_duration_s"].default
    assert default == DEFAULT_UNIFIED_WINDOW_DURATION_S == 20.0


def test_unified_loader_default_eeg_branch_is_single_trial_v4():
    default = inspect.signature(UnifiedPhysiologyWindowDataset).parameters["eeg_signal_branch"].default
    assert default == "single_trial_eeg_artifact_clean_v4"


@pytest.mark.parametrize('coordinate', ['legacy_robust', 'measurement'])
def test_unified_window_disables_artifact_marking_and_invalidity(coordinate):
    dataset = object.__new__(UnifiedPhysiologyWindowDataset)
    dataset.output_coordinate = coordinate
    dataset.window_duration_s = 1.0
    dataset.window_offset_s = 0.0
    record = SimpleNamespace(
        dataset_id="eeg_fnirs_single_trial",
        canonical_subject_id="subject_01",
        base_record_id="session_01",
        signal_branch="homer2_wavelength_pair",
        join_key="eeg_fnirs_single_trial|subject_01|session_01",
    )
    event = {
        "event_type": "trial",
        "label": "MA",
        "label_index": 1,
        "eeg_time_ms": 0.0,
        "fnirs_time_ms": 0.0,
        "onset_ms": 0.0,
        "metadata": {"task": "mental_arithmetic"},
    }
    dataset.windows = [SimpleNamespace(record=record, event=event, window_offset_s=0.0)]
    record_data = {
        "eeg": np.ones((200, 2), dtype=np.float32),
        "fnirs": np.ones((10, 2), dtype=np.float32),
        "eeg_channel_names": ("F3", "F4"),
        "fnirs_channel_names": ("CH1_HbO", "CH1_HbR"),
        "fnirs_component_roles": ("HbO", "HbR"),
        "eeg_preprocessing_state": {"signal_branch": "single_trial_eeg_artifact_clean_v4", "canonical_unit": "uV"},
        "fnirs_preprocessing_state": {"canonical_unit": "uM"},
        "fnirs_processed_valid_mask": np.ones((10, 2), dtype=bool),
        "eeg_quality": {
            "artifact_mask": np.ones(200, dtype=bool),
            "bad_channel_mask": np.zeros(2, dtype=bool),
            "processed_valid_mask": np.column_stack((np.ones(200, dtype=bool), np.zeros(200, dtype=bool))),
        },
        "eeg_geometry": [{"channel_name": "F3"}, {"channel_name": "F4"}],
    }
    dataset._load_canonical_record = lambda _: record_data
    dataset.geometry_index = SimpleNamespace(
        for_channels=lambda **_: [{"channel_name": "CH1_HbO"}, {"channel_name": "CH1_HbR"}]
    )

    sample = dataset[0]

    assert sample["artifact_mask_policy"] == EEG_ARTIFACT_MASK_POLICY
    assert not sample["artifact_mask"]["eeg"].any()
    if coordinate == 'measurement':
        assert sample['unit'] == {'eeg': 'uV', 'fnirs': 'uM'}
        assert sample['channel_valid_mask']['eeg'].shape == sample['eeg'].shape
        assert sample['channel_valid_mask']['eeg'][0].all()
        assert not sample['channel_valid_mask']['eeg'][1].any()
    np.testing.assert_array_equal(
        sample["analysis_valid_mask"]["eeg"],
        sample["valid_mask"]["eeg"],
    )

    event["eeg_time_ms"], event["fnirs_time_ms"] = 12.0, 151.0
    record_data["eeg"][:] = np.arange(200)[:, None]
    record_data["fnirs"][:] = np.arange(10)[:, None]
    sample = dataset[0]
    grid = sample["alignment"]["sample_grid"]
    assert grid["eeg"]["actual_start_ms"] == 10
    assert grid["eeg"]["rounding_error_ms"] == -2
    assert grid["fnirs"]["actual_start_ms"] == 200
    assert grid["fnirs"]["rounding_error_ms"] == 49
    assert sample["eeg"][0, 0] == sample["fnirs"][0, 0] == 2


def test_fnirs_names_are_unified_to_hbo_hbr_components():
    names = canonical_fnirs_channel_names(["CH1_Oxy", "CH1_Deoxy", "CH2_HbO", "CH2_HbR"])
    assert names == ("CH1_HbO", "CH1_HbR", "CH2_HbO", "CH2_HbR")
    assert set(fnirs_component_roles(names)) == set(CANONICAL_FNIRS_COMPONENTS)


def test_common_preprocessing_unifies_rates_and_robust_units(tmp_path):
    time = np.arange(5000) / 500.0
    eeg_native = np.column_stack((20 + 5 * np.sin(2 * np.pi * 10 * time), -8 + np.cos(2 * np.pi * 6 * time)))
    eeg, eeg_state = preprocess_eeg_record(
        NativeEEGRecord(
            values=eeg_native,
            sample_rate_hz=500.0,
            channel_names=("Fz", "Cz"),
            native_unit="uV",
            source_path=tmp_path / "test.edf",
        )
    )
    assert eeg.shape[0] == int(round(eeg_native.shape[0] * CANONICAL_EEG_SAMPLE_RATE_HZ / 500.0))
    assert eeg_state["canonical_unit"] == CANONICAL_UNIT
    assert np.max(np.abs(np.median(eeg, axis=0))) < 0.05

    fnirs_time = np.arange(1000) / 20.0
    fnirs_native = np.column_stack((3 + np.sin(2 * np.pi * 0.08 * fnirs_time), -2 + 0.5 * np.cos(2 * np.pi * 0.05 * fnirs_time)))
    fnirs, fnirs_state = preprocess_fnirs_record(
        fnirs_native,
        sample_rate_hz=20.0,
        native_contract={"native_unit": "mmol/L"},
    )
    assert fnirs.shape[0] == int(round(fnirs_native.shape[0] * CANONICAL_FNIRS_SAMPLE_RATE_HZ / 20.0))
    assert fnirs_state["canonical_unit"] == CANONICAL_UNIT
    assert np.max(np.abs(np.median(fnirs, axis=0))) < 0.05


def test_measurement_coordinate_preserves_paired_amplitude_and_eeg_units(tmp_path):
    time = np.arange(1000)/200
    x = np.column_stack([20*np.sin(2*np.pi*10*time),5*np.sin(2*np.pi*10*time)])
    outputs = []
    for unit,factor in [('uV',1.),('V',1e-6)]:
        record = NativeEEGRecord(x*factor,200.,('F3','F4'),unit,tmp_path/'fixture',unit_evidence='known synthetic')
        y,state,q = preprocess_eeg_record_with_quality(record,output_coordinate='measurement')
        assert y.dtype == np.float64 and state['canonical_unit']=='uV'
        outputs.append(y)
    np.testing.assert_allclose(*outputs,atol=1e-12)
    hb = np.column_stack([4*np.sin(time),-np.sin(time)])
    y,state = preprocess_fnirs_record(hb,sample_rate_hz=10,native_contract={'output_unit':'uM'},
                                    output_coordinate='measurement')
    np.testing.assert_array_equal(y,hb)
    assert state['canonical_unit']=='uM' and y.dtype == np.float64


def test_visual_label_separates_condition_from_event_role():
    label = canonical_label(
        {
            "event_type": "trial",
            "label": "RR",
            "label_index": 0,
            "metadata": {
                "task": "visual_cognitive_motivation",
                "event_role": "stimulus_onset",
                "epoch_type": "RR",
            },
        },
        "visual_cognitive_motivation",
    )
    assert label["schema"] == "canonical_task_label_v1"
    assert label["condition"] == "RR"
    assert label["class_index"] == 0
    assert label["event_role"] == "stimulus_onset"


def _refed_continuous_event(duration_s=5.0, values=None):
    if values is None:
        values = [[float(index), float(100 + index)] for index in range(int(duration_s))]
    return {
        "dataset_id": "refed",
        "subject": "1",
        "record_id": "video_1",
        "event_index": 0,
        "event_type": "video_segment_with_continuous_labels",
        "label": "positive",
        "label_index": 0,
        "eeg_time_ms": 0.0,
        "fnirs_time_ms": 0.0,
        "onset_ms": 0.0,
        "duration_ms": duration_s * 1000.0,
        "metadata": {
            "task": "emotion_video",
            "continuous_label_stream": {
                "names": ["valence", "arousal"],
                "values": values,
                "sample_count": len(values),
            },
        },
    }


def test_refed_continuous_target_window_is_fixed_shape_and_time_aligned():
    target = refed_continuous_target_window(
        _refed_continuous_event(),
        window_start_s=0.0,
        window_duration_s=4.0,
        target_sample_rate_hz=2.0,
    )

    assert target["schema"] == REFED_CONTINUOUS_SEQUENCE_SCHEMA
    assert target["values"].shape == (2, 8)
    assert target["valid_mask"].shape == (2, 8)
    assert target["valid_mask"].all()
    np.testing.assert_allclose(target["time_s"], np.arange(8) / 2.0)
    np.testing.assert_allclose(target["values"][0], np.arange(8) / 2.0)
    np.testing.assert_allclose(target["values"][1], 100.0 + np.arange(8) / 2.0)


def test_refed_continuous_target_masks_partial_support_and_missing_values():
    values = [[0.0, 100.0], [1.0, 101.0], [float("nan"), 102.0], [3.0, 103.0], [4.0, 104.0]]
    target = refed_continuous_target_window(
        _refed_continuous_event(values=values),
        window_start_s=2.0,
        window_duration_s=4.0,
    )

    assert target["values"].shape == (2, 4)
    assert target["valid_mask"][0].tolist() == [False, True, True, False]
    assert target["valid_mask"][1].tolist() == [True, True, True, False]
    assert target["values"][0, 0] == 0.0
    assert target["values"][1, 0] == 102.0


def test_refed_continuous_dataset_expands_video_and_versions_target_contract(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    (cache / "event_index").mkdir(parents=True)
    record = {
        "schema": CLEAN_CACHE_SCHEMA,
        "dataset_id": "refed",
        "subject": "1",
        "record_id": "video_1_hbo_hbr",
        "sample_rate_hz": 47.62,
        "record_npz": str(cache / "unused.npz"),
        "metadata": {},
    }
    event = _refed_continuous_event(
        duration_s=45.0,
        values=[[float(index), float(100 + index)] for index in range(45)],
    )
    report = {
        "dataset_id": "refed",
        "subject": "1",
        "record_id": "video_1",
        "alignment_case": "shared_segment_index_no_marker_stream",
        "label_sequence_match": True,
    }
    (cache / "cache_manifest.json").write_text(json.dumps({"schema": CLEAN_CACHE_SCHEMA, "records": [record]}), encoding="utf-8")
    (cache / "event_index/event_manifest.json").write_text(json.dumps({"event_alignment_schema": EVENT_ALIGNMENT_SCHEMA}), encoding="utf-8")
    (cache / "event_index/events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    (cache / "event_index/alignment_reports.jsonl").write_text(json.dumps(report) + "\n", encoding="utf-8")

    dataset = REFEDContinuousSequenceDataset(cache, window_duration_s=20.0)
    summary = dataset.contract_summary()

    assert len(dataset) == 3
    assert [window.window_offset_s for window in dataset.windows] == [0.0, 20.0, 40.0]
    assert summary["schema"] == REFED_CONTINUOUS_SEQUENCE_SCHEMA
    assert summary["source_event_count"] == 1
    assert summary["partial_window_count"] == 1
    assert summary["target_shape"] == [2, 20]
    assert summary["split_group_keys"] == ["subject"]
    assert summary["window_dependency_group_keys"] == ["subject", "record_id"]
    assert summary["event_index_sha256"]

    monkeypatch.setattr(
        UnifiedPhysiologyWindowDataset,
        "__getitem__",
        lambda self, index: {
            "schema": UNIFIED_PHYSIOLOGY_SCHEMA,
            "label": {"class_index": 0, "condition": "positive"},
            "valid_mask": {
                "eeg": np.ones(4000, dtype=bool),
                "fnirs": np.ones(200, dtype=bool),
            },
            "sample_rate_hz": {"eeg": 200.0, "fnirs": 10.0},
            "event": event,
        },
    )
    sample = dataset[2]
    assert sample["schema"] == REFED_CONTINUOUS_SEQUENCE_SCHEMA
    assert sample["target"].shape == (2, 20)
    assert sample["target_valid_mask"].sum() == 10
    assert sample["target_time_s"][0] == 40.0
    assert sample["label"]["target_type"] == "continuous_sequence_regression"
    assert sample["video_context_label"]["condition"] == "positive"
    assert sample["sample_id"].endswith("event=0|start_ms=40000")
    assert "values" not in sample["event"]["metadata"]["continuous_label_stream"]

    full_only = REFEDContinuousSequenceDataset(
        cache,
        window_duration_s=20.0,
        include_partial_windows=False,
    )
    assert len(full_only) == 2


def test_refed_continuous_collator_stacks_training_fields_and_keeps_provenance_list():
    def sample(index):
        return {
            "schema": REFED_CONTINUOUS_SEQUENCE_SCHEMA,
            "eeg": np.full((2, 4), index, dtype=np.float32),
            "fnirs": np.full((3, 2), index, dtype=np.float32),
            "valid_mask": {"eeg": np.ones(4, dtype=bool), "fnirs": np.ones(2, dtype=bool)},
            "analysis_valid_mask": {"eeg": np.ones(4, dtype=bool), "fnirs": np.ones(2, dtype=bool)},
            "artifact_mask": {"eeg": np.zeros(4, dtype=bool), "fnirs": np.zeros(2, dtype=bool)},
            "bad_channel_mask": {"eeg": np.zeros(2, dtype=bool), "fnirs": np.zeros(3, dtype=bool)},
            "target": np.full((2, 2), index, dtype=np.float32),
            "target_valid_mask": np.ones((2, 2), dtype=bool),
            "target_time_s": np.arange(2, dtype=np.float32),
            "label": {"target_type": "continuous_sequence_regression"},
            "target_names": ["valence", "arousal"],
            "target_sample_rate_hz": 1.0,
            "sample_rate_hz": {"eeg": 2.0, "fnirs": 1.0},
            "channel_names": {"eeg": ["Fz", "Cz"], "fnirs": ["CH1", "CH2", "CH3"]},
            "component_roles": {"eeg": ["electrical_potential"] * 2, "fnirs": ["HbO"] * 3},
            "channel_geometry": {"eeg": [{"x": None}], "fnirs": [{"x": None}]},
            "sample_id": f"sample-{index}",
            "dataset_id": "refed",
            "subject": "1",
            "record_id": "video_1",
            "join_key": "refed|1|video_1",
            "event": {"event_index": 0},
            "alignment": {"event_relative_window_start_s": float(index)},
            "target_metadata": {"source_sample_rate_hz": 1.0},
            "video_context_label": {"condition": "positive"},
            "preprocessing_state": {"eeg": {}, "fnirs": {}},
            "eeg_signal_branch": "raw_with_ocular_artifact",
        }

    batch = collate_refed_continuous_sequences([sample(0), sample(1)])

    assert tuple(batch["eeg"].shape) == (2, 2, 4)
    assert tuple(batch["fnirs"].shape) == (2, 3, 2)
    assert tuple(batch["target"].shape) == (2, 2, 2)
    assert batch["sample_id"] == ["sample-0", "sample-1"]
    assert len(batch["provenance"]) == 2
    assert batch["channel_geometry"]["eeg"][0]["x"] is None


def test_visual_geometry_index_selects_the_probe_encoded_in_record_id(tmp_path):
    geometry_dir = tmp_path / "channel_geometry"
    geometry_dir.mkdir()
    rows = []
    for probe, x in (("Probe1", 1.0), ("Probe2", -1.0)):
        rows.append(
            {
                "dataset_id": "visual_cognitive_motivation",
                "canonical_subject_id": "all",
                "record_id": probe,
                "base_record_id": probe,
                "modality": "fnirs",
                "channel_name": "CH1",
                "x": x,
                "y": 0.0,
                "z": 1.0,
                "coordinate_system": "test",
                "coordinate_units": "test",
                "source_file": "test",
                "metadata": {"coordinate_status": "test_probe_specific"},
            }
        )
    (geometry_dir / "channels.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    index = ChannelGeometryIndex(tmp_path)
    probe1 = index.for_channels(
        record=SimpleNamespace(
            dataset_id="visual_cognitive_motivation",
            canonical_subject_id="S01",
            base_record_id="S01_Part1_Probe1",
        ),
        modality="fnirs",
        channel_names=("CH1_HbO",),
    )
    probe2 = index.for_channels(
        record=SimpleNamespace(
            dataset_id="visual_cognitive_motivation",
            canonical_subject_id="S01",
            base_record_id="S01_Part1_Probe2",
        ),
        modality="fnirs",
        channel_names=("CH1_HbO",),
    )

    assert probe1[0]["x"] == 1.0
    assert probe2[0]["x"] == -1.0


def test_alignment_admission_filter_excludes_unstable_records(tmp_path):
    cache = tmp_path / "cache"
    (cache / "event_index").mkdir(parents=True)
    record = {
        "schema": CLEAN_CACHE_SCHEMA,
        "dataset_id": "refed",
        "subject": "1",
        "record_id": "video_1_hbo_hbr",
        "sample_rate_hz": 10.0,
        "record_npz": str(cache / "unused.npz"),
        "metadata": {},
    }
    (cache / "cache_manifest.json").write_text(json.dumps({"schema": CLEAN_CACHE_SCHEMA, "records": [record]}), encoding="utf-8")
    (cache / "event_index/event_manifest.json").write_text(json.dumps({"event_alignment_schema": EVENT_ALIGNMENT_SCHEMA}), encoding="utf-8")
    event = {
        "dataset_id": "refed",
        "subject": "1",
        "record_id": "video_1",
        "event_type": "trial",
        "label": "x",
        "eeg_time_ms": 0.0,
        "fnirs_time_ms": 0.0,
        "onset_ms": 0.0,
    }
    report = {
        "dataset_id": "refed",
        "subject": "1",
        "record_id": "video_1",
        "alignment_case": "continuous_drift",
        "label_sequence_match": True,
    }
    (cache / "event_index/events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    (cache / "event_index/alignment_reports.jsonl").write_text(json.dumps(report) + "\n", encoding="utf-8")

    from src.data.unified_physiology import UnifiedPhysiologyWindowDataset

    strict = UnifiedPhysiologyWindowDataset(cache, dataset_ids=["refed"])
    diagnostic = UnifiedPhysiologyWindowDataset(cache, dataset_ids=["refed"], admissible_alignment_cases=None)
    assert len(strict) == 0
    assert len(strict.excluded_alignment_records) == 1
    assert len(diagnostic) == 1


def test_unified_loader_admits_restored_dsr_stimulus_labels(tmp_path):
    cache = tmp_path / "cache"
    (cache / "event_index").mkdir(parents=True)
    record = {
        "schema": CLEAN_CACHE_SCHEMA,
        "dataset_id": "simultaneous_eeg_nirs",
        "subject": "VP001",
        "record_id": "cnt_dsr",
        "sample_rate_hz": 10.0,
        "record_npz": str(cache / "unused.npz"),
        "metadata": {},
    }
    (cache / "cache_manifest.json").write_text(json.dumps({"schema": CLEAN_CACHE_SCHEMA, "records": [record]}), encoding="utf-8")
    (cache / "event_index/event_manifest.json").write_text(json.dumps({"event_alignment_schema": EVENT_ALIGNMENT_SCHEMA}), encoding="utf-8")
    event = {
        "dataset_id": "simultaneous_eeg_nirs",
        "subject": "VP001",
        "record_id": "cnt_dsr",
        "event_type": "stimulus",
        "label": "Go",
        "label_index": 0,
        "eeg_time_ms": 0.0,
        "fnirs_time_ms": 0.0,
        "onset_ms": 0.0,
        "metadata": {"task": "dsr"},
    }
    report = {
        "dataset_id": "simultaneous_eeg_nirs",
        "subject": "VP001",
        "record_id": "cnt_dsr",
        "alignment_case": "stable_fixed_offset",
        "label_sequence_match": True,
    }
    (cache / "event_index/events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    (cache / "event_index/alignment_reports.jsonl").write_text(json.dumps(report) + "\n", encoding="utf-8")

    dataset = UnifiedPhysiologyWindowDataset(
        cache,
        dataset_ids=["simultaneous_eeg_nirs"],
        admissible_alignment_cases=None,
    )
    summary = dataset.contract_summary()

    assert FORBIDDEN_TASK_NAMESPACES == frozenset()
    assert len(dataset) == 1
    assert dataset.windows[0].event["label"] == "Go"
    assert summary["forbidden_task_policy"] == "no_hard_exclusions_dsr_restored_v2"
    assert summary["excluded_forbidden_task_window_count_by_namespace"] == {}
    assert summary["excluded_forbidden_task_record_count"] == 0


def test_v4_artifact_cache_loads_only_when_source_stat_and_join_key_match(tmp_path):
    source = tmp_path / "cnt.mat"
    source.write_bytes(b"source")
    cache_root = tmp_path / "cache"
    path = cache_root / "subject_01" / "session_00.npz"
    path.parent.mkdir(parents=True)
    state = {"signal_branch": "single_trial_eeg_artifact_clean_v4"}
    np.savez_compressed(
        path,
        schema=np.asarray("single_trial_eeg_artifact_cache_v4"),
        signal_branch=np.asarray("single_trial_eeg_artifact_clean_v4"),
        join_key=np.asarray("eeg_fnirs_single_trial|subject_01|session_00"),
        source_path=np.asarray("cnt.mat"),
        source_size_bytes=np.asarray(source.stat().st_size, dtype=np.int64),
        source_mtime_ns=np.asarray(source.stat().st_mtime_ns, dtype=np.int64),
        eeg=np.zeros((20, 2), dtype=np.float32),
        artifact_mask=np.ones(20, dtype=bool),
        bad_channel_mask=np.zeros(2, dtype=bool),
        channel_names=np.asarray(["F3", "F4"]),
        preprocessing_state_json=np.asarray(json.dumps(state)),
    )
    repo_root = Path(__file__).resolve().parents[1]
    code_sha256 = {
        "audit": hashlib.sha256(
            (repo_root / "experiments/audit_single_trial_eeg_artifact_v2.py").read_bytes()
        ).hexdigest(),
        "cleaner": hashlib.sha256(
            Path(clean_single_trial_eeg.__code__.co_filename).read_bytes()
        ).hexdigest(),
    }
    (cache_root / "cache_manifest.json").write_text(
        json.dumps(
            {
                "schema": "single_trial_eeg_artifact_cache_v4",
                "signal_branch": "single_trial_eeg_artifact_clean_v4",
                "cleaning_config": EEGArtifactCleaningConfig().to_dict(),
                "code_sha256": code_sha256,
                "records": [
                    {
                        "join_key": "eeg_fnirs_single_trial|subject_01|session_00",
                        "cache_path": str(path),
                        "source_path": "cnt.mat",
                    }
                ],
            }
        )
    )
    dataset = object.__new__(UnifiedPhysiologyWindowDataset)
    dataset.project_root = tmp_path
    dataset.eeg_artifact_cache_root = cache_root
    dataset.eeg_artifact_config = None
    dataset._artifact_cache_manifest = None
    record = SimpleNamespace(
        dataset_id="eeg_fnirs_single_trial",
        canonical_subject_id="subject_01",
        base_record_id="session_00",
        join_key="eeg_fnirs_single_trial|subject_01|session_00",
    )
    loaded = dataset._load_cached_single_trial_eeg(record, "single_trial_eeg_artifact_clean_v4")
    assert loaded is not None
    eeg, names, loaded_state, quality = loaded
    assert eeg.shape == (20, 2)
    assert names == ("F3", "F4")
    assert loaded_state["artifact_cache"]["used"] is True
    assert loaded_state["artifact_cache"]["stored_detected_artifact_fraction"] == 1.0
    assert loaded_state["artifact_cache"]["stored_detected_artifact_mask_exposed"] is False
    assert quality["artifact_mask"].shape == (20,)
    assert not quality["artifact_mask"].any()
    record.join_key = "eeg_fnirs_single_trial|subject_01|session_wrong"
    with pytest.raises(RuntimeError, match="manifest has no record"):
        dataset._load_cached_single_trial_eeg(record, "single_trial_eeg_artifact_clean_v4")
    record.join_key = "eeg_fnirs_single_trial|subject_01|session_00"
    source.write_bytes(b"changed source")
    with pytest.raises(RuntimeError, match="source EEG changed"):
        dataset._load_cached_single_trial_eeg(record, "single_trial_eeg_artifact_clean_v4")


def test_v4_artifact_cache_rejects_stale_code_hash(tmp_path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    (cache_root / "cache_manifest.json").write_text(json.dumps({
        "schema": "single_trial_eeg_artifact_cache_v4",
        "signal_branch": "single_trial_eeg_artifact_clean_v4",
        "cleaning_config": EEGArtifactCleaningConfig().to_dict(),
        "code_sha256": {"audit": "stale", "cleaner": "stale"},
        "records": [],
    }))
    dataset = object.__new__(UnifiedPhysiologyWindowDataset)
    dataset.eeg_artifact_cache_root = cache_root
    dataset._artifact_cache_manifest = None
    with pytest.raises(RuntimeError, match="code hash mismatch"):
        dataset._validated_artifact_cache_manifest()


def test_refed_annotation_clock_is_not_stretched_to_fnirs_duration():
    event = _refed_continuous_event(duration_s=10)
    event["duration_ms"] = 10_020
    target = refed_continuous_target_window(event, window_start_s=2, window_duration_s=4)
    np.testing.assert_allclose(target["values"][0], [2, 3, 4, 5])
    assert target["source_sample_rate_hz"] == 1.0


def test_window_crossing_unverified_session_gap_is_rejected_before_signal_read():
    dataset = object.__new__(UnifiedPhysiologyWindowDataset)
    dataset.window_duration_s = 1.0
    dataset.window_offset_s = 0.0
    event = {"eeg_time_ms": 500.0, "fnirs_time_ms": 500.0,
             "metadata": {"alignment_support_ms": {"eeg": [None, 800], "fnirs": [None, 800]}}}
    dataset.windows = [SimpleNamespace(event=event, record=None, window_offset_s=0.0)]
    dataset._load_canonical_record = lambda _: pytest.fail("must check support before reading signals")
    with pytest.raises(ValueError, match="unverified concatenation interval"):
        dataset[0]
