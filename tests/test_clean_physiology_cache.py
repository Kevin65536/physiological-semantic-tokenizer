import json
from pathlib import Path

import numpy as np

from src.data.event_alignment import EVENT_ALIGNMENT_SCHEMA

from src.data.clean_physiology_cache import (
    CLEAN_CACHE_SCHEMA,
    CleanPhysiologyAlignedWindowDataset,
    CleanPhysiologyCacheIndex,
    base_record_id,
    canonical_subject_id,
    join_key,
    signal_branch,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_canonical_join_key_normalizes_dataset_specific_names():
    assert canonical_subject_id("simultaneous_eeg_nirs", "VP001-NIRS") == "VP001"
    assert canonical_subject_id("simultaneous_eeg_nirs", "VP001") == "VP001"
    assert canonical_subject_id("eeg_fnirs_single_trial", "subject 01") == "subject_01"
    assert base_record_id("refed", "video_1_hbo_hbr") == "video_1"
    assert base_record_id("refed", "video_1_absorbance_780_805_830") == "video_1"
    assert signal_branch("refed", "video_1_absorbance_780_805_830") == "absorbance_780_805_830"
    assert join_key("refed", "1", "video_1_hbo_hbr") == "refed|1|video_1"


def test_cache_index_and_window_dataset_join_refed_branch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "cache"
    record_dir = root / "refed" / "1"
    record_dir.mkdir(parents=True)
    np.savez(
        record_dir / "video_1_hbo_hbr.npz",
        homer2_aligned_fnirs=np.arange(40, dtype=np.float32).reshape(10, 4),
        time_s=np.arange(10, dtype=np.float32),
    )
    manifest_record = {
        "schema": CLEAN_CACHE_SCHEMA,
        "dataset_id": "refed",
        "subject": "1",
        "record_id": "video_1_hbo_hbr",
        "record_npz": "cache/refed/1/video_1_hbo_hbr.npz",
        "sample_rate_hz": 2.0,
        "metadata": {},
    }
    _write_json(root / "cache_manifest.json", {"schema": CLEAN_CACHE_SCHEMA, "records": [manifest_record]})
    _write_json(root / "event_index" / "event_manifest.json", {"event_alignment_schema": EVENT_ALIGNMENT_SCHEMA})
    _write_jsonl(
        root / "event_index" / "events.jsonl",
        [
            {
                "dataset_id": "refed",
                "subject": "1",
                "record_id": "video_1",
                "event_index": 0,
                "event_type": "video_segment",
                "label": "calm",
                "label_index": 3,
                "onset_ms": 1000.0,
            }
        ],
    )
    _write_jsonl(
        root / "event_index" / "alignment_reports.jsonl",
        [{"dataset_id": "refed", "subject": "1", "record_id": "video_1"}],
    )

    index = CleanPhysiologyCacheIndex(root)
    assert index.coverage_summary()["record_keys_without_events"] == []
    assert index.records[0].join_key == "refed|1|video_1"

    dataset = CleanPhysiologyAlignedWindowDataset(root, window_duration_s=2.0, branch_preference="hbo_hbr")
    assert len(dataset) == 1
    item = dataset[0]
    assert item["fnirs"].shape == (4, 4)
    assert item["fnirs"][0].tolist() == [8.0, 12.0, 16.0, 20.0]
    assert item["modality_available"] == {"fnirs": True, "eeg": False}
    assert item["label"] == "calm"


def test_old_cache_is_rejected_before_reading_events_or_signal_arrays(tmp_path):
    import pytest
    root = tmp_path / "cache"
    _write_json(root / "cache_manifest.json", {"schema": "clean_eeg_fnirs_cache_v1"})
    # A malformed event stream must never be opened for a deprecated cache.
    (root / "event_index").mkdir()
    (root / "event_index/events.jsonl").write_text("not JSON")
    with pytest.raises(ValueError, match="Deprecated physiology cache"):
        CleanPhysiologyCacheIndex(root)


def test_current_signal_cache_cannot_use_old_event_index(tmp_path):
    import pytest
    root = tmp_path / "cache"
    _write_json(root / "cache_manifest.json", {"schema": CLEAN_CACHE_SCHEMA, "records": []})
    _write_json(root / "event_index/event_manifest.json", {"event_alignment_schema": "physiology_event_alignment_v1"})
    with pytest.raises(ValueError, match="Deprecated or missing event index"):
        CleanPhysiologyCacheIndex(root)


def test_event_builder_refuses_to_overwrite_old_index_even_with_overwrite(tmp_path, monkeypatch):
    import pytest
    from types import SimpleNamespace
    import experiments.build_clean_event_index as builder
    directory = tmp_path / "event_index"
    path = directory / "event_manifest.json"
    _write_json(path, {"event_alignment_schema": "physiology_event_alignment_v1"})
    original = path.read_bytes()
    monkeypatch.setattr(builder, "parse_args", lambda: SimpleNamespace(output_dir=str(directory), overwrite=True))
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        builder.main()
    assert path.read_bytes() == original


def test_signal_builder_refuses_to_overwrite_old_cache_before_data_read(tmp_path, monkeypatch):
    import pytest
    from types import SimpleNamespace
    import experiments.build_clean_eeg_fnirs_cache as builder
    path = tmp_path / "cache_manifest.json"
    _write_json(path, {"schema": "clean_eeg_fnirs_cache_v1"})
    original = path.read_bytes()
    monkeypatch.setattr(builder, "parse_args", lambda: SimpleNamespace(output_dir=str(tmp_path), overwrite=True))
    monkeypatch.setattr(builder, "iter_records", lambda _: pytest.fail("must not read measured data"))
    with pytest.raises(ValueError, match="Deprecated physiology cache"):
        builder.main()
    assert path.read_bytes() == original
