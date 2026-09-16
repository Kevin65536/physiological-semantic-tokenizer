from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np

from experiments.build_clean_event_index import _simultaneous_dsr_events

from src.data.event_alignment import (
    EVENT_ALIGNMENT_SCHEMA,
    align_paired_marker_streams,
    detect_offset_blocks,
    drift_slope_ms_per_min,
    normalize_marker_targets,
    read_xlsx_rows,
    visual_stimulus_onsets_from_dc9,
)


def _marker(times, labels):
    class_names = ["A", "B"]
    y = np.zeros((2, len(times)), dtype=np.float32)
    for index, label in enumerate(labels):
        y[label, index] = 1.0
    return {"time": np.asarray(times, dtype=np.float64), "y": y, "className": class_names}


def test_normalize_marker_targets_accepts_event_major_or_class_major():
    event_major = np.asarray([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
    class_major = event_major.T

    np.testing.assert_array_equal(normalize_marker_targets(event_major, 3), class_major)
    np.testing.assert_array_equal(normalize_marker_targets(class_major, 3), class_major)


def test_align_paired_marker_streams_records_fixed_offset():
    eeg = _marker([1000, 2000, 3000], [0, 1, 0])
    fnirs = _marker([1500, 2500, 3500], [0, 1, 0])

    events, report = align_paired_marker_streams(
        dataset_id="synthetic",
        subject="s1",
        record_id="r1",
        eeg_marker=eeg,
        fnirs_marker=fnirs,
    )

    assert len(events) == 3
    assert report.schema == EVENT_ALIGNMENT_SCHEMA
    assert report.alignment_case == "stable_fixed_offset"
    assert report.offset_mean_ms == 500.0
    assert report.label_sequence_match is True
    assert events[0].metadata["offset_ms"] == 500.0


def test_dsr_projects_only_go_nogo_stimuli_from_aligned_block_anchors():
    def marker(times, descriptions):
        return SimpleNamespace(
            time=np.asarray(times, dtype=np.float64),
            y=np.ones((1, len(times)), dtype=np.float32),
            className=np.asarray(["event"], dtype=object),
            event=SimpleNamespace(desc=np.asarray(descriptions, dtype=np.uint8)),
        )

    eeg = marker([0, 100, 2100, 10_000, 10_100, 12_100], [48, 16, 32, 48, 32, 16])
    fnirs = marker([500, 10_500], [3, 3])
    events, report = _simultaneous_dsr_events(
        subject="VP001",
        record_id="cnt_dsr",
        eeg_marker_struct=eeg,
        fnirs_marker_struct=fnirs,
        source_files=[],
    )

    assert [event.label for event in events] == ["Go", "No-go", "No-go", "Go"]
    assert [event.fnirs_time_ms for event in events] == [600.0, 2600.0, 10_600.0, 12_600.0]
    assert all(event.event_type == "stimulus" for event in events)
    assert report.metadata["projected_stimulus_count"] == 4


def test_alignment_reports_continuous_drift_slope():
    eeg_times = np.linspace(0, 10 * 60_000, 20)
    residual = np.linspace(0, 500, 20)
    fnirs_times = eeg_times + residual
    eeg = _marker(eeg_times, [0, 1] * 10)
    fnirs = _marker(fnirs_times, [0, 1] * 10)

    _, report = align_paired_marker_streams(
        dataset_id="synthetic",
        subject="s1",
        record_id="drift",
        eeg_marker=eeg,
        fnirs_marker=fnirs,
    )

    assert report.alignment_case == "continuous_drift"
    assert report.drift_slope_ms_per_min is not None
    assert report.drift_slope_ms_per_min > 40.0


def test_detect_offset_blocks_splits_large_jumps():
    blocks = detect_offset_blocks(np.asarray([100, 101, 102, 50_000, 50_001], dtype=np.float64))
    assert len(blocks) == 2
    assert blocks[0]["count"] == 3
    assert blocks[1]["count"] == 2


def test_drift_slope_returns_none_for_single_event():
    assert drift_slope_ms_per_min(np.asarray([1.0]), np.asarray([2.0])) is None


def test_visual_dc9_onsets_use_documented_three_second_stimulus_pair():
    # Trial 2 has no response annotation. Taking every third row would shift
    # trial 3, while the appearance->disappearance relation remains intact.
    triggers_ms = [2_000, 5_000, 6_200, 14_700, 17_698, 27_400, 30_400, 32_100]

    onsets, diagnostics = visual_stimulus_onsets_from_dc9(triggers_ms)

    np.testing.assert_allclose(onsets, [2_000, 14_700, 27_400])
    assert diagnostics["extraction_rule"] == "dc9_followed_by_stimulus_offset_at_3000ms"
    assert diagnostics["stimulus_onset_candidate_count"] == 3


def test_visual_dc9_onsets_remove_duplicated_annotation_rows():
    triggers_ms = [2_000, 2_000, 5_000, 5_000, 6_200, 6_200, 14_700, 14_700, 17_700, 17_700]

    onsets, diagnostics = visual_stimulus_onsets_from_dc9(triggers_ms)

    np.testing.assert_allclose(onsets, [2_000, 14_700])
    assert diagnostics["duplicate_dc9_count"] == 5


def test_read_xlsx_rows_uses_stdlib(tmp_path: Path):
    path = tmp_path / "labels.xlsx"
    with ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "")
        archive.writestr("_rels/.rels", "")
        archive.writestr(
            "xl/sharedStrings.xml",
            """<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
            <si><t>Epoch_ID</t></si><si><t>Type</t></si><si><t>RR</t></si>
            </sst>""",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            """<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
            <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>
            <row r="2"><c r="A2"><v>1</v></c><c r="B2" t="s"><v>2</v></c></row>
            </sheetData></worksheet>""",
        )

    assert read_xlsx_rows(str(path)) == [{"Epoch_ID": "1", "Type": "RR"}]


def _align(eeg_times, fnirs_times):
    return align_paired_marker_streams(
        dataset_id="synthetic", subject="s1", record_id="r1",
        eeg_marker=_marker(eeg_times, [0] * len(eeg_times)),
        fnirs_marker=_marker(fnirs_times, [0] * len(fnirs_times)),
    )


def test_slow_physical_drift_is_rejected_even_below_dispersion_threshold():
    times = np.arange(20) * 60_000.0
    _, report = _align(times, times + np.linspace(0, 300, 20))
    assert report.offset_std_ms < 100
    assert report.drift_slope_ms_per_min > 10
    assert report.alignment_case == "continuous_drift"


def test_jitter_is_not_mislabeled_as_physical_clock_drift():
    times = np.arange(20) * 60_000.0
    _, report = _align(times, times + np.tile([150, -150], 10))
    assert abs(report.drift_slope_ms_per_min) < 10
    assert report.alignment_case == "mixed_or_unstable_offset"


def test_session_jumps_do_not_count_as_within_session_drift():
    times = np.arange(8) * 60_000.0
    _, report = _align(times, times + np.repeat([500, 60_500], 4))
    assert report.drift_slope_ms_per_min > 10
    assert report.alignment_case == "piecewise_constant_offset"
    assert all(abs(b["drift_slope_ms_per_min"]) < 1e-9 for b in report.offset_blocks)


def test_multiple_missing_or_ambiguous_markers_produce_no_pairs():
    events, report = _align([0, 1000, 2000, 3000, 4000], [500, 1500, 2500])
    assert not events
    assert report.alignment_case == "unresolved_marker_count_mismatch"
    events, report = _align([0, 1000, 2000], [500, 1500])
    assert not events
    assert report.alignment_case == "ambiguous_marker_pairing"


def test_single_missing_marker_retains_original_indices():
    events, report = _align([0, 10_000, 20_000, 30_000, 40_000], [500, 10_500, 30_500, 40_500])
    assert report.skipped_marker_indices["eeg_indices"] == [2]
    assert [e.metadata["eeg_source_index"] for e in events] == [0, 1, 3, 4]
    assert [e.metadata["fnirs_source_index"] for e in events] == [0, 1, 2, 3]
    assert all(e.metadata["offset_ms"] == 500 for e in events)


def test_unverified_append_gap_is_not_accepted_as_window_support():
    from src.data.event_alignment import window_within_alignment_support
    events, _ = _align([0, 60_000, 120_000, 180_000], [500, 60_500, 180_500, 240_500])
    assert window_within_alignment_support(events[0].to_dict(), 0, 20)
    assert not window_within_alignment_support(events[1].to_dict(), 0, 20)
    assert not window_within_alignment_support(events[2].to_dict(), -5, 20)
    assert window_within_alignment_support(events[2].to_dict(), 0, 20)


def _visual_csv(path, middle_time="00:00:00.110", middle_value="11"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Header\nSampling Period[s],0.1\nData\nTime,Mark,CH1\n"
        "00:00:00.000,0,0\n"
        f"{middle_time},1,{middle_value}\n"
        "00:00:00.210,2,21\n", encoding="utf-8",
    )


def test_visual_marker_clock_is_shared_with_signal_reader(tmp_path):
    from src.data.event_alignment import read_visual_fnirs_csv
    from experiments.build_clean_event_index import _read_visual_marks
    path = tmp_path / "test.csv"
    _visual_csv(path)
    data = read_visual_fnirs_csv(path, load_signals=True)
    marks, rate = _read_visual_marks(path)
    assert rate == 10
    assert marks[0]["onset_ms"] == 110
    np.testing.assert_allclose(data["time_s"], [0, .11, .21])
    np.testing.assert_allclose(data["values"][:, 0], [0, 11, 21])


def test_visual_invalid_rows_and_gaps_are_not_silently_compressed(tmp_path):
    import pytest
    from src.data.event_alignment import read_visual_fnirs_csv
    path = tmp_path / "test.csv"
    _visual_csv(path, middle_value="bad")
    with pytest.raises(ValueError, match="Invalid ETG row"):
        read_visual_fnirs_csv(path, load_signals=True)
    _visual_csv(path, middle_time="00:00:00.200")
    with pytest.raises(ValueError, match="Missing ETG time support"):
        read_visual_fnirs_csv(path)


def test_visual_builder_rejects_different_oxy_deoxy_clocks(tmp_path):
    import pytest
    from experiments.build_clean_eeg_fnirs_cache import iter_visual
    directory = tmp_path / "S01/fNIRS"
    _visual_csv(directory / "S01_Probe1_Oxy.csv")
    _visual_csv(directory / "S01_Probe1_Deoxy.csv", middle_time="00:00:00.100")
    with pytest.raises(ValueError, match="clock or marker mismatch"):
        list(iter_visual(tmp_path, 1, 1, 0))


def test_visual_future_cache_resamples_native_clock_before_filtering(tmp_path, monkeypatch):
    import experiments.build_clean_eeg_fnirs_cache as builder
    directory = tmp_path / "S01/fNIRS"
    _visual_csv(directory / "S01_Probe1_Oxy.csv")
    _visual_csv(directory / "S01_Probe1_Deoxy.csv")
    record = next(builder.iter_visual(tmp_path, 1, 1, 0))
    observed = []
    def process(values, **kwargs):
        observed.append(values.copy())
        return SimpleNamespace(values=values, state=SimpleNamespace(to_dict=lambda: {}), quality={})
    monkeypatch.setattr(builder, "standardize_fnirs_record", process)
    monkeypatch.setattr(builder, "apply_homer2_aligned_contract", process)
    manifest = builder.build_record(record, tmp_path / "new_cache", False)
    for values in observed:
        np.testing.assert_allclose(values[:, 0], [0, 10, 20])
    with np.load(manifest["record_npz"]) as arrays:
        np.testing.assert_allclose(arrays["time_s"], [0, .1, .2])
        np.testing.assert_allclose(arrays["native_input_time_s"], [0, .11, .21])
        np.testing.assert_allclose(arrays["native_input_fnirs"][:, 0], [0, 11, 21])


def test_legacy_multimodal_path_uses_missing_marker_alignment():
    from src.data.simultaneous_eeg_nirs_dataset import SimultaneousMultiModalDataset
    eeg = _marker([0, 10_000, 20_000, 30_000, 40_000], [0] * 5)
    fnirs = _marker([500, 10_500, 30_500, 40_500], [0] * 4)
    dataset = object.__new__(SimultaneousMultiModalDataset)
    dataset.subject_ids = [1]
    dataset.eeg_loader = SimpleNamespace(load_subject_data=lambda *_: (None, None, {"fs": 200}))
    dataset.fnirs_loader = SimpleNamespace(load_subject_data=lambda *_: (None, None, {"fs": 10}))
    dataset._get_markers = lambda _, modality: eeg if modality == "eeg" else fnirs
    dataset.task = "dsr"
    dataset.window_duration_s = 1
    dataset.window_offset_ms = 0
    dataset.trials = []
    dataset._build_trial_index()
    assert [t.trial_idx for t in dataset.trials] == [0, 1, 3, 4]
    assert dataset.trials[2].eeg_start_sample == 6000
    assert dataset.trials[2].nirs_start_sample == 305
    # The same runtime path must exclude slow drift, not only mismatched counts.
    times = np.arange(20) * 60_000.0
    eeg = _marker(times, [0] * 20)
    fnirs = _marker(times + np.linspace(0, 300, 20), [0] * 20)
    dataset.trials = []
    dataset._build_trial_index()
    assert dataset.trials == []


def test_refed_builder_reports_unknown_clock_error_and_checks_duration(tmp_path, monkeypatch):
    import experiments.build_clean_event_index as builder
    (tmp_path / "data/1").mkdir(parents=True)
    monkeypatch.setattr(builder, "_refed_video_info", lambda _: {})
    monkeypatch.setattr(builder, "_refed_sam", lambda _: {})
    monkeypatch.setattr(builder, "_rel", str)
    arrays = {"EEG_videos.mat": np.zeros((1, 1000)),
              "fNIRS_videos.mat": np.zeros((2, 1, 48)), "1_label.mat": np.zeros((1, 2))}
    monkeypatch.setattr(builder, "loadmat", lambda path, **_: {"video_1": arrays[path.name]})
    events, reports = builder.iter_refed(tmp_path, 1, 1)
    assert reports[0].alignment_case == "shared_segment_index_no_marker_stream"
    assert reports[0].offset_mean_ms is None and reports[0].offset_std_ms is None
    assert events[0].metadata["continuous_label_stream"]["native_sample_rate_hz"] == 1
    assert events[0].metadata["clock_evidence"] == "publisher_segment_origin_assumed_not_measured"
    arrays["fNIRS_videos.mat"] = np.zeros((2, 1, 55))
    _, reports = builder.iter_refed(tmp_path, 1, 1)
    assert reports[0].alignment_case == "segment_duration_mismatch"


def test_legacy_alignment_diagnostic_uses_same_pairs_as_training_path():
    from src.data.simultaneous_eeg_nirs_dataset import SimultaneousCognitiveLoader
    loader = object.__new__(SimultaneousCognitiveLoader)
    loader.task = "dsr"
    eeg = _marker([0, 10_000, 20_000, 30_000, 40_000], [0] * 5)
    fnirs = _marker([500, 10_500, 30_500, 40_500], [0] * 4)
    loader.get_session_markers = lambda _, modality: eeg if modality == "eeg" else fnirs
    report = loader.check_marker_alignment(1)
    assert report["skipped_marker_indices"]["eeg_indices"] == [2]
    assert report["residual_std_ms"] == 0
    assert report["label_index_match"] is True
