"""Frozen-input adapters: dependency, partition, electrode and window boundaries."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.data import downstream_benchmark as data
from comparative_methods.rebench_baselines_v1 import SignalFeatures
from comparative_methods.rebench_eeg_common_v1 import encoding_duration, validate_profile


def visual_row(probe, *, subject="S01", split="train", label=0, event=0):
    return dict(id=f"{subject}_Probe{probe}:{event}", dependency_id=f"{subject}:event{event}",
                dataset="visual_cognitive_motivation", task="visual_cognitive_motivation", subject=subject,
                split=split, class_index=label, eeg_start_s=10., duration_s=30.,
                record=f"{subject}_Part1_Probe{probe}", join_key=f"{subject}_Probe{probe}")


def test_probe_grouping_and_split_boundary_are_resolved_without_signal_access():
    samples = [visual_row(1), visual_row(2), visual_row(1, subject="S02", split="validation"),
               visual_row(1, subject="S03", split="test")]
    assert data.frozen_input_groups(samples, "visual", ["train", "validation"]) == [[0, 1], [2]]
    for field, value in [("class_index", 1), ("subject", "other"), ("eeg_start_s", 11.)]:
        broken = deepcopy(samples); broken[1][field] = value
        with pytest.raises(ValueError, match="disagree"):
            data.frozen_input_groups(broken, "visual", ["train", "validation"])
    with pytest.raises(ValueError, match="distinct documented probes"):
        data.frozen_input_groups([samples[0], samples[0]], "visual", ["train"])


def test_frozen_adapter_joins_probes_preserves_masks_and_selects_by_name(monkeypatch):
    names = data.EEG_ONLY_CHANNELS["visual_cognitive_motivation"] + ["FCz"]
    samples = [visual_row(1), visual_row(2), visual_row(1, subject="S02", split="validation"),
               visual_row(1, subject="S03", split="test")]
    calls = []
    class Reader:
        def __init__(self, root, task):
            self.samples = samples
            meta = dict(eeg_channel_names=names, fnirs_channel_names=["CH1_HbO", "CH1_HbR"],
                        eeg_preprocessing_state=dict(canonical_unit="uV"))
            self.records = {row["join_key"]: SimpleNamespace(base_record_id=row["record"], manifest=dict(measurement=meta)) for row in samples}
            self.geometry = SimpleNamespace(for_channels=lambda **kw: [dict(channel_name=name, x=i, y=i % 2) for i, name in enumerate(kw["channel_names"])])
        def __getitem__(self, i):
            row = samples[i]; calls.append(row["split"])
            assert row["split"] != "test"
            probe = int(row["record"][-1])
            return dict(eeg=np.broadcast_to(np.arange(17)[:, None], (17, 6000)).copy(),
                        fnirs=np.full((2, 300), probe), valid_mask=dict(eeg=np.ones((17, 6000), bool), fnirs=np.ones((2, 300), bool)),
                        label=dict(class_index=0), channel_names=dict(eeg=names, fnirs=["CH1_HbO", "CH1_HbR"]),
                        record=row["record"], unit=dict(eeg="uV", fnirs="relative"))
    monkeypatch.setattr(data, "DownstreamInputDataset", Reader)
    dataset = data.FrozenInputBenchmarkDataset("fixture", task="visual", splits=["train", "validation"])
    assert not calls and len(dataset) == 2
    both, one = dataset[0], dataset[1]
    assert both["eeg"].shape == (16, 6000)
    np.testing.assert_array_equal(both["fnirs"][:, 0], [1, 2, 1, 2])
    assert not one["channel_valid_mask"]["fnirs"][[1, 3]].any()
    assert np.all(one["fnirs"][[1, 3]] == 0)
    assert len(set([both["sample_id"], one["sample_id"]])) == 2
    assert len(both["source_sample_ids"]) == 2
    native = data.FrozenInputBenchmarkDataset("fixture", task="visual", splits=["train"], eeg_panel="native")
    assert native[0]["eeg"].shape == (17, 6000)


def test_new_windows_do_not_reinterpret_legacy_defaults():
    profile = dict(input_window_schema="downstream_input_v1", duration_s=30, window_start_s=-5,
                   sampling_rate_hz=200, channel_names=[f"E{i}" for i in range(31)])
    assert validate_profile("visual", profile)[1] == 6000
    assert encoding_duration("visual", {}) == 8
    with pytest.raises(ValueError, match="fixed event-relative"):
        encoding_duration("visual", {**profile, "window_start_s": 0})
    with pytest.raises(ValueError, match="16"):
        validate_profile("visual", {**profile, "input_window_schema": "downstream_rebench_v1", "duration_s": 8})


def test_segment_features_keep_order_and_refed_has_thirty_actual_targets():
    p = dict(input_window_schema="downstream_input_v1", duration_s=30, window_start_s=-5,
             sampling_rate_hz=200, fnirs_sampling_rate_hz=10,
             channel_names=["C3", "C4"], fnirs_channel_names=["L_HbO", "L_HbR"],
             baseline_modality="fnirs", classification_segments=6)
    m = SignalFeatures("motor_imagery", p)
    eeg = torch.zeros(1, 2, 6000)
    hb = torch.arange(300).float().reshape(1, 1, 300).expand(1, 2, 300)
    x = m(eeg, hb)
    assert x.shape == (1, 6 * 2 * 4) and m.feature_dim == x.shape[1]
    assert x[0, 0] < x[0, 8] < x[0, 16]
    refed = SignalFeatures("refed_regression", {**p, "window_start_s": 0})
    assert refed(eeg, hb).shape == (1, 30, 8)
