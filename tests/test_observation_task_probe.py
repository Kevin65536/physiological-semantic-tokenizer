"""Synthetic event/clock/coordinate, integer-interface and subject-score checks."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis import observation_task_probe as p
from src.tokenizers.factorized_observation import FactorizedObservationTokenizer


def _row(subject, split, i):
    rec = "session_01"
    key = f"{p.DATASET}|{subject}|{rec}"
    condition = p.CLASSES[i % 2]
    ems, hms = 7010. + 30_000 * i, 57480. + 30_000 * i
    meta = dict(task=p.TASK, session_idx=1, eeg_label=condition, fnirs_label=condition,
                alignment_support_ms={"eeg": [None, None], "fnirs": [None, None]})
    event = dict(dataset_id=p.DATASET, canonical_subject_id=subject, base_record_id=rec, join_key=key,
                 event_index=i, event_type="trial", label=condition, eeg_time_ms=ems, fnirs_time_ms=hms,
                 alignment_role="paired_eeg_fnirs_marker", metadata=meta, schema=p.EVENT_ALIGNMENT_SCHEMA)
    return dict(subject=subject, split=split, record=rec, event=event, dataset=p.DATASET, task=p.TASK,
                id=f"{key}|event={i:05d}|offset=-5|duration=30", event_index=i, join_key=key,
                condition=condition, class_index=i % 2, target_kind="classification", partial=False,
                duration_s=30, supported_seconds=30, event_relative_start_s=-5, target_event_time_s=5,
                eeg_start_s=ems/1000 - 5, hb_start_s=hms/1000 - 5, dependency_views=1,
                dependency_id=f"{key}|eeg_event_ms={ems:.6f}")


def _cell(modality, label):
    if modality == "eeg":
        return np.sin(2 * np.pi * (6 if label == 0 else 16) * np.arange(400) / 200)[None, :]
    value = 1 - 2*label
    return np.stack((np.full(20, value), np.full(20, -0.5 * value)))


class _SyntheticPanel:
    """Actual Hb clock is 50 s later than EEG, with a *different* marker phase."""
    def __init__(self, registry_rows):
        self.rows, self.reads = [], []
        self.invalid = set()
        for item in registry_rows:
            if item["split"] not in p.SPLITS:  # not a signal source
                continue
            i, s = item["event_index"], item["subject"]
            record_key = f"{p.DATASET}/{s}/session_01"
            for m, names in p.POSITIONS.items():
                for name in reversed(names):  # array/dict order is NOT physical position order
                    self.rows.append(SimpleNamespace(dataset=p.DATASET, subject=s, record="session_01",
                        split=item["split"], modality=m, position=name, native_rate_hz=200,
                        source_unit="uV" if m == "eeg" else "subject_normalized_Hb", scale=8.,
                        native_start_sample=6000*i, member_clocks=((record_key, f"w{i:05d}", 30.*i, 50.+30*i),),
                        parent=f"{record_key}/parent{i}", start_s=0, i=i))

    def read(self, idx):
        row = self.rows[idx]
        assert row.split in p.SPLITS
        self.reads.append((row.subject, row.i, row.position))
        # Only the true requested 8 s contains the synthetic condition. Outside
        # is a decoy with the opposite signal. The probe must use both clocks.
        m = row.modality
        cell = _cell(m, row.i % 2)
        scale = 1 + p.POSITIONS[m].index(row.position) / 16
        data = np.tile(_cell(m, 1-row.i % 2).T, (15, 1)) * scale
        start = 1802 if m == "eeg" else 95
        data[start:start + 4*cell.shape[1]] = np.tile(cell.T, (4, 1)) * scale
        mask = np.ones_like(data, bool)
        mask[0] = False  # unscored outside the requested event: has no effect
        if (row.subject, row.i, row.position) in self.invalid:
            mask[start] = False
        shape = (15, 2 * p.RATES[m], cell.shape[0])
        return {"x": data.astype(np.float32).reshape(shape), "score_mask": mask.reshape(shape)}


@pytest.fixture
def registry(monkeypatch, tmp_path):
    parts = {"train": ["S1", "S2"], "validation": ["S3", "S4"], "test": ["poison"]}
    rows = [_row(s, split, i) for split, names in parts.items() for s in names for i in range(4)]
    calls = []

    class Reader:
        def __init__(self, root, *, split, task):
            assert root == tmp_path and split in p.SPLITS and task == p.TASK
            calls.append(split)
            self.schema = p.SCHEMA_V2
            self.records = {i: None for i in range(5)}
            self.plan = dict(samples=rows, subject_partitions={p.DATASET: parts}, exclusions=[])
            self.samples = [r for r in rows if r["split"] == split]

        def __getitem__(self, _):
            pytest.fail("The downstream owner provides event metadata only; no signal access")

    panel = _SyntheticPanel(rows)
    def verify():
        (tmp_path / "verification.json").write_text(json.dumps(dict(schema=p.SCHEMA_V2 + "_verification",
            status="passed", errors=[], checked_samples=len(rows), checked_records=5, unchanged_samples_splits_labels=True)))
    verify()
    monkeypatch.setattr(p, "DownstreamInputDataset", Reader)
    monkeypatch.setattr(p, "DOWNSTREAM_ROOT", tmp_path)
    monkeypatch.setattr(p, "_panel", lambda _: panel)
    return SimpleNamespace(root=tmp_path, rows=rows, parts=parts, panel=panel, calls=calls, verify=verify)


@pytest.fixture(scope="module")
def codecs():
    result = {"pattern": {}, "waveform": {}}
    for m in p.POSITIONS:
        cells = np.stack([_cell(m, i % 2) * (1 + i / 40) for i in range(8)])
        for route in result:
            result[route][m] = FactorizedObservationTokenizer(m, routing=route, n_patterns=2, detail_count=4,
                                                           detail_bits=2, seed=17).fit(cells, partition="train")
    return result


def test_real_codec_end_to_end_and_validation_targets_cannot_fit(registry, codecs, monkeypatch):
    monkeypatch.setattr(FactorizedObservationTokenizer, "fit", lambda *a, **k: pytest.fail("refit codec"))
    kwargs = dict(cache_root=registry.root, g0_root="synthetic", eeg_codec=codecs["pattern"]["eeg"],
                  hb_codec=codecs["pattern"]["hb"], waveform_codecs=codecs["waveform"])
    result = p.run_task_probe(**kwargs)
    assert result["status"] == "completed" and result["planned_events"] == result["retained_events"] == 16
    assert registry.calls == ["train", "validation"] and len(registry.panel.reads) == 16 * 7
    assert {row["subject"] for row in result["subjects"]} == {"S1", "S2", "S3", "S4"}
    assert result["scores"]["pattern_ids/fusion"]["subject_balanced_accuracy"] == 1.
    assert result["scores"]["training_majority"]["subject_balanced_accuracy"] == .5
    assert result["bits_per_complete_event"] == {"flat": {"eeg": 144, "hb": 108}, "prototype": {"eeg": 16, "hb": 12}}
    arms = ("pattern_ids", "waveform_ids_same_k", "continuous_attributes", "flat_to_attributes",
            "flat_to_pattern_ids", "pattern_ids_shuffled_train_labels")
    assert set(result["scores"]) == {"training_majority", *(f"{a}/{b}" for a in arms for b in ("eeg", "hb", "fusion"))}
    for row in result["events"]:
        assert ("predicted_condition" in row) == (row["split"] == "validation")
        assert ("shuffled_training_condition" in row) == (row["split"] == "train")
    json.dumps(result, allow_nan=False)
    # Reversing ONLY validation labels changes scores, never predictions or fitted weights.
    for row in registry.rows:
        if row["split"] == "validation":
            row["class_index"] = 1 - row["class_index"]
            c = p.CLASSES[row["class_index"]]
            row["condition"] = row["event"]["label"] = c
            row["event"]["metadata"].update(eeg_label=c, fnirs_label=c)
    other = p.run_task_probe(**kwargs)
    assert other["scores"]["pattern_ids/fusion"]["subject_balanced_accuracy"] == 0.
    assert [r.get("predicted_condition") for r in result["events"]] == [r.get("predicted_condition") for r in other["events"]]


def test_true_clock_ticks_named_order_no_rescaling_mask_and_join_failures(registry):
    events, expected, _ = p._events(registry.root)
    panel = registry.panel
    panel.invalid.add(("S2", 0, "P4"))
    # A missing position excludes the event across *all* arms; no substitute. A
    # complete EEG-only clock join cannot rescue noncontaining Hb support.
    panel.rows[:] = [r for r in panel.rows if (r.subject, r.i, r.position) != ("S4", 2, "F3")]
    for r in panel.rows:
        if r.subject == "S4" and r.i == 3:
            key, w, e, h = r.member_clocks[0]
            r.member_clocks = ((key, w, e, h + 15.),)
    waves, _ = p._event_inputs(events, expected, panel)
    assert waves["eeg"].shape == (13, 4, 4, 1, 400) and waves["hb"].shape == (13, 3, 4, 2, 20)
    for m in p.POSITIONS:
        for j, _ in enumerate(p.POSITIONS[m]):
            np.testing.assert_allclose(waves[m][0, j], np.tile(_cell(m, 0)[None], (4, 1, 1)) * (1 + j/16), atol=1e-6)
    reason = {(r["subject"], r["event_index"]): r["status"] for r in events}
    assert reason["S2", 0] == "incomplete_observed_or_source_scoring_support"
    assert reason["S4", 2] == "missing_fixed_G0_position"
    assert reason["S4", 3] == "no_single_registered_G0_score_crop_on_both_clocks"
    assert not any(s == "S4" and i >= 2 for s, i, pos in panel.reads)


@pytest.mark.parametrize("change,match", [
    (lambda r: r.update(hb_start_s=r["eeg_start_s"]), "clock"),
    (lambda r: r.update(condition="rest"), "label"),
    (lambda r: r.update(dependency_views=2), "shared"),
    (lambda r: r.update(record="session_00"), "identity"),
    (lambda r: r["event"]["metadata"].update(fnirs_label="?"), "label"),
])
def test_invalid_registered_event_rejected(change, match):
    row = _row("S1", "train", 0)
    change(row)
    with pytest.raises(ValueError, match=match):
        p._event(row, "train")


def test_unsafe_root_split_duplicate_support_and_units_fail_before_arrays(registry):
    with pytest.raises(ValueError, match="registered"):
        p._events(registry.root / "unregistered")
    registry.parts["test"].append("S1")
    with pytest.raises(ValueError, match="crosses"):
        p._events(registry.root)
    registry.parts["test"].pop()
    registry.rows.append(deepcopy(registry.rows[0])); registry.verify()
    with pytest.raises(ValueError, match="duplicate event"):
        p._events(registry.root)
    registry.rows.pop(); registry.verify()
    events, expected, _ = p._events(registry.root)
    registry.panel.rows[-1].source_unit = "relative_MBll_not_scaled"
    with pytest.raises(ValueError, match="unit"):
        p._event_inputs(events, expected, registry.panel)
    registry.panel.rows[-1].source_unit = "subject_normalized_Hb"
    registry.panel.rows[-1].split = "test"
    with pytest.raises(ValueError, match="partition"):
        p._event_inputs(events, expected, registry.panel)
    registry.panel.rows[-1].split = "validation"
    registry.panel.rows.append(deepcopy(registry.panel.rows[-1]))
    with pytest.raises(ValueError, match="ambiguous"):
        p._event_inputs(events, expected, registry.panel)
    assert not registry.panel.reads


def test_crop_uses_retained_parent_rounding_on_half_sample_ties(registry):
    _, expected, _ = p._events(registry.root)
    rows = [row for row in registry.panel.rows if row.i == 1 and row.subject == "S1"]
    for row in rows:
        rec, w, _, h = row.member_clocks[0]
        row.member_clocks = ((rec, w, 30.0175, h),)  # 6003.5 requested: re-rounding here gives 6004
        row.native_start_sample = 6003  # G0 actually rounded its parent to tick 3, then added 6000
    index, _ = p._crops(SimpleNamespace(rows=rows), expected)
    assert index["S1", "session_01"][0]["tick"]["eeg"] == 6003
    rows[0].native_start_sample = 6002
    with pytest.raises(ValueError, match="clock"):
        p._crops(SimpleNamespace(rows=rows), expected)


def test_shuffle_and_subject_equal_two_class_metric_include_failed_subjects():
    y = np.array([0]*6 + [1]*2 + [1]*7 + [0]*3)
    groups = [("same_subject", "a")]*6 + [("same_subject", "b")]*3 + [("other", "b")]*9
    shuffled = p._shuffle(y, groups)
    for g in set(groups):
        take = [v == g for v in groups]
        np.testing.assert_array_equal(np.bincount(y[take], minlength=2), np.bincount(shuffled[take], minlength=2))
    np.testing.assert_array_equal(shuffled, p._shuffle(y, groups))
    truth = np.array([0]*20 + [1]*20 + [0, 1])
    pred = truth.copy(); pred[-2:] = 1 - pred[-2:]
    s = np.array(["big"] * 40 + ["small"] * 2)
    assert p._score(truth, pred, s, ["big", "small"])["subject_balanced_accuracy"] == .5
    incomplete = p._score(truth, pred, s, ["big", "missing", "small"])
    assert incomplete["subject_balanced_accuracy"] is None and incomplete["diagnostic_defined_subject_mean"] == .5
    assert incomplete["subjects"][1] == dict(subject="missing", n=0, confusion_true_by_predicted=[[0,0], [0,0]], balanced_accuracy=None)


def test_pattern_rejects_float_ids_and_continuous_payloads():
    codec = SimpleNamespace(n_patterns=2, encode=lambda x, mode: {"pattern": np.ones(len(x))})
    with pytest.raises(ValueError, match="integer ID"):
        p._ids(codec, np.zeros((1, 1, 400)))
    codec.encode = lambda x, mode: {"pattern": np.zeros(len(x), int), "sidecar": x}
    with pytest.raises(ValueError, match="alone"):
        p._ids(codec, np.zeros((1, 1, 400)))
