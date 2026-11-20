"""Report-only pairing and small-cohort uncertainty on invented event decisions."""
from copy import deepcopy
import json

import numpy as np
import pytest

from src.analysis.observation_task_uncertainty import paired_task_contrasts, _interval


def _fixture(*, incomplete=False, waveform=True):
    classes = ["MA", "BL"]
    methods = ["pattern_ids", "pattern_ids_shuffled_train_labels", "continuous_attributes",
               "flat_to_pattern_ids", "flat_to_attributes"] + (["waveform_ids_same_k"] if waveform else [])
    arms = [f"{method}/{branch}" for method in methods for branch in ("eeg", "hb", "fusion")] + ["training_majority"]
    events, subjects = [], []
    for s, split, ys in [("T", "train", [0, 1]), ("A", "validation", [0]*12 + [1]*8),
                         ("B", "validation", [0] if incomplete else [0, 1])]:
        for i, y in enumerate(ys):
            e = dict(id=f"{s}/{i}", dependency_id=f"{s}/{i}/clock", split=split, subject=s, condition=classes[y], status="included")
            if split == "validation":
                e["predicted_condition"] = {}
                for a in arms:
                    if a.startswith("pattern_ids/"):
                        pred = y if s == "A" else 1-y
                    elif a.startswith("waveform_ids_same_k/"):
                        pred = 0 if s == "A" else y
                    else:
                        pred = 1
                    e["predicted_condition"][a] = classes[pred]
            events.append(e)
        events.append(dict(id=s+"/unmatched", dependency_id=s+"/none", subject=s, split=split,
                           condition="MA", status="no_single_registered_G0_score_crop_on_both_clocks"))
        subjects.append(dict(subject=s, split=split, planned=len(ys)+1, counts=dict(included=len(ys), no_single_registered_G0_score_crop_on_both_clocks=1),
                             retained_by_condition={c: ys.count(i) for i, c in enumerate(classes)}))
    scores = {}
    for a in arms:
        rows = []
        for s in ("B", "A"):  # different order from the source/event registry
            cm = [[0, 0], [0, 0]]
            for e in events:
                if e["subject"] == s and e["status"] == "included":
                    cm[classes.index(e["condition"])][classes.index(e["predicted_condition"][a])] += 1
            ns = [sum(row) for row in cm]
            ba = ((cm[0][0]/ns[0] + cm[1][1]/ns[1])/2) if all(ns) else None
            rows.append(dict(subject=s, n=sum(ns), confusion_true_by_predicted=cm, balanced_accuracy=ba))
        values = [r["balanced_accuracy"] for r in rows if r["balanced_accuracy"] is not None]
        mean = sum(values)/len(values)
        scores[a] = dict(subjects=rows, expected_subjects=len(rows), defined_subjects=len(values),
                         diagnostic_defined_subject_mean=mean, subject_balanced_accuracy=mean if not incomplete else None)
    return dict(schema="frozen_observation_ma_probe_v1", dataset="eeg_fnirs_single_trial", task="mental_arithmetic",
                status="not_estimable" if incomplete else "completed", subjects=subjects, class_order=classes,
                scores=scores, events=events, planned_events=len(events), retained_events=sum(r["status"]=="included" for r in events),
                downstream_cache_exclusions=[dict(subject="B", split="validation", reason="pre_cache_clock_gap")],
                waveform_control_present=waveform, bits_per_complete_event=dict(prototype={"eeg": 64, "hb":48}, flat={"eeg":512, "hb":128}))


def test_exact_paired_subject_interval_equal_weights_not_pooled(monkeypatch):
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("report must not open arrays"))
    source = _fixture()
    before = json.dumps(source, sort_keys=True)
    report = paired_task_contrasts(source)
    assert json.dumps(source, sort_keys=True) == before
    json.dumps(report, allow_nan=False)
    by_key = {(r["branch"], r["reference"]): r for r in report["contrasts"]}
    row = by_key["fusion", "waveform_ids_same_k/fusion"]
    # A (20 trials) BA 1-.5=+.5; B (2 trials) 0-1=-1. Equal *subjects*, not 20:2.
    assert row["n_expected_subjects"] == row["n_paired"] == 2 and row["mean_delta_ba"] == -.25
    assert row["ci95"] == [-1., .5] and row["bootstrap_draws"] == 4
    assert row["direction_counts"] == dict(higher=1, lower=1, tie=0, undefined=0)
    assert [r["delta_ba"] for r in row["per_subject"]] == [.5, -1.]
    assert row["pattern_bits_per_trial"] == row["reference_bits_per_trial"] == 112
    assert by_key["fusion", "continuous_attributes/fusion"]["reference_bits_per_trial"] is None
    assert by_key["fusion", "flat_to_pattern_ids/fusion"]["reference_bits_per_trial"] == 640
    assert by_key["hb", "training_majority"]["reference_bits_per_trial"] == 0
    cov = report["coverage"]["validation"]
    assert (cov["planned"], cov["retained"], cov["rejected"]) == (24, 22, 2)
    assert cov["prior_downstream_cache_exclusions"] == {"pre_cache_clock_gap": 1}
    # Exact six-subject bootstrap: in six draws from one '1' plus five '0's,
    # the .025 and .975 quantiles are zero and three successes, respectively.
    ci, n = _interval(np.array([1., 0., 0., 0., 0., 0.]))
    assert n == 46656 and ci == [0., .5]


def test_tied_arms_stay_tied_in_every_paired_resample():
    source = _fixture()
    left, right = "pattern_ids/fusion", "waveform_ids_same_k/fusion"
    source["scores"][right] = deepcopy(source["scores"][left])
    for e in source["events"]:
        if preds := e.get("predicted_condition"):
            preds[right] = preds[left]
    contrast = next(c for c in paired_task_contrasts(source)["contrasts"] if c["reference"] == right)
    assert contrast["direction_counts"] == dict(higher=0, lower=0, tie=2, undefined=0)
    assert contrast["mean_delta_ba"] == 0. and contrast["ci95"] == [0., 0.]


def test_undefined_subjects_remain_in_denominator_and_abstention_survives():
    source = _fixture(incomplete=True, waveform=False)
    report = paired_task_contrasts(source)
    assert all(r["n_paired"] == 1 and r["n_expected_subjects"] == 2 and r["ci95"] is None
               and r["mean_delta_ba"] is None and r["direction_counts"]["undefined"] == 1 for r in report["contrasts"])
    assert not any("waveform_ids" in r["reference"] for r in report["contrasts"])
    assert _interval(np.array([.5])) == (None, 0)
    source["scores"] = {}
    for row in source["events"]:
        row.pop("predicted_condition", None)
    assert paired_task_contrasts(source)["contrasts"] == []


@pytest.mark.parametrize("change,match", [
    (lambda d: d["scores"]["pattern_ids/fusion"]["subjects"][0].update(n=0), "per-subject"),
    (lambda d: d["scores"]["pattern_ids/fusion"]["subjects"][0].update(subject="A"), "unmatched"),
    (lambda d: d["events"][0].update(split="test"), "event outside"),
    (lambda d: d["events"][1].update(dependency_id=d["events"][0]["dependency_id"]), "duplicate event"),
    (lambda d: d["subjects"][1].update(planned=999), "coverage"),
    (lambda d: next(r for r in d["events"] if "predicted_condition" in r)["predicted_condition"].pop("training_majority"), "different support"),
])
def test_reject_inconsistent_frozen_identities_counts_or_predictions(change, match):
    source = _fixture()
    change(source)
    with pytest.raises(ValueError, match=match):
        paired_task_contrasts(source)
