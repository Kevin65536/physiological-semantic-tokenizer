"""Paired, subject-level uncertainty for the frozen public MA task probe.

A pure report reduction: no dataset, array file, model, RNG-dependent selection
or fitted probe is opened. The recorded validation predictions must reproduce
the reported per-subject confusion matrices and BA on exactly the same events.
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np


_CLASSES = ("MA", "BL")
_BRANCHES = ("fusion", "eeg", "hb")
_REFERENCES = ("waveform_ids_same_k", "pattern_ids_shuffled_train_labels", "continuous_attributes",
               "flat_to_pattern_ids", "flat_to_attributes", "training_majority")


def _check(ok, message):
    if not ok:
        raise ValueError(message)


def _ba(cm):
    n0, n1 = cm.sum(axis=1)
    return float((cm[0, 0] / n0 + cm[1, 1] / n1) / 2) if n0 and n1 else None


def _equal_score(reported, recomputed):
    return (reported is recomputed if recomputed is None or reported is None
            else math.isclose(reported, recomputed, abs_tol=1e-12, rel_tol=0))


def _audit(result, arms):
    declared = {r["subject"]: r for r in result["subjects"]}
    _check(len(declared) == len(result["subjects"]) and all(r["split"] in ("train", "validation")
           for r in declared.values()), "duplicate, cross-split or nondevelopment subject")
    validation = sorted(s for s, r in declared.items() if r["split"] == "validation")
    cm = {a: {s: np.zeros((2, 2), dtype=int) for s in validation} for a in arms}
    reasons = {s: Counter() for s in declared}
    retained = {s: Counter() for s in declared}
    ids, dependencies = set(), set()
    for e in result["events"]:
        s, status, label = e["subject"], e["status"], e["condition"]
        _check(s in declared and e["split"] == declared[s]["split"] and label in _CLASSES,
               "event outside its declared subject, partition or classes")
        _check(e["id"] not in ids and e["dependency_id"] not in dependencies, "duplicate event/dependency")
        ids.add(e["id"]); dependencies.add(e["dependency_id"])
        reasons[s][status] += 1
        if status == "included":
            retained[s][label] += 1
        if e["split"] == "validation" and status == "included" and arms:
            preds = e["predicted_condition"]
            _check(set(preds) == set(arms), "validation predictions missing an arm or on different support")
            for a, pred in preds.items():
                _check(pred in _CLASSES, "non-native predicted condition")
                cm[a][s][_CLASSES.index(label), _CLASSES.index(pred)] += 1
        else:
            _check("predicted_condition" not in e, "unexpected prediction outside scored validation support")
    _check(len(ids) == result["planned_events"] and sum(v.total() for v in retained.values()) == result["retained_events"],
           "planned/retained event denominator mismatch")
    prior = {k: Counter() for k in ("train", "validation")}
    for e in result["downstream_cache_exclusions"]:
        _check(e["subject"] in declared and declared[e["subject"]]["split"] == e["split"], "unexpected cache exclusion")
        prior[e["split"]][e["reason"]] += 1
    coverage = {}
    for split in prior:
        subjects, combined = [], Counter()
        for s in sorted(k for k, r in declared.items() if r["split"] == split):
            r, counts = declared[s], reasons[s]
            by_class = {c: retained[s][c] for c in _CLASSES}
            _check(counts.total() == r["planned"] and dict(counts) == r["counts"] and by_class == r["retained_by_condition"],
                   "per-subject coverage inconsistent with event inventory")
            missing = {k: v for k, v in counts.items() if k != "included"}
            subjects.append(dict(subject=s, planned=r["planned"], retained=counts["included"],
                                 rejected=sum(missing.values()), rejection_reasons=missing, retained_by_condition=by_class))
            combined.update(counts)
        coverage[split] = dict(subjects=subjects, planned=combined.total(), retained=combined["included"],
                               rejected=combined.total()-combined["included"],
                               rejection_reasons={k: v for k, v in combined.items() if k != "included"},
                               prior_downstream_cache_exclusions=dict(prior[split]))
    for arm, by_subject in cm.items():
        outcome = result["scores"][arm]
        rows = {r["subject"]: r for r in outcome["subjects"]}
        _check(set(rows) == set(validation) and len(rows) == len(outcome["subjects"]), "unmatched validation subjects")
        bas = []
        for s, matrix in by_subject.items():
            r, ba = rows[s], _ba(matrix)
            _check(r["n"] == matrix.sum() and np.array_equal(matrix, r["confusion_true_by_predicted"])
                   and _equal_score(r["balanced_accuracy"], ba), "frozen per-subject score disagrees with predictions")
            if ba is not None:
                bas.append(ba)
        mean = sum(bas) / len(bas) if bas else None
        _check(len(rows) == outcome["expected_subjects"] and len(bas) == outcome["defined_subjects"]
               and _equal_score(outcome["diagnostic_defined_subject_mean"], mean)
               and _equal_score(outcome["subject_balanced_accuracy"], mean if len(bas) == len(rows) else None),
               "frozen subject-equal score/denominator mismatch")
    return coverage, cm, validation


def _interval(diffs):
    """Exact empirical paired subject bootstrap for this six-subject validation cohort."""
    n = len(diffs)
    if n < 2:
        return None, 0
    _check(n <= 6, "unexpected MA validation cohort size; review before a different bootstrap contract")
    sums = np.array([0.])
    for _ in range(n):
        sums = (sums[:, None] + diffs).reshape(-1)  # all ordered n-out-of-n draws, equal probability
    return np.quantile(sums / n, [.025, .975], method="inverted_cdf").tolist(), len(sums)


def _trial_bits(result, mode, branch):
    if mode is None:  # unsized, unquantized original attributes; not zero bits
        return None
    if mode == "constant":
        return 0
    bits = result["bits_per_complete_event"][mode]
    return bits[branch] if branch != "fusion" else bits["eeg"] + bits["hb"]


def paired_task_contrasts(task_result: dict) -> dict:
    """Summarize pattern-ID minus comparator BA, keyed by validation subject.

    The percentile interval conditions on the already-frozen training draws,
    preprocessing, codebooks, linear probes, eligible events and the one shuffle.
    Do not interpret its sign as a multiplicity-adjusted test or as estimation of
    between-training-run variability. Undefined subjects remain in every denominator.
    """
    result = task_result
    _check(result["schema"] == "frozen_observation_ma_probe_v1" and result["dataset"] == "eeg_fnirs_single_trial"
           and result["task"] == "mental_arithmetic" and tuple(result["class_order"]) == _CLASSES
           and result["status"] in ("completed", "not_estimable"), "frozen Single-Trial MA probe result required")
    refs = [a for a in _REFERENCES if a != "waveform_ids_same_k" or result["waveform_control_present"]]
    expected = {"training_majority"} | {f"{a}/{b}" for a in ["pattern_ids", *refs] if a != "training_majority" for b in _BRANCHES}
    arms = set(result["scores"])
    _check(arms == expected or (not arms and result["status"] == "not_estimable"), "incomplete or unknown frozen task arm set")
    coverage, matrices, subjects = _audit(result, arms)
    if arms:
        complete = bool(subjects) and all(_ba(m) is not None for m in matrices["pattern_ids/fusion"].values())
        _check(complete == (result["status"] == "completed"), "completion status disagrees with validation coverage")
    contrasts = []
    for branch in _BRANCHES if arms else ():
        left = "pattern_ids/" + branch
        for ref in refs:
            right = ref if ref == "training_majority" else f"{ref}/{branch}"
            direction = dict(higher=0, tie=0, lower=0, undefined=0)
            pairs, diffs = [], []
            for s in subjects:
                a, b = matrices[left][s], matrices[right][s]
                n0, n1 = map(int, a.sum(axis=1))
                # Exact sign, including ties, for recalls computed on identical class counts.
                delta = None
                if n0 and n1:
                    numerator = int(a[0, 0] - b[0, 0]) * n1 + int(a[1, 1] - b[1, 1]) * n0
                    delta = numerator / (2 * n0 * n1)
                    diffs.append(delta)
                direction["undefined" if delta is None else "higher" if delta > 0 else "lower" if delta < 0 else "tie"] += 1
                pairs.append(dict(subject=s, n_trials=n0+n1, true_ma=n0, true_bl=n1,
                                  pattern_ba=_ba(a), reference_ba=_ba(b), delta_ba=delta))
            complete = bool(subjects) and len(diffs) == len(subjects)
            avg = sum(diffs) / len(diffs) if diffs else None
            ci, draws = _interval(np.asarray(diffs)) if complete else (None, 0)
            mode = ("constant" if ref == "training_majority" else None if ref == "continuous_attributes"
                    else "flat" if ref.startswith("flat_to_") else "prototype")
            contrasts.append(dict(branch=branch, pattern=left, reference=right, n_expected_subjects=len(subjects),
                                  n_paired=len(diffs), mean_delta_ba=avg if complete else None,
                                  defined_pairs_mean_delta_ba=avg, ci95=ci, bootstrap_draws=draws,
                                  direction_counts=direction, per_subject=pairs,
                                  pattern_bits_per_trial=_trial_bits(result, "prototype", branch),
                                  reference_bits_per_trial=_trial_bits(result, mode, branch)))
    return dict(schema="frozen_observation_ma_paired_contrasts_v1", source_schema=result["schema"],
                dataset=result["dataset"], task=result["task"], source_status=result["status"],
                class_order=list(_CLASSES), unit="balanced accuracy on [0,1]; delta = pattern IDs minus reference",
                uncertainty="Exploratory pointwise 95% percentile intervals, exact empirical CDF of n**n equally likely paired validation-subject resamples, quantiles at .025/.975. No resampling trials, retraining, or multiple-comparison correction. No full-cohort interval if a subject is undefined; <2 subjects cannot estimate uncertainty.",
                coverage=coverage, waveform_control_present=result["waveform_control_present"], contrasts=contrasts)
