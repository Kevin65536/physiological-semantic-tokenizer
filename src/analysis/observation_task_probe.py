"""Frozen observation IDs on one public task: Single-Trial MA vs labelled baseline.

The downstream-v2 owner supplies the *event*, not the waveform. Its EEG is
1–45 Hz filtered and its Single-Trial Hb is relative MBLL. G0 instead stores
native-uV EEG and first-5-s-referenced, subject-scaled Hb. Never feed one
coordinate to the other's dictionary. Join both marker clocks to the same
registered G0 score crop, then slice actual measured X (no S/O or teacher).
This is an offline, previously examined development cohort, not a test result.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data.downstream_input_cache import DownstreamInputDataset, SCHEMA_V2
from src.data.event_alignment import EVENT_ALIGNMENT_SCHEMA, window_within_alignment_support

DOWNSTREAM_ROOT = Path(__file__).resolve().parents[2] / "data/cache/physiology_downstream_v2"
DATASET = "eeg_fnirs_single_trial"
TASK = "mental_arithmetic"
SPLITS = ("train", "validation")
CLASSES = ("MA", "BL")  # downstream_input_v1 class order inherited verbatim by v2
POSITIONS = {"eeg": ("F3", "F4", "P3", "P4"), "hb": ("AF3Fp1", "FpzAFz", "AF4Fp2")}
RATES = {"eeg": 200, "hb": 10}
OFFSET = 2  # [marker+2, marker+10), four consecutive non-overlapping 2-s cells
SECONDS = 8
SEED = 17


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _event(row, split):
    """Revalidate the owner identity/label/two clocks, never infer BL from time."""
    s, r, e = row["subject"], row["record"], row["event"]
    key, i, meta = f"{DATASET}|{s}|{r}", row["event_index"], e["metadata"]
    condition = row["condition"]
    _require(row["dataset"] == DATASET and row["task"] == TASK and row["split"] == split
             and split in SPLITS and r in ("session_01", "session_03", "session_05")
             and row["join_key"] == key and e["join_key"] == key
             and e["canonical_subject_id"] == s and e["base_record_id"] == r
             and e["dataset_id"] == DATASET and e["event_index"] == i
             and e["schema"] == EVENT_ALIGNMENT_SCHEMA and e["event_type"] == "trial"
             and e["alignment_role"] == "paired_eeg_fnirs_marker" and meta["task"] == TASK
             and meta["session_idx"] == int(r[-2:]), "Single-Trial MA event identity changed")
    _require(row["target_kind"] == "classification" and condition in CLASSES
             and row["class_index"] == CLASSES.index(condition)
             and condition == e["label"] == meta["eeg_label"] == meta["fnirs_label"],
             "paired MA/BL label mismatch")
    eeg_ms, hb_ms = float(e["eeg_time_ms"]), float(e["fnirs_time_ms"])
    _require(not row["partial"] and row["duration_s"] == row["supported_seconds"] == 30
             and row["event_relative_start_s"] == -5 and row["target_event_time_s"] == 5
             and np.allclose([row["eeg_start_s"], row["hb_start_s"]],
                             np.array([eeg_ms, hb_ms]) / 1000 - 5, rtol=0, atol=1e-8)
             and window_within_alignment_support(e, -5, 30), "incorrect native event window/clock")
    _require(row["dependency_views"] == 1 and row["id"] == f"{key}|event={i:05d}|offset=-5|duration=30"
             and row["dependency_id"] == f"{key}|eeg_event_ms={eeg_ms:.6f}", "shared/mistagged event view")
    return dict(id=row["id"], dependency_id=row["dependency_id"], split=split, subject=s, record=r,
                event_index=i, condition=condition, marker_ms={"eeg": eeg_ms, "hb": hb_ms},
                start_tick={m: round((ms / 1000 + OFFSET) * RATES[m])
                            for m, ms in (("eeg", eeg_ms), ("hb", hb_ms))})


def _events(cache_root):
    root = Path(cache_root).absolute()
    _require(root == root.resolve() == DOWNSTREAM_ROOT, "exact registered downstream-v2 root required")
    verified = json.loads((root / "verification.json").read_text())
    _require(verified["schema"] == SCHEMA_V2 + "_verification" and verified["status"] == "passed"
             and verified["errors"] == [] and verified["unchanged_samples_splits_labels"] is True,
             "verified downstream_input_v2 events required")
    events, expected, cache_exclusions = [], {}, []
    for split in SPLITS:
        reader = DownstreamInputDataset(root, split=split, task=TASK)
        _require(reader.schema == SCHEMA_V2 and verified["checked_samples"] == len(reader.plan["samples"])
                 and verified["checked_records"] == len(reader.records), "downstream plan/verification mismatch")
        parts = reader.plan["subject_partitions"][DATASET]
        # The central registry includes mixed metadata; never dereference a test event or signal.
        populations = [s for members in parts.values() for s in members]
        _require(len(set(populations)) == len(populations), "downstream subject crosses partitions")
        expected[split] = sorted(parts[split])
        for row in reader.samples:
            _require(row["split"] == split and row["subject"] in expected[split], "unsafe event slice")
            events.append(_event(row, split))
        cache_exclusions += [{k: row[k] for k in ("split", "subject", "record", "event_index", "reason")}
                             for row in reader.plan["exclusions"]
                             if row["split"] == split and row["dataset"] == DATASET and row["task"] == TASK]
        del reader  # no __getitem__, no downstream signal, no nested index kept across folds
    events.sort(key=lambda v: (SPLITS.index(v["split"]), v["subject"], v["record"], v["event_index"]))
    for field in ("id", "dependency_id"):
        _require(len({e[field] for e in events}) == len(events), "duplicate event/dependency")
    ends = {}
    for e in events:
        for m, t in e["start_tick"].items():
            key = e["subject"], e["record"], m
            _require(t >= ends.get(key, 0), "overlapping or nonmonotonic marker-relative input")
            ends[key] = t + SECONDS * RATES[m]
    return events, expected, cache_exclusions


def _panel(g0_root):
    from src.data.observation_tokenizer_panel import PanelLimits, build_panel
    # These ceilings exceed this cohort (17/6 subjects, 6 sessions, <= 30 crops,
    # four/three prespecified channels). No score-, label- or quality-based search.
    return build_panel(g0_root, limits=PanelLimits(64, 64, 128, 8), seed=SEED,
                       datasets=(DATASET,), positions=POSITIONS)


def _crops(panel, expected):
    """Index disjoint *physical* score supports, independent of labels/targets."""
    groups, scales = {}, {m: {} for m in POSITIONS}
    for index, row in enumerate(panel.rows):
        m, pos = row.modality, row.position
        _require(row.dataset == DATASET and row.split in SPLITS and row.subject in expected[row.split],
                 "G0/downstream subject partition mismatch before arrays")
        _require(m in POSITIONS and pos in POSITIONS[m] and row.native_rate_hz == RATES["eeg"]
                 and row.source_unit == {"eeg": "uV", "hb": "subject_normalized_Hb"}[m],
                 "unexpected G0 position, unit or native EEG clock")
        _require(np.isfinite(row.scale) and row.scale > 0
                 and scales[m].setdefault(pos, float(row.scale)) == row.scale, "inconsistent G0 training scale")
        if row.record not in ("session_01", "session_03", "session_05"):
            continue
        _require(len(row.member_clocks) == 1, "ambiguous native member")
        record_key, window, eeg_s, hb_s = row.member_clocks[0]
        # The owner rounded at the *parent* origin, then added integer crop ticks.
        # Re-rounding (requested_origin_s + 30) fails on a half-sample float tie.
        ee, hh = row.native_start_sample, round(hb_s * RATES["hb"])
        _require(record_key == f"{DATASET}/{row.subject}/{row.record}"
                 and abs(eeg_s * RATES["eeg"] - ee) <= 0.5 + 1e-7
                 and abs(hb_s * RATES["hb"] - hh) < 1e-7, "G0 member marker clock mismatch")
        key = (row.subject, row.record, row.parent, row.start_s)
        ticks = {"eeg": ee, "hb": hh}
        crop = groups.setdefault(key, dict(parent=row.parent, start_s=row.start_s, window=window,
                                            tick=ticks, rows={}))
        _require(crop["tick"] == ticks and crop["window"] == window and (m, pos) not in crop["rows"],
                 "ambiguous G0 crop/position identity")
        crop["rows"][m, pos] = index
    by_record = defaultdict(list)
    for (subject, record, _, _), crop in groups.items():
        by_record[subject, record].append(crop)
    for crops in by_record.values():
        crops.sort(key=lambda c: c["tick"]["eeg"])
        # G0 enforces EEG score-crop disjointness. A rounded median-offset Hb clock
        # can differ slightly across segments; refuse ambiguous joins on either clock below.
        for a, b in zip(crops, crops[1:]):
            _require(b["tick"]["eeg"] >= a["tick"]["eeg"] + 30 * RATES["eeg"], "shared G0 scoring support")
    return by_record, scales


def _matches(events, expected, panel):
    """Finish the entire metadata join before touching the first measurement."""
    by_record, scales = _crops(panel, expected)
    matches, used = [], set()
    for e in events:
        hits = [crop for crop in by_record[e["subject"], e["record"]]
                if all(0 <= e["start_tick"][m] - crop["tick"][m] <= (30 - SECONDS) * r
                       for m, r in RATES.items())]
        _require(len(hits) <= 1, "ambiguous dual-clock event-to-crop mapping")
        if not hits:
            e["status"] = "no_single_registered_G0_score_crop_on_both_clocks"
            continue
        c = hits[0]
        e["g0"] = dict(parent=c["parent"], window=c["window"], start_s=c["start_s"], crop_start_tick=c["tick"])
        key = (c["parent"], c["start_s"])
        _require(key not in used, "multiple real trials share one G0 scoring crop")
        used.add(key)
        missing = [f"{m}:{p}" for m, names in POSITIONS.items() for p in names if (m, p) not in c["rows"]]
        if missing:
            e.update(status="missing_fixed_G0_position", unavailable=missing)
            continue
        matches.append((e, c))
    return matches, scales


def _event_inputs(events, expected, panel):
    matches, scales = _matches(events, expected, panel)
    waves = {m: [] for m in POSITIONS}
    for e, c in matches:
        piece, invalid = {}, []
        for m, names in POSITIONS.items():
            rate, colors = RATES[m], 1 if m == "eeg" else 2
            lo = e["start_tick"][m] - c["tick"][m]
            selected = slice(lo, lo + SECONDS * rate)
            piece[m] = []
            for pos in names:
                chunk = panel.read(c["rows"][m, pos])  # only X/support; already divided by row.scale
                x, valid = np.asarray(chunk["x"]), np.asarray(chunk["score_mask"])
                _require(x.shape == valid.shape == (15, 2 * rate, colors) and valid.dtype == np.bool_,
                         "G0 patch grid/support changed")
                x = x.reshape(-1, colors)[selected]
                if not valid.reshape(-1, colors)[selected].all() or not np.isfinite(x).all():
                    invalid.append(f"{m}:{pos}")
                piece[m].append(x.reshape(4, 2 * rate, colors).transpose(0, 2, 1).copy())
        if invalid:
            e.update(status="incomplete_observed_or_source_scoring_support", unavailable=invalid)
            continue
        e["status"] = "included"
        for m in waves:
            waves[m].append(np.stack(piece[m]))  # [position,4 patches,color,time]
    return {m: np.stack(w) for m, w in waves.items() if w}, scales


def _ids(codec, x):
    payload = codec.encode(x, mode="prototype")
    _require(set(payload) == {"pattern"}, "prototype must emit the pattern ID alone")
    values = np.asarray(payload["pattern"])
    _require(values.shape == (len(x),) and values.dtype.kind in "iu"
             and (values >= 0).all() and (values < codec.n_patterns).all(), "one in-range integer ID per patch required")
    out = np.zeros((len(x), codec.n_patterns), np.float32)
    out[np.arange(len(x)), values] = 1.
    return out


def _features(waves, codec, waveform_codec=None):
    n, pos, patches, colors, samples = waves.shape
    x = waves.reshape(-1, colors, samples)
    features = defaultdict(list)
    for begin in range(0, len(x), 256):
        batch = x[begin:begin + 256]
        features["pattern_ids"].append(_ids(codec, batch))
        features["continuous_attributes"].append(codec.pattern_features(batch))
        if waveform_codec is not None:
            features["waveform_ids_same_k"].append(_ids(waveform_codec, batch))
        payload = codec.encode(batch, mode="flat")
        _require(set(payload) == {"detail"} and np.asarray(payload["detail"]).dtype.kind in "iu"
                 and np.asarray(payload["detail"]).shape[0] == len(batch), "integer-only flat payload required")
        decoded = codec.decode({"detail": payload["detail"].copy()}, mode="flat")
        _require(decoded.shape == batch.shape and np.isfinite(decoded).all(), "invalid flat-only replay")
        features["flat_to_attributes"].append(codec.pattern_features(decoded))
        features["flat_to_pattern_ids"].append(_ids(codec, decoded))
    result = {}
    for name, chunks in features.items():
        f = np.concatenate(chunks)
        width = len(codec.pattern_feature_names) if "attributes" in name else codec.n_patterns
        _require(f.shape == (len(x), width) and np.isfinite(f).all(), "invalid per-cell readout feature")
        # The identical time reduction for continuous and categorical arms: within
        # named position, two fixed four-second blocks; one-hots become frequencies.
        result[name] = f.reshape(n, pos, patches // 2, 2, width).mean(axis=3).reshape(n, -1)
    return result


def _shuffle(y, records):
    """Permute event labels within their original training subject/session only."""
    rng, permuted = np.random.default_rng(SEED), y.copy()
    for group in sorted(set(records)):
        idx = [i for i, value in enumerate(records) if value == group]
        permuted[idx] = rng.permutation(y[idx])
    return permuted


def _score(y, pred, subjects, expected):
    rows = []
    for s in expected:
        take = subjects == s
        cm = np.bincount(2 * y[take] + pred[take], minlength=4).reshape(2, 2)
        counts = cm.sum(axis=1)
        ba = float(np.mean(cm.diagonal() / counts)) if np.all(counts > 0) else None
        rows.append(dict(subject=s, n=int(take.sum()), confusion_true_by_predicted=cm.tolist(), balanced_accuracy=ba))
    defined = [v["balanced_accuracy"] for v in rows if v["balanced_accuracy"] is not None]
    avg = float(np.mean(defined)) if defined else None
    return dict(subject_balanced_accuracy=avg if len(defined) == len(expected) else None,
                diagnostic_defined_subject_mean=avg, defined_subjects=len(defined), expected_subjects=len(expected),
                subjects=rows)


def _probes(kept, expected, features):
    y = np.array([CLASSES.index(e["condition"]) for e in kept])
    people = np.array([e["subject"] for e in kept])
    train = np.array([e["split"] == "train" for e in kept])
    groups = [(e["subject"], e["record"]) for e in kept if e["split"] == "train"]
    shuffled = _shuffle(y[train], groups)
    for event, label in zip((e for e in kept if e["split"] == "train"), shuffled, strict=True):
        event["shuffled_training_condition"] = CLASSES[label]
    counts = Counter(people[train])
    weights = np.array([1.0 / counts[s] for s in people[train]])
    weights *= len(weights) / weights.sum()  # equal people; labels not used by scaler/weights
    validation = [e for e in kept if e["split"] == "validation"]
    scores = {}

    def record(name, prediction):
        scores[name] = _score(y[~train], prediction, people[~train], expected["validation"])
        for event, label in zip(validation, prediction, strict=True):
            event.setdefault("predicted_condition", {})[name] = CLASSES[label]

    majority = np.bincount(y[train], weights=weights, minlength=2).argmax()
    record("training_majority", np.full(len(validation), majority, dtype=int))
    for arm in features["eeg"]:
        a, b = features["eeg"][arm], features["hb"][arm]
        for branch, x in (("eeg", a), ("hb", b), ("fusion", np.column_stack((a, b)))):
            z = StandardScaler().fit(x[train], sample_weight=weights)
            xt, xv = z.transform(x[train]), z.transform(x[~train])
            targets = {arm: y[train]}
            if arm == "pattern_ids":
                targets["pattern_ids_shuffled_train_labels"] = shuffled
            for readout, labels in targets.items():
                clf = LogisticRegression(C=1., solver="liblinear", tol=1e-7, max_iter=500, random_state=SEED)
                clf.fit(xt, labels, sample_weight=weights)
                _require(clf.n_iter_.max() < 500, "fixed linear probe did not converge")
                record(readout + "/" + branch, clf.predict(xv))
    return scores, int(np.sum(y[train] != shuffled))


def run_task_probe(*, cache_root, g0_root, eeg_codec, hb_codec, waveform_codecs=None) -> dict:
    """Score frozen G0-train codecs; no tuning, writing, launcher or model fitting on validation.

    Supply the verified downstream-v2 and completed G0 run roots explicitly.
    ``waveform_codecs`` optionally supplies {'eeg', 'hb'} same-K, G0-train
    waveform-routing controls. Main runner owns the frozen codecs' fit provenance
    and persistence of this JSON-serializable evidence. Run only under supervision.
    """
    codecs = {"eeg": eeg_codec, "hb": hb_codec}
    if waveform_codecs is not None:
        _require(set(waveform_codecs) == set(codecs), "both waveform-routing controls required")
    budget = {}
    for m, c in codecs.items():
        _require(c.modality == m and c.routing == "pattern" and c.n_patterns >= 2,
                 "expected the matching frozen pattern-route modality")
        budget[m] = {mode + "_bits_per_position_per_2s": int(c.rate(mode)["bits_per_item"])
                     for mode in ("flat", "prototype")}  # also checks fitted state before input access
        if waveform_codecs is not None:
            w = waveform_codecs[m]
            _require(w.modality == m and w.routing == "waveform" and w.n_patterns == c.n_patterns
                     and w.rate("prototype")["bits_per_item"] == c.rate("prototype")["bits_per_item"]
                     and w.pattern_counts_.sum() == c.pattern_counts_.sum() and w.seed == c.seed,
                     "same-K waveform controls require matched modality, budget, fit count and seed")
    events, expected, exclusions = _events(cache_root)
    waves, scales = _event_inputs(events, expected, _panel(g0_root))
    kept = [e for e in events if e["status"] == "included"]
    counts = []
    for split, subjects in expected.items():
        for s in subjects:
            rows = [e for e in events if (e["split"], e["subject"]) == (split, s)]
            counts.append(dict(split=split, subject=s, planned=len(rows), counts=dict(Counter(e["status"] for e in rows)),
                               retained_by_condition={c: sum(e["status"] == "included" and e["condition"] == c for e in rows)
                                                      for c in CLASSES}))
    result = dict(schema="frozen_observation_ma_probe_v1", dataset=DATASET, task=TASK, class_order=list(CLASSES),
                  positions={m: list(v) for m, v in POSITIONS.items()}, rates_hz=RATES.copy(),
                  codec_budget=budget, patterns_per_modality={m: c.n_patterns for m, c in codecs.items()},
                  bits_per_complete_event={mode: {m: 4 * len(POSITIONS[m]) * b[mode + "_bits_per_position_per_2s"]
                                                   for m, b in budget.items()} for mode in ("flat", "prototype")},
                  waveform_control_present=waveform_codecs is not None,
                  attributes={m: list(c.pattern_feature_names) for m, c in codecs.items()},
                  downstream_source=str(Path(cache_root).resolve()), g0_source=str(Path(g0_root).resolve()),
                  scale_by_modality_position=scales, downstream_cache_exclusions=exclusions,
                  contract=dict(input="G0 X/frozen G0 train position SD (one shared divisor for both Hb colors); no S/O, teacher validity or continuous sidecar read by the ID arms",
                                population="one labelled trial per event; MA=publisher subtraction and BL=explicit baseline trial, not an intertrial or pre-event interval",
                                scope="verified downstream-v2 mixed-metadata registry parsed by owner; consume only original train/validation MA events, zero downstream array reads; no raw/test/protected signal reads",
                                time="separate EEG/Hb marker clocks rounded to nearest sample; [marker+2,+10)s, not proof of cue-free physiology",
                                support="one containing registered disjoint 30s G0 score crop on both clocks; all named positions and all observed/scoring ticks required for every arm; overlapping 90s parent context is not independent",
                                inherited_preprocessing="G0 EEG = native voltage, Hb = offline filtered relative MBLL minus parent first-five-second mean, times existing unlabelled subject-specific Hb factor (including validation); not strict zero-shot or real time",
                                readout="within named position, average features or one-hots in [+2,+6),[+6,+10), concatenate slots; flat arms see only the decoded integer payload; continuous_attributes alone sees original X statistics",
                                fitting="subject-equal train-only StandardScaler and L2 logistic C=1, liblinear, tol=1e-7, max_iter=500; unweighted binary class decision; no tuning; frozen dictionaries and G0 scales not refitted",
                                shuffle=f"one training-label permutation within each subject/session (seed {SEED}); validation labels unchanged; negative control, not a permutation p-value",
                                primary="pattern_ids/fusion: each validation subject's two-class mean recall, then equal subject mean; undefined if any registered validation subject lacks either class",
                                interpretation="offline previously examined public development cohort; no physiological source, causal or confirmatory inference"),
                  subjects=counts, events=events, planned_events=len(events), retained_events=len(kept), scores={})
    if len({e["condition"] for e in kept if e["split"] == "train"}) != 2 or not any(e["split"] == "validation" for e in kept):
        return dict(result, status="not_estimable", reason="incomplete_training_classes_or_no_validation_support")
    features = {m: _features(waves[m], c, waveform_codecs[m] if waveform_codecs else None) for m, c in codecs.items()}
    scores, changed = _probes(kept, expected, features)
    complete = scores["pattern_ids/fusion"]["subject_balanced_accuracy"] is not None
    return dict(result, scores=scores, shuffled_training_labels_changed=changed,
                status="completed" if complete else "not_estimable", reason=None if complete else "incomplete_validation_class_or_subject_coverage")
