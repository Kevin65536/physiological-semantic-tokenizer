"""Synthetic external event contrasts; no measured arrays, retained results or fitted artifacts."""
from copy import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis import observation_task_probe as task, observation_token_events as rel
from src.tokenizers.factorized_observation import FactorizedObservationTokenizer as Codec
from tests.test_observation_task_probe import _row, _SyntheticPanel  # synthetic publisher metadata, not real identifiers


def _sin(freq):
    return np.sin(2 * np.pi * freq * np.arange(400) / 200)[None, :]


def _hb(o, r):
    t = (np.arange(20) - 9.5) / 10
    return np.stack([o * t, r * t])


@pytest.fixture(scope='module')
def codecs():
    # Actual codec, but fitted solely on generated waves. Leave many of the 16 IDs unsupported.
    exemplars = {'eeg': [_sin(6), _sin(10), _sin(20)], 'hb': [_hb(1, -1), _hb(-1, 1)]}
    return {m: Codec(m, n_patterns=16, detail_count=2, detail_bits=5, seed=17).fit(
        np.repeat(x, np.arange(len(x)) + 2, axis=0), partition='train') for m, x in exemplars.items()}


class Panel(_SyntheticPanel):
    def read(self, idx):
        row = self.rows[idx]
        m, pos = row.modality, row.position
        assert row.split in ('train', 'validation')
        self.reads.append((row.subject, row.i, pos))

        def cell(favorable):
            if m == 'hb':
                return 1.4 * (_hb(1, -1) if favorable else _hb(-1, 1))
            # No theta anywhere in the posterior ROI; no alpha in the frontal ROI.
            yes, no = (6, 20) if pos in ('F3', 'F4') else (20, 10)
            return _sin(yes if favorable else no) + 0.5 * _sin(no if favorable else yes)

        # Only the true patch grid on the *two different* clocks carries this sequence.
        start = 1802 if m == 'eeg' else 95
        wave = np.tile(cell(False).T, (15, 1))
        signal = np.concatenate([cell(row.i % 2 == 0 and j < 3).T for j in range(4)])
        wave[start:start + len(signal)] = signal
        mask = np.ones_like(wave, dtype=bool)
        mask[0] = False  # score support outside the event is immaterial
        if (row.subject, row.i, pos) in self.invalid:
            mask[start] = False
        shape = (15, 2 * task.RATES[m], wave.shape[-1])
        return {'x': wave.astype('f').reshape(shape), 'score_mask': mask.reshape(shape)}


@pytest.fixture
def source(tmp_path, monkeypatch):
    parts = {'train': ['T0', 'T1'], 'validation': [f'V{i}' for i in range(6)], 'test': ['never_read']}
    rows = [_row(s, sp, i) for sp, names in parts.items() for s in names for i in range(4)]
    panel, calls = Panel(rows), []

    class MetadataOnly:
        def __init__(self, root, *, split, task):
            assert root == tmp_path and split in ('train', 'validation') and task == 'mental_arithmetic'
            calls.append(split)
            self.schema = rel.task.SCHEMA_V2
            self.records = {i: None for i in range(9)}
            self.plan = dict(samples=rows, subject_partitions={rel.task.DATASET: parts}, exclusions=[])
            self.samples = [r for r in rows if r['split'] == split]

        def __getitem__(self, key):
            pytest.fail('downstream signal payload must never be accessed')

    (tmp_path / 'verification.json').write_text(json.dumps(dict(schema=task.SCHEMA_V2 + '_verification',
        status='passed', errors=[], checked_samples=len(rows), checked_records=9, unchanged_samples_splits_labels=True)))
    monkeypatch.setattr(task, 'DOWNSTREAM_ROOT', tmp_path)
    monkeypatch.setattr(task, 'DownstreamInputDataset', MetadataOnly)
    monkeypatch.setattr(task, '_panel', lambda root: panel if root == 'synthetic_g0' else pytest.fail('unsafe G0 root'))
    return SimpleNamespace(root=tmp_path, rows=rows, parts=parts, panel=panel, calls=calls)


def _run(source, codecs):
    return rel.run_event_relations(cache_root=source.root, g0_root='synthetic_g0',
                                   eeg_codec=codecs['eeg'], hb_codec=codecs['hb'])


def _fingerprint(rows):
    keys = sorted((r['split'], r['subject'], r['record'], r['event_index'], r['condition']) for r in rows)
    return hashlib.sha256(json.dumps(keys, separators=(',', ':')).encode()).hexdigest()


def test_actual_frozen_ids_true_clocks_fixed_windows_rois_and_raw_comparators(source, codecs, monkeypatch):
    monkeypatch.setattr(Codec, 'fit', lambda *a, **kw: pytest.fail('must not refit'))
    monkeypatch.setattr(Codec, 'decode', lambda *a, **kw: pytest.fail('centroid lookup uses integers, not a decoder'))
    monkeypatch.setattr(task, '_probes', lambda *a: pytest.fail('no classifier'))
    out = _run(source, codecs)
    assert out['status'] == 'completed' and out['retained_events'] == out['planned_events'] == 32
    assert source.calls == ['train', 'validation'] and len(source.panel.reads) == 32 * 7
    assert out['kept_keys_sha256'] == _fingerprint([r for r in source.rows if r['split'] != 'test'])
    assert out['periods_s'] == {'all': [2, 10], 'early': [2, 6], 'late': [6, 10]}
    assert out['primary'] == dict(split='validation', key='hb_joint_rise_fall_occupancy/all',
                                  expected_MA_minus_BL='increase', word_support='observed')
    val = out['by_split']['validation']; est = val['estimates']; hb = 'hb_joint_rise_fall_occupancy'
    assert (val['n_paired_subjects'], val['n_registered_subjects'], val['n_paired_sessions']) == (6, 6, 6)
    assert val['paired_by_condition'] == {'MA': 12, 'BL': 12}
    assert out['by_split']['train']['n_paired_subjects'] == 2 and out['by_split']['train']['primary_sign_flip'] is None
    assert val['paired_subject_bootstrap'] == dict(method='exact_empirical_subject_percentile', draws=6**6, seed=None)
    assert est[hb + '/all']['full_cohort_ci95'] == [.75, .75]
    assert est[hb + '/all']['paired_mean_MA'] == .75 and est[hb + '/all']['paired_mean_BL'] == 0
    assert val['primary_sign_flip']['two_sided_p'] == 2/64 and val['primary_sign_flip']['n_sign_vectors'] == 64
    for period, diff in [('all', .75), ('early', 1.), ('late', .5)]:
        def get(name):
            return est[name + '/' + period]['paired_mean_MA_minus_BL']
        assert get(hb) == get('hb_x_joint_rise_fall_fraction') == diff
        assert get('hb_centroid_signed_opposition') == pytest.approx(4 * diff)
        assert get('hb_x_signed_opposition') == pytest.approx(1.4 * 4 * diff, abs=1e-6)
        for b, sign in [('posterior_alpha', -1), ('frontal_theta', 1)]:
            assert get(b + '_enriched_occupancy') == sign * diff
            assert get(b + '_centroid_fraction') == pytest.approx(sign * diff)
            assert get(b + '_x_fraction') == pytest.approx(sign * .6 * diff, abs=1e-6)
    assert val['paired_word_support_all'][hb]['MA'] == dict(hits=108, cells=144, events_with_hit=12, events=12,
                                                           subjects_with_hit=6)
    assert all(e['used_in_contrast'] and 'predicted_condition' not in e for e in out['events'])
    assert all(np.shape(e['pattern_ids']['eeg']) == (4, 4) and np.shape(e['pattern_ids']['hb']) == (3, 4)
               for e in out['events'])
    json.dumps(out, allow_nan=False)


def test_validation_labels_change_external_comparison_not_word_definitions(source, codecs):
    original = _run(source, codecs)
    for row in source.rows:
        if row['split'] == 'validation':
            row['class_index'] = 1 - row['class_index']
            row['condition'] = row['event']['label'] = task.CLASSES[row['class_index']]
            row['event']['metadata'].update(eeg_label=row['condition'], fnirs_label=row['condition'])
    changed = _run(source, codecs)
    assert original['dictionaries'] == changed['dictionaries'] and original['kept_keys_sha256'] != changed['kept_keys_sha256']
    assert original['by_split']['train'] == changed['by_split']['train']
    assert [e['values'] for e in original['events']] == [e['values'] for e in changed['events']]
    assert changed['by_split']['validation']['estimates'][rel._PRIMARY]['full_cohort_mean_MA_minus_BL'] == -.75


def test_incomplete_support_keeps_unpaired_people_events_and_denominators(source, codecs):
    source.panel.invalid.update({('V0', 0, 'P4'), ('V0', 2, 'P4')})  # no retained MA for V0
    out = _run(source, codecs); val = out['by_split']['validation']; effect = val['estimates'][rel._PRIMARY]
    assert out['status'] == 'not_estimable_for_full_validation_cohort'
    assert (val['n_registered_subjects'], val['n_paired_subjects'], val['n_paired_sessions']) == (6, 5, 5)
    assert (val['planned_events'], val['retained_events'], val['paired_events']) == (24, 22, 20)
    assert val['retained_by_condition'] == {'MA': 10, 'BL': 12} and val['paired_by_condition'] == {'MA': 10, 'BL': 10}
    assert val['primary_sign_flip'] is None and val['paired_subject_bootstrap']['draws'] == 5**5
    assert effect['full_cohort_ci95'] is None and effect['full_cohort_mean_MA_minus_BL'] is None
    assert effect['paired_subject_ci95'] == [.75, .75] and effect['subject_direction']['undefined'] == 1
    v0 = [e for e in out['events'] if e['subject'] == 'V0']
    assert all(not e['used_in_contrast'] for e in v0)
    assert [e['status'] for e in v0 if e['condition'] == 'MA'] == ['incomplete_observed_or_source_scoring_support'] * 2
    assert out['kept_keys_sha256'] == _fingerprint([e for e in out['events'] if e['status'] == 'included'])


def test_session_first_class_balancing_then_subject_equal_never_cross_sessions():
    rows, values = [], []
    def add(subject, record, cls, value, n):
        for _ in range(n):
            rows.append(dict(subject=subject, record=record, split='train', condition=cls, status='included', id=str(len(rows))))
            values.append(value)
    add('A', 'session_01', 'MA', 1, 10); add('A', 'session_01', 'BL', 0, 1)   # +1
    add('A', 'session_03', 'MA', 0, 1); add('A', 'session_03', 'BL', 1, 2)    # -1 -> A=0
    add('B', 'session_05', 'MA', 1, 1); add('B', 'session_05', 'BL', 0, 1)    # B=1
    add('C', 'session_01', 'MA', 1, 20); add('C', 'session_03', 'BL', 0, 20)  # not paired
    args = ('train', list('ABCD'), rows, {e['id']: i for i, e in enumerate(rows)}, np.array(values)[:, None],
            [rel._PRIMARY], {}, {})
    result = rel._summarize(*args); effect = result['estimates'][rel._PRIMARY]
    assert result['n_paired_subjects'] == 2 and result['n_paired_sessions'] == 3
    assert result['paired_by_condition'] == {'MA': 12, 'BL': 4}
    assert [p['ma_minus_bl'] for p in result['subjects']] == [{rel._PRIMARY: 0}, {rel._PRIMARY: 1}, None, None]
    assert effect['paired_mean_MA_minus_BL'] == .5 and effect['paired_mean_MA'] == .75 and effect['paired_mean_BL'] == .25
    assert effect['paired_subject_ci95'] == [0, 1] and effect['full_cohort_mean_MA_minus_BL'] is None
    assert rel._summarize(*args) == result  # deterministic subject-bootstrap control


def test_exact_primary_no_spurious_signs_on_cancelling_occupancy_fractions():
    rows, values = [], []
    subjects = [str(j) for j in range(6)]
    for s in subjects:
        # (1/2 - 1/6) + (0 - 1/3) == 0 mathematically but != 0 in ordinary floating subtraction.
        for r, cls, hits in [('session_01', 'MA', 6), ('session_01', 'BL', 2),
                             ('session_03', 'MA', 0), ('session_03', 'BL', 4)]:
            rows.append(dict(subject=s, record=r, split='validation', condition=cls, status='included', id=str(len(rows))))
            values.append(hits / 12)
    result = rel._summarize('validation', subjects, rows, {e['id']: i for i, e in enumerate(rows)},
                            np.array(values)[:, None], [rel._PRIMARY], {}, {})
    assert result['estimates'][rel._PRIMARY]['full_cohort_ci95'] == [0, 0]
    assert result['estimates'][rel._PRIMARY]['subject_direction']['tie'] == 6
    assert result['primary_sign_flip']['two_sided_p'] == 1


def test_word_names_follow_trained_centroids_not_integer_slot_or_unused_fallback(source, codecs):
    original = _run(source, codecs)
    remapped = {}
    for modality, codec in codecs.items():
        new = copy(codec)
        # Permute all learned physical profiles and IDs together, not the validation data.
        new.pattern_counts_ = codec.pattern_counts_[::-1].copy()
        new.pattern_centers_ = codec.pattern_centers_[::-1].copy()
        if modality == 'hb':  # fallback values must never create qualifying words
            new.pattern_centers_[new.pattern_counts_ == 0, 2:] = [20, -20]
        def encode(x, mode, c=codec):
            assert mode == 'prototype'
            return {'pattern': 15 - c.encode(x, mode=mode)['pattern']}
        new.encode = encode
        remapped[modality] = new
    changed = _run(source, remapped)
    a = original['dictionaries']['hb']['groups'][rel._HB_WORD]
    b = changed['dictionaries']['hb']['groups'][rel._HB_WORD]
    assert b['ids'] == [15-a['ids'][0]] and a['n_words'] == b['n_words'] == 1
    assert original['by_split']['validation'] == changed['by_split']['validation']
    # A supported dictionary with no jointly opposite slopes remains the original primary, exactly zero.
    empty = Codec('hb', n_patterns=16, seed=17).fit(np.stack([_hb(1, 2), _hb(-2, -1)]), partition='train')
    out = _run(source, {**codecs, 'hb': empty}); val = out['by_split']['validation']
    assert out['primary']['key'] == rel._PRIMARY and out['primary']['word_support'] == 'empty_training_word_set'
    assert out['dictionaries']['hb']['groups'][rel._HB_WORD]['ids'] == []
    assert val['estimates'][rel._PRIMARY]['full_cohort_ci95'] == [0, 0] and val['primary_sign_flip']['two_sided_p'] == 1
    assert val['estimates']['hb_x_signed_opposition/all']['paired_mean_MA_minus_BL'] > 0


def test_unsafe_roots_or_codecs_rejected_and_no_support_serializes(source, codecs, monkeypatch):
    kwargs = dict(cache_root=source.root, g0_root='synthetic_g0', eeg_codec=codecs['eeg'], hb_codec=codecs['hb'])
    with pytest.raises(ValueError, match='explicit downstream and G0 roots'):
        rel.run_event_relations(**{**kwargs, 'g0_root': None})
    with monkeypatch.context() as m:
        m.setattr(task, 'SPLITS', ('train', 'validation', 'test'))
        with pytest.raises(ValueError, match='event contract drift'):
            rel.run_event_relations(**kwargs)
    with pytest.raises(ValueError, match='registered downstream'):
        rel.run_event_relations(cache_root=source.root / 'other', g0_root='synthetic_g0',
                                eeg_codec=codecs['eeg'], hb_codec=codecs['hb'])
    blind = copy(codecs['eeg']); blind.routing = 'waveform'
    with pytest.raises(ValueError, match='original main pattern'):
        _run(source, {**codecs, 'eeg': blind})
    assert not source.calls and not source.panel.reads
    source.panel.invalid.update({(r.subject, r.i, 'F3') for r in source.panel.rows})
    out = _run(source, codecs)
    assert out['retained_events'] == 0 and out['kept_keys_sha256'] == _fingerprint([])
    assert out['primary']['word_support'] == 'no_paired_events'
    for split in out['by_split'].values():
        assert split['n_paired_subjects'] == 0 and split['primary_sign_flip'] is None
        assert split['estimates'][rel._PRIMARY]['paired_mean_MA_minus_BL'] is None
    json.dumps(out, allow_nan=False)
