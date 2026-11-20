"""Synthetic 2 s ID series only. No G0 arrays, saved model, atlas or run reads."""
from itertools import permutations, product

import numpy as np
import pandas as pd
import pytest

from src.analysis.observation_token_dynamics import _panel, run_dynamics
from src.tokenizers.factorized_observation import FactorizedObservationTokenizer


def _series(ids, *, dataset='A', subject='train1', record='r1', parent='p', start_s=0,
            position='site', patch=None, blind=None):
    return pd.DataFrame(dict(dataset=dataset, subject=subject, record=record, parent=parent,
                             start_s=start_s, position=position, pattern=ids,
                             patch=np.arange(len(ids)) if patch is None else patch,
                             blind=np.zeros(len(ids), dtype=int) if blind is None else blind, rms=1.0))


def _atlas(k, mod='eeg'):
    names = FactorizedObservationTokenizer(mod).pattern_feature_names
    rows = []
    for i in range(k):
        x = np.eye(len(names))[i % len(names)] if mod == 'eeg' else [0., 0., float(i), -float(i)]
        rows.extend(dict(modality=mod, split='train', pattern_id=i, feature=f, median=y) for f, y in zip(names, x))
    a = pd.DataFrame(rows)
    # Impossible validation medians would cause obvious drift if used as training names/features.
    return pd.concat([a, a.assign(split='validation', median=-1e6)], ignore_index=True)


def _table(result, metric, scope='all', route='pattern', level='macro'):
    t = result['scores']
    return t[(t.metric == metric) & (t.scope == scope) & (t.routing == route) & (t.level == level)]


def _balanced(ids, split, **kw):
    return pd.concat([_series(ids, dataset=d, subject=f'{split}{i}', **kw)
                      for d in ('A', 'B') for i in (1, 2)], ignore_index=True)


def _counts(runs, k):
    a = np.zeros((k, k))
    for r in runs:
        for i, j in zip(r[:-1], r[1:]):
            a[i, j] += 1
    return a


def test_matches_enumerated_permutations_and_exact_conditional_log_score():
    k = 3
    train_ids = [0, 0, 1, 2, 0, 1, 1, 0, 2, 2, 0]
    train = _series(train_ids)
    # Two disjoint runs in one site/window; reversing input rows is not time reversal.
    runs = [[0, 0, 1, 2], [2, 1, 2]]
    val = _series(runs[0] + runs[1], patch=[1, 2, 3, 4, 9, 10, 11], subject='val').iloc[::-1]
    exact = np.mean([_counts((u, v), k) for u, v in product(permutations(runs[0]), permutations(runs[1]))], axis=0)
    seen = _counts(runs, k)
    panel = _panel(val, 'validation', 'eeg', k)
    actual, expected = panel.matrices('pattern', k)
    np.testing.assert_array_equal(actual[0], seen)
    np.testing.assert_allclose(expected[0], exact)
    assert actual.sum() == expected.sum() == 5

    tr = _counts([train_ids], k)
    # One dataset/site: hierarchy weights recover the raw train destination/pair counts.
    q = np.full(k, 1/k)
    for _ in range(3):  # global, dataset, site, each train-only
        q = (tr.sum(axis=0) + k*q) / (tr.sum() + k)
    p = (tr + k*q) / (tr.sum(axis=1, keepdims=True) + k)
    result = run_dynamics(train, val, _atlas(k), 'eeg')
    for scope, mask in [('all', np.ones((k, k))), ('switch', 1 - np.eye(k))]:
        row = _table(result, 'history_gain_bits', scope).iloc[0]
        for c, prefix in [(seen, ''), (exact, 'null_')]:
            for prob, name in [(np.broadcast_to(q, (k, k)), 'occupancy_bits'), (p, 'history_bits')]:
                logp = np.log2(prob / (prob * mask).sum(axis=1, keepdims=True))
                assert row[prefix+name] == pytest.approx(-np.sum(c*mask*logp)/np.sum(c*mask))
        assert row.observed == pytest.approx(row.occupancy_bits - row.history_bits)
        assert row.null_mean == pytest.approx(row.null_occupancy_bits - row.null_history_bits)
        assert row.expected_pairs == pytest.approx((exact * mask).sum())
        assert row.delta == pytest.approx(row.observed - row.null_mean)


@pytest.mark.parametrize(('key', 'other'), [('dataset', 'B'), ('subject', 's2'), ('record', 'r2'),
                                          ('parent', 'q'), ('start_s', 30), ('position', 'other')])
def test_no_pair_or_null_across_any_physical_identity(key, other):
    f = _series([0, 1]); f.loc[1, key] = other
    o, e = _panel(f, 'train', 'eeg', 3).matrices('pattern', 3)
    assert o.sum() == e.sum() == 0


def test_forward_gain_reverses_on_heldout_without_changing_edges_labels_or_null():
    ids = [0, 1, 2] * 5
    train = _balanced(ids, 'train')
    atlas = _atlas(3)
    val = _balanced(ids, 'val')
    fwd = run_dynamics(train, val, atlas, 'eeg')
    poison = atlas.loc[atlas.split == 'train'].assign(split='validation', pattern_id=99, median=np.inf)
    rev = run_dynamics(train.sample(frac=1, random_state=7), _balanced(ids[::-1], 'val'),
                       pd.concat([poison, atlas], ignore_index=True), 'eeg')
    for scope in ('all', 'switch'):
        f = _table(fwd, 'history_gain_bits', scope).iloc[0]
        r = _table(rev, 'history_gain_bits', scope).iloc[0]
        assert f.observed_ci_low > 0 and f.ci_low > 0
        assert r.observed_ci_high < 0 and r.ci_high < 0
        assert r.null_mean == pytest.approx(f.null_mean)
        assert f.n_positive == r.n_negative == f.n_subjects == 4
    # The constant blind control has the SAME allowed K, not semantic labels or a made-up switch score.
    blind = _table(fwd, 'self_rate', route='blind').iloc[0]
    assert blind.k == 3 and blind.observed == blind.null_mean == 1
    switch = _table(fwd, 'history_gain_bits', 'switch', route='blind').iloc[0]
    assert switch.n_subjects == switch.n_pairs == 0 and np.isnan(switch.observed)
    assert _table(fwd, 'atlas_closeness', route='blind').empty
    e = fwd['edges'].query("level == 'macro'").sort_values('rank').reset_index(drop=True)
    z = rev['edges'].query("level == 'macro'").sort_values('rank').reset_index(drop=True)
    assert set(zip(e.source_id, e.target_id)) == {(0, 1), (1, 2), (2, 0)}
    cols = ['rank', 'source_id', 'target_id', 'source_label', 'target_label', 'train_delta', 'null_mean', 'expected_count']
    pd.testing.assert_frame_equal(e[cols], z[cols])
    assert e.source_label.str.contains('Hz').all() and (e.train_support_subjects == 4).all()
    assert (e.train_support_datasets == 2).all() and (e.n_support_subjects == 4).all() and (e.ci_low > 0).all()
    # Do not select away absent validation edges, nor withhold negative evidence for lack of hits.
    assert (z.n_support_subjects == 0).all() and (z.observed_count == 0).all()
    assert (z.n_opportunity_subjects == 4).all() and (z.n_negative == 4).all() and (z.ci_high < 0).all()


def test_all_bag_orders_give_no_excess_even_for_a_strong_train_model():
    ids = [0, 1, 2] * 5
    v = pd.concat([_series(p, start_s=30*i, subject='validation')
                   for i, p in enumerate(permutations([0, 1, 2]))], ignore_index=True)
    result = run_dynamics(_series(ids), v, _atlas(3), 'eeg')
    for scope in ('all', 'switch'):
        row = _table(result, 'history_gain_bits', scope).iloc[0]
        assert row.observed == pytest.approx(row.null_mean, abs=1e-12)
        assert row.n_tied == 1
    # Repeated events from just one person/library do not qualify a readable edge.
    assert result['edges'].empty and 'n_opportunity_datasets' in result['edges']


def test_binary_switch_gain_cannot_claim_self_persistence_as_sequence_information():
    ids = [0]*7 + [1]*8
    result = run_dynamics(_balanced(ids, 'train'), _balanced(ids, 'val'), _atlas(2), 'eeg')
    assert _table(result, 'self_rate').iloc[0].delta > .4
    assert _table(result, 'history_gain_bits').iloc[0].delta > .5
    row = _table(result, 'history_gain_bits', 'switch').iloc[0]
    assert row.n_pairs == 4 and row.n_tied == 4
    assert row.occupancy_bits == row.history_bits == row.observed == row.null_mean == 0


def test_hierarchy_weights_unscorable_subject_and_no_patch_level_interval():
    v = pd.concat([_series([0]*15, subject='v1'), _series([0, 1], subject='v1', position='other'),
                   _series([0]*3, subject='v1', start_s=30), _series([0, 1, 0, 1], subject='v1', record='r2'),
                   _series([0, 0], subject='v2'), _series([0, 1, 0, 1], dataset='B', subject='v3'),
                   _series([1], dataset='B', subject='v4')], ignore_index=True)
    result = run_dynamics(_series([0, 1, 0, 0]), v, _atlas(2), 'eeg')
    sub = _table(result, 'self_rate', level='subject').set_index('subject')
    assert sub.loc['v1', 'observed'] == pytest.approx((.5*(.5 + 1) + 0)/2)
    assert np.isnan(sub.loc['v4', 'observed']) and sub.loc['v4', 'n_subjects'] == 0
    assert sub.ci_low.isna().all()
    ds = _table(result, 'self_rate', level='dataset').set_index('dataset')
    assert ds.loc['A', 'observed'] == pytest.approx(11/16) and ds.loc['B', 'observed'] == 0
    row = _table(result, 'self_rate').iloc[0]
    assert row.observed == pytest.approx(11/32)
    assert row.n_subjects == 3 and row.n_planned_subjects == 4
    assert row.n_datasets == 2 and np.isnan(row.ci_low)  # B has only one scored person
    cover = result['coverage'].query("level == 'macro' and split == 'validation'").iloc[0]
    assert (cover.n_patches, cover.n_pairs, cover.n_series, cover.n_subjects, cover.n_series_with_pairs) == (31, 24, 7, 4, 6)


def test_train_hb_trend_distance_has_units_and_handles_unnamed_ids():
    train, val = _balanced([0, 1, 2, 3]*3, 'train'), _balanced([0, 1, 2, 3], 'val')
    atlas = _atlas(5, 'hb'); atlas.loc[atlas.pattern_id == 4, 'median'] = np.nan
    result = run_dynamics(train, val, atlas, 'hb')
    row = _table(result, 'atlas_closeness', 'switch').iloc[0]
    assert row.observed == -1 and row.null_mean == pytest.approx(-5/3)
    assert row.delta == pytest.approx(2/3) and row.ci_low > 0
    assert result['edges'].source_label.str.contains('HbO slope').all() and not result['edges'].empty
    sparse = run_dynamics(train, _series([0, 4, 1], subject='v', position='unseen'), atlas, 'hb')
    sem = _table(sparse, 'atlas_closeness').iloc[0]
    assert np.isnan(sem.observed) and sem.n_pairs == 0
    h = _table(sparse, 'history_gain_bits').iloc[0]
    assert np.isfinite(h.observed) and h.n_pairs == 2  # unseen site and train-inactive ID are smoothed


@pytest.mark.parametrize(('make_val', 'match'), [
    (lambda a: pd.concat([a, a.iloc[:1]]), 'duplicate patch'),
    (lambda a: a.assign(patch=[0, 1, 15]), 'G0 patch'),
    (lambda a: a.assign(patch=[0., 1.5, 2.]), 'integer'),
    (lambda a: a.assign(blind=[0, 1, 3]), 'blind outside'),
    (lambda a: a.assign(subject='train1'), 'subject overlap'),
    (lambda a: a.assign(rms=[1, np.nan, 1]), 'rms'),
    (lambda a: a.assign(split='forbidden'), 'unexpected split'),
    (lambda a: a.assign(modality='hb'), 'unexpected modality'),
])
def test_invalid_contract_is_rejected(make_val, match):
    with pytest.raises(ValueError, match=match):
        run_dynamics(_series([0, 1, 2]), make_val(_series([1, 2, 0], subject='val')), _atlas(3), 'eeg')


def test_no_training_pairs_is_not_evidence_of_an_empty_effect():
    with pytest.raises(ValueError, match='no adjacent training'):
        run_dynamics(_series([0, 1], patch=[1, 3]), _series([0, 1], subject='val'), _atlas(2), 'eeg')
