"""Synthetic DataFrames only. Never load a G0 panel, a dictionary, or a data split."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import observation_token_coupling as c


EFFECT = np.array([1., -2., 3., -1., 2., -3., 4., -4., -1.5, .4, 1.7, -.5, 2.5, -.9, 0., 1.2])
HB_CODES = np.array([7, 14, 0, 11, 3, 8, 13, 5, 12, 4, 2, 15, 9, 6, 1, 10])


def _synth(split, n_subjects=5, crops=3, datasets=('A', 'B'), *, planted=True):
    rng = np.random.default_rng(61 if split == 'train' else 1703)
    eeg, hb = [], []; t = 15
    for d in datasets:
        for s in range(n_subjects):
            for w in range(crops):
                common = dict(dataset=d, subject=f'{split}{s}', record='R', parent='P', start_s=30*w, split=split)
                words = rng.integers(c.K, size=(t, 2))
                # Positive energies of three orthogonal frequency components.
                energy = rng.gamma(2., 1., size=(t, 2, 3))
                powers = energy / energy.sum(axis=2, keepdims=True)
                blind = rng.integers(c.K, size=(t, 2))
                source = words if planted else rng.integers(c.K, size=(t, 2))
                delayed = np.r_[rng.integers(c.K, size=(3, 2)), source[:-3]]
                driver = EFFECT[delayed].mean(axis=1)
                for p in range(t):
                    for pos in range(2):
                        key = dict(**common, patch=p)
                        eeg.append(dict(**key, position=f'E{pos}', modality='eeg', pattern=words[p, pos],
                                        blind=blind[p, pos], rms=np.sqrt(energy[p, pos].sum()),
                                        alpha=powers[p, pos, 0], beta=powers[p, pos, 1],
                                        hbo=np.nan, hbr=np.nan, so=np.nan, sr=np.nan))
                        # Hb labels use a separate, permuted dictionary, never equality of EEG/Hb IDs.
                        o, r = driver[p] + pos + rng.normal(0, .15), -.7*driver[p] + rng.normal(0, .15)
                        hb.append(dict(**key, position=f'H{pos}', modality='hb', pattern=HB_CODES[delayed[p, pos]],
                                       blind=0, rms=np.sqrt((o**2+r**2)/2+.1), hbo=o, hbr=r,
                                       so=rng.normal(), sr=rng.normal(), alpha=np.nan, beta=np.nan))
    return pd.DataFrame(eeg), pd.DataFrame(hb)


def _result(planted):
    return c.run_coupling(*_synth('train', planted=planted), *_synth('validation', planted=planted))


def _score(result, *, lag=6, endpoint='hb_delta', reference='base', candidate='pattern'):
    df = result['summary']
    return df[(df.lag_s == lag) & (df.endpoint == endpoint) & (df.reference == reference)
              & (df.candidate == candidate)].iloc[0]


def test_known_six_second_delay_and_distinct_future_word_endpoint():
    result = _result(True)
    assert result['primary']['status'] == 'above_controls'
    score = _score(result)
    # 15 patches - 3 past for the equal-delay control - 3 future. Two outputs share each origin.
    assert score.n_val_subjects == 10 and score.n_val_datasets == 2
    assert score.n_val_origins == 10 * 3 * 9 and score.n_val_pairs == 2 * score.n_val_origins
    assert score.reference_extra_eeg_bits == 0 and score.candidate_extra_eeg_bits == 8 * score.n_val_origins
    assert score.gain > .25 and score.ci_low > .2
    for control in ['shifted', 'blind', 'continuous']:
        assert _score(result, reference=control).ci_low > .2
    assert _score(result, reference='continuous', candidate='continuous_pattern').ci_low > .2
    assert max(_score(result, lag=4).gain, _score(result, lag=8).gain) < .1
    word = _score(result, endpoint='hb_word')
    assert word.metric == 'brier' and 0 <= word.candidate_loss < word.reference_loss <= 2
    assert word.ci_low > 0
    effects = result['word_effects'].query('lag_s == 6').sort_values('eeg_word')
    assert np.corrcoef(effects.hbo_delta_effect, EFFECT)[0, 1] > .95
    assert np.corrcoef(effects.hbr_delta_effect, EFFECT)[0, 1] < -.95
    assert abs(effects.train_mass @ effects.hbo_delta_effect) < 1e-10
    ds = result['by_dataset'].query("lag_s == 6 and endpoint == 'hb_delta' and reference == 'base' and candidate == 'pattern'")
    ss = result['by_subject'].query("lag_s == 6 and endpoint == 'hb_delta' and reference == 'base' and candidate == 'pattern'")
    assert len(ds) == 2 and len(ss) == 10 and ss.ci_low.isna().all()
    assert score.gain == pytest.approx(ds.gain.mean())
    assert np.allclose(ds.set_index('dataset').gain, ss.groupby('dataset').gain.mean())


def test_iid_independent_eeg_null_has_no_registerable_gain():
    result = _result(False)
    assert result['primary']['status'] == 'not_demonstrated'
    assert abs(_score(result).gain) < .1
    assert _score(result).ci_low <= 0
    assert abs(_score(result, endpoint='hb_word').gain) < .1


def _clean(e, b, split):
    return dict(eeg=c._input(e, 'eeg', split), hb=c._input(b, 'hb', split))


def test_contiguity_per_crop_per_site_no_future_eeg_selection_or_hb_features():
    e, b = _synth('train', 1, 2, ('A',))
    # The EEG subset changes temporarily at t=5; H0 lacks t=7. Other site/crop cannot fill either gap.
    e = e[~((e.start_s == 0) & (e.position == 'E1') & (e.patch == 5))]
    b = b[~((b.start_s == 0) & (b.position == 'H0') & (b.patch == 7))]
    raw = _clean(e, b, 'train'); panels = c._panels(raw)
    g = c._grid(raw, panels); h = 3
    i, j = c._anchors(g, h); m = c._meta(g, i, j)
    p = lambda w, pos: set(m.loc[(m.start_s == w) & (m.position == pos), 'patch'])
    assert p(0, 'H0') == {3, 11}           # t=3 allowed despite the missing future EEG patch at t=5
    assert p(0, 'H1') == {3, 4, 9, 10, 11}
    assert p(30, 'H0') == p(30, 'H1') == set(range(3, 12))
    assert (m.n_eeg == 2).all()
    # Inspect ONE origin's design, repeatedly perturb only its future or its own history.
    one = (m.start_s == 0) & (m.position == 'H0') & (m.patch == 3)
    def example(data):
        grid = c._grid(data, panels)
        return c._blocks(grid, i[one], j[one], h)
    x, delta, ids = example(raw)
    g_hb = b.copy(); target = (g_hb.start_s == 0) & (g_hb.position == 'H0') & (g_hb.patch == 6)
    g_hb.loc[target, ['hbo', 'hbr', 'so', 'sr', 'rms']] += [333, 555, 8e4, 9e3, 1e2]
    g_hb.loc[target, 'pattern'] = (g_hb.loc[target, 'pattern'] + 7) % c.K
    g_eeg = e.copy(); future = (e.start_s == 0) & (e.patch > 3)
    g_eeg.loc[future, ['pattern', 'blind', 'alpha', 'beta', 'rms']] = [15, 15, 0, 1, 1e4]
    changed, d2, ids2 = example(_clean(g_eeg, g_hb, 'train'))
    for key in x:
        np.testing.assert_array_equal(x[key], changed[key])
    np.testing.assert_allclose(d2 - delta, [[333, 555]])
    assert ids2[0] != ids[0]
    # No circular roll: the wrong time is t-h=0, with the same site set, not any future word.
    e0 = e[(e.patch == 0) & (e.start_s == 0)]
    np.testing.assert_allclose(x['shifted'][0], np.bincount(e0.pattern, minlength=c.K)/len(e0))
    e3 = e[(e.patch == 3) & (e.start_s == 0)]
    np.testing.assert_allclose(x['continuous'][0, :2], e3[['alpha', 'beta']].mean())
    hist = b.copy(); past = (b.start_s == 0) & (b.position == 'H0') & (b.patch == 1)
    hist.loc[past, ['so', 'hbr', 'pattern']] += [5, 4, 0]
    x2, *_ = example(_clean(e, hist, 'train'))
    assert not np.array_equal(x['base'], x2['base'])


def test_training_sites_frozen_partial_montage_no_unseen_replacement():
    te, tb = _synth('train', 2, 2, ('A',)); ve, vb = _synth('validation', 1, 1, ('A',))
    # An unused categorical level is not an observed training site.
    one = te[te.position == 'E0'].copy()
    one['position'] = pd.Categorical(one.position, categories=['E0', 'E1'])
    assert c._panels(_clean(one, tb, 'train'))['A']['eeg'] == ('E0',)
    # Rare training site ranks third; even a widely supplied validation-only site cannot promote itself.
    te = pd.concat([te, te.iloc[:1].assign(position='E9')], ignore_index=True)
    ve = pd.concat([ve[ve.position == 'E0'], ve[ve.position == 'E1'].assign(position='unused')], ignore_index=True)
    vb = vb[vb.position == 'H1']
    ve = pd.concat([ve, ve.assign(dataset='unknown')], ignore_index=True)
    vb = pd.concat([vb, vb.assign(dataset='unknown')], ignore_index=True)
    result = c.run_coupling(te, tb, ve, vb)
    assert result['panels'].to_dict('records') == [dict(dataset='A', eeg_positions=('E0','E1'), hb_positions=('H0','H1'))]
    usage = result['panel_usage'].query("split == 'validation' and lag_s == 6").iloc[0]
    assert usage.eeg_positions == ('E0',) and usage.hb_positions_used == ('H1',)
    assert usage.n_pairs == usage.n_origins == 9 and usage.total_eeg_word_bits == 9 * 4
    cov = result['coverage'].query("split == 'validation' and dataset == 'unknown' and lag_s == 6").iloc[0]
    assert cov.n_used_pairs == 0 and cov.status == 'no_training_panel' and cov.n_eeg_input > 0
    assert np.isnan(_score(result).ci_low) and result['primary']['status'] == 'insufficient_independent_units'
    # Training coefficients, scales and montage cannot depend on validation signals, counts or targets.
    rerun = c.run_coupling(te, tb, ve.sample(frac=1, random_state=9).assign(alpha=0., beta=1.),
                            vb.sample(frac=1, random_state=8).assign(hbo=1e2*vb.hbo, pattern=7))
    for key in ('panels', 'word_effects', 'training_scales'):
        pd.testing.assert_frame_equal(rerun[key], result[key])


def test_empty_support_no_backfill_and_no_future_pair_over_gap():
    te, tb = _synth('train', 1, 1, ('A',)); ve, vb = _synth('validation', 1, 1, ('A',))
    result = c.run_coupling(te, tb, ve, vb[vb.patch % 2 == 0])
    assert result['status'] == result['primary']['status'] == 'no_paired_support'
    assert result['summary'].empty
    cv = result['coverage'].query("split == 'validation'")
    assert cv.n_used_pairs.eq(0).all() and cv.status.eq('no_contiguous_paired_support').all()
    empty = c.run_coupling(te.iloc[:0], tb.iloc[:0], ve, vb)
    assert empty['primary']['status'] == 'no_paired_support' and empty['panels'].empty
    assert empty['coverage'].status.eq('no_training_panel').all()


def test_refuses_duplicate_bad_codes_split_tags_and_cross_modal_subject_overlap():
    te, tb = _synth('train', 1, 1, ('A',)); ve, vb = _synth('validation', 1, 1, ('A',))
    for message, bad in [('duplicate', pd.concat([te, te.iloc[:1]])),
                         ('pattern', te.assign(pattern=16)), ('blind', te.assign(blind=.5)),
                         ('relative power', te.assign(alpha=1.1)), ('split', te.assign(split='other'))]:
        with pytest.raises(ValueError, match=message):
            c.run_coupling(bad, tb, ve, vb)
    with pytest.raises(ValueError, match='subject overlap'):
        c.run_coupling(te, tb, ve, vb.assign(subject=te.subject.iloc[0]))


def test_nested_unit_weighting_and_paired_resampling_not_patch_resampling():
    # Unequal origins per subject, unequal sites at an origin, repeated subject names across datasets.
    m = pd.DataFrame([
        ['A', '1', 'r', 'p', 0, 'h0', 1, 1, 1], ['A', '1', 'r', 'p', 0, 'h1', 1, 1, 1],
        ['A', '1', 'r', 'p', 0, 'h0', 2, 1, 1], ['A', '2', 'r', 'p', 0, 'h0', 1, 1, 1],
        ['B', '1', 'r', 'p', 0, 'h0', 1, 1, 1]], columns=c.META)
    np.testing.assert_allclose(c._weights(m), [1/16, 1/16, 1/8, 1/4, 1/2])
    # Large subject-specific intercept makes unpaired intervals wide. Paired deltas are exactly zero
    # throughout A, exactly two throughout B. Resampling datasets too must span both extremes.
    L = np.zeros((len(m), len(c.METHODS), len(c.ENDPOINTS)))
    L[:, 0, :] = np.array([50., 50., 50., .5, 25.])[:, None]
    L[:, 1, :] = L[:, 0, :] - (m.dataset == 'B').to_numpy()[:, None]*2
    macro, ds, _ = c._summaries(m, L, 6)
    corners = [r for r in macro + ds if r['reference'] == 'base' and r['candidate'] == 'pattern' and r['endpoint'] == 'hb_delta']
    assert [(r['gain'], r['ci_low'], r['ci_high']) for r in corners[:2]] == [(1., 0., 2.), (0., 0., 0.)]
    assert corners[2]['gain'] == 2 and np.isnan(corners[2]['ci_low'])
