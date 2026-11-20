"""Adjacent *observation* IDs on the G0 2 s grid. In-memory tables only, no I/O.

``run_dynamics`` returns ``scores``, ``edges``, ``coverage``. A pair is (t,t+1)
within the same dataset/subject/record/parent/start_s/native position. Never
join across a missing patch. For a contiguous run of length L with histogram n,
a uniform permutation gives exactly n[i]*(n[j] - (i==j))/L expected adjacent
i→j edges. Sum these counts within site/window. No simulated null, seed search,
validation fit or circular wrap. The null holds even each *run's* bag fixed.

* history_gain_bits: occupancy minus one-step history cross-entropy (bits per
  eligible pair). Both estimate the same train pair destinations. Train series'
  empirical pair distributions get equal conditional dataset/subject/record/
  window/site mass, rescaled to the total train pair count. Global occupancy
  shrinks to uniform; dataset to global; site to dataset. History rows shrink
  to their own occupancy. Each shrinkage has fixed total prior mass K. An unseen
  site uses its dataset model; an unseen dataset uses the global model. No
  history/occupancy counts are updated at validation. ``switch`` conditions BOTH
  forecasts on a change (diagonal removed and each row renormalized). Its null
  is the score of a random bag-order adjacency CONDITIONAL on a switch, i.e.
  ratio of expected counts, not the mean of per-shuffle ratios. A positive
  excess over the bag alone does not imply improved forecasting: check that the
  actual held-out gain over occupancy is positive too.
* self_rate: fraction of pairs repeating an ID. atlas_closeness: negative mean
  absolute difference between train ID medians, over the six EEG relative power
  fractions or the two Hb slopes (shared train X scale/second). Higher = closer.
  ID numbers have no ordinal distance. Ignore a pair in this endpoint if an
  endpoint lacks train medians. No semantic coordinates are borrowed for blind
  waveform IDs; both routes otherwise have the same K, fits, nulls and metrics.
* edges: at most six nonself pattern edges, ranked ONLY by positive balanced
  train joint-rate excess over the same bag null (ties: ID order). Require ≥8
  train occurrences, ≥4 dataset-qualified train subjects and ≥2 train datasets.
  Validation uses frozen edges/labels, including absent or reversed effects.
  observed_count/expected_count are unweighted audit counts; observed/null_mean
  are hierarchical rates per eligible adjacency, NOT conditional on its source.

Within a site/window each eligible pair has equal weight. Then average sites,
windows, records, subjects within datasets, and datasets equally. Only matched
site/windows with both denominators >0 enter the pair of means. n_pairs and
expected_pairs audit those denominators; they are NOT inferential sample sizes.
Absent subjects stay visible (NaN rates); an absent entire dataset makes the
all-dataset macro point unavailable. ``coverage`` counts the supplied scored
patch rows including isolated ones, not unobserved positions/planned raw crops.
``delta`` = observed - exact bag null_mean; n_positive/negative/tied count
subjects (tie tolerance 1e-12). ci_low/high: deterministic 95% paired,
dataset-stratified *subject* percentile bootstrap on delta; observed_ci_* on
the actual score. Frozen train fit, labels, selected edges, validation bags and
dataset strata; no patch-level intervals or multiplicity-adjusted edge tests.
Dataset intervals require ≥2 contributing people; macro requires ≥2 datasets
and ≥2 people in every dataset. Edge intervals additionally require bag
opportunities in ≥2 people (macro: ≥4 people in ≥2 datasets), never a positive
observed validation edge. Always inspect actual and opportunity support.

Slow-signal continuity, occupancy bursts, offline filtering and changing RMS
can all give predictable observations. The RMS column is audited for validity,
not conditioned away or used as a future input. These scores do not establish
new physiology, latent-state mechanisms, causal transitions or superiority of
one tokenizer from separate intervals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analysis.observation_tokenizer_evaluation import KEYS, hierarchy_weights
from src.tokenizers.factorized_observation import FactorizedObservationTokenizer


_BOOT_SEED = 48203
_BOOT_REPS = 2000
_TOL = 1e-12
_PRIOR_EDGE_MIN = (8, 4, 2)  # raw train occurrences, distinct subjects, datasets
_CE = ['occupancy_bits', 'history_bits', 'null_occupancy_bits', 'null_history_bits']
_COUNTS = ['n_pairs', 'expected_pairs', 'n_series', 'n_subjects', 'n_planned_subjects',
           'n_positive', 'n_negative', 'n_tied']
_EDGE_COUNTS = ['observed_count', 'expected_count', 'n_support_subjects', 'n_opportunity_subjects']
_CI = ['observed_ci_low', 'observed_ci_high', 'ci_low', 'ci_high']
_EDGE_COLS = [
    'modality', 'k', 'rank', 'source_id', 'target_id', 'source_label', 'target_label',
    'train_observed', 'train_null_mean', 'train_delta', 'train_observed_count',
    'train_expected_count', 'train_support_subjects', 'train_support_datasets',
    'level', 'dataset', 'subject', 'observed', 'null_mean', 'delta', *_CI, *_COUNTS,
    'n_datasets', 'n_planned_datasets', *_EDGE_COUNTS, 'n_support_datasets',
    'n_opportunity_datasets', 'interval_status',
]


def _codes(values, name, upper):
    if not pd.api.types.is_integer_dtype(values.dtype) or values.isna().any():
        raise ValueError(f'{name} must contain nonmissing integer indices')
    if not ((values >= 0) & (values < upper)).all():
        raise ValueError(f'{name} outside [0, {upper})')
    return values.to_numpy(dtype=np.int64)


def _atlas(atlas, modality):
    needed = {'modality', 'split', 'pattern_id', 'feature', 'median'}
    if not needed.issubset(atlas.columns) or not atlas.columns.is_unique:
        raise ValueError('missing or ambiguous pattern atlas columns')
    if not atlas['split'].isin(['train', 'validation']).all():
        raise ValueError('development atlas only')
    a = atlas.loc[(atlas['modality'] == modality) & (atlas['split'] == 'train'),
                  ['pattern_id', 'feature', 'median']].copy()
    # Neither the vocabulary nor any feature value/label comes from held-out IDs.
    ids = pd.to_numeric(a['pattern_id'], errors='raise')
    if a.empty or not np.isfinite(ids).all() or not (ids == np.floor(ids)).all():
        raise ValueError('missing or invalid train pattern atlas IDs')
    k = ids.nunique()
    if k < 2 or ids.min() != 0 or ids.max() != k - 1:
        raise ValueError('train atlas must enumerate the shared alphabet 0..K-1')
    a['pattern_id'] = ids.astype(int)
    names = FactorizedObservationTokenizer(modality).pattern_feature_names
    features = list(names if modality == 'eeg' else names[-2:])
    a = a[a['feature'].isin(features)]
    if len(a) != k * len(features) or a.duplicated(['pattern_id', 'feature']).any():
        raise ValueError('missing or duplicate train atlas feature per ID')
    x = a.pivot(index='pattern_id', columns='feature', values='median')[features].to_numpy(dtype=float)
    named = np.isfinite(x).all(axis=1)
    if np.isinf(x).any() or (np.isfinite(x).any(axis=1) & ~named).any():
        raise ValueError('incomplete train median vector')
    if modality == 'eeg' and ((x[named] < 0) | (x[named] > 1 + _TOL)).any():
        raise ValueError('train relative-power median outside [0,1]')
    labels = []
    for row, available in zip(x, named):
        if not available:
            labels.append('no train median')
        elif modality == 'hb':
            labels.append(f'HbO slope {row[0]:+.3g}, HbR slope {row[1]:+.3g} (train X scale/s)')
        elif row.max() - row.min() <= _TOL:
            labels.append('no unique dominant median EEG band')
        else:
            # Report ties, not a physiological name chosen by validation examples.
            bands = [name.removeprefix('eeg_rel_').removesuffix('hz').replace('p', '.').replace('_', '–')
                     for name, value in zip(features, row) if abs(value - row.max()) <= _TOL]
            labels.append(f'peak median power fraction: {"/".join(bands)} Hz ({row.max():.2f})')
    closeness = -np.abs(x[:, None, :] - x[None, :, :]).mean(axis=2)
    return k, labels, closeness


@dataclass
class _Panel:
    frame: pd.DataFrame
    series: pd.DataFrame                 # exactly one row per observed site/window
    group: np.ndarray                    # patch -> site/window row
    run_starts: np.ndarray               # including final sentinel
    before: np.ndarray                   # admitted source patch rows, target = before + 1

    def matrices(self, routing, k, bag=True):
        ids = self.frame[routing].to_numpy(dtype=np.int64)
        i = self.before
        cells = (self.group[i] * k + ids[i]) * k + ids[i + 1]
        shape = (len(self.series), k, k)
        observed = np.bincount(cells, minlength=int(np.prod(shape))).reshape(shape)
        if not bag:
            return observed, None
        expected = np.zeros(shape)
        diag = np.diag_indices(k)
        for start, end in zip(self.run_starts[:-1], self.run_starts[1:]):
            if end - start < 2:
                continue
            n = np.bincount(ids[start:end], minlength=k)
            outer = n[:, None] * n[None, :]
            outer[diag] -= n
            expected[self.group[start]] += outer / (end - start)
        return observed, expected


def _panel(frame, split, modality, k):
    columns = KEYS + ['patch', 'pattern', 'blind', 'rms']
    if frame.empty or not set(columns).issubset(frame.columns) or not frame.columns.is_unique:
        raise ValueError('empty or incomplete patch frame')
    for key, required in [('split', split), ('modality', modality)]:
        if key in frame and not frame[key].isin([required]).all():
            raise ValueError(f'unexpected {key} in {split} {modality} frame')
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS + ['patch']).any():
        raise ValueError('missing identity or duplicate patch')
    rms = frame['rms'].to_numpy(dtype=float)
    if not np.isfinite(rms).all() or (rms < 0).any():
        raise ValueError('expected finite nonnegative real-patch AC rms')
    f = frame[columns].sort_values(KEYS + ['patch']).reset_index(drop=True)
    patch = _codes(f['patch'], 'G0 patch', 15)
    for route in ['pattern', 'blind']:
        _codes(f[route], route, k)
    first = ~f.duplicated(KEYS).to_numpy()
    group = first.cumsum() - 1
    run = first.copy()
    run[1:] |= patch[1:] != patch[:-1] + 1
    before = np.flatnonzero(~run[1:])
    series = f.loc[first, KEYS].reset_index(drop=True)
    series['n_patches'] = np.bincount(group)
    series['n_runs'] = np.bincount(group[run], minlength=len(series))
    series['n_pairs'] = np.bincount(group[before], minlength=len(series))
    return _Panel(f, series, group, np.r_[np.flatnonzero(run), len(f)], before)


def _coverage(panel, split):
    s = panel.series.assign(n_series=1, n_series_with_pairs=(panel.series.n_pairs > 0).astype(int))
    counts = ['n_patches', 'n_pairs', 'n_runs', 'n_series', 'n_series_with_pairs']
    sub = s.groupby(KEYS[:2], as_index=False, observed=True)[counts].sum().assign(n_datasets=1, n_subjects=1)
    sub['n_subjects_with_pairs'] = (sub.n_pairs > 0).astype(int)
    cols = counts + ['n_subjects', 'n_subjects_with_pairs']
    ds = sub.groupby('dataset', as_index=False, observed=True)[cols].sum().assign(n_datasets=1, subject='')
    macro = ds[cols + ['n_datasets']].sum().to_frame().T.assign(level='macro', dataset='', subject='')
    return pd.concat([macro, ds.assign(level='dataset'), sub.assign(level='subject')],
                     ignore_index=True).assign(split=split)


def _training_weights(series):
    valid = series.n_pairs > 0
    if not valid.any():
        raise ValueError('no adjacent training pairs')
    w = np.zeros(len(series))
    w[valid] = hierarchy_weights(series.loc[valid]) / series.loc[valid, 'n_pairs'].to_numpy()
    return w  # w times counts gives equal-hierarchy empirical joint rates


def _fit(series, obs, weights):
    k = obs.shape[1]
    weighted = obs * (weights * series.n_pairs.sum())[:, None, None]

    def model(counts, parent_occupancy):
        q = (counts.sum(axis=0) + k * parent_occupancy) / (counts.sum() + k)
        p = (counts + k * q) / (counts.sum(axis=1, keepdims=True) + k)
        return q, p

    global_model = model(weighted.sum(axis=0), np.full(k, 1 / k))
    ds, site = {}, {}
    for d, g in series.groupby('dataset', observed=True).indices.items():
        c = weighted[g].sum(axis=0)
        if c.sum() > 0:
            ds[d] = model(c, global_model[0])
    for key, g in series.groupby(['dataset', 'position'], observed=True).indices.items():
        c = weighted[g].sum(axis=0)
        if c.sum() > 0:
            site[key] = model(c, ds[key[0]][0])
    return global_model, ds, site


def _dot(counts, matrix):
    return np.einsum('sij,ij->s', counts, matrix)


def _ratio(x, n):
    return np.divide(x, n, out=np.full(len(n), np.nan), where=n > 0)


def _history(series, obs, bag, fit, mask):
    global_model, ds, site = fit
    n, e = _dot(obs, mask), _dot(bag, mask)
    out = {name: np.full(len(series), np.nan) for name in _CE}
    for key, rows in series.groupby(['dataset', 'position'], observed=True).indices.items():
        q, p = site.get(key, ds.get(key[0], global_model))
        for name, probs in [('occupancy_bits', np.broadcast_to(q, p.shape)), ('history_bits', p)]:
            # For switches the row normalizer is P(next != previous | context).
            cond = probs / (probs * mask).sum(axis=1, keepdims=True)
            bits = np.where(mask, -np.log2(cond), 0.)
            out[name][rows] = _ratio(_dot(obs[rows], bits), n[rows])
            out['null_' + name][rows] = _ratio(_dot(bag[rows], bits), e[rows])
    return dict(observed=out['occupancy_bits'] - out['history_bits'],
                null_mean=out['null_occupancy_bits'] - out['null_history_bits'],
                n_pairs=n, expected_pairs=e, **out)


def _measurement(obs, bag, mask, matrix):
    n, e = _dot(obs, mask), _dot(bag, mask)
    value = np.where(mask, matrix, 0.)
    return dict(observed=_ratio(_dot(obs, value), n), null_mean=_ratio(_dot(bag, value), e),
                n_pairs=n, expected_pairs=e)


def _summarize(series, raw, edge=False):
    means = ['observed', 'null_mean'] + [name for name in _CE if name in raw]
    rates = series[KEYS].assign(**{name: raw[name] for name in means})
    matched = (raw['n_pairs'] > 0) & (raw['expected_pairs'] > 0)
    rates.loc[~matched, means] = np.nan
    for keys in (KEYS[:5], KEYS[:3], KEYS[:2]):
        rates = rates.groupby(keys, as_index=False, observed=True)[means].mean()
    rates['delta'] = rates.observed - rates.null_mean
    means += ['delta']
    counts = ['n_pairs', 'expected_pairs'] + (['observed_count', 'expected_count'] if edge else [])
    counts = series[KEYS[:2]].assign(**{c: np.where(matched, raw[c], 0) for c in counts},
                                     n_series=matched.astype(int))
    sub = rates.merge(counts.groupby(KEYS[:2], as_index=False, observed=True).sum(),
                      on=KEYS[:2], validate='one_to_one')
    valid = sub.delta.notna()
    sub['n_subjects'] = valid.astype(int)
    sub['n_planned_subjects'] = 1
    sub['n_positive'] = (sub.delta > _TOL).astype(int)
    sub['n_negative'] = (sub.delta < -_TOL).astype(int)
    sub['n_tied'] = (valid & (sub.delta.abs() <= _TOL)).astype(int)
    if edge:
        sub['n_support_subjects'] = (sub.observed_count > 0).astype(int)
        sub['n_opportunity_subjects'] = (sub.expected_count > 0).astype(int)
    additive = _COUNTS + (_EDGE_COUNTS if edge else [])
    ds = sub.groupby('dataset', as_index=False, observed=True).agg(
        {**dict.fromkeys(means, 'mean'), **dict.fromkeys(additive, 'sum')})
    dataset_counts = ['n_datasets', 'n_planned_datasets']
    for table in (sub, ds):
        table['n_datasets'] = (table.n_subjects > 0).astype(int)
        table['n_planned_datasets'] = 1
        if edge:
            for attr in ('support', 'opportunity'):
                table[f'n_{attr}_datasets'] = (table[f'n_{attr}_subjects'] > 0).astype(int)
    if edge:
        dataset_counts += ['n_support_datasets', 'n_opportunity_datasets']
    macro = {**ds[means].mean(skipna=False).to_dict(), **ds[additive + dataset_counts].sum().to_dict(),
             'level': 'macro', 'dataset': '', 'subject': ''}

    def interval(draws):
        lo, hi = np.quantile(draws, [.025, .975], axis=0)
        return dict(zip(_CI, [lo[0], hi[0], lo[1], hi[1]]), interval_status='available')

    for c in _CI:
        sub[c] = ds[c] = macro[c] = np.nan
    sub['interval_status'] = 'subject_is_unit'
    ds['interval_status'] = macro['interval_status'] = 'too_few_subjects_or_datasets'
    rng = np.random.default_rng(_BOOT_SEED)
    reps = []
    for idx, d in zip(ds.index, ds.dataset):
        group = sub.loc[(sub.dataset == d) & valid]
        x = group[['observed', 'delta']].to_numpy()
        if len(x) < 2:
            continue
        draw = x[rng.integers(0, len(x), size=(_BOOT_REPS, len(x)))].mean(axis=1)
        reps.append(draw)
        if edge and group.n_opportunity_subjects.sum() < 2:
            ds.loc[idx, 'interval_status'] = 'too_few_bag_opportunities'
        else:
            for column, value in interval(draw).items():
                ds.loc[idx, column] = value
    if len(ds) >= 2 and len(reps) == len(ds):
        if edge and (macro['n_opportunity_subjects'] < 4 or macro['n_opportunity_datasets'] < 2):
            macro['interval_status'] = 'too_few_bag_opportunities'
        else:
            macro.update(interval(np.mean(reps, axis=0)))
    return pd.concat([pd.DataFrame([macro]), ds.assign(level='dataset', subject=''),
                      sub.assign(level='subject')], ignore_index=True)


def _select_edges(series, obs, bag, w, closeness, labels):
    rate = np.einsum('s,sij->ij', w, obs)
    null = np.einsum('s,sij->ij', w, bag)
    delta = rate - null
    seen = []
    for keys in (KEYS[:2], KEYS[:1]):
        groups = series.groupby(keys, observed=True).indices.values()
        seen.append(sum(obs[g].sum(axis=0) > 0 for g in groups))
    total, expectation = obs.sum(axis=0), bag.sum(axis=0)
    n, ns, nd = _PRIOR_EDGE_MIN
    eligible = ((total >= n) & (seen[0] >= ns) & (seen[1] >= nd) & (delta > _TOL)
                & np.isfinite(closeness) & ~np.eye(len(labels), dtype=bool))
    candidates = sorted(zip(*np.nonzero(eligible)), key=lambda ij: (-round(delta[ij], 12), *ij))
    result = []
    for rank, (a, b) in enumerate(candidates[:6], 1):
        result.append(dict(rank=rank, source_id=a, target_id=b, source_label=labels[a], target_label=labels[b],
                           train_observed=rate[a, b], train_null_mean=null[a, b], train_delta=delta[a, b],
                           train_observed_count=total[a, b], train_expected_count=expectation[a, b],
                           train_support_subjects=seen[0][a, b], train_support_datasets=seen[1][a, b]))
    return result


def run_dynamics(train: pd.DataFrame, val: pd.DataFrame, atlas: pd.DataFrame,
                 modality: str) -> dict[str, pd.DataFrame]:
    """Score supplied complete train/held-out patches; return frames, never read files.

    Missing/unscored patches must be absent, not imputed/sentinel IDs. ``patch``
    is the original integer 0..14 within the 30 s window, not a compacted index.
    ``pattern`` and ``blind`` are separate integer alphabets with the same K,
    enumerated by the train atlas. If present, frame split/modality must agree.
    RMS is the real patch AC RMS in frozen train coordinates, not a code feature.
    """
    if modality not in ('eeg', 'hb'):
        raise ValueError('expected eeg or hb')
    k, labels, closeness = _atlas(atlas, modality)
    t, v = _panel(train, 'train', modality, k), _panel(val, 'validation', modality, k)
    if not t.series[KEYS[:2]].drop_duplicates().merge(v.series[KEYS[:2]].drop_duplicates(), on=KEYS[:2]).empty:
        raise ValueError('dataset-qualified subject overlap across train and validation')
    weights = _training_weights(t.series)
    same = np.eye(k, dtype=bool)
    scopes = {'all': np.ones_like(same), 'switch': ~same}
    scores, edge_rows = [], []
    for route in ('pattern', 'blind'):
        ot, bt = t.matrices(route, k, bag=(route == 'pattern'))
        model = _fit(t.series, ot, weights)
        selected = _select_edges(t.series, ot, bt, weights, closeness, labels) if route == 'pattern' else []
        obs, bag = v.matrices(route, k)

        def add(raw, metric, scope):
            scores.append(_summarize(v.series, raw).assign(routing=route, metric=metric, scope=scope))

        add(_measurement(obs, bag, scopes['all'], same), 'self_rate', 'all')
        for scope, mask in scopes.items():
            add(_history(v.series, obs, bag, model, mask), 'history_gain_bits', scope)
            if route == 'pattern':
                add(_measurement(obs, bag, mask & np.isfinite(closeness), closeness), 'atlas_closeness', scope)
        for edge in selected:
            a, b = edge['source_id'], edge['target_id']
            n = v.series.n_pairs.to_numpy()
            raw = dict(observed=_ratio(obs[:, a, b], n), null_mean=_ratio(bag[:, a, b], n),
                       observed_count=obs[:, a, b], expected_count=bag[:, a, b], n_pairs=n, expected_pairs=n)
            edge_rows.append(_summarize(v.series, raw, edge=True).assign(**edge))
    coverage = pd.concat([_coverage(t, 'train'), _coverage(v, 'validation')], ignore_index=True)
    edges = pd.concat(edge_rows, ignore_index=True) if edge_rows else pd.DataFrame()
    return {'scores': pd.concat(scores, ignore_index=True).assign(modality=modality, k=k),
            'edges': edges.assign(modality=modality, k=k).reindex(columns=_EDGE_COLS),
            'coverage': coverage.assign(modality=modality, k=k)}
