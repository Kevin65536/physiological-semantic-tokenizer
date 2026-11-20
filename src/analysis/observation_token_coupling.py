"""Delayed *association* of frozen K16 observation words; no I/O, Granger or causal test.

One input row is one complete, scored, disjoint 2-s patch. ``start_s`` identifies
its crop within ``parent``; ``patch`` is the integer ordinal *within that crop*.
The caller must audit native member clocks, disjoint crop supports, the scoring
mask and the training-only dictionaries/scales before supplying either modality.
Matching crop keys cannot establish synchronization or sensor co-location.

The origin is the end of patch t. A lead h is start-to-start: the 6-s target
patch starts 4 s after the visible patch ends. Only Hb targets and the presence of
contiguous scorable Hb support may be inspected past t. Results concern offline,
parent-preprocessed observations on the supplied supports, not online decoding.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from src.analysis.observation_tokenizer_evaluation import KEYS


SUBJECT = KEYS[:2]
RECORD = KEYS[:3]
CROP = KEYS[:5]
CELL = CROP + ['patch']
ROW = KEYS + ['patch']
K = 16
HISTORY = 2  # Hb at t, t-1 and t-2
LAGS = (2, 3, 4)  # 2-second patches; primary = 3
RIDGE = 0.01  # loss weights sum to one, not number of correlated site/patch rows
VAR_FLOOR = 1e-12
BOOTSTRAPS = 2000
SEED = 6803
ENDPOINTS = ('hb_delta', 'hbo_delta', 'hbr_delta', 'hb_word')
METHODS = ('base', 'pattern', 'shifted', 'blind', 'continuous', 'continuous_pattern')
CHARGED = {'pattern', 'shifted', 'blind', 'continuous_pattern'}
CONTRASTS = (('base', 'pattern'), ('base', 'shifted'), ('base', 'blind'), ('base', 'continuous'),
             ('shifted', 'pattern'), ('blind', 'pattern'), ('continuous', 'pattern'),
             ('continuous', 'continuous_pattern'), ('pattern', 'continuous_pattern'))
META = ROW + ['eeg_mask', 'n_eeg']


def _input(frame, modality, split):
    active = ['pattern', 'rms'] + (['blind', 'alpha', 'beta'] if modality == 'eeg'
                                   else ['hbo', 'hbr', 'so', 'sr'])
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError('expected a DataFrame with unique columns')
    missing = set(ROW + active).difference(frame.columns)
    if missing:
        raise ValueError(f'missing {modality} columns: {sorted(missing)}')
    for tag, value in [('split', split), ('modality', modality)]:
        if tag in frame and (frame[tag].isna().any() or not frame[tag].eq(value).all()):
            raise ValueError(f'wrong {tag}; expected {value}')
    # In particular do not consume placeholder Hb fields in EEG or band powers in Hb.
    f = frame[ROW + active].copy().reset_index(drop=True)
    if f[KEYS].isna().any().any():
        raise ValueError('missing patch identity')
    for col in ['start_s', 'patch', *active]:
        try:
            a = pd.to_numeric(f[col], errors='raise').to_numpy(dtype=np.float64)
        except (ValueError, TypeError, OverflowError) as err:
            raise ValueError(f'invalid {modality}.{col}') from err
        if not np.isfinite(a).all():
            raise ValueError(f'nonfinite {modality}.{col}')
        if col in ('patch', 'pattern', 'blind'):
            bound = K if col != 'patch' else 2**53  # exact integer representation
            if np.any((a < 0) | (a >= bound) | (a != np.floor(a))):
                raise ValueError(f'invalid integer {modality}.{col}')
            a = a.astype(np.int64)
        if col in ('start_s', 'rms') and (a < 0).any():
            raise ValueError(f'negative {modality}.{col}')
        if col in ('alpha', 'beta') and np.any((a < 0) | (a > 1 + 1e-7)):
            raise ValueError(f'{col} must be actual relative power')
        f[col] = a
    if f.duplicated(ROW).any():
        raise ValueError(f'duplicate {split} {modality} patch identity')
    return f


def _panels(train):
    """Rank named sites by complete training rows; ties by name, never outcome/validation."""
    panels = {}
    for modality, f in train.items():
        for dataset, g in f.groupby('dataset', observed=True, sort=True):
            counts = g.groupby('position', sort=False, observed=True).size()
            rank = sorted(counts.index, key=lambda p: (-int(counts[p]), str(p), type(p).__name__))
            panels.setdefault(dataset, {})[modality] = tuple(rank[:2])
    return {d: {m: p.get(m, ()) for m in ('eeg', 'hb')} for d, p in panels.items()}


def _selected(f, panels, modality):
    sites = [(d, p, i) for d, panel in panels.items() for i, p in enumerate(panel[modality])]
    if not sites or f.empty:
        return f.iloc[:0].assign(slot=np.zeros(0, dtype=int))
    lookup = pd.DataFrame(sites, columns=['dataset', 'position', 'slot'])
    return f.merge(lookup, on=['dataset', 'position'], how='inner', validate='many_to_one')


def _runs(frame, keys):
    """Available immediately adjacent rows on the left/right of every sorted row."""
    i = np.arange(len(frame))
    if not len(i):
        return i, i
    group = frame.groupby(keys, sort=False, observed=True).ngroup().to_numpy()
    cut = (group[1:] != group[:-1]) | (np.diff(frame.patch.to_numpy()) != 1)
    left = np.maximum.accumulate(np.where(np.r_[True, cut], i, 0))
    right = np.minimum.accumulate(np.where(np.r_[cut, True], i, len(i))[::-1])[::-1]
    return i - left, right - i


@dataclass
class _Grid:
    hb: pd.DataFrame
    hb_left: np.ndarray
    hb_right: np.ndarray
    eeg_index: np.ndarray  # one lookup per Hb row, -1 for no same-crop EEG
    cells: pd.DataFrame
    eeg_left: np.ndarray
    pattern: np.ndarray
    blind: np.ndarray
    amplitude: np.ndarray
    bands: np.ndarray


def _grid(data, panels):
    e = _selected(data['eeg'], panels, 'eeg').sort_values(CELL + ['slot']).reset_index(drop=True)
    b = _selected(data['hb'], panels, 'hb').sort_values(ROW).reset_index(drop=True)
    first = ~e.duplicated(CELL)
    cells = e.loc[first, CELL].reset_index(drop=True)
    group = first.to_numpy().cumsum(dtype=int) - 1
    count = np.bincount(group, minlength=len(cells))
    slot = e.slot.to_numpy(dtype=int)
    mask = np.zeros(len(cells), dtype=int)
    np.bitwise_or.at(mask, group, 1 << slot)
    cells['eeg_mask'], cells['n_eeg'] = mask, count

    def hist(word):
        a = np.zeros((len(cells), K))
        np.add.at(a, (group, e[word].to_numpy(dtype=int)), 1.)
        return a / count[:, None]

    # Two train-named slots. A missing slot has zero here AND an explicit mask.
    amp = np.zeros((len(cells), 2))
    amp[group, slot] = np.log(np.maximum(e.rms.to_numpy(), 1e-8))
    powers = e[['alpha', 'beta']].to_numpy()
    means = np.zeros((len(cells), 2)); ss = np.zeros_like(means)
    np.add.at(means, group, powers)
    means /= count[:, None]
    np.add.at(ss, group, (powers - means[group])**2)
    bands = np.column_stack((means, np.sqrt(ss / count[:, None])))
    match = pd.MultiIndex.from_frame(cells[CELL]).get_indexer(pd.MultiIndex.from_frame(b[CELL]))
    left, right = _runs(b, KEYS)
    eeg_left, _ = _runs(cells, CROP + ['eeg_mask'])
    return _Grid(b, left, right, match, cells, eeg_left, hist('pattern'), hist('blind'), amp, bands)


def _anchors(grid, h):
    # Only the *target site's* Hb availability is needed beyond t. Future EEG is irrelevant.
    # No Hb gap even between the shifted control time and the start of the Hb history.
    i = np.flatnonzero((grid.hb_left >= max(h, HISTORY)) & (grid.hb_right >= h) & (grid.eeg_index >= 0))
    j = grid.eeg_index[i]
    keep = grid.eeg_left[j] >= h  # same actual EEG sensor subset, no gaps from t-h through t
    return i[keep], j[keep]


def _meta(grid, i, j):
    f = grid.hb.iloc[i][ROW].reset_index(drop=True)
    f[['eeg_mask', 'n_eeg']] = grid.cells.iloc[j][['eeg_mask', 'n_eeg']].to_numpy()
    return f


def _in_training_scope(train, val):
    """All contributing positions need fit support, not necessarily the full joint montage."""
    site = ['dataset', 'position']
    hb_ok = pd.MultiIndex.from_frame(val[site]).isin(pd.MultiIndex.from_frame(train[site]))
    masks = train.groupby('dataset', observed=True).eeg_mask.agg(lambda x: np.bitwise_or.reduce(x))
    allowed = masks.reindex(pd.Index(val.dataset), fill_value=0).to_numpy(dtype=int)
    return hb_ok & ((val.eeg_mask.to_numpy(dtype=int) & ~allowed) == 0)


def _blocks(grid, i, j, h):
    """Index separation: every design block ends at t; only y and word use i+h."""
    b = grid.hb
    ids = b.pattern.to_numpy(dtype=int)
    obs = b[['hbo', 'hbr', 'so', 'sr']].to_numpy()
    log_rms = np.log(np.maximum(b.rms.to_numpy(), 1e-8))
    base = []
    for lag in range(HISTORY + 1):
        p = i - lag
        base.extend([obs[p], log_rms[p], np.eye(K)[ids[p]]])
    phase = b.patch.to_numpy()[i].astype(float)  # visible crop-relative time, no target clock
    base.extend([grid.amplitude[j], grid.amplitude[j-h], phase, phase**2])
    blocks = dict(base=np.column_stack(base), pattern=grid.pattern[j], shifted=grid.pattern[j-h],
                  blind=grid.blind[j], continuous=grid.bands[j])
    return blocks, obs[i+h, :2] - obs[i, :2], ids[i+h]


def _weights(meta):
    """Equal Hb sites at origin → origins at crop → crops at record → records/subject → subjects/dataset."""
    w = pd.Series(np.ones(len(meta)), index=meta.index)
    for keys in [CELL, CROP, RECORD, SUBJECT, ['dataset']]:
        total = w.groupby([meta[k] for k in keys], sort=False, observed=True).transform('sum')
        w = w / total
    return (w / w.sum()).to_numpy()


def _categories(train, val):
    def indicators(keys):
        levels = pd.MultiIndex.from_frame(train[keys].drop_duplicates())
        def block(meta):
            idx = levels.get_indexer(pd.MultiIndex.from_frame(meta[keys]))
            mat = np.zeros((len(meta), len(levels)))
            good = np.flatnonzero(idx >= 0)
            mat[good, idx[good]] = 1
            return mat
        return block(train), block(val)

    ds = indicators(['dataset']); site = indicators(['dataset', 'position'])
    result = []
    for m, d, s in zip((train, val), ds, site):
        mask = m.eeg_mask.to_numpy(dtype=int)[:, None]
        result.append(np.column_stack([d, s, d * ((mask & 1) != 0), d * ((mask & 2) != 0)]))
    return result


def _simplex(scores):
    """Euclidean projection of each linear probability readout onto the K simplex."""
    u = np.sort(scores, axis=1)[:, ::-1]
    thresholds = (np.cumsum(u, axis=1) - 1.) / np.arange(1, K + 1)
    k = (u > thresholds).sum(axis=1) - 1
    return np.maximum(0., scores - thresholds[np.arange(len(u)), k, None])


def _fit(train, val, data_t, data_v, lag_s):
    """One common weighted Gram matrix; each model/target has its own ridge coefficients."""
    bt, yt, it = data_t; bv, yv, iv = data_v
    w = _weights(train)
    cats = _categories(train, val)
    blocks = ['base', 'pattern', 'shifted', 'blind', 'continuous']
    xt, xv, indices = [], [], {}; end = 0
    for key in blocks:
        t, v = bt[key], bv[key]
        if key == 'base':
            t, v = np.column_stack([t, cats[0]]), np.column_stack([v, cats[1]])
        xt.append(t); xv.append(v)
        indices[key] = np.arange(end, end + t.shape[1]); end += t.shape[1]
    xt, xv = np.column_stack(xt), np.column_stack(xv)
    xm = np.average(xt, axis=0, weights=w)
    xt -= xm; xv -= xm
    xs = np.sqrt(np.average(xt**2, axis=0, weights=w))
    # Constant train columns cannot support a readout. Drop them rather than
    # amplifying roundoff in their coefficients for out-of-range validation data.
    const = xs < 1e-10
    xs[const] = 1.; xt[:, const] = 0.; xv[:, const] = 0.
    xt /= xs; xv /= xs
    ym = np.average(yt, axis=0, weights=w)
    var = np.average((yt - ym)**2, axis=0, weights=w)
    ys = np.sqrt(np.maximum(var, VAR_FLOOR))
    y = np.column_stack([(yt - ym) / ys, np.eye(K)[it]])
    intercept = np.average(y, axis=0, weights=w)
    yv = (yv - ym) / ys
    wx = (xt * w[:, None]).T
    gram = wx @ xt; rhs = wx @ (y - intercept)
    losses = []; effects = []
    for method in METHODS:
        extra = [] if method == 'base' else (['continuous', 'pattern'] if method == 'continuous_pattern' else [method])
        use = np.concatenate([indices['base'], *(indices[k] for k in extra)])
        g = gram[np.ix_(use, use)].copy()
        g.flat[::len(g)+1] += RIDGE
        coef = solve(g, rhs[use], assume_a='pos', check_finite=True)
        pred = xv[:, use] @ coef + intercept
        se = (pred[:, :2] - yv)**2
        probs = _simplex(pred[:, 2:])
        brier = ((probs - np.eye(K)[iv])**2).sum(axis=1)
        losses.append(np.column_stack([se.mean(axis=1), se, brier]))
        if method == 'pattern':
            # Unit histogram versus the training mixture, holding base fixed.
            # One site replaced in an m-site histogram changes 1/m of its mass.
            mass = np.average(bt['pattern'], axis=0, weights=w)
            slope = coef[-K:, :2] * ys / xs[indices['pattern'], None]
            centered = slope - mass @ slope
            centered[mass == 0] = np.nan  # no empirical direction for an unseen word
            for k in range(K):
                effects.append(dict(lag_s=lag_s, eeg_word=k, train_mass=mass[k],
                                    hbo_delta_effect=centered[k, 0], hbr_delta_effect=centered[k, 1]))
    scales = dict(lag_s=lag_s, hbo_delta_mean=ym[0], hbr_delta_mean=ym[1], hbo_delta_var=var[0],
                  hbr_delta_var=var[1], hbo_delta_scale=ys[0], hbr_delta_scale=ys[1])
    loss = np.stack(losses, axis=1)  # target-site rows x methods x independent endpoints
    if not np.isfinite(loss).all():
        raise ValueError('nonfinite validation loss')
    return loss, effects, scales


def _counts(meta):
    cells = meta.drop_duplicates(CELL)
    return dict(n_val_pairs=len(meta), n_val_origins=len(cells),
                n_val_records=len(cells[RECORD].drop_duplicates()),
                n_val_crops=len(cells[CROP].drop_duplicates()),
                eeg_bits=int(4 * cells.n_eeg.sum()))


def _bootstrap(units, datasets, seed):
    """Resample dataset then subject within EACH drawn dataset copy. Never resample patches."""
    rng = np.random.default_rng(seed)
    b = BOOTSTRAPS
    levels = pd.unique(datasets)
    nested_draw = rng.integers(len(levels), size=(b, len(levels)))
    pooled = np.zeros((b, units.shape[1])); intervals = {}
    def interval(a):
        return np.quantile(a, [.025, .975], axis=0)
    for k, dataset in enumerate(levels):
        unit = units[datasets == dataset]; n = len(unit)
        def draw(size):
            counts = rng.multinomial(n, np.full(n, 1. / n), size=size)
            return (counts / n) @ unit
        intervals[dataset] = interval(draw(b)) if n > 1 else np.full((2, units.shape[1]), np.nan)
        row = np.nonzero(nested_draw == k)[0]
        np.add.at(pooled, row, draw(len(row)) / len(levels))
    macro = interval(pooled) if len(units) > 1 else np.full_like(pooled[:2], np.nan)
    return macro, intervals


def _summaries(meta, loss, lag_s):
    w = _weights(meta)
    subjects = []; datasets = []; counts = []; unit = []
    for (d, s), r in meta.groupby(SUBJECT, sort=True, observed=True).indices.items():
        datasets.append(d); subjects.append(s); counts.append(_counts(meta.iloc[r]))
        unit.append(np.average(loss[r], axis=0, weights=w[r]))
    unit = np.asarray(unit); datasets = np.asarray(datasets, dtype=object)
    ref, cand = np.array([(METHODS.index(r), METHODS.index(c)) for r, c in CONTRASTS]).T
    diff = (unit[:, ref] - unit[:, cand]).reshape(len(unit), -1)
    ci, within = _bootstrap(diff, datasets, SEED + lag_s)
    def rows(point, indices, limits, *, dataset=None, subject=None):
        n = {key: sum(counts[i][key] for i in indices) for key in counts[0]}
        bits = n.pop('eeg_bits')
        n.update(n_val_subjects=len(indices), n_val_datasets=len(pd.unique(datasets[indices])))
        result = []
        for j, (r, c) in enumerate(CONTRASTS):
            for k, endpoint in enumerate(ENDPOINTS):
                a, b = point[ref[j], k], point[cand[j], k]
                lo, hi = (np.nan, np.nan) if limits is None else limits[:, j*len(ENDPOINTS)+k]
                row = dict(lag_s=lag_s, endpoint=endpoint, metric='brier' if endpoint == 'hb_word' else 'train_scaled_mse',
                           reference=r, candidate=c, reference_loss=a, candidate_loss=b, gain=a-b,
                           ci_low=lo, ci_high=hi, ci_above_zero=bool(lo > 0) if np.isfinite(lo) else None,
                           reference_extra_eeg_bits=bits if r in CHARGED else 0,
                           candidate_extra_eeg_bits=bits if c in CHARGED else 0, **n)
                if dataset is not None: row['dataset'] = dataset
                if subject is not None: row['subject'] = subject
                result.append(row)
        return result
    per_subject = []; per_dataset = []; means = []
    for i, (d, s) in enumerate(zip(datasets, subjects)):
        per_subject.extend(rows(unit[i], [i], None, dataset=d, subject=s))
    for d in pd.unique(datasets):
        i = np.flatnonzero(datasets == d)
        point = unit[i].mean(axis=0); means.append(point)
        per_dataset.extend(rows(point, i, within[d], dataset=d))
    macro = rows(np.mean(means, axis=0), np.arange(len(unit)), ci)
    return macro, per_dataset, per_subject


def _inventory(data):
    counts = [f.groupby(SUBJECT, observed=True).size().rename('n_'+m+'_input') for m, f in data.items()]
    return pd.concat(counts, axis=1).fillna(0).astype(int)


def _coverage(inventory, eligible, used, split, lag_s, panels):
    table = inventory.copy()
    for name, meta in [('eligible', eligible), ('used', used)]:
        for suffix, frame in [('pairs', meta), ('origins', meta.drop_duplicates(CELL))]:
            table[f'n_{name}_{suffix}'] = frame.groupby(SUBJECT, observed=True).size().reindex(table.index, fill_value=0)
    rows = []
    for (d, s), counts in table.iterrows():
        if counts.n_used_pairs:
            status = 'ok' if counts.n_used_pairs == counts.n_eligible_pairs else 'partial_training_support'
        elif not panels.get(d, {}).get('eeg') or not panels.get(d, {}).get('hb'):
            status = 'no_training_panel'
        else:
            status = 'no_contiguous_paired_support' if not counts.n_eligible_pairs else 'positions_without_fit_support'
        rows.append(dict(split=split, dataset=d, subject=s, lag_s=lag_s, status=status, **counts.to_dict()))
    return rows


def _usage(meta, split, lag_s, panels):
    result = []
    for (d, mask), g in meta.groupby(['dataset', 'eeg_mask'], sort=True, observed=True):
        names = tuple(p for i, p in enumerate(panels[d]['eeg']) if mask & (1 << i))
        n = len(g[CELL].drop_duplicates())
        result.append(dict(split=split, lag_s=lag_s, dataset=d, eeg_positions=names,
                           hb_positions_used=tuple(g.position.unique()), n_pairs=len(g), n_origins=n,
                           eeg_bits_per_origin=4 * len(names), total_eeg_word_bits=4 * len(names) * n))
    return result


def run_coupling(train_eeg, train_hb, val_eeg, val_hb):
    """Evaluate 4/6/8-s leads on strictly separate, dataset-qualified subjects.

    Inputs: DataFrames with ``KEYS + ['patch']`` unique per modality/split;
    integer ``pattern`` in [0,15] and nonnegative ``rms``. EEG also supplies
    same-K ``blind`` and measured relative ``alpha,beta``; Hb supplies signed
    means ``hbo,hbr`` and observable slopes ``so,sr``. Unused columns are not
    read. Optional ``split``/``modality`` tags must agree with the argument.

    Returns a dict (no files): ``panels`` (train-selected site lists), ``coverage``
    (all supplied dataset/subjects, incl. zero support), ``panel_usage`` (actual
    contributing sites and charged bits), ``summary``, ``by_dataset``, ``by_subject``
    (long paired contrast tables), ``training_scales``, ``word_effects`` (training
    descriptive contrasts), ``protocol``, and a ``primary`` decision on the
    supported validation cohort only. No usable pairs => explicit status, empty
    estimates, and the available coverage, never a different sensor/model.
    """
    train = {m: _input(f, m, 'train') for m, f in [('eeg', train_eeg), ('hb', train_hb)]}
    val = {m: _input(f, m, 'validation') for m, f in [('eeg', val_eeg), ('hb', val_hb)]}
    tinv, vinv = _inventory(train), _inventory(val)
    if len(tinv.index.intersection(vinv.index)):
        raise ValueError('dataset-qualified subject overlap between train and validation')
    panels = _panels(train)
    tg, vg = _grid(train, panels), _grid(val, panels)
    summary = []; datasets = []; subjects = []; coverage = []; usage = []; effects = []; scales = []
    for h in LAGS:
        lag_s = 2*h
        i, j = _anchors(tg, h); u, v = _anchors(vg, h)
        tf, eligible_v = _meta(tg, i, j), _meta(vg, u, v)
        keep = _in_training_scope(tf, eligible_v)
        vf = eligible_v.loc[keep].reset_index(drop=True)
        coverage.extend(_coverage(tinv, tf, tf, 'train', lag_s, panels))
        coverage.extend(_coverage(vinv, eligible_v, vf, 'validation', lag_s, panels))
        usage.extend(_usage(tf, 'train', lag_s, panels)); usage.extend(_usage(vf, 'validation', lag_s, panels))
        if not len(tf) or not len(vf):
            continue
        with threadpool_limits(limits=1, user_api='blas'):
            loss, coeffs, scale = _fit(tf, vf, _blocks(tg, i, j, h), _blocks(vg, u[keep], v[keep], h), lag_s)
            macro, by_dataset, by_subject = _summaries(vf, loss, lag_s)
        summary.extend(macro); datasets.extend(by_dataset); subjects.extend(by_subject)
        effects.extend(coeffs); scales.append(scale)

    def at_primary(row):
        return row['lag_s'] == 6 and row['endpoint'] == 'hb_delta' and row['candidate'] == 'pattern'
    required = ('base', 'shifted', 'blind')
    primary = {r['reference']: r for r in summary if at_primary(r) and r['reference'] in required}
    if not primary:
        decision = 'no_paired_support'
    elif any(v['ci_above_zero'] is None for v in primary.values()):
        decision = 'insufficient_independent_units'
    elif all(primary[r]['ci_above_zero'] for r in required):
        decision = 'above_controls'
    else:
        decision = 'not_demonstrated'
    panel_rows = [dict(dataset=d, eeg_positions=p['eeg'], hb_positions=p['hb']) for d, p in panels.items()]
    return dict(status='ok' if summary else 'no_paired_support',
                protocol=dict(patch_s=2, k=K, primary_lag_s=6, primary_endpoint='hb_delta',
                              lead='target patch start minus visible patch start; at 6s the post-origin gap is 4s',
                              association_only=True, colocation_assumed=False, native_clock_audit='caller required',
                              sites='per modality/dataset: <=2 by training patch coverage; tie by name',
                              base='dataset, Hb site, EEG site mask, crop phase/phase^2, log EEG RMS per site at t and t-h; '
                                   'target-site Hb word, levels, slopes, log RMS at t/t-1/t-2',
                              shifted='same histogram moved from t to strictly past t-h, fit on wrong-time train inputs',
                              continuous='measured current alpha/beta mean and SD over the same EEG sites',
                              delta='target-site signed mean at t+h minus mean at t; train variance per color/lag',
                              word='future Hb pattern, independent ridge one-hot target, simplex projection, summed K-way Brier',
                              effects='pattern fit: unit EEG histogram minus weighted train mixture, base fixed, raw delta units',
                              gain='reference_loss - candidate_loss; positive favors candidate', ridge_penalty=RIDGE,
                              target_variance_floor=VAR_FLOOR, eeg_bits_per_site_patch=4,
                              rate_scope='incremental EEG word bits per distinct origin x contributing EEG sites, not per Hb target; '
                                         'do not sum horizons. Common features and physical EEG attributes are side information, '
                                         'not a zero-rate end-to-end codec.',
                              averaging='equal sites/origin, origins/crop, crops/record, records/subject, subjects/dataset, datasets',
                              ci='paired 95% percentile; macro resamples datasets then subjects within each drawn copy; '
                                 'per-dataset resamples subjects; no window/position replicates or subject-level CI',
                              bootstrap_draws=BOOTSTRAPS, bootstrap_seed_base=SEED, ci_conditional_on='frozen models and scored validation cohort',
                              multiplicity='6s joint delta prespecified: all three paired lower limits >0; other endpoints/contrasts descriptive'),
                primary=dict(status=decision, references=required, ci_by_reference={r: [v['ci_low'], v['ci_high']] for r, v in primary.items()}),
                panels=pd.DataFrame(panel_rows, columns=['dataset', 'eeg_positions', 'hb_positions']),
                coverage=pd.DataFrame(coverage), panel_usage=pd.DataFrame(usage), summary=pd.DataFrame(summary),
                by_dataset=pd.DataFrame(datasets), by_subject=pd.DataFrame(subjects),
                training_scales=pd.DataFrame(scales), word_effects=pd.DataFrame(effects))
