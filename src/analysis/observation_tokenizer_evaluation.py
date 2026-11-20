"""Observation-only tokenizer endpoints. The subject within dataset is the independent unit."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder


# Ordered from coarse to fine. A position and a patch are not independent subjects.
KEYS = ['dataset', 'subject', 'record', 'parent', 'start_s', 'position']
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def hierarchy_weights(frame):
    """Each level (including the remaining patches) has equal conditional mass."""
    if len(frame) == 0 or frame[KEYS].isna().any().any():
        raise ValueError('empty or missing evaluation identity')
    w = np.ones(len(frame), dtype=float) / frame.dataset.nunique()
    # parent and start_s jointly identify one physical window within a record.
    for group, nextkeys in [(KEYS[:1], ['subject']), (KEYS[:2], ['record']),
                            (KEYS[:3], ['parent', 'start_s']), (KEYS[:5], ['position'])]:
        unique = frame[group + nextkeys].drop_duplicates()
        counts = unique.groupby(group, sort=False).size().rename('_count').reset_index()
        w /= frame[group].merge(counts, on=group, how='left', sort=False)['_count'].to_numpy()
    w /= frame.groupby(KEYS, sort=False).subject.transform('size').to_numpy()
    if not np.isclose(w.sum(), 1):
        raise ValueError('incomplete panel hierarchy')
    return w


def aggregate(frame, values):
    """Patch→site/window→window→record→subject→dataset, then equal datasets."""
    a = pd.DataFrame(values, index=frame.index)
    if len(a) != len(frame) or not np.isfinite(a.to_numpy()).all():
        raise ValueError('nonfinite or incomplete endpoint')
    columns = a.columns.tolist()
    a = pd.concat([frame[KEYS], a], axis=1)
    for keys in [KEYS, KEYS[:5], KEYS[:3], KEYS[:2]]:
        a = a.groupby(keys, sort=True, as_index=False)[columns].mean()
    dataset = a.groupby('dataset', sort=True, as_index=False)[columns].mean()
    macro = dataset[columns].mean().to_dict()
    return pd.concat([pd.DataFrame([dict(level='macro', dataset='', subject='', **macro)]),
                      dataset.assign(level='dataset', subject=''), a.assign(level='subject')], ignore_index=True)


def _cosine(a, b):
    dot = np.real(np.sum(a.conj() * b, axis=-1))
    n = np.sqrt(np.sum(abs(a)**2, axis=-1) * np.sum(abs(b)**2, axis=-1))
    return np.divide(dot, n, out=np.zeros_like(dot), where=n > 1e-24)


def reconstruction(frame, x, pred, vx, modality):
    x, pred = np.asarray(x, dtype=float), np.asarray(pred, dtype=float)
    if x.shape != pred.shape or x.ndim != 3 or vx.shape != x.shape[:2] or not (vx > 0).all():
        raise ValueError('expected [patch,color,time] and paired training variances')
    r = x - pred
    level = (np.mean(r, -1)**2 / vx).mean(1)
    shape = (np.var(r, -1) / vx).mean(1)
    a, b = x - x.mean(-1, keepdims=True), pred - pred.mean(-1, keepdims=True)
    mse = (np.mean(r*r, -1) / vx)
    out = dict(x_nmse=mse.mean(1), level_nmse=level, shape_nmse=shape,
               centered_cosine=_cosine(a, b).mean(1))
    if modality == 'hb':
        out.update(hbo_nmse=mse[:, 0], hbr_nmse=mse[:, 1])
    else:
        win = np.hanning(a.shape[-1]); f = np.fft.rfftfreq(a.shape[-1], 1/200.)
        t, y = np.fft.rfft(a[:, 0] * win), np.fft.rfft(b[:, 0] * win)
        for lo, hi in BANDS:
            use = (f >= lo) & (f < hi); u, v = t[:, use], y[:, use]
            # A shared floor in frozen training-X coordinates (never fitted to a validation patch).
            out[f'band_{lo}_{hi}_db'] = 10*np.log10((np.sum(abs(v)**2, -1)+1e-12) /
                                                     (np.sum(abs(u)**2, -1)+1e-12))
            out[f'band_{lo}_{hi}_cosine'] = _cosine(u, v)
    return aggregate(frame, out)


def secondary_attributes(x, modality):
    """Not the encoder's feature vector. Signs/amplitudes are computed from held-out X."""
    x = np.asarray(x, dtype=float)
    if modality == 'eeg':
        a = x[:, 0] - x[:, 0].mean(1, keepdims=True)
        energy = np.sum(a*a, 1)
        return np.stack([np.sum(a[:, :-lag] * a[:, lag:], 1) / np.maximum(energy, 1e-20)
                         for lag in (4, 8, 12, 20, 32, 50)], axis=1)
    return np.mean(x[:, :, -5:], axis=-1) - np.mean(x[:, :, :5], axis=-1)


def _background(train, test, a, b, weights, modality):
    def raw(frame, x):
        rms = np.sqrt(np.mean(x.astype(float)**2, (1, 2)))
        coarse = np.digitize(rms, [.25, .5, 1., 2., 4.]).astype(str)
        site = (frame['dataset'].astype(str) + '/' + frame['position'].astype(str)).to_numpy()
        bins = np.asarray([s + '/' + v for s, v in zip(site, coarse)])
        amp = np.log(np.maximum(rms, 1e-8))
        nums = np.column_stack([amp, amp**2])
        if modality == 'hb':
            nums = np.column_stack([nums, x.mean(axis=2)])
        return np.column_stack([site, coarse, bins]), nums
    cat, num = raw(train, a); cat2, num2 = raw(test, b)
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    from scipy.sparse import hstack
    m = np.average(num, axis=0, weights=weights)
    sd = np.sqrt(np.maximum(np.average((num-m)**2, axis=0, weights=weights), 1e-8))
    return (hstack([enc.fit_transform(cat), (num-m)/sd], format='csr'),
            hstack([enc.transform(cat2), (num2-m)/sd], format='csr'))


def _pairs(frame, lag):
    """Never predicts across a missing patch, site, window, record or subject."""
    if frame.duplicated(KEYS + ['patch']).any():
        raise ValueError('duplicate patch identity')
    if lag == 0:
        return np.arange(len(frame)), np.arange(len(frame))
    index = pd.MultiIndex.from_frame(frame[KEYS + ['patch']]); target = frame[KEYS + ['patch']].copy()
    target['patch'] += lag
    after = index.get_indexer(pd.MultiIndex.from_frame(target)); before = np.flatnonzero(after >= 0)
    return before, after[before]


def roll_patterns(frame, ids):
    """Half-window rotation within the observed complete patches of each physical series."""
    if not frame.index.equals(pd.RangeIndex(len(frame))):
        raise ValueError('contiguous row indices required')
    shifted = np.array(ids, copy=True)
    for rows in frame.groupby(KEYS, sort=False).indices.values():
        rows = rows[np.argsort(frame.patch.to_numpy()[rows])]
        shifted[rows] = np.roll(ids[rows], len(rows)//2)
    return shifted


def probe_attributes(train, val, xt, xv, train_ids, val_ids, modality, k):
    """Frozen additive ridge against a train-only background, and a matched assignment null."""
    from scipy.sparse import csr_matrix, hstack
    targets = [secondary_attributes(x, modality) for x in (xt, xv)]
    blocks = []
    for lag in [0, 1]:
        i, j = _pairs(train, lag); u, v = _pairs(val, lag)
        tf = train.iloc[i].reset_index(drop=True); vf = val.iloc[u].reset_index(drop=True)
        weights = hierarchy_weights(tf)
        a, b = _background(tf, vf, xt[i], xv[u], weights, modality)
        yt, yv = targets[0][j], targets[1][v]
        mean = np.average(yt, axis=0, weights=weights)
        sd = np.sqrt(np.maximum(np.average((yt-mean)**2, axis=0, weights=weights), 1e-12))
        yt, yv = (yt-mean)/sd, (yv-mean)/sd
        onehot = np.eye(k)
        loss = {}
        for key in ['background', 'pattern', 'wave_kmeans', 'rolled_pattern']:
            xa, xb = a, b
            if key != 'background':
                name = 'pattern' if key == 'rolled_pattern' else key
                ia, ib = train_ids[name], val_ids[name]
                # Apply a frozen genuine readout to the wrong site/time-matched ID at validation.
                if key == 'rolled_pattern':
                    ib = roll_patterns(val, ib)
                xa = hstack([a, csr_matrix(onehot[ia[i]])], format='csr')
                xb = hstack([b, csr_matrix(onehot[ib[u]])], format='csr')
            reg = Ridge(alpha=1., solver='lsqr', tol=1e-6).fit(xa, yt, sample_weight=weights*len(i))
            loss[key] = ((yv - reg.predict(xb))**2).mean(1)
        table = aggregate(vf, loss)
        for name in ['pattern', 'wave_kmeans', 'rolled_pattern']:
            table[name+'_gain'] = 1. - table[name] / table.background
        blocks.append(table.assign(lag_patches=lag, modality=modality, n_train_pairs=len(i), n_validation_pairs=len(u)))
    return pd.concat(blocks, ignore_index=True)


def pattern_atlas(frame, ids, features, names, k, modality, split):
    """Empirical weighted quantiles; support counts are distinct dataset-qualified units."""
    features = np.asarray(features)
    if features.shape != (len(frame), len(names)) or not np.isfinite(features).all():
        raise ValueError('nonfinite or mismatched atlas features')
    weights = hierarchy_weights(frame)
    result = []
    for code in range(k):
        sel = ids == code; local = frame.loc[sel]; w = weights[sel]
        count = dict(n_subjects=len(local[KEYS[:2]].drop_duplicates()), n_records=len(local[KEYS[:3]].drop_duplicates()),
                     n_datasets=local.dataset.nunique(), n_patches=len(local), mass=float(w.sum()))
        for col, label in enumerate(names):
            q = [np.nan]*3
            if len(w):
                vals = features[sel, col]; order = np.argsort(vals); cumul = w[order].cumsum()/w.sum()
                q = vals[order[np.searchsorted(cumul, [.25, .5, .75])]].tolist()
            result.append(dict(modality=modality, split=split, pattern_id=code, feature=label,
                               q25=q[0], median=q[1], q75=q[2], **count))
    return pd.DataFrame(result)


def paired_contrast(table, metric, reference, candidate, seed=4801):
    """Dataset-stratified paired subject bootstrap; tuples are (method, bits_per_patch)."""
    t = table[table.level == 'subject']
    def vec(key):
        return t[(t.method == key[0]) & (t.bits_per_patch == key[1])].set_index(KEYS[:2])[metric].sort_index()
    a, b = vec(reference), vec(candidate)
    if len(a) == 0 or not a.index.equals(b.index) or not a.index.is_unique:
        raise ValueError('incomplete paired denominator')
    delta = b - a; rng = np.random.default_rng(seed); reps = []
    for _, col in delta.groupby(level=0):
        y = col.to_numpy(); reps.append(y[rng.integers(0, len(y), (4000, len(y)))].mean(1))
    lo, hi = np.quantile(np.mean(reps, 0), [.025, .975])
    return dict(metric=metric, reference=list(reference), candidate=list(candidate),
                candidate_minus_reference=float(delta.groupby(level=0).mean().mean()),
                ci95=[float(lo), float(hi)], n_subjects=len(a), n_lower=int((delta < 0).sum()))
