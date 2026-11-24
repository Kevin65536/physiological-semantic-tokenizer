"""Frozen low-vocabulary EEG sequences distilled into internal high-code atoms."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .spatial_waveform_dictionary import spectral_features


def integer_fields(ids, branch, neighbors, *, vocabulary=512):
    """Eight categorical fields within one 30-second integer context; 512 is a boundary sentinel."""
    if set(ids) != {'S', 'O'}:
        raise ValueError('both declared integer streams required')
    a, b = np.asarray(ids[branch]), np.asarray(ids['O' if branch == 'S' else 'S'])
    if (a.ndim != 3 or a.shape != b.shape or a.shape[-1] != 15
            or any(v.dtype.kind not in 'iu' or np.any(v < 0) or np.any(v >= vocabulary) for v in (a, b))):
        raise ValueError('matching in-vocabulary [batch,native,15] integers required')
    neighbors = np.asarray(neighbors)
    if neighbors.shape != (a.shape[1], 2) or neighbors.dtype.kind not in 'iu' or np.any(neighbors < 0) or np.any(neighbors >= a.shape[1]):
        raise ValueError('two fixed native neighbor indices per site required')
    past, future = np.full_like(a, vocabulary), np.full_like(a, vocabulary)
    past[..., 1:], future[..., :-1] = a[..., :-1], a[..., 1:]
    return np.stack((past, a, future, b, a[:, neighbors[:, 0]], a[:, neighbors[:, 1]],
                     b[:, neighbors[:, 0]], b[:, neighbors[:, 1]]), axis=-1)


def neighbor_graph(source):
    """Two training-frozen synthesis-row neighbors of each site, within the same spectral role."""
    graph = {}
    for dataset, names in source.metadata['positions'].items():
        for branch in ('S', 'O'):
            result = np.empty((len(names), 2), np.int32)
            roles = source.roles(dataset, branch)
            for band in range(5):
                slots = np.flatnonzero(roles == band)
                w = source.weights[dataset, branch, band]
                if w.shape[0] != len(slots) or len(slots) < 3:
                    raise ValueError('at least three same-role synthesis rows required')
                unit = w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-15)
                similarity = unit @ unit.T
                np.fill_diagonal(similarity, -np.inf)
                for j, slot in enumerate(slots):
                    result[slot] = slots[np.argsort(-similarity[j], kind='stable')[:2]]
            graph[dataset, branch] = result
    return graph


def fit_additive(fields, residual, weight, *, prior=16., passes=5, current_only=False):
    fields, residual, weight = np.asarray(fields), np.asarray(residual, float), np.asarray(weight, float)
    if (fields.ndim != 2 or fields.shape[1] != 8 or residual.ndim != 2 or len(fields) != len(residual)
            or weight.shape != (len(fields),) or len(fields) == 0 or np.any(weight <= 0)
            or not np.isfinite(residual).all()):
        raise ValueError('finite paired training residuals and positive sample weights required')
    weight = weight * len(weight) / weight.sum()
    intercept = np.average(residual, axis=0, weights=weight)
    active = (1,) if current_only else tuple(range(8))
    tables = np.zeros((8, 513, residual.shape[1]))
    prediction = np.broadcast_to(intercept, residual.shape).copy()
    order = {}
    for field in active:
        idx = np.argsort(fields[:, field], kind='stable')
        unique, first = np.unique(fields[idx, field], return_index=True)
        mass = np.add.reduceat(weight[idx], first)
        order[field] = idx, unique, first, mass
    losses = []
    for _ in range(1 if current_only else passes):
        for field in active:
            index = fields[:, field]
            prediction -= tables[field, index]
            idx, unique, first, mass = order[field]
            totals = np.add.reduceat(((residual-prediction)*weight[:, None])[idx], first, axis=0)
            tables[field] = 0
            tables[field, unique] = totals / (mass[:, None]+prior)
            prediction += tables[field, index]
        losses.append(float(np.average(np.mean((residual-prediction)**2, axis=1), weights=weight)))
    return {'tables': tables, 'intercept': intercept}, dict(samples=len(fields), prior=prior, passes=len(losses),
        active_fields=list(active), training_weighted_residual_MSE=losses,
        observed_categories=[int(len(np.unique(fields[:, j]))) for j in range(8)])


def predict_features(fields, baseline, state):
    value = np.asarray(baseline, float)+state['intercept']
    for field in range(8):
        value = value+state['tables'][field, np.asarray(fields)[..., field]]
    return value


def nearest_internal(features, teacher_features, *, block=128):
    """Hard nearest teacher atom. ID zero remains reserved for missing telemetry."""
    features = np.asarray(features, float)
    teacher = np.asarray(teacher_features, float)
    flat = features.reshape(-1, features.shape[-1])
    norm = np.square(teacher[1:]).sum(1)
    values = []
    for start in range(0, len(flat), block):
        item = flat[start:start+block]
        distance = np.square(item).sum(1, keepdims=True)+norm-2*item@teacher[1:].T
        values.append(distance.argmin(1)+1)
    return np.concatenate(values).reshape(features.shape[:-1]).astype(np.int32)


class EEGIntegerDistillationDecoder:
    """Only K512 payloads enter; inferred high IDs are private deterministic decoder state."""
    def __init__(self, low, high, states, graph, metadata):
        self.low, self.high, self.states, self.graph, self.metadata = low, high, states, graph, metadata
        source = getattr(high, 'decoder', high)
        source = getattr(source, 'source', source)
        self.teacher = {(b, k): spectral_features(source.atoms[b, k], k) for b in ('S', 'O') for k in range(5)}
        self.baseline = {(b, k): spectral_features(low.atoms[b, k], k) for b in ('S', 'O') for k in range(5)}

    def infer_ids(self, ids, dataset):
        result = {}
        if dataset not in self.metadata['positions']:
            raise ValueError('fixed native layout required')
        for branch in ('S', 'O'):
            fields = integer_fields(ids, branch, self.graph[dataset, branch])
            roles = self.low.roles(dataset, branch)
            codes = np.asarray(ids[branch])
            if codes.shape[1] != len(roles):
                raise ValueError('complete fixed native array required')
            out = np.zeros(codes.shape, np.int32)
            for band in range(5):
                sites = np.flatnonzero(roles == band)
                local = codes[:, sites]
                feature = predict_features(fields[:, sites], self.baseline[branch, band][local], self.states[branch, band])
                out[:, sites] = nearest_internal(feature, self.teacher[branch, band])
            out[codes == 0] = 0
            result[branch] = out
        return result

    def decode_ids(self, ids, dataset):
        # No observations, teacher IDs, targets, local scales, or caller state accepted.
        return self.high.decode_ids(self.infer_ids(ids, dataset), dataset)


def load_model(path):
    from .spatial_waveform_dictionary import load_model as load_spatial
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z['metadata']))
        if metadata['schema'] != 'eeg_low_integer_teacher_decoder_v1':
            raise ValueError('versioned low-integer-only student required')
        references = {}
        for name in ('low_decoder', 'high_decoder'):
            ref = metadata[name]
            p = (path.parent/ref['path']).resolve()
            if hashlib.sha256(p.read_bytes()).hexdigest() != ref['sha256']:
                raise ValueError('frozen decoder identity changed')
            references[name] = p
        book_ref = metadata['student_state']
        book_path = (path.parent/book_ref['path']).resolve()
        if hashlib.sha256(book_path.read_bytes()).hexdigest() != book_ref['sha256']:
            raise ValueError('frozen student state changed')
    with np.load(book_path, allow_pickle=False) as z:
        states = {(b, k): {f: z[f'{b}_{k}_{f}'].copy() for f in ('tables', 'intercept')} for b in ('S', 'O') for k in range(5)}
        graph = {(d, b): z[f'neighbors_{d}_{b}'].copy() for d in metadata['positions'] for b in ('S', 'O')}
    return EEGIntegerDistillationDecoder(load_spatial(references['low_decoder']), load_spatial(references['high_decoder']), states, graph, metadata)


def synthesize_full_atoms(high_decoder, atoms, availability_ids, dataset):
    """Route internal full 400-sample atoms through exactly the frozen teacher synthesis.

    Private routing indices identify computed atoms, never transmitted information.
    No original decoder object or frozen weight is mutated.
    """
    import copy
    from .eeg_component_transfer import IntegerSumSourceDecoder
    from .spatial_local_gain import LocalEnergyGainDecoder
    outer = copy.copy(high_decoder)
    gain = copy.copy(outer.decoder) if isinstance(outer, IntegerSumSourceDecoder) else copy.copy(outer)
    source = copy.copy(gain.source) if isinstance(gain, LocalEnergyGainDecoder) else copy.copy(gain)
    source.atoms = {}
    route = {}
    maximum = 0
    for branch in ('S', 'O'):
        given = np.asarray(availability_ids[branch])
        values = np.asarray(atoms[branch], float)
        if (given.ndim != 3 or given.shape[0] != 1 or values.shape != (*given.shape, 400)
                or not np.isfinite(values).all()):
            raise ValueError('one complete native context and finite internal full atoms required')
        roles = source.roles(dataset, branch)
        if len(roles) != given.shape[1]:raise ValueError('fixed native support required')
        route[branch] = np.zeros_like(given)
        for band in range(5):
            sites = np.flatnonzero(roles == band)
            block = values[0, sites].reshape(-1, 400)
            source.atoms[branch, band] = np.vstack((np.zeros((1, 400)), block))
            route[branch][:, sites] = np.arange(1, len(block)+1).reshape(1, len(sites), 15)
            maximum = max(maximum, len(block))
        route[branch][given == 0] = 0
    source.metadata = dict(source.metadata, codebook_size=maximum+1)
    source.codebook_size = maximum+1
    if isinstance(gain, LocalEnergyGainDecoder):gain.source = source
    else:gain = source
    if isinstance(outer, IntegerSumSourceDecoder):outer.decoder = gain
    else:outer = gain
    return outer.decode_ids(route, dataset)


def fit_full_atom_projection(features, atoms, weighted_counts, *, ridge_fraction=1e-6):
    """Fit-only linear full-wave dictionary, preserving role-band and predicting cross-band terms."""
    f, y, count = np.asarray(features,float), np.asarray(atoms,float), np.asarray(weighted_counts,float)
    if f.shape[0] != y.shape[0] or y.shape[1] != 400 or count.shape != (len(f),) or np.any(count < 0) or count[0] != 0:
        raise ValueError('matched teacher features/full atoms and fit-only positive-ID support required')
    keep=count>0;f,y,w=f[keep],y[keep],count[keep];w=w/w.sum()
    fm,ym=np.average(f,axis=0,weights=w),np.average(y,axis=0,weights=w)
    fc,yc=f-fm,y-ym
    gram=(fc*w[:,None]).T@fc;cross=(fc*w[:,None]).T@yc
    ridge=ridge_fraction*max(float(np.trace(gram))/len(gram),1e-20)
    coefficient=np.linalg.solve(gram+ridge*np.eye(len(gram)),cross)
    intercept=ym-fm@coefficient
    error=f@coefficient+intercept-y
    return dict(coefficient=coefficient,intercept=intercept),dict(fit_used_IDs=int(keep.sum()),ridge_fraction=ridge_fraction,
        fit_full_atom_weighted_MSE=float(np.sum(w*np.mean(error**2,axis=1))),
        fit_full_atom_target_energy=float(np.sum(w*np.mean(y*y,axis=1))),
        semantics='full400_samples_fitted_from_actual_fit_source_f_high_atom_occupancy_no_held_counts')


class DirectEEGIntegerDecoder:
    """Low integers determine continuous internal atoms; the transmitted alphabet stays K512."""
    def __init__(self, student, projections):
        self.student, self.projections = student, projections
        self.metadata = dict(student.metadata, schema='eeg_low_integer_direct_full_atom_v1')

    def full_atoms(self, ids, dataset, *, teacher_ids=None):
        result={}
        for branch in ('S','O'):
            low=np.asarray(ids[branch]);roles=self.student.low.roles(dataset,branch)
            fields=integer_fields(ids,branch,self.student.graph[dataset,branch])
            full=np.zeros((*low.shape,400))
            for band in range(5):
                slots=np.flatnonzero(roles==band)
                if teacher_ids is None:
                    feature=predict_features(fields[:,slots],self.student.baseline[branch,band][low[:,slots]],self.student.states[branch,band])
                else:
                    feature=self.student.teacher[branch,band][np.asarray(teacher_ids[branch])[:,slots]]
                projection=self.projections[branch,band]
                full[:,slots]=feature@projection['coefficient']+projection['intercept']
            full[low==0]=0
            result[branch]=full
        return result

    def decode_ids(self, ids, dataset):
        return synthesize_full_atoms(self.student.high,self.full_atoms(ids,dataset),ids,dataset)

    def projection_control(self, low_ids, teacher_ids, dataset):
        """Explicit privileged control, separate from the candidate ID-only interface."""
        return synthesize_full_atoms(self.student.high,self.full_atoms(low_ids,dataset,teacher_ids=teacher_ids),low_ids,dataset)


def load_direct_model(path):
    path=Path(path)
    with np.load(path,allow_pickle=False) as z:metadata=json.loads(str(z['metadata']))
    if metadata['schema']!='eeg_low_integer_direct_full_atom_v1':raise ValueError('versioned direct atom decoder required')
    refs={}
    for name in ('student','projection'):
        p=(path.parent/metadata[name]['path']).resolve()
        if hashlib.sha256(p.read_bytes()).hexdigest()!=metadata[name]['sha256']:raise ValueError('frozen direct decoder changed')
        refs[name]=p
    with np.load(refs['projection'],allow_pickle=False) as z:
        projections={(b,k):{f:z[f'{b}_{k}_{f}'].copy() for f in ('coefficient','intercept')} for b in ('S','O') for k in range(5)}
    return DirectEEGIntegerDecoder(load_model(refs['student']),projections)
