"""Explicit finite time-shift/sign EEG atoms; every expanded atom is one paid ID."""
from __future__ import annotations
import numpy as np
from .anchored_carrier_dictionary import AnchoredCarrierDictionary, load_model as load_base
from .spatial_waveform_dictionary import BANDS, spectral_features


def frequency_indices(band):
    f = np.fft.rfftfreq(400, 1/200)
    lo, hi = BANDS[band]
    return np.flatnonzero((f >= lo) & (f < hi))


def waveform_features(features, band):
    q = np.asarray(features, float)
    bins = frequency_indices(band); n = len(bins)
    if q.ndim != 2 or q.shape[1] != 2*n or not np.isfinite(q).all():
        raise ValueError('finite fixed-band real Fourier coordinates required')
    z = np.zeros((len(q), 201), complex)
    z[:, bins] = (q[:, :n]+1j*q[:, n:])/np.sqrt(2)
    return np.fft.irfft(z, n=400, norm='ortho')


def expansion(parents, band, shifts):
    p = np.asarray(parents, float); bins = frequency_indices(band); n = len(bins)
    phase = np.exp(-2j*np.pi*np.asarray(shifts)[:, None]*bins[None]/400)
    complex_parent = p[:, :n]+1j*p[:, n:]
    expanded = complex_parent[:, None, None] * phase[None, :, None] * np.array([1., -1.])[None, None, :, None]
    return np.concatenate((expanded.real, expanded.imag), -1).reshape(-1, 2*n)


def nearest(features, atoms, chunk=512):
    q, a = np.asarray(features, float), np.asarray(atoms, float)
    ids = np.empty(len(q), np.int32); cost = np.empty(len(q))
    norm = (a*a).sum(1)
    for start in range(0, len(q), chunk):
        values = q[start:start+chunk]
        distance = (values*values).sum(1)[:, None]+norm-2*values@a.T
        index = distance.argmin(1)
        ids[start:start+len(values)] = index
        cost[start:start+len(values)] = distance[np.arange(len(values)), index].clip(min=0)
    return ids, cost


def fit_book(carriers, band, *, method, parent_count, shifts, iterations, seed):
    from sklearn.cluster import MiniBatchKMeans
    values = np.asarray(carriers, float).reshape(-1, 400)
    q = spectral_features(values, band)
    expanded_count = parent_count*len(shifts)*2
    if len(q) < expanded_count or method not in ('ordinary', 'orbit'):
        raise ValueError('matched finite vocabulary must fit training population')
    rng = np.random.default_rng(seed)
    if method == 'ordinary':
        km = MiniBatchKMeans(n_clusters=expanded_count, n_init=1, random_state=seed,
            batch_size=2048, max_iter=8, reassignment_ratio=0., max_no_improvement=8)
        centers = km.fit(q).cluster_centers_
    else:
        parents = [q[rng.integers(len(q))]]
        minimum = np.full(len(q), np.inf)
        for _ in range(parent_count-1):
            _, d = nearest(q, expansion(np.asarray(parents[-1:]), band, shifts))
            minimum = np.minimum(minimum, d)
            take = rng.choice(len(q), p=minimum/minimum.sum()) if minimum.sum() > 1e-12 else rng.integers(len(q))
            parents.append(q[take])
        parents = np.stack(parents)
        centers = expansion(parents, band, shifts)
    history = []
    bins = frequency_indices(band); n = len(bins)
    for _ in range(iterations):
        labels, costs = nearest(q, centers)
        history.append(float(costs.mean()/400))
        if method == 'ordinary':
            counts = np.bincount(labels, minlength=expanded_count)
            total = np.zeros_like(centers); np.add.at(total, labels, q)
            occupied = counts > 0
            centers[occupied] = total[occupied]/counts[occupied, None]
        else:
            parent = labels//(len(shifts)*2)
            shift = np.asarray(shifts)[(labels//2) % len(shifts)]
            sign = 1-2*(labels % 2)
            rotation = sign[:, None]*np.exp(2j*np.pi*shift[:, None]*bins[None]/400)
            aligned = (q[:, :n]+1j*q[:, n:])*rotation
            aligned = np.concatenate((aligned.real, aligned.imag), 1)
            counts = np.bincount(parent, minlength=parent_count)
            total = np.zeros_like(parents); np.add.at(total, parent, aligned)
            occupied = counts > 0
            parents[occupied] = total[occupied]/counts[occupied, None]
            centers = expansion(parents, band, shifts)
    labels, costs = nearest(q, centers)
    counts = np.bincount(labels, minlength=expanded_count)
    atoms = np.vstack((np.zeros((1, 400)), waveform_features(centers, band)))
    report = dict(method=method, training_patches=len(q), positive_atoms=expanded_count,
        parent_count=parent_count if method == 'orbit' else None, shifts_samples=list(shifts),
        projection_loss=float(np.mean((values-waveform_features(q, band))**2)),
        initial_and_iteration_projected_MSE=history, final_projected_MSE=float(costs.mean()/400),
        occupied_fit_sample_atoms=int((counts>0).sum()), atom_RMS_quantiles=np.quantile(np.sqrt((atoms[1:]**2).mean(1)), [0,.1,.5,.9,1]).tolist())
    if method == 'orbit':
        parent = labels//(len(shifts)*2)
        count = np.bincount(parent, minlength=parent_count)
        raw = np.zeros_like(parents); np.add.at(raw, parent, q)
        raw /= np.maximum(count[:, None], 1)
        den = max(float((q*q).sum()), 1e-12)
        report.update(aligned_parent_mean_energy_fraction=float((count[:, None]*parents**2).sum()/den),
            unaligned_same_parent_mean_energy_fraction=float((count[:, None]*raw**2).sum()/den),
            final_ID_mapping='1+2*(16*parent+shift_index)+sign_index; sign_index0=+1,1=-1')
    # Keep the existing frozen carrier decoder schema; the last three columns
    # are explicitly unused and do not receive true or estimated attributes.
    book = dict(mean=np.zeros(q.shape[1]+3), weight=np.r_[np.ones(q.shape[1]), np.zeros(3)],
        centers=np.column_stack((centers, np.zeros((len(centers), 3)))), atoms=atoms,
        counts=np.r_[0, counts], iterations=iterations)
    return book, report


class PhaseOrbitDictionary(AnchoredCarrierDictionary):
    def assign(self, carriers, physical, dataset, observed):
        shape = np.shape(carriers['S']); roles = self.roles(dataset); observed = np.asarray(observed)
        if (len(shape) != 4 or shape[1:] != (len(roles), 15, 400) or np.shape(carriers['O']) != shape
                or observed.shape != shape[:2] or observed.dtype != bool):
            raise ValueError('matched complete native carrier shapes and observation flags required')
        result = {}
        for branch in 'SO':
            codes = np.zeros(shape[:3], np.int32)
            for band in range(5):
                selected = np.flatnonzero(roles == band)
                query = spectral_features(carriers[branch][:, selected].reshape(-1, 400), band)
                index, _ = nearest(query, self.books[branch, band]['centers'][:, :-3])
                codes[:, selected] = (index+1).reshape(shape[0], len(selected), 15)
            codes[~observed] = 0
            result[branch] = codes
        return result


def load_model(path):
    source = load_base(path)
    if source.metadata.get('experiment_family') != 'phase_orbit_dictionary_v1':
        raise ValueError('versioned phase-orbit experiment required')
    return PhaseOrbitDictionary(source.source, source.books, source.metadata)
