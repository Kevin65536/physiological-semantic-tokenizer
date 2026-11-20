"""Fixed-rate reference codec for *observations*, not a source/S–O model.

One complete, offline 2 s item is [1,400] EEG (200 Hz) or paired [2,20]
HbO/HbR (10 Hz). Supply finite, G0 training-position-scaled samples. A caller
owns split selection and masks: do not impute missing positions into this API.
Only ``fit(..., partition='train')`` learns anything. No per-item scale, phase,
index list, continuous embedding, neighbouring window or task label is passed
through to the decoder. A fitted instance is a reusable, pickleable dictionary.

* ``prototype``: a K-way ID from standardized relative Fourier powers (EEG)
  or joint levels/slopes (Hb); decode its TRAIN conditional mean waveform. In
  particular phase-invariant IDs need not reconstruct an EEG carrier well.
* ``full``: same ID, plus m fixed-width integers quantizing signed residual
  coefficients about its prototype. Its TRAIN-derived coordinate order and
  ranges depend on the ID. EEG uses four nonoverlapping 100-sample DCT-IIs;
  Hb uses orthonormal sum/difference and a 20-sample DCT-II on each mixture.
* ``flat``: no ID. Independent train mean, scalar ranges and per-coordinate
  bit allocation on that same transform, with the *entire full bit budget*.
  ``flat_pattern`` is the analogous flat codec at the prototype-only budget.
* ``routing='waveform'`` replaces only the assignment features with a fixed,
  phase-sensitive signed random projection of the input waveform. All routed
  parameters are refit on the same training inputs. This provides the same-K,
  same-bit blind routing control (both prototype and full).

With pre-shared frozen dictionaries/config, the hard payload per item is
``p=(K-1).bit_length()`` for prototype (including unused IDs), ``p+m*b`` for
full and flat, and ``p`` for flat_pattern. Divide by 2 for bits/s **per input
item** (one EEG position or paired Hb position); don't divide by channel count.
No entropy credit. ``pack`` concatenates fields MSB first, padding ONLY the
last byte of the whole batch. Container headers, shared model distribution,
modality, mode, version, item count and temporal/position framing are not
in-band or free learned per-item data. Hold that metadata fixed out of band.

An ID shuffle with identical residual integers tests conditional routing, not
physiological semantics. Compare independent outcomes at matched rate and
with the same readout (and include the waveform-route, continuous-property
and null controls). The codec fits no outcome and makes no predictive claim.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.fft import dct, idct, rfft
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits


_EEG_BANDS = ((0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45), (45, 100))
_EEG_NAMES = ('eeg_rel_0p5_4hz', 'eeg_rel_4_8hz', 'eeg_rel_8_13hz',
              'eeg_rel_13_30hz', 'eeg_rel_30_45hz', 'eeg_rel_45_100hz')
_HB_NAMES = ('hbo_mean', 'hbr_mean', 'hbo_slope_per_s', 'hbr_slope_per_s')
_TIME_HB = (np.arange(20) - 9.5) / 10
_SHAPES = {'eeg': (1, 400), 'hb': (2, 20)}
_MODES = ('prototype', 'full', 'flat', 'flat_pattern')
_MAX_WIDTH = 24  # All detail symbols and their endpoints fit in int32.


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f'{name} must be an integer')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f'{name} must be in [{low},{high}]')
    return value


def _wave(x, modality):
    if modality not in _SHAPES:
        raise ValueError("modality must be 'eeg' or 'hb'")
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[1:] != _SHAPES[modality]:
        raise ValueError(f'expected [n,{_SHAPES[modality][0]},{_SHAPES[modality][1]}]')
    if x.dtype.kind not in 'iuf' or not np.isfinite(x).all():
        raise ValueError('expected finite real observations')
    # Explicit failure, not silent waveform clipping. Includes the whole
    # float32 range; leaves headroom for DCT sums, squares and distances.
    if np.max(x, initial=0).item() > 1e120 or np.min(x, initial=0).item() < -1e120:
        raise ValueError('G0-scaled samples outside safe float64 moment range')
    return np.asarray(x, dtype=np.float64)


def _properties(x, modality):
    if modality == 'hb':
        means = x.mean(axis=2)
        slopes = np.einsum('nct,t->nc', x - means[:, :, None], _TIME_HB) / (_TIME_HB @ _TIME_HB)
        return np.concatenate((means, slopes), axis=1)
    # Per-item rescaling ONLY for dimensionless spectral fractions. No taper:
    # fractions are invariant to circular shifts and DFT phase replacement,
    # not promised invariant to cropping an arbitrary nonstationary record.
    wave = x[:, 0]
    peak = np.max(np.abs(wave), axis=1, keepdims=True)
    wave = wave / np.where(peak == 0, 1, peak)
    power = np.abs(rfft(wave - wave.mean(axis=1, keepdims=True), axis=-1, workers=1)) ** 2
    # One-sided real-signal energy: both members of every conjugate pair;
    # DC is excluded and the single Nyquist bin is counted only once.
    power[:, 1:-1] *= 2
    bands = np.column_stack([
        power[:, int(2 * lo):(None if hi == 100 else int(2 * hi))].sum(axis=1)
        for lo, hi in _EEG_BANDS
    ])
    total = bands.sum(axis=1, keepdims=True)
    return np.divide(bands, total, out=np.zeros_like(bands), where=total > 0)


def _coefficients(x, modality):
    n = len(x)
    if modality == 'eeg':
        blocks = x.reshape(n, 4, 100)
    else:
        blocks = np.stack((x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]), axis=1) / np.sqrt(2.)
    return dct(blocks, type=2, norm='ortho', axis=-1, workers=1).reshape(n, np.prod(_SHAPES[modality]))


def _synthesize(coeff, modality):
    n = len(coeff)
    blocks = coeff.reshape((n, 4, 100) if modality == 'eeg' else (n, 2, 20))
    x = idct(blocks, type=2, norm='ortho', axis=-1, workers=1)
    if modality == 'eeg':
        return x.reshape(n, 1, 400)
    return np.stack((x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]), axis=1) / np.sqrt(2.)


def _quantize(residual, radius, width):
    # Midrise when radius>0; a zero training radius codes a zero residual.
    # Clip before division: an extreme test value / tiny TRAIN range must
    # saturate at an integer endpoint, not produce Inf or wrap around.
    r = np.clip(residual, -radius, radius)
    ratio = np.divide(r, radius, out=np.zeros_like(r), where=radius > 0)
    last = np.left_shift(1, width) - 1
    return np.rint((ratio + 1) * (last / 2)).astype(np.int32)


def _dequantize(symbol, radius, width):
    last = np.left_shift(1, width) - 1
    return (symbol.astype(np.float64) * (2. / last) - 1) * radius


def _fit_radii(r, width):
    """Select scalar endpoint minimizing TRAIN coefficient MSE over a small grid."""
    if not r.shape[1]:
        return np.empty(0)
    sd = np.sqrt(np.mean(r * r, axis=0))
    # Include observed extrema (and for 1-bit, the best symmetric sign level).
    # No evaluation quantile, item-specific step or dynamic scale is used.
    radius = np.max(np.abs(r), axis=0)
    best = np.mean((_dequantize(_quantize(r, radius, width), radius, width) - r) ** 2, axis=0)
    for factor in (1.25, 2., 3., 4.):
        q = factor * sd
        mse = np.mean((_dequantize(_quantize(r, q, width), q, width) - r) ** 2, axis=0)
        improve = mse < best
        radius = np.where(improve, q, radius)
        best = np.minimum(mse, best)
    return np.where(width == 1, np.mean(np.abs(r), axis=0), radius)


@dataclass
class _Flat:
    index: np.ndarray
    widths: np.ndarray
    radius: np.ndarray


def _fit_flat(coeff, mean, variance, budget):
    # Greedily minimize the TRAIN variance proxy sum_j v_j * 4**(-b_j).
    # Not claimed an exact MSE optimizer. A high-variance coefficient may
    # receive several bits before a low-variance one receives its first.
    widths = np.zeros(len(variance), dtype=np.int32)
    next_gain = variance.copy()
    for _ in range(budget):
        j = next_gain.argmax()
        widths[j] += 1
        next_gain[j] = -np.inf if widths[j] == _MAX_WIDTH else next_gain[j] * .25
    index = np.flatnonzero(widths)
    widths = widths[index]
    radius = _fit_radii(coeff[:, index] - mean[index], widths)
    return _Flat(index, widths, radius)


class FactorizedObservationTokenizer:
    """Numpy CPU API; no mutable running batch statistics at inference.

    ``detail_count`` counts scalar residual coefficients, not time blocks or
    channels. Defaults are 48 for EEG and 8 for Hb, at 4 bits each. To target
    14 bits with K=16 use (count=2,bits=5) in either modality. Examples at
    b=4 are EEG m=63/127 for 256/512 bits and Hb m=7/15 for 32/64 bits.
    A requested alphabet may have unused IDs; report ``pattern_counts_``.
    """

    def __init__(self, modality, *, n_patterns=16, detail_count=None, detail_bits=4,
                 seed=0, routing='pattern'):
        if modality not in _SHAPES:
            raise ValueError("modality must be 'eeg' or 'hb'")
        if routing not in ('pattern', 'waveform'):
            raise ValueError("routing must be 'pattern' or 'waveform'")
        self.modality = modality
        self.routing = routing
        self.n_patterns = _integer(n_patterns, 'n_patterns', 1, 65536)
        if detail_count is None:
            detail_count = 48 if modality == 'eeg' else 8
        self.detail_count = _integer(detail_count, 'detail_count', 0, np.prod(_SHAPES[modality]))
        self.detail_bits = _integer(detail_bits, 'detail_bits', 1, 16)
        self.seed = _integer(seed, 'seed', 0, 2 ** 32 - 1)
        # Same *fixed* sketch across training/KMeans seeds and rate arms.
        if routing == 'waveform':
            d = int(np.prod(_SHAPES[modality]))
            self._sketch = np.random.default_rng(1729).choice((-1., 1.), size=(d, len(self.pattern_feature_names))) / np.sqrt(d)
        self._fitted = False

    @property
    def pattern_feature_names(self):
        return _EEG_NAMES if self.modality == 'eeg' else _HB_NAMES

    def pattern_features(self, x):
        """Physical features BEFORE train standardization; no fit required.

        For EEG all non-DC power (0.5–100 Hz) is partitioned [lo,hi), last
        band inclusive; a constant signal has all-zero fractions. Hb channels
        are HbO, HbR, and slopes are OLS vs centered time in seconds.
        These same named properties are available even with blind routing.
        """
        return _properties(_wave(x, self.modality), self.modality)

    @property
    def readout_feature_names(self):
        extra = ('eeg_mean', 'eeg_log1p_ac_rms') if self.modality == 'eeg' else ('hbo_detrended_rms', 'hbr_detrended_rms')
        return self.pattern_feature_names + extra

    def observation_features(self, x):
        """Same continuous feature coordinate for raw and every decoded arm."""
        x = _wave(x, self.modality)
        f = _properties(x, self.modality)
        if self.modality == 'eeg':
            mean = x[:, 0].mean(axis=1)
            rms = np.sqrt(np.mean((x[:, 0] - mean[:, None]) ** 2, axis=1))
            return np.column_stack((f, mean, np.log1p(rms)))
        trend = f[:, :2, None] + f[:, 2:, None] * _TIME_HB
        return np.column_stack((f, np.sqrt(np.mean((x - trend) ** 2, axis=2))))

    def _route_features(self, x, physical=None):
        if self.routing == 'pattern':
            return _properties(x, self.modality) if physical is None else physical
        return np.einsum('nd,df->nf', x.reshape(len(x), len(self._sketch)), self._sketch)

    def _assign(self, features):
        z = (features - self._route_mean) / self._route_sd
        # Bound temporary distance storage independently of evaluation batch.
        ids = np.empty(len(z), dtype=np.int32)
        step = max(1, 1_000_000 // len(self._centers))
        for i in range(0, len(z), step):
            ids[i:i + step] = cdist(z[i:i + step], self._centers, 'sqeuclidean').argmin(axis=1)
        return ids

    def fit(self, x_train, *, partition='train'):
        """Fit only on caller-supplied eligible training items, equally weighted."""
        if partition != 'train':
            raise ValueError('dictionary fitting requires partition=train')
        x = _wave(x_train, self.modality)
        if not len(x):
            raise ValueError('need nonempty training data')
        self._fitted = False
        physical = _properties(x, self.modality)
        f = self._route_features(x, physical)
        self._route_mean = f.mean(axis=0)
        self._route_sd = np.maximum(f.std(axis=0), 1e-8)
        z = (f - self._route_mean) / self._route_sd
        unique = np.unique(z, axis=0)
        if len(unique) <= self.n_patterns:
            centers = unique  # May leave unused IDs; do not shrink the rate.
        else:
            with threadpool_limits(limits=1):
                centers = KMeans(n_clusters=self.n_patterns, n_init=3, max_iter=50,
                                 random_state=self.seed, algorithm='lloyd').fit(z).cluster_centers_
        # Canonical lexicographic order for identical unordered centers.
        self._centers = centers[np.lexsort(centers.T[::-1])]
        pattern = self._assign(f)
        self.pattern_counts_ = np.bincount(pattern, minlength=self.n_patterns)
        coeff = _coefficients(x, self.modality)
        self._mean = coeff.mean(axis=0)
        variance = coeff.var(axis=0)
        pbits = (self.n_patterns - 1).bit_length()
        self._flat = {mode: _fit_flat(coeff, self._mean, variance, budget) for mode, budget in
                      (('flat', pbits + self.detail_count * self.detail_bits), ('flat_pattern', pbits))}
        k, m = self.n_patterns, self.detail_count
        # Finite/global fallbacks for valid but never occupied IDs; encoder
        # does not invent training examples or revive codes on held-out data.
        self._prototype = np.tile(self._mean, (k, 1))
        self._index = np.tile(np.argsort(-variance, kind='stable')[:m], (k, 1))
        self._radius = np.zeros((k, m))
        self.pattern_centers_ = np.tile(physical.mean(axis=0), (k, 1))
        for p in np.flatnonzero(self.pattern_counts_):
            member = pattern == p
            group = coeff[member]
            self.pattern_centers_[p] = physical[member].mean(axis=0)
            self._prototype[p] = group.mean(axis=0)
            residual = group - self._prototype[p]
            index = np.argsort(-np.mean(residual ** 2, axis=0), kind='stable')[:m]
            self._index[p] = index
            self._radius[p] = _fit_radii(residual[:, index], self.detail_bits)
        self._fitted = True
        return self

    def _mode(self, mode):
        if mode not in _MODES:
            raise ValueError(f'mode must be one of {_MODES}')
        if not self._fitted:
            raise RuntimeError('fit on training data first')

    def _widths(self, mode):
        if mode == 'prototype':
            return np.empty(0, dtype=np.int32)
        if mode == 'full':
            return np.full(self.detail_count, self.detail_bits, dtype=np.int32)
        return self._flat[mode].widths

    def rate(self, mode='full'):
        """Exact fixed payload, excluding frozen dictionaries and framing."""
        self._mode(mode)
        pbits = (self.n_patterns - 1).bit_length() if mode in ('prototype', 'full') else 0
        widths = tuple(map(int, self._widths(mode)))
        detail = sum(widths)
        return dict(pattern_bits=pbits, detail_bits=detail, detail_widths=widths,
                    bits_per_item=pbits + detail, bits_per_second=(pbits + detail) / 2)

    def encode(self, x, mode='full'):
        """Return just integer fields. No wave, float scale or phase is saved."""
        self._mode(mode)
        x = _wave(x, self.modality)
        if mode in ('full', 'prototype'):
            p = self._assign(self._route_features(x))
            if mode == 'prototype':
                return {'pattern': p}
            i = self._index[p]
            residual = np.take_along_axis(_coefficients(x, self.modality), i, axis=1) - self._prototype[p[:, None], i]
            return {'pattern': p, 'detail': _quantize(residual, self._radius[p], self.detail_bits)}
        f = self._flat[mode]
        residual = _coefficients(x, self.modality)[:, f.index] - self._mean[f.index]
        return {'detail': _quantize(residual, f.radius, f.widths)}

    def _fields(self, codes, mode):
        self._mode(mode)
        expected = {'prototype': {'pattern'}, 'full': {'pattern', 'detail'},
                    'flat': {'detail'}, 'flat_pattern': {'detail'}}[mode]
        if not isinstance(codes, Mapping) or set(codes) != expected:
            raise ValueError(f'{mode} requires exactly the fields {sorted(expected)}')
        fields = {name: np.asarray(codes[name]) for name in expected}
        if any(v.dtype.kind not in 'iu' for v in fields.values()):
            raise TypeError('decoder accepts only integer symbol arrays')
        p, d = fields.get('pattern'), fields.get('detail')
        if p is not None and (p.ndim != 1 or np.any(p < 0) or np.any(p >= self.n_patterns)):
            raise ValueError('invalid pattern shape or ID')
        if d is not None:
            widths = self._widths(mode)
            if (d.ndim != 2 or d.shape[1] != len(widths) or (p is not None and len(p) != len(d))
                    or np.any(d < 0) or np.any(d >= np.left_shift(1, widths))):
                raise ValueError('invalid detail shape or integer outside its bit field')
        return p, d

    def decode(self, codes, mode='full'):
        """Decode using only the supplied bounded integers and frozen state."""
        p, d = self._fields(codes, mode)
        if p is not None:
            coeff = self._prototype[p].copy()
            if d is not None:
                coeff[np.arange(len(p))[:, None], self._index[p]] += _dequantize(d, self._radius[p], self.detail_bits)
        else:
            f = self._flat[mode]
            coeff = np.tile(self._mean, (len(d), 1))
            coeff[:, f.index] += _dequantize(d, f.radius, f.widths)
        return _synthesize(coeff, self.modality)

    def readout(self, codes, mode='full'):
        """Identical observed-property features of integer reconstructions in every arm."""
        return self.observation_features(self.decode(codes, mode))

    def pack(self, codes, mode='full'):
        """Pack the charged bits MSB first in item/field order, zero pad the tail."""
        p, d = self._fields(codes, mode)
        widths = self._widths(mode)
        n = len(d) if p is None else len(p)
        rate = self.rate(mode)
        bits = np.empty((n, rate['bits_per_item']), dtype=np.uint8)
        offset = 0
        fields = [] if p is None else [(p, rate['pattern_bits'])]
        if d is not None:
            fields.extend(zip(d.T, widths))
        for symbol, width in fields:
            for shift in range(width - 1, -1, -1):
                bits[:, offset] = (symbol >> shift) & 1
                offset += 1
        return np.packbits(bits.ravel(), bitorder='big').tobytes()

    def unpack(self, payload, n_items, mode='full'):
        """Invert pack with external item count; reject invalid or noncanonical data."""
        rate = self.rate(mode)
        n = _integer(n_items, 'n_items', 0, np.iinfo(np.intp).max)
        if not isinstance(payload, bytes):
            raise TypeError('expected packed bytes')
        total = n * rate['bits_per_item']
        if len(payload) != (total + 7) // 8:
            raise ValueError('wrong packed byte length')
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder='big')
        if bits[total:].any():
            raise ValueError('nonzero trailing padding')
        bits = bits[:total].reshape(n, rate['bits_per_item'])
        offset = 0
        def field(width):
            nonlocal offset
            result = np.zeros(n, dtype=np.int32)
            for _ in range(width):
                result = (result << 1) | bits[:, offset]
                offset += 1
            return result
        codes = {}
        if mode in ('prototype', 'full'):
            codes['pattern'] = field(rate['pattern_bits'])
        if mode != 'prototype':
            widths = rate['detail_widths']
            d = np.empty((n, len(widths)), dtype=np.int32)
            for j, width in enumerate(widths):
                d[:, j] = field(width)
            codes['detail'] = d
        self._fields(codes, mode)  # e.g. reject unused bit patterns for non-power-of-two K
        return codes
