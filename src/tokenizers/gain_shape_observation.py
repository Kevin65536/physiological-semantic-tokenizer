"""EEG gain/shape ablation at fixed B=256, 512 or 1024 bits per 2 s item.

Transmit the frozen original-waveform spectral ID (4 bits), observed sample
mean (8-bit level ID), AC RMS (6-bit gain ID), and B-18 signed-coefficient
shape bits. The latter use exactly the adaptive reference's DCT transform,
training-only variance allocation and scalar quantizer on zero-mean, unit-AC-
RMS waveforms. Both arms send the same level, gain and pattern symbols. Only
``conditional`` routes the shape dictionary by ID; ``independent`` pools it.

Level representatives are 255 evenly spaced empirical TRAIN quantiles plus
exact zero, sorted; encode by nearest level. Gain zero is reserved for zero
AC. The 63 positive representatives are log-uniform across positive TRAIN
AC RMS extrema; encode by nearest log gain. Values outside the fitted gains
or levels saturate; a positive gain below the smallest positive TRAIN gain
is not extrapolated. With no AC in training all gains decode to zero. Duplicate
representatives do NOT reduce the charge. No per-item floating-point statistic
is passed to the decoder. The bit-decoded shape is centered and normalized
before applying the *decoded* gain; a numerically zero decoded shape cannot
synthesize a carrier. Controlling total RMS does not guarantee band fidelity.

Pattern/level/gain store one integer each per row. Detail is a row-padded
uint8 array. ``pack`` emits the 4+8+6 header bits then B-18 detail bits per
row, MSB first, with NO row padding: precisely n*B/8 bytes. Shared codebooks,
arm, item count and frame metadata are out of band.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import numpy as np

from .adaptive_observation_detail import _Dictionary, _decode_detail, _encode_detail
from .factorized_observation import FactorizedObservationTokenizer, _coefficients, _integer, _synthesize, _wave


def _decompose(x):
    wave = x[:, 0]
    # Subtract the first value before averaging: an exactly constant window,
    # including a non-binary decimal constant, has *exactly* zero residual.
    delta = wave - wave[:, :1]
    offset = delta.mean(axis=1, keepdims=True)
    level = wave[:, 0] + offset[:, 0]
    ac = delta - offset
    peak = np.max(np.abs(ac), axis=1, keepdims=True)
    scaled = ac / np.where(peak > 0, peak, 1.)
    rms = np.sqrt(np.mean(scaled * scaled, axis=1, keepdims=True))
    gain = (peak * rms)[:, 0]
    unit = scaled / np.where(rms > 0, rms, 1.)
    return level, gain, unit[:, None, :]


def _nearest(v, representatives):
    mid = (representatives[:-1] + representatives[1:]) * .5
    return np.searchsorted(mid, v, side='left').astype(np.uint8)


class GainShapeObservation:
    """Gain/shape comparison for an already-fitted 16-way spectral EEG router."""

    def __init__(self, codec, total_bits):
        if not isinstance(codec, FactorizedObservationTokenizer):
            raise TypeError('expected FactorizedObservationTokenizer')
        if not codec._fitted or (codec.modality, codec.n_patterns, codec.routing) != ('eeg', 16, 'pattern'):
            raise ValueError('need a training-fitted EEG K=16 spectral router')
        self.total_bits = _integer(total_bits, 'total_bits', 256, 1024)
        if self.total_bits not in (256, 512, 1024):
            raise ValueError('total_bits must be 256, 512 or 1024')
        self.detail_bits = self.total_bits - 18
        self._codec = deepcopy(codec)
        self._global = None

    def fit(self, x, *, partition='train'):
        if partition != 'train':
            raise ValueError('fit requires partition=train')
        x = _wave(x, 'eeg')
        if not len(x):
            raise ValueError('need nonempty training data')
        # Crucial: assignment uses exactly the original frozen encoder, not a
        # reclustering or a second ID evaluation on a normalized waveform.
        p = self._codec.encode(x, mode='prototype')['pattern']
        level, gain, shape = _decompose(x)
        coeff = _coefficients(shape, 'eeg')
        pooled = _Dictionary.fit(coeff, self.detail_bits)
        counts = np.bincount(p, minlength=16)
        local = {int(i): pooled if counts[i] == len(x) else _Dictionary.fit(coeff[p == i], self.detail_bits)
                 for i in np.flatnonzero(counts)}
        self.level_range_ = (float(level.min()), float(level.max()))
        self.level_lut_ = np.sort(np.r_[0., np.quantile(level, np.linspace(0, 1, 255), method='linear')])
        pos = gain[gain > 0]
        self.positive_gain_range_ = (float(pos.min()), float(pos.max())) if len(pos) else None
        self.gain_lut_ = np.zeros(64)
        if len(pos):
            lo, hi = self.positive_gain_range_
            self.gain_lut_[1:] = np.clip(np.exp(np.linspace(np.log(lo), np.log(hi), 63)), lo, hi)
            self.gain_lut_[[1, -1]] = [lo, hi]
        self._global, self._local, self.pattern_counts_ = pooled, local, counts
        return self

    def _check(self, mode):
        if mode not in ('conditional', 'independent'):
            raise ValueError("mode must be 'conditional' or 'independent'")
        if self._global is None:
            raise RuntimeError('fit on training data first')

    def rate(self, mode='conditional'):
        self._check(mode)
        return dict(pattern_bits=4, level_bits=8, gain_bits=6, detail_bits=self.detail_bits,
                    bits_per_item=self.total_bits, bits_per_second=self.total_bits / 2)

    def detail_layout(self, pattern_id, mode='conditional'):
        self._check(mode)
        p = _integer(pattern_id, 'pattern_id', 0, 15)
        f = (self._local.get(p, self._global) if mode == 'conditional' else self._global).layout
        return dict(indices=tuple(map(int, f.index)), widths=tuple(map(int, f.widths)))

    def encode(self, x, mode='conditional'):
        self._check(mode)
        x = _wave(x, 'eeg')
        p = self._codec.encode(x, mode='prototype')['pattern']
        level, gain, shape = _decompose(x)
        level_id = _nearest(level, self.level_lut_)
        gain_id = np.zeros(len(x), dtype=np.uint8)
        if self.positive_gain_range_ is not None:
            pos = gain > 0
            gain_id[pos] = 1 + _nearest(np.log(gain[pos]), np.log(self.gain_lut_[1:]))
        coeff = _coefficients(shape, 'eeg')
        if mode == 'independent':
            detail = _encode_detail(coeff, self._global)
        else:
            detail = np.empty((len(x), (self.detail_bits + 7) // 8), dtype=np.uint8)
            for i in np.unique(p):
                rows = p == i
                detail[rows] = _encode_detail(coeff[rows], self._local.get(i, self._global))
        return dict(pattern=p, level=level_id, gain=gain_id, detail=detail)

    def _fields(self, codes):
        names = ('pattern', 'level', 'gain', 'detail')
        if not isinstance(codes, Mapping) or set(codes) != set(names):
            raise ValueError(f'expected only the four fields {names}')
        p, level, gain, d = (np.asarray(codes[name]) for name in names)
        for field, limit in ((p, 16), (level, 256), (gain, 64)):
            if field.dtype.kind not in 'iu':
                raise TypeError('header accepts only integer IDs')
            if field.ndim != 1 or field.shape != p.shape or np.any(field < 0) or np.any(field >= limit):
                raise ValueError('invalid header ID or shape')
        if d.dtype != np.uint8:
            raise TypeError('detail must be packed uint8 rows')
        if d.shape != (len(p), (self.detail_bits + 7) // 8) or np.any(d[:, -1] & 3):
            raise ValueError('wrong detail shape or nonzero two-bit row padding')
        return p, level, gain, d

    def decode(self, codes, mode='conditional'):
        self._check(mode)
        p, level, gain, d = self._fields(codes)
        if mode == 'independent':
            coeff = _decode_detail(d, self._global)
        else:
            coeff = np.empty((len(p), 400))
            for i in np.unique(p):
                rows = p == i
                coeff[rows] = _decode_detail(d[rows], self._local.get(i, self._global))
        _, ac_rms, shape = _decompose(_synthesize(coeff, 'eeg'))
        # Avoid normalizing roundoff into a fabricated unit-amplitude carrier.
        shape[ac_rms <= 1e-12] = 0.
        return self.level_lut_[level, None, None] + self.gain_lut_[gain, None, None] * shape

    def readout(self, codes, mode='conditional'):
        return self._codec.observation_features(self.decode(codes, mode))

    def pack(self, codes, mode='conditional'):
        self._check(mode)
        p, level, gain, d = self._fields(codes)
        header = (p.astype(np.int32) << 14) | (level.astype(np.int32) << 6) | gain.astype(np.int32)
        rows = np.empty((len(p), self.total_bits), dtype=np.uint8)
        rows[:, :18] = (header[:, None] >> np.arange(17, -1, -1)) & 1
        rows[:, 18:] = np.unpackbits(d, axis=1, count=self.detail_bits, bitorder='big')
        return np.packbits(rows, axis=1, bitorder='big').tobytes()

    def unpack(self, payload, n_items, mode='conditional'):
        self._check(mode)
        n = _integer(n_items, 'n_items', 0, np.iinfo(np.intp).max)
        if not isinstance(payload, bytes):
            raise TypeError('expected packed bytes')
        if len(payload) != n * (self.total_bits // 8):
            raise ValueError('wrong packed byte length')
        rows = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder='big').reshape(n, self.total_bits)
        h = rows[:, :18] @ np.left_shift(1, np.arange(17, -1, -1, dtype=np.int32))
        return dict(pattern=h >> 14, level=((h >> 6) & 255).astype(np.uint8), gain=(h & 63).astype(np.uint8),
                    detail=np.packbits(rows[:, 18:], axis=1, bitorder='big'))
