"""Matched-allocation conditional and attached-ID controls for 2 s observations.

Take a snapshot of a TRAIN-fitted ``FactorizedObservationTokenizer`` and keep
its routing frozen. Fit both reconstruction dictionaries on the same supplied
training rows (normally the router's own training panel). Given K possible
IDs and B bits per window, both arms transmit p=(K-1).bit_length() bits for
the identical ID and R=B-p bits of detail. They use exactly the reference
flat codec's transform, bit allocator and scalar quantizer; only the training
pool for the reconstruction dictionary changes:

* ``conditional``: the ID chooses a training conditional mean waveform,
  coefficient support, integer widths and ranges. Unoccupied training IDs
  fall back to the fitted global dictionary; see ``pattern_counts_``.
* ``independent``: one global training dictionary receives R bits. The ID is
  charged and available to a separate reader, but has *no effect* on the
  detail encoder or waveform decoder. The ordinary reference ``flat`` arm at
  B bits remains a different, no-ID competitor.

``encode`` returns ONLY ``pattern`` (integer [n]) and ``detail`` (uint8
[n,ceil(R/8)]). Detail rows are MSB-first concatenations of unsigned fields
in fitted coefficient order, zero-padded at the low end of the last byte.
This row-aligned in-memory representation is not the transmission rate.
``pack`` removes that alignment, writing the ID followed by R actual bits for
every item: ceil(n*B/8) bytes, with at most seven zero bits at the very end.
The model, arm, item count, normalization and framing are pre-shared, not
additional per-window measurements. A route shuffle at fixed detail bytes
is a decoder-dependence test, not standalone evidence for semantics.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .factorized_observation import (
    FactorizedObservationTokenizer, _Flat, _MAX_WIDTH, _coefficients,
    _dequantize, _fit_flat, _integer, _quantize, _synthesize, _wave,
)


@dataclass
class _Dictionary:
    mean: np.ndarray
    layout: _Flat

    @classmethod
    def fit(cls, coeff, budget):
        mean = coeff.mean(axis=0)
        return cls(mean, _fit_flat(coeff, mean, coeff.var(axis=0), budget))


def _bit_columns(widths):
    """Field index and within-field right shift for each chronological bit."""
    ends = np.cumsum(widths, dtype=np.int32)
    fields = np.repeat(np.arange(len(widths), dtype=np.int32), widths)
    shifts = ends[fields] - 1 - np.arange(len(fields), dtype=np.int32)
    return fields, shifts, ends - widths


def _detail_bytes(symbols, widths):
    fields, shifts, _ = _bit_columns(widths)
    out = np.empty((len(symbols), (len(fields) + 7) // 8), dtype=np.uint8)
    if len(fields):
        step = max(1, 1_000_000 // len(fields))
        for i in range(0, len(out), step):
            # Slice rows first, then gather columns. Never build n*B symbols
            # for an entire large panel at once.
            bits = (symbols[i:i + step][:, fields] >> shifts) & 1
            out[i:i + step] = np.packbits(bits.astype(np.uint8), axis=1, bitorder='big')
    return out


def _detail_symbols(data, widths):
    _, shifts, starts = _bit_columns(widths)
    out = np.empty((len(data), len(widths)), dtype=np.int32)
    if len(shifts):
        weights = np.left_shift(1, shifts)
        step = max(1, 1_000_000 // len(shifts))
        for i in range(0, len(out), step):
            bits = np.unpackbits(data[i:i + step], axis=1, count=len(shifts), bitorder='big')
            out[i:i + step] = np.add.reduceat(bits * weights, starts, axis=1, dtype=np.int32)
    return out


def _encode_detail(coeff, dictionary):
    f = dictionary.layout
    symbol = _quantize(coeff[:, f.index] - dictionary.mean[f.index], f.radius, f.widths)
    return _detail_bytes(symbol, f.widths)


def _decode_detail(data, dictionary):
    f = dictionary.layout
    symbol = _detail_symbols(data, f.widths)
    coeff = np.tile(dictionary.mean, (len(data), 1))
    coeff[:, f.index] += _dequantize(symbol, f.radius, f.widths)
    return coeff


class AdaptiveObservationDetail:
    """Reference allocator on either ID-local or pooled DCT residuals.

    ``total_bits`` is the whole payload, not a residual-only allowance.
    At most 24 detail bits per available coefficient: for K=16, B up to
    9604 for EEG, 964 for Hb. B=p (no detail), and K=1 are supported.
    """

    def __init__(self, codec, total_bits):
        if not isinstance(codec, FactorizedObservationTokenizer):
            raise TypeError('codec must be a fitted FactorizedObservationTokenizer')
        if not codec._fitted:
            raise ValueError('fit the base router on training data first')
        self.pattern_bits = (codec.n_patterns - 1).bit_length()
        self.total_bits = _integer(total_bits, 'total_bits', self.pattern_bits,
                                   self.pattern_bits + len(codec._mean) * _MAX_WIDTH)
        self.detail_bits = self.total_bits - self.pattern_bits
        self._nbytes = (self.detail_bits + 7) // 8
        # Subsequent refits of the caller's router cannot silently change IDs.
        self._codec = deepcopy(codec)
        self._global = None

    def fit(self, x, *, partition='train'):
        """Train dictionaries on the same rows; never update the base routing."""
        if partition != 'train':
            raise ValueError('detail dictionary fitting requires partition=train')
        x = _wave(x, self._codec.modality)
        if not len(x):
            raise ValueError('need nonempty training data')
        pattern = self._codec.encode(x, mode='prototype')['pattern']
        coeff = _coefficients(x, self._codec.modality)
        pooled = _Dictionary.fit(coeff, self.detail_bits)
        counts = np.bincount(pattern, minlength=self._codec.n_patterns)
        local = {}
        for p in np.flatnonzero(counts):
            local[int(p)] = pooled if counts[p] == len(x) else _Dictionary.fit(coeff[pattern == p], self.detail_bits)
        self._global, self._local, self.pattern_counts_ = pooled, local, counts
        return self

    def _check(self, mode):
        if mode not in ('conditional', 'independent'):
            raise ValueError("mode must be 'conditional' or 'independent'")
        if self._global is None:
            raise RuntimeError('fit on training data first')

    def rate(self, mode='conditional'):
        """Hard bits per 2 s item; model and item framing are out-of-band."""
        self._check(mode)
        return dict(pattern_bits=self.pattern_bits, detail_bits=self.detail_bits,
                    bits_per_item=self.total_bits, bits_per_second=self.total_bits / 2)

    def detail_layout(self, pattern_id, mode='conditional'):
        """Return pre-shared coefficient indices and widths for one valid ID."""
        self._check(mode)
        p = _integer(pattern_id, 'pattern_id', 0, self._codec.n_patterns - 1)
        dictionary = self._local.get(p, self._global) if mode == 'conditional' else self._global
        return dict(indices=tuple(map(int, dictionary.layout.index)),
                    widths=tuple(map(int, dictionary.layout.widths)))

    def encode(self, x, mode='conditional'):
        self._check(mode)
        x = _wave(x, self._codec.modality)
        p = self._codec.encode(x, mode='prototype')['pattern']
        coeff = _coefficients(x, self._codec.modality)
        if mode == 'independent':
            detail = _encode_detail(coeff, self._global)  # No p here.
        else:
            detail = np.empty((len(p), self._nbytes), dtype=np.uint8)
            for value in np.unique(p):
                rows = p == value
                detail[rows] = _encode_detail(coeff[rows], self._local.get(value, self._global))
        return {'pattern': p, 'detail': detail}

    def _fields(self, codes):
        if not isinstance(codes, Mapping) or set(codes) != {'pattern', 'detail'}:
            raise ValueError('expected exactly the integer fields pattern and detail')
        p, d = np.asarray(codes['pattern']), np.asarray(codes['detail'])
        if p.dtype.kind not in 'iu' or d.dtype != np.uint8:
            raise TypeError('expected integer pattern IDs and packed uint8 detail rows')
        if p.ndim != 1 or np.any(p < 0) or np.any(p >= self._codec.n_patterns):
            raise ValueError('invalid pattern shape or ID')
        if d.shape != (len(p), self._nbytes):
            raise ValueError('wrong packed detail shape')
        pad = -self.detail_bits % 8
        if pad and np.any(d[:, -1] & ((1 << pad) - 1)):
            raise ValueError('nonzero row padding')
        return p, d

    def decode(self, codes, mode='conditional'):
        self._check(mode)
        p, d = self._fields(codes)
        if mode == 'independent':
            coeff = _decode_detail(d, self._global)  # p is validated, never used.
        else:
            coeff = np.empty((len(p), len(self._global.mean)), dtype=np.float64)
            for value in np.unique(p):
                rows = p == value
                coeff[rows] = _decode_detail(d[rows], self._local.get(value, self._global))
        return _synthesize(coeff, self._codec.modality)

    def readout(self, codes, mode='conditional'):
        """The same frozen-codec observed properties on every reconstruction."""
        return self._codec.observation_features(self.decode(codes, mode))

    def pack(self, codes, mode='conditional'):
        """Pack fixed B bits per item, no row padding, only final byte padding."""
        self._check(mode)
        p, d = self._fields(codes)
        rows = np.empty((len(p), self.total_bits), dtype=np.uint8)
        shifts = np.arange(self.pattern_bits - 1, -1, -1, dtype=np.int32)
        rows[:, :self.pattern_bits] = (p.astype(np.int32)[:, None] >> shifts) & 1
        rows[:, self.pattern_bits:] = np.unpackbits(d, axis=1, count=self.detail_bits, bitorder='big')
        return np.packbits(rows.ravel(), bitorder='big').tobytes()

    def unpack(self, payload, n_items, mode='conditional'):
        """Unpack externally framed items; reject reserved IDs/extra bytes/padding."""
        self._check(mode)
        n = _integer(n_items, 'n_items', 0, np.iinfo(np.intp).max)
        if not isinstance(payload, bytes):
            raise TypeError('expected packed bytes')
        total = n * self.total_bits
        if len(payload) != (total + 7) // 8:
            raise ValueError('wrong packed byte length')
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder='big')
        if bits[total:].any():
            raise ValueError('nonzero trailing padding')
        rows = bits[:total].reshape(n, self.total_bits)
        weights = np.left_shift(1, np.arange(self.pattern_bits - 1, -1, -1, dtype=np.int32))
        p = np.sum(rows[:, :self.pattern_bits] * weights, axis=1, dtype=np.int32)
        d = np.packbits(rows[:, self.pattern_bits:], axis=1, bitorder='big')
        codes = {'pattern': p, 'detail': d}
        self._fields(codes)
        return codes
