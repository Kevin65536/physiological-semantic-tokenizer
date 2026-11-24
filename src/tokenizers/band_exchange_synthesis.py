"""Frozen paid EEG words followed by train-only stationary complex synthesis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .band_exchange_words import load_model as load_base


FREQUENCY_GROUPS = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))


def frequency_indices(samples):
    frequency = np.fft.rfftfreq(samples, 1 / 200)
    return [
        np.flatnonzero((frequency >= low) & ((frequency <= high) if high == 45 else (frequency < high)))
        for low, high in FREQUENCY_GROUPS
    ]


def fit_grouped_heads(gram, cross, output_group, sites, ridge_fraction=0.05):
    """Use identical penalties for all-montage and two-stream same-site controls.

    Gram is [band, mask-group, input, input]; cross is [band, output, input].
    Sharing a Gram is legal only for outputs with identical FIT target support
    and hierarchy weights. Both branches at one electrode are the local inputs.
    """
    gram = np.asarray(gram, np.complex128)
    cross = np.asarray(cross, np.complex128)
    groups = np.asarray(output_group, np.int64)
    channels = 2 * sites
    if (gram.ndim != 4 or gram.shape[0] != len(FREQUENCY_GROUPS)
            or gram.shape[2:] != (channels, channels)
            or cross.shape != (len(FREQUENCY_GROUPS), channels, channels)
            or groups.shape != (channels,) or np.any(groups < 0)
            or np.any(groups >= gram.shape[1]) or not 0 < ridge_fraction <= 1):
        raise ValueError("matched two-stream native grouped spectral moments required")
    full = np.zeros_like(cross)
    local = np.zeros_like(cross)
    for band in range(len(FREQUENCY_GROUPS)):
        for group in range(gram.shape[1]):
            members = np.flatnonzero(groups == group)
            if not len(members):
                continue
            matrix = (gram[band, group] + gram[band, group].conj().T) / 2
            diagonal = matrix.diagonal().real.clip(min=0)
            average = float(diagonal.mean())
            if average <= 1e-20:
                continue
            penalty = ridge_fraction * np.maximum(diagonal, 1e-6 * average)
            full[band, members] = np.linalg.solve(
                matrix + np.diag(penalty), cross[band, members].T
            ).T
            for output in members:
                site = output % sites
                inputs = np.asarray([site, sites + site])
                local[band, output, inputs] = np.linalg.solve(
                    matrix[np.ix_(inputs, inputs)] + np.diag(penalty[inputs]),
                    cross[band, output, inputs],
                )
    return {"montage": full, "same_site": local}


def apply_synthesis(tracks, coefficients):
    """Read only decoded S/O tracks; infer on the complete visible 30-s context."""
    s, o = (np.asarray(tracks[b], dtype=np.float64) for b in "SO")
    if s.ndim == 3:
        s, o = s[None], o[None]
        squeeze = True
    else:
        squeeze = False
    if s.ndim != 4 or s.shape != o.shape or s.shape[-2:] != (15, 400):
        raise ValueError("matching [batch,native-site,15,400] decoded EEG tracks required")
    sites = s.shape[1]
    if np.shape(coefficients) != (len(FREQUENCY_GROUPS), 2 * sites, 2 * sites):
        raise ValueError("frozen two-stream native synthesis matrix required")
    values = np.concatenate((s, o), axis=1).reshape(len(s), 2 * sites, 6000)
    spectrum = np.fft.rfft(values, axis=-1)
    predicted = np.zeros_like(spectrum)
    for indices, coefficient in zip(frequency_indices(6000), coefficients, strict=True):
        predicted[:, :, indices] = np.einsum("bif,oi->bof", spectrum[:, :, indices], coefficient, optimize=True)
    waveform = np.fft.irfft(predicted, n=6000, axis=-1).reshape(len(s), 2 * sites, 15, 400)
    result = dict(S=waveform[:, :sites], O=waveform[:, sites:])
    result["sum"] = result["S"] + result["O"]
    return {key: value[0] for key, value in result.items()} if squeeze else result


class BandExchangeSynthesis:
    """Portable original integer decoder plus one frozen all-montage synthesis."""

    def __init__(self, base, metadata, coefficients):
        if metadata["schema"] != "band_exchange_synthesis_v1":
            raise ValueError("versioned frozen integer synthesis required")
        if metadata["positions"] != base.metadata["positions"]:
            raise ValueError("base and synthesis native coordinates disagree")
        if len(metadata["positions"]) != 1:
            raise ValueError("one native dataset per fitted synthesis checkpoint required")
        sites = len(next(iter(metadata["positions"].values())))
        if (np.shape(coefficients) != (5, 2 * sites, 2 * sites)
                or not np.isfinite(coefficients).all()):
            raise ValueError("finite five-group two-stream synthesis coefficients required")
        self.base, self.metadata, self.coefficients = base, metadata, coefficients

    def encode(self, values, dataset, observed):
        return self.base.encode(values, dataset, observed)

    def decode_ids(self, ids, dataset):
        return apply_synthesis(self.base.decode_ids(ids, dataset), self.coefficients)

    def save(self, path, base_file):
        path, base_file = Path(path), Path(base_file)
        if path.parent.resolve() != base_file.parent.resolve():
            raise ValueError("portable head and frozen base must share one directory")
        metadata = dict(self.metadata, base_file=base_file.name,
                        base_sha256=hashlib.sha256(base_file.read_bytes()).hexdigest())
        with path.open("xb") as stream:
            np.savez_compressed(stream, metadata=np.asarray(json.dumps(metadata)), coefficients=self.coefficients)


def load_model(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        coefficients = archive["coefficients"].copy()
    base_file = path.parent / metadata["base_file"]
    if base_file.name != metadata["base_file"] or hashlib.sha256(base_file.read_bytes()).hexdigest() != metadata["base_sha256"]:
        raise ValueError("portable frozen base identity changed")
    return BandExchangeSynthesis(load_base(base_file), metadata, coefficients)
