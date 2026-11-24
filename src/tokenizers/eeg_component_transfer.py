"""Frozen frequency-band residual regression for voltage components on a native EEG montage."""

import numpy as np

from .spatial_waveform_dictionary import BANDS


def frequency_groups(samples=6000):
    frequencies = np.fft.rfftfreq(samples, 1 / 200)
    return [
        np.flatnonzero((frequencies >= lo) & (frequencies < hi)) for lo, hi in BANDS
    ]


def component_residual(s_target, o_target, s_prior, o_prior, variances):
    """Minimum weighted S/O squared-error transfer with unchanged observed-sum estimate."""
    v_s, v_o = np.asarray(variances)
    if np.any(v_s <= 0) or np.any(v_o <= 0):
        raise ValueError("positive training component variances required")
    return ((s_target - s_prior) * v_o + (o_prior - o_target) * v_s) / (v_o + v_s)


def transfer_coefficients(gram, cross, ridge_fraction):
    """Fit equal-penalty regularized local and full-montage models for each output."""
    if gram.ndim != 4 or cross.shape != gram.shape[:-1] or not 0 < ridge_fraction <= 1:
        raise ValueError(
            "expected band, output, native input dimensions and positive ridge"
        )
    local = np.zeros_like(cross)
    full = np.zeros_like(cross)
    for band in range(len(gram)):
        for site, matrix in enumerate(gram[band]):
            diag = matrix.diagonal().real.clip(min=0)
            scale = diag.mean()
            if scale <= 0:
                continue
            penalty = ridge_fraction * np.maximum(diag, 1e-6 * scale)
            target = cross[band, site]
            full[band, site] = np.linalg.solve(matrix + np.diag(penalty), target)
            local[band, site, site] = target[site] / (diag[site] + penalty[site])
    return local, full


def transfer_waveform(spectrum, coefficients):
    """Use only the input spectrum; each output site is a frozen combination of native inputs."""
    if (
        spectrum.ndim != 2
        or spectrum.shape[0] != 3001
        or coefficients.shape != (len(BANDS), spectrum.shape[1], spectrum.shape[1])
    ):
        raise ValueError(
            "one 30-second native montage and its own fitted coefficients required"
        )
    correction = np.zeros_like(spectrum)
    for indices, projection in zip(frequency_groups(), coefficients, strict=True):
        correction[indices] = spectrum[indices] @ projection.T
    return np.fft.irfft(correction, n=6000, axis=0)


class IntegerSumSourceDecoder:
    """Allocate a frozen spatial decoder's total voltage using only its two transmitted integer streams."""

    def __init__(self, decoder, estimator, *, legacy=False, strength=1.0):
        if not 0 < strength <= 1:
            raise ValueError("positive convex allocation weight required")
        self.decoder = decoder
        self.metadata = dict(
            decoder.metadata,
            schema="integer_sum_source_decoder_v1",
            positions=estimator.metadata["positions"],
        )
        self.estimator = estimator
        self.legacy = legacy
        self.strength = strength

    def decode_ids(self, ids, dataset):
        if set(ids) != {"S", "O"} or any(
            np.asarray(v).dtype.kind not in "iu" for v in ids.values()
        ):
            raise ValueError("two integer streams required")
        wave = self.decoder.decode_ids(ids, dataset)
        return self.allocate(wave, ids, dataset)

    def allocate(self, wave, ids, dataset):
        seen = np.asarray(ids["S"]).any(axis=-1)
        if not np.array_equal(seen, np.asarray(ids["O"]).any(axis=-1)):
            raise ValueError("missing observation slots must agree between streams")
        if (
            self.decoder.metadata["positions"][dataset]
            != self.estimator.metadata["positions"][dataset]
        ):
            raise ValueError("mismatched acquisition layout")
        total = wave["sum"]
        available = seen[..., None, None]
        if self.legacy:
            observed_wave = np.where(available, total, 0)
            spectrum = np.fft.rfft(observed_wave.reshape(*seen.shape, 6000), axis=-1)
            source = np.fft.irfft(
                spectrum * np.asarray(self.estimator.transfer["S"]).reshape(1, 1, 3001),
                n=6000,
                axis=-1,
            ).reshape(total.shape)
        else:
            source = self.estimator.predict_continuous(total, dataset, seen)["S"]
        s = np.where(
            available, wave["S"] + self.strength * (source - wave["S"]), wave["S"]
        )
        return {"sum": total, "S": s, "O": total - s}


def load_integer_decoder(path):
    import hashlib
    import json
    from pathlib import Path
    from .eeg_transient_gain import load_model as load_source
    from .spatial_waveform_dictionary import load_model as load_spatial

    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
    if metadata["schema"] != "integer_sum_source_decoder_v1":
        raise ValueError("expected a frozen integer-sum allocation manifest")
    files = {}
    for key in ("integer_decoder", "component_estimator"):
        files[key] = (path.parent / metadata[key]["path"]).resolve()
        with files[key].open("rb") as stream:
            if (
                hashlib.file_digest(stream, "sha256").hexdigest()
                != metadata[key]["sha256"]
            ):
                raise ValueError(f"frozen {key} changed")
    decoder = load_spatial(files["integer_decoder"])
    estimator = load_source(files["component_estimator"])
    for key in ("input_identity", "scales_identity"):
        if decoder.metadata[key] != estimator.metadata[key]:
            raise ValueError(f"component source and integer decoder disagree on {key}")
    if any(
        decoder.metadata["positions"].get(d) != names
        for d, names in estimator.metadata["positions"].items()
    ):
        raise ValueError("component source and integer decoder disagree on positions")
    return IntegerSumSourceDecoder(
        decoder, estimator, legacy=metadata["legacy"], strength=metadata["strength"]
    )
