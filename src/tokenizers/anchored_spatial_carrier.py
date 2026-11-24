"""Continuous EEG spatial coordinates with a fixed physical anchor for each band slot."""

import json
from pathlib import Path

import numpy as np

from .spatial_waveform_dictionary import BANDS, band_filter


def fit_band(covariance, scales, sites, ridge_fraction):
    """Fit equal-rank virtual and physical-anchor reconstructions of a band.

    The covariance is for Wiener carriers in G0 target-SD coordinates.
    The returned operators act in the caller's original G0 coordinates.
    """
    sites = np.asarray(sites, dtype=int)
    count = len(scales)
    if not 0 < len(sites) <= count or np.any(scales <= 0):
        raise ValueError(
            "positive scales and at least one native site per band required"
        )
    covariance = (covariance + covariance.T) / 2
    _, eig = np.linalg.eigh(covariance)
    subspace = eig[:, -len(sites) :]
    left, _, right = np.linalg.svd(
        subspace.T @ covariance[:, sites], full_matrices=False
    )
    oriented = subspace @ (left @ right)
    latent_var = np.diag(oriented.T @ covariance @ oriented).clip(min=1e-18)
    slot_scale = (
        np.sqrt(covariance.diagonal()[sites].clip(min=1e-18) / latent_var)
        * scales[sites]
    )
    analysis = (oriented / scales[:, None]) * slot_scale[None, :]
    synthesis = (oriented.T * scales[None, :]) / slot_scale[:, None]
    # Same carriers, band ranks and training covariance, using physical electrodes only.
    gram = covariance[np.ix_(sites, sites)]
    ridge = max(1e-12, ridge_fraction * np.trace(gram) / len(sites))
    physical = np.linalg.solve(gram + ridge * np.eye(len(sites)), covariance[sites])
    physical = (physical * scales[None, :]) / scales[sites, None]
    return analysis, synthesis, physical


class AnchoredSpatialCarrier:
    def __init__(self, metadata, transfer, analysis, synthesis):
        self.metadata = metadata
        self.transfer = transfer
        self.analysis = analysis
        self.synthesis = synthesis

    def roles(self, dataset):
        return np.arange(len(self.metadata["positions"][dataset])) % len(BANDS)

    def continuous_source(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 4 or x.shape[-2:] != (15, 400) or not np.isfinite(x).all():
            raise ValueError("finite [batch,native site,15,400] observed EEG required")
        spectrum = np.fft.rfft(x.reshape(*x.shape[:2], 6000), axis=-1)
        out = {
            b: np.fft.irfft(
                spectrum * self.transfer[b].reshape(-1), n=6000, axis=-1
            ).reshape(x.shape)
            for b in ("S", "O")
        }
        out["sum"] = out["S"] + out["O"]
        return out

    def analyze(self, prediction, dataset, observed):
        roles = self.roles(dataset)
        observed = np.asarray(observed)
        shape = np.shape(prediction["S"])
        if (
            len(shape) != 4
            or shape[1:] != (len(roles), 15, 400)
            or observed.dtype != bool
            or observed.shape != shape[:2]
            or np.shape(prediction["O"]) != shape
        ):
            raise ValueError("native montage layout and observation mask required")
        out = {}
        for b in ("S", "O"):
            flat = np.asarray(prediction[b]).reshape(*shape[:2], 6000)
            flat = np.where(observed[..., None], flat, 0.0)
            slots = np.zeros_like(flat)
            for k in range(len(BANDS)):
                band = band_filter(flat, k)
                sites = np.flatnonzero(roles == k)
                slots[:, sites] = self.analysis[dataset, b, k].T @ band
            out[b] = np.where(observed[..., None], slots, 0.0).reshape(shape)
        return out

    def synthesize(self, carriers, dataset):
        roles = self.roles(dataset)
        shape = np.shape(carriers["S"])
        if (
            len(shape) != 4
            or shape[1:] != (len(roles), 15, 400)
            or np.shape(carriers["O"]) != shape
        ):
            raise ValueError("matching native-layout carrier slots required")
        out = {}
        for b in ("S", "O"):
            flat = np.asarray(carriers[b]).reshape(*shape[:2], 6000)
            reconstructed = np.zeros(flat.shape, dtype=np.float64)
            for k in range(len(BANDS)):
                band = band_filter(flat[:, roles == k], k)
                reconstructed += self.synthesis[dataset, b, k].T @ band
            out[b] = reconstructed.reshape(shape)
        out["sum"] = out["S"] + out["O"]
        return out

    def save(self, path):
        arrays = {"metadata": np.asarray(json.dumps(self.metadata))}
        for b, v in self.transfer.items():
            arrays["transfer_" + b] = v
        for kind, values in [
            ("analysis", self.analysis),
            ("synthesis", self.synthesis),
        ]:
            for (d, b, k), v in values.items():
                arrays[f"{kind}_{d}_{b}_{k}"] = v
        path = Path(path)
        temporary = path.with_suffix(".pending.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["metadata"]))
            if meta["schema"] != "anchored_spatial_carrier_v1" or meta["bands_hz"] != [
                list(x) for x in BANDS
            ]:
                raise ValueError("frozen anchored spatial carrier required")
            op = [
                {
                    (d, b, k): f[f"{name}_{d}_{b}_{k}"]
                    for d in meta["positions"]
                    for b in "SO"
                    for k in range(len(BANDS))
                }
                for name in ["analysis", "synthesis"]
            ]
            return cls(meta, {b: f["transfer_" + b] for b in "SO"}, *op)
