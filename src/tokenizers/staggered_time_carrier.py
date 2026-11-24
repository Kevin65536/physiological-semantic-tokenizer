"""Continuous real time samples and stationary array LMMSE interpolation."""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def sample_layout(positions, samples, staggered):
    if samples not in (4, 16, 32) or positions < 1:
        raise ValueError("declared 4/16/32 real samples and native positions required")
    phase = (
        np.floor(np.arange(positions) * 400 / (samples * positions)).astype(int)
        if staggered
        else np.zeros(positions, int)
    )
    center = np.floor((np.arange(samples) + 0.5) * 400 / samples).astype(int)
    indices = np.sort((center[None] + phase[:, None]) % 400, axis=1)
    return indices, phase


def neighbor_layout(indices):
    """Two preceding and two following encoded samples, within the same 30s."""
    count, samples = indices.shape
    sites = np.repeat(np.arange(count), 4)
    offsets = np.empty((400, 4 * count), np.int64)
    patch = np.empty_like(offsets)
    slot = np.empty_like(offsets)
    for position in range(count):
        times = np.concatenate(
            [indices[position] - 400, indices[position], indices[position] + 400]
        )
        for phase in range(400):
            pivot = np.searchsorted(times, phase)
            chosen = np.arange(pivot - 2, pivot + 2)
            values = times[chosen]
            take = slice(position * 4, (position + 1) * 4)
            offsets[phase, take] = values - phase
            patch[phase, take] = np.floor_divide(values, 400)
            slot[phase, take] = chosen % samples
    return sites, offsets, patch, slot


def stationary_moments(wave, observed, maximum_lag=400):
    """E[target(t) observed(t+lag)] without window centering or side statistics."""
    wave = np.asarray(wave, float)
    observed = np.asarray(observed)
    if (
        wave.ndim != 2
        or wave.shape[1] != 6000
        or observed.shape != wave.shape[:1]
        or observed.dtype != bool
    ):
        raise ValueError(
            "finite full 30s native carrier and observation flags required"
        )
    if not np.isfinite(wave).all() or not 0 <= maximum_lag < 3000:
        raise ValueError("finite carrier and bounded lag support required")
    target = np.fft.rfft(wave, axis=-1)
    inputs = target * observed[:, None]
    lags = np.arange(-maximum_lag, maximum_lag + 1) % 6000

    def correlation(left, right):
        values = (
            np.fft.irfft(left[:, None].conj() * right[None], n=6000, axis=-1) / 6000
        )
        return values[:, :, lags].copy()

    return dict(
        observation=correlation(inputs, inputs),
        target_observation=correlation(target, inputs),
    )


def fit_interpolator(moments, indices, ridge_fraction=0.001):
    sites, offsets, patch, slot = neighbor_layout(indices)
    obs, cross = moments["observation"], moments["target_observation"]
    maximum_lag = (obs.shape[-1] - 1) // 2
    if np.max(offsets.max(1) - offsets.min(1)) > maximum_lag:
        raise ValueError("stationary covariance lacks declared temporal neighborhood")
    unique = {}
    operators = []
    assignment = []
    penalties = []
    for lag in offsets:
        key = lag.tobytes()
        if key not in unique:
            covariance = obs[
                sites[:, None], sites[None, :], lag[None] - lag[:, None] + maximum_lag
            ]
            covariance = (covariance + covariance.T) * 0.5
            penalty = ridge_fraction * max(
                float(np.trace(covariance) / len(covariance)), 1e-10
            )
            covariance.flat[:: len(covariance) + 1] += penalty
            target = cross[:, sites, lag + maximum_lag]
            operators.append(cho_solve(cho_factor(covariance, lower=True), target.T).T)
            unique[key] = len(operators) - 1
            penalties.append(penalty)
        assignment.append(unique[key])
    return dict(
        sites=sites,
        patch=patch,
        slot=slot,
        phase_to_operator=np.asarray(assignment),
        weights=np.stack(operators),
        maximum_lag=maximum_lag,
        ridge_penalties=np.asarray(penalties),
    )


class StaggeredTimeCarrier:
    def __init__(self, metadata, scales, indices, operators):
        self.metadata, self.scales, self.indices, self.operators = (
            metadata,
            scales,
            indices,
            operators,
        )
        self.coordinates = int(metadata["real_samples_per_slot"])
        for dataset, names in metadata["positions"].items():
            if np.shape(indices[dataset]) != (len(names), self.coordinates):
                raise ValueError("fixed physical position/sample map required")
            for branch in "SO":
                if np.shape(scales[dataset, branch]) != (len(names),) or np.any(
                    scales[dataset, branch] <= 0
                ):
                    raise ValueError("positive fixed training scales required")
                if operators[dataset, branch]["weights"].shape[1:] != (
                    len(names),
                    4 * len(names),
                ):
                    raise ValueError(
                        "fixed array/time interpolation dimensions required"
                    )

    def analyze(self, prediction, dataset, observed):
        count = len(self.metadata["positions"][dataset])
        observed = np.asarray(observed)
        shape = np.shape(prediction["S"])
        if (
            len(shape) != 4
            or shape[1:] != (count, 15, 400)
            or np.shape(prediction["O"]) != shape
            or observed.shape != shape[:2]
            or observed.dtype != bool
        ):
            raise ValueError(
                "matched full native 30s carriers and observation flags required"
            )
        result = {}
        for branch in "SO":
            wave = np.asarray(prediction[branch], float)
            if not np.isfinite(wave).all():
                raise ValueError("finite observation-derived carrier required")
            value = np.take_along_axis(
                wave, self.indices[dataset][None, :, None, :], axis=-1
            )
            result[branch] = np.where(
                observed[:, :, None, None],
                value / self.scales[dataset, branch][None, :, None, None],
                0.0,
            )
        return result

    def synthesize(self, coordinates, dataset):
        count = len(self.metadata["positions"][dataset])
        shape = np.shape(coordinates["S"])
        if (
            len(shape) != 4
            or shape[1:] != (count, 15, self.coordinates)
            or np.shape(coordinates["O"]) != shape
        ):
            raise ValueError("declared continuous local sample vectors required")
        result = {}
        for branch in "SO":
            values = np.asarray(coordinates[branch])
            if values.dtype.kind != "f" or not np.isfinite(values).all():
                raise ValueError(
                    "real floating sample vectors; no integer model is claimed"
                )
            op = self.operators[dataset, branch]
            output = np.empty((*shape[:3], 400))
            for phase in range(400):
                patch = (np.arange(15)[:, None] + op["patch"][phase][None]) % 15
                features = values[:, op["sites"][None], patch, op["slot"][phase][None]]
                output[..., phase] = np.einsum(
                    "cf,bpf->bcp",
                    op["weights"][op["phase_to_operator"][phase]],
                    features,
                    optimize=True,
                )
            result[branch] = output * self.scales[dataset, branch][None, :, None, None]
        result["sum"] = result["S"] + result["O"]
        return result

    def save(self, path):
        fields = dict(metadata=np.asarray(json.dumps(self.metadata)))
        for dataset in self.metadata["positions"]:
            fields["indices_" + dataset] = self.indices[dataset]
            for branch in "SO":
                fields["scale_" + dataset + "_" + branch] = self.scales[dataset, branch]
                for key, value in self.operators[dataset, branch].items():
                    fields[key + "_" + dataset + "_" + branch] = value
        file = Path(path)
        temporary = file.with_suffix(".pending.npz")
        np.savez_compressed(temporary, **fields)
        temporary.replace(file)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["metadata"]))
            if meta["schema"] != "staggered_time_carrier_v1":
                raise ValueError("versioned time-sample interpolation model required")
            scales = {}
            indices = {}
            operators = {}
            for dataset in meta["positions"]:
                indices[dataset] = f["indices_" + dataset]
                for branch in "SO":
                    scales[dataset, branch] = f["scale_" + dataset + "_" + branch]
                    operators[dataset, branch] = {
                        key: f[key + "_" + dataset + "_" + branch]
                        for key in [
                            "sites",
                            "patch",
                            "slot",
                            "phase_to_operator",
                            "weights",
                            "maximum_lag",
                            "ridge_penalties",
                        ]
                    }
        return cls(meta, scales, indices, operators)
