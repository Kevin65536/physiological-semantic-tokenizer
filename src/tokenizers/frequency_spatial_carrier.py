"""Compact continuous Fourier-spatial coordinates assigned to fixed physical slots."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

FREQUENCY_INDICES = np.flatnonzero(
    (np.fft.rfftfreq(400, 1 / 200) >= 1) & (np.fft.rfftfreq(400, 1 / 200) < 45)
)


def fit_modes(covariance, coordinates, *, frequency_indices=None):
    """Choose exactly C*m train-energy modes and assign m to each physical site."""
    covariance = np.asarray(covariance, np.complex128)
    indices = (
        FREQUENCY_INDICES
        if frequency_indices is None
        else np.asarray(frequency_indices, dtype=np.int64)
    )
    if (
        covariance.ndim != 3
        or covariance.shape[0] != len(indices)
        or covariance.shape[1] != covariance.shape[2]
        or indices.ndim != 1
        or len(np.unique(indices)) != len(indices)
        or np.any((indices < 0) | (indices >= 201))
    ):
        raise ValueError(
            "one full-native Hermitian covariance per retained frequency required"
        )
    count = covariance.shape[1]
    if not isinstance(coordinates, int) or not 1 <= coordinates <= len(indices):
        raise ValueError(
            "positive fixed number of complex coordinates per physical slot required"
        )
    values, vectors = np.linalg.eigh(
        (covariance + covariance.conj().transpose(0, 2, 1)) * 0.5
    )
    values = values.clip(min=0)
    flat = np.arange(values.size)
    # Stable index tie-break makes repeated low-rank and zero eigenvalues reproducible.
    energy_weights = np.where(indices == 0, 1.0, 2.0)
    ordering_energy = values * energy_weights[:, None]
    rank_values = values if frequency_indices is None else ordering_energy
    order = np.lexsort((flat, -rank_values.ravel()))[: count * coordinates]
    frequency = order // count
    mode = order % count
    selected = np.stack([vectors[f, :, p] for f, p in zip(frequency, mode)])
    energy = values[frequency, mode]
    site_labels = np.repeat(np.arange(count), coordinates)
    affinity_energy = (
        energy if frequency_indices is None else energy * energy_weights[frequency]
    )
    affinity = affinity_energy[:, None] * np.abs(selected[:, site_labels]) ** 2
    rows, cols = linear_sum_assignment(-affinity)
    arrangement = np.empty(count * coordinates, np.int64)
    arrangement[cols] = rows
    return dict(
        frequency=indices[frequency[arrangement]].reshape(count, coordinates),
        vectors=selected[arrangement].reshape(count, coordinates, count),
        eigenvalues=energy[arrangement].reshape(count, coordinates),
        selected_mode_number=mode[arrangement].reshape(count, coordinates),
        training_total_energy=float(2 * values.sum())
        if frequency_indices is None
        else float(ordering_energy.sum()),
        training_selected_energy=float(2 * energy.sum())
        if frequency_indices is None
        else float((energy * energy_weights[frequency]).sum()),
    )


class FrequencySpatialCarrier:
    def __init__(self, metadata, scales, modes):
        self.metadata = metadata
        self.scales = scales
        self.modes = modes
        self.coordinates = int(metadata["complex_coordinates_per_slot"])
        permitted = metadata.get("local_frequency_indices", FREQUENCY_INDICES)
        for dataset, names in metadata["positions"].items():
            for branch in "SO":
                scale = np.asarray(scales[dataset, branch])
                basis = modes[dataset, branch]
                if (
                    scale.shape != (len(names),)
                    or not np.isfinite(scale).all()
                    or np.any(scale <= 0)
                    or basis["frequency"].shape != (len(names), self.coordinates)
                    or basis["vectors"].shape
                    != (len(names), self.coordinates, len(names))
                    or not np.isin(basis["frequency"], permitted).all()
                ):
                    raise ValueError(
                        "fixed native units and in-support frequency modes required"
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
                "30s native montage and observation-only availability required"
            )
        result = {}
        for branch in "SO":
            waves = np.asarray(prediction[branch], np.float64)
            if not np.isfinite(waves).all():
                raise ValueError("finite observation-derived carriers required")
            waves = np.where(observed[..., None, None], waves, 0)
            spectra = (
                np.fft.rfft(waves, axis=-1)
                / 400
                / self.scales[dataset, branch][None, :, None, None]
            )
            basis = self.modes[dataset, branch]
            latent = np.empty((*shape[:3], self.coordinates), np.complex128)
            for site in range(count):
                for coordinate in range(self.coordinates):
                    f = int(basis["frequency"][site, coordinate])
                    v = basis["vectors"][site, coordinate]
                    latent[:, site, :, coordinate] = np.einsum(
                        "bpq,p->bq", spectra[:, :, :, f], v.conj()
                    )
            result[branch] = np.where(observed[..., None, None], latent, 0)
        return result

    def synthesize(self, coordinates, dataset):
        count = len(self.metadata["positions"][dataset])
        shape = np.shape(coordinates["S"])
        if (
            len(shape) != 4
            or shape[1:] != (count, 15, self.coordinates)
            or np.shape(coordinates["O"]) != shape
        ):
            raise ValueError(
                "same declared complex coordinates at every physical slot required"
            )
        result = {}
        frequency = np.fft.rfftfreq(6000, 1 / 200)
        keep = (frequency >= 1) & (frequency < 45)
        for branch in "SO":
            value = np.asarray(coordinates[branch], np.complex128)
            if not np.isfinite(value).all():
                raise ValueError("finite continuous diagnostic coordinates required")
            spectrum = np.zeros((shape[0], count, 15, 201), np.complex128)
            basis = self.modes[dataset, branch]
            for site in range(count):
                for coordinate in range(self.coordinates):
                    f = int(basis["frequency"][site, coordinate])
                    v = basis["vectors"][site, coordinate]
                    spectrum[:, :, :, f] += np.einsum(
                        "bq,p->bpq", value[:, site, :, coordinate], v
                    )
            waves = np.fft.irfft(
                spectrum * 400 * self.scales[dataset, branch][None, :, None, None],
                n=400,
                axis=-1,
            )
            full = waves.reshape(shape[0], count, 6000)
            full = np.fft.irfft(np.fft.rfft(full, axis=-1) * keep, n=6000, axis=-1)
            result[branch] = full.reshape(shape[0], count, 15, 400)
        result["sum"] = result["S"] + result["O"]
        return result

    def save(self, path):
        fields = {"metadata": np.asarray(json.dumps(self.metadata))}
        for (dataset, branch), values in self.modes.items():
            fields[f"scale_{dataset}_{branch}"] = self.scales[dataset, branch]
            for name, value in values.items():
                fields[f"{name}_{dataset}_{branch}"] = np.asarray(value)
        destination = Path(path)
        temporary = destination.with_suffix(".pending.npz")
        np.savez_compressed(temporary, **fields)
        temporary.replace(destination)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as fields:
            metadata = json.loads(str(fields["metadata"]))
            if metadata["schema"] not in [
                "frequency_spatial_carrier_v1",
                "frequency_spatial_carrier_v2",
            ]:
                raise ValueError("versioned compact frequency-spatial carrier required")
            scales = {}
            modes = {}
            for dataset in metadata["positions"]:
                for branch in "SO":
                    scales[dataset, branch] = fields[f"scale_{dataset}_{branch}"]
                    modes[dataset, branch] = {
                        key: fields[f"{key}_{dataset}_{branch}"]
                        for key in [
                            "frequency",
                            "vectors",
                            "eigenvalues",
                            "selected_mode_number",
                            "training_total_energy",
                            "training_selected_energy",
                        ]
                    }
        return cls(metadata, scales, modes)
