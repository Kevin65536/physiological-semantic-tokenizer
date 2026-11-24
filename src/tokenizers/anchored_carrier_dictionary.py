"""One integer per native EEG slot from train-fitted spatial carrier dictionaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .spatial_waveform_dictionary import BANDS, spectral_features
from .spectral_waveform_dictionary import raw_features


def assignment_features(carriers, physical_estimates, band):
    """Carrier phase plus local full-patch attributes from the observed-X estimate."""
    carrier = np.asarray(carriers).reshape(-1, 400)
    physical = np.asarray(physical_estimates).reshape(-1, 400)
    if (
        carrier.shape != physical.shape
        or not np.isfinite(carrier).all()
        or not np.isfinite(physical).all()
    ):
        raise ValueError(
            "matched finite carrier and physical-estimate patches required"
        )
    return np.column_stack(
        (
            spectral_features(carrier, band),
            raw_features(physical, None, None, mode="theta_alpha_rms"),
        )
    )


def feature_coordinate(features, semantic_fraction):
    """Freeze a pooled carrier scale and three train-only attribute scales."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 4 or len(values) < 2:
        raise ValueError("multiple complete training patches required")
    if not 0 <= semantic_fraction < 1:
        raise ValueError("semantic distance fraction must be in [0,1)")
    variance = values.var(axis=0)
    phase_scale = max(float(variance[:-3].sum()), 1e-12)
    weight = np.r_[
        np.full(values.shape[1] - 3, np.sqrt((1 - semantic_fraction) / phase_scale)),
        np.sqrt(semantic_fraction / (3 * np.maximum(variance[-3:], 1e-12))),
    ]
    return values.mean(axis=0), weight


def fit_book(
    carriers,
    physical_estimates,
    band,
    *,
    positive_atoms,
    semantic_fraction,
    seed,
    max_iter=15,
    batch_size=2048,
):
    """Fit centers and carrier atoms from the same observed-X training population."""
    from sklearn.cluster import MiniBatchKMeans

    carriers = np.asarray(carriers, dtype=np.float64).reshape(-1, 400)
    features = assignment_features(carriers, physical_estimates, band)
    mean, weight = feature_coordinate(features, semantic_fraction)
    scaled = ((features - mean) * weight).astype(np.float32)
    if not 1 <= positive_atoms <= len(carriers):
        raise ValueError("positive vocabulary must fit within the training population")
    clustering = MiniBatchKMeans(
        n_clusters=positive_atoms,
        random_state=seed,
        n_init=1,
        batch_size=batch_size,
        max_iter=max_iter,
        max_no_improvement=8,
        reassignment_ratio=0.0,
    )
    labels = clustering.fit_predict(scaled)
    count = np.bincount(labels, minlength=positive_atoms)
    atoms = np.zeros((positive_atoms + 1, 400), np.float64)
    np.add.at(atoms[1:], labels, carriers)
    occupied = count > 0
    atoms[1:][occupied] /= count[occupied, None]
    # A never-occupied center receives a real training carrier as a finite fallback.
    if not occupied.all():
        for code in np.flatnonzero(~occupied):
            distance = np.square(scaled - clustering.cluster_centers_[code]).sum(axis=1)
            atoms[code + 1] = carriers[int(distance.argmin())]
    return dict(
        mean=mean,
        weight=weight,
        centers=clustering.cluster_centers_.astype(np.float64),
        atoms=atoms,
        counts=np.r_[0, count],
        iterations=int(clustering.n_iter_),
    )


class AnchoredCarrierDictionary:
    def __init__(self, source, books, metadata):
        self.source = source
        self.books = books
        self.metadata = dict(metadata, positions=source.metadata["positions"])
        self.codebook_size = int(metadata["codebook_size"])
        if (
            metadata["schema"] != "anchored_carrier_dictionary_v1"
            or self.codebook_size < 2
        ):
            raise ValueError("versioned anchored-carrier vocabulary required")
        for branch in ("S", "O"):
            for band in range(5):
                book = books[branch, band]
                width = spectral_features(np.zeros((1, 400)), band).shape[1] + 3
                shapes = dict(
                    mean=(width,),
                    weight=(width,),
                    centers=(self.codebook_size - 1, width),
                    atoms=(self.codebook_size, 400),
                    counts=(self.codebook_size,),
                )
                if any(
                    np.shape(book[key]) != shape or not np.isfinite(book[key]).all()
                    for key, shape in shapes.items()
                ):
                    raise ValueError(
                        "finite carrier atoms and matching assignment coordinates required"
                    )
                if np.any(book["counts"] < 0) or not np.array_equal(
                    book["atoms"][0], np.zeros(400)
                ):
                    raise ValueError(
                        "reserved missing ID zero must decode to a zero carrier"
                    )

    def roles(self, dataset, branch=None):
        return self.source.roles(dataset)

    def encode(self, values, dataset, observed):
        values = np.asarray(values, dtype=np.float64)
        observed = np.asarray(observed)
        positions = len(self.metadata["positions"][dataset])
        if (
            values.ndim != 4
            or values.shape[1:] != (positions, 15, 400)
            or observed.shape != values.shape[:2]
            or observed.dtype != bool
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "finite 30-second native montage and observation flags required"
            )
        physical = self.source.continuous_source(values)
        carriers = self.source.analyze(physical, dataset, observed)
        return self.assign(carriers, physical, dataset, observed)

    def assign(self, carriers, physical, dataset, observed):
        shape = np.shape(carriers["S"])
        roles = self.roles(dataset)
        observed = np.asarray(observed)
        if (
            len(shape) != 4
            or shape[1:] != (len(roles), 15, 400)
            or np.shape(carriers["O"]) != shape
            or observed.shape != shape[:2]
            or observed.dtype != bool
        ):
            raise ValueError(
                "matching native carrier shapes and observation flags required"
            )
        result = {}
        for branch in ("S", "O"):
            ids = np.zeros(shape[:3], np.int32)
            for band in range(5):
                selected = np.flatnonzero(roles == band)
                book = self.books[branch, band]
                raw = assignment_features(
                    carriers[branch][:, selected], physical[branch][:, selected], band
                )
                features = (raw - book["mean"]) * book["weight"]
                norm = np.square(book["centers"]).sum(axis=1)
                assigned = []
                for start in range(0, len(features), 256):
                    query = features[start : start + 256]
                    distances = (
                        np.square(query).sum(axis=1)[:, None]
                        + norm
                        - 2 * query @ book["centers"].T
                    )
                    assigned.append(distances.argmin(axis=1) + 1)
                if assigned:
                    ids[:, selected] = np.concatenate(assigned).reshape(
                        shape[0], len(selected), 15
                    )
            ids[~observed] = 0
            result[branch] = ids
        return result

    def decode_ids(self, ids, dataset):
        roles = self.roles(dataset)
        positions = len(roles)
        if set(ids) != {"S", "O"}:
            raise ValueError("both declared EEG integer sequences required")
        carriers = {}
        expected = None
        for branch in ("S", "O"):
            codes = np.asarray(ids[branch])
            if (
                codes.ndim != 3
                or codes.shape[1:] != (positions, 15)
                or codes.dtype.kind not in "iu"
                or np.any(codes < 0)
                or np.any(codes >= self.codebook_size)
            ):
                raise ValueError("valid native-position integer sequences required")
            if expected is not None and codes.shape != expected:
                raise ValueError("the two branch integer supports must match")
            expected = codes.shape
            output = np.zeros((*codes.shape, 400))
            for band in range(5):
                selected = np.flatnonzero(roles == band)
                output[:, selected] = self.books[branch, band]["atoms"][
                    codes[:, selected]
                ]
            carriers[branch] = output
        return self.source.synthesize(carriers, dataset)

    def predict_continuous(self, values, dataset, observed):
        physical = self.source.continuous_source(values)
        return self.source.synthesize(
            self.source.analyze(physical, dataset, observed), dataset
        )

    def save(self, path, source_path):
        path, source_path = Path(path), Path(source_path).resolve()
        import os

        metadata = dict(
            self.metadata,
            source_file=os.path.relpath(source_path, path.parent),
            source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        )
        arrays = {"metadata": np.asarray(json.dumps(metadata))}
        for (branch, band), book in self.books.items():
            for key, value in book.items():
                arrays[f"{branch}_{band}__{key}"] = value
        pending = path.with_suffix(".pending.npz")
        np.savez_compressed(pending, **arrays)
        pending.replace(path)


def load_model(path):
    from .anchored_spatial_carrier import AnchoredSpatialCarrier

    path = Path(path)
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        if metadata["schema"] != "anchored_carrier_dictionary_v1":
            raise ValueError("versioned anchored-carrier dictionary required")
        source_path = (path.parent / metadata["source_file"]).resolve()
        if (
            hashlib.sha256(source_path.read_bytes()).hexdigest()
            != metadata["source_sha256"]
        ):
            raise ValueError("frozen anchored-carrier source identity changed")
        books = {
            (branch, band): {
                key: saved[f"{branch}_{band}__{key}"].copy()
                for key in (
                    "mean",
                    "weight",
                    "centers",
                    "atoms",
                    "counts",
                    "iterations",
                )
            }
            for branch in ("S", "O")
            for band in range(5)
        }
    return AnchoredCarrierDictionary(
        AnchoredSpatialCarrier.load(source_path), books, metadata
    )
