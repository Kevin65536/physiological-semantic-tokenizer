"""Single-word vector quantization of closed-loop spatial voltage innovations."""

import json
from pathlib import Path
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def band_limit(values):
    shape = values.shape
    wave = np.asarray(values, float).reshape(*shape[:-2], 6000)
    f = np.fft.rfftfreq(6000, 1 / 200)
    spectrum = np.fft.rfft(wave, axis=-1)
    spectrum[..., (f < 1) | (f > 45)] = 0
    return np.fft.irfft(spectrum, n=6000, axis=-1).reshape(shape)


def fit_tree(covariance):
    variance = np.diag(covariance).clip(1e-12)
    correlation = np.abs(covariance / np.sqrt(variance[:, None] * variance[None]))
    root = int(np.argmax(correlation.sum(1)))
    parents = np.full(len(variance), -1, np.int32)
    order = [root]
    gains = np.zeros(len(variance))
    while len(order) < len(variance):
        remaining = np.setdiff1d(np.arange(len(variance)), order)
        ij = np.unravel_index(
            np.argmax(correlation[np.ix_(order, remaining)]),
            (len(order), len(remaining)),
        )
        parent, child = order[ij[0]], remaining[ij[1]]
        parents[child] = parent
        gains[child] = np.clip(covariance[child, parent] / variance[parent], -1.5, 1.5)
        order.append(int(child))
    residual = variance.copy()
    for child in order[1:]:
        parent = parents[child]
        residual[child] += (
            gains[child] ** 2 * variance[parent]
            - 2 * gains[child] * covariance[child, parent]
        )
    return np.array(order), parents, gains, np.sqrt(residual.clip(1e-10))


def fit_book(samples, seed, size=512):
    samples = np.asarray(samples, np.float32)
    mean = samples.mean(0)
    covariance = (samples - mean).T @ (samples - mean) / len(samples)
    _, vectors = np.linalg.eigh(covariance)
    basis = vectors[:, -64:][:, ::-1].copy()
    features = (samples - mean) @ basis
    learner = MiniBatchKMeans(
        n_clusters=size - 1,
        n_init=1,
        random_state=seed,
        batch_size=512,
        max_iter=16,
        max_no_improvement=5,
        reassignment_ratio=0.0,
    )
    labels = learner.fit_predict(features)
    centers = learner.cluster_centers_.copy()
    # Physical conditional means preserve all 400 samples for each emitted ID.
    atoms = np.zeros((size, 400), np.float64)
    for code in range(size - 1):
        members = samples[labels == code]
        atoms[code + 1] = (
            members.mean(0) if len(members) else mean + centers[code] @ basis.T
        )
    return dict(
        mean=mean,
        basis=basis,
        centers=centers,
        atoms=atoms,
        fit_samples=len(samples),
        iterations=int(learner.n_iter_),
    )


def nearest(values, book):
    feature = (np.asarray(values, np.float32) - book["mean"]) @ book["basis"]
    center = book["centers"]
    norm = np.sum(center * center, 1)
    result = []
    for start in range(0, len(feature), 256):
        q = feature[start : start + 256]
        distance = np.sum(q * q, 1)[:, None] + norm - 2 * q @ center.T
        result.append(distance.argmin(1) + 1)
    return np.concatenate(result).astype(np.int32)


class SpatialPredictiveDictionary:
    def __init__(self, metadata, trees, books):
        self.metadata = metadata
        self.trees = trees
        self.books = books
        self.codebook_size = 512

    def encode_source(self, source, observed):
        shape = source["S"].shape
        n, channels = shape[:2]
        codes = {b: np.zeros((n, channels, 15), np.int32) for b in "SO"}
        for branch in "SO":
            tree = self.trees[branch]
            decoded = np.zeros(shape)
            for site in tree["order"]:
                good = observed[:, site]
                if not good.any():
                    continue
                parent = int(tree["parent"][site])
                prediction = (
                    tree["gain"][site] * decoded[good, parent] if parent >= 0 else 0.0
                )
                innovation = (source[branch][good, site] - prediction) / tree["scale"][
                    site
                ]
                word = nearest(innovation.reshape(-1, 400), self.books[branch]).reshape(
                    -1, 15
                )
                codes[branch][good, site] = word
                decoded[good, site] = (
                    prediction + tree["scale"][site] * self.books[branch]["atoms"][word]
                )
        return codes

    def decode_raw_ids(self, codes):
        if set(codes) != {"S", "O"}:
            raise ValueError("both integer branches required")
        first = np.asarray(codes["S"])
        channels = len(self.trees["S"]["order"])
        if first.ndim != 3 or first.shape[1:] != (channels, 15):
            raise ValueError("native 30-second integer layout required")
        result = {}
        for branch in "SO":
            word = np.asarray(codes[branch])
            if (
                word.shape != first.shape
                or word.dtype.kind not in "ui"
                or np.any(word < 0)
                or np.any(word >= 512)
            ):
                raise ValueError("integer vocabulary indices required")
            output = np.zeros((*word.shape, 400))
            tree = self.trees[branch]
            for site in tree["order"]:
                present = np.any(word[:, site] != 0, axis=1)
                if np.any((word[:, site] != 0).any(1) != (word[:, site] != 0).all(1)):
                    raise ValueError("missing site must use all zero words")
                parent = int(tree["parent"][site])
                prediction = (
                    tree["gain"][site] * output[present, parent] if parent >= 0 else 0.0
                )
                output[present, site] = (
                    prediction
                    + tree["scale"][site]
                    * self.books[branch]["atoms"][word[present, site]]
                )
            result[branch] = output
        result["sum"] = result["S"] + result["O"]
        return result

    def decode_ids(self, codes):
        raw = self.decode_raw_ids(codes)
        result = {b: band_limit(raw[b]) for b in "SO"}
        result["sum"] = result["S"] + result["O"]
        return result

    def project_source(self, source, observed):
        """Matched continuous PCA diagnostic using the same reconstructed prefix."""
        result = {}
        for branch in "SO":
            output = np.zeros_like(source[branch], dtype=float)
            tree = self.trees[branch]
            book = self.books[branch]
            for site in tree["order"]:
                present = observed[:, site]
                if not present.any():
                    continue
                parent = int(tree["parent"][site])
                prediction = (
                    tree["gain"][site] * output[present, parent] if parent >= 0 else 0.0
                )
                value = (source[branch][present, site] - prediction) / tree["scale"][
                    site
                ]
                coordinates = (value - book["mean"]) @ book["basis"]
                restored = book["mean"] + coordinates @ book["basis"].T
                output[present, site] = prediction + tree["scale"][site] * restored
            result[branch] = band_limit(output)
        result["sum"] = result["S"] + result["O"]
        return result

    def save(self, path):
        arrays = {"metadata": np.array(json.dumps(self.metadata))}
        for b in "SO":
            arrays.update({b + "_tree_" + k: v for k, v in self.trees[b].items()})
            arrays.update(
                {
                    b + "_book_" + k: v
                    for k, v in self.books[b].items()
                    if isinstance(v, np.ndarray)
                }
            )
        np.savez_compressed(path, **arrays)


def load_model(path):
    with np.load(Path(path), allow_pickle=False) as a:
        meta = json.loads(str(a["metadata"]))
        if meta["schema"] != "spatial_predictive_dictionary_v1":
            raise ValueError("frozen spatial predictive model required")
        trees = {
            b: {
                k: a[b + "_tree_" + k].copy()
                for k in ("order", "parent", "gain", "scale")
            }
            for b in "SO"
        }
        books = {
            b: {
                k: a[b + "_book_" + k].copy()
                for k in ("mean", "basis", "centers", "atoms")
            }
            for b in "SO"
        }
    return SpatialPredictiveDictionary(meta, trees, books)
