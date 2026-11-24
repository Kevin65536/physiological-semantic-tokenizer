"""Fixed, training-learned electrode to band assignments for single-ID EEG."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .spatial_waveform_dictionary import BANDS, SpatialWaveformDictionary


class LearnedRoleDictionary(SpatialWaveformDictionary):
    def __init__(
        self, metadata, transfer, centers, atoms, weights, feature_weights=None
    ):
        super().__init__(
            metadata, transfer, centers, atoms, weights, feature_weights=feature_weights
        )
        for dataset, names in metadata["positions"].items():
            expected = np.bincount(
                np.arange(len(names)) % len(BANDS), minlength=len(BANDS)
            )
            for branch in ("S", "O"):
                roles = self.roles(dataset, branch)
                if (
                    roles.shape != (len(names),)
                    or roles.dtype.kind not in "iu"
                    or np.any(roles < 0)
                    or np.any(roles >= len(BANDS))
                    or not np.array_equal(
                        np.bincount(roles, minlength=len(BANDS)), expected
                    )
                ):
                    raise ValueError(
                        "fixed native band roles must preserve every original band capacity"
                    )
                for band in range(len(BANDS)):
                    if np.shape(weights[dataset, branch, band]) != (
                        expected[band],
                        len(names),
                    ):
                        raise ValueError(
                            "spatial coefficients must match learned native roles"
                        )

    def roles(self, dataset, branch=None):
        if branch is None:
            return np.asarray(
                self.metadata["native_roles"][dataset]["S"], dtype=np.int64
            )
        return np.asarray(
            self.metadata["native_roles"][dataset][branch], dtype=np.int64
        )

    @classmethod
    def from_source(cls, source, roles, weights, metadata=None):
        details = deepcopy(source.metadata)
        details.update(
            schema="learned_role_dictionary_v1",
            native_roles=roles,
            encoder_rule="fixed_Wiener_then_learned_native_band_role_nearest_atom",
        )
        details.update(metadata or {})
        return cls(
            details,
            source.transfer,
            source.centers,
            source.atoms,
            weights,
            source.feature_weights,
        )


def load_model(path):
    with np.load(Path(path), allow_pickle=False) as fields:
        metadata = json.loads(str(fields["metadata"]))
        if metadata["schema"] != "learned_role_dictionary_v1":
            raise ValueError("versioned learned native roles required")
        transfer = {branch: fields["transfer_" + branch] for branch in ("S", "O")}
        centers, atoms, feature_weights, weights = {}, {}, {}, {}
        for branch in ("S", "O"):
            for band in range(len(BANDS)):
                centers[branch, band] = fields[f"centers_{branch}_{band}"]
                atoms[branch, band] = fields[f"atoms_{branch}_{band}"]
                key = f"feature_weights_{branch}_{band}"
                if key in fields:
                    feature_weights[branch, band] = fields[key]
                for dataset in metadata["positions"]:
                    weights[dataset, branch, band] = fields[
                        f"weights_{dataset}_{branch}_{band}"
                    ]
    return LearnedRoleDictionary(
        metadata, transfer, centers, atoms, weights, feature_weights
    )


def fit_band(gram, cross, target_energy, anchors, ridge_fraction):
    """One target-specific zero-intercept fit on the actual scored samples."""
    anchors = np.asarray(anchors, dtype=np.int64)
    matrix = gram[:, anchors][:, :, anchors].copy()
    right = cross[:, anchors]
    penalty = ridge_fraction * np.trace(matrix, axis1=1, axis2=2) / len(anchors) + 1e-12
    diagonal = np.arange(len(anchors))
    matrix[:, diagonal, diagonal] += penalty[:, None]
    coefficients = np.linalg.solve(matrix, right[..., None])[..., 0]
    # Evaluate the fitted predictor against the unregularized sample moments.
    error = (
        target_energy
        - 2 * np.einsum("pi,pi->p", coefficients, right)
        + np.einsum(
            "pi,pij,pj->p", coefficients, gram[:, anchors][:, :, anchors], coefficients
        )
    )
    return coefficients.T, np.maximum(error, 0.0)


def fit_roles(statistics, roles, ridge_fraction):
    weights = {}
    error = []
    for band in range(len(BANDS)):
        selected = np.flatnonzero(roles == band)
        weight, residual = fit_band(
            statistics["gram"][band],
            statistics["cross"][band],
            statistics["energy"][band],
            selected,
            ridge_fraction,
        )
        weights[band] = weight
        error.append(float(np.mean(residual)))
    return weights, np.asarray(error)


def balanced_roles(statistics, original, ridge_fraction, max_swaps, min_relative_gain):
    """Greedy pair exchange keeps the exact source count at each frequency band."""
    roles = np.asarray(original, dtype=np.int64).copy()
    weights, loss = fit_roles(statistics, roles, ridge_fraction)
    trace = [dict(swap=0, objective=float(loss.sum()), roles=roles.tolist())]
    for step in range(max_swaps):
        best = None
        best_gain = 0.0
        for left in range(len(roles)):
            for right in range(left + 1, len(roles)):
                a, b = int(roles[left]), int(roles[right])
                if a == b:
                    continue
                selected_a = np.flatnonzero(roles == a)
                selected_b = np.flatnonzero(roles == b)
                selected_a[selected_a == left] = right
                selected_b[selected_b == right] = left
                wa, ea = fit_band(
                    statistics["gram"][a],
                    statistics["cross"][a],
                    statistics["energy"][a],
                    np.sort(selected_a),
                    ridge_fraction,
                )
                wb, eb = fit_band(
                    statistics["gram"][b],
                    statistics["cross"][b],
                    statistics["energy"][b],
                    np.sort(selected_b),
                    ridge_fraction,
                )
                new_loss = float(ea.mean() + eb.mean())
                gain = float(loss[a] + loss[b] - new_loss)
                if gain > best_gain:
                    best_gain = gain
                    best = left, right, a, b, wa, wb, float(ea.mean()), float(eb.mean())
        if best is None or best_gain <= min_relative_gain * max(
            float(loss.sum()), 1e-12
        ):
            break
        left, right, a, b, wa, wb, ea, eb = best
        roles[left], roles[right] = roles[right], roles[left]
        weights[a], weights[b] = wa, wb
        loss[a], loss[b] = ea, eb
        trace.append(
            dict(
                swap=step + 1,
                left=left,
                right=right,
                objective=float(loss.sum()),
                improvement=best_gain,
                roles=roles.tolist(),
            )
        )
    return roles, weights, trace
