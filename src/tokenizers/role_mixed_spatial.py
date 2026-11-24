"""One integer per site/branch using a fixed dictionary for each native frequency role."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .spatial_waveform_dictionary import SpatialWaveformDictionary, load_model
from .spatial_local_gain import LocalEnergyGainDecoder
from .eeg_component_transfer import IntegerSumSourceDecoder


class FixedRoleMixedDecoder:
    """Construct atoms and spatial readouts before decoding the final integers."""

    def __init__(self, high, low, masks, *, mode="raw", estimator=None):
        if set(masks) != {"S", "O"} or any(
            not 0 <= int(m) < 32 for m in masks.values()
        ):
            raise ValueError("two fixed five-bit role choices required")
        if not isinstance(high, LocalEnergyGainDecoder) or mode not in ["raw", "gain"]:
            raise ValueError(
                "frozen high-energy calibration and explicit correction mode required"
            )
        self.high, self.low, self.masks, self.mode, self.estimator = (
            high,
            low,
            dict(masks),
            mode,
            estimator,
        )
        high_raw = high.source
        if any(
            high_raw.metadata.get(key) != low.metadata.get(key)
            for key in ["input_identity", "scales_identity"]
        ):
            raise ValueError(
                "role dictionaries must use identical data and training scales"
            )
        if (
            high_raw.metadata["positions"] != low.metadata["positions"]
            or high_raw.readout
            or low.readout
            or high_raw.metadata.get("output_projection_hz") != [1, 45]
            or low.metadata.get("output_projection_hz") != [1, 45]
        ):
            raise ValueError(
                "matching separable native band dictionaries and 1–45Hz output required"
            )
        atoms, centers, weights = {}, {}, {}
        sizes = {}
        for dataset in high_raw.metadata["positions"]:
            for branch in "SO":
                np.testing.assert_array_equal(
                    high_raw.roles(dataset, branch), low.roles(dataset, branch)
                )
        for branch in "SO":
            for role in range(5):
                book = high_raw if (int(masks[branch]) >> role) & 1 else low
                atoms[branch, role] = book.atoms[branch, role]
                centers[branch, role] = book.centers[branch, role]
                sizes[branch, role] = len(atoms[branch, role])
                for dataset in high_raw.metadata["positions"]:
                    weights[dataset, branch, role] = book.weights[dataset, branch, role]
        self.sizes = sizes
        metadata = dict(
            high_raw.metadata,
            codebook_size=max(sizes.values()),
            role_dictionary_sizes={b: [sizes[b, r] for r in range(5)] for b in "SO"},
            mixed_high_masks={b: int(m) for b, m in masks.items()},
            construction="only selected role atoms and matching spatial matrices; no unemitted integer streams",
        )
        raw = SpatialWaveformDictionary(
            metadata, high_raw.transfer, centers, atoms, weights
        )
        self.raw = raw
        decoder = (
            raw
            if mode == "raw"
            else LocalEnergyGainDecoder(
                raw,
                high.states,
                high.fixed_gains,
                high.settings,
                high.strength,
                metadata,
            )
        )
        if estimator is not None:
            decoder = IntegerSumSourceDecoder(decoder, estimator, strength=1.0)
        self.decoder = decoder
        self.metadata = dict(
            metadata,
            schema="fixed_role_mixed_spatial_v1",
            mode=mode,
            integer_sum_source=estimator is not None,
            decoder_float_side_information=0,
            positions=decoder.metadata["positions"],
        )

    def emit_ids(self, high_ids, low_ids, dataset):
        """Two searches may run at the encoder; exactly one selected ID leaves it."""
        emitted = {}
        for branch in "SO":
            high, low = np.asarray(high_ids[branch]), np.asarray(low_ids[branch])
            roles = self.raw.roles(dataset, branch)
            if (
                high.shape != low.shape
                or high.ndim != 3
                or high.shape[1:] != (len(roles), 15)
                or high.dtype.kind not in "iu"
                or low.dtype.kind not in "iu"
            ):
                raise ValueError(
                    "two matching complete native integer searches required"
                )
            use_high = ((int(self.masks[branch]) >> roles) & 1).astype(bool)
            emitted[branch] = np.where(use_high[None, :, None], high, low).astype(
                np.int32
            )
        self._validate(emitted, dataset)
        return emitted

    def _validate(self, ids, dataset):
        if dataset not in self.metadata["positions"]:
            raise ValueError("dataset absent from this frozen source/decoder")
        if set(ids) != {"S", "O"}:
            raise ValueError("only two emitted EEG branch integer sequences required")
        shape = None
        for branch in "SO":
            value = np.asarray(ids[branch])
            roles = self.raw.roles(dataset, branch)
            if (
                value.dtype.kind not in "iu"
                or value.ndim != 3
                or value.shape[1:] != (len(roles), 15)
            ):
                raise ValueError("integer [batch,native site,15] interface required")
            if shape is not None and value.shape != shape:
                raise ValueError("matching physical branch support required")
            shape = value.shape
            bound = np.array([self.sizes[branch, r] for r in roles])
            if np.any(value < 0) or np.any(value >= bound[None, :, None]):
                raise ValueError("integer exceeds its declared role dictionary")

    def decode_ids(self, ids, dataset):
        self._validate(ids, dataset)
        return self.decoder.decode_ids(ids, dataset)

    def rate(self, dataset):
        bits = {
            b: np.array(
                [
                    int(np.ceil(np.log2(self.sizes[b, r])))
                    for r in self.raw.roles(dataset, b)
                ]
            )
            for b in "SO"
        }
        maximum = max(int(v.max()) for v in bits.values())
        return dict(
            role_sizes=self.metadata["role_dictionary_sizes"],
            role_bits={
                b: [int(np.ceil(np.log2(self.sizes[b, r]))) for r in range(5)]
                for b in "SO"
            },
            mean_payload_bits_per_second_per_position=float(
                sum(v.sum() for v in bits.values()) / (2 * len(bits["S"]))
            ),
            fixed_maximum_width_bits_each_ID=maximum,
            all_slots_at_max_width_bits_per_second_per_position=maximum,
            integers_per_branch_per_position_30s=15,
            extra_integers=0,
            extra_floating_side_information=0,
        )

    def save(self, path, *, high_path, low_path, source_path=None):
        files = dict(
            high_calibrated_decoder=Path(high_path), low_original_decoder=Path(low_path)
        )
        if self.estimator is not None:
            if source_path is None:
                raise ValueError("frozen integer-sum source reference required")
            files["component_source"] = Path(source_path)
        metadata = dict(
            self.metadata,
            files={
                key: dict(
                    path=str(file.resolve()),
                    sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                )
                for key, file in files.items()
            },
        )
        target = Path(path)
        temporary = target.with_suffix(".pending.npz")
        np.savez_compressed(temporary, metadata=np.asarray(json.dumps(metadata)))
        temporary.replace(target)


def load_role_mixed_decoder(path, *, frozen_cache=None):
    from .eeg_transient_gain import load_model as load_source

    path = Path(path)
    with np.load(path, allow_pickle=False) as f:
        metadata = json.loads(str(f["metadata"]))
    if metadata["schema"] != "fixed_role_mixed_spatial_v1":
        raise ValueError("fixed role mixed dictionary manifest required")
    files = {}
    for key, value in metadata["files"].items():
        file = (path.parent / value["path"]).resolve()
        if hashlib.sha256(file.read_bytes()).hexdigest() != value["sha256"]:
            raise ValueError("frozen role dictionary/source changed")
        files[key] = file

    def read(file, loader):
        if frozen_cache is None:
            return loader(file)
        key = str(file)
        if key not in frozen_cache:
            frozen_cache[key] = loader(file)
        return frozen_cache[key]

    estimator = (
        read(files["component_source"], load_source)
        if "component_source" in files
        else None
    )
    return FixedRoleMixedDecoder(
        read(files["high_calibrated_decoder"], load_model),
        read(files["low_original_decoder"], load_model),
        metadata["mixed_high_masks"],
        mode=metadata["mode"],
        estimator=estimator,
    )
