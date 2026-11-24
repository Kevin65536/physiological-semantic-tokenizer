"""Single atomic EEG words containing distributed signed linear measurements."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .band_exchange_words import source_properties
from .eeg_transient_gain import BoundedTransientSource

CAPACITY = 1024
HARD_ARMS = ("spatial_attributes", "local_attributes", "spatial_carrier")


def probes(sites, seed):
    """Fixed full-400-point temporal directions, with no target-dependent phase."""
    values = np.random.default_rng(seed).normal(size=(sites, 10, 400))
    values -= values.mean(-1, keepdims=True)
    values /= np.linalg.norm(values, axis=-1, keepdims=True)
    return values


def spatial_directions(covariance):
    covariance = np.asarray(covariance, float)
    count = len(covariance)
    if covariance.shape != (count, count) or not np.isfinite(covariance).all():
        raise ValueError("one finite same-fold native covariance required")
    _, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    vectors = vectors[:, ::-1]
    result = np.empty((count, 10, count))
    for site in range(count):
        for number in range(10):
            mode = vectors[:, (3 * site + number) % min(count, 8)].copy()
            mode *= 1 if mode[site] >= 0 else -1
            mode[site] += 1  # Equal-norm shared mode plus the physical local axis.
            result[site, number] = mode / np.linalg.norm(mode)
    return result


def measure(waveform, seen, state, layout):
    wave = np.asarray(waveform, float)
    seen = np.asarray(seen, bool)
    count = len(seen)
    if wave.shape != (count, 15, 400) or layout not in ("spatial", "local"):
        raise ValueError("one finite 30-second native source and declared layout required")
    if not np.isfinite(wave).all():
        raise ValueError("nonfinite source")
    wave = np.where(seen[:, None, None], wave / state["SD"][:, None, None], 0)
    if layout == "local":
        values = np.einsum("cpt,cjt->cpj", wave, state["temporal"], optimize=True)
    else:
        values = np.empty((count, 15, 10))
        for number in range(10):
            spatial = state["spatial"][:, number] @ wave.reshape(count, 6000)
            values[:, :, number] = np.einsum(
                "cpt,ct->cp", spatial.reshape(count, 15, 400), state["temporal"][:, number], optimize=True
            )
    return np.where(seen[:, None, None], values, 0)


def atomic_ids(measurements, properties, seen, medians, arm):
    if arm not in HARD_ARMS or measurements.shape[-1] != 10:
        raise ValueError("fixed ten-bit atomic measurement scheme required")
    bits = measurements[..., :7] > 0 if arm.endswith("attributes") else measurements > 0
    if arm.endswith("attributes"):
        bits = np.concatenate((bits, properties > medians[:, None]), axis=-1)
    code = np.sum(bits.astype(np.uint16) << np.arange(10, dtype=np.uint16), axis=-1)
    # Reserve missing zero without silently adding an eleventh paid bit. Cells
    # 1022/1023 merge; they differ only in carrier sign zero, never attributes.
    paid = np.minimum(code.astype(np.uint16) + 1, 1023)
    return np.where(np.asarray(seen)[:, None], paid, 0).astype(np.uint16)


def word_features(ids, gain, property_levels, arm):
    ids = np.asarray(ids)
    if (arm not in HARD_ARMS or ids.ndim != 2 or ids.shape[1] != 15
            or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= CAPACITY)):
        raise ValueError("actual single integer in 0..1023 at each native slot required")
    code = np.maximum(ids.astype(np.int32) - 1, 0)
    bits = ((code[..., None] >> np.arange(10)) & 1).astype(float)
    signed = 2 * bits - 1
    signed[ids == 1023, 0] = 0  # The single merged carrier sign has no received sign.
    if arm.endswith("attributes"):
        values = np.empty((*ids.shape, 10))
        values[..., :7] = signed[..., :7] * gain[:, None, :7]
        for attribute in range(3):
            values[..., 7 + attribute] = np.take_along_axis(
                property_levels[:, None, attribute, :], bits[..., 7 + attribute, None].astype(int), axis=-1
            )[..., 0]
    else:
        values = signed * gain[:, None]
    return np.where(ids[..., None] != 0, values, 0)


def infer(features, head):
    values = np.asarray(features, np.float32)
    count = values.shape[0]
    if values.shape != (count, 15, 10):
        raise ValueError("ten decoded measurement coordinates per native atomic word required")
    x = values.transpose(1, 0, 2).reshape(15, -1)
    output = (x - head["xmean"]) @ head["coefficient"] + head["ymean"]
    return output.reshape(15, count, 400).transpose(1, 0, 2)


class DistributedMeasurement:
    def __init__(self, metadata, source, states, heads):
        if metadata["schema"] != "distributed_measurement_v1" or metadata["arm"] not in HARD_ARMS:
            raise ValueError("versioned single-word measurement model required")
        self.metadata, self.source, self.states, self.heads = metadata, source, states, heads

    def encode(self, x, dataset, observed):
        x, observed = np.asarray(x), np.asarray(observed)
        if dataset != self.metadata["dataset"] or x.shape != (1, len(self.metadata["names"]), 15, 400):
            raise ValueError("one complete native array and matching dataset required")
        dense = self.source.predict_continuous(x, dataset, observed)
        arm = self.metadata["arm"]
        layout = "local" if arm.startswith("local") else "spatial"
        result = {}
        for branch in "SO":
            state = self.states[branch]
            waveform = dense[branch][0].astype(np.float32)
            value = measure(waveform, observed[0], state, layout)
            result[branch] = atomic_ids(value, source_properties(waveform), observed[0], state["medians"], arm)[None]
        return result

    def decode_ids(self, ids, dataset):
        if set(ids) != {"S", "O"} or dataset != self.metadata["dataset"]:
            raise ValueError("two native integer streams and fixed dataset metadata required")
        arm = self.metadata["arm"]
        layout = "local" if arm.startswith("local") else "spatial"
        result = {}
        for branch in "SO":
            state = self.states[branch]
            values = np.asarray(ids[branch])
            if values.shape != (1, len(self.metadata["names"]), 15):
                raise ValueError("one complete native integer array required")
            features = word_features(values[0], state[layout + "_gain"], state["property_levels"], arm)
            result[branch] = infer(features, self.heads[branch])[None]
        result["sum"] = result["S"] + result["O"]
        return result

    def save(self, path):
        values = dict(metadata=np.asarray(json.dumps(self.metadata)),
                      source_metadata=np.asarray(json.dumps(self.source.metadata)))
        dataset = self.metadata["dataset"]
        for branch in "SO":
            values["transfer_" + branch] = self.source.transfer[branch]
            for name, value in self.states[branch].items():
                values[f"state_{branch}_{name}"] = value
            for name, value in self.heads[branch].items():
                values[f"head_{branch}_{name}"] = value
        for name, value in self.source.models[dataset].items():
            values["source_" + name] = value
        with Path(path).open("xb") as stream:
            np.savez(stream, **values)


def load_model(path):
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z["metadata"]))
        sm = json.loads(str(z["source_metadata"]))
        source = BoundedTransientSource({b: z["transfer_" + b].copy() for b in "SO"},
                {metadata["dataset"]: {name: z["source_" + name].copy() for name in ("background", "beta")}}, sm)
        states, heads = {}, {}
        for branch in "SO":
            for prefix, destination in (("state", states), ("head", heads)):
                start = f"{prefix}_{branch}_"
                destination[branch] = {k[len(start):]: z[k].copy() for k in z if k.startswith(start)}
    return DistributedMeasurement(metadata, source, states, heads)
