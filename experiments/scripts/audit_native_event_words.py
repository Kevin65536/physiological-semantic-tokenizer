"""Independent train-only integer replay and information audit of native event words."""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/scripts")]
from audit_montage_word_information import restore_montage_streams
from src.analysis.dual_component_tokenizer import base_training_aggregate, joint_reconstruction_metrics
from src.analysis.tokenizer_pattern_audit import SparsePatternAudit, feature_means
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_training_data
from src.tokenizers.band_exchange_words import load_fold_books
from src.tokenizers.montage_word_search import synthesis_matrix
from src.utils.io import save_npz_atomic

FIRST, PULSE_BASE, CAPACITY = 15351, 26743, 58743
ARMS = ("frozen", "semantic64_prior", "semantic64_event", "waveform_prior", "waveform_event")
FIELDS = {"theta": (0, 1), "alpha": (1, 2), "RMS": (2, 10)}
GROUPS = ("all", "ordinary", "sinusoid", "Ricker", "Gaussian_derivative", "event")
STATE = {}


def write(path, value):
    def scalar(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)
    path = Path(path)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=scalar) + "\n")
    pending.replace(path)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def budget():
    unit = os.environ.get("NATIVE_EVENT_AUDIT_UNIT")
    if unit:
        raw = subprocess.check_output(["systemctl", "--user", "show", unit, "-p", "CPUUsageNSec", "--value"], text=True)
        if int(raw) / 1e9 >= 1770:
            raise RuntimeError("1800 CPU-second audit budget exhausted; retained outputs are resumable")


def event_attributes(words):
    """Intrinsic decoded type/parameters; no target or continuous source is consulted."""
    words = np.asarray(words, dtype=np.int32)
    if np.any((words < 0) | (words >= CAPACITY)):
        raise ValueError("outside actual K58743 alphabet")
    kind = np.zeros(words.shape, np.int8)
    kind[(words >= FIRST) & (words < PULSE_BASE)] = 1
    pulse = words >= PULSE_BASE
    kind[pulse] = 2 + (((words[pulse] - PULSE_BASE) // 32) % 2)
    return kind


def type_role(words):
    value = np.asarray(words, dtype=np.int32)
    role = np.where(value > 0, (value - 1) // 1023 + 1, 0)
    second = (value >= 5116) & (value < FIRST)
    role[second] = 6 + (value[second] - 5116) // 2047
    kind = event_attributes(value)
    role[kind > 0] = 10 + kind[kind > 0]
    if np.any((role < 0) | (role >= 14)):
        raise ValueError("invalid actual type-role")
    return role


def event_wave_independent(words):
    code = np.asarray(words, dtype=np.int32)
    if code.ndim != 1 or np.any((code < FIRST) | (code >= CAPACITY)):
        raise ValueError("actual event IDs required")
    levels = 2 ** np.linspace(-4, 6, 16)
    result = np.zeros((len(code), 400))
    harmonic = code < PULSE_BASE
    local = code[harmonic] - FIRST
    bins, phase = 2 + local // 128, (local // 16) % 8
    result[harmonic] = levels[local % 16, None] * np.cos(2 * np.pi * (bins[:, None] * np.arange(400) / 400 + phase[:, None] / 8))
    local = code[~harmonic] - PULSE_BASE
    shape = local // 32
    centers = (shape // 10) * 4
    widths = np.asarray([3, 6, 12, 24, 48])[(shape // 2) % 5]
    u = (np.arange(400) - centers[:, None]) / widths[:, None]
    bell = np.exp(-0.5 * u * u)
    waves = np.where((shape % 2 == 0)[:, None], (1 - u * u) * bell, u * bell)
    waves -= waves.mean(1, keepdims=True)
    waves /= np.max(np.abs(waves), axis=1, keepdims=True)
    result[~harmonic] = levels[local % 16, None] * np.where((local // 16) % 2 == 0, 1.0, -1.0)[:, None] * waves
    return result


def transport16(values):
    values = np.asarray(values)
    if values.dtype.kind not in "iu" or np.any((values < 0) | (values >= CAPACITY)):
        raise ValueError("actual K58743 unsigned integers required")
    payload = values.astype("<u2").tobytes(order="C")
    restored = np.frombuffer(payload, dtype="<u2").reshape(values.shape).copy()
    np.testing.assert_array_equal(restored, values)
    return payload, restored


def decode_integer_events(sent, old, new, state, scales):
    """Only integers, frozen atoms/state/scales and fixed native order are arguments."""
    codes = np.stack([sent[b] for b in "SO"])
    if codes.ndim != 3 or codes.shape[2] != 15 or codes.dtype.kind not in "iu" or np.any(codes >= CAPACITY):
        raise ValueError("two arrays of fifteen actual IDs per native site required")
    n = codes.shape[1]
    codes = codes.reshape(2 * n, 15)
    seen = (codes != 0).all(1)
    if not np.array_equal(seen, (codes != 0).any(1)) or not np.array_equal(seen[:n], seen[n:]):
        raise ValueError("inconsistent native missing observation")
    role = np.full(2 * n, -1, np.int32)
    sites = np.flatnonzero(seen[:n])
    role[sites] = sites % 5
    role[n + sites] = np.roll(role[sites], 2)
    pilot = np.r_[np.arange(n) % 2 == 0, np.arange(n) % 2 == 1]
    atoms = np.zeros((2 * n, 15, 400))
    local = np.zeros_like(atoms)
    for j in np.flatnonzero(seen):
        branch, band = "SO"[j // n], int(role[j])
        event = codes[j] >= FIRST
        ordinary = codes[j, ~event].astype(np.int32)
        if pilot[j]:
            index = ordinary - (1 + band * 1023)
            book = old[branch][band]
            if np.any((index < 0) | (index >= len(book["balanced_atoms"]))):
                raise ValueError("ordinary pilot ID differs from fixed native role")
            atoms[j, ~event] = book["balanced_atoms"][index].astype(float) * book["balanced_gain"][index, None]
        else:
            index = ordinary - (5116 + band * 2047)
            book = new[branch][band]
            if np.any((index < 0) | (index >= len(book["innovation_atoms"]))):
                raise ValueError("ordinary residual ID differs from fixed native role")
            atoms[j, ~event] = book["innovation_atoms"][index]
        local[j, event] = scales[j] * event_wave_independent(codes[j, event])
    matrix = synthesis_matrix(role, state)
    unfiltered = (matrix @ atoms.reshape(2 * n, -1)).reshape(3 * n, 15, 400)
    unfiltered[:2 * n] += local
    unfiltered[2 * n:] += local[:n] + local[n:]
    spectrum = np.fft.rfft(unfiltered.reshape(3 * n, 6000))
    frequency = np.fft.rfftfreq(6000, 1 / 200)
    spectrum[:, (frequency < 1) | (frequency > 45)] = 0
    value = np.fft.irfft(spectrum, n=6000).reshape(3 * n, 15, 400)
    return {"S": value[:n], "O": value[n:2 * n], "sum": value[:n] + value[n:2 * n]}


def initialize(config):
    cfg = yaml.safe_load(Path(config).read_text())
    if cfg["data"] != {"fit": "original_train_other_people_attribute_means_only", "score": "original_train_outer_held_people", "validation": "forbidden", "confirmation": "forbidden", "test": "forbidden"}:
        raise ValueError("explicit original train-only audit contract required before reads")
    producer = yaml.safe_load((ROOT / cfg["producer_config"]).read_text())
    if producer["cap_rounding"] != "floor" or producer["interface"] != {"K": CAPACITY, "context_s": 30, "slot_s": 2, "bits_per_ID": 16, "EEG_rate_Hz": 200}:
        raise ValueError("strict-floor v2 and unchanged single-integer signal interface required")
    run = ROOT / cfg["run"]
    model = run / cfg["model"]
    out = model / cfg["output"]
    out.mkdir(exist_ok=True)
    pop = json.loads((run / cfg["population"]).read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(yaml.safe_load((ROOT / "experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml").read_text()), ROOT)
    if (pop["cfg_identity"], pop["scales_identity"]) != (manifest["cfg_identity"], manifest["scales_identity"]):
        raise ValueError("original data identity or scale changed")
    rows = manifest["rows"]
    if len(pop["cases"]) != 400 or sum(c["count"] for c in pop["cases"]) != 17040:
        raise ValueError("original complete population changed")
    for c in pop["cases"]:
        r = rows[c["crop"]["row"]]
        if (r["split"], r["modality"], r["dataset"], r["subject"]) != ("train", "eeg", c["dataset"], c["subject"]):
            raise ValueError("non-original or non-train sample before signal read")
    names = sorted(pop["native_positions"])
    cases = {d: [c for c in pop["cases"] if c["dataset"] == d] for d in names}
    STATE.update(cfg=cfg, producer=producer, run=run, model=model, out=out, pop=pop, rows=rows, scales=scales,
                 names=names, cases=cases, frozen=run / producer["frozen"], pilot=run / producer["pilot"], warm=run / producer["warm_search"])
    return out


def selected_examples():
    path = STATE["run"] / "component_source_transfer/integer_sum_source_v1/training_examples/waveform_examples.json"
    examples = json.loads(path.read_text())["examples"]
    if len(examples) != 17:
        raise ValueError("all original seventeen examples required")
    chosen = {}
    for index, example in enumerate(examples):
        key = tuple(example[k] for k in ("row", "start_s", "position"))
        if key in chosen:
            raise ValueError("original selection unexpectedly duplicates identity")
        chosen[key] = {**example, "audit_index": index, "audit_selection": "all original17, chosen before current event results"}
    for identity in [(242, 30, 17), (2384, 60, 9), (1890, 30, 19), (1896, 0, 19), (3643, 30, 20)]:
        if identity not in chosen:
            raise ValueError("predeclared P7/CP5/FC2/VisualF7 identity absent")
    write(STATE["out"] / "selected_examples.json", {"source": str(path), "source_sha256": sha(path), "examples": list(chosen.values()), "no_improvement_based_selection": True})
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["prepare", "waves", "information", "all"], default="prepare")
    args = parser.parse_args()
    out = initialize(args.config)
    with (out / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        STATE["selected"] = selected_examples()
        write(out / "resolved_audit_config.json", STATE["cfg"])
        if args.stage != "prepare":
            raise RuntimeError("Implementation preparation only; synthetic decoder checks precede measured audit")


if __name__ == "__main__":
    main()
