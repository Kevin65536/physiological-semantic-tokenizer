"""Score two frozen historical EEG references on the exact held-training carrier coordinates."""

import argparse
from collections import defaultdict, OrderedDict
from concurrent.futures import ProcessPoolExecutor
import json, multiprocessing, sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluate_joint_tokenizer import load_candidate, predict, input_scales, write
from apply_joint_tokenizer import checked_path
from src.data.dual_component_tokenizer import (
    base_tokenizer_training_data,
    base_tokenizer_joint_eeg,
    base_tokenizer_batch,
)
from src.analysis.dual_component_tokenizer import (
    joint_reconstruction_metrics,
    base_training_aggregate,
)
from src.tokenizers.spatial_analysis_synthesis import load_encoder
from src.tokenizers.spatial_local_gain import load_decoder


def worker(dataset):
    rows, scales, manifest, args, bundle = STATE
    encoder_path = checked_path(args.bundle, bundle["eeg"]["encoder"])
    decoder = load_decoder(checked_path(args.bundle, bundle["eeg"]))
    encoder = load_encoder(encoder_path)
    if (
        encoder.metadata["positions"] != decoder.metadata["positions"]
        or json.loads(encoder_path.read_text())["source_sha256"]
        != decoder.metadata["source_decoder_sha256"]
    ):
        raise ValueError("matched spatial encoder and decoder required")
    legacy_args = SimpleNamespace(
        kind="legacy",
        modality="eeg",
        continuous=False,
        legacy_run=args.legacy_run,
        checkpoint=args.legacy_run
        / "models/eeg__waveform__vq128/checkpoint_step_10000.pt",
    )
    old = load_candidate(legacy_args, manifest, scales)
    with np.load(args.carrier / (dataset + "_held.npz")) as original:
        examples = original["examples"]
    groups = defaultdict(list)
    for ordinal, (i, t, _) in enumerate(examples):
        if rows[i]["dataset"] != dataset or rows[i]["split"] != "train":
            raise ValueError("training dataset boundary before arrays")
        groups[(int(i), int(t))].append(ordinal)
    saved = {}
    for f in sorted(args.archived_ids.glob(dataset + "_*.npz")):
        with np.load(f) as z:
            if (
                json.loads(str(z["signature"]))["encoder_sha256"]
                != bundle["eeg"]["encoder"]["sha256"]
            ):
                raise ValueError("archived integers encoded by a different model")
            keep = [
                (j, tuple(pair))
                for j, pair in enumerate(z["crops"])
                if tuple(pair) in groups
            ]
            if keep:
                ids = {b: z[b] for b in "SO"}
                for j, key in keep:
                    if key in saved:
                        raise ValueError("duplicate retained crop")
                    saved[key] = {b: ids[b][j : j + 1].copy() for b in "SO"}
    native = decoder.metadata["positions"][dataset]
    look = {v.upper(): n for n, v in enumerate(native)}
    cache = OrderedDict()
    out = defaultdict(list)
    ordered = []
    searched = []
    for (i, t), ordinals in groups.items():
        take = examples[ordinals]
        batch = base_tokenizer_batch(
            rows, take, scales, array_cache=cache, cache_limit=16
        )
        full = saved.get((i, t))
        if full is None:
            fields, valid, _ = base_tokenizer_joint_eeg(
                rows,
                dict(row=i, start_s=t, positions=[]),
                scales,
                native,
                cache,
                allowed_splits=("train",),
            )
            full = encoder.encode(
                fields["x"].T.reshape(1, len(native), 15, 400), dataset, valid[None]
            )
            searched.append((i, t))
        else:
            for ids in full.values():
                if ids.shape != (1, len(native), 15):
                    raise ValueError("retained full native layout mismatch")
        selected = [look[rows[i]["positions"][p].upper()] for _, _, p in take]
        spatial = {
            b: v[0, selected] for b, v in decoder.decode_ids(full, dataset).items()
        }
        hist, codes = predict(old, batch, legacy_args)
        replay, _ = predict(old, batch, legacy_args, codes)
        for key in hist:
            np.testing.assert_allclose(hist[key], replay[key], rtol=1e-5, atol=2e-6)
        for arm, v in [("as_local_gain", spatial), ("G1_K128", replay)]:
            a = joint_reconstruction_metrics(
                v, batch, 1, input_scales(rows, take, scales)
            )
            if a["support"]["nonfinite_prediction_points"].any():
                raise ValueError("nonfinite reference prediction")
            for field in (
                "nmse",
                "rms_gain",
                "band_4_8_abs_db",
                "band_8_13_abs_db",
                "band_30_45_abs_db",
                "local_rms_correlation",
            ):
                out[arm + "__" + field].append(a["metrics"][field])
        ordered.extend(ordinals)
    order = np.argsort(ordered)
    np.testing.assert_array_equal(np.asarray(ordered)[order], np.arange(len(examples)))
    np.savez_compressed(
        args.output / (dataset + ".npz"),
        examples=examples,
        **{k: np.concatenate(v)[order] for k, v in out.items()},
    )
    return dict(
        dataset=dataset,
        positions=len(examples),
        crops=len(groups),
        reused_AS_crops=len(saved),
        newly_encoded_AS_crops=searched,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ["carrier", "bundle", "legacy-run", "archived-ids", "output"]:
        parser.add_argument("--" + k, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    if (args.output / "summary.json").exists():
        raise FileExistsError("completed control audit")
    cfg = json.loads((args.carrier / "resolved_config.json").read_text())
    g0 = yaml.safe_load((ROOT / cfg["source"]["g0_config"]).read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(g0, ROOT)
    spec = json.loads(args.bundle.read_text())
    held = json.loads((args.carrier / "summary.json").read_text())
    if held["validation_signal_reads"] or not held["continuous_diagnostic"]:
        raise ValueError("training carrier audit required")
    datasets = sorted(held["seconds_by_dataset"])
    torch.set_num_threads(1)
    global STATE
    STATE = manifest["rows"], scales, manifest, args, spec
    with ProcessPoolExecutor(
        len(datasets), mp_context=multiprocessing.get_context("fork")
    ) as pool:
        records = list(pool.map(worker, datasets))
    bank = defaultdict(list)
    for d in datasets:
        with np.load(args.output / (d + ".npz")) as z:
            for k in z:
                bank[k].append(z[k])
    full = {k: np.concatenate(v) for k, v in bank.items()}
    arms = {
        arm: base_training_aggregate(
            {k.split("__")[1]: v for k, v in full.items() if k.startswith(arm + "__")},
            full["examples"],
            manifest["rows"],
        )
        for arm in ["as_local_gain", "G1_K128"]
    }
    write(
        args.output / "summary.json",
        dict(
            status="completed",
            records=records,
            arms=arms,
            exposure="Both historical controls fitted on original training subjects. Comparison of frozen outputs on identical masks; this is not an out-of-sample comparison for the old models.",
            validation_reads=0,
            test_reads=0,
        ),
    )
    print({k: round(v["selection_score"], 5) for k, v in arms.items()})


if __name__ == "__main__":
    main()
