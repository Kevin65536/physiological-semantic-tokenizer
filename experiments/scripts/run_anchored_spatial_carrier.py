"""Evaluate train-fitted spatial band coordinates on held training subjects, then freeze the basis."""

import argparse
from collections import defaultdict, OrderedDict
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.dual_component_tokenizer import (
    base_training_aggregate,
    joint_reconstruction_metrics,
)
from src.data.dual_component_tokenizer import (
    base_tokenizer_batch,
    base_tokenizer_joint_eeg,
    base_tokenizer_training_data,
)
from src.tokenizers.anchored_spatial_carrier import AnchoredSpatialCarrier, fit_band
from src.tokenizers.spatial_waveform_dictionary import BANDS, load_model
from evaluate_joint_tokenizer import input_scales, write


def make_pair(dataset, subject_covariances, people, all_scales, source, cfg):
    native = source.metadata["positions"][dataset]
    roles = np.arange(len(native)) % len(BANDS)
    meta = dict(
        schema="anchored_spatial_carrier_v1",
        positions={dataset: native},
        bands_hz=[list(b) for b in BANDS],
        input_identity=source.metadata["input_identity"],
        scales_identity=source.metadata["scales_identity"],
    )
    virtual, inverse, selector, physical = {}, {}, {}, {}
    for b in "SO":
        for k in range(5):
            cov = np.mean([subject_covariances[s][b, k] for s in people], axis=0)
            slots = np.flatnonzero(roles == k)
            key = dataset, b, k
            virtual[key], inverse[key], physical[key] = fit_band(
                cov, all_scales[b], slots, cfg["model"]["baseline_ridge_trace_fraction"]
            )
            selector[key] = np.eye(len(native))[:, slots]
    return (
        AnchoredSpatialCarrier(meta, source.transfer, virtual, inverse),
        AnchoredSpatialCarrier(meta, source.transfer, selector, physical),
    )


def subject_covariances(dataset, crops, branch_sd, names, rows, scales, source, cache):
    moments = defaultdict(lambda: defaultdict(list))
    for crop in crops:
        fields, _, _ = base_tokenizer_joint_eeg(
            rows, dict(crop, positions=[]), scales, names, cache
        )
        x = fields["x"].T.reshape(1, len(names), 15, 400)
        estimates = source.predict_continuous(x)
        for b in "SO":
            values = estimates[b][0].reshape(len(names), 6000) / branch_sd[b][:, None]
            # Parseval moments use the real part of the one-sided cross-spectrum.
            spec = np.fft.rfft(values, axis=1) / 6000
            freq = np.fft.rfftfreq(6000, 1 / 200)
            for k, (lo, hi) in enumerate(BANDS):
                band = spec[:, (lo <= freq) & (freq < hi)]
                moments[b, k][rows[crop["row"]]["record"]].append(
                    2 * (band @ band.conj().T).real
                )
    return {
        key: np.mean([np.mean(mats, axis=0) for mats in group.values()], axis=0)
        for key, group in moments.items()
    }


def alignment(virtual, local, mask):
    # Alignment is a secondary coordinate diagnostic on fully scored 30-s windows.
    a = virtual - virtual.mean(axis=1, keepdims=True)
    b = local - local.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.divide(
        (a * b).sum(1),
        den,
        out=np.full(len(a), np.nan),
        where=(den > 1e-15) & mask.all(1),
    )


def dataset_worker(dataset):
    cfg, rows, scales, plan, source, folds, out = STATE
    start = time.monotonic()
    names = source.metadata["positions"][dataset]
    lookup = {name.upper(): i for i, name in enumerate(names)}
    refs = {name.upper(): v for name, v in scales["positions"]["eeg"][dataset].items()}
    branch_sd = {
        b: np.sqrt([refs[n.upper()]["target_variance"][b.lower()][0] for n in names])
        for b in "SO"
    }
    people = sorted(plan[dataset])
    cache = OrderedDict()
    cov = {
        subject: subject_covariances(
            dataset,
            plan[dataset][subject],
            branch_sd,
            names,
            rows,
            scales,
            source,
            cache,
        )
        for subject in people
    }
    parts = defaultdict(list)
    examples = []
    example_folds = []
    for fold in range(cfg["evaluation"]["folds"]):
        fitting = [s for s in people if folds[dataset][s] != fold]
        virtual, physical = make_pair(dataset, cov, fitting, branch_sd, source, cfg)
        virtual.save(out / f"{dataset}_fold_{fold}.npz")
        for s in people:
            if s in fitting:
                continue
            for crop in plan[dataset][s]:
                i, t = crop["row"], crop["start_s"]
                ex = [(i, t, p) for p in crop["positions"]]
                if not ex:
                    raise ValueError("empty registered training crop")
                native = [lookup[rows[i]["positions"][p].upper()] for _, _, p in ex]
                fields, seen, _ = base_tokenizer_joint_eeg(
                    rows, dict(crop, positions=[]), scales, names, cache
                )
                x = fields["x"].T.reshape(1, len(names), 15, 400)
                predicted = source.predict_continuous(x)
                batch = base_tokenizer_batch(
                    rows, ex, scales, array_cache=cache, cache_limit=16
                )
                local = physical.analyze(predicted, dataset, seen[None])
                slots = virtual.analyze(predicted, dataset, seen[None])
                for arm, pred in [
                    ("physical", physical.synthesize(local, dataset)),
                    ("anchored", virtual.synthesize(slots, dataset)),
                ]:
                    selected = {b: v[0, native] for b, v in pred.items()}
                    audit = joint_reconstruction_metrics(
                        selected, batch, 1, input_scales(rows, ex, scales)
                    )
                    for key in (
                        "nmse",
                        "rms_gain",
                        "correlation",
                        "local_rms_correlation",
                        "band_4_8_abs_db",
                        "band_8_13_abs_db",
                        "band_13_30_abs_db",
                        "band_30_45_abs_db",
                    ):
                        parts[arm + "__" + key].append(audit["metrics"][key])
                    if audit["support"]["nonfinite_prediction_points"].any():
                        raise ValueError("nonfinite waveform output")
                corr = np.stack(
                    [
                        alignment(
                            slots[b][0, native].reshape(-1, 6000),
                            local[b][0, native].reshape(-1, 6000),
                            batch["mask"].reshape(-1, 6000),
                        )
                        for b in "SO"
                    ],
                    axis=1,
                )
                parts["anchored__local_alignment"].append(corr[:, :, None])
                parts["valid_points"].append(batch["mask"].sum(axis=(1, 2)))
                examples.extend(ex)
                example_folds.extend([fold] * len(ex))
        write(
            out / f"{dataset}_progress.json",
            dict(
                completed_folds=fold + 1,
                crops=sum(
                    len(plan[dataset][s]) for s in people if folds[dataset][s] <= fold
                ),
                elapsed_seconds=time.monotonic() - start,
            ),
        )
    final, _ = make_pair(dataset, cov, people, branch_sd, source, cfg)
    final.save(out / f"{dataset}_all.npz")
    np.savez_compressed(
        out / f"{dataset}_held.npz",
        examples=np.array(examples),
        fold=np.array(example_folds),
        **{key: np.concatenate(v) for key, v in parts.items()},
    )
    return dataset, time.monotonic() - start


def merge_basis(out, datasets, kind, meta_details):
    models = [AnchoredSpatialCarrier.load(out / f"{d}_{kind}.npz") for d in datasets]
    meta = dict(
        models[0].metadata,
        positions={d: m.metadata["positions"][d] for d, m in zip(datasets, models)},
        **meta_details,
    )
    return AnchoredSpatialCarrier(
        meta,
        models[0].transfer,
        {k: v for m in models for k, v in m.analysis.items()},
        {k: v for m in models for k, v in m.synthesis.items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "summary.json").exists():
        raise FileExistsError("retained evaluation requires fresh output")
    src = cfg["source"]
    g0 = yaml.safe_load((ROOT / src["g0_config"]).read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(g0, ROOT)
    rows = manifest["rows"]
    source = load_model(ROOT / src["waveform_dictionary"])
    if (
        source.metadata["input_identity"] != manifest["cfg_identity"]
        or source.metadata["scales_identity"] != manifest["scales_identity"]
    ):
        raise ValueError("frozen G0 and Wiener identity mismatch")
    plan = json.loads((ROOT / src["native_train_plan"]).read_text())["plan"]
    for d, people in plan.items():
        for s, crops in people.items():
            for crop in crops:
                row = rows[crop["row"]]
                if (
                    row["split"] != "train"
                    or row["dataset"] != d
                    or row["subject"] != s
                    or row["modality"] != "eeg"
                ):
                    raise ValueError("training subject boundary before arrays")
    nf = cfg["evaluation"]["folds"]
    seed = cfg["evaluation"]["seed"]
    folds = {
        d: {
            s: i % nf
            for i, s in enumerate(
                sorted(
                    people,
                    key=lambda s: hashlib.sha256(
                        f"{seed}/{d}/{s}".encode()
                    ).hexdigest(),
                )
            )
        }
        for d, people in plan.items()
    }
    write(out / "fold_subjects.json", folds)
    write(out / "resolved_config.json", cfg)
    global STATE
    STATE = cfg, rows, scales, plan, source, folds, out
    with ProcessPoolExecutor(
        max_workers=cfg["resources"]["dataset_workers"],
        mp_context=multiprocessing.get_context("fork"),
    ) as pool:
        times = dict(pool.map(dataset_worker, sorted(plan)))
    parts = defaultdict(list)
    for d in sorted(plan):
        with np.load(out / f"{d}_held.npz") as f:
            for k in f.files:
                parts[k].append(f[k])
    full = {k: np.concatenate(v) for k, v in parts.items()}
    ex = full["examples"]
    expected = sorted(
        (c["row"], c["start_s"], p)
        for people in plan.values()
        for crops in people.values()
        for c in crops
        for p in c["positions"]
    )
    if sorted(map(tuple, ex.tolist())) != expected or np.any(full["valid_points"] <= 0):
        raise ValueError("held-training panel identities or valid support differ")

    def aggregate(arm, keep):
        return base_training_aggregate(
            {
                k.split("__", 1)[1]: v[keep]
                for k, v in full.items()
                if k.startswith(arm + "__")
            },
            ex[keep],
            rows,
        )

    arms = {a: aggregate(a, slice(None)) for a in ["physical", "anchored"]}
    held = {a: [aggregate(a, full["fold"] == f) for f in range(nf)] for a in arms}
    base, virt = arms["physical"], arms["anchored"]
    rule = cfg["evaluation"]["advancement"]
    nm = lambda v: np.asarray(v["metrics"]["nmse"]["macro"])[:, 0]
    fd = [
        float(nm(held["physical"][f])[1] - nm(held["anchored"][f])[1])
        for f in range(nf)
    ]
    dd = {
        d: float(
            base["metrics"]["nmse"]["by_dataset"][d][1][0]
            - virt["metrics"]["nmse"]["by_dataset"][d][1][0]
        )
        for d in plan
    }
    loc = np.array(virt["metrics"]["local_alignment"]["macro"]).ravel()
    gates = {
        "macro": virt["selection_score"]
        <= (1 - rule["held_macro_NMSE_relative_improvement_min"])
        * base["selection_score"],
        "S": nm(virt)[1]
        <= (1 - rule["held_S_NMSE_relative_improvement_min"]) * nm(base)[1],
        "datasets": sum(v > 0 for v in dd.values())
        >= rule["required_improving_datasets_S"],
        "folds": sum(v > 0 for v in fd) >= rule["required_improving_folds_S"],
        "components": np.all(
            nm(virt) <= rule["maximum_any_component_NMSE_ratio_to_physical"] * nm(base)
        ),
        "alignment": np.all(
            loc >= rule["minimum_local_carrier_correlation_each_branch"]
        ),
    }
    admit = all(gates.values())
    if admit:
        for f in range(nf):
            merge_basis(out, sorted(plan), f"fold_{f}", dict(held_fold=f)).save(
                out / f"fold_model_{f}.npz"
            )
        merge_basis(out, sorted(plan), "all", dict(fitting_split="train")).save(
            out / "model.npz"
        )
    write(
        out / "summary.json",
        dict(
            status="completed",
            admitted_to_hard_training=admit,
            gates={k: bool(v) for k, v in gates.items()},
            arms=arms,
            held_by_fold=held,
            S_improvement_by_dataset=dd,
            S_improvement_by_fold=fd,
            local_carrier_correlation_S_O=loc,
            seconds_by_dataset=times,
            validation_signal_reads=0,
            confirmation_reads=0,
            test_reads=0,
            continuous_diagnostic=True,
        ),
    )
    print(
        json.dumps(
            dict(
                admitted=admit,
                physical=base["selection_score"],
                anchored=virt["selection_score"],
                local_corr=loc.tolist(),
            )
        )
    )


if __name__ == "__main__":
    main()
