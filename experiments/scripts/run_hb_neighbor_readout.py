"""Train-crossfit one low-rank native-array Hb readout using saved branch integers."""

import argparse
from collections import defaultdict, OrderedDict
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluate_joint_tokenizer import write, input_scales
from src.analysis.dual_component_tokenizer import (
    joint_reconstruction_metrics,
    joint_reconstruction_summary,
)
from src.analysis.tokenizer_shape_diagnostics import training_weights
from src.data.dual_component_tokenizer import (
    base_tokenizer_examples,
    base_tokenizer_batch,
    base_tokenizer_joint_hb,
    base_tokenizer_training_data,
)
from src.tokenizers.contextual_latent_dictionary import load_model as load_source
from src.tokenizers.hb_neighbor_readout import (
    apply_coefficients,
    decode_native,
    fit_site,
    HbNeighborReadout,
    load_model,
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rank(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def plan(rows, cfg):
    groups = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    for i, t, p in base_tokenizer_examples(rows, "hb", "train"):
        row = rows[i]
        groups[row["dataset"]][row["subject"]][row["record"]][i, t].append(p)
    chosen = []
    spec = cfg["training"]
    for d, subjects in sorted(groups.items()):
        for s, records in sorted(subjects.items()):
            for r in sorted(records, key=lambda r: rank([spec["seed"], d, s, r]))[
                : spec["records_per_subject"]
            ]:
                crops = records[r]
                for i, t in sorted(
                    crops, key=lambda c: rank([spec["seed"], d, s, r, *c])
                )[: spec["crops_per_record"]]:
                    chosen.append(
                        dict(
                            dataset=d,
                            subject=s,
                            record=r,
                            row=i,
                            start_s=t,
                            positions=crops[i, t],
                        )
                    )
    return chosen


def encode_crop(row, item, names, scales, source):
    values, observed = base_tokenizer_joint_hb(
        [row],
        dict(row=0, start_s=item["start_s"]),
        scales,
        allowed_splits=(row["split"],),
    )
    lookup = {name: j for j, name in enumerate(names)}
    available = np.zeros(len(names), bool)
    full = np.zeros((len(names), 15, 40), np.float32)
    for local, name in enumerate(row["positions"]):
        pos = lookup[name]
        available[pos] = observed[:, local].all()
        if available[pos]:
            full[pos] = values[:, local].reshape(15, 40)
    ids = source.encode(full)
    for b in ids:
        ids[b][~available] = {"S": 129, "O": 257}[b]
    return ids, available


def crop_metric(prediction, crop, rows, scales):
    return joint_reconstruction_metrics(
        prediction, crop["batch"], 2, input_scales(rows, crop["examples"], scales)
    )


def selected_prediction(values, coefficient, crop):
    pred = apply_coefficients(values, crop["missing"], crop["quiet"], coefficient)
    return {key: value[crop["native"]] for key, value in pred.items()}


def collect(dataset, cfg, rows, scales, items, names, out):
    source = load_source(ROOT / cfg["source"]["checkpoint"])
    cache = OrderedDict()
    examples = np.array(
        [
            (item["row"], item["start_s"], p)
            for item in items
            for p in item["positions"]
        ],
        np.int32,
    )
    weights = training_weights(examples, rows)
    crops = []
    offset = 0
    for ordinal, item in enumerate(items):
        row = rows[item["row"]]
        if (
            row["split"] != "train"
            or row["subject"] != item["subject"]
            or row["dataset"] != dataset
        ):
            raise ValueError("train-only crop before waveform arrays")
        codes, available = encode_crop(row, item, names, scales, source)
        decoded, missing, quiet = decode_native(source, codes, {"S": 129, "O": 257})
        take = np.array(
            [(item["row"], item["start_s"], p) for p in item["positions"]], np.int32
        )
        batch = base_tokenizer_batch(
            rows, take, scales, array_cache=cache, cache_limit=32
        )
        native = np.array([names.index(row["positions"][p]) for p in item["positions"]])
        record = dict(
            identity=item,
            examples=take,
            native=native,
            ids=codes,
            decoded=decoded,
            missing=missing,
            quiet=quiet,
            batch=batch,
            weight=weights[offset : offset + len(take)],
        )
        offset += len(take)
        crops.append(record)
        write(
            out / f"{dataset}_progress.json",
            dict(
                stage="collect",
                crops=ordinal + 1,
                planned=len(items),
                CPU_seconds=time.process_time(),
            ),
        )
    # The input and its integers are retained once, before any leave-subject-out fit.
    payload = {}
    for j, crop in enumerate(crops):
        for k in ("S", "O"):
            payload[f"ids_{k}_{j}"] = crop["ids"][k]
        payload[f"native_{j}"] = crop["native"]
    np.savez_compressed(out / f"{dataset}_train_ids.npz", **payload)
    return crops


def summarize(crops, predictions, rows, scales):
    metrics, support = defaultdict(list), defaultdict(list)
    for crop, pred in zip(crops, predictions):
        audit = crop_metric(pred, crop, rows, scales)
        for k, v in audit["metrics"].items():
            metrics[k].append(v)
        for k, v in audit["support"].items():
            support[k].append(v)
    values = {k: np.concatenate(v) for k, v in metrics.items()}
    counts = {k: np.concatenate(v) for k, v in support.items()}
    ex = np.concatenate([c["examples"] for c in crops])
    return (
        joint_reconstruction_summary(values, counts, ex.tolist(), rows),
        values,
        counts,
    )


def moments(crops, count):
    width = 4 * count
    cov = np.zeros((count, width, width))
    cross = np.zeros((count, width, 4))
    total = np.zeros(count)
    variances = np.full((count, 3, 2), np.nan)
    for crop in crops:
        f = crop["decoded"].transpose(1, 0, 2, 3).reshape(300, width)
        target = np.stack(
            [crop["batch"][b].reshape(-1, 300, 2) for b in ("s", "o")], axis=2
        )
        for j, p in enumerate(crop["native"]):
            valid = crop["batch"]["mask"][j].reshape(300, 2).all(1)
            # Quiet and missing retain the exact frozen decoder value and have no
            # trainable residual. Scoring still includes every admitted target.
            if crop["quiet"][p] or crop["missing"][p] or not valid.any():
                continue
            residual = (target[j] - crop["decoded"][p]).reshape(300, 4)
            w = crop["weight"][j] / valid.sum()
            a = f[valid]
            y = residual[valid]
            cov[p] += w * (a.T @ a)
            cross[p] += w * (a.T @ y)
            total[p] += crop["weight"][j]
            v = np.array([crop["batch"]["v" + b][j, 0, :2] for b in ("x", "s", "o")])
            if np.isfinite(variances[p]).all():
                np.testing.assert_array_equal(variances[p], v)
            variances[p] = v
    return cov, cross, total, variances


def fit_all(stats, count, low_rank, ridge):
    cov, cross, total, var = stats
    result = np.zeros((count, 4 * count, 4))
    for p in range(count):
        if total[p] > 0:
            result[p] = fit_site(
                cov[p] / total[p], cross[p] / total[p], var[p], p, low_rank, ridge
            )
    return result


def dataset_worker(payload):
    dataset, cfg, rows, scales, items, names, out = payload
    resource.setrlimit(resource.RLIMIT_CPU, (690, 700))
    torch.set_num_threads(1)
    began = time.process_time()
    with threadpool_limits(1):
        crops = collect(dataset, cfg, rows, scales, items, names, out)
        people = sorted(
            {c["identity"]["subject"] for c in crops},
            key=lambda s: rank([cfg["training"]["seed"], dataset, s]),
        )
        folds = {s: i % cfg["training"]["folds"] for i, s in enumerate(people)}
        baseline = [
            selected_prediction(
                c["decoded"], np.zeros((len(names), 4 * len(names), 4)), c
            )
            for c in crops
        ]
        summary, base_metrics, _ = summarize(crops, baseline, rows, scales)
        specs = [
            (r, a)
            for r in [0, *cfg["model"]["neighbor_ranks"]]
            for a in cfg["model"]["ridge_fractions"]
        ]
        held = {key: [None] * len(crops) for key in specs}
        final_fits = {}
        for fold in range(cfg["training"]["folds"]):
            fit = [c for c in crops if folds[c["identity"]["subject"]] != fold]
            stats = moments(fit, len(names))
            tests = [
                i
                for i, c in enumerate(crops)
                if folds[c["identity"]["subject"]] == fold
            ]
            for low_rank, ridge in specs:
                coef = fit_all(stats, len(names), low_rank, ridge)
                for j in tests:
                    held[low_rank, ridge][j] = selected_prediction(
                        crops[j]["decoded"], coef, crops[j]
                    )
            write(
                out / f"{dataset}_progress.json",
                dict(
                    stage="CV",
                    completed_fold=fold + 1,
                    planned=cfg["training"]["folds"],
                    CPU_seconds=time.process_time(),
                ),
            )
        result = {}
        metrics = {}
        for r, a in specs:
            key = f"rank{r}_ridge{a:g}"
            result[key], values, _ = summarize(crops, held[r, a], rows, scales)
            metrics[key] = values["nmse"]
        stats = moments(crops, len(names))
        for r, a in specs:
            final_fits[f"rank{r}_ridge{a:g}"] = fit_all(stats, len(names), r, a)
        np.savez_compressed(out / f"{dataset}_fits.npz", **final_fits)
        ex = np.concatenate([c["examples"] for c in crops])
        np.savez_compressed(
            out / f"{dataset}_held_nmse.npz",
            examples=ex,
            baseline=base_metrics["nmse"],
            **metrics,
        )
        # Retain predicted arrays only for this bounded training panel, allowing
        # one fixed choice to be independently checked and visualized later.
        idsfile = out / f"{dataset}_held_predictions.npz"
        payload = dict(
            examples=ex,
            **{
                k: np.concatenate([c["batch"][k] for c in crops])
                for k in ("x", "s", "o", "mask")
            },
        )
        for spec, seq in [
            ("baseline", baseline),
            *[(f"rank{r}_ridge{a:g}", held[r, a]) for r, a in specs],
        ]:
            for b in ("S", "O", "sum"):
                payload[f"{spec}__{b}"] = np.concatenate([p[b] for p in seq])
        np.savez_compressed(idsfile, **payload)
        quiet = sum(int(c["quiet"].sum()) for c in crops)
        missing = sum(int(c["missing"].sum()) for c in crops)
        record = dict(
            dataset=dataset,
            subjects=people,
            folds=folds,
            summary=summary,
            candidates=result,
            physical_crops=len(crops),
            emitted_integer_count=sum(
                sum(v.size for v in c["ids"].values()) for c in crops
            ),
            quiet_native_position_windows=quiet,
            missing_native_position_windows=missing,
            CPU_seconds=time.process_time() - began,
            positions=names,
        )
        write(out / f"{dataset}_summary.json", record)
        return record


def merge_summaries(results, key):
    # Each dataset summary already uses the original within-dataset hierarchy.
    selected = [
        r["summary"] if key == "baseline" else r["candidates"][key] for r in results
    ]
    metrics = {}
    for name in selected[0]["metrics"]:
        by_dataset = {
            d: v for s in selected for d, v in s["metrics"][name]["by_dataset"].items()
        }
        subjects = [v for s in selected for v in s["metrics"][name]["by_subject"]]
        metrics[name] = dict(
            macro=np.mean(list(by_dataset.values()), axis=0).tolist(),
            by_dataset=by_dataset,
            by_subject=subjects,
        )
    return dict(
        metrics=metrics,
        selection_score=float(np.mean(metrics["nmse"]["macro"])),
        inputs=sum(s["inputs"] for s in selected),
        subjects=sum(s["subjects"] for s in selected),
        records=sum(s["records"] for s in selected),
        windows=sum(s["windows"] for s in selected),
    )


def decision(current, local, base, cfg):
    p = cfg["selection"]
    nm = lambda s: np.asarray(s["metrics"]["nmse"]["macro"])
    s_gain = 1 - nm(current)[1:].mean() / nm(local)[1:].mean()
    b_gain = 1 - nm(current)[1:].mean() / nm(base)[1:].mean()
    datasets = list(base["metrics"]["nmse"]["by_dataset"])
    ds = sum(
        np.mean(current["metrics"]["nmse"]["by_dataset"][d][1:])
        < np.mean(local["metrics"]["nmse"]["by_dataset"][d][1:])
        for d in datasets
    )
    index = {
        (v["dataset"], v["subject"]): np.asarray(v["value"])[1:].mean()
        for v in local["metrics"]["nmse"]["by_subject"]
    }
    better = sum(
        np.asarray(v["value"])[1:].mean() < index[v["dataset"], v["subject"]]
        for v in current["metrics"]["nmse"]["by_subject"]
    ) / len(index)
    guards = dict(
        increment_over_local=s_gain
        >= p["cross_position_vs_local_macro_SO_relative_improvement_min"],
        increment_over_frozen=b_gain
        >= p["cross_position_vs_frozen_macro_SO_relative_improvement_min"],
        datasets=ds >= p["minimum_improving_datasets_SO"],
        subjects=better >= p["minimum_improving_subject_fraction"],
        macro_components=bool(
            np.all(
                nm(current)
                <= p["maximum_any_component_color_NMSE_ratio_vs_frozen"] * nm(base)
            )
        ),
        X_vs_local=bool(
            np.all(
                nm(current)[0]
                <= p["maximum_X_color_NMSE_ratio_vs_local"] * nm(local)[0]
            )
        ),
        O_gain=bool(
            np.all(
                np.asarray(current["metrics"]["rms_gain"]["macro"])[2]
                >= p["minimum_O_gain_vs_base_ratio"]
                * np.asarray(base["metrics"]["rms_gain"]["macro"])[2]
            )
        ),
        dataset_components=all(
            np.all(
                np.asarray(current["metrics"]["nmse"]["by_dataset"][d])
                <= p["maximum_dataset_component_color_NMSE_ratio_vs_frozen"]
                * np.asarray(base["metrics"]["nmse"]["by_dataset"][d])
            )
            for d in datasets
        ),
    )
    return dict(
        eligible=all(guards.values()),
        checks=guards,
        relative_SO_improvement_vs_local=s_gain,
        relative_SO_improvement_vs_frozen=b_gain,
        improved_datasets=ds,
        improved_subject_fraction=better,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / "summary.json").exists():
        raise FileExistsError("completed experiment requires new output")
    cfg = yaml.safe_load(args.config.read_text())
    g0 = yaml.safe_load((ROOT / cfg["source"]["g0_config"]).read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(g0, ROOT)
    rows = manifest["rows"]
    items = plan(rows, cfg)
    positions = {}
    for item in items:
        r = rows[item["row"]]
        if r["split"] != "train":
            raise ValueError("train boundary before waveform reads")
        positions.setdefault(r["dataset"], [])
        for name in r["positions"]:
            if name not in positions[r["dataset"]]:
                positions[r["dataset"]].append(name)
    write(out / "resolved_config.json", cfg)
    write(out / "training_plan.json", items)
    payload = [
        (d, cfg, rows, scales, [c for c in items if c["dataset"] == d], names, out)
        for d, names in sorted(positions.items())
    ]
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = list(pool.map(dataset_worker, payload))
    frozen = merge_summaries(results, "baseline")
    records = []
    locals = {
        a: merge_summaries(results, f"rank0_ridge{a:g}")
        for a in cfg["model"]["ridge_fractions"]
    }
    for r in cfg["model"]["neighbor_ranks"]:
        for a in cfg["model"]["ridge_fractions"]:
            key = f"rank{r}_ridge{a:g}"
            current = merge_summaries(results, key)
            records.append(
                dict(
                    key=key,
                    rank=r,
                    ridge=a,
                    summary=current,
                    local_control=locals[a],
                    decision=decision(current, locals[a], frozen, cfg),
                )
            )
    eligible = [r for r in records if r["decision"]["eligible"]]
    selected = (
        min(
            eligible,
            key=lambda r: (r["summary"]["selection_score"], r["rank"], -r["ridge"]),
        )
        if eligible
        else None
    )
    write(
        out / "training_crossfit.json",
        dict(
            frozen=frozen,
            local=locals,
            candidates=records,
            selected=None if selected is None else selected["key"],
            validation_arrays_read=0,
            source_train_exposure="source_model_and_original_G0_scales_seen_all_train_subjects_only_new_readout_crossfit",
        ),
    )
    if selected:
        coefficients = {}
        for d in positions:
            with np.load(out / f"{d}_fits.npz") as f:
                coefficients[d] = f[selected["key"]]
        metadata = dict(
            schema="hb_neighbor_readout_v1",
            positions=positions,
            source_checkpoint=str((ROOT / cfg["source"]["checkpoint"]).resolve()),
            source_sha256=digest(ROOT / cfg["source"]["checkpoint"]),
            missing_ids={"S": 129, "O": 257},
            branch_codebook_sizes={"S": 130, "O": 258},
            cfg_identity=manifest["cfg_identity"],
            scales_identity=manifest["scales_identity"],
            rank=selected["rank"],
            ridge=selected["ridge"],
        )
        HbNeighborReadout(None, metadata, coefficients).save(out / "model.npz")
    write(
        out / "summary.json",
        dict(
            status="train_admitted" if selected else "train_rejected",
            selected=None if selected is None else selected["key"],
            training_subjects=frozen["subjects"],
            training_examples=frozen["inputs"],
            physical_crops=sum(r["physical_crops"] for r in results),
            CPU_seconds_by_dataset={r["dataset"]: r["CPU_seconds"] for r in results},
            screen_read=False,
            confirmation_read=False,
            test_read=False,
            source_integer_count=sum(r["emitted_integer_count"] for r in results),
            missing_position_windows=sum(
                r["missing_native_position_windows"] for r in results
            ),
        ),
    )
    print("selected", None if selected is None else selected["key"])


if __name__ == "__main__":
    main()
