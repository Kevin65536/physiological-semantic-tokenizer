"""Run the registered four-stream follow-up sequence on bounded public inputs."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import multiprocessing as mp
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
import torch
import yaml

import audit_aligned_tokenizer as audit
from run_dual_component_tokenizer import owner_lock
from src.analysis.four_stream_bridge import select_ridge
from src.analysis.conditional_evidence import cell_scores, p1_tables, verify_reader_scores
from src.analysis.single_vq_relations import interval
from src.data.downstream_input_cache import DownstreamInputDataset
from src.data.unified_physiology import load_cached_fnirs_record
from src.inference.four_stream_training import fit_reader, make_model, predictions, training_tensors
from src.inference.conditional_slot_training import setup, run_conditional, new_model, forward
from src.utils.io import save_npz_atomic

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "experiments/configs/physiology_semantic_tokenizer/four_stream_conditional_v2.yaml"
CFG = yaml.safe_load(CONTRACT.read_text())
OUT = ROOT / CFG["run"]
PARENT = OUT.parents[1] / "four_stream_stage2"
DS = "simultaneous_eeg_nirs"
DATA = ROOT / "data/cache/physiology_downstream_v2"
TR = CFG["training"]


def read(path):
    return json.loads(path.read_text())


def save(path, obj):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def load_parent(coord):
    p = PARENT / CFG["parents"]["native" if coord == "strict_common" else coord]
    if (
        read(p / "verification.json")["status"] != "passed"
        or read(p / "summary.json")["status"] != "completed"
    ):
        raise ValueError("unverified parent")
    meta = pd.read_csv(p / "panel.csv")
    if (
        set(meta.split) != {"train", "validation"}
        or set(meta.dataset) != {DS}
        or set(meta.record) != {"cnt_nback"}
        or meta.id.duplicated().any()
        or meta.groupby("subject").split.nunique().max() != 1
    ):
        raise ValueError("panel drift")
    counts = meta.groupby("split").size()
    if counts.to_dict() != {"train": 309, "validation": 91} or meta.groupby(
        "split"
    ).subject.nunique().tolist() != [15, 4]:
        raise ValueError("split support mismatch")
    donor = meta.iloc[meta.donor].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        meta[["subject", "record", "condition", "split"]], donor[["subject", "record", "condition", "split"]]
    )
    if sorted(meta.donor) != list(range(len(meta))) or any(
        (abs(meta[m + "_start"] - donor[m + "_start"]) < 40 - 1e-6).any() for m in ("eeg", "hb")
    ):
        raise ValueError("donor not disjoint")
    with np.load(p / "panel.npz") as z:
        data = dict(z)
    if coord == "strict_common":
        with np.load(OUT / "strict_common" / "codes.npz") as codes:
            data["hb_ids"] = codes["ids"]
    return p, meta, data


def restore(arm, seed, t, old, p):
    model = make_model(arm, t, old["width"], old["rank"])
    state = torch.load(p / "fits" / f"{arm}_{seed}" / "readout.pt", weights_only=True, map_location="cpu")
    if set(state) != {k for k, v in model.named_parameters() if v.requires_grad}:
        raise ValueError("checkpoint state")
    model.load_state_dict(state, strict=False)
    return model.eval()


def p0_replay(coord):
    path, meta, data = load_parent(coord)
    old = yaml.safe_load((path / "resolved.yaml").read_text())["training"]
    t, norm = training_tensors(data, meta)
    dev = t["dev"]
    rows = []
    curves = []
    for f in sorted((path / "fits").glob("*/curve.csv")):
        arm, seed = f.parent.name.rsplit("_", 1)
        c = pd.read_csv(f)
        if len(c) != old["steps"] or c.step.tolist() != list(range(1, old["steps"] + 1)):
            raise ValueError("missing curve")
        curves.append(c.assign(arm=arm, seed=int(seed), origin="parent"))
    for mod in ("eeg", "hb"):
        y = data[mod + "_target"][dev].astype(float)
        mean = np.array(norm[mod]["mean"])
        sd = np.array(norm[mod]["sd"])
        for k, v in [("zero", np.zeros_like(y)), ("training_mean", np.broadcast_to(mean, y.shape))]:
            rows.extend(cell_scores(meta.iloc[dev], y, v, sd, mod, k, 0, training="fixed", input="aligned"))
        for seed in TR["seeds"]:
            for family in ["add", "interaction"] + (["A"] if mod == "hb" else []):
                for wrong in (False, True):
                    arm = (
                        ("A_wrong" if wrong else "A_HE")
                        if family == "A"
                        else mod + "_" + ("wrong" if wrong and family == "interaction" else family)
                    )
                    if family == "add" and wrong:
                        altered = dict(t)
                        altered[mod + "_ids"] = t[mod + "_ids"].clone()
                        altered[mod + "_ids"][:, 1] = t[mod + "_ids"][t["donor"], 1]
                        model, loss = fit_reader(arm, seed, altered, old)
                        curves.append(
                            pd.DataFrame(
                                dict(
                                    step=np.arange(len(loss)) + 1,
                                    loss=loss,
                                    arm=mod + "_wrong_add",
                                    seed=seed,
                                    origin="new",
                                )
                            )
                        )
                    else:
                        model = restore(arm, seed, t, old, path)
                    anchor = (
                        torch.from_numpy(np.load(path / "fits" / f"A_base_{seed}" / "anchor.npy"))
                        if family == "A"
                        else None
                    )
                    for swap in (False, True):
                        with torch.no_grad():
                            z = predictions(model, arm, t, dev, anchor, swap).numpy()
                        variant = "mismatched" if swap else "aligned"
                        if not (family == "add" and wrong):
                            with np.load(path / "fits" / f"{arm}_{seed}" / "dev_predictions.npz") as prev:
                                if variant in prev:
                                    np.testing.assert_allclose(z, prev[variant], atol=1e-6)
                        rows.extend(
                            cell_scores(
                                meta.iloc[dev],
                                y,
                                z.reshape(y.shape) * sd + mean,
                                sd,
                                mod,
                                family,
                                seed,
                                training="wrong" if wrong else "true",
                                input=variant,
                            )
                        )
    folder = OUT / coord
    folder.mkdir(exist_ok=True)
    pd.concat(curves).to_csv(folder / "training_curves.csv.gz", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(folder / "p0_cells.csv.gz", index=False)
    mat = df.groupby(["modality", "arm", "training", "input", "subject"]).nmse.mean().unstack("input")
    mat["cost"] = mat.mismatched - mat.aligned
    mat.reset_index().to_csv(folder / "pairing_matrix.csv", index=False)
    di = mat.cost.unstack("arm")
    di["extra_mismatch_cost"] = di.interaction - di["add"]
    di[di.extra_mismatch_cost.notna()].to_csv(folder / "extra_mismatch_by_person.csv")


def wave_input(coord, path, meta, data):
    orig = PARENT / CFG["parents"]["native"]
    pieces = {}
    if coord == "native":
        for sub in meta.subject.unique():
            folder = orig / "records"
            ledger = [r["id"] for r in read(folder / (sub + ".json"))["ledger"] if r["status"] == "complete"]
            with np.load(folder / (sub + ".npz")) as z:
                pieces.update(zip(ledger, z["hb_wave"], strict=True))
        wave = np.stack([pieces[k] for k in meta.id])
    else:
        # Read float64 cached histories before centering as in the parent, not rounded panel histories.
        histories = []
        records = {}
        for split in ["train", "validation"]:
            r = DownstreamInputDataset(DATA, split=split, task="nback")
            for sub in meta.loc[meta.split == split, "subject"].unique():
                key = f"{DS}|{sub}|cnt_nback"
                if sub not in r.plan["subject_partitions"][DS][split]:
                    raise ValueError("cross-split record")
                rec = r.records[key]
                if rec.manifest["Hb_reuse"]["operation"] != "unchanged_array_link_or_copy":
                    raise ValueError("Hb coordinates")
                h = load_cached_fnirs_record(
                    ROOT, replace(rec, npz_path=DATA / "records" / DS / sub / "cnt_nback")
                )
                ix = [h["channel_names"].index(v + "_HbO") // 2 for v in ["AFF3h", "AFF4h", "P3h", "P4h"]]
                records[sub] = h["values"][:, ix]
        for row in meta.itertuples():
            t = round(row.hb_start * 10)
            hist = records[row.subject][t : t + 300]
            histories.append((hist - hist[:50].mean(0)).transpose(1, 0, 2).reshape(4, 15, 40))
        wave = np.stack(histories).astype("float32")
    scales = read(OUT.parents[1] / "dual_component_tokenizer/20261122_unified_base_g0_v1/scales.json")
    if scales["fitting_split"] != "train":
        raise ValueError("invalid scales")
    gain = read(path / "input_coordinates.json")["training_hb_gain"]
    scale = [scales["positions"]["hb"][DS][v]["scale"] for v in ["AFF3h", "AFF4h", "P3h", "P4h"]]
    return (wave * np.float64(gain) / np.asarray(scale)[None, :, None, None]).astype("float32")


def bridge(coord):
    path, meta, data = load_parent(coord)
    x = wave_input(coord, path, meta, data)
    net = audit.network("hb", 1024, 17, "cpu")
    pre = []
    decoded = []
    with torch.inference_mode():
        for i in range(0, len(x), 8):
            batch = torch.from_numpy(np.ascontiguousarray(x[i : i + 8].reshape(-1, 15, 40)))
            result = net(batch)
            ids = {
                b: torch.tensor(data["hb_ids"][i : i + 8, j].reshape(-1, 15).astype("int64"))
                for j, b in enumerate("SO")
            }
            for b in "SO":
                torch.testing.assert_close(ids[b], result["ids"][b])
            replay = net.decode_ids(ids)["sum"]
            torch.testing.assert_close(replay, result["sum"], rtol=0, atol=0)
            decoded.append(replay.reshape(-1, 4, 15, 40).numpy())
            pre.append(torch.stack([result["latent"][b] for b in "SO"], 1).reshape(-1, 4, 2, 15, 128).numpy())
    vectors = np.stack([data["hb" + b + "_book"][data["hb_ids"][:, j]] for j, b in enumerate("SO")], axis=2)
    tr = np.flatnonzero(meta.split.eq("train"))
    dev = np.flatnonzero(meta.split.eq("validation"))
    y = data["hb_target"].astype(float)
    sd = y[tr].std(0)
    rows = []
    sel = []
    for label, features in [
        ("uncentered", data["hb_history"]),
        ("exact", x),
        ("decoded", np.concatenate(decoded)),
        ("pre_VQ", np.concatenate(pre)),
        ("ID_embeddings", vectors),
    ]:
        p, alpha, loss = select_ridge(features, y, meta.subject.to_numpy(), tr, dev, CFG["ridge_grid"])
        rows.extend(cell_scores(meta.iloc[dev], y[dev], p, sd, "hb", label, 0))
        sel.extend(
            dict(input=label, ridge=a, group_cv_nmse=v, selected=a == alpha)
            for a, v in zip(CFG["ridge_grid"], loss)
        )
    for seed in TR["seeds"]:
        with np.load(path / "fits" / f"A_H_{seed}" / "dev_predictions.npz") as z:
            pred = z["aligned"].reshape(y[dev].shape) * sd + y[tr].mean(0)
        rows.extend(cell_scores(meta.iloc[dev], y[dev], pred, sd, "hb", "historical_A_H", seed))
    folder = OUT / coord
    folder.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / "bridge_cells.csv.gz", index=False)
    pd.DataFrame(sel).to_csv(folder / "ridge_selection.csv", index=False)


def job(args):
    stage, coord = args
    start = time.monotonic()
    if stage == "replay":
        p0_replay(coord)
    elif stage == "bridge":
        bridge(coord)
    else:
        raise ValueError(stage)
    return dict(
        stage=stage,
        coordinate=coord,
        seconds=time.monotonic() - start,
        peak_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )


def p1_job(task):
    coord, kind, seed = task
    tag = f"{coord}_{kind}_{seed}"
    folder = OUT / "readers" / tag
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "completion.json").exists():
        return read(folder / "completion.json")
    start = time.monotonic()
    _, meta, data = load_parent(coord)
    tr = np.flatnonzero(meta.split == "train")
    dev = np.flatnonzero(meta.split == "validation")
    t = setup(data, meta, kind)
    predictions_, models, mu, sd, sel, cal, cv = run_conditional(t, seed, tr, dev, TR)
    rows = []
    payload = {}
    for arm, z in predictions_.items():
        for flip, variant in enumerate(["aligned", "mismatched"]):
            for adjusted in (False, True):
                value = z[flip]
                if adjusted:
                    value = cal[arm][0] + cal[arm][1] * value
                rows.extend(
                    cell_scores(
                        meta.iloc[dev],
                        t["y"][dev].numpy(),
                        value.numpy(),
                        sd.numpy(),
                        t["mod"],
                        arm,
                        seed,
                        task=kind,
                        coordinate=coord,
                        input=variant,
                        calibrated=adjusted,
                    )
                )
            payload[arm + "_" + variant] = z[flip].numpy()
        payload[arm + "_oof"] = cv[arm].numpy()
    pd.DataFrame(rows).to_csv(folder / "cells.csv.gz", index=False)
    pd.DataFrame(t["curve"], columns=["partition", "arm", "step", "loss"]).to_csv(
        folder / "curves.csv.gz", index=False
    )
    save(folder / "selection.json", sel)
    save_npz_atomic(folder / "predictions.npz", **payload)
    torch.save(
        dict(models={k: m.state_dict() for k, m in models.items()}, mu=mu, sd=sd, calibration=cal),
        folder / "readers.pt",
    )
    # Reload and check all final raw-unit development predictions against saved values.
    for arm, model in models.items():
        check = new_model(kind, arm, TR, 0, t["y"].shape[-1])
        check.load_state_dict(torch.load(folder / "readers.pt", weights_only=True)["models"][arm])
        with torch.no_grad():
            for k in (0, 1):
                got = forward(check, t, dev, arm, bool(k)) * sd + (
                    mu if arm == "anchor" else predictions_["anchor"][k]
                )
                torch.testing.assert_close(got, predictions_[arm][k])
    npar = {arm: sum(p.numel() for p in model.parameters()) for arm, model in models.items()}
    if kind == "A" and len({npar[a] for a in ["capacity", "true", "wrong"]}) != 1:
        raise ValueError("unequal budget")
    status = dict(
        job=tag,
        status="completed",
        seconds=time.monotonic() - start,
        peak_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        parameters=npar,
    )
    save(folder / "completion.json", status)
    return status


def later(stage):
    if stage == "p2":
        from analyze_four_stream_transfer import evaluate

        evaluate()
        return
    if not (OUT / "p0.json").exists():
        raise ValueError("P0 incomplete")
    jobs = [
        (c, k, s)
        for c in CFG["parents"]
        for k in ("A", "B_hb", "B_eeg")
        for s in TR["seeds"]
        if k != "B_eeg" or c == "native"
    ]
    # Four concurrent P0 jobs measured 0.6–1.7 GiB/worker. Six single-threaded P1
    # workers fit the 48-core quota alongside the existing encoder diagnostics.
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context("fork")) as pool:
        results = []
        for r in pool.map(p1_job, jobs):
            results.append(r)
            save(OUT / "p1_progress.json", dict(complete=len(results), planned=len(jobs), jobs=results))
            print(r["job"], round(r["seconds"], 1), flush=True)
    cells = pd.concat([pd.read_csv(OUT / "readers" / r["job"] / "cells.csv.gz") for r in results])
    comp = p1_tables(OUT, cells)
    save(
        OUT / "p1.json",
        dict(
            status="completed",
            workers=6,
            jobs=results,
            primary=comp.query(
                "input=='aligned' and not calibrated and ((task=='A' and contrast=='over_Hb_capacity') or (task!='A' and contrast=='over_frozen_anchor'))"
            ).to_dict("records"),
        ),
    )


@torch.inference_mode()
def common_coordinate():
    spec = CONTRACT.with_name("four_stream_cutoff_coordinate_v2.yaml")
    folder = OUT / "strict_common"
    folder.mkdir(exist_ok=True)
    if (folder / "resolved.yaml").exists() and (folder / "resolved.yaml").read_text() != spec.read_text():
        raise ValueError("strict coordinate contract drift")
    (folder / "resolved.yaml").write_text(spec.read_text())
    _, m, data = load_parent("native")
    train = m.split.eq("train").to_numpy()
    if (m.hb_stop - m.hb_marker > 4 + 1e-8).any() or (m.hb_future_0_s - m.hb_stop < 2 - 1e-7).any():
        raise ValueError("strict support mismatch")
    h = data["hb_history"].astype(float)
    h = h - h[:, :, :50].mean(2, keepdims=True)
    gain = 0.025 / np.sqrt(np.mean(h[train] ** 2))
    sites = ["AFF3h", "AFF4h", "P3h", "P4h"]
    scales = read(OUT.parents[1] / "dual_component_tokenizer/20261122_unified_base_g0_v1/scales.json")[
        "positions"
    ]["hb"][DS]
    h = (
        (h * gain / np.array([scales[s]["scale"] for s in sites])[None, :, None, None])
        .astype("float32")
        .reshape(len(m), 4, 15, 40)
    )
    model = audit.network("hb", 1024, 17)
    ids = []
    for i in range(0, len(m), 8):
        result = model(torch.from_numpy(np.ascontiguousarray(h[i : i + 8].reshape(-1, 15, 40))))["ids"]
        ids.append(torch.stack([result[b] for b in "SO"], 1).reshape(-1, 4, 2, 15).transpose(1, 2).numpy())
    save_npz_atomic(folder / "codes.npz", ids=np.concatenate(ids).astype("uint16"))
    save(
        folder / "coordinates.json",
        dict(gain=gain, sites=sites, site_scales=[scales[s]["scale"] for s in sites]),
    )
    # Leave inference_mode before forking the training workers.


def common_fits():
    common_coordinate()
    jobs = [("strict_common", kind, seed) for kind in ("A", "B_hb") for seed in TR["seeds"]]
    with ProcessPoolExecutor(4, mp_context=mp.get_context("fork")) as pool:
        results = list(pool.map(p1_job, jobs))
    folder = OUT / "strict_common"
    comp = p1_tables(
        folder, pd.concat([pd.read_csv(OUT / "readers" / r["job"] / "cells.csv.gz") for r in results])
    )
    save(
        folder / "summary.json",
        dict(
            status="completed",
            jobs=results,
            primary=comp.query(
                "input=='aligned' and not calibrated and ((task=='A' and contrast=='over_Hb_capacity') or (task!='A' and contrast=='over_frozen_anchor'))"
            ).to_dict("records"),
        ),
    )


def finalize():
    required = ["p0.json", "p1.json", "p2.json", "strict_common/summary.json"]
    stages = {f: read(OUT / f) for f in required}
    if any(result["status"] != "completed" for result in stages.values()):
        raise ValueError("incomplete stage")
    verified = verify_reader_scores(OUT, TR["seeds"], load_parent, read)
    if len(verified) != 14:
        raise ValueError("missing verified reader family or seed")
    comparisons = []
    for coord in ("native", "cached"):
        a = (
            pd.read_csv(OUT / coord / "bridge_cells.csv.gz")
            .groupby(["arm", "subject"])
            .nmse.mean()
            .unstack("arm")
        )
        for ref, arm in [
            ("uncentered", "exact"),
            ("exact", "decoded"),
            ("pre_VQ", "ID_embeddings"),
            ("historical_A_H", "ID_embeddings"),
        ]:
            d = a[ref] - a[arm]
            v, lo, hi = interval(d, np.repeat(DS, len(d)))
            comparisons.append(
                dict(
                    coordinate=coord,
                    reference=ref,
                    candidate=arm,
                    delta=v,
                    ci_low=lo,
                    ci_high=hi,
                    improved=int((d > 0).sum()),
                    people=len(d),
                )
            )
    pd.DataFrame(comparisons).to_csv(OUT / "bridge_contrasts.csv", index=False)
    save(
        OUT / "summary.json",
        dict(
            experiment_id=CFG["experiment_id"],
            status="completed",
            stages=required,
            verified_reader_seed_jobs=verified,
            primary_nback=stages["p1.json"]["primary"] + stages["strict_common/summary.json"]["primary"],
            primary_transfer=stages["p2.json"]["components"]["wg"]["primary"],
            bridge_contrasts=comparisons,
            scientific_verdict="exploratory; no replicated conditional edge or systematic discrete bottleneck",
            tokenizer_refit=False,
            protected_access=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["p0", "p1", "p2", "common", "finalize"])
    args = parser.parse_args()
    with owner_lock(OUT, "common" if args.stage == "common" else "controller"):
        torch.set_num_threads(1)
        if (OUT / "resolved.yaml").exists() and yaml.safe_load((OUT / "resolved.yaml").read_text()) != CFG:
            raise ValueError("run identity drift")
        if not (OUT / "resolved.yaml").exists():
            (OUT / "resolved.yaml").write_text(CONTRACT.read_text())
        if args.stage == "p0":
            if (OUT / "p0.json").exists():
                return
            with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("fork")) as pool:
                stats = list(pool.map(job, [(s, c) for c in CFG["parents"] for s in ["replay", "bridge"]]))
            save(OUT / "p0.json", dict(status="completed", jobs=stats))
        elif args.stage == "finalize":
            finalize()
        elif args.stage == "common":
            common_fits()
        else:
            later(args.stage)


if __name__ == "__main__":
    main()
