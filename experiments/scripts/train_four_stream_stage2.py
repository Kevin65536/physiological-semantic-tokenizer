"""Train matched four-stream readers in canonical cached and native audit Hb coordinates."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
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
from src.analysis.single_vq_relations import interval, fit_ridge
from src.data.channel_geometry import project_head_topview
from src.data.downstream_input_cache import DownstreamInputDataset
from src.data.event_alignment import window_within_alignment_support
from src.data.four_stream_future import event_observations, lowpass_split, sample_limits
from src.data.unified_physiology import (
    load_cached_fnirs_record,
    load_native_eeg_record,
    load_native_fnirs_record,
)
from src.data.window_exclusions import paired_window_excluded
from src.foundation.aligned_masked import condition_permutation
from src.foundation.four_stream_relations import check_history_support
from src.inference.four_stream_training import fit_reader, make_model, predictions, training_tensors
from src.utils.io import save_npz_atomic

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "experiments/configs/physiology_semantic_tokenizer/four_stream_stage2_v1.yaml"
CFG = yaml.safe_load(CONTRACT.read_text())
OUT = ROOT / CFG["run"]
DATA = ROOT / "data/cache/physiology_downstream_v2"
DS = CFG["dataset"]
TR = CFG["training"]
SPLITS = ("train", "validation")
READERS = {}
ARRAYS = {}
CONTROLS = [
    "continuous_H",
    "continuous_HE",
    "continuous_wrong",
    "continuous_H_alpha",
    "continuous_alpha_history",
]
ARMS = ["A_H", "A_HE", "A_wrong", "A_generic"] + [
    f"{m}_{a}"
    for m in ("eeg", "hb")
    for a in ("S", "O", "add", "interaction", "generic", "wrong", "alt_add", "alt_interaction")
]


def save_json(path, data):
    tmp = path.with_suffix(".json.partial")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def check_contract():
    if (
        CFG["scope"]
        != dict(splits=list(SPLITS), subjects="phase1_record_roster", test="forbidden", protected="forbidden")
        or DS != "simultaneous_eeg_nirs"
        or (CFG["task"], CFG["record"]) != ("nback", "cnt_nback")
        or CFG["timing"]
        != dict(
            history_seconds=30,
            cutoff_from_marker_s=4,
            horizon_starts_after_cutoff_s=[2, 4, 6],
            horizon_width_s=2,
        )
        or CFG["tokens"] != dict(seed=17, dimension=128, eeg_K=8192, hb_K=1024, patches=15, frozen=True)
        or len(set(CFG["eeg_sites"])) != 4
        or len(set(CFG["hb_sites"])) != 4
    ):
        raise ValueError("unexpected stage 2 protocol")
    if not (OUT / "resolved.yaml").exists():
        (OUT / "resolved.yaml").write_text(CONTRACT.read_text())
    elif yaml.safe_load((OUT / "resolved.yaml").read_text()) != CFG:
        raise ValueError("existing run has a different contract")


def panel_tasks():
    atlas = ROOT / CFG["atlas"]
    if (
        read_json(atlas / "summary.json")["status"] != "completed"
        or read_json(DATA / "verification.json")["status"] != "passed"
    ):
        raise ValueError("unverified panel parent")
    roster = pd.read_csv(atlas / "records.csv").query("dataset==@DS and record=='cnt_nback'")
    if roster.duplicated("subject").any() or set(roster.split) != set(SPLITS):
        raise ValueError("invalid subject roster")
    tasks = []
    for split in SPLITS:
        reader = DownstreamInputDataset(DATA, split=split, task=CFG["task"])
        READERS[split] = reader
        for sub in sorted(roster.loc[roster.split.eq(split), "subject"]):
            if sub not in reader.plan["subject_partitions"][DS][split]:
                raise ValueError("split disagreement before record read")
            key = f"{DS}|{sub}|{CFG['record']}"
            planned = [s for s in reader.plan["samples"] if s["join_key"] == key and s["split"] == split]
            admitted = {s["id"] for s in reader.samples if s["join_key"] == key}
            tasks.append((split, sub, key, planned, admitted))
    return tasks


def extract(task):
    split, sub, key, planned, admitted = task
    start = time.monotonic()
    dest = OUT / "records" / sub
    if dest.with_suffix(".json").exists():
        return read_json(dest.with_suffix(".json"))["performance"]
    reader = READERS[split]
    rec = reader.records[key]
    if (rec.dataset_id, rec.canonical_subject_id, rec.base_record_id) != (
        DS,
        sub,
        CFG["record"],
    ) or split not in SPLITS:
        raise ValueError("unexpected native record")
    # Relocate only the two exact registered file identities after the workspace move.
    files = []
    for item in rec.manifest["source_files"]:
        p = Path(item["path"])
        folder = p.parent.name
        if (
            folder not in (sub + "-EEG", sub + "-NIRS")
            or p.name != "cnt_nback.mat"
            or p.parent.parent.name != "Simultaneous EEG&NIRS"
        ):
            raise ValueError("native file disagreement")
        path = ROOT / "data/Simultaneous EEG&NIRS" / folder / p.name
        if path.resolve() != path:
            raise ValueError("native path redirection")
        files.append(dict(item, path=str(path)))
    if len(files) != 2 or len({i["path"] for i in files}) != 2:
        raise ValueError("expected paired native sources")
    rec = replace(rec, manifest=dict(rec.manifest, source_files=files))
    e = load_native_eeg_record(ROOT, rec)
    h = load_native_fnirs_record(ROOT, rec)
    if (e.native_unit, e.sample_rate_hz, h["provenance"]["native_unit"], h["sample_rate_hz"]) != (
        "uV",
        200,
        "mmol/L",
        10,
    ):
        raise ValueError("unexpected native units or rates")
    names = CFG["eeg_sites"], CFG["hb_sites"]
    ei = [e.channel_names.index(v) for v in names[0]]
    hi = [h["channel_names"].index(v + "_HbO") // 2 for v in names[1]]
    for v, i in zip(names[1], hi):
        if h["channel_names"][2 * i : 2 * i + 2] != [v + "_HbO", v + "_HbR"]:
            raise ValueError("invalid Hb pairing")
    folder = DATA / "records" / DS / sub / CFG["record"]
    ee = rec.manifest["measurement"]["eeg_channel_names"]
    valid_e = np.load(folder / "eeg_supported_channels.npy")[[ee.index(v) for v in names[0]]].all()
    valid_h = np.load(folder / "fnirs_supported_channels.npy").reshape(-1, 2)[hi].all()
    xy = np.stack(
        [
            [
                project_head_topview(v)
                for v in reader.geometry.for_channels(record=rec, modality=m, channel_names=n)
            ]
            for m, n in zip(("eeg", "fnirs"), names)
        ]
    )
    eeg = e.values[:, ei]
    hb = h["values"][:, hi] * 1000
    ledger = []
    arrays = []
    for s in planned:
        ev = s["event"]
        row = dict(
            dataset=DS,
            split=split,
            subject=sub,
            record=rec.base_record_id,
            condition=s["condition"],
            event=s["event_index"],
            id=s["id"],
            eeg_marker=ev["eeg_time_ms"] / 1000,
            hb_marker=ev["fnirs_time_ms"] / 1000,
        )
        e0, e1, et = sample_limits(ev["eeg_time_ms"], 200)
        h0, h1, ht = sample_limits(ev["fnirs_time_ms"], 10)
        row.update(eeg_start=e0 / 200, hb_start=h0 / 10, eeg_stop=e1 / 200, hb_stop=h1 / 10)
        if not valid_e or not valid_h:
            reason = "unsupported sensor"
        elif s["id"] not in admitted:
            reason = "upstream synchronized QC"
        elif ev["alignment_role"] != "paired_eeg_fnirs_marker" or not ev["metadata"].get(
            "alignment_support_ms"
        ):
            reason = "missing paired marker support"
        elif not window_within_alignment_support(ev, -26.1, 38.2):
            reason = "native alignment segment"
        elif paired_window_excluded(DS, split, sub, rec.base_record_id, e0 / 200, h0 / 10, 38.1):
            reason = "overlap with registered QC exclusion"
        else:
            try:
                values = event_observations(eeg, hb, ev["eeg_time_ms"], ev["fnirs_time_ms"])
            except ValueError as ex:
                reason = str(ex)
            else:
                arrays.append(dict(values, xy=xy))
                reason = "complete"
        ledger.append(dict(row, status=reason))
    if arrays:
        save_npz_atomic(dest.with_suffix(".npz"), **{k: np.stack([a[k] for a in arrays]) for k in arrays[0]})
    perf = dict(
        subject=sub,
        seconds=time.monotonic() - start,
        peak_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        source_MiB=sum(Path(i["path"]).stat().st_size for i in files) / 2**20,
        windows=len(arrays),
    )
    save_json(dest.with_suffix(".json"), dict(ledger=ledger, performance=perf, source_files=files))
    return perf


def decode_inputs(model, wave, batch=32):
    codes = []
    decoded = []
    for i in range(0, len(wave), batch):
        result = model(torch.from_numpy(np.ascontiguousarray(wave[i : i + batch])).float())
        codes.append(torch.stack([result["ids"][b] for b in "SO"], 1).numpy())
        decoded.append(torch.stack([result[b] for b in "SO"], 1).numpy())
    return np.concatenate(codes), np.concatenate(decoded)


def morphology(wave, mod):
    colors = 2 if mod == "hb" else 1
    series = wave.reshape(len(wave), -1, colors)
    return np.array([np.mean(series**2), np.mean(np.diff(series, axis=1) ** 2)])


@torch.inference_mode()
def encode(meta, arrays, *, reused=None):
    train = meta.split.eq("train").to_numpy()
    scales = read_json(
        ROOT
        / "experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261122_unified_base_g0_v1/scales.json"
    )
    if scales["fitting_split"] != "train":
        raise ValueError("invalid G0 scales")
    gain = 0.025 / np.sqrt(np.mean(arrays["hb_wave"][train].astype(float) ** 2))
    out = {k: arrays[k] for k in ("xy", "hb_target", "eeg_target", "hb_history", "eeg_alpha")}
    diag = {}
    for mod in ("eeg", "hb"):
        if mod == "eeg" and reused is not None:
            out.update(reused)
            diag[mod] = read_json(ROOT / CFG["paired_run"] / "input_coordinates.json")["nonphysical_control"][
                mod
            ]
            continue
        names = CFG[mod + "_sites"]
        scale = np.array([scales["positions"][mod][DS][v]["scale"] for v in names])
        x = (arrays[mod + "_wave"] * (gain if mod == "hb" else 1) / scale[None, :, None, None]).astype(
            "float32"
        )
        net = audit.network(mod, CFG["tokens"][mod + "_K"], 17, "cpu")
        flat = x.reshape(-1, *x.shape[2:])
        ids, decoded = decode_inputs(net, flat)
        out[mod + "_ids"] = ids.reshape(len(x), 4, 2, 15).transpose(0, 2, 1, 3).astype("uint16")

        def store_book(b):
            return net.branches[b]["quantizer"].codebook.detach().numpy()

        for b in "SO":
            out[mod + b + "_book"] = store_book(b)
        # Pick an engineering control on training histories only; evaluate distortion on validation.
        tr = np.repeat(train, 4)
        ref = morphology(decoded[tr, 0], mod)
        grid = [2, 4, 8, 12, 20, 35] if mod == "eeg" else [0.0334, 0.067, 0.1, 0.167, 0.233, 0.4]
        distances = []
        for freq in grid:
            smooth, _ = lowpass_split(flat[tr], mod, freq)
            distances.append(float(np.linalg.norm(np.log((morphology(smooth, mod) + 1e-12) / (ref + 1e-12)))))
        freq = grid[np.argmin(distances)]
        pieces = lowpass_split(flat, mod, freq)
        np.testing.assert_allclose(pieces[0] + pieces[1], flat, rtol=1e-5, atol=2e-5)
        alt_ids = []
        alt_wave = 0
        for j, piece in enumerate(pieces):
            alt, dec = decode_inputs(net, piece)
            alt_ids.append(alt[:, j])
            alt_wave = alt_wave + dec[:, j]
        out[mod + "_alt"] = (
            np.stack(alt_ids, 1).reshape(len(x), 4, 2, 15).transpose(0, 2, 1, 3).astype("uint16")
        )
        dev = ~tr
        diag[mod] = dict(
            selected_Hz=freq,
            grid_Hz=grid,
            train_log_moment_distance=distances,
            train_physical_S_moments=ref.tolist(),
            train_alternative_S_moments=morphology(pieces[0][tr], mod).tolist(),
            dev_physical_X_mse=float(np.mean((decoded[dev].sum(1) - flat[dev]) ** 2)),
            dev_alternative_X_mse=float(np.mean((alt_wave[dev] - flat[dev]) ** 2)),
        )
    save_json(
        OUT / "input_coordinates.json",
        dict(
            training_hb_gain=gain,
            sites={m: CFG[m + "_sites"] for m in ("hb", "eeg")},
            nonphysical_control=diag,
        ),
    )
    save_npz_atomic(OUT / "panel.npz", **out)


def prepare_cached():
    """Replay identical IDs for EEG and identical event/donor rows on registered cached Hb."""
    source = ROOT / CFG["paired_run"]
    original = read_json(source / "prepared.json")
    if original["status"] != "completed" or read_json(source / "verification.json")["status"] != "passed":
        raise ValueError("matched native-coordinate audit is incomplete")
    meta = pd.read_csv(source / "panel.csv")
    records = {}
    for split, sub, key, _, admitted in panel_tasks():
        rec = READERS[split].records[key]
        if (rec.dataset_id, rec.canonical_subject_id, rec.base_record_id) != (DS, sub, CFG["record"]):
            raise ValueError("wrong record in canonical cache")
        rows = meta[meta.subject.eq(sub)]
        if not rows.id.isin(admitted).all() or not rows.split.eq(split).all():
            raise ValueError("paired events no longer on canonical public panel")
        if rec.manifest["Hb_reuse"] != dict(
            source_cache="data/cache/physiology_semantic_clean_v5",
            source_record=key,
            operation="unchanged_array_link_or_copy",
        ):
            raise ValueError("expected verified byte-identical v5/none measurement copy")
        folder = DATA / "records" / DS / sub / CFG["record"]
        h = load_cached_fnirs_record(ROOT, replace(rec, npz_path=folder))
        if (h["provenance"]["output_unit"], h["provenance"]["preprocessing_support"]) != (
            "uM",
            "complete_registered_record_offline",
        ):
            raise ValueError("invalid cached Hb coordinates")
        indices = [h["channel_names"].index(v + "_HbO") // 2 for v in CFG["hb_sites"]]
        vals = h["values"][:, indices]
        records[sub] = (vals, h["provenance"]["cache_path"])
    waves = []
    histories = []
    targets = []
    raw_stops = []
    for row in meta.itertuples():
        vals, _ = records[row.subject]
        lo, hi, starts = sample_limits(row.hb_marker * 1000, 10)
        if lo < 0 or starts[-1] + 20 > len(vals):
            raise ValueError("matched cache support changed")
        hist = vals[lo:hi]
        future = np.stack([vals[i : i + 20].mean(0) for i in starts], axis=1)
        if not np.isfinite(hist).all() or not np.isfinite(future).all():
            raise ValueError("matched cache contains unsupported Hb")
        histories.append(hist.transpose(1, 0, 2))
        waves.append((hist - hist[:50].mean(0)).transpose(1, 0, 2).reshape(4, 15, 40))
        targets.append(future - hist[-20:].mean(0)[:, None])
        raw_stops.append(len(vals) / 10)
    meta["hb_processing_record_stop_s"] = raw_stops
    meta.to_csv(OUT / "panel.csv", index=False)
    with np.load(source / "panel.npz") as z:
        reused = {k: z[k] for k in z.files if k.startswith("eeg") or k == "xy"}
    arrays = dict(
        reused,
        hb_history=np.stack(histories).astype("float32"),
        hb_wave=np.stack(waves).astype("float32"),
        hb_target=np.stack(targets).astype("float32"),
    )
    encode(meta, arrays, reused=reused)
    (OUT / "support.csv").write_bytes((source / "support.csv").read_bytes())
    save_json(
        OUT / "prepared.json",
        dict(
            original,
            paired_to=CFG["paired_run"],
            hb_support="complete_registered_record_offline",
            cache_paths={s: p for s, (_, p) in records.items()},
        ),
    )


def prepare():
    tasks = panel_tasks()
    (OUT / "records").mkdir(exist_ok=True)
    pilot_task = next(t for t in tasks if t[0] == "train")
    with mp.get_context("fork").Pool(1) as pool:
        pilot = pool.apply(extract, (pilot_task,))
    workers = min(4, len(tasks))
    save_json(
        OUT / "preparation_pilot.json",
        dict(
            pilot=pilot,
            workers=workers,
            threads_each=1,
            rationale="Concurrent independent records, bounded at four readers on rotational storage; one CPU thread per numerical worker.",
        ),
    )
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
        list(pool.map(extract, tasks))
    parts = []
    ledger = []
    for _, sub, *_ in tasks:
        p = OUT / "records" / sub
        rows = read_json(p.with_suffix(".json"))["ledger"]
        ledger.extend(rows)
        if any(r["status"] == "complete" for r in rows):
            with np.load(p.with_suffix(".npz")) as z:
                parts.append(dict(z))
    full = pd.DataFrame(ledger)
    if full.id.duplicated().any() or full.groupby("subject").split.nunique().max() != 1:
        raise ValueError("duplicate exposure or split leakage")
    pop = full.query("status=='complete'").reset_index(drop=True)
    donor = condition_permutation(pop, TR["donor_separation_s"], TR["donor_seed"])
    keep = np.flatnonzero(donor >= 0)
    meta = pop.iloc[keep].copy().reset_index(drop=True)
    meta["donor"] = np.searchsorted(keep, donor[keep])
    full.loc[full.id.isin(pop.loc[donor < 0, "id"]), "status"] = "no disjoint matched donor"
    full.to_csv(OUT / "support.csv", index=False)
    if set(meta.split) != set(SPLITS) or len(meta[meta.split == "validation"].subject.unique()) < 2:
        raise ValueError("insufficient paired subject support")
    arrays = {k: np.concatenate([p[k] for p in parts])[keep] for k in parts[0]}
    for mod in ("hb", "eeg"):
        future = torch.from_numpy(arrays[mod + "_target_start"])
        check_history_support(
            torch.from_numpy(arrays["encoder_start"]),
            torch.from_numpy(arrays["raw_stop"]),
            torch.full((len(meta),), 4.0, dtype=torch.float64),
            future,
            future + 2,
        )
        for i in range(3):
            meta[f"{mod}_future_{i}_s"] = arrays[mod + "_target_start"][:, i] + meta[mod + "_marker"]
    meta.to_csv(OUT / "panel.csv", index=False)
    encode(meta, arrays)
    save_json(
        OUT / "prepared.json",
        dict(
            status="completed",
            counts=meta.groupby("split").size().to_dict(),
            subjects=meta.groupby("split").subject.nunique().to_dict(),
            planned=len(full),
            excluded=full.status.value_counts().to_dict(),
        ),
    )


def fit_cell(job, *, pilot=False):
    arm, seed = job
    folder = OUT / "fits" / f"{arm}_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "completion.json"
    if marker.exists() and not pilot:
        return read_json(marker)
    t = ARRAYS["t"]
    path = OUT / "fits" / f"A_base_{seed}" / "anchor.npy"
    base = torch.from_numpy(np.load(path)) if arm.startswith("A_") and arm != "A_base" else None
    start = time.monotonic()
    model, loss = fit_reader(arm, seed, t, TR, base, steps=12 if pilot else None)
    trainable = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad}
    result = dict(
        arm=arm,
        seed=seed,
        steps=len(loss),
        parameters=sum(v.numel() for v in trainable.values()),
        seconds=time.monotonic() - start,
        peak_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        first_loss=loss[0],
        last_loss=loss[-1],
    )
    if pilot:
        return result
    torch.save(trainable, folder / "readout.pt")
    with torch.inference_mode():
        if arm == "A_base":
            anchor = torch.cat(
                [
                    predictions(model, arm, t, index)
                    for index in np.array_split(np.arange(len(t["xy"])), max(1, len(t["xy"]) // 64))
                ]
            )
            np.save(folder / "anchor.npy", anchor.numpy())
        variants = ["aligned"]
        if arm in ("A_HE", "A_wrong", "eeg_wrong", "hb_wrong", "eeg_interaction", "hb_interaction"):
            variants.append("mismatched")
        vals = {
            v: predictions(model, arm, t, t["dev"], base, swap=v == "mismatched").numpy() for v in variants
        }
        save_npz_atomic(folder / "dev_predictions.npz", **vals)
    pd.DataFrame({"step": np.arange(len(loss)) + 1, "loss": loss}).to_csv(folder / "curve.csv", index=False)
    save_json(marker, dict(result, status="completed"))
    return result


def continuous_controls(data, meta, t):
    tr, dev = t["train"], t["dev"]
    h = data["hb_history"]
    # Physical samples and 15 two-second means, including raw level, local changes and global course.
    z = h.reshape(len(h), 4, 15, 20, 2).mean(3)
    site = np.broadcast_to(np.eye(4)[None], (len(h), 4, 4))
    common = np.concatenate(
        [
            site,
            z.reshape(len(h), 4, -1),
            np.broadcast_to(z.mean(1).reshape(len(h), 1, -1), (len(h), 4, 30)),
            (h - h[:, :, -20:].mean(2, keepdims=True)).reshape(len(h), 4, -1),
        ],
        axis=-1,
    )
    rng = np.random.default_rng(1126)
    projection = rng.normal(size=(128, 8)) / np.sqrt(128)
    descriptors = []
    for j, b in enumerate("SO"):
        code = data["eeg" + b + "_book"] @ projection
        feature = code[data["eeg_ids"][:, j]].reshape(len(h), 4, 3, 5, 8).mean(3)
        descriptors.append(feature.reshape(len(h), -1))

    def append(x, extra):
        return np.concatenate([x, np.broadcast_to(extra[:, None], (len(h), 4, extra.shape[-1]))], axis=-1)

    tokens = np.concatenate(descriptors, axis=1)
    donor = meta.donor.to_numpy()
    for arm, x, mod in [
        ("continuous_H", common, "hb"),
        ("continuous_HE", append(common, tokens), "hb"),
        ("continuous_wrong", append(common, tokens[donor]), "hb"),
        ("continuous_H_alpha", append(common, data["eeg_alpha"].reshape(len(h), -1)), "hb"),
        ("continuous_alpha_history", np.concatenate([site, data["eeg_alpha"]], axis=-1), "eeg"),
    ]:
        y = t[mod + "_target"].numpy()
        fit = fit_ridge(x[tr].reshape(-1, x.shape[-1]), y[tr].reshape(-1, y.shape[-1]))
        pred = fit(x[dev].reshape(-1, x.shape[-1])).reshape(len(dev), 4, -1)
        if not np.isfinite(pred).all():
            raise ValueError("nonfinite continuous comparator")
        save_npz_atomic(OUT / (arm + ".npz"), aligned=pred)


def summarize(meta, t):
    dev = meta.iloc[t["dev"]].reset_index(drop=True)
    scores = []

    def score(arm, seed, mod, preds):
        y = t[mod + "_target"][t["dev"]].numpy()
        for kind, values in preds.items():
            if values.shape != y.shape or not np.isfinite(values).all():
                raise ValueError("missing development predictions")
            errors = ((y - values) ** 2).reshape(len(dev), 4, 3, -1)
            for (person, record), rows in dev.groupby(["subject", "record"]):
                for site, curve in enumerate(errors[rows.index].mean(0)):
                    for lag, channels in zip([2, 4, 6], curve):
                        for target, err in zip(
                            ["HbO", "HbR"] if mod == "hb" else ["alpha_fraction"], channels
                        ):
                            scores.append(
                                dict(
                                    arm=arm,
                                    seed=seed,
                                    input=kind,
                                    subject=person,
                                    record=record,
                                    events=len(rows),
                                    site=CFG[mod + "_sites"][site],
                                    lag_s=lag,
                                    target=target,
                                    nmse=err,
                                )
                            )

    for arm in ["A_base", *ARMS]:
        for seed in TR["seeds"]:
            with np.load(OUT / "fits" / f"{arm}_{seed}" / "dev_predictions.npz") as z:
                score(arm, seed, "hb" if arm.startswith("A_") else arm.split("_")[0], z)
    for arm in CONTROLS:
        with np.load(OUT / (arm + ".npz")) as z:
            score(arm, 0, "eeg" if arm == "continuous_alpha_history" else "hb", z)
    df = pd.DataFrame(scores)
    df.to_csv(OUT / "site_horizon_scores.csv.gz", index=False)
    subjects = df.groupby(["arm", "input", "subject", "seed"], as_index=False).nmse.mean()
    subjects.to_csv(OUT / "subject_scores.csv", index=False)
    table = subjects.groupby(["arm", "input", "subject"]).nmse.mean().unstack(["arm", "input"])
    comparisons = [
        ("A", "A_H", "A_HE"),
        ("A donor training", "A_wrong", "A_HE"),
        ("generic fusion", "A_generic", "A_HE"),
        ("continuous history", "continuous_H", "continuous_HE"),
        ("continuous donor", "continuous_wrong", "continuous_HE"),
    ]
    for mod in ("eeg", "hb"):
        comparisons.extend(
            [
                (mod + " B1", mod + "_S", mod + "_add"),
                (mod + " B1 given O", mod + "_O", mod + "_add"),
                (mod + " B2", mod + "_add", mod + "_interaction"),
                (mod + " generic", mod + "_generic", mod + "_interaction"),
                (mod + " donor training", mod + "_wrong", mod + "_interaction"),
                (mod + " alternative B2", mod + "_alt_add", mod + "_alt_interaction"),
            ]
        )
    results = []
    for name, ref, arm in comparisons:
        left, right = table[ref, "aligned"], table[arm, "aligned"]
        delta = left - right
        mean, lo, hi = interval(delta, np.repeat(DS, len(delta)))
        results.append(
            dict(
                contrast=name,
                reference=ref,
                candidate=arm,
                reference_nmse=left.mean(),
                candidate_nmse=right.mean(),
                difference=mean,
                ci_low=lo,
                ci_high=hi,
                people=len(delta),
                improved=int((delta > 0).sum()),
            )
        )
    pd.DataFrame(results).to_csv(OUT / "contrasts.csv", index=False)
    save_json(
        OUT / "summary.json",
        dict(
            status="completed",
            experiment_id=CFG["experiment_id"],
            fits=(len(ARMS) + 1) * len(TR["seeds"]),
            continuous_controls=len(CONTROLS),
            primary=[r for r in results if r["contrast"] in ("A", "eeg B2", "hb B2")],
            panel=read_json(OUT / "prepared.json"),
            scope="Single-task public development; subject bootstrap exploratory.",
            hb_support=(
                "complete_registered_record_offline" if "paired_run" in CFG else "released_history_only"
            ),
        ),
    )


def train_suite():
    if read_json(OUT / "prepared.json")["status"] != "completed":
        raise ValueError("missing completed cutoff-isolated panel")
    if (OUT / "summary.json").exists():
        return
    meta = pd.read_csv(OUT / "panel.csv")
    with np.load(OUT / "panel.npz") as z:
        data = dict(z)
    t, norm = training_tensors(data, meta)
    ARRAYS["t"] = t
    save_json(OUT / "target_normalization.json", norm)
    ctx = mp.get_context("fork")
    # Benchmark one and concurrent small-model CPU fits at actual training dimensions.
    with ctx.Pool(1) as pool:
        single = pool.apply(partial(fit_cell, pilot=True), [("hb_interaction", 17)])
    pilots = [("hb_interaction", 17), ("eeg_interaction", 17), ("hb_add", 43), ("eeg_add", 43)]
    with ctx.Pool(len(pilots)) as pool:
        started = time.monotonic()
        parallel = pool.map(partial(fit_cell, pilot=True), pilots)
        elapsed = time.monotonic() - started
    workers = 4 if len(pilots) / elapsed >= 0.9 / single["seconds"] else 1
    save_json(
        OUT / "training_pilot.json",
        dict(
            single=single,
            parallel=parallel,
            parallel_wall_s=elapsed,
            workers=workers,
            threads_each=1,
            rationale="Measured simultaneous fits, CPU-only; both permitted GPUs occupied.",
        ),
    )
    total = (len(ARMS) + 1) * len(TR["seeds"])
    done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for arms in [["A_base"], ARMS]:
            jobs = [(arm, seed) for arm in arms for seed in TR["seeds"]]
            for result in pool.map(fit_cell, jobs, chunksize=1):
                done += 1
                save_json(OUT / "progress.json", dict(completed=done, planned=total, latest=result))
                print(
                    f"{done}/{total} {result['arm']} seed={result['seed']} {result['seconds']:.1f}s",
                    flush=True,
                )
    continuous_controls(data, meta, t)
    summarize(meta, t)


@torch.inference_mode()
def verify():
    summary = read_json(OUT / "summary.json")
    meta = pd.read_csv(OUT / "panel.csv")
    ledger = pd.read_csv(OUT / "support.csv")
    assert meta.id.is_unique and set(meta.id) == set(ledger.query("status=='complete'").id)
    assert meta.groupby("subject").split.nunique().eq(1).all()
    donor = meta.iloc[meta.donor].reset_index(drop=True)
    assert sorted(meta.donor) == list(range(len(meta)))
    for col in ["dataset", "split", "subject", "record", "condition"]:
        pd.testing.assert_series_equal(meta[col], donor[col])
    for m in ("eeg", "hb"):
        assert (np.abs(meta[m + "_start"] - donor[m + "_start"]) >= TR["donor_separation_s"] - 1e-9).all()
        stop = meta[m + "_stop"] - meta[m + "_marker"]
        np.testing.assert_allclose(meta[m + "_stop"] - meta[m + "_start"], 30, atol=1e-8)
        assert (stop <= 4 + 1e-9).all()
        for j, lag in enumerate([2, 4, 6]):
            offset = meta[f"{m}_future_{j}_s"] - meta[m + "_marker"]
            assert offset.between(4 + lag - 1e-9, 4 + lag + 1 / (200 if m == "eeg" else 10) + 1e-9).all()
    with np.load(OUT / "panel.npz") as z:
        data = dict(z)
    if "paired_run" in CFG:
        parent = ROOT / CFG["paired_run"]
        paired = pd.read_csv(parent / "panel.csv")
        pd.testing.assert_frame_equal(meta[paired.columns], paired, check_exact=False, rtol=0, atol=1e-9)
        with np.load(parent / "panel.npz") as z:
            for k in data:
                if k.startswith("eeg") or k == "xy":
                    np.testing.assert_array_equal(data[k], z[k])
        assert (meta.hb_processing_record_stop_s >= meta.hb_future_2_s + 2 - 1e-9).all()
        assert read_json(OUT / "prepared.json")["hb_support"] == "complete_registered_record_offline"
    t, norm = training_tensors(data, meta)
    assert norm == read_json(OUT / "target_normalization.json")
    dev = t["dev"]
    replay = {}
    rows = []
    for seed in TR["seeds"]:
        base = None
        for arm in ["A_base", *ARMS]:
            dest = OUT / "fits" / f"{arm}_{seed}"
            completed = read_json(dest / "completion.json")
            assert completed["steps"] == TR["steps"] and completed["status"] == "completed"
            model = make_model(arm, t, TR["width"], TR["rank"]).eval()
            weights = torch.load(dest / "readout.pt", weights_only=True, map_location="cpu")
            assert set(weights) == {k for k, v in model.named_parameters() if v.requires_grad}
            model.load_state_dict(weights, strict=False)
            if arm == "A_base":
                base = predictions(model, arm, t, np.arange(len(meta)))
                np.testing.assert_allclose(base.numpy(), np.load(dest / "anchor.npy"), rtol=1e-5, atol=1e-6)
            mod = "hb" if arm.startswith("A_") else arm.split("_")[0]
            with np.load(dest / "dev_predictions.npz") as saved:
                for variant in saved.files:
                    value = predictions(model, arm, t, dev, base, swap=variant == "mismatched").numpy()
                    np.testing.assert_allclose(value, saved[variant], rtol=1e-5, atol=1e-6)
                    y = data[mod + "_target"][dev].astype(float)
                    sd = np.array(norm[mod]["sd"])
                    raw = value.reshape(y.shape) * sd + norm[mod]["mean"]
                    err = ((raw - y) / sd) ** 2
                    for who, indices in (
                        meta.iloc[dev].reset_index(drop=True).groupby("subject").indices.items()
                    ):
                        rows.append(
                            dict(arm=arm, seed=seed, input=variant, subject=who, nmse=err[indices].mean())
                        )
            replay[f"{arm}_{seed}"] = completed["parameters"]
    for name in ("A_HE", "A_wrong", "A_generic"):
        for seed in TR["seeds"]:
            assert replay[f"{name}_{seed}"] == replay[f"A_H_{seed}"]
    keys = ["arm", "seed", "input", "subject"]
    scored = pd.read_csv(OUT / "subject_scores.csv")
    joined = pd.DataFrame(rows).merge(scored, on=keys, how="outer", validate="one_to_one")
    neural = joined.loc[~joined.arm.isin(CONTROLS)]
    np.testing.assert_allclose(neural.nmse_x, neural.nmse_y, rtol=2e-6)
    tab = neural.query("input=='aligned'").groupby(["subject", "arm"]).nmse_x.mean().unstack()
    for r in summary["primary"]:
        delta = tab[r["reference"]] - tab[r["candidate"]]
        np.testing.assert_allclose(delta.mean(), r["difference"], atol=1e-6)
        assert (delta > 0).sum() == r["improved"] and len(delta) == r["people"]
    assert len(replay) == summary["fits"] and len(CONTROLS) == summary["continuous_controls"]
    for arm in CONTROLS:
        with np.load(OUT / (arm + ".npz")) as pred:
            mod = "eeg" if arm == "continuous_alpha_history" else "hb"
            loss = (pred["aligned"] - t[mod + "_target"][dev].numpy()) ** 2
            measured = (
                pd.Series(loss.reshape(len(dev), -1).mean(1))
                .groupby(meta.iloc[dev].subject.reset_index(drop=True))
                .mean()
            )
            expected = scored[scored.arm.eq(arm)].set_index("subject").nmse.reindex(measured.index)
            np.testing.assert_allclose(measured, expected, rtol=1e-6)
    save_json(
        OUT / "verification.json",
        dict(
            status="passed",
            checkpoint_replays=len(replay),
            continuous_scores=len(CONTROLS),
            development_people=len(tab),
            development_events=len(dev),
            paired_donor_and_native_clocks=True,
            audited_primary_contrasts=len(summary["primary"]),
        ),
    )


def main():
    global CONTRACT, CFG, OUT, TR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "train", "verify"])
    parser.add_argument(
        "--native-audit", action="store_true", help="replay the paired native-window sensitivity coordinate"
    )
    parser.add_argument(
        "--all", action="store_true", help="after preparation run the fits and checkpoint replay"
    )
    args = parser.parse_args()
    if not args.native_audit:
        CONTRACT = CONTRACT.with_name("four_stream_stage2_cached_v1.yaml")
        CFG = yaml.safe_load(CONTRACT.read_text())
        OUT = ROOT / CFG["run"]
        TR = CFG["training"]
    if args.all and args.stage != "prepare":
        parser.error("--all requires prepare")
    with owner_lock(OUT, "stage2"):
        check_contract()
        torch.set_num_threads(1)
        if args.stage == "prepare":
            if not (OUT / "prepared.json").exists():
                if args.native_audit:
                    prepare()
                else:
                    prepare_cached()
            if args.all:
                train_suite()
                verify()
        elif args.stage == "train":
            train_suite()
        else:
            verify()


if __name__ == "__main__":
    main()
