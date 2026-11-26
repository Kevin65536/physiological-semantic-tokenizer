"""Frozen-reader transfer and subject-level atlas follow-up on public panels."""

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import multiprocessing as mp
import os
import subprocess

import numpy as np
import pandas as pd
import torch

import audit_aligned_tokenizer as audit
import run_four_stream_conditional as suite
from src.analysis.conditional_evidence import cell_scores, p1_tables
from src.analysis.single_vq_relations import interval, wave_attribute
from src.data.channel_geometry import project_head_topview
from src.data.downstream_input_cache import DownstreamInputDataset
from src.data.dual_component_tokenizer import base_tokenizer_batch
from src.data.event_alignment import window_within_alignment_support
from src.data.four_stream_future import event_observations, sample_limits
from src.data.unified_physiology import load_cached_fnirs_record, load_native_eeg_record
from src.data.window_exclusions import paired_window_excluded
from src.foundation.aligned_masked import condition_permutation
from src.inference.conditional_slot_training import forward, new_model, setup

OUT = suite.OUT / "transfer"
DS = suite.DS
SITES = {"eeg": ["F1", "F2", "P3", "P4"], "hb": ["AFF3h", "AFF4h", "P3h", "P4h"]}
G0 = suite.OUT.parents[1] / "dual_component_tokenizer/20261122_unified_base_g0_v1"


def wg_panel(output=OUT):
    r = DownstreamInputDataset(suite.DATA, split="validation", task="wg")
    roster = r.plan["subject_partitions"][DS]["validation"]
    if set(roster) & set(
        r.plan["subject_partitions"][DS]["train"] + r.plan["subject_partitions"][DS]["test"]
    ):
        raise ValueError("subject partition drift")
    ledger = []
    waves = []
    admitted = {s["id"] for s in r.samples}
    for sub in sorted(roster):
        key = f"{DS}|{sub}|cnt_wg"
        rec = r.records[key]
        folder = suite.DATA / "records" / DS / sub / "cnt_wg"
        if (rec.dataset_id, rec.canonical_subject_id, rec.base_record_id) != (
            DS,
            sub,
            "cnt_wg",
        ) or rec.manifest["Hb_reuse"]["operation"] != "unchanged_array_link_or_copy":
            raise ValueError("unexpected measurement identity")
        h = load_cached_fnirs_record(suite.ROOT, replace(rec, npz_path=folder))
        e = load_native_eeg_record(suite.ROOT, rec)
        if (
            e.sample_rate_hz,
            e.native_unit,
            h["sample_rate_hz"],
            h["provenance"]["output_unit"],
            h["provenance"]["preprocessing_support"],
        ) != (200, "uV", 10, "uM", "complete_registered_record_offline"):
            raise ValueError("unexpected task coordinates")
        if (
            e.source_path.resolve()
            != (suite.ROOT / "data/Simultaneous EEG&NIRS" / (sub + "-EEG") / "cnt_wg.mat").resolve()
        ):
            raise ValueError("wrong EEG record")
        ei = [e.channel_names.index(v) for v in SITES["eeg"]]
        hi = [h["channel_names"].index(v + "_HbO") // 2 for v in SITES["hb"]]
        eeg = e.values[:, ei]
        hb = h["values"][:, hi]
        names = rec.manifest["measurement"]["eeg_channel_names"]
        supported = np.load(folder / "eeg_supported_channels.npy")[
            [names.index(v) for v in SITES["eeg"]]
        ].all()
        xy = np.stack(
            [
                [
                    project_head_topview(v)
                    for v in r.geometry.for_channels(record=rec, modality=m, channel_names=n)
                ]
                for m, n in [("eeg", SITES["eeg"]), ("fnirs", SITES["hb"])]
            ]
        )
        for s in r.plan["samples"]:
            if s["split"] != "validation" or s["join_key"] != key or s["task"] != "wg":
                continue
            ev = s["event"]
            em = ev["eeg_time_ms"]
            hm = ev["fnirs_time_ms"]
            e0 = sample_limits(em, 200)[0] / 200
            h0 = sample_limits(hm, 10)[0] / 10
            row = dict(
                dataset=DS,
                subject=sub,
                record="cnt_wg",
                split="validation",
                condition=s["condition"],
                event=s["event_index"],
                id=s["id"],
                eeg_start=e0,
                hb_start=h0,
                eeg_marker_s=em / 1000,
                hb_marker_s=hm / 1000,
            )
            if s["id"] not in admitted or not supported:
                reason = "upstream QC or unsupported channel"
            elif (
                ev["alignment_role"] != "paired_eeg_fnirs_marker"
                or not ev["metadata"].get("alignment_support_ms")
                or not window_within_alignment_support(ev, -26.1, 38.2)
            ):
                reason = "clock support"
            elif paired_window_excluded(DS, "validation", sub, "cnt_wg", e0, h0, 38.1):
                reason = "QC overlap"
            else:
                try:
                    v = event_observations(eeg, hb, em, hm, cached_hb=True)
                except ValueError as exc:
                    reason = str(exc)
                else:
                    reason = "complete"
                    waves.append(dict(v, xy=xy))
            ledger.append(dict(row, status=reason))
    full = pd.DataFrame(ledger)
    m = full.query("status=='complete'").reset_index(drop=True)
    donor = condition_permutation(m, 40, 1126)
    keep = np.flatnonzero(donor >= 0)
    full.loc[full.id.isin(m.loc[donor < 0, "id"]), "status"] = "unmatched donor"
    full.to_csv(output / "wg_support.csv", index=False)
    m = m.iloc[keep].reset_index(drop=True)
    m["donor"] = np.searchsorted(keep, donor[keep])
    if m.id.duplicated().any() or len(m) == 0:
        raise ValueError("invalid WG panel")
    m.to_csv(output / "wg_panel.csv", index=False)
    return m, {k: np.stack([waves[i][k] for i in keep]) for k in waves[0]}


@torch.inference_mode()
def transfer():
    meta, raw = wg_panel()
    path, source, books = suite.load_parent("cached")
    scales = suite.read(G0 / "scales.json")["positions"]
    gain = suite.read(path / "input_coordinates.json")["training_hb_gain"]
    data = {k: raw[k] for k in ["hb_target", "eeg_target", "xy"]}
    for mod in SITES:
        scale = np.array([scales[mod][DS][v]["scale"] for v in SITES[mod]])
        x = (
            raw[mod + "_wave"] * (np.float64(gain) if mod == "hb" else 1) / scale[None, :, None, None]
        ).astype("float32")
        net = audit.network(mod, 1024 if mod == "hb" else 8192, 17)
        ids = []
        for i in range(0, len(x), 8):
            vals = net(torch.from_numpy(np.ascontiguousarray(x[i : i + 8].reshape(-1, 15, x.shape[-1]))))[
                "ids"
            ]
            ids.append(torch.stack([vals[b] for b in "SO"], 1).reshape(-1, 4, 2, 15).transpose(1, 2).numpy())
        data[mod + "_ids"] = np.concatenate(ids)
        for b in "SO":
            data[mod + b + "_book"] = books[mod + b + "_book"]
            np.testing.assert_array_equal(
                net.branches[b]["quantizer"].codebook.numpy(), data[mod + b + "_book"]
            )
    rows = []
    for task in ["A", "B_eeg", "B_hb"]:
        t = setup(data, meta, task)
        idx = np.arange(len(meta))
        mod = t["mod"]
        coord = "native" if task == "B_eeg" else "cached"
        for seed in suite.TR["seeds"]:
            state = torch.load(
                suite.OUT / "readers" / f"{coord}_{task}_{seed}" / "readers.pt", weights_only=True
            )
            for arm, weights in state["models"].items():
                model = new_model(task, arm, suite.TR, seed, t["y"].shape[-1])
                model.load_state_dict(weights)
                model.eval()
                if arm == "anchor":
                    base = [forward(model, t, idx, arm, k) * state["sd"] + state["mu"] for k in [False, True]]
                for k, variant in enumerate(["aligned", "mismatched"]):
                    pred = (
                        base[k]
                        if arm == "anchor"
                        else base[k] + forward(model, t, idx, arm, bool(k)) * state["sd"]
                    )
                    for calibrated in (False, True):
                        value = pred
                        if calibrated:
                            a, b = state["calibration"][arm]
                            value = a + b * pred
                        rows.extend(
                            cell_scores(
                                meta,
                                t["y"].numpy(),
                                value.numpy(),
                                state["sd"].numpy(),
                                mod,
                                arm,
                                seed,
                                task=task,
                                coordinate="WG_cached",
                                input=variant,
                                calibrated=calibrated,
                            )
                        )
    cells = pd.DataFrame(rows)
    cells.to_csv(OUT / "wg_cells.csv.gz", index=False)
    comp = p1_tables(OUT, cells)
    support = pd.read_csv(OUT / "wg_support.csv")
    return dict(
        planned=len(support),
        events=len(meta),
        people=meta.subject.nunique(),
        people_new_to_nback_development=sorted(set(meta.subject) - set(source.subject)),
        status_counts=support.status.value_counts().to_dict(),
        primary=comp.query(
            "input=='aligned' and not calibrated and ((task=='A' and contrast=='over_Hb_capacity') or (task!='A' and contrast=='over_frozen_anchor'))"
        ).to_dict("records"),
    )


def atlas_inputs():
    audit.readout.panel()
    return audit.relations.maps()


def family(mod, b):
    m = suite.read(audit.OUT / f"{mod}_17" / f"{b}_family.json")["mapping"]
    return np.asarray(m)


def temporal_null(frame, mod):
    """Enumerate legal whole-window O rotations within person, position, record and observed condition."""
    joint = []
    coverage = []
    for keys, g in frame.groupby(["dataset", "subject", "record", "position", "condition"], dropna=False):
        g = g.sort_values("start_time")
        s = np.stack(g.s)
        o = np.stack(g.o)
        start = g.start_time.to_numpy()
        shifts = [k for k in range(1, len(g)) if np.all(np.abs(start - np.roll(start, k)) >= 30 - 1e-6)]
        info = dict(zip(["dataset", "subject", "record", "position", "condition"], keys))
        coverage.append(dict(info, windows=len(g), legal_shifts=len(shifts)))
        if not shifts:
            continue

        def counts(v):
            code = 4 * s + v
            return np.r_[
                np.bincount(code.ravel(), minlength=16),
                np.bincount((16 * code[:, :-1] + code[:, 1:]).ravel(), minlength=256),
            ]

        obs = counts(o)
        null = np.mean([counts(np.roll(o, k, axis=0)) for k in shifts], axis=0)
        for i in range(len(obs)):
            joint.append(
                dict(
                    info,
                    modality=mod,
                    kind="occupancy" if i < 16 else "transition",
                    cell=i if i < 16 else i - 16,
                    observed=int(obs[i]),
                    expected=null[i],
                    opportunities=len(g) * (15 if i < 16 else 14),
                )
            )
    return joint, coverage


@torch.inference_mode()
def spatial_temporal():
    rows, scales = atlas_inputs()
    raw = pd.read_csv(audit.OUT / "positions.csv", dtype={"subject": str}).query("split=='validation'")
    lookup = {(r["parent"], r["modality"]): i for i, r in enumerate(rows) if r["split"] == "validation"}
    # The source atlas used a CUDA batch of four neighboring sites. Reproduce
    # that arithmetic before assigning a word near a quantizer boundary.
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=pci.bus_id,uuid", "--format=csv,noheader"], text=True
    )
    permitted = [
        line.split(",")[1].strip()
        for line in inventory.splitlines()
        if line.split(",")[0].strip() == "00000000:4B:00.0"
    ]
    if len(permitted) != 1 or torch.cuda.is_initialized():
        raise ValueError("expected a permitted cold CUDA device")
    os.environ["CUDA_VISIBLE_DEVICES"] = permitted[0]
    torch.set_float32_matmul_precision("highest")
    samples = []
    sequences = []
    matches = []
    coverage = []
    joint = []
    original = pd.read_csv(audit.OUT / "paired_panel.csv").query("status=='complete'")
    dev = original.split.eq("validation").to_numpy()
    with np.load(audit.OUT / "paired_features.npz") as z:
        bins = {m + b: z[m + b][dev].reshape(-1, 3, 4) for m in ["eeg", "hb"] for b in "SO"}
    for mod in ["eeg", "hb"]:
        net = audit.network(mod, 8192 if mod == "eeg" else 1024, 17, "cuda:0")
        maps = [family(mod, b) for b in "SO"]
        entries = raw[raw["mod"] == mod].to_dict("records")
        cache = OrderedDict()
        for start in range(0, len(entries), 4):
            batch = entries[start : start + 4]
            ex = []
            for r in batch:
                idx = lookup[r["parent"], mod]
                src = rows[idx]
                if (r["dataset"], r["subject"], r["record"]) != (
                    src["dataset"],
                    src["subject"],
                    src["record"],
                ):
                    raise ValueError("site join mismatch")
                ex.append((idx, r["start"], src["positions"].index(r["position"])))
            v = base_tokenizer_batch(rows, ex, scales, array_cache=cache)
            if not v["mask"].all():
                raise ValueError("incomplete atlas site")
            x = v["x"]
            ids = net(torch.from_numpy(x.astype("float32")).to("cuda:0"))["ids"]
            seq = np.stack([ids[b].cpu().numpy() for b in "SO"], 1)
            fam = np.stack([m[seq[:, j]] for j, m in enumerate(maps)], 1)
            for j, b in enumerate("SO"):
                np.testing.assert_allclose(
                    np.eye(4)[fam[:, j]].reshape(4, 3, 5, 4).mean(2), bins[mod + b][start : start + 4]
                )
            observed = wave_attribute(x, mod)
            np.testing.assert_allclose(observed.mean(1), [r["observed"] for r in batch], atol=1e-6)
            np.testing.assert_allclose((fam[:, 0] == 3).mean(1), [r["Q4"] for r in batch])
            for i, r in enumerate(batch):
                attrs = dict(
                    dataset=r["dataset"],
                    subject=r["subject"],
                    record=r["record"],
                    position=r["position"],
                    condition=r["condition"] if pd.notna(r["condition"]) else "unlabelled",
                    start_time=r[mod + "_start"],
                    parent=r["parent"],
                    start=r["start"],
                )
                sequences.append(dict(attrs, mod=mod, s=fam[i, 0], o=fam[i, 1]))
                if mod == "eeg":
                    power = np.abs(np.fft.rfft(x[i], axis=-1)) ** 2
                    line = power[:, 98:103].sum(1) / np.maximum(power[:, 1:].sum(1), 1e-20)
                    scale = scales["positions"]["eeg"][r["dataset"]][r["position"]]["scale"]
                    amp = np.sqrt((x[i] ** 2).mean(1)) * scale
                    for j in range(15):
                        samples.append(
                            dict(
                                attrs,
                                back=r["y"] < 0,
                                q4=int(fam[i, 0, j] == 3),
                                alpha=observed[i, j],
                                amplitude=amp[j],
                                line_proxy=line[j],
                            )
                        )
                elif attrs["condition"] != "unlabelled":
                    deoxy = wave_attribute(x[i].reshape(15, 20, 2)[:, :, ::-1].reshape(15, 40), "hb")
                    for j in range(15):
                        matches.append(
                            dict(
                                attrs,
                                phase=j // 5,
                                rms_bin=np.digitize(np.sqrt((x[i, j] ** 2).mean()), [0.5, 1, 2]),
                                patch=j,
                                s_word=seq[i, 0, j],
                                o_word=seq[i, 1, j],
                                oxy=observed[i, j],
                                deoxy=deoxy[j],
                            )
                        )
        seqdf = pd.DataFrame([r for r in sequences if r["mod"] == mod])
        j, c = temporal_null(seqdf, mod)
        joint += j
        coverage += [dict(r, modality=mod) for r in c]
    pairs = []
    for (ds, sub), g in pd.DataFrame(samples).groupby(["dataset", "subject"]):
        ix = ["record", "condition", "amp_bin", "quality_bin"]
        g = g.assign(
            amp_bin=pd.qcut(g.amplitude, 3, labels=False, duplicates="drop"),
            quality_bin=pd.qcut(g.line_proxy, 2, labels=False, duplicates="drop"),
        )
        base = g.groupby(["back", "position"])[["q4", "alpha"]].mean().groupby("back").mean().diff().iloc[-1]
        means = g.groupby(ix + ["back"])[["q4", "alpha"]].mean().unstack().dropna()
        count = g.groupby(ix + ["back"]).size().unstack().loc[means.index]
        w = count.min(axis=1)
        pairs.append(
            dict(
                dataset=ds,
                subject=sub,
                patches=len(g),
                strata=len(w),
                paired_opportunities=int(w.sum()),
                q4=base.q4,
                alpha=base.alpha,
                adjusted_q4=np.average(means.q4[True] - means.q4[False], weights=w),
                adjusted_alpha=np.average(means.alpha[True] - means.alpha[False], weights=w),
            )
        )
    people = pd.DataFrame(pairs)
    people.to_csv(OUT / "spatial_people.csv", index=False)
    ss = []
    for ds, g in people.groupby("dataset"):
        for metric in ("q4", "alpha", "adjusted_q4", "adjusted_alpha"):
            val, lo, hi = interval(g[metric], g.dataset)
            ss.append(
                dict(
                    dataset=ds,
                    metric=metric,
                    mean=val,
                    ci_low=lo,
                    ci_high=hi,
                    people=len(g),
                    positive=int((g[metric] > 0).sum()),
                )
            )
    pd.DataFrame(ss).to_csv(OUT / "spatial_contrasts.csv", index=False)
    pd.DataFrame(coverage).to_csv(OUT / "sequence_coverage.csv", index=False)
    j = (
        pd.DataFrame(joint)
        .groupby(["dataset", "subject", "modality", "kind", "cell"])[
            ["observed", "expected", "opportunities"]
        ]
        .sum()
        .reset_index()
    )
    j["excess"] = (j.observed - j.expected) / j.opportunities
    j.to_csv(OUT / "joint_family_null.csv.gz", index=False)
    md = pd.DataFrame(matches)
    paired = []
    for b, word in [("s_word", 704), ("o_word", 866)]:
        for key, g in md.groupby(
            ["dataset", "subject", "record", "position", "condition", "phase", "rms_bin"]
        ):
            m = g[b].eq(word)
            n = int(m.sum())
            if n:
                paired.append(
                    dict(
                        zip(
                            ["dataset", "subject", "record", "position", "condition", "phase", "rms_bin"], key
                        ),
                        branch=b[0].upper(),
                        word=word,
                        members=n,
                        controls=int((~m).sum()),
                        member_oxy=g[m].oxy.median(),
                        member_deoxy=g[m].deoxy.median(),
                        control_oxy=g[~m].oxy.median(),
                        control_deoxy=g[~m].deoxy.median(),
                    )
                )
    pd.DataFrame(paired).to_csv(OUT / "hb_condition_matched.csv", index=False)
    return dict(
        windows=raw[["parent", "start"]].drop_duplicates().shape[0],
        people=len(people),
        matched_effect_rows=len(paired),
        joint_subject_cells=len(j),
    )


def vocabulary():
    """Inventory both entire Hb dictionaries; naming direction is determined on train only."""
    rows, scales = atlas_inputs()
    people = []
    cache = OrderedDict()
    for role in ["train", "validation"]:
        parts = []
        for task in audit.tasks(role):
            mod, ds, _, lo, hi = task
            if mod != "hb":
                continue
            with np.load(audit.inputs(task)) as z:
                ex = z["examples"]
                valid = z["valid"]
                ids = [z[b + "_17_id"] for b in "SO"]
                original = z["y"][:, 0]
            expected = audit.readout.capacity.PANEL["examples"][mod, role, ds][lo:hi]
            np.testing.assert_array_equal(ex, expected)
            sub = []
            scores = []
            for i in range(0, len(ex), 96):
                block = ex[i : i + 96]
                mask = valid[i : i + len(block)]
                if any(rows[k]["dataset"] != ds or rows[k]["split"] != role for k in block[:, 0]):
                    raise ValueError("shard scope drift")
                x = base_tokenizer_batch(rows, block, scales, array_cache=cache)
                np.testing.assert_array_equal(mask, x["mask"].all(-1))
                v = x["x"][mask]
                oxy = wave_attribute(v, "hb")
                deoxy = wave_attribute(v.reshape(-1, 20, 2)[:, :, ::-1].reshape(-1, 40), "hb")
                scores.append(np.stack([oxy, deoxy], 1))
                sub.extend(np.repeat([rows[k]["subject"] for k in block[:, 0]], mask.sum(1)))
            vals = np.concatenate(scores)
            np.testing.assert_allclose(vals[:, 0], original, atol=2e-5, rtol=1e-5)
            parts.append(
                pd.DataFrame(
                    dict(dataset=ds, subject=sub, S=ids[0], O=ids[1], oxy=vals[:, 0], deoxy=vals[:, 1])
                )
            )
        d = pd.concat(parts, ignore_index=True)
        for b in "SO":
            g = (
                d.groupby([b, "dataset", "subject"])
                .agg(n=("oxy", "size"), oxy=("oxy", "median"), deoxy=("deoxy", "median"))
                .reset_index()
                .rename(columns={b: "word"})
            )
            people.append(g.assign(branch=b, split=role))
    frame = pd.concat(people, ignore_index=True)
    frame.to_csv(OUT / "hb_word_subjects.csv.gz", index=False)
    words = []
    for b in "SO":
        for word in range(1024):
            g = frame[(frame.branch == b) & (frame.word == word) & (frame.n >= 3)]
            tr = g[g.split == "train"]
            dv = g[g.split == "validation"]
            supported = len(tr) >= 3 and len(dv) >= 3

            # Ties or exact zeros count as no majority; zero is not a positive or negative slope.
            def labels(v):
                return np.where(
                    (v.oxy == 0) | (v.deoxy == 0), -1, (v.oxy > 0).astype(int) * 2 + (v.deoxy > 0).astype(int)
                )

            a = labels(tr)
            z = labels(dv)
            votes = np.bincount(a[a >= 0], minlength=4)
            q = int(votes.argmax()) if votes.max() > len(tr) / 2 else -1
            same = int(np.sum(z == q)) if q >= 0 else 0
            words.append(
                dict(
                    branch=b,
                    word=word,
                    train_people=len(tr),
                    dev_people=len(dv),
                    supported=supported,
                    train_quadrant=q,
                    dev_same=same,
                    replicate_majority=supported and q >= 0 and same > len(dv) / 2,
                    replicate_all=supported and q >= 0 and same == len(dv),
                )
            )
    tab = pd.DataFrame(words)
    tab.to_csv(OUT / "hb_dictionary_replication.csv", index=False)
    return (
        tab.groupby("branch")[["supported", "replicate_majority", "replicate_all"]]
        .sum()
        .astype(int)
        .to_dict("index")
    )


@torch.inference_mode()
def time_reencode():
    # Replace one real EEG interval at all four sites using the registered same-condition donor.
    from src.foundation.four_stream_relations import check_history_support

    path, m, data = suite.load_parent("native")
    dev = np.flatnonzero(m.split == "validation")
    meta = m.iloc[dev].reset_index(drop=True)
    donor = np.searchsorted(dev, meta.donor.to_numpy())
    np.testing.assert_array_equal(dev[donor], meta.donor)
    waves = {}
    for sub in meta.subject.unique():
        r = path / "records"
        ledger = [v["id"] for v in suite.read(r / (sub + ".json"))["ledger"] if v["status"] == "complete"]
        with np.load(r / (sub + ".npz")) as z:
            waves.update(zip(ledger, z["eeg_wave"], strict=True))
            check_history_support(
                torch.from_numpy(z["encoder_start"]),
                torch.from_numpy(z["raw_stop"]),
                torch.full((len(ledger),), 4.0, dtype=torch.float64),
                torch.from_numpy(z["hb_target_start"]),
                torch.from_numpy(z["hb_target_start"]) + 2,
            )
    scale = suite.read(G0 / "scales.json")["positions"]["eeg"][DS]
    x = (
        np.stack([waves[k] for k in meta.id])
        / np.array([scale[s]["scale"] for s in SITES["eeg"]])[None, :, None, None]
    ).astype("float32")
    t = setup(data, m, "A")
    net = audit.network("eeg", 8192, 17)
    models = {}
    for seed in suite.TR["seeds"]:
        state = torch.load(suite.OUT / "readers" / f"native_A_{seed}" / "readers.pt", weights_only=True)
        for arm in ["anchor", "true"]:
            mdl = new_model("A", arm, suite.TR, seed, 2)
            mdl.load_state_dict(state["models"][arm])
            models[seed, arm] = mdl.eval()
        models[seed, "sd"] = state["sd"]
        models[seed, "base"] = forward(models[seed, "anchor"], t, dev, "anchor") * state["sd"] + state["mu"]
        models[seed, "original"] = (
            models[seed, "base"] + forward(models[seed, "true"], t, dev, "true") * state["sd"]
        )
    results = []
    for slot in range(-1, 15):
        changed = x.copy()
        if slot >= 0:
            changed[:, :, slot] = x[donor, :, slot]
        ids = []
        for i in range(0, len(meta), 8):
            vals = net(torch.from_numpy(np.ascontiguousarray(changed[i : i + 8].reshape(-1, 15, 400))))["ids"]
            ids.append(torch.stack([vals[b] for b in "SO"], 1).reshape(-1, 4, 2, 15).numpy())
        code = np.concatenate(ids)
        initial = data["eeg_ids"][dev].transpose(0, 2, 1, 3)
        if slot < 0:
            np.testing.assert_array_equal(code, initial)
            continue
        for j, b in enumerate("SO"):
            t["v"]["eeg"][dev, :, j] = torch.from_numpy(data["eeg" + b + "_book"][code[:, :, j]])
        changed_slot = code != initial
        for seed in suite.TR["seeds"]:
            sd = models[seed, "sd"]
            pred = models[seed, "base"] + forward(models[seed, "true"], t, dev, "true") * sd
            orig = models[seed, "original"]
            y = t["y"][dev]
            loss = (((pred - y) / sd).square() - ((orig - y) / sd).square()).numpy()
            shift = ((pred - orig) / sd).square().numpy()
            for who, idx in meta.groupby("subject").indices.items():
                for lag in range(3):
                    results.append(
                        dict(
                            subject=who,
                            seed=seed,
                            slot=slot,
                            start_before_cutoff_s=-30 + 2 * slot,
                            target_lag_s=[2, 4, 6][lag],
                            delta_nmse=loss[idx, :, lag].mean(),
                            prediction_rms_train_sd=np.sqrt(shift[idx, :, lag].mean()),
                            other_slot_id_change=changed_slot[idx][:, :, :, np.arange(15) != slot].mean(),
                        )
                    )
    pd.DataFrame(results).to_csv(OUT / "real_interval_reencode.csv", index=False)
    return dict(events=len(meta), perturbation_windows=15 * len(meta), baseline_id_replay=True)


def part(name):
    f = OUT / (name + "_complete.json")
    if f.exists():
        return suite.read(f)
    methods = dict(atlas=spatial_temporal, vocabulary=vocabulary, wg=transfer, reencode=time_reencode)
    result = methods[name]()
    suite.save(f, dict(status="completed", result=result))
    return result


def evaluate():
    if suite.read(suite.OUT / "p1.json")["status"] != "completed":
        raise ValueError("P1 incomplete")
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("fork")) as pool:
        list(pool.map(part, ["wg", "atlas", "vocabulary", "reencode"]))
    suite.save(
        suite.OUT / "p2.json",
        dict(
            status="completed",
            components={
                k: suite.read(OUT / (k + "_complete.json"))["result"]
                for k in ["wg", "atlas", "vocabulary", "reencode"]
            },
        ),
    )
