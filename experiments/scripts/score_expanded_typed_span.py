"""Terminal replay of actual ordered IDs, paired fragments and typed attention."""

from pathlib import Path
import math

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

import run_expanded_typed_span as run
from src.analysis.typed_span_patterns import (
    comparisons,
    exact_orders,
    transfer_selected,
    family_ids,
    lag_profile,
    pattern_fields,
)
from src.foundation.typed_span import LagPath
from src.utils.io import save_npz_atomic

base, OUT = run.base, run.OUT
PATHS = ["S→S", "S→O", "O→S", "O→O"]


def known_delay():
    """Independent positive control on the attention measurement with asymmetric clocks and a null key."""
    rng = torch.Generator().manual_seed(803)
    out = []
    for planted in (0, 2, 4):
        link = LagPath(16, [-6, 12], False)
        with torch.no_grad():
            for layer in (link.query, link.key):
                layer.weight.copy_(torch.eye(16))
        e = torch.randn(96, 1, 15, 16, generator=rng) * 2
        h = torch.randn(e.shape, generator=rng)
        h[:, :, planted:] = e[:, :, : 15 - planted] + 0.03 * torch.randn(
            h[:, :, planted:].shape, generator=rng
        )
        mask = torch.ones(h.shape[:3], dtype=torch.bool)
        mask[:, :, :planted] = False
        for name, s in (("true", e), ("wrong", e.roll(1, 0))):
            w = link.attention(h, s, True, False)
            mass, opp = lag_profile(w, mask, True)
            density = (mass.sum(0) / opp.sum(0).clamp_min(1)).detach().numpy()
            out.append(
                dict(
                    planted_s=planted * 2,
                    control=name,
                    peak_s=2 * (int(np.argmax(density)) - 14),
                    mass_at_planted=float(mass[:, 14 + planted].sum().detach() / mask.sum()),
                )
            )
    if any(
        r["peak_s"] != r["planted_s"] or r["mass_at_planted"] < 0.8 for r in out if r["control"] == "true"
    ):
        raise ValueError("synthetic lag orientation/recovery failed")
    return out


@torch.inference_mode()
def dictionary(model, data):
    m = data["m"]
    ids = []
    for ix in np.array_split(np.arange(len(m)), math.ceil(len(m) / 12)):
        empty = torch.zeros((len(ix), 4, 15), dtype=torch.bool, device="cuda")
        streams = [v for mod in (0, 1) for v in base.codes(model.nets[mod], data, mod, ix, empty)[1]]
        ids.append(torch.stack(streams, 1).cpu().numpy().astype("uint16"))
    books = [net.branches[b]["quantizer"].codebook.cpu().numpy() for net in model.nets for b in "SO"]
    return np.concatenate(ids), books


def lookup(ids, books, ix, mod):
    return [
        F.embedding(torch.as_tensor(ids[ix, mod * 2 + j].astype("int64"), device="cuda"), books[mod * 2 + j])
        for j in (0, 1)
    ]


@torch.inference_mode()
def effects(model, data, ids):
    m = data["m"]
    dev = data["dev"]
    dev = dev[m.donor.to_numpy()[dev] >= 0]
    books = [net.branches[b]["quantizer"].codebook for net in model.nets for b in "SO"]
    rows = []
    profiles = []
    shifts = []
    for ix in np.array_split(dev, math.ceil(len(dev) / 12)):
        donor = m.donor.to_numpy()[ix]
        where = m.iloc[ix]
        for mod in (0, 1):
            true = lookup(ids, books, ix, 1 - mod)
            wrong = lookup(ids, books, donor, 1 - mod)
            for length in [base.P["mask"]["span_slots_eeg_hb"][mod], 15]:
                mask = base.score_mask(ix, mod, length, "cuda")
                tv = base.codes(model.nets[mod], data, mod, ix, mask)[0]
                control = [
                    ("true", true, (0, 1)),
                    ("wrong", wrong, (0, 1)),
                    ("reverse", [v.flip(2) for v in true], (0, 1)),
                    ("none", true, ()),
                    ("S", true, (0,)),
                    ("O", true, (1,)),
                ]
                if length == 15:
                    # Delay the source token contents by two slots. Compare identical
                    # query/key supports and discard the two wrapped positions in both views.
                    shifted = [v.roll(2, 2) for v in true]
                    _, wa = model.reader(true, tv, data["xy"][ix], mask, mod, (0, 1), model.arm, inspect=True)
                    _, wb = model.reader(
                        shifted, tv, data["xy"][ix], mask, mod, (0, 1), model.arm, inspect=True
                    )
                    q = torch.arange(5, 10, device="cuda")[:, None]
                    k = torch.arange(2, 15, device="cuda")[None, :]
                    lag = 2 * (q - k) if mod else 2 * (k - q)
                    for path in wa:

                        def center(w):
                            real = w[:, :, 5:10, 2:15]
                            return ((real * lag).sum(-1) / real.sum(-1).clamp_min(1e-12)).mean((1, 2))

                        delta = (center(wb[path]) - center(wa[path])).cpu().numpy()
                        for kidx, d in enumerate(delta):
                            shifts.append(
                                dict(
                                    task=where.task.iloc[kidx],
                                    subject=where.subject.iloc[kidx],
                                    event=ix[kidx],
                                    target=mod,
                                    path=path,
                                    centroid_change_s=d,
                                    expected_s=-4 if mod else 4,
                                )
                            )
                for name, sv, eligible in control:
                    scores, weights = model.reader(
                        sv,
                        tv,
                        data["xy"][ix],
                        mask,
                        mod,
                        eligible,
                        model.arm,
                        inspect=True,
                        source_xy=data["xy"][donor if name == "wrong" else ix, 1 - mod],
                    )
                    for j, z in enumerate(scores):
                        stream = 2 * mod + j
                        label = torch.as_tensor(ids[ix, stream].astype("int64"), device="cuda")
                        bits = F.cross_entropy(z[mask], label[mask], reduction="none").reshape(
                            len(ix), -1
                        ).mean(1).cpu().numpy() / math.log(2)
                        for k, value in enumerate(bits):
                            rows.append(
                                dict(
                                    task=where.task.iloc[k],
                                    subject=where.subject.iloc[k],
                                    event=ix[k],
                                    stream=stream,
                                    slots=length,
                                    control=name,
                                    bits=value,
                                )
                            )
                    if name not in ("true", "wrong", "reverse"):
                        continue
                    for path, w in weights.items():
                        variants = [(name, w)]
                        if name == "true":
                            zeros = sv[0].new_zeros((*mask.shape, model.reader.mask.shape[-1]))
                            link = model.reader.paths[path]
                            variants.append(
                                ("kernel", link.attention(zeros, zeros, bool(mod), model.arm != "no_lag"))
                            )
                        for kind, v in variants:
                            mass, opp = lag_profile(v, mask, bool(mod))
                            n = mask.sum((1, 2))
                            result = (
                                torch.cat(
                                    [
                                        mass / n[:, None],
                                        opp / n[:, None],
                                        ((v[..., -1] * mask).sum((1, 2)) / n)[:, None],
                                    ],
                                    1,
                                )
                                .cpu()
                                .numpy()
                            )
                            for k, vec in enumerate(result):
                                profiles.append(
                                    (
                                        where.task.iloc[k],
                                        where.subject.iloc[k],
                                        mod,
                                        length,
                                        path,
                                        kind,
                                        *vec.tolist(),
                                    )
                                )
    cols = (
        ["task", "subject", "target", "slots", "path", "control"]
        + [f"m{k}" for k in range(-14, 15)]
        + [f"n{k}" for k in range(-14, 15)]
        + ["null"]
    )
    profiles = pd.DataFrame.from_records(profiles, columns=cols).groupby(cols[:6]).mean().reset_index()
    if not np.allclose(profiles[[f"m{k}" for k in range(-14, 15)]].sum(1) + profiles.null, 1, atol=3e-6):
        raise ValueError("lost attention mass")
    return pd.DataFrame(rows), profiles, pd.DataFrame(shifts)


@torch.inference_mode()
def export_examples(model, data, ids, selected, members, folder):
    from src.foundation.typed_span import observed_view

    if members.empty:
        return
    examples = []
    arrays = {}
    for (task, path), g in members.groupby(["task", "path"]):
        # One first matching held event per person per train-selected candidate.
        for member in g.groupby("subject", sort=True).head(1).itertuples():
            ix = member.event_index
            site = member.site
            spec = selected[(selected.task == task) & (selected.path == path)].iloc[0]
            key = f"member_{len(examples)}"
            meta = data["m"].iloc[ix]
            examples.append(
                dict(
                    key=key,
                    id=meta.id,
                    task=task,
                    subject=meta.subject,
                    record=meta.record,
                    eeg_start_s=meta.eeg_start,
                    hb_start_s=meta.hb_start,
                    site=site,
                    path=path,
                    word=member.word,
                    eeg_slot=member.eeg_slot,
                    hb_slot=member.hb_slot,
                    lag_s=spec.lag_seconds,
                    eeg_site=run.CFG["sites"][task][0][site],
                    hb_site=run.CFG["sites"][task][1][site],
                )
            )
            arrays[key + "_ids"] = ids[ix, :, site]
            for mod in (0, 1):
                mask = torch.zeros(1, 4, 15, device="cuda", dtype=torch.bool)
                obs = observed_view(data["raw"][mod][ix : ix + 1], mask, mod, data["gain"])
                decoded = model.nets[mod].decode_ids(
                    {
                        b: torch.tensor(ids[ix : ix + 1, 2 * mod + j, site].astype("int64"), device="cuda")
                        for j, b in enumerate("SO")
                    }
                )
                arrays[key + f"_obs{mod}"] = obs[0, site].cpu().numpy()
                for b in "SO":
                    arrays[key + f"_{mod}{b}"] = decoded[b][0].cpu().numpy()
    if examples:
        valid = pd.DataFrame(examples)
        sel = selected.assign(
            priority=lambda x: np.minimum(x.train_true - x.train_wrong, x.train_true - x.train_reverse)
        ).sort_values("priority", ascending=False)
        chosen = next(
            row for row in sel.itertuples() if ((valid.task == row.task) & (valid.path == row.path)).any()
        )
        item = valid[(valid.task == chosen.task) & (valid.path == chosen.path)].iloc[0]
        idx = np.flatnonzero(data["m"].id == item.id)
        other = data["m"].donor.to_numpy()[idx]
        empty = torch.zeros(1, 4, 15, dtype=torch.bool, device="cuda")
        mask = ~empty
        tgt = base.codes(model.nets[1], data, 1, idx, mask)[0]
        maps = {}
        for key, which in (("true", idx), ("wrong", other)):
            src = base.codes(model.nets[0], data, 0, which, empty)[0]
            _, w = model.reader(
                src,
                tgt,
                data["xy"][idx],
                mask,
                1,
                (0, 1),
                model.arm,
                inspect=True,
                source_xy=data["xy"][which, 0],
            )
            maps[key] = w[int(item.path)][0, int(item.site)].cpu().numpy()
        zeros = tgt[0].new_zeros(1, 1, 15, model.reader.mask.shape[-1])
        maps["kernel"] = (
            model.reader.paths[int(item.path)].attention(zeros, zeros, True, True)[0, 0].cpu().numpy()
        )
        save_npz_atomic(
            folder / "example_attention.npz",
            event=np.asarray(item.id),
            donor=np.asarray(data["m"].id.iloc[other[0]]),
            path_index=np.asarray(item.path),
            site=np.asarray(item.site),
            **maps,
        )
    pd.DataFrame(examples).to_csv(folder / "examples.csv", index=False)
    save_npz_atomic(folder / "examples.npz", **arrays)


def score(arm, seed):
    folder = OUT / "fits" / f"{arm}_{seed}"
    with run.owner_lock(folder, "scoring"):
        if (folder / "scored.json").exists():
            return
        fit = base.read(folder / "done.json")
        if fit["steps"] != run.TR["steps"] or fit["unique_exposed"] != fit["train_windows"]:
            raise ValueError("incomplete exposure")
        model = base.load_model(arm, seed, "cuda").eval()
        model.load_state_dict(torch.load(folder / "weights.pt", weights_only=True, map_location="cpu"))
        data, anchors = run.load("cuda")
        ids, books = dictionary(model, data)
        m = data["m"]
        save_npz_atomic(folder / "tokens.npz", ids=ids, events=m.id.to_numpy(dtype=str))
        if arm == "joint":
            fam, maps = family_ids(ids, books, data["train"], seed)
            save_npz_atomic(folder / "families.npz", **{f"stream{i}": v for i, v in enumerate(maps)})
            exact_orders(ids, m).to_csv(folder / "exact_orders.csv", index=False)
            sel, people, members = comparisons(fam, m)
            sel.to_csv(folder / "selected.csv", index=False)
            people.to_csv(folder / "motifs_by_person.csv", index=False)
            members.to_csv(folder / "members.csv", index=False)
            export_examples(model, data, ids, sel, members, folder)
            if seed == 17:
                # Same events and scales before joint updates; label alignment sees only training subjects.
                parent_ids, parent_books = dictionary(base.load_model("joint", seed, "cuda").eval(), data)
                before, _ = family_ids(parent_ids, parent_books, data["train"], seed)
                exact_orders(parent_ids, m).to_csv(folder / "warm_start_exact.csv", index=False)
                replay, mapping = transfer_selected(fam, before, sel, m)
                replay.to_csv(folder / "warm_start_motifs.csv", index=False)
                base.save(folder / "warm_start_alignment.json", mapping)
        scores, profiles, shifts = effects(model, data, ids)
        shifts.groupby(["task", "subject", "target", "path"])[
            ["centroid_change_s", "expected_s"]
        ].mean().to_csv(folder / "shift_people.csv")
        scores.to_csv(folder / "masked.csv.gz", index=False)
        profiles.to_csv(folder / "attention.csv.gz", index=False)
        guard = []
        with torch.inference_mode():
            for task, ad, stats, am, _ in anchors:
                dev = np.flatnonzero(am.split == "validation")
                for ix in np.array_split(dev, max(1, math.ceil(len(dev) / 12))):
                    if not len(ix):
                        continue
                    _, nmse = base.anchor_loss(model, ad, ix, stats, False)
                    for j, values in zip(ix, nmse.cpu().tolist()):
                        guard.append(
                            dict(
                                task=task,
                                subject=am.subject.iloc[j],
                                **dict(zip(["eeg_X", "eeg_S", "eeg_O", "hb_X", "hb_S", "hb_O"], values)),
                            )
                        )
        pd.DataFrame(guard).groupby(["task", "subject"]).mean().to_csv(folder / "anchor_nmse.csv")
        base.save(
            folder / "scored.json",
            dict(
                status="completed",
                events=len(m),
                coded_shape=list(ids.shape),
                paired_dev=scores.event.nunique(),
                seed=seed,
                arm=arm,
            ),
        )


@torch.inference_mode()
def paired_attention(arm, seed):
    folder = OUT / "fits" / f"{arm}_{seed}"
    with run.owner_lock(folder, "paired_attention"):
        model = base.load_model(arm, seed, "cuda").eval()
        model.load_state_dict(torch.load(folder / "weights.pt", weights_only=True, map_location="cpu"))
        data, _ = run.load("cuda")
        m = data["m"]
        with np.load(folder / "tokens.npz") as z:
            np.testing.assert_array_equal(z["events"], m.id)
            ids = z["ids"]
        books = [n.branches[b]["quantizer"].codebook for n in model.nets for b in "SO"]
        dev = np.flatnonzero((m.split == "validation") & (m.donor >= 0))
        rows = []
        for ix in np.array_split(dev, math.ceil(len(dev) / 12)):
            donor = m.donor.to_numpy()[ix]
            for mod in (0, 1):
                for slots in (base.P["mask"]["span_slots_eeg_hb"][mod], 15):
                    mask = base.score_mask(ix, mod, slots, "cuda")
                    tgt = base.codes(model.nets[mod], data, mod, ix, mask)[0]
                    observed = []
                    for idx in (ix, donor):
                        _, w = model.reader(
                            lookup(ids, books, idx, 1 - mod),
                            tgt,
                            data["xy"][ix],
                            mask,
                            mod,
                            (0, 1),
                            model.arm,
                            inspect=True,
                            source_xy=data["xy"][idx, 1 - mod],
                        )
                        observed.append(w)
                    for path in range(4):
                        true, wrong = [v[path] for v in observed]
                        tv = ((true - wrong).abs().sum(-1) * 0.5)[mask].reshape(len(ix), -1).mean(1)
                        peaks = [lag_profile(v, mask, bool(mod))[0].argmax(1) for v in (true, wrong)]
                        for i, delta in enumerate(tv.cpu().numpy()):
                            rows.append(
                                dict(
                                    event=ix[i],
                                    subject=m.subject.iloc[ix[i]],
                                    task=m.task.iloc[ix[i]],
                                    slots=slots,
                                    target=mod,
                                    path=path,
                                    query_tv=delta,
                                    true_peak_s=2 * (peaks[0][i].item() - 14),
                                    wrong_peak_s=2 * (peaks[1][i].item() - 14),
                                )
                            )
        pd.DataFrame(rows).to_csv(folder / "paired_attention.csv.gz", index=False)


def summarize():
    from report_expanded_typed_span import report

    report()
