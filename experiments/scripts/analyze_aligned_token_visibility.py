"""Disentangle missing-ID and raw-center visibility; measure controlled, directional ID edits."""

from collections import OrderedDict, defaultdict
import numpy as np
import pandas as pd
import torch
import audit_aligned_tokenizer as audit
from src.analysis.single_vq_relations import wave_attribute
from src.analysis.frozen_token_readout import ResidualRidge, Means
from src.data.dual_component_tokenizer import base_tokenizer_batch
from analyze_aligned_token_controls import save, words
from single_vq_final_c_probes import context_pairs


def native_errors(mod, rows, scales):
    d = pd.read_csv(audit.OUT / f"{mod}_wave_errors.csv")
    refs = []
    for (i, p), g in d.groupby(["row", "position"]):
        row = rows[i]
        scale = np.atleast_1d(scales["positions"][mod][row["dataset"]][p]["scale"])
        for color in range(1 if mod == "eeg" else 2):
            refs.append(
                dict(row=i, position=p, color=color, native_scale=float(scale[min(len(scale) - 1, color)]))
            )
    d = d.merge(pd.DataFrame(refs), on=["row", "position", "color"], validate="many_to_one")
    d["native_error"] = d.error * d.native_scale**2
    d["native_target"] = d.target * d.native_scale**2
    d.to_csv(audit.OUT / f"{mod}_native_errors.csv", index=False)


@torch.inference_mode()
def run(mod):
    out = audit.OUT
    rows, scales = audit.relations.maps()
    native_errors(mod, rows, scales)
    net = audit.network(mod, audit.PLAN["models"][mod]["K"], 17, "cuda:0")
    seq = audit.collect(mod, 17, "validation")
    pairs, _ = context_pairs(seq["examples"], seq["valid"], rows)
    maps = {b: words(mod, b) for b in "SO"}
    cache = OrderedDict()
    rng = np.random.default_rng(1126)
    vis = []
    edit = []
    for chunk in np.array_split(pairs, int(np.ceil(len(pairs) / 48))):
        batches = [
            base_tokenizer_batch(rows, seq["examples"][ids], scales, array_cache=cache) for ids in chunk.T
        ]
        x = batches[0]["x"]
        donor = batches[1]["x"]
        alt = x.copy()
        alt[:, 7] = donor[:, 7]
        original, replaced, source = [net(torch.as_tensor(v, device="cuda:0")) for v in (x, alt, donor)]
        for b in "SO":
            ids = original["ids"][b].cpu().numpy()
            altid = replaced["ids"][b].cpu().numpy()
            other = source["ids"][b].cpu().numpy()
            word = maps[b]
            fam = word.family.to_numpy()
            freq = word.train_patches.to_numpy()
            score = word.family_score.to_numpy()
            base = wave_attribute(original[b][:, 7].cpu().numpy(), mod)
            rowmeta = []
            for i, (r, t, p) in enumerate(seq["examples"][chunk[:, 0]]):
                meta = dict(
                    dataset=rows[r]["dataset"],
                    subject=rows[r]["subject"],
                    row=r,
                    start=t,
                    position=rows[r]["positions"][p],
                    branch=b,
                )
                rowmeta.append(meta)
                # ID mask applied AFTER encoding leaves each of the 14 visible neighbors unchanged, by construction.
                for offset in range(1, 8):
                    k = [7 - offset, 7 + offset]
                    u = ids[i, k]
                    v = altid[i, k]
                    vis.append(
                        dict(
                            **meta,
                            distance_s=2 * offset,
                            raw_changed=np.mean(u != v),
                            raw_family_changed=np.mean(fam[u] != fam[v]),
                            idmask_visible_changed=0,
                        )
                    )
            choices = defaultdict(list)
            for i, center in enumerate(ids[:, 7]):
                pool = np.unique(other[i])
                pool = pool[
                    (pool != center)
                    & (freq[pool] > 0)
                    & (abs(np.log((freq[pool] + 1) / (freq[center] + 1))) <= np.log(2))
                ]
                if len(pool) < 2:
                    continue
                vals = score[pool]
                candidates = {
                    "highest": pool[np.argmax(vals)],
                    "lowest": pool[np.argmin(vals)],
                    "random": rng.choice(pool),
                }
                if vals.max() <= vals.min():
                    continue
                for name, token in candidates.items():
                    choices[name].append((i, token))
            for name, choice in choices.items():
                ii, token = np.array(choice).T
                stream = {k: v.clone() for k, v in original["ids"].items()}
                stream[b][ii, 7] = torch.as_tensor(token, device="cuda:0")
                v = net.decode_ids(stream)[b].cpu().numpy()
                attr = wave_attribute(v[ii, 7], mod)
                d = v[ii] - original[b].cpu().numpy()[ii]
                for k, i in enumerate(ii):
                    edit.append(
                        dict(
                            **rowmeta[i],
                            kind=name,
                            word=ids[i, 7],
                            new=token[k],
                            trained_delta=score[token[k]] - score[ids[i, 7]],
                            decoded_delta=attr[k] - base[i],
                            outside_rms=np.sqrt(np.mean(np.delete(d[k], 7, axis=0) ** 2)),
                        )
                    )
    save(vis, mod + "_visibility")
    save(edit, mod + "_directed_edit")
    panel = pd.read_csv(out / f"{mod}_local_panel.csv")
    blocks = defaultdict(list)
    for chunk in np.array_split(np.arange(len(panel)), int(np.ceil(len(panel) / 48))):
        g = panel.iloc[chunk]
        ex = [(r.row, r.start, rows[r.row]["positions"].index(r.position)) for r in g.itertuples()]
        raw = base_tokenizer_batch(rows, ex, scales, array_cache=cache)
        x = raw["x"]
        res = net(torch.as_tensor(x, device="cuda:0"))
        blocks["rms"].extend(np.sqrt((x[:, 7] ** 2).mean(1)))

        def attributes(w):
            if mod == "hb":
                a = w.reshape(-1, 20, 2)
                return ["HbO_rms", "HbR_rms"], np.sqrt((a * a).mean(1))
            f = np.fft.rfftfreq(400, 1 / 200)
            ok = ((f >= 2) & (f <= 5)) | ((f >= 15) & (f <= 30))

            def amp(x):
                return np.sqrt((x * x).mean(-1))

            z = w - w.mean(1, keepdims=True)
            taper = np.hanning(400)
            p = abs(np.fft.rfft(z * taper)) ** 2 * 2 / (400 * (taper @ taper))
            xx = np.c_[np.ones(ok.sum()), np.log10(f[ok])]
            y = np.log10(np.maximum(p[:, ok], 1e-15)).T
            co = np.linalg.lstsq(xx, y, rcond=None)[0]
            r2 = 1 - ((xx @ co - y) ** 2).sum(0) / np.maximum(((y - y.mean(0)) ** 2).sum(0), 1e-15)
            return ["offset", "slope", "fit_R2", "RMS", *[f"env{i}" for i in range(8)]], np.column_stack(
                (co.T, r2, amp(w), amp(w.reshape(-1, 8, 50)))
            )

        for b in "SO":
            names, y = attributes(raw[b.lower()][:, 7])
            blocks[b + "y"].extend(y)
            blocks[b + "id"].extend(res["ids"][b][:, 7].cpu().numpy())
            blocks[b + "z"].extend(res["latent"][b][:, 7].cpu().numpy())
    bg = pd.MultiIndex.from_arrays(
        [panel.dataset, panel.position, np.digitize(blocks["rms"], [0.5, 1, 2])]
    ).factorize()[0]
    tr = panel.role.eq("train").to_numpy()
    results = []
    for b in "SO":
        y = np.array(blocks[b + "y"])
        ids = np.array(blocks[b + "id"])
        book = net.branches[b]["quantizer"].codebook.cpu().numpy()
        mean = Means(bg[tr], y[tr])
        ref = mean.lookup(bg[~tr])[0]
        for name, X in [
            ("latent", np.array(blocks[b + "z"])),
            ("embedding", book[ids]),
            ("random_geometry", book[rng.permutation(len(book))][ids]),
        ]:
            rd = ResidualRidge(bg[tr], X[tr], y[tr], mean)
            pred = rd.predict(bg[~tr], X[~tr], rd.coefficients(0.01))
            target = y[~tr]
            err = (pred - target) ** 2
            base = (ref - target) ** 2
            for (ds, sub), idx in (
                panel[~tr].reset_index(drop=True).groupby(["dataset", "subject"]).groups.items()
            ):
                for j, attr in enumerate(names):
                    results.append(
                        dict(
                            dataset=ds,
                            subject=sub,
                            branch=b,
                            method=name,
                            attr=attr,
                            mse=err[idx, j].mean(),
                            baseline=base[idx, j].mean(),
                        )
                    )
    save(results, mod + "_envelope_probes")


@torch.inference_mode()
def paired_example():
    out = audit.OUT
    meta = pd.read_csv(out / "paired_panel.csv").query("status=='complete'").reset_index(drop=True)
    feat = np.load(out / "paired_features.npz")
    eligible = np.flatnonzero((meta.split == "validation") & (meta.condition == "MA"))
    observed = feat["eeg_y"][eligible].mean((1, 2))
    i = eligible[np.argmin(abs(observed - np.median(observed)))]
    s = meta.iloc[i]
    rows, scales = audit.relations.maps()
    arrays = {}
    led = []
    names = pd.read_csv(out / "positions.csv")
    for m in ["eeg", "hb"]:
        row = next(j for j, r in enumerate(rows) if r["parent"] == s.parent and r["modality"] == m)
        pos = names[
            (names.parent == s.parent) & (names.start == s.start) & (names["mod"] == m)
        ].position.to_list()
        site = pos[0]
        ex = [(row, int(s.start), rows[row]["positions"].index(site))]
        raw = base_tokenizer_batch(rows, ex, scales)
        assert raw["mask"].all()
        model = audit.network(m, audit.PLAN["models"][m]["K"], 17, "cpu")
        res = model(torch.as_tensor(raw["x"]))
        for b in "SO":
            a = res["ids"][b][0].numpy()
            arrays[m + b + "_id"] = a
            arrays[m + b + "_family"] = words(m, b).family.to_numpy()[a]
        for b, key in zip("XSO", ["sum", "S", "O"]):
            arrays[m + b] = raw[b.lower()][0]
            arrays[m + b + "_decoded"] = res[key][0].numpy()
        led.append(
            dict(
                mod=m,
                position=site,
                dataset=s.dataset,
                subject=s.subject,
                record=s.record,
                event=int(s.event),
                eeg_start=s.eeg_start,
                hb_start=s.hb_start,
                eeg_cue=s.cue,
                hb_cue=s.hb_cue,
                eligible=len(eligible),
                median_observed=np.median(observed),
                selected_observed=feat["eeg_y"][i].mean(),
            )
        )
    save(led, "paired_example")
    np.savez_compressed(out / "paired_example.npz", **arrays)


if __name__ == "__main__":
    torch.set_num_threads(1)
    audit.readout.panel()
    paired_example()
