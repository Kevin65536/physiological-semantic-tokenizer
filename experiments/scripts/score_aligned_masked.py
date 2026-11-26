"""Fixed development evaluation on the common true-pair panel and the independent Hb-history panel."""

import argparse
from collections import Counter, defaultdict
import json
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, brier_score_loss
from train_aligned_masked import CFG, TR, OUT, PARENT, STREAMS, tensors, reader
from run_dual_component_tokenizer import identity, read_json, write_json, owner_lock, now
from src.foundation.aligned_masked import descriptors, hidden_sites
from src.analysis.single_vq_relations import fit_ridge, interval
import audit_aligned_tokenizer as audit


def load(books, arm):
    m = reader(books, arm)
    state = torch.load(OUT / arm / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert state["step"] == TR["steps"] and state["signature"] == identity([CFG, arm])
    m.load_state_dict(state["model"])
    for j, k in enumerate(m.streams):
        assert torch.equal(m.books[j].weight.cpu(), books[k])
    return m.eval()


class Priors:
    def __init__(self, panel, ids):
        self.total = Counter()
        self.cells = defaultdict(Counter)
        self.transitions = defaultdict(Counter)
        self.freq = defaultdict(Counter)
        for i, r in panel[panel.split.eq("train")].iterrows():
            for k, s in enumerate(CFG["streams"]):
                sites = r[s[:-1] + "_sites"].split("|")
                for p, site in enumerate(sites):
                    for t, target in enumerate(ids[i, k, p]):
                        y = int(target)
                        self.freq[k][y] += 1
                        self.total[k] += 1
                        self.cells[k, r.dataset, site, t][y] += 1
                        if t:
                            self.transitions[k, int(ids[i, k, p, t - 1])][y] += 1
                        if t < 14:
                            self.transitions[k, ~int(ids[i, k, p, t + 1])][y] += 1
        self.cell_total = {k: sum(v.values()) for k, v in self.cells.items()}
        self.edge_total = {k: sum(v.values()) for k, v in self.transitions.items()}

    def bits(self, r, k, seq, hidden):
        ans = []
        sites = r[CFG["streams"][k][:-1] + "_sites"].split("|")
        for p, t in zip(*np.where(hidden)):
            y = int(seq[p, t])
            marg = (self.freq[k][y] + 0.5) / (self.total[k] + 0.5 * CFG["vocab"][k])
            cell = (k, r.dataset, sites[p], t)
            cond = (self.cells.get(cell, {}).get(y, 0) + 16 * marg) / (self.cell_total.get(cell, 0) + 16)
            prior = cond
            neighbors = []
            if t and not hidden[p, t - 1]:
                neighbors.append(int(seq[p, t - 1]))
            if t < 14 and not hidden[p, t + 1]:
                neighbors.append(~int(seq[p, t + 1]))
            if neighbors:
                prior = np.mean(
                    [
                        (self.transitions.get((k, n), {}).get(y, 0) + 16 * cond)
                        / (self.edge_total.get((k, n), 0) + 16)
                        for n in neighbors
                    ]
                )
            ans.append(-np.log2([marg, cond, prior]))
        return np.mean(ans, 0)


@torch.inference_mode()
def representations(net, ids, xy, source="trained"):
    data = []
    for i in range(0, len(ids), 16):
        x = ids[i : i + 16]
        coords = xy[i : i + 16]
        v = torch.zeros((len(x), 2, *x.shape[2:]), dtype=torch.bool, device=x.device)
        if source == "fixed":
            z = torch.stack([net.books[j](x[:, k]) for j, k in enumerate(net.streams)], 1)
        else:
            z = net.encode(x, coords, v)
        # All probes receive identical physical coordinate and time information on their visible streams.
        native = torch.stack([coords[:, s // 2] for s in net.streams], 1)[:, :, :, None].expand(
            *z.shape[:-1], 2
        )
        clock = (
            torch.arange(15, device=z.device, dtype=z.dtype)
            .reshape(1, 1, 1, 15, 1)
            .expand(*z.shape[:-1], 1)
            / 14
        )
        data.append(descriptors(torch.cat((z, native, clock), -1)))
    return np.concatenate(data)


@torch.inference_mode()
def score():
    panel = pd.read_csv(OUT / "panel.csv")
    ids, xy, books = tensors()
    cpu = ids.cpu().numpy()
    val = np.flatnonzero(panel.split.eq("validation"))
    donor = panel.donor.to_numpy()
    priors = Priors(panel, cpu)
    fam = [
        torch.as_tensor(
            pd.read_csv(PARENT / f"{s[:-1]}_17/{s[-1]}_words.csv").family.to_numpy(), device="cuda"
        )
        for s in CFG["streams"]
    ]
    decoder = [audit.network(m, CFG["vocab"][2 * j], 17, "cuda:0") for j, m in enumerate(("eeg", "hb"))]
    scores = []
    base = []
    features = {}
    random = reader(books, "M3").eval()
    for name in ("fixed", "random"):
        features[name] = representations(random, ids, xy, name)
    for arm in STREAMS:
        net = load(books, arm)
        features[arm] = representations(net, ids, xy)
        for mode in ("true", "mismatch", "reverse") if arm == "M3" else ("true",):
            for task_index, task in enumerate(["span", "hb_hidden", "eeg_hidden"]):
                if (arm == "M0_EEG" and task == "hb_hidden") or (arm == "M0_Hb" and task == "eeg_hidden"):
                    continue
                if mode != "true" and task == "span":
                    continue
                for rep in range(CFG["evaluation"]["span_repeats"] if task == "span" else 1):
                    rng = np.random.default_rng(CFG["evaluation"]["seed"] + task_index + rep * 100)
                    for ix in np.array_split(val, int(np.ceil(len(val) / 8))):
                        x = ids[ix].clone()
                        loc = xy[ix].clone()
                        if mode != "true":
                            src = 0 if task == "hb_hidden" else 1
                            if mode == "mismatch":
                                x[:, 2 * src : 2 * src + 2] = ids[donor[ix], 2 * src : 2 * src + 2]
                                loc[:, src] = xy[donor[ix], src]
                            else:
                                x[:, 2 * src : 2 * src + 2] = x[:, 2 * src : 2 * src + 2].flip([-1])
                        hm = hidden_sites(len(x), 4, 15, task, rng)
                        mask = torch.as_tensor(hm, device="cuda")
                        z = net.encode(x, loc, mask)
                        completed = ids[ix].clone()
                        for j, k in enumerate(net.streams):
                            sel = mask[:, k // 2]
                            if not sel.any():
                                continue
                            prob = net.output[j](z[:, j]).float().log_softmax(-1)
                            truth = ids[ix, k]
                            logp = prob.gather(-1, truth[..., None]).squeeze(-1)
                            conf, pred = prob.max(-1)
                            conf = conf.exp()
                            ok = pred == truth
                            entropy = -(prob.exp() * prob).sum(-1) / np.log(2)
                            completed[:, k] = torch.where(sel, pred, truth)
                            fam_ok = fam[k][pred] == fam[k][truth]
                            for n, idx in enumerate(ix):
                                keep = sel[n]
                                c = conf[n][keep]
                                a = ok[n][keep].float()
                                ece = sum(
                                    abs(
                                        (
                                            (c - a)
                                            * ((c >= edge) & (c < (edge + 0.1) if edge < 0.89 else c <= 1))
                                        )
                                        .mean()
                                        .item()
                                    )
                                    for edge in np.arange(10) / 10
                                )
                                row = panel.iloc[idx]
                                meta = dict(
                                    subject=row.subject,
                                    dataset=row.dataset,
                                    record=row.record,
                                    event=int(row.event),
                                    task=task,
                                    rep=rep,
                                    stream=CFG["streams"][k],
                                )
                                scores.append(
                                    dict(
                                        **meta,
                                        arm=arm,
                                        pairing=mode,
                                        n=int(keep.sum()),
                                        bits=-float(logp[n][keep].mean()) / np.log(2),
                                        accuracy=float(a.mean()),
                                        confidence=float(c.mean()),
                                        entropy_bits=float(entropy[n][keep].mean()),
                                        ece_trial=ece,
                                        family_acc=float(fam_ok[n][keep].float().mean()),
                                    )
                                )
                                if mode == "true" and arm == "M2":
                                    v = priors.bits(row, k, cpu[idx, k], hm[n, k // 2])
                                    base.append(dict(**meta, unigram=v[0], condition=v[1], markov=v[2]))
                        for j, m in enumerate(("eeg", "hb")):
                            if 2 * j not in net.streams or not mask[:, j].any():
                                continue

                            def decode(arr):
                                return decoder[j].decode_ids(
                                    {b: arr[:, 2 * j + bi].reshape(-1, 15) for bi, b in enumerate("SO")}
                                )

                            pred, true = decode(completed), decode(ids[ix])
                            for bi, b in enumerate("SO"):
                                err = (pred[b] - true[b]).square().mean(-1).reshape(len(ix), 4, 15)
                                # Find these exact rows in the recently added result block.
                                start = len(scores) - len(ix) * sum(
                                    mask[:, k // 2].any().item() for k in net.streams
                                )
                                picked = [v for v in scores[start:] if v["stream"] == m + b]
                                assert len(picked) == len(ix)
                                for n, row in enumerate(picked):
                                    row["oracle_decode_mse"] = float(err[n][mask[n, j]].mean())
    frame = pd.DataFrame(scores)
    frame.to_csv(OUT / "masked_trial_scores.csv", index=False)
    baseline = pd.DataFrame(base)
    baseline.to_csv(OUT / "prior_trial_scores.csv", index=False)
    keys = ["arm", "pairing", "stream", "task"]
    metrics = [
        "bits",
        "accuracy",
        "confidence",
        "entropy_bits",
        "ece_trial",
        "family_acc",
        "oracle_decode_mse",
    ]
    people = (
        frame.groupby(keys + ["dataset", "subject", "record"])[metrics]
        .mean()
        .groupby(keys + ["dataset", "subject"])
        .mean()
        .reset_index()
    )
    people.to_csv(OUT / "masked_subjects.csv", index=False)
    macro = people.groupby(keys + ["dataset"])[metrics].mean().groupby(keys).mean().reset_index()
    macro.to_csv(OUT / "masked_macro.csv", index=False)
    diffs = []
    for (s, t), g in people.groupby(["stream", "task"]):
        p = g.pivot_table(index=["dataset", "subject"], columns=["arm", "pairing"], values="bits")
        delta = p[("M4", "true")] - p[("M3", "true")]
        avg, lo, hi = interval(delta, delta.index.get_level_values("dataset"))
        diffs.append(dict(stream=s, task=t, people=len(delta), M4_minus_M3_bits=avg, ci_low=lo, ci_high=hi))
    pd.DataFrame(diffs).to_csv(OUT / "paired_differences.csv", index=False)
    classification = []
    phys = []
    probs = {}
    with np.load(OUT / "tokens.npz") as data:
        attr = data["attributes"]
    tr = panel.split.eq("train").to_numpy()
    ma = panel.condition.isin(["MA", "BL"]).to_numpy()
    target = panel.condition.eq("MA").to_numpy()
    for name, x in features.items():
        pred = fit_ridge(x[tr], attr[tr])(x[~tr])
        for idx, i in enumerate(val):
            for j, m in enumerate(["eeg_alpha", "hb_slope"]):
                phys.append(
                    dict(
                        arm=name,
                        dataset=panel.iloc[i].dataset,
                        subject=panel.iloc[i].subject,
                        record=panel.iloc[i].record,
                        event=int(panel.iloc[i].event),
                        attribute=m,
                        mse=(pred[idx, j] - attr[i, j]) ** 2,
                    )
                )
        ss = StandardScaler().fit(x[tr & ma])
        clf = LogisticRegression(C=0.1, max_iter=1000).fit(ss.transform(x[tr & ma]), target[tr & ma])
        probs[name] = clf.predict_proba(ss.transform(x))[:, 1]
    probs["M1"] = (probs["M0_EEG"] + probs["M0_Hb"]) / 2
    for name, prob in probs.items():
        for person, g in panel[~tr & ma].groupby("subject"):
            i = g.index
            y = target[i]
            classification.append(
                dict(
                    arm=name,
                    subject=person,
                    n=len(i),
                    n_ma=int(y.sum()),
                    balanced_accuracy=(
                        balanced_accuracy_score(y, prob[i] >= 0.5) if len(set(y)) == 2 else np.nan
                    ),
                    brier=brier_score_loss(y, prob[i]),
                )
            )
    pd.DataFrame(classification).to_csv(OUT / "ma_readouts.csv", index=False)
    pd.DataFrame(phys).to_csv(OUT / "physiology_readouts.csv", index=False)


@torch.inference_mode()
def history():
    import single_vq_event_coupling as ev
    from src.analysis.observation_task_probe import DOWNSTREAM_ROOT
    from src.data.unified_physiology import ChannelGeometryIndex
    from src.data.channel_geometry import project_head_topview
    from types import SimpleNamespace

    evdir = PARENT / "events_17"
    paired = pd.read_csv(evdir / "coupling_pairs.csv").query("donor_event>=0")
    audit.readout.panel()
    rows, _ = audit.relations.maps()
    geo = ChannelGeometryIndex(DOWNSTREAM_ROOT)
    net = audit.network("eeg", 8192, 17, "cuda:0")
    ids = []
    coords = []
    base = []
    cont = []
    targets = []
    meta = []
    for key, events, record, _ in ev.record_tasks(OUT):
        split, sub, rec = key
        use = paired[(paired.split == split) & (paired.subject == sub) & (paired.record == rec)]
        if not len(use):
            continue
        wanted = set(use.event)
        track = ev.eeg_track(ev.SUPPORTED[key])
        ds = rows[ev.SUPPORTED[key][0]]["dataset"]
        geometry = geo.for_channels(
            record=SimpleNamespace(dataset_id=ds, canonical_subject_id=sub, base_record_id=rec),
            modality="eeg",
            channel_names=list(ev.EEG[:2]),
        )
        xy = np.array([project_head_topview(g) for g in geometry], dtype="float32")
        with np.load(evdir / "events_by_record" / f"{split}_{sub}_{rec}.npz") as z:
            for i, val in enumerate(z["D"]):
                r = json.loads(val)
                if r["event"] not in wanted:
                    continue
                x = ev.context(track, r["eeg_marker"], -26, 2)
                assert x is not None
                code = net(torch.as_tensor(x.copy(), device="cuda"))["ids"]
                ids.append(np.stack([code[b].cpu().numpy() for b in "SO"]))
                coords.append(xy)
                base.append(z["base"][i])
                cont.append(z["continuous"][i])
                targets.append(z["y"][i, 2])
                meta.append(r)
    meta = pd.DataFrame(meta)
    tr = meta.split.eq("train").to_numpy()
    assert tr.sum() == 614 and (~tr).sum() == 107 and meta[~tr].subject.nunique() == 3
    ids = torch.tensor(np.array(ids), device="cuda")
    xy = torch.tensor(np.array(coords), device="cuda")
    # EEG-only independent history. Use the trained EEG streams from each backbone with the two native sensors.
    _, _, books = tensors()
    feat = {}
    y = np.asarray(targets)
    base = np.asarray(base)
    var = y[tr].var(0)
    for arm in ("M0_EEG", "M2", "M3", "M4", "random", "fixed"):
        full = reader(books, "M3").eval() if arm in ("random", "fixed") else load(books, arm)
        full.streams = (0, 1)
        feat[arm] = representations(full, ids, xy[:, None], "fixed" if arm == "fixed" else "trained")
    feat.update(continuous=np.array(cont), history=np.empty((len(y), 0)))
    results = []
    for name, v in feat.items():
        x = np.column_stack([base, v])
        pred = fit_ridge(x[tr], y[tr])(x[~tr])
        temp = meta[~tr].copy()
        for j, clr in enumerate(["HbO", "HbR"]):
            temp["nmse"] = (pred[:, j] - y[~tr, j]) ** 2 / var[j]
            val = (
                temp.groupby(["subject", "record", "condition"])
                .nmse.mean()
                .unstack()
                .dropna()
                .mean(1)
                .groupby("subject")
                .mean()
            )
            for sub, err in val.items():
                results.append(dict(arm=name, color=clr, subject=sub, nmse=err))
    score = pd.DataFrame(results)
    score.to_csv(OUT / "hb_history_readouts.csv", index=False)
    old = (
        pd.read_csv(evdir / "coupling_subject.csv")
        .query("method=='base'")
        .sort_values(["color", "subject"])
    )
    np.testing.assert_allclose(
        score.query("arm=='history'").sort_values(["color", "subject"]).nmse, old.nmse, rtol=1e-6
    )


def summarize():
    prepared = read_json(OUT / "prepared.json")
    assert prepared["signature"] == identity(CFG)
    arms = {arm: read_json(OUT / arm / "result.json") for arm in STREAMS}
    for arm, result in arms.items():
        assert result["signature"] == identity([CFG, arm]) and result["status"] == "completed"
        assert result["step"] == TR["steps"] and (OUT / result["checkpoint"]).is_file()
    primary = pd.read_csv(OUT / "paired_differences.csv")
    whole = primary[primary.task != "span"]
    assert set(whole.stream) == set(CFG["streams"]) and whole.people.eq(prepared["people"][1]).all()
    macro = pd.read_csv(OUT / "masked_macro.csv")
    prior = pd.read_csv(OUT / "prior_trial_scores.csv")
    metrics = ["unigram", "condition", "markov"]
    keys = ["task", "stream"]
    baselines = (
        prior.groupby(keys + ["dataset", "subject", "record"])[metrics]
        .mean()
        .groupby(keys + ["dataset", "subject"])
        .mean()
        .groupby(keys + ["dataset"])
        .mean()
        .groupby(keys)
        .mean()
    )
    comparisons = macro.merge(baselines.reset_index(), on=keys, validate="many_to_one")
    for metric in metrics:
        comparisons[metric + "_minus_model_bits"] = comparisons[metric] - comparisons.bits
    comparisons.to_csv(OUT / "baseline_gains.csv", index=False)
    m3 = comparisons.query("arm=='M3' and pairing=='true' and task!='span'")
    assert len(m3) == 4 and np.isfinite(comparisons.select_dtypes("number")).all().all()
    classification = pd.read_csv(OUT / "ma_readouts.csv")
    measured = pd.read_csv(OUT / "hb_history_readouts.csv")
    assert set(measured.color) == {"HbO", "HbR"} and measured.subject.nunique() == 3
    downstream = classification.groupby("arm").balanced_accuracy.mean().to_dict()
    per_subject = classification.query("arm=='M3'")
    summary_path = OUT / "summary.json"
    finished = read_json(summary_path)["finished"] if summary_path.is_file() else now()
    write_json(
        summary_path,
        dict(
            status="completed",
            finished=finished,
            contract=identity(CFG),
            seed=TR["seed"],
            arm_steps={k: v["step"] for k, v in arms.items()},
            late_fusion=CFG["M1"],
            train_validation_windows=prepared["counts"],
            train_validation_people=prepared["people"],
            primary="paired_differences.csv",
            accuracy="masked_macro.csv",
            baselines="baseline_gains.csv",
            physiology="physiology_readouts.csv",
            downstream="ma_readouts.csv",
            measured_Hb="hb_history_readouts.csv",
            whole_modality_M4_minus_M3_bits=whole.to_dict("records"),
            whole_modality_M3_unigram_minus_model_bits=m3.set_index(
                "stream"
            ).unigram_minus_model_bits.to_dict(),
            MA_vs_BL=dict(
                windows=int(per_subject.n.sum()), people=len(per_subject), mean_balanced_accuracy=downstream
            ),
            measured_Hb_mean_nmse=measured.groupby("arm").nmse.mean().to_dict(),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with owner_lock(OUT, "evaluation"):
        if not args.summary_only:
            score()
            history()
        summarize()
