"""Score completed ablation cells with a common anchor and independent measured endpoints."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
from scipy.linalg import solve
import torch

import run_frozen_science_screen as screen
import run_single_vq_capacity as capacity
from prepare_science_ablations import CFG, OUT, check
from run_science_ablations import ARMS, TR, read, save
from run_dual_component_tokenizer import owner_lock, identity
from src.analysis.conditional_evidence import effect_tables
from src.analysis.frozen_science_screen import gram, slot_vectors, centered, residual, interaction_blocks
from src.analysis.single_vq_relations import interval
from src.utils.io import save_npz_atomic


def normalized(x, tr):
    mu = x[tr].mean((0, 3), keepdims=True)
    sd = x[tr].std((0, 3), keepdims=True)
    return (x - mu) / np.where(sd > 1e-7, sd, 1)


def slot_adapter(x, tr):
    if x.shape[-1] <= 128:
        return slot_vectors(x, tr)
    # All dimensions remain available to the full linear reader. Only the P1 adapter needs compression.
    vectors = []
    for j in range(2):
        a = x[tr, :, j].reshape(-1, x.shape[-1]).astype(float)
        mu = a.mean(0)
        _, v = np.linalg.eigh(np.cov(a, rowvar=False))
        vectors.append(((x[:, :, j] - mu) @ v[:, -128:]).astype("float32"))
    return slot_vectors(np.stack(vectors, 2), tr)


def paired_ci(v):
    a, lo, hi = interval(v, np.repeat(screen.parent.DS, len(v)))
    return dict(delta=a, ci_low=lo, ci_high=hi, people=len(v), improved=int((np.asarray(v) > 0).sum()))


def branch_kernels(x, tr):
    ref = normalized(x, tr)
    kernels = []
    for j in range(2):
        k = gram(ref[:, :, j])
        d = k.diagonal()
        dist = np.maximum(d[:, None] + d[None, :] - 2 * k, 0)
        bandwidth = max(dist[np.ix_(tr, tr)].mean(), 1e-7)
        kernels.append(np.exp(-dist / bandwidth))
    return kernels


def kernel_prediction(k, y, tr, ev):
    a = centered(k, tr, tr)
    b = centered(k, tr, ev)
    mu = y[tr].mean(0)
    return mu + b @ solve(a + np.eye(len(tr)), y[tr] - mu, assume_a="pos")


def so_scores(x, m, tr, dev, folder, label):
    ks, ko = branch_kernels(x, tr)
    # Actual future EEG alpha-fraction change, same sites and horizons as P1.
    _, pm, d = screen.parent.load_parent("strict_common")
    np.testing.assert_array_equal(pm.id, m.id)
    y = d["eeg_target"].reshape(len(m), -1)

    # Keep the main-effect kernel and ridge scale fixed while adding the interaction.
    def block(k):
        return centered(k, tr, tr), centered(k, tr, dev)

    a, c = block(ks)
    b, e = block(ko)
    ab, ce = block(ks * ko)
    models = {"S": (a, c), "O": (b, e), "additive": (a + b, c + e), "interaction": (a + b + ab, c + e + ce)}
    rows = []
    for name, (train, valid) in models.items():
        pred = y[tr].mean(0) + valid @ solve(train + np.eye(len(tr)), y[tr] - y[tr].mean(0), assume_a="pos")
        err = (((pred - y[dev]) / y[tr].std(0)) ** 2).mean(1)
        for who, idx in m.iloc[dev].reset_index(drop=True).groupby("subject").indices.items():
            rows.append(dict(subject=who, arm=name, nmse=err[idx].mean(), representation=label))
    return rows


def audit_interaction_cell(task):
    arm, seed = task
    folder = (
        OUT
        / ("dimensions" if arm == "dimension" else "models")
        / (str(seed) if arm == "dimension" else f"{arm}_{seed}")
    )
    m, tr, dev = screen.panel()
    _, pm, d = screen.parent.load_parent("strict_common")
    np.testing.assert_array_equal(m.id, pm.id)
    y = d["eeg_target"].reshape(len(m), -1).astype(float)
    with np.load(folder / "science.npz") as z:
        first = z["level0"]
        variants = {"full": first}
        if "level1" in z:
            second = z["level1"]
            variants = {"full": first + second, "level1_only": first, "level2_only": second}
    rows = []
    for name, x in variants.items():
        kernels = branch_kernels(x, tr)
        for kind, (a, b) in interaction_blocks(*kernels, tr, dev, m.donor.to_numpy()).items():
            pred = y[tr].mean(0) + b @ solve(a + np.eye(len(tr)), y[tr] - y[tr].mean(0), assume_a="pos")
            err = (((pred - y[dev]) / y[tr].std(0)) ** 2).mean(1)
            for subject, idx in m.iloc[dev].reset_index(drop=True).groupby("subject").indices.items():
                rows.append(
                    dict(
                        model=f"d{seed}" if arm == "dimension" else arm,
                        tokenizer_seed=17 if arm == "dimension" else seed,
                        subject=subject,
                        representation=name,
                        arm=kind,
                        nmse=err[idx].mean(),
                    )
                )
    return pd.DataFrame(rows)


def attributes(x, ids, m, tr, ev):
    alpha = screen.array("raw_bands")[:, :, 0, :, 2]
    rows = []
    for j, b in enumerate("SO"):
        vec = x[:, :, j]
        a = vec[tr].reshape(-1, x.shape[-1]).astype(float)
        mu = a.mean(0)
        a -= mu
        y = alpha[tr].ravel()
        ym = y.mean()
        cov = a.T @ a
        coef = solve(
            cov + np.eye(len(mu)) * (max(cov.trace() / len(mu), 1e-8) * 0.01), a.T @ (y - ym), assume_a="pos"
        )
        pred = (vec[ev] - mu) @ coef + ym
        # Train-only ID-level observed alpha family. An unseen composite ID gets the global prior.
        words = ids[:, :, j].astype(np.int64)
        uniq, code = np.unique(words[tr], return_inverse=True)
        cnt = np.bincount(code.ravel())
        sums = np.bincount(code.ravel(), weights=y)
        profile = (sums + 16 * ym) / (cnt + 16)
        threshold = np.quantile(profile[code.ravel()], 0.75)
        where = np.searchsorted(uniq, words[ev])
        match = uniq[np.minimum(where, len(uniq) - 1)] == words[ev]
        value = np.where(match, profile[np.minimum(where, len(uniq) - 1)], ym)
        hi = value > threshold
        for who, idx in m.iloc[ev].reset_index(drop=True).groupby("subject").indices.items():
            v = hi[idx]
            cond = m.condition.to_numpy()[ev[idx]]
            denom = v[:, :, :-1].sum()
            rows.append(
                dict(
                    subject=who,
                    branch=b,
                    alpha_nmse=np.mean((pred[idx] - alpha[ev[idx]]) ** 2) / np.var(y),
                    seen_fraction=match[idx].mean(),
                    posterior_minus_frontal=v[:, 2:].mean() - v[:, :2].mean(),
                    high_minus_low_load=v[cond == "3-back session"].mean()
                    - v[cond == "0-back session"].mean(),
                    high_persistence=(v[:, :, :-1] & v[:, :, 1:]).sum() / denom if denom else np.nan,
                    transition_support=int(denom),
                    events=len(idx),
                )
            )
    return rows


def dimension(dim):
    folder = OUT / "dimensions" / str(dim)
    with owner_lock(folder):
        if (folder / "done.json").exists():
            return
        k = CFG["E3A"]["K"]
        seed = 17
        ref = capacity.train_folder("eeg", dim, seed) / "models" / f"eeg__waveform__vq{k}"
        record = read(ref / "summary.json")
        if (
            record["signature"]["config"] != identity(capacity.resolved("eeg", dim, seed))
            or record["completed_step"] != 10000
        ):
            raise ValueError("dimension identity mismatch")
        model = capacity.load_scoring_model("candidate", "eeg", dim, k, seed)
        wave = screen.array("wave")
        ids, (_, dec) = screen.encode(model, wave, details=True)
        x = np.stack(
            [model.branches[b]["quantizer"].codebook.numpy()[ids[:, :, j]] for j, b in enumerate("SO")],
            axis=2,
        )
        save_npz_atomic(folder / "science.npz", level0=x, ids0=ids.astype("uint16"))
        m, _, ev = screen.panel()
        loss = ((dec[ev] - wave[ev]) ** 2).mean((1, 2, 3)) / (wave[ev] ** 2).mean((1, 2, 3))
        pd.DataFrame(dict(subject=m.subject.values[ev], x_nmse=loss)).groupby("subject").mean().to_csv(
            folder / "reconstruction.csv"
        )
        save(
            folder / "done.json",
            dict(status="completed", source=str(ref.relative_to(screen.ROOT)), K=k, dimension=dim),
        )


def score_one(task):
    arm, seed = task
    folder = (
        OUT
        / ("dimensions" if arm == "dimension" else "models")
        / (str(seed) if arm == "dimension" else f"{arm}_{seed}")
    )
    with owner_lock(folder, "score"):
        if (folder / "score.json").is_file():
            return
        if read(folder / "done.json")["status"] != "completed":
            raise ValueError("incomplete model")
        with np.load(folder / "science.npz") as z:
            x = z["level0"]
            ids = z["ids0"].astype("int64")
            variants = {}
            if "level1" in z:
                variants = {"level1_only": x, "level2_only": z["level1"]}
                x = x + variants["level2_only"]
                ids = ids * CFG["E3"][arm][1] + z["ids1"]
            exchanges = {k: z[k] for k in ("exchange_wrong", "without_eeg", "without_hb") if k in z}
        m, tr, ev = screen.panel()
        rows = []
        attr = []
        taskrows = []
        sorows = []
        hk = gram(screen.hb_slots(screen.array("hb"), tr))
        for name, vec in {"full": x, **variants}.items():
            sk = gram(normalized(vec, tr))
            rows.append(screen.score_sources(name, sk, slot_adapter(vec, tr) if name == "full" else None))
            taskrows.extend(screen.task_scores(name, hk, sk))
            sorows.extend(so_scores(vec, m, tr, ev, folder, name))
            if name == "full":
                attrids = ids
            else:
                with np.load(folder / "science.npz") as z:
                    attrids = z["ids0" if name == "level1_only" else "ids1"]
            attr.extend(dict(**r, representation=name) for r in attributes(vec, attrids, m, tr, ev))
        pd.concat(rows).to_csv(folder / "readout.csv.gz", index=False)
        pd.DataFrame(taskrows).to_csv(folder / "task.csv", index=False)
        pd.DataFrame(sorows).to_csv(folder / "so.csv", index=False)
        pd.DataFrame(attr).to_csv(folder / "attributes.csv", index=False)
        if exchanges:
            # Crossed pairing starts inside the encoder. The downstream readout is fitted in the same
            # training condition and held fixed across paired and wrong exchanges at evaluation.
            training_vec = exchanges["exchange_wrong"] if arm == "cross_wrong" else x
            y = screen.array("target").reshape(len(m), -1)
            sd = y[tr].std(0)
            with np.load(screen.OUT / "anchor.npz") as a:
                base = a["base"]
                r = y[tr] - a["oof"]
            donor = m.donor.to_numpy()
            sq = gram(screen.hb_slots(screen.array("hb"), tr) ** 2)
            cap = base + residual(hk, sq, r, tr, ev, donor)
            results = []
            for key, val in {"aligned": x, **exchanges}.items():
                vv = training_vec.copy()
                vv[ev] = val[ev]
                sk = gram(normalized(vv, tr))
                pred = base + residual(hk, sk, r, tr, ev, donor)
                if key == "without_hb":
                    # Full input-chain EEG-only probe: remove the Hb anchor and Hb residual features too.
                    ek = gram(normalized(val, tr))
                    pure = kernel_prediction(ek, y, tr, ev)
                elif key == "without_eeg":
                    pure = base + residual(hk, gram(normalized(val, tr)), r, tr, ev, donor)
                else:
                    pure = pred
                for who, idx in m.iloc[ev].reset_index(drop=True).groupby("subject").indices.items():
                    loss = lambda v: np.mean(((v[idx] - y[ev[idx]]) / sd) ** 2)
                    results.append(
                        dict(
                            subject=who,
                            input=key,
                            conditional_nmse=loss(pred),
                            capacity_delta=loss(cap) - loss(pred),
                            full_chain_nmse=loss(pure),
                        )
                    )
            pd.DataFrame(results).to_csv(folder / "exchange.csv", index=False)
        save(
            folder / "score.json",
            dict(status="completed", arm=arm, seed=seed, train_events=len(tr), development_events=len(ev)),
        )


def compare():
    tasks = [(a, s) for a in ARMS for s in TR["seeds"]] + [("dimension", d) for d in CFG["E3A"]["dimensions"]]
    stores = defaultdict(list)
    for arm, seed in tasks:
        folder = (
            OUT
            / ("dimensions" if arm == "dimension" else "models")
            / (str(seed) if arm == "dimension" else f"{arm}_{seed}")
        )
        if read(folder / "score.json")["status"] != "completed":
            raise ValueError("incomplete scores")
        label = f"d{seed}" if arm == "dimension" else arm
        for name, file in dict(
            readout="readout.csv.gz",
            task="task.csv",
            so="so.csv",
            attributes="attributes.csv",
            guardrails="guardrails.csv",
            exchange="exchange.csv",
            dimension_reconstruction="reconstruction.csv",
        ).items():
            if (folder / file).is_file():
                stores[name].append(
                    pd.read_csv(folder / file, dtype={"arm": str}).assign(
                        model=label, tokenizer_seed=17 if arm == "dimension" else seed
                    )
                )
        if (folder / "usage.json").exists():
            stores["usage"].append(
                pd.DataFrame(read(folder / "usage.json")).assign(
                    model=label,
                    tokenizer_seed=seed,
                    parameters=read(folder / "done.json")["parameters"],
                    seconds=read(folder / "done.json")["seconds"],
                )
            )
    for k, v in stores.items():
        stores[k] = pd.concat(v)
        if k != "readout":
            stores[k].to_csv(OUT / (k + ".csv"), index=False)
    df = stores["readout"]
    person = df.groupby(
        ["model", "representation", "reader", "input", "subject", "arm"], as_index=False
    ).nmse.mean()
    person.to_csv(OUT / "people.csv", index=False)
    effects = effect_tables(
        person.query("input=='aligned'"),
        "nmse",
        [("full_H", "anchor", "true"), ("matched_H", "capacity", "true"), ("wrong_train", "wrong", "true")],
        ["model", "representation", "reader"],
    )
    effects.to_csv(OUT / "increments.csv", index=False)
    matrix = person.query("input=='aligned' and representation=='full' and reader=='slot'").pivot(
        index="subject", columns=["arm", "model"], values="nmse"
    )
    deltas = matrix["capacity"] - matrix["true"]
    pairs = [("B0", a) for a in ARMS if a != "B0"] + [
        ("s_off", "both_off"),
        ("o_off", "both_off"),
        ("single14", "dual14"),
        ("dual13", "dual14"),
        ("dual13", "dual13_first"),
        ("independent_size", "shared"),
        ("shuffled_neighbor", "neighbor"),
        ("cross_wrong", "cross_true"),
        ("eog_wrong", "eog_true"),
        ("d128", "d64"),
        ("d128", "d256"),
    ]
    comparisons = pd.DataFrame(
        [dict(baseline=a, candidate=b, **paired_ci(deltas[b] - deltas[a])) for a, b in pairs]
    )
    comparisons.to_csv(OUT / "paired_ablation.csv", index=False)
    return comparisons


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["dimension", "score", "summary"])
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    check()
    torch.set_num_threads(1)
    if args.stage == "summary":
        with owner_lock(OUT, "summary"):
            compare()
        return
    with owner_lock(OUT, args.stage):
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("fork")) as pool:
            if args.stage == "dimension":
                list(pool.map(dimension, CFG["E3A"]["dimensions"]))
                list(pool.map(score_one, [("dimension", d) for d in CFG["E3A"]["dimensions"]]))
            else:
                pending = {(a, s) for a in ARMS for s in TR["seeds"]}
                while pending:
                    ready = [
                        t
                        for t in sorted(pending)
                        if (OUT / "models" / f"{t[0]}_{t[1]}" / "done.json").is_file()
                    ]
                    if ready:
                        list(pool.map(score_one, ready))
                        pending.difference_update(ready)
                    else:
                        time.sleep(10)


if __name__ == "__main__":
    main()
