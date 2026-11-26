"""Exploratory time-block transfer and quantization bridge on the same frozen events."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score

import run_nback_function_screen as run
from src.analysis.nback_function_screen import flat, join, phase, probe
from src.utils.io import save_npz_atomic, write_json

CONFIG = run.CONFIG.with_name("nback_order_check_v1.yaml")
CFG = yaml.safe_load(CONFIG.read_text())
OUT = run.OUT / CFG["output_subdirectory"]


@torch.inference_mode()
def prepare(g):
    from audit_aligned_tokenizer import network

    torch.set_num_threads(1)
    sub = g.subject.iloc[0]
    dest = OUT / (sub + ".npz")
    if dest.exists():
        return sub
    ledger, e, h = run.samples(g)
    sc = run.read(run.ROOT / run.CFG["scales"])
    gain = run.read(run.ROOT / run.CFG["hb_coordinates"])["training_hb_gain"]
    payload = {}
    with np.load(run.OUT / "subjects" / (sub + ".npz")) as parent:
        ids = ledger.query("status=='complete'").id.to_numpy(dtype=str)
        np.testing.assert_array_equal(ids, parent["id"])
        payload["id"] = ids
        for mod, x, key in (("eeg", e, "ids"), ("hb", h.reshape(len(h), 4, 15, 40) * gain, "hids")):
            names = run.CFG["sites"][mod]
            scales = np.array([sc["positions"][mod][run.DATASET][n]["scale"] for n in names], dtype="f")
            net = network(mod, run.CFG["model"][mod], 17, "cpu")
            # Match the original Hb coordinate operation exactly at the integer-assignment check.
            if mod == "hb":
                x = h.reshape(len(h), 4, 15, 40)
                scales = scales / gain
            x = (x / scales[None, :, None, None]).reshape(-1, 15, x.shape[-1])
            native = parent[key].reshape(-1, 2, 15)
            outputs = []
            for start in range(0, len(x), 16):
                result = net(torch.from_numpy(x[start : start + 16].copy()))
                for j, b in enumerate("SO"):
                    np.testing.assert_array_equal(native[start : start + 16, j], result["ids"][b].numpy())
                outputs.append(np.stack([result["latent"][b].numpy() for b in "SO"], axis=1))
            latent = np.concatenate(outputs).reshape(len(e), 4, 2, 15, 128)
            payload[mod] = join(*[phase(latent[:, :, b]) for b in range(2)]).astype("f")
    save_npz_atomic(dest, **payload)
    return sub


def score():
    m = pd.read_csv(run.OUT / "support.csv").query("status=='complete'").reset_index(drop=True)
    p = m.subject.to_numpy()
    y = np.eye(3)[m.condition.map(run.CFG["classes"].index)]
    tr = np.flatnonzero(m.split == "train")
    ev = np.flatnonzero(m.split == "validation")
    features = {}
    with np.load(run.PARENT / "panel.npz") as book:
        for sub, g in m.groupby("subject", sort=False):
            with np.load(OUT / (sub + ".npz")) as pre, np.load(run.OUT / "subjects" / (sub + ".npz")) as w:
                np.testing.assert_array_equal(g.id, pre["id"])
                e = w["bands_X"]
                h = w["h"].reshape(len(e), 4, 15, 20, 2).mean(3)
                hard = []
                for mod, key in (("eeg", "ids"), ("hb", "hids")):
                    hard.append(
                        join(*[phase(book[mod + b + "_book"][w[key][:, :, j]]) for j, b in enumerate("SO")])
                    )
                diff = e[:, :, 5:15].mean(2) - e[:, :, :2].mean(2)
                for k, v in dict(
                    continuous_E=phase(e),
                    continuous_H=phase(h),
                    continuous_EH=join(phase(e), phase(h)),
                    token_E=hard[0],
                    token_EH=join(*hard),
                    pre_E=pre["eeg"],
                    pre_EH=join(pre["eeg"], pre["hb"]),
                    theta_alpha=join(diff[:, :2, 1].mean(1), diff[:, 2:, 2].mean(1)),
                ).items():
                    features.setdefault(k, []).append(flat(v))
    rows = []

    def record(ix, name, scope, pr, reg):
        for i, cls in zip(ix, pr, strict=True):
            rows.append(
                dict(
                    id=m.id[i],
                    subject=p[i],
                    event=int(m.event[i]),
                    method=name,
                    scope=scope,
                    true=int(y[i].argmax()),
                    pred=int(cls),
                    ridge=reg,
                )
            )

    lookup = pd.crosstab(m.event[tr], m.condition[tr]).reindex(columns=run.CFG["classes"], fill_value=0)
    if not (lookup.gt(0).sum(1) == 1).all() or not set(m.event[ev]) <= set(lookup.index):
        raise ValueError("ordinal lookup tied or absent")
    lookup.to_csv(OUT / "ordinal_training_counts.csv")
    record(ev, "ordinal", "people", lookup.loc[m.event[ev]].to_numpy().argmax(1), None)
    trial = m.event.to_numpy()
    ranges = CFG["windows"]
    if not np.all(sum((trial >= lo) & (trial <= hi) for lo, hi in ranges) == 1):
        raise ValueError("invalid ordinal coverage")
    for name, parts in features.items():
        x = np.concatenate(parts)
        for tag, bounds in [("people", None)] + [("people_and_range", bounds) for bounds in ranges]:
            a, b = tr, ev
            if bounds is not None:
                held = (trial >= bounds[0]) & (trial <= bounds[1])
                a = tr[~held[tr]]
                b = ev[held[ev]]
                assert set(trial[a]).isdisjoint(trial[b])
            pr, _, reg = probe(
                x, y, p, a, b, classification=True, ridge=run.CFG["ridge"], folds=run.CFG["folds"]
            )
            record(b, name, tag, pr.argmax(1), reg)
    d = pd.DataFrame(rows)
    if (
        d.duplicated(["id", "scope", "method"]).any()
        or (d.groupby(["scope", "method"]).size() != len(ev)).any()
    ):
        raise ValueError("incomplete block evaluation")
    d.to_csv(OUT / "predictions.csv", index=False)
    stats = (
        d.groupby(["scope", "method", "subject"])
        .apply(lambda g: balanced_accuracy_score(g.true, g.pred), include_groups=False)
        .rename("ba")
        .reset_index()
    )
    stats.to_csv(OUT / "people.csv", index=False)
    write_json(
        OUT / "summary.json",
        dict(
            status="completed",
            experiment_id=CFG["experiment_id"],
            events=len(ev),
            development_people=len(set(p[ev])),
            ordinals=len(lookup),
            unseen=0,
            ties=0,
            means={scope: g.groupby("method").ba.mean().to_dict() for scope, g in stats.groupby("scope")},
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    (OUT / "resolved.yaml").write_text(CONFIG.read_text())
    m = pd.read_csv(run.PARENT / "panel.csv")
    run.check_panel(m)
    # Software check for all disjoint interval boundaries before any signal access.
    labels = np.arange(27)
    assert np.all(sum((labels >= lo) & (labels <= hi) for lo, hi in CFG["windows"]) == 1)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("fork")) as ex:
        for sub in ex.map(prepare, [g for _, g in m.groupby("subject")]):
            print(sub, flush=True)
    score()
