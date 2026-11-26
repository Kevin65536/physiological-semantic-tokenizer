"""Evaluate the frozen candidate library, grouped readouts, donor controls and transfer."""

from itertools import product
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from run_task_pattern_expansion import CFG, OUT
from src.analysis.nback_function_screen import join, probe
from src.analysis.task_pattern_expansion import (
    disjoint_donors,
    features,
    family_features,
    auc_columns,
    PRE,
    ACTIVE,
    LATE,
    TAIL,
)
from src.data.task_pattern_panel import load_task_panel
from src.utils.io import write_json


def measures(m, x, names, strata):
    scans = []
    group = ["split", "subject", "record"] + (["bin"] if strata else [])
    for ks, ix in m.assign(bin=m.event // 5).groupby(group).indices.items():
        y = m.label.to_numpy()[ix]
        if len(set(y)) != 2:
            continue
        auc = auc_columns(x[ix], y)
        diff = x[ix][y == 1].mean(0) - x[ix][y == 0].mean(0)
        scans.extend(
            dict(zip(group, ks), family=f, name=n, auc=v, delta=de, events=len(ix))
            for (f, n), v, de in zip(names, auc, diff)
        )
    frame = pd.DataFrame(scans)
    if strata:
        frame = frame.groupby(group[:-1] + ["family", "name"], as_index=False)[["auc", "delta"]].mean()
    return frame.groupby(["split", "subject", "family", "name"], as_index=False)[["auc", "delta"]].mean()


def screen(m, a, h):
    raw = features(a["e"], h, a["ef"], a["hf"])
    names = list(raw)
    x = np.column_stack(list(raw.values()))
    frame = pd.DataFrame(x, columns=["__".join(k) for k in names])
    pd.concat([m[["id", "subject", "split", "record", "event", "label"]], frame], axis=1).to_csv(
        OUT / m.task[0] / "pattern_trials.csv.gz", index=False
    )
    people = measures(m, x, names, False).merge(
        measures(m, x, names, True).rename(columns={"auc": "strata_auc", "delta": "strata_delta"}),
        on=["split", "subject", "family", "name"],
        how="left",
        validate="one_to_one",
    )
    rows = []
    for (fam, name), g in people.groupby(["family", "name"]):
        train = g[g.split == "train"]
        dev = g[g.split == "validation"]
        sign = 1 if train.auc.mean() >= 0.5 else -1
        orient = lambda v: 0.5 + sign * (v - 0.5)
        rows.append(
            dict(
                family=fam,
                name=name,
                sign=sign,
                train=orient(train.auc).mean(),
                development=orient(dev.auc).mean(),
                stratified=orient(dev.strata_auc).mean(),
                strata_people=dev.strata_auc.count(),
                dev_delta=dev.delta.mean(),
                train_positive=(orient(train.auc) > 0.5).sum(),
                train_people=len(train),
                positive=(orient(dev.auc) > 0.5).sum(),
                people=len(dev),
            )
        )
    table = (
        pd.DataFrame(rows)
        .sort_values(["family", "train", "name"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    eligible = table.train.ge(0.55) & (table.train_positive >= table.train_people * 2 / 3)
    selected = table[eligible].groupby("family", sort=False).head(2).index
    table["selected"] = table.index.isin(selected)
    table["passes"] = (
        table.selected
        & table.development.ge(0.58)
        & (table.positive >= 0.75 * table.people)
        & table.people.ge(3)
        & table.strata_people.ge(3)
        & table.stratified.ge(0.55)
    )
    table["max_p"] = np.nan
    if len(selected):
        dev = people.query("split=='validation'").pivot(
            index="subject", columns=["family", "name"], values="auc"
        )
        sel = table.loc[selected]
        effects = np.array([(dev[r.family, r.name].to_numpy() - 0.5) * r.sign for r in sel.itertuples()]).T
        if not np.isfinite(effects).all():
            raise ValueError("incomplete paired max-statistic")
        signs = np.array(list(product([-1, 1], repeat=len(dev))))
        mx = (signs @ effects / len(dev)).max(1)
        table.loc[selected, "max_p"] = [np.mean(mx >= v - 1e-14) for v in effects.mean(0)]
    return people, table


def models(m, a, h):
    tr = np.flatnonzero(m.split == "train")
    ev = np.flatnonzero(m.split == "validation")
    y = np.eye(2)[m.label]
    p = m.subject.to_numpy()

    def phased(v):
        return join(*[v[:, :, s].mean(2) for s in (PRE, ACTIVE, LATE, TAIL)])

    e = phased(a["e"])
    hh = phased(h)
    eo, et = family_features(a["ef"])
    ho, ht = family_features(a["hf"])
    inputs = dict(
        EEG=e,
        Hb=hh,
        EH=join(e, hh),
        pre_only=join(a["e"][:, :, PRE].mean(2), h[:, :, PRE].mean(2)),
        E_family=eo,
        H_family=ho,
        families=join(eo, ho),
        ordered=join(eo, ho, et, ht),
        hybrid=join(e, hh, eo, ho),
    )
    preds = []

    def record(name, scores, reg):
        for i, z in zip(ev, scores, strict=True):
            preds.append(
                dict(
                    id=m.id[i],
                    subject=p[i],
                    record=m.record[i],
                    event=m.event[i],
                    label=m.label[i],
                    model=name,
                    pred=int(z.argmax()),
                    score=float(z[1] - z[0]),
                    ridge=reg,
                )
            )

    for name, x in inputs.items():
        v, _, reg = probe(x, y, p, tr, ev, classification=True)
        record(name, v, reg)
    census = m.iloc[tr].groupby(["record", "event"]).label.mean()
    pri = m.label.iloc[tr].mean()
    vals = np.array([census.get((m.record[i], m.event[i]), pri) for i in ev])
    record("ordinal", np.column_stack([1 - vals, vals]), np.nan)
    df = pd.DataFrame(preds)
    pp = (
        df.groupby(["model", "subject"])
        .apply(lambda g: balanced_accuracy_score(g.label, g.pred), include_groups=False)
        .rename("ba")
        .reset_index()
    )
    return df, pp


def coupling(m, a, h):
    d = disjoint_donors(m)
    good = np.flatnonzero(d >= 0)
    y = (h[:, :, LATE].mean(2) - h[:, :, ACTIVE].mean(2)).reshape(len(h), -1)
    # Only same-condition different trials with disjoint 30-s supports enter any arm.
    p = m.subject.to_numpy()
    tr = good[m.split.to_numpy()[good] == "train"]
    ev = good[m.split.to_numpy()[good] == "validation"]
    base = join(m.label.to_numpy(), h[:, :, :7])
    eeg = a["e"][:, :, :7]
    rows = []
    sd = y[tr].std(0)
    for model, x in [("H", base), ("HE", join(base, eeg)), ("wrong", join(base, eeg[d.clip(min=0)]))]:
        pr, _, reg = probe(x, y, p, tr, ev, classification=False)
        for j, i in enumerate(ev):
            for k in range(8):
                rows.append(
                    dict(
                        id=m.id[i],
                        subject=p[i],
                        donor=m.id[d[i]],
                        site=k // 2,
                        color=("HbO", "HbR")[k % 2],
                        model=model,
                        pred=pr[j, k],
                        observed=y[i, k],
                        train_sd=sd[k],
                        ridge=reg,
                        nmse=((pr[j, k] - y[i, k]) / sd[k]) ** 2,
                    )
                )
    return pd.DataFrame(rows), dict(
        train=len(tr), validation=len(ev), dev_people=len(set(p[ev])), unmatched=int((d < 0).sum())
    )


def main():
    ledgers = []
    tables = {}
    persons = {}
    summary = {}
    for task in CFG["tasks"]:
        folder = OUT / task
        folder.mkdir(exist_ok=True)
        ledger, m, a = load_task_panel(OUT, task)
        ledgers.append(ledger)
        h = a["h"].reshape(len(m), 4, 15, 20, 2).mean(3)
        pers, tab = screen(m, a, h)
        tables[task] = tab
        persons[task] = pers
        pers.to_csv(folder / "pattern_people.csv", index=False)
        tab.to_csv(folder / "pattern_selection.csv", index=False)
        pr, ba = models(m, a, h)
        pr.to_csv(folder / "predictions.csv.gz", index=False)
        ba.to_csv(folder / "model_people.csv", index=False)
        c, support = coupling(m, a, h)
        c.to_csv(folder / "coupling_predictions.csv.gz", index=False)
        cp = c.groupby(["subject", "model"]).nmse.mean().unstack()
        cp.to_csv(folder / "coupling_people.csv")
        scores = ba.pivot(index="subject", columns="model", values="ba")
        change = scores.subtract(scores.EH, axis=0)
        gain = lambda k: bool(change[k].mean() >= 0.02 and (change[k] > 0).mean() >= 0.75)
        summary[task] = dict(
            planned=ledger.groupby("split").size().to_dict(),
            scored=m.groupby("split").size().to_dict(),
            subjects=m.groupby("split").subject.nunique().to_dict(),
            selected=int(tab.selected.sum()),
            passed=tab.loc[tab.passes, "name"].tolist(),
            ordinal_warning=bool(scores.ordinal.mean() >= 0.65),
            BA=scores.mean().to_dict(),
            hybrid_gain=gain("hybrid"),
            family_gain=gain("families"),
            coupling_support=support,
            coupling_nmse=cp.mean().to_dict(),
            coupling_gain_over_H=float((cp.H - cp.HE).mean()),
            coupling_gain_over_wrong=float((cp.wrong - cp.HE).mean()),
            better_than_both=int(((cp.HE < cp.H) & (cp.HE < cp.wrong)).sum()),
        )
        print(task, summary[task], flush=True)
    trans = []
    for source, target in [("mental_arithmetic", "wg"), ("wg", "mental_arithmetic")]:
        tab = tables[source]
        dev = persons[target].query("split=='validation'")
        for r in tab[tab.selected].itertuples():
            g = dev[(dev.family == r.family) & (dev.name == r.name)]
            v = 0.5 + r.sign * (g.auc - 0.5)
            q = 0.5 + r.sign * (g.strata_auc - 0.5)
            trans.append(
                dict(
                    source=source,
                    target=target,
                    family=r.family,
                    name=r.name,
                    sign=r.sign,
                    source_dev_auc=r.development,
                    target_dev_auc=v.mean(),
                    target_stratified=q.mean(),
                    positive=(v > 0.5).sum(),
                    people=len(v),
                    passes_both=bool(
                        r.passes
                        and v.mean() >= 0.58
                        and (v > 0.5).mean() >= 0.75
                        and q.mean() >= 0.55
                        and q.count() >= 3
                    ),
                )
            )
    pd.DataFrame(trans).to_csv(OUT / "transfer.csv", index=False)
    pd.concat(ledgers).to_csv(OUT / "support.csv", index=False)
    write_json(
        OUT / "summary.json", dict(experiment_id=CFG["experiment_id"], status="completed", tasks=summary)
    )


if __name__ == "__main__":
    main()
