"""Decompose the selected beta asymmetry and test a matched continuous Hb trend control."""

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from run_task_pattern_expansion import CFG, OUT, CONTRACT
from src.data.task_pattern_panel import load_task_panel
from src.analysis.task_pattern_expansion import ACTIVE, LATE, TAIL, PRE, auc_columns
from src.analysis.nback_function_screen import probe, join
from src.utils.io import write_json


def main():
    folder = OUT / "localization"
    folder.mkdir(exist_ok=True)
    (folder / "resolved.yaml").write_text(
        CONTRACT.with_name("task_pattern_localization_v1.yaml").read_text()
    )
    rows = []
    models = []
    examples = []
    thresholds = []
    for task in CFG["tasks"]:
        ledger, m, a = load_task_panel(OUT, task)
        e = a["e"]
        h = a["h"].reshape(len(m), 4, 15, 20, 2)
        delta = e[:, :, TAIL].mean(2) - e[:, :, ACTIVE].mean(2)
        eb = delta[:, 4:6, 2]
        tr = np.flatnonzero(m.split == "train")
        ev = np.flatnonzero(m.split == "validation")
        word = a["hf"][:, :2, :, 0] == 3

        def timing(q):
            return q[:, :, LATE].mean((1, 2)) - q[:, :, ACTIVE].mean((1, 2))

        seconds = (np.arange(20) - 9.5) / 10
        slopes = np.einsum("nspc,c->nsp", h[:, :2, :, :, 0], seconds) / np.sum(seconds**2)
        cuts = np.quantile(slopes[tr], 0.75, axis=(0, 2))
        rawq = slopes > cuts[None, :, None]
        thresholds.append(
            dict(task=task, left=cuts[0], right=cuts[1], unit=CFG["sites"][m.dataset[0]]["hb_unit"] + "/s")
        )
        cols = dict(
            beta_lr=eb[:, 0] - eb[:, 1],
            beta_left=eb[:, 0],
            beta_right=eb[:, 1],
            beta_mean=eb.mean(1),
            beta_lr_pre=e[:, 4, PRE, 2].mean(1) - e[:, 5, PRE, 2].mean(1),
            alpha_lr=delta[:, 4, 1] - delta[:, 5, 1],
            Hb_word_timing=timing(word),
            Hb_measured_slope_timing=timing(rawq),
        )
        values = np.column_stack(list(cols.values()))
        prior = ledger.set_index(["subject", "record", "event"])
        prev = []
        for r in m.itertuples():
            k = (r.subject, r.record, r.event - 1)
            if k in prior.index:
                last = prior.loc[k]
                valid = all(
                    0 < r0 - last[c] <= 40
                    for r0, c in [(r.eeg_start, "eeg_start"), (r.hb_start, "hb_start")]
                )
                prev.append(int(last.label) if valid else -1)
            else:
                prev.append(-1)
        m = m.assign(prev=prev)
        for scope in ("all", "previous_condition", "at_least_five_per_class"):
            use = m.iloc[ev]
            if scope == "previous_condition":
                use = use[use.prev >= 0]
            if scope == "at_least_five_per_class":
                nn = pd.crosstab(use.subject, use.label)
                use = use[use.subject.isin(nn.index[nn.min(axis=1) >= 5])]
            groups = ["subject", "record"] + (["prev"] if scope == "previous_condition" else [])
            for keys, g in use.groupby(groups):
                y = g.label.to_numpy()
                if len(set(y)) != 2:
                    continue
                x = values[g.index]
                au = auc_columns(x, y)
                for j, n in enumerate(cols):
                    rows.append(
                        dict(
                            task=task,
                            scope=scope,
                            subject=keys[0],
                            record=keys[1],
                            previous=keys[2] if len(keys) > 2 else -1,
                            n1=sum(y),
                            n0=len(y) - sum(y),
                            name=n,
                            raw_auc=au[j],
                            effect=x[y == 1, j].mean() - x[y == 0, j].mean(),
                        )
                    )
        hmean = h.mean(3)
        base = join(*[hmean[:, :, s].mean(2) for s in (PRE, ACTIVE, LATE, TAIL)])
        labels = np.eye(2)[m.label]
        for name, x in (
            ("Hb", base),
            ("Hb_plus_word", join(base, cols["Hb_word_timing"])),
            ("Hb_plus_slope", join(base, cols["Hb_measured_slope_timing"])),
        ):
            pr, _, reg = probe(x, labels, m.subject.to_numpy(), tr, ev, classification=True)
            for r, p in zip(m.iloc[ev].itertuples(), pr):
                models.append(
                    dict(
                        task=task,
                        subject=r.subject,
                        id=r.id,
                        label=r.label,
                        model=name,
                        pred=int(p.argmax()),
                        ridge=reg,
                    )
                )
        for i in ev:
            examples.append(
                dict(
                    task=task,
                    id=m.id[i],
                    subject=m.subject[i],
                    record=m.record[i],
                    event=m.event[i],
                    condition=m.condition[i],
                    prev=prev[i],
                    **{c: x[i] for c, x in cols.items()}
                )
            )
    df = pd.DataFrame(rows)
    df.to_csv(folder / "record_controls.csv", index=False)
    pp = (
        df.groupby(["task", "scope", "name", "subject", "record"])[["raw_auc", "effect"]]
        .mean()
        .groupby(level=[0, 1, 2, 3])
        .mean()
        .reset_index()
    )
    pp.to_csv(folder / "people.csv", index=False)
    pd.DataFrame(examples).to_csv(folder / "development_features.csv", index=False)
    pd.DataFrame(thresholds).to_csv(folder / "slope_thresholds.csv", index=False)
    d = pd.DataFrame(models)
    d.to_csv(folder / "predictions.csv.gz", index=False)
    d.groupby(["task", "subject", "model"]).apply(
        lambda g: balanced_accuracy_score(g.label, g.pred), include_groups=False
    ).rename("ba").reset_index().to_csv(folder / "model_people.csv", index=False)
    write_json(
        folder / "summary.json",
        dict(
            status="completed",
            experiment_id="TK-TASK-PATTERN-LOCALIZATION-v1",
            development_events=len(examples),
        ),
    )


if __name__ == "__main__":
    main()
