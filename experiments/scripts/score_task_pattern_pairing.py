"""Within-condition empirical early EEG versus delayed Hb response probabilities."""

import numpy as np
import pandas as pd
import yaml
from run_task_pattern_expansion import ROOT, CFG, OUT, CONTRACT
from src.data.task_pattern_panel import load_task_panel
from src.analysis.task_pattern_expansion import disjoint_donors
from src.analysis.task_pattern_expansion import ACTIVE, LATE, features
from src.utils.io import write_json

CON = CONTRACT.with_name("task_pattern_pairing_v1.yaml")
RULE = yaml.safe_load(CON.read_text())


def main():
    folder = OUT / RULE["output_child"]
    folder.mkdir(exist_ok=True)
    (folder / "resolved.yaml").write_text(CON.read_text())
    totals = []
    allrows = []
    scales = []
    coverage = []
    for task in CFG["tasks"]:
        _, m, a = load_task_panel(OUT, task)
        d = disjoint_donors(m)
        h = a["h"].reshape(len(m), 4, 15, 20, 2).mean(3)
        attributes = {
            k[1]: v for k, v in features(a["e"], h, a["ef"], a["hf"]).items() if k[0] in ["EEG", "EEGword"]
        }
        slow = h[:, :, LATE].mean(2) - h[:, :, ACTIVE].mean(2)
        observed = {
            "frontal_HbO": slow[:, :2, 0].mean(1),
            "frontal_HbR": slow[:, :2, 1].mean(1),
            "motor_HbO_lr": slow[:, 2, 0] - slow[:, 3, 0],
        }
        for name, src, d1, tgt, d2 in RULE["candidates"]:
            x = attributes[src] * d1
            y = observed[tgt] * d2
            train = m.split.eq("train") & (d >= 0)
            lo = float(np.median(x[train]))
            hi = float(np.median(y[train]))
            scales.append(
                dict(
                    task=task,
                    pair=name,
                    x_threshold=lo,
                    y_threshold=hi,
                    x_sign=d1,
                    y_sign=d2,
                    training_events=int(train.sum()),
                )
            )
            event = x > lo
            response = y > hi
            for (split, sub, rec, condition), idx in m.groupby(
                ["split", "subject", "record", "condition"]
            ).indices.items():
                idx = idx[d[idx] >= 0]
                on = idx[event[idx]]
                off = idx[~event[idx]]
                ok = min(len(on), len(off)) >= 2
                coverage.append(
                    dict(
                        task=task,
                        pair=name,
                        split=split,
                        subject=sub,
                        record=rec,
                        condition=condition,
                        on=len(on),
                        off=len(off),
                        supported=ok,
                    )
                )
                if ok:
                    pon = response[on].mean()
                    poff = response[off].mean()
                    wrong = response[d[on]].mean() - response[d[off]].mean()
                    allrows.append(
                        dict(
                            task=task,
                            pair=name,
                            split=split,
                            subject=sub,
                            record=rec,
                            condition=condition,
                            on=len(on),
                            off=len(off),
                            p_on=pon,
                            p_off=poff,
                            diff=pon - poff,
                            wrong=wrong,
                        )
                    )
    df = pd.DataFrame(allrows)
    pp = (
        df.groupby(["task", "pair", "split", "subject", "record"])[["diff", "wrong", "p_on", "p_off"]]
        .mean()
        .groupby(level=[0, 1, 2, 3])
        .mean()
        .reset_index()
    )
    for (task, name), g in pp.groupby(["task", "pair"]):
        tr = g[g.split == "train"]
        dev = g[g.split == "validation"]
        s = 1 if tr["diff"].mean() >= 0 else -1
        v = dev["diff"] * s
        inc = (dev["diff"] - dev.wrong) * s
        trainpass = abs(tr["diff"].mean()) >= 0.10 and (s * tr["diff"] > 0).mean() >= 2 / 3
        totals.append(
            dict(
                task=task,
                pair=name,
                train_selected=trainpass,
                train_effect=tr["diff"].mean(),
                dev_effect=dev["diff"].mean(),
                wrong=dev.wrong.mean(),
                aligned_minus_wrong=inc.mean(),
                dev_people=len(dev),
                positive=(v > 0).sum(),
                p_on=dev.p_on.mean(),
                p_off=dev.p_off.mean(),
                passes=bool(
                    trainpass
                    and len(dev) >= 3
                    and v.mean() >= 0.10
                    and inc.mean() >= 0.05
                    and (inc > 0).mean() >= 0.75
                    and (v > 0).mean() >= 0.75
                ),
            )
        )
    out = pd.DataFrame(totals)
    df.to_csv(folder / "record_condition.csv", index=False)
    pp.to_csv(folder / "people.csv", index=False)
    pd.DataFrame(coverage).to_csv(folder / "support.csv", index=False)
    pd.DataFrame(scales).to_csv(folder / "thresholds.csv", index=False)
    out.to_csv(folder / "contrasts.csv", index=False)
    write_json(
        folder / "summary.json",
        dict(
            status="completed",
            experiment_id=RULE["experiment_id"],
            planned=len(CFG["tasks"]) * len(RULE["candidates"]),
            scored=len(out),
            passed=out.loc[out.passes, ["task", "pair"]].to_dict("records"),
        ),
    )


if __name__ == "__main__":
    main()
