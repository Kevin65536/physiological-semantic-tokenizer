"""Produce paired descriptive summaries, independent annotation controls, and blinded member material."""

import json
import numpy as np
import pandas as pd
import audit_aligned_tokenizer as audit
from analyze_aligned_token_controls import save
from src.analysis.single_vq_relations import interval
from src.analysis.observation_task_probe import DOWNSTREAM_ROOT

OUT = audit.OUT


def aggregate():
    rng = np.random.default_rng(1126)
    # Use a common event denominator for both pseudo clocks, shuffle condition assignments within session.
    d = pd.read_csv(OUT / "pseudo_cues.csv")
    index = ["subject", "record", "event", "condition"]
    both = d.pivot(index=index, columns="offset", values="delta").dropna().reset_index()

    def per_subject(frame, key):
        s = frame.groupby(["subject", "record", "condition"])[key].mean().unstack().dropna()
        return (s.MA - s.BL).groupby("subject").mean()

    place = []
    for k in [-13, 0, 13]:
        people = per_subject(both, k)
        for sub, v in people.items():
            place.append(dict(offset=k, subject=sub, contrast=v, n=int((both.subject == sub).sum())))
    save(place, "pseudo_contrasts")
    draws = []
    for _ in range(499):
        b = both.copy()
        b["condition"] = b.groupby(["subject", "record"]).condition.transform(lambda v: rng.permutation(v))
        draws.append(per_subject(b, 0).mean())
    obs = per_subject(both, 0).mean()
    (OUT / "cue_permutation.json").write_text(
        json.dumps(
            dict(
                common=len(both),
                people=len(per_subject(both, 0)),
                observed=obs,
                permutations=len(draws),
                two_sided_p=(1 + np.sum(np.abs(draws) >= abs(obs) - 1e-14)) / (1 + len(draws)),
                seed=1126,
            )
        )
    )
    # Separate the conditional cue comparison and the broad same-window paired task classifier.
    pop = (
        pd.read_csv(OUT / "paired_panel.csv", dtype={"subject": str})
        .query("status=='complete'")
        .reset_index(drop=True)
    )
    a = np.load(OUT / "paired_features.npz")
    lookup = {}
    for s in json.loads((DOWNSTREAM_ROOT / "input_plan.json").read_text())["samples"]:
        if s["dataset"] == "refed" and s["split"] in ["train", "validation"] and not s["partial"]:
            ev = s["event"]
            stream = ev["metadata"]["continuous_label_stream"]
            assert stream["names"] == ["valence", "arousal"] and stream["native_sample_rate_hz"] == 1
            lookup[s["split"], s["subject"], s["record"]] = (
                np.array(stream["values"]),
                ev["eeg_time_ms"] / 1000,
            )
    tbl = []
    for i, r in pop[pop.dataset.eq("refed")].iterrows():
        if (r.split, r.subject, r.record) not in lookup:
            continue
        val, origin = lookup[r.split, r.subject, r.record]
        offset = r.eeg_start - origin
        if abs(offset - round(offset)) > 0.15 or offset < 0 or round(offset) + 30 > len(val):
            continue
        y = val[round(offset) : round(offset) + 30].reshape(3, 10, 2).mean(1)
        for j in range(3):
            tbl.append(
                dict(
                    split=r.split,
                    subject=r.subject,
                    record=r.record,
                    start=r.eeg_start + 10 * j,
                    arousal=y[j, 1],
                    valence=y[j, 0],
                    **{m + b: a[m + b][i, :, j, 3].mean() for m in ("eeg", "hb") for b in "SO"},
                )
            )
    t = pd.DataFrame(tbl)
    t.to_csv(OUT / "refed_annotations.csv", index=False)
    scores = []
    for name in ["arousal", "valence"]:
        for sub, g in t[t.split == "validation"].groupby("subject"):
            centered = g[[name, "eegS", "eegO", "hbS", "hbO"]] - g.groupby("record")[
                [name, "eegS", "eegO", "hbS", "hbO"]
            ].transform("mean")
            for k in ("eegS", "eegO", "hbS", "hbO"):
                scores.append(
                    dict(
                        subject=sub,
                        attribute=name,
                        branch=k,
                        windows=len(g),
                        records=g.record.nunique(),
                        r=centered[name].corr(centered[k]),
                    )
                )
    save(scores, "refed_state")
    # Use the native sample/baseline structure of the completed 30-second crop, reports on people.
    fusion = pd.read_csv(OUT / "fusion.csv")
    contrasts = []
    for mode, g in fusion.groupby("mode"):
        for metric in ["ba", "brier"]:
            table = g.pivot(index="subject", columns="model", values=metric)
            for base, cand in [
                ("eeg", "linear"),
                ("hb", "linear"),
                ("late", "linear"),
                ("nonlinear_main", "interaction"),
            ]:
                diff = (table[cand] - table[base]).dropna() * (1 if metric == "ba" else -1)
                avg, lo, hi = interval(diff, np.zeros(len(diff)))
                contrasts.append(
                    dict(
                        mode=mode,
                        metric=metric,
                        reference=base,
                        candidate=cand,
                        people=len(diff),
                        benefit=avg,
                        ci_low=lo,
                        ci_high=hi,
                    )
                )
    save(contrasts, "fusion_contrasts")
    ranking = pd.read_csv(OUT / "paired_retrieval.csv")
    ranking["chance_MRR"] = [sum(1 / np.arange(1, n + 1)) / n for n in ranking.candidates]
    ranking.groupby(["source", "subject"])[
        ["R1", "chance", "MRR", "chance_MRR"]
    ].mean().reset_index().to_csv(OUT / "ranking_people.csv", index=False)
    dose = pd.read_csv(OUT / "nback_dose.csv")
    dose["load"] = dose.condition.str[0].astype(int)
    dose.groupby(["split", "subject", "load"])[
        ["eeg_Q4", "hb_Q4", "eeg_obs", "hb_obs"]
    ].mean().reset_index().to_csv(OUT / "dose_people.csv", index=False)
    occ = []
    for mod in ("eeg", "hb"):
        for b in "SO":
            w = pd.read_csv(OUT / f"{mod}_17/{b}_words.csv")
            for role in ("train", "dev"):
                n = w[role + "_patches"].to_numpy()
                p = n[n > 0] / n.sum()
                h = -np.sum(p * np.log2(p))
                occ.append(
                    dict(
                        mod=mod,
                        branch=b,
                        split=role,
                        capacity=len(w),
                        used=len(p),
                        words_eligible=w.eligible.sum(),
                        words_readable=w.readable.sum(),
                        patches=n.sum(),
                        entropy_bits=h,
                        perplexity=2**h,
                    )
                )
    save(occ, "occupancy")


def blind():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import fitz
    import io

    out = OUT / "blind"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(1126)
    items = []
    for mod in ("eeg", "hb"):
        for b in "SO":
            root = OUT / f"{mod}_17"
            meta = pd.read_csv(root / f"{b}_members.csv")
            with np.load(root / f"{b}_member_waves.npz") as y:
                for (g, word), group in meta[meta.split == "validation"].groupby(["group", "word"]):
                    for kind in ("member", "background"):
                        v = group[group.kind == kind]
                        i = rng.integers(len(v))
                        r = v.iloc[i]
                        curve = (
                            y[f"{g}_{word}_validation_{kind}"][r["pair"], 0]
                            .copy()
                            .reshape(-1, 1 if mod == "eeg" else 2)
                        )
                        items.append(
                            (
                                dict(
                                    mod=mod,
                                    branch=b,
                                    group=g,
                                    word=word,
                                    kind=kind,
                                    subject=r.subject,
                                    parent=r.parent,
                                    position=r.position,
                                    patch=r.patch,
                                ),
                                curve,
                            )
                        )
    pdf = fitz.open()
    key = []
    form = []
    for j, idx in enumerate(rng.permutation(len(items))):
        row, x = items[idx]
        code = f"B{j+1:03}"
        if j % 4 == 0:
            pg = pdf.new_page(width=812, height=620)
        fig, ax = plt.subplots(figsize=(5.1, 3.3), layout="constrained")
        for color, w in enumerate(x.T):
            lab = "EEG" if row["mod"] == "eeg" else ["HbO", "HbR"][color]
            ax.plot(
                np.arange(len(w)) / (200 if row["mod"] == "eeg" else 10),
                w,
                label=lab,
                c=["#155684", "#ad4d30"][color],
                ls=["-", "--"][color],
                lw=1,
            )
        ax.set(title=code, xlabel="Time (s)", ylabel="Training input SD")
        ax.legend()
        buf = io.BytesIO()
        fig.savefig(buf, dpi=230, format="png")
        plt.close(fig)
        c = j % 4 % 2
        r = (j % 4) // 2
        pg.insert_image(
            fitz.Rect(6 + c * 400, 10 + r * 305, 402 + c * 400, 309 + r * 305), stream=buf.getvalue()
        )
        key.append(dict(case=code, **row))
        form.append(
            dict(case=code, modality=row["mod"], morphology="", quality="", confidence="", rater="")
        )
    pdf.save(out / "blinded_waveforms.pdf", deflate=True)
    pdf.close()
    pd.DataFrame(key).to_csv(OUT / "blinding_key.csv", index=False)
    pd.DataFrame(form).to_csv(out / "ratings.csv", index=False)
    (out / "instructions.txt").write_text(
        "请两位评阅者分别填写 ratings.csv。每例为留出被试的独立呈现片段。单位为训练观测标准差，短横线表示 HbR。EEG 形态可选节律、瞬态、慢变、混合、不确定；Hb 形态可选同向、反向、单色主导、平坦、不确定。quality 填可判/不可判，confidence 填 1–3，rater 填匿名评阅者编号。先独立提交，再按模态计算含不确定类的一致率与 Cohen κ。样例用于定性审阅，同一受试者或窗口可重复。解盲表另存。\n"
    )


if __name__ == "__main__":
    aggregate()
    blind()
