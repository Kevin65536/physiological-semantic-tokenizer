"""Audit completion and aggregate the two-seed screen at the person level."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from prepare_science_ablations import CFG, OUT, check
from run_science_ablations import ARMS, TR, read, save, schedule
from score_science_ablations import compare, paired_ci, audit_interaction_cell
from run_dual_component_tokenizer import owner_lock


def audit():
    torch.set_num_threads(1)
    meta = pd.read_csv(OUT / "crops.csv")
    rows = []
    for seed in TR["seeds"]:
        order, init = schedule(meta, seed)
        if order.shape != (1280, 16) or not (meta.iloc[order.ravel()].split == "train").all():
            raise ValueError("sample budget")
        for arm in ARMS:
            f = OUT / "models" / f"{arm}_{seed}"
            done, scored = read(f / "done.json"), read(f / "score.json")
            wg = read(OUT / "wg" / f"{arm}_{seed}" / "done.json")
            p = read(f / "progress.json")
            if not (
                done["status"] == scored["status"] == wg["status"] == "completed"
                and done["seed"] == seed
                and done["arm"] == arm
                and p["step"] == p["total"] == done["experimental_steps"] == 1280
                and [v["step"] for v in p["history"]] == list(range(128, 1281, 128))
                and np.isfinite([v["loss"] for v in p["history"]]).all()
                and scored["development_events"] == 91
                and wg["WG_events"] == 237
            ):
                raise ValueError(f"incomplete cell {arm} {seed}")
            cells = pd.read_csv(f / "readout.csv.gz", dtype={"arm": str})
            group = cells.query("reader=='slot' and representation=='full'").groupby(
                ["arm", "input", "seed", "subject"]
            )
            if len(group) != 6 * 2 * 4 or not (group.size() == 24).all() or not np.isfinite(cells.nmse).all():
                raise ValueError("reader support")
            with np.load(f / "science.npz") as z:
                levels = CFG["E3"].get(arm, [8192])
                ids = [z[f"ids{j}"] for j in range(len(levels))]
                for j, (codes, k) in enumerate(zip(ids, levels)):
                    if (
                        codes.shape != (400, 4, 2, 15)
                        or not np.issubdtype(codes.dtype, np.integer)
                        or codes.max() >= k
                    ):
                        raise ValueError("integer contract")
                # Effective joint entropy on the same dev events used in the scientific readout.
                from run_frozen_science_screen import panel

                _, _, dev = panel()
                for b in range(2):
                    joint = ids[0][dev, :, b].astype("int64")
                    for j in range(1, len(levels)):
                        joint = joint * levels[j] + ids[j][dev, :, b]
                    _, counts = np.unique(joint, return_counts=True)
                    prob = counts / counts.sum()
                    rows.append(
                        dict(
                            model=arm,
                            seed=seed,
                            branch="SO"[b],
                            nominal_bits=sum(np.log2(levels)),
                            joint_bits=float(-np.sum(prob * np.log2(prob))),
                            observed_codes=len(prob),
                            planned_vocabulary=int(np.prod(levels)),
                            trainable_parameters=done["parameters"],
                            two_branch_codebook_values=2 * sum(levels) * 128,
                            elapsed_s=done["seconds"],
                            gpu_peak_process_mib=done["peak_gpu_mib"],
                            training_position_exposures=int(order.size * 4),
                        )
                    )
    pd.DataFrame(rows).to_csv(OUT / "rates_and_cost.csv", index=False)
    return len(rows) // 2


def longitudinal_scores():
    people = pd.read_csv(OUT / "people.csv", dtype={"arm": str})
    n = people.query("representation=='full' and reader=='slot' and input=='aligned'").pivot(
        index="subject", columns=["model", "arm"], values="nmse"
    )
    w = (
        pd.read_csv(OUT / "wg/people.csv", dtype={"arm": str})
        .query("input=='aligned'")
        .pivot(index="subject", columns=["model", "arm"], values="nmse")
    )
    g = pd.read_csv(OUT / "guardrails.csv").groupby(["model", "subject"]).mean(numeric_only=True)
    t = (
        pd.read_csv(OUT / "task.csv")
        .query("representation=='full'")
        .groupby(["model", "arm", "subject"])
        .brier.mean()
        .unstack("arm")
    )
    attr = (
        pd.read_csv(OUT / "attributes.csv")
        .query("representation=='full'")
        .groupby(["model", "subject"])
        .alpha_nmse.mean()
    )
    ss = (
        pd.read_csv(OUT / "so_controlled.csv")
        .query("representation=='full'")
        .groupby(["model", "subject", "arm"])
        .nmse.mean()
        .unstack("arm")
    )
    audit_rows = []
    parts = []
    for arm in ARMS:
        v = n[arm]
        wg = w[arm]
        d = v.capacity - v.true
        dw = wg.capacity - wg.true
        values = {
            "EEG_or_joint_over_capacity": d,
            "over_full_Hb": v.anchor - v.true,
            "over_wrong_training": v.wrong - v.true,
            "change_vs_B0": d - (n["B0"].capacity - n["B0"].true),
            "WG_over_capacity": dw,
            "WG_over_full_Hb": wg.anchor - wg.true,
            "WG_over_wrong": wg.wrong - wg.true,
            "WG_change_vs_B0": dw - (w["B0"].capacity - w["B0"].true),
            "alpha_error_gain_vs_B0": attr.loc["B0"] - attr.loc[arm],
            "waveform_error_gain_vs_B0": g.loc["B0"].x_nmse - g.loc[arm].x_nmse,
            "task_E_given_H": t.loc[arm].H - t.loc[arm].HE,
            "task_H_given_E": t.loc[arm].E - t.loc[arm].HE,
            "S_gain_from_O": ss.loc[arm].S - ss.loc[arm].additive,
            "O_gain_from_S": ss.loc[arm].O - ss.loc[arm].additive,
            "extra_interaction": ss.loc[arm].additive - ss.loc[arm].interaction_true_aligned,
        }
        ci = {k: paired_ci(vv) for k, vv in values.items()}
        for k, vv in ci.items():
            parts.append(dict(model=arm, endpoint=k, **vv))
        # Descriptive conjunction; historical development exposure precludes confirmation.
        support = all(
            ci[k]["ci_low"] > 0
            for k in ("change_vs_B0", "EEG_or_joint_over_capacity", "over_full_Hb", "over_wrong_training")
        )
        transfer = all(
            ci[k]["ci_low"] > 0
            for k in ("WG_change_vs_B0", "WG_over_capacity", "WG_over_wrong", "WG_over_full_Hb")
        )
        audit_rows.append(
            dict(
                model=arm,
                source=(
                    "joint EEG/Hb"
                    if arm in ("cross_true", "cross_wrong")
                    else ("Hb only" if arm == "cross_hb_only" else "EEG only")
                ),
                **{k: v["delta"] for k, v in ci.items()},
                positive_paired_lower_bounds=bool(support),
                positive_WG_lower_bounds=bool(transfer),
            )
        )
    df = pd.DataFrame(audit_rows)
    df.to_csv(OUT / "unified_endpoints.csv", index=False)
    pd.DataFrame(parts).to_csv(OUT / "endpoint_intervals.csv", index=False)
    return df, n


def factorial(n):
    delta = {a: n[a].capacity - n[a].true for a in ARMS}
    f = {
        "S_off_with_O_on": delta["s_off"] - delta["B0"],
        "S_off_with_O_off": delta["both_off"] - delta["o_off"],
        "O_off_with_S_on": delta["o_off"] - delta["B0"],
        "O_off_with_S_off": delta["both_off"] - delta["s_off"],
        "S_by_O": delta["both_off"] - delta["o_off"] - delta["s_off"] + delta["B0"],
        "extra_bit_single": delta["single14"] - delta["B0"],
        "extra_bit_dual": delta["dual14"] - delta["dual13"],
        "dual_at_13": delta["dual13"] - delta["B0"],
        "dual_at_14": delta["dual14"] - delta["single14"],
        "rate_by_organization": delta["dual14"] - delta["dual13"] - delta["single14"] + delta["B0"],
        "first_attribute_vs_joint": delta["dual13_first"] - delta["dual13"],
    }
    pd.DataFrame([dict(contrast=k, **paired_ci(v)) for k, v in f.items()]).to_csv(
        OUT / "factorial.csv", index=False
    )


def decomposition_checks():
    from run_science_ablations import arrays, create, extract

    meta, data = arrays()
    idx = np.flatnonzero((meta.split == "validation") & (meta.donor >= 0))
    results = []
    for arm in CFG["E2"]:
        waves = {}
        for seed in TR["seeds"]:
            model = create(arm, seed, "cpu").eval()
            state = torch.load(
                OUT / "models" / f"{arm}_{seed}" / "state.pt", map_location="cpu", weights_only=False
            )
            model.load_state_dict(state["model"])
            z = extract(model, data["x"][idx], data["h"][idx], "cpu", metrics=True)
            waves[seed] = [z[b].reshape(len(idx), 4, -1) for b in "SO"]
            s, o = waves[seed]
            es = (s * s).mean(2)
            eo = (o * o).mean(2)
            total = ((s + o) ** 2).mean(2)
            ratio = (es + eo) / np.maximum(total, 1e-9)
            share = es / np.maximum(es + eo, 1e-9)
            dot = (s * o).mean(2) / np.sqrt(np.maximum(es * eo, 1e-12))
            for person, ii in meta.iloc[idx].reset_index(drop=True).groupby("subject").indices.items():
                results.append(
                    dict(
                        model=arm,
                        seed=seed,
                        subject=person,
                        site_crops=share[ii].size,
                        cancellation_median=np.median(ratio[ii]),
                        cancellation_p95=np.quantile(ratio[ii], 0.95),
                        min_cosine=float(dot[ii].min()),
                        opposed_cosine_below_minus_point8=int((dot[ii] < -0.8).sum()),
                        near_single_branch_count=int(((share[ii] < 0.05) | (share[ii] > 0.95)).sum()),
                        s_energy_share=share[ii].mean(),
                    )
                )
        xenergy = (data["x"][idx] ** 2).mean((2, 3))
        for person, ii in meta.iloc[idx].reset_index(drop=True).groupby("subject").indices.items():
            changes = [
                (((one - two) ** 2).mean(2) / np.maximum(xenergy, 1e-9))[ii].mean()
                for one, two in zip(waves[17], waves[43])
            ]
            results.append(
                dict(
                    model=arm,
                    seed="paired",
                    subject=person,
                    site_crops=len(ii) * 4,
                    seed_difference_S_over_X_energy=changes[0],
                    seed_difference_O_over_X_energy=changes[1],
                )
            )
    pd.DataFrame(results).to_csv(OUT / "E2_stability.csv", index=False)
    matching = []
    for (person, cond), g in meta[meta.donor >= 0].groupby(["subject", "condition"]):
        src = meta.iloc[g.donor]
        if not np.all(src.subject.values == person) or not np.all(src.condition.values == cond):
            raise ValueError("sham source mismatched conditioning stratum")
        matching.append(
            dict(
                subject=person,
                condition=cond,
                split=g.split.iloc[0],
                crops=len(g),
                distinct_donors=g.donor.nunique(),
                minimum_separation_s=np.min(abs(g.start.values - src.start.values)),
                median_phase_distance_s=np.median(abs(g.phase.values - src.phase.values)),
            )
        )
    pd.DataFrame(matching).to_csv(OUT / "teacher_donor_matching.csv", index=False)


def full_chain():
    """Refit exactly the same linear probe for every encoder-side modality intervention."""
    import run_frozen_science_screen as screen
    from scipy.linalg import solve
    from src.analysis.frozen_science_screen import centered, gram
    from score_science_ablations import normalized

    m, tr, ev = screen.panel()
    y = screen.array("target").reshape(len(m), -1).astype(float)
    ym = y[tr].mean(0)
    sd = y[tr].std(0)
    h = gram(screen.hb_slots(screen.array("hb"), tr))
    rows = []
    for arm in ("cross_true", "cross_wrong", "cross_eeg_only", "cross_hb_only"):
        for seed in TR["seeds"]:
            with np.load(OUT / "models" / f"{arm}_{seed}" / "science.npz") as z:
                aligned = z["level0"]
                wrong = z["exchange_wrong"]
                for scope, fitvec, evalvec, h_present in [
                    ("real_exchange", wrong if arm == "cross_wrong" else aligned, aligned, True),
                    ("wrong_exchange", wrong if arm == "cross_wrong" else aligned, wrong, True),
                    ("remove_EEG", z["without_eeg"], z["without_eeg"], True),
                    ("remove_Hb", z["without_hb"], z["without_hb"], False),
                ]:
                    vectors = fitvec.copy()
                    vectors[ev] = evalvec[ev]
                    k = gram(normalized(vectors, tr))
                    a = centered(k, tr, tr)
                    b = centered(k, tr, ev)
                    if h_present:
                        a += centered(h, tr, tr)
                        b += centered(h, tr, ev)
                    pred = ym + b @ solve(a + np.eye(len(tr)), y[tr] - ym, assume_a="pos")
                    err = (((pred - y[ev]) / sd) ** 2).mean(1)
                    for who, idx in m.iloc[ev].reset_index(drop=True).groupby("subject").indices.items():
                        rows.append(
                            dict(model=arm, seed=seed, input=scope, subject=who, nmse=err[idx].mean())
                        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "full_chain.csv", index=False)
    return df


def time_contrasts():
    folder = OUT / "time_views"
    df = pd.read_csv(folder / "person_scores.csv", dtype={"arm": str})
    m = df.query("input=='aligned'").pivot(index="subject", columns=["view", "arm"], values="nmse")
    result = []
    for v in ("history", "bidir"):
        for ref in ("anchor", "capacity", "wrong"):
            result.append(dict(view=v, contrast=ref + "_minus_true", **paired_ci(m[v][ref] - m[v].true)))
    result.append(
        dict(view="paired", contrast="bidir_minus_history", **paired_ci(m.bidir.true - m.history.true))
    )
    pd.DataFrame(result).to_csv(folder / "contrasts.csv", index=False)
    fam = pd.read_csv(folder / "family_effects.csv")
    rows = []
    for b, g in fam.groupby("branch"):
        for col in ("posterior_minus_frontal", "load_difference", "transition"):
            mat = g.pivot(index="subject", columns="view", values=col).dropna()
            if len(mat):
                rows.append(dict(branch=b, endpoint=col, **paired_ci(mat.history - mat.bidir)))
    pd.DataFrame(rows).to_csv(folder / "paired_family_effects.csv", index=False)


def atlas():
    p = pd.read_csv(OUT / "attributes.csv").query("representation=='full'")
    p = p.groupby(["model", "branch", "subject"], as_index=False).mean(numeric_only=True)
    rows = []
    for (model, b), g in p.groupby(["model", "branch"]):
        for name in ("posterior_minus_frontal", "high_minus_low_load", "high_persistence"):
            val = g[name].dropna()
            if len(val):
                rows.append(dict(model=model, branch=b, metric=name, **paired_ci(val)))
    for arm in ARMS:
        tables = []
        for s in TR["seeds"]:
            path = OUT / "models" / f"{arm}_{s}" / "readout.csv.gz"
            tables.append(
                pd.read_csv(path, dtype={"arm": str}).query(
                    "representation=='full' and reader=='slot' and input=='aligned'"
                )
            )
        v = pd.concat(tables).groupby(["subject", "lag_s", "arm"]).nmse.mean().unstack("arm")
        for lag, g in v.groupby(level="lag_s"):
            rows.append(
                dict(
                    model=arm,
                    branch="joint",
                    metric=f"future_Hb_gain_{lag}s",
                    **paired_ci(g.capacity - g.true),
                )
            )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "atlas_effects.csv", index=False)
    b0 = table.query("model=='B0'").set_index(["branch", "metric"])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
    specs = [
        (
            [("S", "posterior_minus_frontal"), ("O", "posterior_minus_frontal")],
            ["S", "O"],
            "Posterior minus frontal",
            "High-family occupancy difference",
        ),
        (
            [("joint", f"future_Hb_gain_{i}s") for i in (2, 4, 6)],
            ["2", "4", "6"],
            "Future Hb horizon (s)",
            "Hb capacity minus paired NMSE",
        ),
        (
            [("S", "high_persistence"), ("O", "high_persistence")],
            ["S", "O"],
            "Consecutive slots",
            "P(high next | high now)",
        ),
    ]
    for ax, (keys, labels, name, unit) in zip(axes, specs):
        g = b0.loc[keys]
        ax.errorbar(
            np.arange(len(g)),
            g.delta,
            yerr=np.stack([g.delta - g.ci_low, g.ci_high - g.delta]),
            fmt="o",
            capsize=4,
        )
        ax.axhline(0, color="grey", lw=0.7)
        ax.set_xticks(np.arange(len(g)), labels)
        ax.set_title(name)
        ax.set_ylabel(unit)
        ax.set_xlim(-0.5, len(g) - 0.5)
    fig.suptitle("B0 · n-back development · 4 people · paired person-bootstrap 95% intervals")
    fig.savefig(OUT / "B0_atlas.png", dpi=240)
    plt.close(fig)


def decision_table():
    """One comparison per intervention, preserving the endpoint and reference for each modality."""
    from run_science_hb_replication import ARMS as hb_arms

    rows = []
    for modality, root, models in [
        ("EEG", OUT, ARMS + ["d64", "d128", "d256"]),
        (
            "Hb",
            OUT / "hb_replication",
            hb_arms,
        ),
    ]:
        if modality == "EEG":
            n = pd.read_csv(root / "people.csv", dtype={"arm": str}).query(
                "representation=='full' and reader=='slot' and input=='aligned'"
            )
            w = pd.read_csv(root / "wg/people.csv", dtype={"arm": str}).query("input=='aligned'")
            g = pd.read_csv(root / "guardrails.csv")
            dim = pd.read_csv(root / "dimension_reconstruction.csv")
            g = pd.concat([g, dim], ignore_index=True)
            att = (
                pd.read_csv(root / "attributes.csv")
                .query("representation=='full'")
                .assign(measured=lambda t: t.alpha_nmse)
            )
        else:
            n = pd.read_csv(root / "people.csv", dtype={"arm": str}).query("input=='aligned'")
            w = pd.read_csv(root / "wg_people.csv", dtype={"arm": str}).query("input=='aligned'")
            g = pd.read_csv(root / "guardrails.csv")
            att = pd.read_csv(root / "attributes.csv").assign(
                measured=lambda t: (t.hbo_slope_nmse + t.hbr_slope_nmse) / 2
            )

        def scores(d):
            return d.groupby(["model", "subject", "arm"]).nmse.mean().unstack("arm")

        ns = scores(n)
        ws = scores(w)
        guard = g.groupby(["model", "subject"]).x_nmse.mean()
        attr = att.groupby(["model", "subject"]).measured.mean()

        def increment(tab, model):
            d = tab.loc[model]
            return d.capacity - d.true

        for model in models:
            ref = "d128" if model in ("d64", "d128", "d256") else "B0"
            if model == ref:
                continue
            d = ns.loc[model]
            delta = paired_ci(increment(ns, model) - increment(ns, ref))
            measured = paired_ci(attr.loc[ref] - attr.loc[model])
            rec = paired_ci(guard.loc[ref] - guard.loc[model])
            ctr = [paired_ci(d[col] - d.true) for col in ("capacity", "anchor", "wrong")]
            wg = None
            if model in ws.index.get_level_values("model"):
                wg = paired_ci(increment(ws, model) - increment(ws, ref))
            if delta["ci_low"] > 0 and min(v["ci_low"] for v in ctr) > 0:
                decision = (
                    "开发组条件增量和独立属性均改善" if measured["ci_low"] > 0 else "开发组条件增量改善"
                )
            elif wg and delta["ci_high"] < 0 and wg["ci_high"] < 0:
                decision = "两任务配对条件增量均劣于基线"
            elif measured["ci_low"] > 0:
                decision = "实测属性读出改善；条件增量未确立"
            elif rec["ci_low"] > 0:
                decision = "重建误差下降；条件增量未确立"
            else:
                decision = "共同科学终点未定"
            row = dict(
                modality=modality,
                model=model,
                reference=ref,
                decision=decision,
                people=delta["people"],
                source="joint" if model in ("cross_true", "cross_wrong") else "single_modality_encoder",
            )
            for k, v in [
                ("conditional_change", delta),
                ("over_capacity", ctr[0]),
                ("over_full_Hb", ctr[1]),
                ("over_wrong", ctr[2]),
                ("measured_attribute_gain", measured),
                ("observed_reconstruction_gain", rec),
                ("WG_conditional_change", wg),
            ]:
                if v:
                    row.update({k: v["delta"], k + "_lo": v["ci_low"], k + "_hi": v["ci_high"]})
            rows.append(row)
    tc = pd.read_csv(OUT / "time_views/contrasts.csv")
    row = tc.query("contrast=='bidir_minus_history'").iloc[0]
    rows.append(
        dict(
            modality="EEG",
            model="history_view",
            reference="bidir_view",
            decision="严格历史视图的条件增量差区间跨零，合成延迟未改善",
            people=4,
            conditional_change=row.delta,
            conditional_change_lo=row.ci_low,
            conditional_change_hi=row.ci_high,
            source="same_local_patch",
        )
    )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "decision_table.csv", index=False)
    return dict(table.decision.value_counts().astype(int).items())


def main():
    check()
    with owner_lock(OUT, "final"):
        count = audit()
        compare()
        tasks = [(arm, s) for arm in ARMS for s in TR["seeds"]] + [
            ("dimension", dim) for dim in CFG["E3A"]["dimensions"]
        ]
        with ProcessPoolExecutor(6, mp_context=mp.get_context("fork")) as pool:
            controlled = pd.concat(pool.map(audit_interaction_cell, tasks))
        controlled.to_csv(OUT / "so_controlled.csv", index=False)
        from src.analysis.conditional_evidence import effect_tables

        people = controlled.groupby(["model", "representation", "subject", "arm"], as_index=False).nmse.mean()
        effect_tables(
            people,
            "nmse",
            [
                ("O_given_S", "S", "additive"),
                ("S_given_O", "O", "additive"),
                ("beyond_additive", "additive", "interaction_true_aligned"),
                ("beyond_additive_expansion", "additive_expansion", "interaction_true_aligned"),
                ("beyond_wrong", "interaction_wrong_aligned", "interaction_true_aligned"),
            ],
            ["model", "representation"],
        ).to_csv(OUT / "so_contrasts.csv", index=False)
        df, n = longitudinal_scores()
        factorial(n)
        decomposition_checks()
        full_chain()
        time_contrasts()
        atlas()
        decisions = decision_table()
        hb = read(OUT / "hb_replication/summary.json")
        if hb["status"] != "completed" or hb["completed_fits"] != 24:
            raise ValueError("incomplete Hb replication")
        w = read(OUT / "wg/summary.json")
        timing = read(OUT / "time_views/summary.json")
        if count != 44 or w["models"] != 44 or w["status"] != "completed" or timing["status"] != "completed":
            raise ValueError("incomplete suite")
        eligible = df.query(
            "model!='B0' and positive_paired_lower_bounds and positive_WG_lower_bounds"
        ).model.tolist()
        save(
            OUT / "summary.json",
            dict(
                status="completed",
                experiment_id=CFG["experiment_id"],
                planned_fits=68,
                completed_fits=count + hb["completed_fits"],
                EEG_fits=count,
                Hb_fits=hb["completed_fits"],
                Hb_replication=hb,
                decision_counts={key: int(n) for key, n in decisions.items()},
                seeds=TR["seeds"],
                training_steps_per_fit=1280,
                position_exposures_per_fit=81920,
                historical_dimension_models=3,
                endpoint=CFG["primary"],
                reference="two matched short refits; original full-budget tokenizer separate",
                training_subjects=15,
                development_subjects=4,
                train_events=309,
                dev_events=91,
                train_crops=463,
                dev_crops=124,
                cross_task=w,
                time_views=timing,
                scientific_verdict="exploratory",
                independent_confirmation="not evaluated",
                descriptive_screen_and_WG_conjunction=eligible,
                selection_note="Descriptive 95% paired bootstrap intervals; no multiplicity correction. WG contains only one additional person. Positive point estimates alone do not establish improvements.",
                files=[
                    "decision_table.csv",
                    "unified_endpoints.csv",
                    "endpoint_intervals.csv",
                    "paired_ablation.csv",
                    "factorial.csv",
                    "increments.csv",
                    "guardrails.csv",
                    "E2_stability.csv",
                    "teacher_donor_matching.csv",
                    "rates_and_cost.csv",
                    "so_controlled.csv",
                    "so_contrasts.csv",
                    "full_chain.csv",
                    "exchange.csv",
                    "task.csv",
                    "attributes.csv",
                    "atlas_effects.csv",
                    "wg/contrasts.csv",
                    "hb_replication/paired.csv",
                    "hb_replication/so_contrasts.csv",
                    "time_views/person_scores.csv",
                    "time_views/synthetic_lag.csv",
                ],
            ),
        )


if __name__ == "__main__":
    main()
