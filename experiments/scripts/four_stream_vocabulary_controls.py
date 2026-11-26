"""Conditional amplitude/frequency controls for frozen Hb-S/O word semantics."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time
import numpy as np
import pandas as pd

from analyze_four_stream_transfer import atlas_inputs, hb_patch_blocks
import run_four_stream_evidence as stage
from src.analysis.hb_word_organization import quadrants, word_census, composition, frequency_match

WORK = {}
FOLDER = stage.OUT / "words"


def collect():
    rows, scales = atlas_inputs()
    parts = []
    for role, split in enumerate(["train", "validation"]):
        for frame in hb_patch_blocks(split, rows, scales):
            parts.append(frame.assign(role=role))
    full = pd.concat(parts, ignore_index=True)
    cols = ["role", "dataset", "subject", "position", "rms_bin"]
    groups = list(full.groupby(cols, sort=False).indices.values())
    y = full[["oxy", "deoxy"]].to_numpy()
    persons = full[["role", "dataset", "subject"]].drop_duplicates().reset_index(drop=True)
    pid = full.merge(
        persons.reset_index(names="person"),
        on=["role", "dataset", "subject"],
        sort=False,
        validate="many_to_one",
    ).person.to_numpy()
    ds, levels = pd.factorize(persons.dataset)
    zone = np.array(
        [np.quantile(np.abs(y[(full.dataset == v) & (full.role == 0)]), 0.10, axis=0) for v in levels]
    )
    pd.DataFrame(zone, columns=["HbO", "HbR"], index=levels).to_csv(FOLDER / "training_deadzone.csv")
    baselines = []
    for i, d in enumerate(levels):
        for split in (0, 1):
            vals = y[(full.dataset == d) & (full.role == split)]
            for label, tol in [("sign", np.zeros(2)), ("q10", zone[i])]:
                votes = np.bincount(quadrants(vals, tol) + 1, minlength=5)
                baselines.append(
                    dict(
                        dataset=d,
                        split=split,
                        zone=label,
                        patches=len(vals),
                        uncertain=votes[0],
                        **{f"q{j}": votes[j + 1] for j in range(4)},
                    )
                )
    pd.DataFrame(baselines).to_csv(FOLDER / "dataset_background.csv", index=False)
    plans = {}
    for b in "SO":
        key = pid * 1024 + full[b].to_numpy(dtype=int)
        codes, inv, n = np.unique(key, return_inverse=True, return_counts=True)
        who = codes // 1024
        plans[b] = dict(
            inv=inv, n=n, who=who, word=codes % 1024, role=persons.role.to_numpy()[who], ds=ds[who]
        )
    WORK.update(
        y=y,
        groups=groups,
        plans=plans,
        persons=persons,
        zone=zone,
        background=baselines,
        levels=levels,
        ds=ds,
    )
    return len(full)


def one(draw):
    y = WORK["y"]
    index = np.arange(len(y))
    if draw:
        rng = np.random.default_rng(stage.CFG["semantics"]["permutation_seed"] + draw)
        for g in WORK["groups"]:
            index[g] = rng.permutation(g)
    values = pd.DataFrame(y[index], columns=["oxy", "deoxy"])
    words = []
    obs = []
    for b, plan in WORK["plans"].items():
        med = values.groupby(plan["inv"], sort=True).median().to_numpy()
        who = plan["who"]
        n = plan["n"]
        w = plan["word"]
        role = plan["role"]
        for zone, threshold in [("sign", 0.0), ("q10", WORK["zone"][plan["ds"]])]:
            label = quadrants(med, threshold)
            tab = word_census(who, w, role, n, label)
            words.append(tab.assign(branch=b, zone=zone))
            if draw == 0:
                people = WORK["persons"].iloc[who].reset_index(drop=True)
                obs.append(
                    people.assign(
                        branch=b,
                        word=w,
                        patches=n,
                        zone=zone,
                        oxy=med[:, 0],
                        deoxy=med[:, 1],
                        quadrant=label,
                        train_name=tab.quadrant.to_numpy()[w],
                    )
                )
    all_words = pd.concat(words, ignore_index=True)
    table = composition(all_words)
    if draw == 0:
        pe = pd.concat(obs, ignore_index=True)
        pe.to_csv(FOLDER / "subject_word_statistics.csv.gz", index=False)
        all_words.to_csv(FOLDER / "word_census.csv.gz", index=False)
        freq = frequency_match(all_words)
        freq.to_csv(FOLDER / "frequency_matched.csv", index=False)
        old = pd.read_csv(stage.prev.OUT / "transfer/hb_dictionary_replication.csv").sort_values(
            ["branch", "word"]
        )
        new = all_words.query("zone=='sign'").sort_values(["branch", "word"])
        for a, b in [
            ("named", None),
            ("majority", "replicate_majority"),
            ("unanimous", "replicate_all"),
            ("quadrant", "train_quadrant"),
        ]:
            if b:
                np.testing.assert_array_equal(new[a], old[b])
        bg = pd.DataFrame(WORK["background"]).query("split==0")
        bg["dataset_quadrant"] = bg[[f"q{i}" for i in range(4)]].to_numpy().argmax(1)
        v = (
            pe.query("role==1 and patches>=3")
            .merge(all_words.query("named")[["zone", "branch", "word"]], on=["zone", "branch", "word"])
            .merge(bg[["dataset", "zone", "dataset_quadrant"]], on=["dataset", "zone"])
        )
        v["named_agreement"] = v.train_name.eq(v.quadrant)
        v["background_agreement"] = v.dataset_quadrant.eq(v.quadrant)
        v.groupby(["branch", "dataset", "subject", "zone"]).agg(
            words=("word", "size"),
            named_agreement=("named_agreement", "mean"),
            background_agreement=("background_agreement", "mean"),
        ).to_csv(FOLDER / "background_agreement.csv")
    return table.assign(draw=draw)


def run():
    FOLDER.mkdir(exist_ok=True)
    n = collect()
    real = one(0)
    start = time.monotonic()
    pilot = one(1)
    secs = time.monotonic() - start
    # Shared immutable arrays; four processes each materialize one shuffled pair table at a time.
    with ProcessPoolExecutor(4, mp_context=mp.get_context("fork")) as pool:
        sims = list(pool.map(one, range(2, stage.CFG["semantics"]["permutation_repeats"] + 1)))
    table = pd.concat([real, pilot, *sims], ignore_index=True)
    summarize_draws(table)
    return dict(
        valid_emissions=n,
        draws=len(sims) + 1,
        single_draw_seconds=secs,
        workers=4,
        frequency_matched=pd.read_csv(FOLDER / "frequency_matched.csv").to_dict("records"),
    )


def summarize_draws(table):
    table = table.copy()
    table.loc[table.denominator.eq(0), "anti"] = np.nan
    table.to_csv(FOLDER / "composition_draws.csv.gz", index=False)
    table[table.draw.eq(0)].to_csv(FOLDER / "composition.csv", index=False)
    wide = table.pivot(
        index=["zone", "tier", "weight", "draw"], columns="branch", values="anti"
    ).reset_index()
    wide["S_minus_O"] = wide.S - wide.O
    out = []
    for key, g in wide.groupby(["zone", "tier", "weight"]):
        value = g[g.draw == 0].S_minus_O.iat[0]
        null = g[g.draw > 0].S_minus_O.dropna()
        out.append(
            dict(
                zip(["zone", "tier", "weight"], key),
                observed=value,
                null_mean=null.mean(),
                null_p025=null.quantile(0.025),
                null_p975=null.quantile(0.975),
                upper_tail=(1 + (null >= value).sum()) / (len(null) + 1) if len(null) else np.nan,
                valid_draws=len(null),
                planned_draws=len(g) - 1,
            )
        )
    pd.DataFrame(out).to_csv(FOLDER / "null_contrasts.csv", index=False)
