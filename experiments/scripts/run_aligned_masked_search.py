"""Execute the staged frozen-code search. Each worker owns one GPU and one fit at a time."""

import argparse
import hashlib
import itertools
from pathlib import Path
from functools import lru_cache
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import train_aligned_masked as parent
from run_dual_component_tokenizer import identity, now, owner_lock, read_json, write_json
from src.foundation.aligned_masked import FrozenCodeReader, compose, descriptors
from src.foundation.aligned_masked_search import (
    check_disjoint_windows,
    coverage_draws,
    log_priors,
    nominees,
    search_mask,
    set_prior_head,
    task_at,
    window_weights,
)
from src.utils.io import save_npz_atomic

CONTRACTS = {"audit": "aligned_masked_search_v2.yaml", "full": "aligned_masked_search_full_v3.yaml"}
CONTRACT = parent.CFG_PATH.with_name(CONTRACTS[os.environ.get("ALIGNED_SEARCH_PANEL", "audit")])
CFG = yaml.safe_load(CONTRACT.read_text())
ROOT = parent.audit.aligned.ROOT
OUT = ROOT / CFG["run"]
GRID = {}
for values in itertools.product(*CFG["grid"].values()):
    GRID["-".join(map(str, values))] = dict(zip(CFG["grid"], values))
STAGES = CFG["stages"]
TRAIN = CFG["training"]
MODS = ("eeg", "hb")


def full_window_panel(split, rows, audited):
    import analyze_aligned_token_pairing as pairing

    coords = pairing.native_coordinates()

    @lru_cache(maxsize=32)
    def mask_for(row_index):
        return np.load(Path(rows[row_index]["folder"]) / "mask.npy", mmap_mode="r")

    known = {(r.parent, r.start): r.Index for r in audited.itertuples()}
    ledger, positions = [], []
    for meta, pair in pairing.panel(split_filter=split, background_limit=None):
        status = "complete"
        sites = []
        for mod in MODS:
            row_index, allowed = pair[mod]
            row = rows[row_index]
            xy = coords(meta["dataset"], meta["subject"], meta["record"], mod, tuple(row["positions"]))
            selected = pairing.four_landmarks(xy, allowed)
            if len(selected) != 4:
                status = mod + "_geometry"
                break
            t, rate = meta["start"], row["rate"]
            if not mask_for(row_index)[t * rate : (t + 30) * rate, selected].all():
                status = mod + "_incomplete"
                break
            for k in selected:
                sites.append((meta["parent"], t, mod, row["positions"][k], *xy[k]))
        eeg = rows[pair["eeg"][0]]
        native = round(eeg["native_start_sample"] * 200 / eeg["native_rate_hz"]) + meta["start"] * 200
        ledger.append(
            dict(
                **meta,
                grid_start_sample=native,
                status=status,
                audit_index=known.get((meta["parent"], meta["start"]), -1),
            )
        )
        if status == "complete":
            positions.extend(sites)
    ledger = pd.DataFrame(ledger)
    panel = ledger.query("status=='complete'").reset_index(drop=True)
    check_disjoint_windows(panel)
    assert panel.groupby("dataset").size().to_dict() == CFG["full"]["by_dataset"][split]
    assert set(panel.loc[panel.audit_index >= 0, "audit_index"]) == set(
        audited[audited.split.eq(split)].index
    )
    sites = pd.DataFrame(positions, columns=["parent", "start", "mod", "position", "x", "y"]).groupby(
        ["parent", "start", "mod"], sort=False
    )
    return ledger, panel, sites


def prepare(split):
    assert split in CFG["scope"]["splits"] and len(GRID) == 24
    if split == "validation":
        assert any(read_json(OUT / "B_selection.json").values())
    folder = OUT / split
    with owner_lock(folder, "prepare"):
        if (folder / "ready.json").exists():
            assert read_json(folder / "ready.json")["signature"] == identity(CFG)
            return
        torch.set_float32_matmul_precision("highest")
        assert (
            ROOT / CFG["parent"] == parent.OUT
            and read_json(parent.OUT / "summary.json")["status"] == "completed"
        )
        ledger = pd.read_csv(parent.PARENT / "paired_panel.csv")
        full = ledger.query("status=='complete'").reset_index(drop=True)
        parent.audit.readout.panel()
        rows, scales = parent.audit.relations.maps()
        if CFG["full"].get("source") == "all_G0_nonoverlapping_score_windows":
            ledger, panel, sites = full_window_panel(split, rows, full)
        else:
            panel = full[full.split.eq(split)].reset_index(names="audit_index")
            sites = pd.read_csv(parent.PARENT / "positions.csv").groupby(
                ["parent", "start", "mod"], sort=False
            )
        strict = pd.read_csv(parent.OUT / "panel.csv")
        mapping = dict(zip(panel.audit_index, panel.index))
        panel["donor"] = -1
        for row in strict[strict.split.eq(split)].itertuples():
            donor = strict.iloc[row.donor]
            assert all(
                getattr(row, k) == donor[k] for k in ("dataset", "split", "subject", "record", "condition")
            )
            assert (
                row.event != donor.event
                and min(abs(row.eeg_start - donor.eeg_start), abs(row.hb_start - donor.hb_start)) >= 60
            )
            panel.loc[mapping[row.audit_index], "donor"] = mapping[donor.audit_index]
        panel["fold"] = -1
        for dataset, g in panel.groupby("dataset"):
            order = sorted(
                set(g.subject), key=lambda s: hashlib.sha256(f"1126|{dataset}|{s}".encode()).hexdigest()
            )
            panel.loc[g.index, "fold"] = (
                g.subject.map({s: i % 2 for i, s in enumerate(order)}) if split == "train" else 2
            )
        matched = panel[panel.donor >= 0]
        assert (
            set(matched.index) == set(matched.donor)
            and len(panel) == CFG["full"][split]
            and len(matched) == CFG["paired"][split]
        )
        assert not panel.duplicated(["dataset", "subject", "record", "eeg_start", "hb_start"]).any()
        if split == "train":
            for g, k in [(panel, "full"), (matched, "paired")]:
                if "fitting_folds" in CFG[k]:
                    assert g.groupby("fold").size().tolist() == CFG[k]["fitting_folds"]
        ids, xy, attributes, books, checks = parent.frozen_panel_arrays(panel, rows, scales, sites)
        previous = strict[strict.split.eq(split)]
        with np.load(parent.OUT / "tokens.npz") as z:
            overlap = [mapping[k] for k in previous.audit_index]
            np.testing.assert_array_equal(ids[overlap], z["ids"][previous.index])
            np.testing.assert_array_equal(xy[overlap], z["xy"][previous.index])
        save_npz_atomic(folder / "tokens.npz", ids=ids, xy=xy, attributes=attributes, **books)
        panel.to_csv(folder / "panel.csv", index=False)
        ledger[ledger.split.eq(split)].to_csv(folder / "support.csv", index=False)
        write_json(
            folder / "ready.json",
            dict(
                signature=identity(CFG),
                complete=len(panel),
                people=panel[["dataset", "subject"]].drop_duplicates().shape[0],
                by_fold=panel.groupby("fold").size().to_dict(),
                audited_overlap=int(panel.audit_index.ge(0).sum()),
                strict=len(matched),
                shape=ids.shape,
                checkpoints=checks,
                exact_parent_replay=len(overlap),
                finished=now(),
            ),
        )


def arrays(fold):
    panels = []
    features = []
    for split in ["train", "validation"] if fold == 2 else ["train"]:
        directory = OUT / split
        assert read_json(directory / "ready.json")["signature"] == identity(CFG)
        p = pd.read_csv(directory / "panel.csv")
        if panels:
            assert set(zip(p.dataset, p.subject)).isdisjoint(zip(panels[0].dataset, panels[0].subject))
            p.loc[p.donor >= 0, "donor"] += len(panels[0])
        panels.append(p)
        with np.load(directory / "tokens.npz") as z:
            features.append(tuple(z[k] for k in ("ids", "xy", "attributes")))
            books = [torch.from_numpy(z[f"book{k}"]) for k in range(4)]
    panel = pd.concat(panels, ignore_index=True)
    ids, xy, y = [np.concatenate(v) for v in zip(*features)]
    train = np.flatnonzero(panel.fold.ne(2) if fold == 2 else panel.fold.eq(fold))
    val = np.setdiff1d(panel.index, train)
    return (
        panel,
        torch.as_tensor(ids.astype("int64"), device="cuda"),
        torch.as_tensor(xy, device="cuda"),
        y,
        books,
        train,
        val,
    )


def reader(spec, books, prior, arm, seed):
    torch.manual_seed(seed)
    width = spec["width"]
    streams = {"eeg": (0, 1), "hb": (2, 3)}.get(arm, (0, 1, 2, 3))
    net = FrozenCodeReader(
        books,
        streams,
        dict(
            width=width,
            heads=width // 32,
            layers=width // 64,
            feedforward=width * 2,
            dropout=TRAIN["dropout"],
        ),
    ).cuda()
    if spec["head"] == "prior":
        set_prior_head(net, prior)
    return net


def fit_path(stage, candidate, fold, seed, arm):
    return OUT / stage / candidate / f"fold{fold}_seed{seed}_{arm}"


def fit(stage, candidate, fold, seed, arm, pilot=False):
    assert (
        fold in STAGES[stage]["folds"] and seed in STAGES[stage]["seeds"] and arm in STAGES[stage]["arms"]
    )
    spec = GRID[candidate]
    dest = fit_path("pilot" if pilot else stage, candidate, fold, seed, arm)
    signature = identity([CFG, stage, candidate, fold, seed, arm, pilot])
    with owner_lock(dest, "fit"):
        if (dest / "result.json").exists():
            assert read_json(dest / "result.json")["signature"] == signature
            return
        panel, ids, xy, y, books, tr, va = arrays(fold)
        prior = log_priors(ids, tr, CFG["tensor"]["vocab"])
        net = reader(spec, books, prior, arm, seed)
        optim = torch.optim.AdamW(
            (p for p in net.parameters() if p.requires_grad),
            lr=TRAIN["lr"],
            weight_decay=TRAIN["weight_decay"],
            fused=True,
        )
        total = 32 if pilot else STAGES[stage]["steps"]
        strict = tr[panel.donor.to_numpy()[tr] >= 0]
        rng = np.random.default_rng(seed)
        schedules = [
            rng.choice(pool, (total, TRAIN["batch"]), p=window_weights(panel, pool))
            for pool in (tr, strict)
        ]
        require_coverage = "sampling" in CFG["full"] and not pilot
        if require_coverage:
            local_slots = schedules[0][::2]
            local_slots[:] = coverage_draws(tr, local_slots.size, window_weights(panel, tr), rng).reshape(
                local_slots.shape
            )
        donors = panel.donor.to_numpy()
        ck = dest / "checkpoint.pt"
        step = 0
        exposure = np.zeros((2, len(panel)), int)
        hidden = np.zeros(4, int)
        if ck.exists():
            state = torch.load(ck, map_location="cpu", weights_only=False)
            assert state["signature"] == signature
            net.load_state_dict(state["model"])
            optim.load_state_dict(state["optimizer"])
            torch.cuda.set_rng_state(state["rng"])
            step = state["step"]
            exposure = state["exposure"]
            hidden = state["hidden"]
        tick = time.monotonic()
        start = step
        losses = []
        for i in range(step, total):
            is_pair, task = task_at(i, spec["mask"], spec["whole_percent"], arm)
            ix = schedules[int(is_pair)][i]
            batch, loc = compose(ids, xy, ix, donors[ix] if is_pair and arm == "wrong_pair" else None)
            mask = search_mask(len(ix), task, np.random.default_rng(seed * 100000 + i))
            for k in net.streams:
                hidden[k] += mask[:, k // 2].sum()
            np.add.at(exposure[int(is_pair)], ix, 1)
            optim.param_groups[0]["lr"] = TRAIN["lr"] * min(1, (i + 1) / TRAIN["warmup"])
            optim.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = net.objective(batch, loc, torch.as_tensor(mask, device="cuda"), isolated=not is_pair)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), TRAIN["clip"], error_if_nonfinite=True)
            optim.step()
            losses.append(loss.detach())
            if (i + 1) % 32 == 0:
                recent = torch.stack(losses).mean().item()
                losses = []
                if not np.isfinite(recent):
                    raise ValueError("nonfinite loss")
                write_json(
                    dest / "progress.json",
                    dict(
                        step=i + 1,
                        budget=total,
                        loss=recent,
                        steps_per_second=(i + 1 - start) / (time.monotonic() - tick),
                        updated=now(),
                    ),
                )
            if not pilot and ((i + 1) % TRAIN["checkpoint_every"] == 0 or i + 1 == total):
                parent.checkpoint(
                    ck,
                    dict(
                        signature=signature,
                        step=i + 1,
                        model=net.state_dict(),
                        optimizer=optim.state_dict(),
                        rng=torch.cuda.get_rng_state(),
                        exposure=exposure,
                        hidden=hidden,
                    ),
                )
        train_seconds = time.monotonic() - tick
        if require_coverage:
            assert (exposure[0, tr] > 0).all() and not exposure[:, va].any()
        exposure_frame = panel[["dataset", "subject", "record", "start"]].copy()
        exposure_frame["full_draws"], exposure_frame["strict_draws"] = exposure
        exposure_frame.to_csv(dest / "exposure.csv", index=False)
        tick = time.monotonic()
        scores = evaluate(net, spec, arm, seed, panel, ids, xy, y, books, prior, tr, va, dest)
        write_json(
            dest / "result.json",
            dict(
                status="completed",
                signature=signature,
                candidate=candidate,
                stage=stage,
                fold=fold,
                seed=seed,
                arm=arm,
                steps=total,
                hidden_counts=hidden.tolist(),
                train_seconds=train_seconds,
                scoring_seconds=time.monotonic() - tick,
                peak_gpu_mib=torch.cuda.max_memory_reserved() / 2**20,
                scores=scores,
                finished=now(),
            ),
        )


def macro(panel, indices, values):
    frame = panel.loc[indices, ["dataset", "subject", "record"]].assign(value=values)
    return float(
        frame.groupby(["dataset", "subject", "record"])
        .value.mean()
        .groupby(level=[0, 1])
        .mean()
        .groupby(level=0)
        .mean()
        .mean()
    )


@torch.inference_mode()
def vectors(net, ids, xy, fixed=False):
    outputs = {m: [] for m in set(k // 2 for k in net.streams)}
    for x, coords in zip(ids.split(16), xy.split(16)):
        mask = torch.zeros((len(x), 2, 4, 15), dtype=torch.bool, device=x.device)
        if fixed:
            z = torch.stack([net.books[j](x[:, k]) for j, k in enumerate(net.streams)], 1)
        else:
            z = net.encode(x, coords, mask, isolated=True)
        for m, parts in outputs.items():
            choose = [j for j, k in enumerate(net.streams) if k // 2 == m]
            values = z[:, choose]
            shape = values.shape[:-1]
            sites = coords[:, m, None, :, None].expand(*shape, 2)
            clock = torch.linspace(0, 1, 15, device=x.device).reshape(1, 1, 1, 15, 1).expand(*shape, 1)
            parts.append(descriptors(torch.cat([values, sites, clock], -1)))
    return {k: np.concatenate(v) for k, v in outputs.items()}


@torch.inference_mode()
def evaluate(net, spec, arm, seed, panel, ids, xy, y, books, prior, tr, va, dest):
    net.eval()
    for j, k in enumerate(net.streams):
        assert torch.equal(net.books[j].weight.cpu(), books[k])
    random = reader(spec, books, prior, arm, seed).eval()
    local = {}
    cross = {}
    measurements = panel.loc[va, ["dataset", "subject", "record", "event", "start"]].copy()
    for name, features in [
        ("fixed", vectors(random, ids, xy, True)),
        ("random", vectors(random, ids, xy)),
        ("trained", vectors(net, ids, xy)),
    ]:
        for m, x in features.items():
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(x[tr], y[tr, m])
            error = np.square(model.predict(x[va]) - y[va, m]) / y[tr, m].var()
            measurements[f"{MODS[m]}_{name}_nmse"] = error
            local.setdefault(MODS[m], {})[name] = macro(panel, va, error)
            if arm in ("eeg", "hb") and panel.fold.eq(2).any() and name == "trained":
                from sklearn.linear_model import LogisticRegression

                select = panel.condition.isin(["MA", "BL"]).to_numpy()
                fit_rows = tr[select[tr]]
                eval_rows = va[select[va] & panel.donor.ge(0).to_numpy()[va]]
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)).fit(
                    x[fit_rows], panel.condition.eq("MA").to_numpy()[fit_rows]
                )
                panel.loc[eval_rows, ["dataset", "subject", "record", "event", "condition"]].assign(
                    probability=clf.predict_proba(x[eval_rows])[:, 1]
                ).to_csv(dest / "ma_probabilities.csv", index=False)
    for v in local.values():
        v["gain"] = min(v["fixed"], v["random"]) - v["trained"]
    paired = va[panel.donor.to_numpy()[va] >= 0]
    donor = panel.donor.to_numpy()
    for m, mod in enumerate(MODS):
        if 2 * m not in net.streams:
            continue
        cross[mod] = {}
        for mode in ("prior", "true", "wrong", "reverse"):
            chunks = []
            for ix in np.array_split(paired, int(np.ceil(len(paired) / 8))):
                x = ids[ix].clone()
                loc = xy[ix].clone()
                source = 1 - m
                sl = slice(2 * source, 2 * source + 2)
                if mode == "wrong":
                    x[:, sl] = ids[donor[ix], sl]
                    loc[:, source] = xy[donor[ix], source]
                if mode == "reverse":
                    x[:, sl] = x[:, sl].flip([-1])
                mask = torch.zeros((len(ix), 2, 4, 15), dtype=torch.bool, device=x.device)
                mask[:, m] = True
                z = None if mode == "prior" else net.encode(x, loc, mask)
                streams = []
                for j, k in enumerate(net.streams):
                    if k // 2 != m:
                        continue
                    truth = ids[ix, k]
                    lp = (
                        prior[k][truth]
                        if mode == "prior"
                        else net.output[j](z[:, j])
                        .float()
                        .log_softmax(-1)
                        .gather(-1, truth[..., None])
                        .squeeze(-1)
                    )
                    streams.append((-lp.mean((1, 2)) / np.log(2)).cpu().numpy())
                chunks.append(np.stack(streams, 1))
            bits = np.concatenate(chunks)
            for j, s in enumerate("SO"):
                measurements.loc[paired, f"{mod}{s}_{mode}_bits"] = bits[:, j]
            cross[mod][mode] = macro(panel, paired, bits.mean(1))
    measurements.to_csv(dest / "held_windows.csv", index_label="panel_row")
    if dest.relative_to(OUT).parts[0] != "pilot":
        fixed_mask_scores(net, ids, xy, va, panel, dest)
    assert all(np.isfinite(list(v.values())).all() for v in [*cross.values(), *local.values()])
    return dict(local=local, cross=cross)


@torch.inference_mode()
def fixed_mask_scores(net, ids, xy, va, panel, dest):
    rows = []
    for shape in ("span", "scatter", "site"):
        for rep in range(4):
            masks = search_mask(len(va), shape, np.random.default_rng(1126 + rep))
            for start in range(0, len(va), 16):
                ix = va[start : start + 16]
                mask = torch.as_tensor(masks[start : start + len(ix)], device=ids.device)
                z = net.encode(ids[ix], xy[ix], mask, isolated=True)
                for j, k in enumerate(net.streams):
                    sel = mask[:, k // 2]
                    logits = net.output[j](z[:, j][sel]).float()
                    loss = torch.nn.functional.cross_entropy(
                        logits, ids[ix, k][sel], reduction="none"
                    ) / np.log(2)
                    counts = sel.sum((1, 2)).tolist()
                    bits = [float(v.mean()) for v in loss.split(counts)]
                    rows.extend(zip(ix, [shape] * len(ix), [rep] * len(ix), [k] * len(ix), bits))
    pd.DataFrame(rows, columns=["panel_row", "mask", "repeat", "stream", "bits"]).to_csv(
        dest / "common_masks.csv", index=False
    )


def jobs(stage, previous):
    cand = (
        sorted(GRID)
        if stage == "A"
        else sorted({r["candidate"] for values in previous.values() for r in values})
    )
    s = STAGES[stage]
    return list(itertools.product(cand, s["folds"], s["seeds"], s["arms"]))


def device_uuids():
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=pci.bus_id,uuid", "--format=csv,noheader"], text=True
    )
    devices = dict(line.split(", ") for line in rows.splitlines())
    return [devices["00000000:" + bus + ":00.0"] for bus in ("4B", "CA")]


def child(uuid, args, log):
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=uuid,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="1",
        PYTHONPATH=str(ROOT),
    )
    with log.open("a") as output:
        subprocess.run(
            [sys.executable, "-u", __file__, *map(str, args)],
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )


def controller():
    with owner_lock(OUT, "controller"):
        devices = device_uuids()
        (OUT / "resolved.yaml").write_text(CONTRACT.read_text())
        previous = {}
        for stage in STAGES:
            work = jobs(stage, previous)
            if not work:
                break
            if stage == "C":
                child(devices[0], ["prepare", "--split", "validation"], OUT / "validation_prepare.log")
            write_json(
                OUT / "status.json", dict(status="running", stage=stage, fits=len(work), updated=now())
            )

            # One persistent worker per GPU amortizes interpreter startup.
            def lane(i):
                child(devices[i], ["lane", "--stage", stage, "--lane", i], OUT / f"{stage}_lane{i}.log")

            with ThreadPoolExecutor(max_workers=len(devices)) as pool:
                list(pool.map(lane, range(len(devices))))
            scores = {}
            results = []
            for c, f, s, a in work:
                r = read_json(fit_path(stage, c, f, s, a) / "result.json")
                results.append(r)
                assert r["status"] == "completed" and r["steps"] == STAGES[stage]["steps"]
                scores[c, f, a] = r["scores"]
            pd.json_normalize(results).to_csv(OUT / f"{stage}_results.csv", index=False)
            if stage != "C":
                previous = nominees(scores, {c: GRID[c] for c, _, _, _ in work}, stage, previous)
                write_json(OUT / f"{stage}_selection.json", previous)
            else:
                fuse_late(work)
        write_json(
            OUT / "training_summary.json",
            dict(
                status="completed",
                last_trained_stage=stage if work else chr(ord(stage) - 1),
                locks=previous,
                finished=now(),
                fits=len(list(OUT.glob("[ABC]/*/*/result.json"))),
            ),
        )
        write_json(OUT / "status.json", read_json(OUT / "training_summary.json"))


def fuse_late(work):
    for c, _, seed, arm in work:
        if arm != "eeg":
            continue
        left, right = [pd.read_csv(fit_path("C", c, 2, seed, k) / "ma_probabilities.csv") for k in MODS]
        merged = left.merge(
            right, on=["dataset", "subject", "record", "event", "condition"], validate="one_to_one"
        )
        assert len(merged) == len(left) == len(right)
        merged["probability"] = (merged.probability_x + merged.probability_y) / 2
        merged.to_csv(OUT / "C" / c / f"seed{seed}_late_fusion.csv", index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "pilot", "lane", "controller"])
    p.add_argument("--split", choices=["train", "validation"], default="train")
    p.add_argument("--stage", choices=STAGES, default="A")
    p.add_argument("--lane", type=int, choices=[0, 1], default=0)
    args = p.parse_args()
    torch.set_num_threads(2)
    if args.command == "controller":
        controller()
        return
    assert os.environ["CUDA_VISIBLE_DEVICES"] in device_uuids() and torch.cuda.device_count() == 1
    torch.set_float32_matmul_precision("high")
    if args.command == "prepare":
        prepare(args.split)
    elif args.command == "pilot":
        c = f"span-25-{(128,256)[args.lane]}-prior"
        fit("A", c, 0, 17, "true_pair", True)
    else:
        prev = {} if args.stage == "A" else read_json(OUT / f"{chr(ord(args.stage)-1)}_selection.json")
        for c, f, s, a in jobs(args.stage, prev)[args.lane :: 2]:
            print(now(), args.stage, c, f, s, a, flush=True)
            fit(args.stage, c, f, s, a)


if __name__ == "__main__":
    main()
