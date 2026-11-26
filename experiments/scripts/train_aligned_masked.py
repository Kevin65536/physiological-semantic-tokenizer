"""Train and score the M0–M4 frozen-128 masked-ID experiment."""

import argparse
from collections import OrderedDict
import time

import numpy as np
import pandas as pd
import torch
import yaml

import audit_aligned_tokenizer as audit
from run_dual_component_tokenizer import read_json, write_json, identity, owner_lock, now
from src.data.dual_component_tokenizer import base_tokenizer_batch
from src.foundation.aligned_masked import compose, condition_permutation, hidden_sites, FrozenCodeReader
from src.analysis.single_vq_relations import wave_attribute
from src.utils.io import save_npz_atomic

CFG_PATH = audit.CONTRACT.with_name("aligned_masked_pretraining_v1.yaml")
CFG = yaml.safe_load(CFG_PATH.read_text())
OUT = audit.aligned.ROOT / CFG["run"]
PARENT = audit.OUT
STREAMS = {"M0_EEG": (0, 1), "M0_Hb": (2, 3), "M2": (0, 1, 2, 3), "M3": (0, 1, 2, 3), "M4": (0, 1, 2, 3)}
TR = CFG["training"]


def tensors():
    with np.load(OUT / "tokens.npz") as z:
        return (
            torch.as_tensor(z["ids"].astype("int64"), device="cuda"),
            torch.as_tensor(z["xy"], device="cuda"),
            [torch.as_tensor(z[f"book{i}"]) for i in range(4)],
        )


@torch.inference_mode()
def prepare():
    torch.set_float32_matmul_precision("highest")
    assert PARENT == audit.aligned.ROOT / CFG["audit"]
    upstream = read_json(PARENT / "summary.json")
    assert upstream["status"] == "completed" and (upstream["seed"], upstream["dimension"]) == (17, 128)
    assert (
        CFG["scope"]["splits"] == ["train", "validation"]
        and CFG["scope"]["test"] == CFG["scope"]["confirmation"] == "forbidden"
    )
    audit.readout.panel()
    rows, scales = audit.relations.maps()
    full = pd.read_csv(PARENT / "paired_panel.csv").query("status=='complete'").reset_index(drop=True)
    cand = full[full.condition.notna() & full.event.ge(0)].copy().reset_index(names="audit_index")
    site = pd.read_csv(PARENT / "positions.csv").groupby(["parent", "start", "mod"], sort=False)
    assert (
        set(cand.split) == {"train", "validation"}
        and not cand.duplicated(["dataset", "subject", "record", "event"]).any()
    )
    donor = condition_permutation(
        cand, CFG["panel"]["min_donor_separation_seconds"], CFG["panel"]["match_seed"]
    )
    keep = np.flatnonzero(donor >= 0)
    panel = cand.iloc[keep].reset_index(drop=True)
    panel["donor"] = np.searchsorted(keep, donor[keep])
    panel["donor_event"] = panel.event.to_numpy()[panel.donor]
    cand.assign(included=donor >= 0).to_csv(OUT / "eligibility.csv", index=False)
    pp = CFG["panel"]
    counts = panel.groupby("split", sort=True).size().tolist()
    people = (
        panel.groupby("split", sort=True)
        .apply(lambda g: len(g[["dataset", "subject"]].drop_duplicates()), include_groups=False)
        .tolist()
    )
    assert counts == [pp["expected_train"], pp["expected_validation"]] and people == pp["expected_people"]
    sets = [set(map(tuple, g[["dataset", "subject"]].to_numpy())) for _, g in panel.groupby("split")]
    assert not sets[0] & sets[1]
    ids, xy, target, books, checks = frozen_panel_arrays(panel, rows, scales, site)
    save_npz_atomic(OUT / "tokens.npz", ids=ids, xy=xy, attributes=target, **books)
    panel.to_csv(OUT / "panel.csv", index=False)
    write_json(
        OUT / "prepared.json",
        dict(
            status="completed",
            counts=counts,
            people=people,
            eligible_before_matching=len(cand),
            shape=ids.shape,
            checkpoints=checks,
            family_replay="exact for all individual positions and all 10-second bins",
            signature=identity(CFG),
            created=now(),
        ),
    )


@torch.inference_mode()
def frozen_panel_arrays(panel, rows, scales, site):
    """Replay the audited four-position tokenizer pass; audit_index names the original complete window."""
    lookup = {(r["parent"], r["modality"]): i for i, r in enumerate(rows)}
    ids = np.zeros((len(panel), 4, 4, 15), "uint16")
    xy = np.empty((len(panel), 2, 4, 2), "float32")
    books = {}
    checks = {}
    cache = OrderedDict()
    with np.load(PARENT / "paired_features.npz") as f:
        # Targets used only in downstream readouts.
        retained = panel.audit_index.ge(0).to_numpy()
        reference = panel.loc[retained, "audit_index"].to_numpy(dtype=int)
        target = np.empty((len(panel), 2), dtype=np.float64)
        for j, mod in enumerate(("eeg", "hb")):
            net = audit.network(mod, CFG["vocab"][2 * j], 17, "cuda:0")
            maps = [pd.read_csv(PARENT / f"{mod}_17/{b}_words.csv").family.to_numpy() for b in "SO"]
            for k, b in enumerate("SO"):
                books["book" + str(2 * j + k)] = (
                    net.branches[b]["quantizer"].codebook.detach().cpu().numpy()
                )
            batch = []
            for i, r in panel.iterrows():
                s = site.get_group((r.parent, r.start, mod))
                assert len(s) == s.position.nunique() == 4
                idx = lookup[r.parent, mod]
                owner = rows[idx]
                assert (owner["dataset"], owner["subject"], owner["record"], owner["split"]) == (
                    r.dataset,
                    r.subject,
                    r.record,
                    r.split,
                )
                xy[i, j] = s[["x", "y"]]
                panel.loc[i, mod + "_sites"] = "|".join(s.position)
                batch.extend((idx, r.start, owner["positions"].index(p)) for p in s.position)
            for off in range(0, len(batch), 4):
                raw = base_tokenizer_batch(rows, batch[off : off + 4], scales, array_cache=cache)
                assert raw["mask"].all()
                target[off // 4, j] = wave_attribute(raw["x"], mod).mean()
                v = net(torch.as_tensor(raw["x"], device="cuda"))["ids"]
                for k, b in enumerate("SO"):
                    arr = v[b].cpu().numpy()
                    ids[off // 4 : (off + len(arr)) // 4, 2 * j + k] = arr.reshape(-1, 4, 15)
            for k, b in enumerate("SO"):
                v = ids[:, 2 * j + k]
                assert v.max() < CFG["vocab"][2 * j + k]
                q = np.eye(4)[maps[k][v]].reshape(len(panel), 4, 3, 5, 4).mean(3)
                np.testing.assert_array_equal(q[retained], f[mod + b][reference])
            np.testing.assert_allclose(
                target[retained, j], f[mod + "_y"][reference].mean((1, 2)), rtol=1e-7, atol=1e-12
            )
            checks[mod] = read_json(audit.PARENT / mod / "result.json")["checkpoint"]
    assert np.isfinite(xy).all() and np.isfinite(target).all()
    return ids, xy, target, books, checks


def reader(books, arm):
    torch.manual_seed(TR["seed"])
    torch.cuda.manual_seed_all(TR["seed"])
    return FrozenCodeReader(books, STREAMS[arm], CFG["model"]).cuda()


def checkpoint(path, payload):
    tmp = path.with_suffix(".pending")
    torch.save(payload, tmp)
    tmp.replace(path)


def train(arm, pilot):
    folder = OUT / arm
    folder.mkdir(exist_ok=True)
    assert read_json(OUT / "prepared.json")["signature"] == identity(CFG)
    ids, xy, books = tensors()
    net = reader(books, arm)
    optim = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad],
        lr=TR["lr"],
        weight_decay=TR["weight_decay"],
        fused=True,
    )
    panel = pd.read_csv(OUT / "panel.csv")
    train_idx = np.flatnonzero(panel.split.eq("train"))
    don = panel.donor.to_numpy()
    rng = np.random.default_rng(TR["seed"])
    schedule = np.concatenate(
        [rng.permutation(train_idx) for _ in range(1 + TR["steps"] * TR["batch"] // len(train_idx))]
    )[: TR["steps"] * TR["batch"]].reshape(-1, TR["batch"])
    total = 16 if pilot else TR["steps"]
    start = 0
    dest = folder / "checkpoint.pt"
    if not pilot and dest.exists():
        state = torch.load(dest, map_location="cpu", weights_only=False)
        assert state["signature"] == identity([CFG, arm])
        start = state["step"]
        net.load_state_dict(state["model"])
        optim.load_state_dict(state["optimizer"])
        torch.cuda.set_rng_state(state["rng"])
    tick = time.monotonic()
    recent = []
    elapsed = 0
    for i in range(start, total):
        if i == 4:
            torch.cuda.synchronize()
            elapsed = time.monotonic()
        train_rng = np.random.default_rng(31000 + i)
        task = CFG["mask"]["mixed_cycle"][i % 4] if arm in ("M3", "M4") else "span"
        ix = schedule[i]
        batch, loc = compose(ids, xy, ix, don[ix] if arm == "M4" else None)
        mask = torch.as_tensor(hidden_sites(len(ix), 4, 15, task, train_rng), device="cuda")
        optim.param_groups[0]["lr"] = TR["lr"] * min(1, (i + 1) / TR["warmup"])
        optim.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = net.objective(batch, loc, mask)
        if not torch.isfinite(loss):
            raise FloatingPointError((arm, i))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), TR["clip"], error_if_nonfinite=True)
        optim.step()
        recent.append(float(loss.detach()))
        if not pilot and ((i + 1) % 16 == 0 or i == 0):
            write_json(
                folder / "progress.json",
                dict(
                    step=i + 1,
                    budget=total,
                    loss_last16=np.mean(recent[-16:]),
                    updated=now(),
                    steps_per_second=(i - start + 1) / (time.monotonic() - tick),
                ),
            )
        if not pilot and ((i + 1) % TR["checkpoint_every"] == 0 or i + 1 == total):
            checkpoint(
                dest,
                dict(
                    step=i + 1,
                    signature=identity([CFG, arm]),
                    model=net.state_dict(),
                    optimizer=optim.state_dict(),
                    rng=torch.cuda.get_rng_state(),
                ),
            )
            print(arm, i + 1, total, np.mean(recent[-16:]), flush=True)
    if pilot:
        torch.cuda.synchronize()
        write_json(
            folder / "pilot.json",
            dict(
                steps=total,
                timed_steps=total - 4,
                steps_per_second=(total - 4) / (time.monotonic() - elapsed),
                peak_mib=torch.cuda.max_memory_reserved() / 2**20,
                gpu=torch.cuda.get_device_name(),
                finite=True,
            ),
        )
        return
    write_json(
        folder / "result.json",
        dict(
            status="completed",
            arm=arm,
            step=total,
            signature=identity([CFG, arm]),
            checkpoint=str(dest.relative_to(OUT)),
            updated=now(),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "pilot", "train"])
    parser.add_argument("--arm", choices=STREAMS, default="M3")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    with owner_lock(OUT, ("prepare" if args.stage == "prepare" else args.arm)):
        if args.stage == "prepare":
            if (OUT / "prepared.json").exists():
                assert read_json(OUT / "prepared.json")["signature"] == identity(CFG)
            else:
                prepare()
        else:
            train(args.arm, args.stage == "pilot")


if __name__ == "__main__":
    main()
