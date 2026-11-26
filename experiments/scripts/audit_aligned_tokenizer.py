"""Frozen aligned single-VQ audit on the registered public G0 and event panels."""

import argparse
from collections import OrderedDict, defaultdict
from functools import lru_cache
import sys
import time

import numpy as np
import torch
import yaml

import retrain_single_vq_aligned as aligned
import run_single_vq_readout as readout
import run_single_vq_relations as relations
from run_dual_component_tokenizer import _base_training_model, identity, owner_lock, read_json, write_json
from src.data.dual_component_tokenizer import base_tokenizer_batch
from src.utils.io import save_npz_atomic

CONTRACT = aligned.CONTRACT.with_name("aligned_token_audit_v1.yaml")
PLAN = yaml.safe_load(CONTRACT.read_text())
OUT = aligned.ROOT / PLAN["run"]
PARENT = aligned.ROOT / PLAN["parent"]


@lru_cache(maxsize=4)
def network(mod, size, seed, device="cpu"):
    cfg, task = aligned.settings(mod)
    ref = read_json(PARENT / mod / "result.json")
    path = aligned.ROOT / ref["checkpoint"]
    state = torch.load(path, map_location="cpu", weights_only=False)
    sig = state["signature"]
    if (
        size != PLAN["models"][mod]["K"]
        or seed != cfg["training"]["seed"]
        or state["step"] != 10000
        or sig != read_json(path.parent / "summary.json")["signature"]
        or sig["task"] != task
        or sig["config"] != identity(cfg)
        or ref["status"] != "completed"
        or (mod == "eeg" and sig["spectral_objective"] != identity(aligned.objective()))
    ):
        raise ValueError("frozen tokenizer identity drift")
    model = _base_training_model(cfg, task, single_id_spectrum=mod == "eeg")
    model.load_state_dict(state["model_state"])
    if any(tuple(model.branches[b]["quantizer"].codebook.shape) != (size, 128) for b in "SO"):
        raise ValueError("wrong frozen codebook")
    return model.eval().to(device)


def tasks(role=None):
    return [t for t in readout.shards() if role is None or t[2] == role]


def inputs(task):
    return OUT / "inputs" / readout.destination(task).name


@torch.inference_mode()
def extract(task, pilot=False):
    mod, ds, role, lo, hi = task
    if inputs(task).exists() and not pilot:
        return
    model = network(mod, PLAN["models"][mod]["K"], 17, "cuda:0")
    rows, scales = relations.maps()
    ex = readout.capacity.PANEL["examples"][mod, role, ds][lo:hi]
    with np.load(relations.path(task)) as prev:
        np.testing.assert_array_equal(ex, prev["examples"])
        data = {k: prev[k] for k in ("examples", "valid", "y", "bg")}
    kept = defaultdict(list)
    cache = OrderedDict()
    start = time.monotonic()
    for i in range(0, min(64, len(ex)) if pilot else len(ex), 64):
        take = ex[i : i + 64]
        batch = base_tokenizer_batch(rows, take, scales, array_cache=cache)
        valid = batch["mask"].all(-1)
        np.testing.assert_array_equal(valid, data["valid"][i : i + len(take)])
        out = model(torch.from_numpy(batch["x"]).to("cuda:0"))
        replay = model.decode_ids(out["ids"])
        for b in "SO":
            torch.testing.assert_close(replay[b], out[b], atol=0, rtol=0)
            ids = out["ids"][b].cpu().numpy()[valid]
            if ids.max() >= PLAN["models"][mod]["K"] or ids.min() < 0:
                raise ValueError("invalid code")
            key = f"{b}_17_"
            kept[key + "id"].append(ids.astype(np.uint16))
            kept[key + "z"].append(out["latent"][b].cpu().numpy()[valid])
            if role == "validation":
                kept[key + "wave"].append(
                    readout.attributes(replay[b].cpu().numpy(), mod)[0][valid.ravel()]
                )
    if pilot:
        return dict(
            task=task,
            examples=min(64, len(ex)),
            seconds=time.monotonic() - start,
            peak_bytes=torch.cuda.max_memory_reserved(),
            exact_replay=True,
        )
    data.update({k: np.concatenate(v) for k, v in kept.items()})
    if mod == "eeg":
        saved = aligned.ROOT / read_json(PARENT / mod / "result.json")["checkpoint"]
        with np.load(saved.parent / "evaluation" / f"{ds}_{role}.npz") as z:
            np.testing.assert_array_equal(ex, z["examples"][lo:hi])
            mask = z["valid"]
            np.testing.assert_array_equal(data["valid"], mask[lo:hi])
            np.testing.assert_array_equal(data["S_17_id"], z["id"][mask[:lo].sum() : mask[:hi].sum()])
    save_npz_atomic(inputs(task), **data)


def collect(mod, seed, role):
    parts = defaultdict(list)
    assert seed == 17
    for t in tasks(role):
        if t[0] != mod:
            continue
        with np.load(inputs(t)) as z:
            for k in ["examples", "valid", "bg", "y"]:
                parts[k].append(z[k])
            parts["ids"].append(np.column_stack([z[b + "_17_id"] for b in "SO"]).astype(int))
    a = {k: np.concatenate(v) for k, v in parts.items()}
    rows, _ = relations.maps()
    a["persons"] = np.array(
        [readout.PEOPLE[rows[r]["dataset"], rows[r]["subject"]] for r in a["examples"][:, 0]]
    )
    a["person"] = np.repeat(a["persons"], a["valid"].sum(1))
    return a


def run(stage, mod):
    source = sys.modules[__name__]
    if stage == "extract":
        jobs = [t for t in tasks() if t[0] == mod]
        write_json(OUT / f"{mod}_pilot.json", extract(jobs[0], True))
        for i, t in enumerate(jobs):
            extract(t)
            print(i + 1, len(jobs), t, flush=True)
    elif stage == "members":
        import single_vq_member_sequence as member

        member.analyze((mod, 17), source=source)
        member.supplementary((mod, 17), source=source)
    elif stage == "events":
        from single_vq_event_coupling import evaluate_events

        evaluate_events(6, out=OUT / "events_17", network=network, seed=17, families=OUT)
    elif stage == "budget":
        from analyze_aligned_token_fusion_controls import matched_main, fusion_null

        matched_main(mod)
        if mod == "eeg":
            fusion_null()
    elif stage == "visibility":
        from analyze_aligned_token_visibility import run as visibility

        visibility(mod)
    elif stage == "specificity":
        from analyze_aligned_token_specificity import run as specificity

        specificity(mod)
    elif stage == "spatial":
        from analyze_aligned_token_spatial import run as spatial

        spatial(mod)
    elif stage in ("waveform", "eye", "event_controls"):
        from analyze_aligned_token_controls import run as controls

        controls(mod, stage)
    elif stage in ("S", "O"):
        from single_vq_final_c_probes import readout_job

        readout_job((17, stage), source=source, mod=mod, include_latent=True)
    else:
        from analyze_aligned_token_diagnostics import analyze

        analyze(stage, mod)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        choices=[
            "extract",
            "members",
            "events",
            "S",
            "O",
            "local",
            "cross",
            "extended",
            "waveform",
            "eye",
            "event_controls",
            "spatial",
            "specificity",
            "visibility",
            "budget",
        ],
    )
    p.add_argument("mod", choices=["eeg", "hb"])
    a = p.parse_args()
    torch.set_num_threads(1)
    with owner_lock(OUT, a.mod + "_" + a.stage):
        marker = OUT / f"{a.mod}_{a.stage}_status.json"
        if marker.exists() and read_json(marker)["status"] == "completed":
            return
        write_json(marker, dict(status="running"))
        try:
            readout.panel()
            run(a.stage, a.mod)
            write_json(marker, dict(status="completed"))
        except Exception as exc:
            write_json(marker, dict(status="failed", error=repr(exc)))
            raise


if __name__ == "__main__":
    sys.modules["audit_aligned_tokenizer"] = sys.modules[__name__]
    main()
