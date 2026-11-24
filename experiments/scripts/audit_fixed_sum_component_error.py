"""Separate original component NMSE into a fixed-sum floor and allocation residual for saved integer outputs."""

import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.dual_component_tokenizer import base_training_aggregate
from src.data.dual_component_tokenizer import (
    base_tokenizer_batch,
    base_tokenizer_training_data,
)

STATE = {}


def decompose(s_error, o_error, mask, vs, vo):
    """Minimum joint weighted squared error achievable by moving signal between fixed-sum outputs."""
    if (
        vs.shape != vo.shape
        or np.any(vs <= 0)
        or np.any(vo <= 0)
        or np.any(mask.sum(1) == 0)
    ):
        raise ValueError(
            "positive matching component variances and scored time support required"
        )

    def mse(e):
        return np.where(mask, e**2, 0.0).sum(1) / mask.sum(1)

    vs, vo = np.asarray(vs, float), np.asarray(vo, float)
    e_sum = s_error + o_error
    residual = s_error - e_sum * (vs / (vs + vo))[:, None, :]
    s = mse(s_error) / vs
    o = mse(o_error) / vo
    floor = mse(e_sum) / (vs + vo)
    allocation = mse(residual) * (1 / vs + 1 / vo)
    np.testing.assert_allclose(s + o, floor + allocation, rtol=1e-12, atol=1e-12)
    return np.stack([s, o, s + o, floor, allocation], axis=1)


def measure(wave, batch, channels):
    shape = (len(batch["x"]), -1, channels)
    mask = batch["mask"].reshape(shape)
    target = {b: np.asarray(batch[b], float).reshape(shape) for b in ["x", "s", "o"]}
    pred = {b: np.asarray(wave[b], float).reshape(shape) for b in ["sum", "S", "O"]}
    va = {
        b: np.asarray(batch["v" + b], float)[:, 0, :channels] for b in ["x", "s", "o"]
    }
    parts = decompose(
        pred["S"] - target["s"], pred["O"] - target["o"], mask, va["s"], va["o"]
    )
    xmse = np.where(mask, (pred["sum"] - target["x"]) ** 2, 0).sum(1) / (
        mask.sum(1) * va["x"]
    )
    difference = max(
        float(np.max(np.abs(np.where(mask, pred["sum"] - pred["S"] - pred["O"], 0)))),
        0.0,
    )
    target_gap = float(
        np.max(np.abs(np.where(mask, target["x"] - target["s"] - target["o"], 0)))
    )
    return np.concatenate([parts, xmse[:, None, :]], axis=1), difference, target_gap


def worker(group):
    from experiments.scripts.apply_joint_tokenizer import checked_path, load_branch_pair

    cfg, run, out, rows, scales = tuple(
        STATE[k] for k in ["config", "run", "out", "rows", "scales"]
    )
    mode = "hb" if group == "hb" else "eeg"
    bundle = run / cfg["bundle"]
    spec = json.loads(bundle.read_text())[mode]
    model = load_branch_pair(spec["kind"], checked_path(bundle, spec))
    expected = STATE[mode]
    ref = STATE[mode + "_metrics"]
    if mode == "eeg":
        ix = np.flatnonzero(
            np.asarray([rows[i]["dataset"] == group for i, _, _ in expected])
        )
    else:
        ix = np.arange(len(expected))
    lookup = {tuple(expected[j]): j for j in ix}
    cache = OrderedDict()
    packed = []
    indices = []
    max_diffs = np.zeros(3)

    def score(ex, wave):
        if any(tuple(row) not in lookup for row in ex):
            raise ValueError("scoring outside frozen development support")
        selected = np.array([lookup[tuple(e)] for e in ex])
        batch = base_tokenizer_batch(
            rows, ex, scales, array_cache=cache, cache_limit=32
        )
        values, closure, target_gap = measure(wave, batch, 2 if mode == "hb" else 1)
        repeated = values[:, [5, 0, 1], :]
        np.testing.assert_allclose(repeated, ref[selected], rtol=3e-12, atol=2e-12)
        max_diffs[:] = np.maximum(
            max_diffs, [np.abs(repeated - ref[selected]).max(), closure, target_gap]
        )
        packed.append(values)
        indices.extend(selected.tolist())

    if mode == "hb":
        with np.load(run / cfg["hb"] / "integer_ids.npz") as stored:
            np.testing.assert_array_equal(expected, stored["examples"])
            ids = {b: stored[b] for b in "SO"}
        for i in range(0, len(expected), 128):
            sl = slice(i, i + 128)
            score(expected[sl], model.decode_ids({b: v[sl] for b, v in ids.items()}))
    else:
        names = model.metadata["positions"][group]
        native = {s.upper(): j for j, s in enumerate(names)}
        for path in sorted((run / cfg["eeg"] / group).glob("forward_*.npz")):
            with np.load(path) as block:
                signature = json.loads(str(block["signature"]))
                if (
                    signature["checkpoint_sha256"] != spec["sha256"]
                    or signature["panel_sha256"] != cfg["panel_identity"]
                ):
                    raise ValueError("wrong frozen source or panel")
                ex = block["examples"]
                if any(tuple(e) not in lookup for e in ex):
                    raise ValueError("unregistered crop")
                ids = {b: block["ids_" + b] for b in "SO"}
                for j, (row, start) in enumerate(block["crop_row_start"]):
                    take = ex[(ex[:, 0] == row) & (ex[:, 1] == start)]
                    sites = [
                        native[rows[i]["positions"][p].upper()] for i, t, p in take
                    ]
                    result = model.decode_ids(
                        {b: v[j : j + 1] for b, v in ids.items()}, group
                    )
                    score(take, {b: v[0, sites] for b, v in result.items()})
    indices = np.array(indices)
    if len(indices) != len(ix) or set(indices) != set(ix):
        raise ValueError("incomplete matched panel")
    np.savez_compressed(
        out / (group + ".npz"),
        indices=indices,
        examples=expected[indices],
        parts=np.concatenate(packed),
    )
    return group, max_diffs.tolist()


def main():
    from evaluate_joint_tokenizer import write

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    run = ROOT / cfg["run"]
    merged = base_tokenizer_training_data(
        yaml.safe_load((ROOT / cfg["g0"]).read_text()), ROOT
    )
    rows = merged[0]["rows"]
    scales = merged[1]
    with (run / cfg["panel"]).open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != cfg["panel_identity"]:
            raise ValueError("wrong panel")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        raise FileExistsError("retain previous audit")
    STATE.update(config=cfg, run=run, out=args.output, rows=rows, scales=scales)
    with np.load(run / cfg["panel"]) as panels:
        for b in ("hb", "eeg"):
            ex = panels[b + "_development"]
            if any(
                rows[i]["modality"] != b or rows[i]["split"] != "validation"
                for i in np.unique(ex[:, 0])
            ):
                raise ValueError("development identities required")
            with np.load(run / cfg[b] / "metrics.npz") as evidence:
                np.testing.assert_array_equal(evidence["examples"], ex)
                STATE[b + "_metrics"] = evidence["nmse"]
            STATE[b] = ex
    groups = sorted({rows[i]["dataset"] for i in STATE["eeg"][:, 0]})
    torch.set_num_threads(1)
    with ProcessPoolExecutor(
        max_workers=cfg["resources"]["workers"],
        mp_context=multiprocessing.get_context("fork"),
    ) as pool:
        checks = dict(pool.map(worker, [*groups, "hb"]))
    arms = {}
    for b, files in [("eeg", groups), ("hb", ["hb"])]:
        values = []
        ex = []
        for name in files:
            with np.load(args.output / (name + ".npz")) as f:
                values.append(f["parts"])
                ex.extend(f["examples"])
        stats = base_training_aggregate({"nmse": np.concatenate(values)}, ex, rows)[
            "metrics"
        ]["nmse"]
        macro = np.array(stats["macro"])
        arms[b] = dict(
            position_windows=len(ex),
            subjects=len(stats["by_subject"]),
            metrics=stats,
            fraction_from_sum_error=(macro[3] / macro[2]).tolist(),
            fraction_from_allocation=(macro[4] / macro[2]).tolist(),
        )
    write(
        args.output / "summary.json",
        dict(
            status="complete",
            arms=arms,
            checks=checks,
            check_order=[
                "maximum_original_NMSE_difference",
                "maximum_output_sum_roundoff",
                "maximum_raw_target_closure_in_train_units",
            ],
            config=cfg,
        ),
    )
    print({b: r["fraction_from_allocation"] for b, r in arms.items()})


if __name__ == "__main__":
    main()
