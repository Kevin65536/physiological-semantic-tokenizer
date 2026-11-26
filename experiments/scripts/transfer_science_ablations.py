"""Apply identically trained n-back readouts to the previously admitted WG panel."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time
import numpy as np
import pandas as pd
import torch
import yaml

import analyze_four_stream_transfer as transfer
import run_frozen_science_screen as screen
from prepare_science_ablations import OUT, CONFIG, check
from run_science_ablations import ARMS, TR, read, save, create, extract
from run_dual_component_tokenizer import owner_lock
from src.analysis.frozen_science_screen import anchor, slot_vectors
from src.inference.conditional_slot_training import fit, forward
from src.utils.io import save_npz_atomic

DEST = OUT / "wg"
PROTOCOL = CONFIG.with_name("science_ablation_transfer_v1.yaml")


def prepare():
    m, raw = transfer.wg_panel(output=DEST)
    retained = pd.read_csv(screen.BASE / "transfer/wg_panel.csv")
    pd.testing.assert_frame_equal(
        m[["id", "subject", "condition", "donor"]], retained[["id", "subject", "condition", "donor"]]
    )
    if len(m) != 237 or m.subject.nunique() != 5:
        raise ValueError("WG support")
    _, n, d = screen.parent.load_parent("cached")
    np.testing.assert_array_equal(n.id, screen.panel()[0].id)
    tr = np.flatnonzero(n.split == "train")
    ev = np.arange(len(n), len(n) + len(m))
    y = np.concatenate([d["hb_target"], raw["hb_target"]])
    h = np.concatenate([d["hb_history"], raw["hb_history"]])
    xy = np.concatenate([d["xy"], raw["xy"]])
    if h.shape != (637, 4, 300, 2) or not np.isfinite(y).all():
        raise ValueError("matched coordinates")
    people = np.r_[n.subject.to_numpy(), m.subject.to_numpy()]
    oof, base, alpha = anchor(h, y, people, tr, ev, screen.GRID)
    sc = np.array([r["scale"] for r in read(OUT / "scales.json")])
    x = (raw["eeg_wave"] / sc[None, :, None, None]).astype("float32")
    save_npz_atomic(DEST / "panel.npz", y=y, h=h, xy=xy, x=x, oof=oof, base=base)
    return dict(
        fit_events=len(tr),
        test_events=len(m),
        test_people=len(set(m.subject)),
        ridge=alpha,
        public_WG_new_to_nback=sorted(set(m.subject) - set(n.query("split=='validation'").subject)),
    )


def one(task):
    arm, seed = task
    f = OUT / "models" / f"{arm}_{seed}"
    out = DEST / f"{arm}_{seed}"
    with owner_lock(out):
        if (out / "done.json").exists():
            return
        state = torch.load(f / "state.pt", map_location="cpu", weights_only=False)
        if state["step"] != 1280 or state["arm"] != arm or state["seed"] != seed:
            raise ValueError("checkpoint identity")
        model = create(arm, seed, "cpu").eval()
        model.load_state_dict(state["model"])
        del state
        n, tr, _ = screen.panel()
        m = pd.read_csv(DEST / "wg_panel.csv")
        with np.load(DEST / "panel.npz") as z:
            h, y, xy, x, oof, base = [z[k] for k in ("h", "y", "xy", "x", "oof", "base")]
        ev = np.arange(len(n), len(n) + len(m))

        def embed(x, h):
            result = extract(model, x, h.reshape(len(x), 4, 15, 40), "cpu")
            return sum(v for k, v in result.items() if k.startswith("level"))

        # Cached Hb differs from the strict coordinate; re-encode the exchange stage in that coordinate.
        if arm.startswith("cross_"):
            ne = embed(screen.array("wave"), h[: len(n)])
        else:
            with np.load(f / "science.npz") as z:
                ne = sum(z[k] for k in z.files if k.startswith("level"))
        e = slot_vectors(np.concatenate([ne, embed(x, h[ev])]), tr)
        hh = screen.hb_slots(h, tr)
        donor = np.r_[n.donor, m.donor + len(n)]
        y = y.astype(float)
        sd = y[tr].std(0)
        base = base.reshape(y[ev].shape)
        r = y[tr] - oof.reshape(y[tr].shape)
        t = dict(
            kind="A",
            people=np.r_[n.subject, m.subject],
            donor=donor,
            y=torch.tensor(y, dtype=torch.float32),
            xy=torch.tensor(xy, dtype=torch.float32),
            v=dict(hb=torch.from_numpy(hh), eeg=torch.from_numpy(e)),
            curve=[],
        )
        target = torch.tensor(r / sd, dtype=torch.float32)
        rows = []
        for reader_seed in screen.SEEDS:
            for which in ["anchor", "capacity", "true", "wrong"]:
                if which != "anchor":
                    mdl, _ = fit(
                        t,
                        which,
                        reader_seed,
                        tr,
                        target,
                        screen.parent.TR["residual_steps"],
                        screen.parent.TR,
                    )
                for swap in [False] if which in ("anchor", "capacity") else [False, True]:
                    with torch.no_grad():
                        pred = (
                            base
                            if which == "anchor"
                            else base + forward(mdl, t, ev, which, swap).numpy() * sd
                        )
                    loss = (((pred - y[ev]) / sd) ** 2).reshape(len(ev), -1).mean(1)
                    for who, idx in m.groupby("subject").indices.items():
                        rows.append(
                            dict(
                                model=arm,
                                tokenizer_seed=seed,
                                reader_seed=reader_seed,
                                arm=which,
                                input="mismatched" if swap else "aligned",
                                subject=who,
                                events=len(idx),
                                nmse=loss[idx].mean(),
                            )
                        )
        pd.DataFrame(rows).to_csv(out / "people.csv", index=False)
        save(out / "done.json", dict(status="completed", arm=arm, seed=seed, WG_events=len(ev)))


def main():
    check()
    torch.set_num_threads(1)
    cfg = yaml.safe_load(PROTOCOL.read_text())
    if (
        cfg["parent"] != CONFIG.name
        or cfg["output_child"] != "wg"
        or cfg["scope"]["protected"] != "forbidden"
    ):
        raise ValueError("transfer scope")
    with owner_lock(DEST):
        (DEST / "resolved.yaml").write_text(PROTOCOL.read_text())
        if not (DEST / "prepared.json").exists():
            save(DEST / "prepared.json", prepare())
        pending = {(a, s) for a in ARMS for s in TR["seeds"]}
        with ProcessPoolExecutor(5, mp_context=mp.get_context("fork")) as pool:
            while pending:
                jobs = [
                    t for t in sorted(pending) if (OUT / "models" / f"{t[0]}_{t[1]}" / "done.json").exists()
                ]
                if jobs:
                    list(pool.map(one, jobs))
                    pending.difference_update(jobs)
                else:
                    time.sleep(10)
        rows = pd.concat(
            [
                pd.read_csv(DEST / f"{a}_{s}" / "people.csv", dtype={"arm": str})
                for a in ARMS
                for s in TR["seeds"]
            ]
        )
        means = rows.groupby(["model", "arm", "input", "subject"], as_index=False).nmse.mean()
        means.to_csv(DEST / "people.csv", index=False)
        from src.analysis.conditional_evidence import effect_tables

        effects = effect_tables(
            means.query("input=='aligned'"),
            "nmse",
            [
                ("over_full", "anchor", "true"),
                ("over_capacity", "capacity", "true"),
                ("over_wrong", "wrong", "true"),
            ],
            ["model"],
        )
        effects.to_csv(DEST / "contrasts.csv", index=False)
        save(
            DEST / "summary.json",
            dict(
                status="completed",
                models=44,
                subjects=5,
                events=237,
                coordinate="offline canonical Hb on both tasks; all reader fitting on n-back train",
                new_subjects=["VP017"],
            ),
        )


if __name__ == "__main__":
    main()
