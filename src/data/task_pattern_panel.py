"""Load retained three-task screen shards with their exact public record identity."""

import numpy as np
import pandas as pd


def record_key(g):
    return "_".join(str(g.iloc[0][k]) for k in ("task", "subject", "record"))


def load_task_panel(run_root, task):
    frame = pd.read_csv(run_root / "plan.csv").query("task==@task")
    if (
        frame.empty
        or set(frame.split) != {"train", "validation"}
        or frame.groupby("subject").split.nunique().max() != 1
    ):
        raise ValueError("invalid public task roster")
    rows = []
    parts = {k: [] for k in ["e", "h", "ef", "hf"]}
    for _, g in frame.groupby(["subject", "record"]):
        name = record_key(g)
        folder = run_root / "records"
        m = pd.read_csv(folder / (name + ".csv"))
        rows.append(m)
        cols = [
            "id",
            "task",
            "dataset",
            "split",
            "subject",
            "record",
            "condition",
            "label",
            "event",
            "eeg_start",
            "hb_start",
        ]
        pd.testing.assert_frame_equal(m[cols].reset_index(drop=True), g[cols].reset_index(drop=True))
        with np.load(folder / (name + ".npz")) as z:
            ids = m.loc[m.status.eq("complete"), "id"]
            np.testing.assert_array_equal(z["ids"], ids)
            if not len(ids):
                continue
            with np.load(run_root / "hb" / (name + ".npz")) as hb:
                np.testing.assert_array_equal(hb["ids"], ids)
                for k in ("e", "ef", "h"):
                    parts[k].append(z[k])
                parts["hf"].append(hb["hf"])
    ledger = pd.concat(rows, ignore_index=True)
    a = {k: np.concatenate(v) for k, v in parts.items()}
    m = ledger.query("status=='complete'").reset_index(drop=True)
    assert all(len(v) == len(m) for v in a.values())
    return ledger, m, a
