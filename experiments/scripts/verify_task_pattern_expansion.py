"""Recompute the retained ranking, prediction and denominator evidence independently."""

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score
from run_task_pattern_expansion import OUT, read
from src.utils.io import write_json


def verify():
    summary = read(OUT / "summary.json")["tasks"]
    support = pd.read_csv(OUT / "support.csv")
    assert support.id.is_unique and set(support.id) == set(pd.read_csv(OUT / "plan.csv").id)
    assert (
        set(support.split) == {"train", "validation"}
        and support.groupby(["dataset", "subject"]).split.nunique().max() == 1
    )
    assert len(support) == 3880 and sum(support.status == "complete") == 2777
    audit = {}
    for task, s in summary.items():
        folder = OUT / task
        trial = pd.read_csv(folder / "pattern_trials.csv.gz", float_precision="round_trip")
        fields = [c for c in trial if "__" in c]
        assert len(fields) == 27
        rows = []
        for (split, sub, _), g in trial.groupby(["split", "subject", "record"]):
            if g.label.nunique() == 2:
                for c in fields:
                    family, name = c.split("__")
                    rows.append((split, sub, family, name, roc_auc_score(g.label, g[c])))
        pp = (
            pd.DataFrame(rows, columns=["split", "subject", "family", "name", "auc"])
            .groupby(["split", "subject", "family", "name"])
            .auc.mean()
        )
        tgt = pd.read_csv(folder / "pattern_people.csv").set_index(pp.index.names).auc
        np.testing.assert_allclose(pp, tgt.reindex(pp.index), rtol=0, atol=1e-14)
        pairs = pd.read_csv(folder / "coupling_predictions.csv.gz", float_precision="round_trip")
        valid = support.set_index("id")
        m = valid.loc[pairs.id].reset_index(drop=True)
        d = valid.loc[pairs.donor].reset_index(drop=True)
        assert (
            (m.subject == d.subject)
            & (m.record == d.record)
            & (m.condition == d.condition)
            & (m.split == "validation")
        ).all()
        assert ((m.eeg_start - d.eeg_start).abs() >= 30).all() and (
            (m.hb_start - d.hb_start).abs() >= 30
        ).all()
        # Restore the source float32 measurement and training SD stored using shortest CSV decimals.
        obs = pairs.observed.to_numpy(dtype="float32").astype(float)
        sd = pairs.train_sd.to_numpy(dtype="float32").astype(float)
        np.testing.assert_allclose(((pairs.pred - obs) / sd) ** 2, pairs.nmse, rtol=1e-11, atol=1e-13)
        df = pd.read_csv(folder / "predictions.csv.gz")
        va = (
            df.groupby(["model", "subject"])
            .apply(lambda g: recall_score(g.label, g.pred, average="macro"), include_groups=False)
            .groupby("model")
            .mean()
        )
        for model, val in va.items():
            np.testing.assert_allclose(val, s["BA"][model], atol=1e-14)
        audit[task] = dict(
            auc_person_values=len(pp),
            model_predictions=len(df),
            coupling_cells=len(pairs),
            development=int((trial.split == "validation").sum()),
        )
    cross = pd.read_csv(OUT / "transfer.csv")
    assert (
        cross.passes_both.sum() == 1
        and cross.query("passes_both").iloc[0]["name"] == "motor_beta_rebound_lr"
    )
    ev = pd.read_csv(OUT / "event_pairs/record_condition.csv")
    assert ev[["on", "off"]].min().min() >= 2
    np.testing.assert_allclose(ev.p_on - ev.p_off, ev["diff"])
    cat = pd.read_csv(OUT / "report_v1/pattern_catalog.csv")
    assert (len(cat), int(cat.selected.sum()), int(cat.passes.sum())) == (81, 12, 4)
    loc = pd.read_csv(OUT / "localization/development_features.csv")
    np.testing.assert_allclose(loc.beta_left - loc.beta_right, loc.beta_lr, atol=2e-6)
    assert not pd.read_csv(OUT / "event_pairs/contrasts.csv").passes.any()
    write_json(
        OUT / "verification.json",
        dict(
            status="passed",
            tasks=audit,
            rankings_verified=81,
            paired_donors_and_nmse_verified=True,
            macro_balanced_accuracy_verified=True,
            selected=12,
            within_task_passed=4,
            transfer_passed=1,
            event_pairs_passed=0,
            report="report_v1/verification.json",
            units=[
                "task-pattern-expansion-1127-" + name + ".service"
                for name in ["pilot", "extract", "score", "pairs", "localize"]
            ],
        ),
    )


if __name__ == "__main__":
    verify()
