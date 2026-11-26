"""Sensor-level atlas of the frozen four-stream audit's retained observations."""

import numpy as np
import pandas as pd

from src.analysis.single_vq_relations import interval, wave_attribute


PERSON = ["dataset", "subject"]
WINDOW = [*PERSON, "record", "parent", "start"]
SITE = ["dataset", "split", "layer", "position", "x", "y", "family"]


def check_partitions(rows):
    if rows[[*PERSON, "split", "record"]].isna().any().any():
        raise ValueError("missing subject, record or split")
    if not set(rows.split).issubset({"train", "validation"}):
        raise ValueError("unexpected split")
    if rows.groupby(PERSON).split.nunique().gt(1).any():
        raise ValueError("subject crosses partitions")


def spatial_counts(panel, positions, arrays):
    """Recover discrete emission counts and verify the upstream array row order."""
    check_partitions(panel)
    if panel.duplicated(WINDOW).any() or not np.isfinite(panel[["eeg_start", "hb_start"]]).all().all():
        raise ValueError("duplicate window or missing native time")
    complete = panel.loc[panel.status.eq("complete")].reset_index(drop=True)
    if set(positions["mod"]) != {"eeg", "hb"}:
        raise ValueError("unexpected modalities")
    chunks = []
    for mod in ("eeg", "hb"):
        sites = positions.loc[positions["mod"].eq(mod)].reset_index(drop=True)
        keys = WINDOW + ["split", "eeg_start", "hb_start"]
        expected = complete.loc[complete.index.repeat(4), keys].reset_index(drop=True)
        if not expected.equals(sites[keys]) or sites.duplicated(WINDOW + ["position"]).any():
            raise ValueError("position identities or ordering do not match complete windows")
        if not np.isfinite(sites[["x", "y"]]).all().all():
            raise ValueError("invalid sensor position")
        observed = arrays[mod + "_y"]
        if observed.shape != (len(complete), 4, 15):
            raise ValueError("unexpected observed attribute dimensions")
        np.testing.assert_allclose(observed.mean(-1).ravel(), sites.observed, atol=1e-6)
        for branch in "SO":
            f = arrays[mod + branch]
            if f.shape != (len(complete), 4, 3, 4):
                raise ValueError("unexpected family tensor dimensions")
            votes = np.rint(5 * f)
            if not (
                np.isfinite(f).all()
                and (votes >= 0).all()
                and (votes.sum(-1) == 5).all()
                and np.allclose(votes, 5 * f, atol=1e-7, rtol=0)
            ):
                raise ValueError("family counts do not conserve five valid patches per bin")
            counts = votes.sum(2).astype(int).reshape(-1, 4)
            if branch == "S":
                np.testing.assert_allclose(counts[:, 3] / 15, sites.Q4, atol=1e-7)
            chunk = sites[keys + ["condition", "event", "position", "x", "y"]].copy()
            chunk["layer"] = mod.upper() + "-" + branch
            for family in range(4):
                chunks.append(chunk.assign(family=family + 1, emissions=counts[:, family], opportunities=15))
    return pd.concat(chunks, ignore_index=True)


def summarize_sites(counts):
    subjects = counts.groupby(SITE + ["subject"], as_index=False).agg(
        emissions=("emissions", "sum"), opportunities=("opportunities", "sum"), records=("record", "nunique")
    )
    subjects["occupancy"] = subjects.emissions / subjects.opportunities
    rows = []
    for key, group in subjects.groupby(SITE):
        mean, lo, hi = interval(group.occupancy, group.dataset)
        rows.append(
            dict(
                zip(SITE, key),
                occupancy=mean,
                people=len(group),
                opportunities=group.opportunities.sum(),
                emissions=group.emissions.sum(),
                ci_low=lo if len(group) > 1 else None,
                ci_high=hi if len(group) > 1 else None,
            )
        )
    return subjects, pd.DataFrame(rows)


def hb_member_slopes(manifest, waves, mapping, branch):
    """Keep measured X and its teacher component, linked to each actual wave in the audit."""
    check_partitions(manifest)
    selected = manifest.loc[manifest.group.isin(["low", "middle", "high"]) & manifest.kind.eq("member")]
    rows = []
    for (group, word, split), data in selected.groupby(["group", "word", "split"]):
        key = f"{group}_{word}_{split}_member"
        data = data.sort_values("pair").copy()
        a = waves[key]
        if not np.array_equal(data.pair, np.arange(len(data))) or a.shape != (len(data), 3, 40):
            raise ValueError("member manifest / waveform mismatch")
        if not np.isfinite(a).all() or not data.patch.between(0, 14).all():
            raise ValueError("invalid member support")
        data["family"] = int(mapping[word]) + 1
        data["layer"] = "HB-" + branch
        data["wave_key"] = key
        data["patch_start_in_parent_s"] = data.crop_s + 2 * data.patch
        for label, idx, attr in [
            ("X", 0, "x_attribute"),
            (branch, "SO".index(branch) + 1, "branch_attribute"),
        ]:
            wave = a[:, idx]
            oxy = wave_attribute(wave, "hb")
            deoxy = wave_attribute(wave.reshape(-1, 20, 2)[:, :, ::-1].reshape(-1, 40), "hb")
            np.testing.assert_allclose(oxy, data[attr], atol=2e-5, rtol=1e-5)
            rows.append(data.assign(signal=label, HbO_slope=oxy, HbR_slope=deoxy))
    return pd.concat(rows, ignore_index=True)


def paired_readout(frame, mod):
    table = frame.loc[frame.target.eq("X") & frame.method.isin(["S", "SO"])].pivot(
        index=PERSON, columns="method", values="mse"
    )
    if table.empty or not np.isfinite(table[["S", "SO"]]).all().all() or (table < 0).any().any():
        raise ValueError("incomplete paired readout")
    table["difference"] = table.S - table.SO
    macro = table.groupby(level="dataset").mean().mean()
    change, lo, hi = interval(table.difference, table.index.get_level_values("dataset"))
    return table.reset_index().assign(modality=mod), dict(
        modality=mod,
        people=len(table),
        improved=int((table.difference > 0).sum()),
        S_mse=macro.S,
        SO_mse=macro.SO,
        relative_gain=1 - macro.SO / macro.S,
        difference=change,
        ci_low=lo,
        ci_high=hi,
    )
