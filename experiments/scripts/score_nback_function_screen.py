"""Train-person-selected linear probes and subject-level decisions for the n-back screen."""

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from run_nback_function_screen import CFG, CONFIG, OUT, ROOT, PARENT, alpha_power, check_panel, pulse, read
from src.analysis.four_stream_block_ridge import band_histories
from src.analysis.nback_function_screen import flat, join, path_features, phase, probe
from src.utils.io import write_json


def synthetic():
    n = 24
    rng = np.random.default_rng(17)
    g = np.repeat(np.arange(6), 4)
    y = np.eye(3)[np.arange(n) % 3]
    x = y + 0.01 * rng.normal(size=y.shape)
    pred, _, _ = probe(
        x,
        y,
        g,
        np.arange(18),
        np.arange(18, n),
        classification=True,
        ridge=CFG["ridge"],
        folds=CFG["folds"],
    )
    np.testing.assert_array_equal(pred.argmax(1), y[18:].argmax(1))
    xx = np.broadcast_to(pulse(), (2, 4, 15, 400))
    assert band_histories(xx, CFG["bands"]).shape == (2, 4, 15, 5)
    assert alpha_power(xx).shape == (2, 4, 15)
    a = np.array([[[1, 1, 0, 3], [2, 1, 1, 0]]])
    occ, order = path_features(a)
    np.testing.assert_array_equal(occ, path_features(a[..., ::-1])[0])
    assert not np.array_equal(order, path_features(a[..., ::-1])[1])
    assert CFG["timing"] == dict(start_s=-5, stop_s=25, eeg_hz=200, hb_hz=10, slots=15)
    assert CFG["scope"]["splits"] == ["train", "validation"] and CFG["model"]["seed"] == 17
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "resolved.yaml").write_text(CONFIG.read_text())
    write_json(
        OUT / "synthetic.json",
        dict(
            status="passed",
            fixture="separable 3-class readout, grouped fitting, order reversal preserves occupancy/run lengths, pulse and band shapes",
        ),
    )


def family_lut():
    df = pd.read_csv(ROOT / CFG["families"])
    d = df[df.layer.eq("EEG-S")]
    # The atlas exports training-observed IDs only. Full frozen training mapping also assigns unseen words.
    src = read(ROOT / d.definition_source.iloc[0])
    lookup = np.asarray(src["mapping"], dtype=int)
    np.testing.assert_array_equal(lookup[d.id], d.family - 1)
    assert lookup.shape == (8192,) and set(lookup) == set(range(4))
    return lookup


def score(meta):
    check_panel(meta)
    ledger = []
    arrays = {}
    noisy = {}
    for sub, g in meta.groupby("subject"):
        base = OUT / "subjects" / sub
        frame = pd.read_csv(base.with_suffix(".csv"))
        assert list(frame.id) == list(g.id)
        ledger.append(frame)
        with np.load(base.with_suffix(".npz")) as z:
            np.testing.assert_array_equal(z["id"], frame.query("status=='complete'").id)
            for k in z.files:
                if k == "id":
                    continue
                store = (
                    noisy
                    if k.startswith(("dirty", "power_", "transfer_")) or k in ("e", "wave_S")
                    else arrays
                )
                store.setdefault(k, []).append(z[k])
    ledger = pd.concat(ledger, ignore_index=True)
    ledger.to_csv(OUT / "support.csv", index=False)
    m = ledger.query("status=='complete'").reset_index(drop=True)
    a = {k: np.concatenate(v) for k, v in arrays.items()}
    z = {k: np.concatenate(v) for k, v in noisy.items()}
    p = m.subject.to_numpy()
    tr = np.flatnonzero(m.split.eq("train"))
    ev = np.flatnonzero(m.split.eq("validation"))
    dev = m.iloc[ev].reset_index(drop=True)
    y = np.eye(3)[m.condition.map(CFG["classes"].index)]
    if not (
        len(set(p[ev])) == 4
        and len(set(p[tr])) == 15
        and (dev.groupby("subject").condition.nunique() == 3).all()
    ):
        raise ValueError("incomplete subject-class support")
    with np.load(PARENT / "panel.npz") as books:

        def embeds(ids, mod):
            return join(*[phase(books[mod + b + "_book"][ids[:, :, j]]) for j, b in enumerate("SO")])

        E = embeds(a["ids"], "eeg")
        H = embeds(a["hids"], "hb")
    h = a["h"]
    ht = h.reshape(len(h), 4, 15, 20, 2).mean(3)
    hc = phase(ht)
    ec = phase(a["bands_X"])
    method_inputs = {
        "continuous_E": ec,
        "continuous_H": hc,
        "continuous_EH": join(ec, hc),
        "token_E": E,
        "token_H": H,
        "token_EH": join(E, H),
    }
    predrows = []
    select = []

    def record(name, scores, condition):
        for row, values in zip(dev.itertuples(), scores, strict=True):
            predrows.append(
                dict(
                    id=row.id,
                    subject=row.subject,
                    true=CFG["classes"].index(row.condition),
                    method=name,
                    input=condition,
                    pred=int(np.argmax(values)),
                    **{f"score_{j}": float(v) for j, v in enumerate(values)},
                )
            )

    for name, x in method_inputs.items():
        pr, _, reg = probe(x, y, p, tr, ev, classification=True, ridge=CFG["ridge"], folds=CFG["folds"])
        record(name, pr, "clean")
        select.append(dict(method=name, ridge=reg))
    x = join(ec, hc)
    scaler = StandardScaler().fit(x[tr])
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
        scaler.transform(x[tr]), y[tr].argmax(1)
    )
    record("hybrid_LDA", lda.decision_function(scaler.transform(x[ev])), "clean")
    for method in ("X", "HP", "S", "O", "SO"):
        pred, dist, reg = probe(
            phase(a["bands_" + method]),
            y,
            p,
            tr,
            ev,
            classification=True,
            dirty=phase(z["dirty_bands_" + method]),
            ridge=CFG["ridge"],
            folds=CFG["folds"],
        )
        record(method, pred, "clean")
        record(method, dist, "pulse")
        select.append(dict(method=method, ridge=reg))
    pred = pd.DataFrame(predrows)
    pred.to_csv(OUT / "task_predictions.csv", index=False)
    people = (
        pred.groupby(["method", "input", "subject"])
        .apply(lambda g: balanced_accuracy_score(g.true, g.pred), include_groups=False)
        .rename("ba")
        .reset_index()
    )
    people.to_csv(OUT / "task_people.csv", index=False)
    fam = family_lut()[a["ids"][:, :, 0]]
    early = fam[:, :, 3:7]  # edges +1,+3,+5,+7,+9 s
    occ, order = path_features(early)
    rng = np.random.default_rng(CFG["seed"])
    flipped = rng.random(early.shape[:2]) < 0.5
    placebo = np.where(flipped[..., None], early[..., ::-1], early)
    oc2, ord2 = path_features(placebo)
    np.testing.assert_array_equal(occ, oc2)
    target = (h[:, :, 140:240].mean(2) - h[:, :, 60:140].mean(2)).reshape(len(h), 2, 2, 2).mean(2)
    base = join(y, ht[:, :, 3:7], a["bands_X"][:, :, 3:7], occ)
    reg_rows = []
    for name, x in (
        ("continuous_occ", base),
        ("order", join(base, order)),
        ("random_direction", join(base, ord2)),
    ):
        pr, _, reg = probe(
            x, flat(target), p, tr, ev, classification=False, ridge=CFG["ridge"], folds=CFG["folds"]
        )
        select.append(dict(method=name, ridge=reg))
        for i, row in enumerate(dev.itertuples()):
            for k, (obs, est, sd) in enumerate(zip(flat(target)[ev[i]], pr[i], flat(target)[tr].std(0))):
                reg_rows.append(
                    dict(
                        id=row.id,
                        subject=row.subject,
                        method=name,
                        region=("frontal", "parietal")[k // 2],
                        color=("HbO", "HbR")[k % 2],
                        observed=obs,
                        pred=est,
                        train_sd=sd,
                        nmse=((obs - est) / sd) ** 2,
                    )
                )
    response = pd.DataFrame(reg_rows)
    response.to_csv(OUT / "response_predictions.csv", index=False)
    pd.DataFrame(select).to_csv(OUT / "selection.csv", index=False)
    fidelity = []
    for k in ("X", "HP", "S", "O", "SO"):
        for i, row in enumerate(dev.itertuples()):
            fidelity.append(
                dict(
                    id=row.id,
                    subject=row.subject,
                    method=k,
                    transfer=z["transfer_" + k][i],
                    alpha_ratio=z["power_" + k][i],
                )
            )
    fidelity = pd.DataFrame(fidelity)
    fidelity.to_csv(OUT / "fidelity.csv", index=False)
    ba = people.query("input=='clean'").pivot(index="subject", columns="method", values="ba")
    nms = response.groupby(["subject", "method"]).nmse.mean().unstack()
    contrasts = {}

    def compare(name, diff, threshold):
        contrasts[name] = dict(
            mean=float(diff.mean()),
            positive=int((diff > 0).sum()),
            by_person=diff.to_dict(),
            passes=bool(diff.mean() >= threshold and (diff > 0).sum() >= 3),
        )

    for candidate, controls in [
        ("continuous_EH", ["continuous_E", "continuous_H"]),
        ("token_EH", ["token_E", "token_H", "continuous_EH", "hybrid_LDA"]),
    ]:
        for ref in controls:
            compare(candidate + "-" + ref, ba[candidate] - ba[ref], 0.02)
    for ref in ("continuous_occ", "random_direction"):
        compare("order_over_" + ref, nms[ref] - nms.order, 0.02)
    s = fidelity.query("method=='S'").groupby("subject")[["transfer", "alpha_ratio"]].median()
    s["pulse_ba_minus_x"] = people.query("method=='S' and input=='pulse'").set_index("subject").ba - ba.X
    s["passes"] = (s.transfer < 0.5) & s.alpha_ratio.between(0.8, 1.25) & (s.pulse_ba_minus_x >= -0.02)
    s.to_csv(OUT / "denoise_gate.csv")
    avg = s.mean()
    gates = {
        key: all(v["passes"] for name, v in contrasts.items() if name.startswith(prefix))
        for key, prefix in (
            ("continuous_complement", "continuous_EH-"),
            ("token_advantage", "token_EH-"),
            ("order", "order_over_"),
        )
    }
    gates["denoise"] = bool(
        s.passes.sum() >= 3
        and avg.transfer < 0.5
        and 0.8 <= avg.alpha_ratio <= 1.25
        and avg.pulse_ba_minus_x >= -0.02
    )
    summary = dict(
        experiment_id=CFG["experiment_id"],
        status="completed",
        task="nback",
        planned=ledger.groupby("split").size().to_dict(),
        scored=m.groupby("split").size().to_dict(),
        subjects=m.groupby("split").subject.nunique().to_dict(),
        exclusions=ledger.query("status!='complete'").status.value_counts().to_dict(),
        balanced_accuracy=ba.mean().to_dict(),
        response_nmse=nms.mean().to_dict(),
        contrasts=contrasts,
        gates=gates,
        branch_S_person_median_mean=avg.to_dict(),
    )
    write_json(OUT / "summary.json", summary)


if __name__ == "__main__":
    score(pd.read_csv(PARENT / "panel.csv"))
