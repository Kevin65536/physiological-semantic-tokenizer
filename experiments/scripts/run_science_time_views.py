"""Equal-support local attribution with the frozen EEG tokenizer, measured and synthetic."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
from scipy.linalg import solve

from prepare_science_ablations import OUT, CFG, check
import run_frozen_science_screen as screen
from run_dual_component_tokenizer import owner_lock
from src.analysis.frozen_science_screen import anchor, gram, residual

DEST = OUT / "time_views"
save = screen.parent.save


@torch.inference_mode()
def local(net, wave, slot):
    """Extract one pair of actual hard IDs using exactly one contiguous 30s voltage history."""
    shape = wave.shape[:-2]
    ids = []
    for i in range(0, np.prod(shape), 32):
        xx = np.array(wave.reshape(-1, 15, 400)[i : i + 32], dtype="float32", copy=True)
        zs = net.encode_continuous(torch.from_numpy(xx))
        ii = []
        for b in "SO":
            q = net.branches[b]["quantizer"]
            ii.append(q._assignment_logits(zs[b][:, slot], q.codebook).argmax(1).numpy())
        ids.append(np.stack(ii, axis=-1))
    return np.concatenate(ids).reshape(*shape, 2)


def vectors(net, ids):
    return np.stack(
        [net.branches[b]["quantizer"].codebook.detach().numpy()[ids[..., j]] for j, b in enumerate("SO")],
        axis=-2,
    )


def measured():
    m, _, _ = screen.panel()
    history = {}
    valid = {}
    for s in m.subject.unique():
        f = pd.read_csv(OUT / "data" / (s + "_history.csv"))
        with np.load(OUT / "data" / (s + "_views.npz")) as z:
            history.update(zip(f.id[f.valid], z["history"], strict=True))
        valid.update(zip(f.id, f.valid))
    direct = np.array([valid[i] for i in m.id])
    mask = direct.copy()
    while np.any(mask & ~mask[m.donor]):
        mask &= mask[m.donor]
    panel = m.assign(time_support=direct, donor_cycle_support=mask)
    panel.to_csv(DEST / "support.csv", index=False)
    keep = np.flatnonzero(mask)
    p = m.iloc[keep].reset_index(drop=True)
    donor = np.searchsorted(keep, p.donor)
    np.testing.assert_array_equal(keep[donor], p.donor)
    tr = np.flatnonzero(p.split == "train")
    ev = np.flatnonzero(p.split == "validation")
    w = np.stack([history[i] for i in p.id])
    orig = screen.array("wave")[keep]
    np.testing.assert_allclose(w[:, :, 7:], orig[:, :, :8], atol=2e-6)
    timeline = np.concatenate([w[:, :, :7], orig], axis=2)
    net = screen.audit.network("eeg", 8192, 17)
    hist = local(net, w, 14)
    bidir = local(net, orig, 7)
    np.testing.assert_array_equal(bidir, screen.array("ids")[keep, :, :, 7])
    h = screen.array("hb")[keep]
    hk = gram(screen.hb_slots(h, tr))
    y = screen.array("target")[keep].astype(float).reshape(len(p), -1)
    sd = y[tr].std(0)
    oof, base, _ = anchor(h, y, p.subject.values, tr, ev, screen.GRID)
    err = y[tr] - oof
    square = gram(screen.hb_slots(h, tr) ** 2)
    rows = []
    spatial = []
    maps = screen.array("map64")
    families = lambda a: np.stack([maps[j, a[:, :, j]] for j in range(2)], axis=-1)
    # For transitions only, use two windows per view with 32s total support in each view.
    prior = dict(bidir=local(net, timeline[:, :, 6:21], 8), history=hist)
    nxt = dict(bidir=local(net, orig, 8), history=local(net, timeline[:, :, 1:16], 14))
    for view, ids in [("history", hist), ("bidir", bidir)]:
        x = vectors(net, ids)
        mu = x[tr].mean(0)
        xs = x[tr].std(0)
        sk = gram((x - mu) / np.where(xs > 1e-7, xs, 1))
        for arm in ["anchor", "capacity", "true", "wrong"]:
            for swap in [False] if arm in ("anchor", "capacity") else [False, True]:
                pred = (
                    base
                    if arm == "anchor"
                    else base
                    + residual(
                        hk, square if arm == "capacity" else sk, err, tr, ev, donor, arm == "wrong", swap
                    )
                )
                loss = (((pred - y[ev]) / sd) ** 2).mean(1)
                for who, idx in p.iloc[ev].reset_index(drop=True).groupby("subject").indices.items():
                    rows.append(
                        dict(
                            subject=who,
                            view=view,
                            arm=arm,
                            input="mismatched" if swap else "aligned",
                            nmse=loss[idx].mean(),
                            events=len(idx),
                        )
                    )
        f = families(ids)
        a = families(prior[view])
        b = families(nxt[view])
        for who, idx in p.iloc[ev].reset_index(drop=True).groupby("subject").indices.items():
            ix = ev[idx]
            c = p.condition.values[ix]
            for j, branch in enumerate("SO"):
                # Unseen IDs (64) remain outside the upper-quartile class.
                high = (f[ix, :, j] >= 48) & (f[ix, :, j] < 64)
                first = (a[ix, :, j] >= 48) & (a[ix, :, j] < 64)
                second = (b[ix, :, j] >= 48) & (b[ix, :, j] < 64)
                spatial.append(
                    dict(
                        subject=who,
                        view=view,
                        branch=branch,
                        events=len(ix),
                        seen_fraction=(f[ix, :, j] < 64).mean(),
                        posterior_minus_frontal=high[:, 2:].mean() - high[:, :2].mean(),
                        n_3back=int(sum(c == "3-back session")),
                        n_0back=int(sum(c == "0-back session")),
                        load_difference=(
                            high[c == "3-back session"].mean() - high[c == "0-back session"].mean()
                            if {"3-back session", "0-back session"} <= set(c)
                            else np.nan
                        ),
                        start_support=int(first.sum()),
                        transition=(first & second).sum() / first.sum() if first.any() else np.nan,
                    )
                )
    pd.DataFrame(rows).to_csv(DEST / "person_scores.csv", index=False)
    pd.DataFrame(spatial).to_csv(DEST / "family_effects.csv", index=False)
    return dict(
        planned=len(m),
        history_available=int(direct.sum()),
        closed_donor_panel=len(p),
        train=len(tr),
        development=len(ev),
        prediction_support=dict(bidir=[-30, 0], history=[-44, -14], same_patch=[-16, -14]),
        transition_seconds_per_view=32,
    )


def synthetic(seed):
    rng = np.random.default_rng(seed)
    n = 112
    length = n + 28
    latent = rng.normal(size=length)
    # A 10 Hz carrier with local stochastic amplitude plus independent broadband voltage.
    t = np.arange(400) / 200
    wave = (
        np.exp(0.55 * latent[:, None]) * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi, (length, 1)))
        + 0.40 * rng.normal(size=(length, 400))
    ).astype("float32")
    truth = np.arange(14, 14 + n)
    windows = lambda shifts: np.stack([wave[i + shifts : i + shifts + 15] for i in truth])
    net = screen.audit.network("eeg", 8192, 17)
    views = {"history": local(net, windows(-14), 14), "bidir": local(net, windows(-7), 7)}
    # Responses in the common chronology with a two-slot lag (4 seconds).
    pos = latent[truth - 2] + rng.normal(0, 0.45, n)
    null = rng.normal(size=n)
    return {k: vectors(net, v).reshape(n, -1) for k, v in views.items()}, latent[truth], pos, null


def correlations(x, y):
    # Identical inner support at every lag, no circular time shifts in the estimator.
    i = np.arange(5, len(x) - 5)
    lags = np.arange(-5, 6)
    return np.array([np.corrcoef(x[i - lag], y[i])[0, 1] for lag in lags])


def synthetic_panel(cases):
    mid = CFG["E4"]["null_and_positive_seeds"]
    rows = []
    for view in ["history", "bidir"]:
        x = np.concatenate([a[0][view] for a in cases[:mid]]).astype(float)
        y = np.concatenate([a[1] for a in cases[:mid]])
        mu = x.mean(0)
        xc = x - mu
        cov = xc.T @ xc
        coef = solve(
            cov + 0.01 * cov.trace() / len(mu) * np.eye(len(mu)), xc.T @ (y - y.mean()), assume_a="pos"
        )
        for j, (features, _, pos, null) in enumerate(cases[mid:]):
            pred = (features[view] - mu) @ coef + y.mean()
            cp = correlations(pred, pos)
            cn = correlations(pred, null)
            sur = [max(abs(correlations(pred, np.roll(null, s)))) for s in range(12, len(null) - 11)]
            est = np.argmax(cp) - 5
            rows.append(
                dict(
                    seed=1100 + mid + j,
                    view=view,
                    estimated_seconds=int(2 * est),
                    absolute_error_seconds=abs(2 * est - 4),
                    positive_peak=cp.max(),
                    null_peak=max(abs(cn)),
                    null_p=(1 + sum(v >= max(abs(cn)) for v in sur)) / (1 + len(sur)),
                )
            )
    table = pd.DataFrame(rows)
    table.to_csv(DEST / "synthetic_lag.csv", index=False)
    return {
        k: dict(
            seeds=len(g),
            mean_lag_absolute_error_s=float(g.absolute_error_seconds.mean()),
            null_false_peaks=int(sum(g.null_p <= 0.05)),
        )
        for k, g in table.groupby("view")
    }


def main():
    check()
    torch.set_num_threads(1)
    with owner_lock(DEST):
        measured_summary = measured()
        with ProcessPoolExecutor(8, mp_context=mp.get_context("fork")) as pool:
            cases = list(pool.map(synthetic, range(1100, 1100 + 2 * CFG["E4"]["null_and_positive_seeds"])))
        save(
            DEST / "summary.json",
            dict(
                status="completed",
                measured=measured_summary,
                synthetic=synthetic_panel(cases),
                endpoint="paired single-patch views, same windows per quantity; synthetic results qualify only the lag estimator",
            ),
        )


if __name__ == "__main__":
    main()
