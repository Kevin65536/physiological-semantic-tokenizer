"""Subject-cross-fitted anchors and zero-initialized conditional residuals."""

import numpy as np
import torch
from src.analysis.four_stream_bridge import grouped_folds
from src.foundation.conditional_slot_readers import AllHb, HbIncrement, AdditiveSlots, LocalInteraction


def slots(data):
    return {
        m: torch.tensor(
            np.stack([data[m + b + "_book"][data[m + "_ids"][:, j]] for j, b in enumerate("SO")], axis=2)
        )
        for m in ("eeg", "hb")
    }


def setup(data, meta, kind):
    mod = "eeg" if kind == "B_eeg" else "hb"
    y = torch.tensor(data[mod + "_target"], dtype=torch.float32).reshape(len(meta), 4, 3, -1)
    return dict(
        v=slots(data),
        xy=torch.tensor(data["xy"], dtype=torch.float32),
        y=y,
        mod=mod,
        kind=kind,
        people=meta.subject.to_numpy(),
        donor=meta.donor.to_numpy(),
        curve=[],
    )


def forward(model, t, idx, arm, swap=False):
    v, xy = t["v"], t["xy"]
    j = t["donor"][idx] if swap else idx
    if t["kind"] == "A":
        if arm == "anchor":
            return model(v["hb"][idx], xy[idx, 1])
        src = "hb" if arm == "capacity" else "eeg"
        src_idx = idx if src == "hb" else j
        return model(v["hb"][idx], xy[idx, 1], v[src][src_idx], xy[src_idx, int(src == "hb")])
    x = v[t["mod"]]
    pair = torch.stack([x[idx, :, 0], x[j, :, 1]], 2)
    return model(pair, xy[idx, int(t["mod"] == "hb")])


def new_model(kind, arm, cfg, seed, colors):
    torch.manual_seed(seed)
    if kind == "A":
        return AllHb(cfg["width"]) if arm == "anchor" else HbIncrement(cfg["width"])
    return (
        AdditiveSlots(cfg["width"], colors)
        if arm == "anchor"
        else LocalInteraction(cfg["rank"], colors, additive=arm == "additive_residual")
    )


def fit(t, arm, seed, indices, target, steps, cfg, *, check=(), score=None, name=""):
    model = new_model(t["kind"], arm, cfg, seed, t["y"].shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(seed)
    names, inverse, counts = np.unique(t["people"][indices], return_inverse=True, return_counts=True)
    prob = 1 / counts[inverse]
    prob /= prob.sum()
    outputs = {}
    for step in range(1, steps + 1):
        sel = rng.choice(len(indices), cfg["batch"], p=prob)
        z = forward(model, t, indices[sel], arm, arm == "wrong")
        loss = (z - target[sel]).square().mean()
        if arm != "anchor":
            loss = loss + cfg["residual_l2"] * z.square().mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite fit")
        opt.zero_grad()
        loss.backward()
        opt.step()
        t["curve"].append((name, arm, step, loss.item()))
        if step in check:
            with torch.no_grad():
                outputs[step] = [forward(model, t, score, arm, wrong).detach() for wrong in (False, True)]
    return model, outputs


def cross_anchor(t, seed, tr, ev, cfg, fold_count, label):
    """Select steps with held-person losses, return physical OOF anchor for each training event."""
    steps = cfg["anchor_step_grid"]
    y = t["y"]
    oof = {n: torch.empty(2, len(y), *y.shape[1:]) for n in steps}
    losses = {n: [] for n in steps}
    splits = grouped_folds(t["people"], tr, fold_count)
    for f, (a, b) in enumerate(splits):
        mu = y[a].mean(0)
        sd = y[a].std(0, unbiased=False)
        if sd.min() <= 1e-9:
            raise ValueError("degenerate fold")
        _, z = fit(
            t,
            "anchor",
            seed,
            a,
            (y[a] - mu) / sd,
            max(steps),
            cfg,
            check=steps,
            score=b,
            name=f"{label}/fold{f}",
        )
        for n in steps:
            err = (z[n][0] - (y[b] - mu) / sd).square().flatten(1).mean(1).numpy()
            losses[n].extend(err[t["people"][b] == p].mean() for p in np.unique(t["people"][b]))
            oof[n][:, b] = torch.stack(z[n]) * sd + mu
    chosen = min(steps, key=lambda n: np.mean(losses[n]))
    mu = y[tr].mean(0)
    sd = y[tr].std(0, unbiased=False)
    model, z = fit(
        t, "anchor", seed, tr, (y[tr] - mu) / sd, chosen, cfg, check=[chosen], score=ev, name=label + "/refit"
    )
    detail = dict(
        chosen=chosen,
        mean_cv={n: float(np.mean(losses[n])) for n in steps},
        folds=[sorted(set(t["people"][b])) for _, b in splits],
    )
    return model, oof[chosen][:, tr], torch.stack(z[chosen]) * sd + mu, mu, sd, detail


def nested_fits(t, seed, tr, ev, cfg, nfold, label):
    anchor, oof, base, mu, sd, selection = cross_anchor(t, seed, tr, ev, cfg, nfold, label)
    models = {"anchor": anchor}
    pred = {"anchor": base}
    for arm in ["capacity", "true", "wrong"] if t["kind"] == "A" else ["true", "wrong"]:
        flip = int(arm == "wrong")
        r, _ = fit(t, arm, seed, tr, (t["y"][tr] - oof[flip]) / sd, cfg["residual_steps"], cfg, name=label)
        models[arm] = r
        with torch.no_grad():
            pred[arm] = torch.stack([base[k] + forward(r, t, ev, arm, bool(k)) * sd for k in (0, 1)])
    return pred, models, mu, sd, selection


def run_conditional(t, seed, tr, dev, cfg):
    # Outer evaluation of the WHOLE procedure, including inner OOF residual target formation.
    cv = {}
    selections = {}
    for k, (a, b) in enumerate(grouped_folds(t["people"], tr, cfg["folds"])):
        pred, _, _, _, sel = nested_fits(t, seed, a, b, cfg, 2, f"outer{k}")
        selections[f"outer{k}"] = dict(held=sorted(set(t["people"][b])), inner=sel)
        for arm, z in pred.items():
            if arm not in cv:
                cv[arm] = torch.empty_like(t["y"])
            cv[arm][b] = z[0]
    pred, models, mu, sd, sel = nested_fits(t, seed, tr, dev, cfg, cfg["folds"], "full")
    selections["full"] = sel
    calibrations = {}
    for arm in pred:
        x = cv[arm][tr]
        y = t["y"][tr]
        xm = x.mean(0)
        ym = y.mean(0)
        # Same affine calibration rule for every arm; coefficients fitted only on outer OOF predictions.
        var = (x - xm).square().mean(0)
        beta = ((x - xm) * (y - ym)).mean(0) / var.clamp_min(1e-12)
        calibrations[arm] = (ym - beta * xm, beta)
    return pred, models, mu, sd, selections, calibrations, {a: z[tr] for a, z in cv.items()}
