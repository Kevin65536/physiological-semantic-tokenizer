"""Sampling and ranking rules for the frozen-code search."""

import numpy as np
import torch
from .aligned_masked import hidden_sites


def search_mask(batch, policy, rng):
    if policy in ("span", "eeg_hidden", "hb_hidden"):
        return hidden_sites(batch, 4, 15, policy, rng)
    if policy == "scatter":
        ranks = rng.random((batch, 2, 4, 15)).argsort(-1).argsort(-1)
        return ranks < rng.integers(7, 9, (batch, 2, 4, 1))
    if policy == "site":
        ranks = rng.random((batch, 2, 4, 1)).argsort(2).argsort(2)
        return np.broadcast_to(ranks < 2, (batch, 2, 4, 15)).copy()
    raise ValueError(policy)


def task_at(step, mask, percent, arm):
    strict = step % 2 == 1
    slot = (step % 16) // 2
    if strict and arm in ("true_pair", "wrong_pair") and slot < {25: 2, 50: 4}[percent]:
        return strict, ("hb_hidden", "eeg_hidden")[slot % 2]
    return strict, mask


def window_weights(panel, indices):
    p = panel.loc[indices]
    people = p[["dataset", "subject"]].drop_duplicates().groupby("dataset").size()
    counts = p.groupby(["dataset", "subject"]).subject.transform("size")
    weights = 1 / (p.dataset.map(people) * counts)
    return (weights / weights.sum()).to_numpy()


def log_priors(ids, indices, vocab):
    return [
        torch.bincount(ids[indices, k].flatten(), minlength=n).add(0.5).log().log_softmax(0)
        for k, n in enumerate(vocab)
    ]


def set_prior_head(model, priors):
    with torch.no_grad():
        for head, k in zip(model.output, model.streams):
            head.weight.zero_()
            # Learnable bias equals a fixed log prior plus a trainable residual bias.
            # Exclude the initial prior from decoupled weight decay by keeping it in a buffer.
            head.bias.zero_()
            torch.nn.utils.parametrize.register_parametrization(head, "bias", PriorOffset(priors[k]))


class PriorOffset(torch.nn.Module):
    def __init__(self, prior):
        super().__init__()
        self.register_buffer("prior", prior.detach().clone())

    def forward(self, bias):
        return self.prior + bias


def nominees(scores, grid, stage, previous):
    selected = {}
    for route, names in [("local", ["eeg", "hb"]), ("cross", ["hb", "eeg"])]:
        ranked = []
        for candidate, spec in grid.items():
            allowed = (
                names
                if stage == "A"
                else [v["target"] for v in previous.get(route, []) if v["candidate"] == candidate]
            )
            choices = []
            for target in allowed:
                margins = []
                for fold in (0, 1):
                    main = scores[candidate, fold, "true_pair"]
                    if route == "local":
                        margins.append(main["local"][target]["gain"])
                    else:
                        m = main["cross"][target]
                        margins.extend([m["prior"] - m["true"], m["wrong"] - m["true"]])
                        if stage == "B":
                            margins.append(
                                scores[candidate, fold, "wrong_pair"]["cross"][target]["true"] - m["true"]
                            )
                choices.append((min(margins), target))
            if choices:
                margin, target = max(choices, key=lambda x: x[0])
                if margin > 0:
                    ranked.append(dict(candidate=candidate, target=target, margin=margin))
        ranked.sort(key=lambda r: (-r["margin"], grid[r["candidate"]]["width"], r["candidate"]))
        selected[route] = ranked[: 2 if stage == "A" else 1]
    return selected


def coverage_draws(pool, draws, weights, rng):
    """Include every available fitting window once before balanced replacement draws."""
    if draws < len(pool):
        raise ValueError("local budget cannot cover the full fitting panel")
    first = rng.choice(pool, len(pool), replace=False, p=weights)
    return np.concatenate((first, rng.choice(pool, draws - len(pool), p=weights)))


def check_disjoint_windows(panel, samples_per_window=6000):
    """Check physical support in the G0 unified 200 Hz acquisition grid."""
    for _, group in panel.groupby(["dataset", "subject", "record"]):
        starts = np.sort(group["grid_start_sample"].to_numpy(dtype=np.int64))
        if np.any(np.diff(starts) < samples_per_window):
            raise ValueError("overlapping 30-second windows within a native record")
