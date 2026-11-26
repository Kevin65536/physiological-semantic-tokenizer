"""Task-locked continuous and frozen-family pattern features for six EEG and four Hb sites."""

import numpy as np
from scipy.stats import rankdata

BANDS = ((4, 8), (8, 13), (13, 30))
PRE, ACTIVE, LATE, TAIL = (slice(0, 2), slice(3, 7), slice(7, 11), slice(11, 15))


def spectral_db(wave):
    """Hann periodogram powers in dB re 1 microvolt squared, one value per 2 s."""
    if wave.shape[-3:] != (6, 15, 400) or not np.isfinite(wave).all():
        raise ValueError("invalid 6-site 200 Hz EEG")
    win = np.hanning(400)
    power = np.abs(np.fft.rfft((wave - wave.mean(-1, keepdims=True)) * win, axis=-1)) ** 2
    power *= 2 / (400 * np.sum(win**2))
    return np.stack(
        [10 * np.log10(np.maximum(power[..., 2 * lo : 2 * hi].sum(-1), 1e-20)) for lo, hi in BANDS], -1
    )


def features(e, h, ef, hf):
    """e,h and integer family arrays are event, site, time, attribute/branch."""
    if (
        e.shape[1:] != (6, 15, 3)
        or h.shape[1:] != (4, 15, 2)
        or ef.shape[1:] != (6, 15, 2)
        or hf.shape[1:] != (4, 15, 2)
    ):
        raise ValueError("unexpected panel shape")

    def delta(v, win, ref):
        return v[:, :, win].mean(2) - v[:, :, ref].mean(2)

    ed = delta(e, ACTIVE, PRE)
    rebound = delta(e, TAIL, ACTIVE)
    # Hb is already referenced to the actual first 5s.
    hd = h[:, :, LATE].mean(2)

    def bilateral(v, i, k):
        return v[:, i : i + 2, k].mean(1)

    result = {}

    def add(group, name, x):
        result[(group, name)] = x

    ft = bilateral(ed, 0, 0)
    pa = bilateral(ed, 2, 1)
    for name, x in dict(
        frontal_theta=ft,
        posterior_alpha=pa,
        theta_minus_alpha=ft - pa,
        frontal_alpha=bilateral(ed, 0, 1),
        motor_alpha_lr=ed[:, 4, 1] - ed[:, 5, 1],
        motor_beta_lr=ed[:, 4, 2] - ed[:, 5, 2],
        posterior_alpha_rebound=bilateral(rebound, 2, 1),
        motor_beta_rebound_lr=rebound[:, 4, 2] - rebound[:, 5, 2],
    ).items():
        add("EEG", name, x)
    for j, col in enumerate(("HbO", "HbR")):
        add("Hb", "frontal_" + col, bilateral(hd, 0, j))
        add("Hb", "motor_" + col + "_lr", hd[:, 2, j] - hd[:, 3, j])
        add("Hb", "frontal_" + col + "_late_minus_early", bilateral(delta(h, LATE, ACTIVE), 0, j))
    add("Hb", "frontal_HbO_minus_HbR", bilateral(hd, 0, 0) - bilateral(hd, 0, 1))
    for j, branch in enumerate("SO"):
        q = (ef[:, :, :, j] == 3).astype(float)
        v = q[:, :, ACTIVE].mean(2) - q[:, :, PRE].mean(2)
        add("EEGword", branch + "_posterior_Q4", v[:, 2:4].mean(1))
        add("EEGword", branch + "_motor_Q4_lr", v[:, 4] - v[:, 5])
        add("EEGword", branch + "_frontal_Q4", v[:, :2].mean(1))
        hbq = (hf[:, :, :, j] == 3).astype(float)
        add("Hbword", branch + "_frontal_Q4", hbq[:, :2, LATE].mean((1, 2)) - hbq[:, :2, PRE].mean((1, 2)))
        add("Hbword", branch + "_motor_Q4_lr", hbq[:, 2, LATE].mean(1) - hbq[:, 3, LATE].mean(1))
        add(
            "Hbword",
            branch + "_frontal_Q4_timing",
            hbq[:, :2, LATE].mean((1, 2)) - hbq[:, :2, ACTIVE].mean((1, 2)),
        )
    return result


def auc_columns(values, labels):
    values = np.asarray(values)
    labels = np.asarray(labels, dtype=bool)
    n = labels.sum()
    b = len(labels) - n
    if not n or not b:
        return np.full(values.shape[1], np.nan)
    ranks = rankdata(values, axis=0)
    return (ranks[labels].sum(0) - n * (n + 1) / 2) / (n * b)


def family_features(families):
    eye = np.eye(4)[families]
    # One occupancy vector per site/branch/phase. Adjacent directed transitions inside the same phase.
    phases = (PRE, ACTIVE, LATE, TAIL)
    occ = np.column_stack([eye[:, :, s].mean(2).reshape(len(eye), -1) for s in phases])
    transitions = []
    for s in phases:
        q = eye[:, :, s]
        trans = np.einsum("nstbi,nstbj->nsbij", q[:, :, :-1], q[:, :, 1:]) / (q.shape[2] - 1)
        transitions.append(trans.reshape(len(eye), -1))
    return occ, np.column_stack(transitions)


def disjoint_donors(m):
    ix = np.full(len(m), -1)
    for _, ind in m.groupby(["subject", "record", "label"]).indices.items():
        v = ind[np.argsort(m.event.to_numpy()[ind])]
        for k in sorted(range(1, len(v)), key=lambda k: abs(k - len(v) / 2)):
            other = np.roll(v, k)
            if (
                min(
                    np.min(abs(m[c].to_numpy()[v] - m[c].to_numpy()[other]))
                    for c in ["hb_start", "eeg_start"]
                )
                >= 30
            ):
                ix[v] = other
                break
    return ix
