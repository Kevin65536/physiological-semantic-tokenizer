"""Losslessly reorder waveform words using only an already decoded predecessor."""

import numpy as np


def previous_states(words, state_by_word):
    words = np.asarray(words)
    if words.dtype.kind not in "ui" or words.ndim != 2 or words.shape[1] < 1:
        raise ValueError("expected integer sequences")
    if np.any(words < 0) or np.any(words >= len(state_by_word)):
        raise ValueError("original word outside vocabulary")
    states = np.zeros_like(words, dtype=np.uint8)
    states[:, 1:] = state_by_word[words[:, :-1]]
    return states


def word_stats(words, attributes, size):
    ids = np.ravel(words)
    counts = np.bincount(ids, minlength=size)
    totals = np.stack([np.bincount(ids, weights=v, minlength=size) for v in attributes.T], axis=1)
    return counts, totals


def order_table(probability, attributes=None):
    """Keep zero fixed. Balance dictionary cardinality, never slice out a rare word."""
    size = len(probability)
    words = np.arange(1, size)
    if size < 5 or size > 65536 or (size % 4):
        raise ValueError("expected a multiple of four entries including missing zero")
    table = np.zeros(size, np.uint16)
    if attributes is None:
        order = np.lexsort((words, -probability[words]))
        table[words[order]] = np.arange(1, size)
        return table
    # Interleaved slots retain different spectral meanings at equal frequency rank.
    slots = [np.arange(1 + g, size, 4) for g in range(4)]
    by_theta = words[np.lexsort((words, attributes[words, 0]))]
    split = len(slots[0]) + len(slots[1])
    for side, selected in enumerate((by_theta[:split], by_theta[split:])):
        by_alpha = selected[np.lexsort((selected, attributes[selected, 1]))]
        first = 2 * side
        cut = len(slots[first])
        for group, part in ((first, by_alpha[:cut]), (first + 1, by_alpha[cut:])):
            ranking = np.lexsort((part, -probability[part]))
            table[part[ranking]] = slots[group]
    return table


def fit_site(words, valid, properties, atom_energy, prior_counts, prior_properties):
    """Fit three reversible models on one native site's outer training subjects."""
    size = len(prior_counts)
    used = words[valid]
    if not len(used):
        raise ValueError("site has no valid training words")
    cuts = np.quantile(atom_energy[used], [0.25, 0.5, 0.75])
    state = (1 + np.searchsorted(cuts, atom_energy, side="right")).astype(np.uint8)
    state[0] = 0
    states = previous_states(words, state)
    site_count, total = word_stats(used, properties[valid], size)
    site_mean = (total + 8 * prior_properties) / (site_count[:, None] + 8)
    site_weight = site_count + 32 * prior_counts / max(1, prior_counts.sum())
    base = order_table(site_weight)
    rank, semantic = [], []
    for i in range(5):
        take = valid & (states == i)
        count, total = word_stats(words[take], properties[take], size)
        weight = count + 32 * site_weight / site_weight.sum()
        mean = (total + 8 * site_mean) / (count[:, None] + 8)
        rank.append(order_table(weight))
        semantic.append(order_table(weight, mean))
    return state, dict(static=base[None], rank=np.array(rank), semantic=np.array(semantic))


def encode_words(words, state, table):
    states = previous_states(words, state)
    return table[0, words] if len(table) == 1 else table[states, words]


def decode_words(sent, state, inverse):
    sent = np.asarray(sent)
    if (
        sent.dtype.kind not in "ui"
        or sent.ndim != 2
        or sent.shape[1] < 1
        or np.any(sent < 0)
        or np.any(sent >= inverse.shape[1])
    ):
        raise ValueError("valid transmitted integers required")
    decoded = np.empty_like(sent)
    row = np.zeros(len(sent), dtype=np.uint8)
    for t in range(sent.shape[1]):
        decoded[:, t] = inverse[row, sent[:, t]]
        if len(inverse) > 1:
            row = state[decoded[:, t]]
    return decoded


def compact_words(words, state, representatives, waveforms):
    """Select a compact index relative to the already reconstructed prefix."""
    words = np.asarray(words)
    previous_states(words, state)  # Validate high-rate source integers.
    size = len(state)
    if (
        representatives.ndim != 2
        or len(representatives) not in (1, 5)
        or np.any(representatives[:, 0] != 0)
        or np.any(representatives[:, 1:] == 0)
        or waveforms.shape[0] != size
    ):
        raise ValueError("expected fixed missing zero and a complete native atom dictionary")
    used = np.unique(words)
    nearest = np.zeros((len(representatives), size), dtype=np.uint16)
    for index, atoms in enumerate(representatives):
        # The original atom takes precedence over any numerically equal prototype.
        nearest[index, atoms] = np.arange(len(atoms), dtype=np.uint16)
        unknown = used[(used != 0) & (nearest[index, used] == 0)]
        if not len(unknown):
            continue
        centers = np.asarray(waveforms[atoms[1:]], np.float64)
        norms = np.einsum("ij,ij->i", centers, centers)
        for start in range(0, len(unknown), 256):
            ids = unknown[start : start + 256]
            dot = np.asarray(waveforms[ids], np.float64) @ centers.T
            nearest[index, ids] = 1 + np.argmin(norms[None, :] - 2 * dot, axis=1)
    sent = np.empty_like(words, dtype=np.uint16)
    reconstructed = np.empty_like(words)
    context = np.zeros_like(words, dtype=np.uint8)
    for t in range(words.shape[1]):
        if t and len(representatives) > 1:
            context[:, t] = state[reconstructed[:, t - 1]]
        row = context[:, t]
        sent[:, t] = nearest[row, words[:, t]]
        reconstructed[:, t] = representatives[row, sent[:, t]]
    return sent, reconstructed, context
