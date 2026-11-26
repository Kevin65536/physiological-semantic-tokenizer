"""The comparative local baselines must predict hidden IDs using only visible neighbours."""

import numpy as np
import pandas as pd
from experiments.scripts.analyze_full_masked_search import count_lookup, priors
from src.foundation.aligned_masked_search import search_mask


def test_sparse_priors_reference_score():
    panel = pd.DataFrame(
        {"dataset": ["d"] * 3, "fold": [0, 0, 1], "eeg_sites": ["A|B|C|D"] * 3, "hb_sites": ["A|B|C|D"] * 3}
    )
    ids = np.random.default_rng(123).integers(0, 4, (3, 4, 4, 15))
    scores = priors(panel, ids, 0, [4] * 4)
    np.testing.assert_array_equal(
        count_lookup(np.array([0, 0, 3]), np.array([0, 1, 2, 3, 4])), [2, 0, 0, 1, 0]
    )
    train = ids[:2, 0]
    truth = ids[2, 0]
    for scheme in ("site", "scatter", "span"):
        expected = []
        for repeat in range(4):
            mask = search_mask(1, scheme, np.random.default_rng(1126 + repeat))[0, 0]
            bits = []
            for i, t in zip(*np.where(mask)):
                y = truth[i, t]
                marginal = ((train == y).sum() + 0.5) / (train.size + 2)
                cond = ((train[:, i, t] == y).sum() + 16 * marginal) / 18
                edges = []
                for dt in (-1, 1):
                    if 0 <= t + dt < 15 and not mask[i, t + dt]:
                        if dt == -1:
                            src, dest = train[:, :, :-1], train[:, :, 1:]
                        else:
                            src, dest = train[:, :, 1:], train[:, :, :-1]
                        matched = src == truth[i, t + dt]
                        edges.append(((matched & (dest == y)).sum() + 16 * cond) / (matched.sum() + 16))
                bits.append(-np.log2([marginal, cond, np.mean(edges) if edges else cond]))
            expected.append(np.mean(bits, axis=0))
        np.testing.assert_allclose(scores.loc[(2, scheme, 0)].to_numpy(), np.mean(expected, axis=0))
