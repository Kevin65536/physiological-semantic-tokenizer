from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/scripts"))
import run_expanded_typed_span as run
from src.foundation.typed_span import TypedSpanReader


def test_scope_and_donors():
    run.check()
    m = pd.DataFrame(
        dict(
            dataset=["x"] * 5,
            subject=["p"] * 5,
            split=["train"] * 5,
            task=["t"] * 5,
            record=["a"] * 5,
            condition=["A"] * 4 + ["B"],
            eeg_start=[0, 10, 80, 120, 300],
            hb_start=[10, 20, 90, 130, 310],
        )
    )
    d = run.pair_donors(m)
    assert d[-1] == -1 and sorted(d[:-1]) == list(range(4))
    assert min(abs(m.eeg_start.values[:4] - m.eeg_start.values[d[:4]])) >= 30


def test_attention_replays_forward_and_null():
    torch.manual_seed(28)
    net = TypedSpanReader([16] * 4, 8, 12, run.base.P["paths"]["support_seconds"]).eval()
    x = [torch.randn(3, 4, 15, 8) for _ in "SO"]
    xy = torch.randn(3, 2, 4, 2)
    for mod in (0, 1):
        mask = torch.rand(3, 4, 15) > 0.5
        z, w = net(x, x, xy, mask, mod, (0, 1), "joint", inspect=True)
        normal = net(x, x, xy, mask, mod, (0, 1), "joint")
        for u, v in zip(z, normal):
            torch.testing.assert_close(u, v)
        for i, a in w.items():
            torch.testing.assert_close(a.sum(-1), torch.ones_like(a[..., 0]))
            allowed = net.paths[i].allowed if mod else net.paths[i].allowed.T
            assert a[..., :-1][..., ~allowed].count_nonzero() == 0


def test_order_ids_and_known_lags():
    from src.analysis.typed_span_patterns import pattern_codes, pattern_fields
    from score_expanded_typed_span import known_delay

    a = np.array([[1, 3, 2, 0, 6]], dtype=np.uint8)
    b = np.array([[0, 0, 5, 7, 0]], dtype=np.uint8)
    assert pattern_fields(pattern_codes(a, b, 2)[0, 0]) == [1, 3, 5, 7]
    checks = known_delay()
    assert [r["peak_s"] for r in checks if r["control"] == "true"] == [0, 4, 8]


def test_motif_freeze_and_label_transfer():
    from src.analysis.typed_span_patterns import comparisons, transfer_selected

    rng = np.random.default_rng(191)
    meta = pd.DataFrame(
        dict(
            task=["t"] * 24,
            split=["train"] * 16 + ["validation"] * 8,
            subject=np.repeat(list("abcdef"), 4),
            donor=np.arange(24).reshape(-1, 4)[:, [1, 2, 3, 0]].ravel(),
        )
    )
    fam = rng.integers(0, 8, (24, 4, 4, 15), dtype=np.uint8)
    selected, people, _ = comparisons(fam, meta)
    altered = fam.copy()
    altered[16:] = 0
    pd.testing.assert_frame_equal(comparisons(altered, meta)[0], selected)
    transferred, mapping = transfer_selected(fam, (fam + 1) % 8, selected, meta)
    assert all(v["train_agreement"] == 1 for v in mapping)
    for name in ["true_count", "wrong_count", "reverse_count", "opportunities"]:
        np.testing.assert_array_equal(transferred[name], people[name])
