from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from src.foundation.aligned_masked import FrozenCodeReader, condition_permutation, hidden_sites


def test_visibility_and_four_frozen_vocabs():
    c = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "experiments/configs/physiology_semantic_tokenizer/aligned_masked_pretraining_v1.yaml"
        ).read_text()
    )
    assert (
        c["vocab"] == [8192, 8192, 1024, 1024]
        and c["scope"]["test"] == c["scope"]["confirmation"] == "forbidden"
    )
    books = [torch.randn(k, 128) for k in [17, 17, 9, 9]]
    net = FrozenCodeReader(books, range(4), dict(width=16, layers=1, heads=2, feedforward=32, dropout=0))
    ids = torch.stack([torch.randint(len(v), (2, 4, 15)) for v in books], 1)
    xy = torch.randn(2, 2, 4, 2)
    rng = np.random.default_rng(17)
    for task in ["span", "eeg_hidden", "hb_hidden"]:
        mask = torch.tensor(hidden_sites(2, 4, 15, task, rng))
        if task == "span":
            assert ((mask.sum(-1) == 7) | (mask.sum(-1) == 8)).all()
            assert (torch.diff(mask.int(), dim=-1).abs().sum(-1) <= 2).all()
        else:
            target = int(task == "hb_hidden")
            assert mask[:, target].all() and not mask[:, 1 - target].any()
        altered = ids.masked_fill(mask.repeat_interleave(2, 1), -123)
        torch.testing.assert_close(net.encode(ids, xy, mask), net.encode(altered, xy, mask), rtol=0, atol=0)
        loss = net.objective(ids, xy, mask)
        loss.backward()
        assert torch.isfinite(loss) and all(v.weight.grad is None for v in net.books)


def test_donors_are_conditional_disjoint_bijections():
    frame = pd.DataFrame(
        dict(
            split=["train"] * 5,
            dataset=["d"] * 5,
            subject=["p"] * 5,
            record=["r"] * 5,
            condition=["MA"] * 4 + ["BL"],
            event=range(5),
            eeg_start=[0, 29, 60, 120, 240],
            hb_start=[5, 34, 65, 125, 245],
        )
    )
    d = condition_permutation(frame, 60, 17)
    k = np.flatnonzero(d >= 0)
    assert d[4] == -1 and len(k) >= 2
    assert set(k) == set(d[k]) and (k != d[k]).all()
    assert (abs(frame.eeg_start.to_numpy()[k] - frame.eeg_start.to_numpy()[d[k]]) >= 60).all()
