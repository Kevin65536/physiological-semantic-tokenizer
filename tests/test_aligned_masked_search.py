import numpy as np
import torch
import yaml
from pathlib import Path
from src.foundation.aligned_masked import FrozenCodeReader
from src.foundation.aligned_masked_search import search_mask, set_prior_head, log_priors, task_at, nominees


def test_search_contract_and_masks():
    cfg = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "experiments/configs/physiology_semantic_tokenizer/aligned_masked_search_v2.yaml"
        ).read_text()
    )
    assert np.prod([len(v) for v in cfg["grid"].values()]) == 24
    assert cfg["tensor"]["vocab"] == [8192, 8192, 1024, 1024] and cfg["scope"]["protected"] == "forbidden"
    rng = np.random.default_rng(17)
    for shape in ("span", "scatter", "site", "eeg_hidden", "hb_hidden"):
        mask = search_mask(16, shape, rng)
        assert mask.shape == (16, 2, 4, 15)
        if shape == "site":
            assert (mask.sum(2) == 2).all() and (mask == mask[..., :1]).all()
        elif shape in ("span", "scatter"):
            assert np.isin(mask.sum(-1), [7, 8]).all()
        else:
            assert (
                mask[:, int(shape == "hb_hidden")].all() and not mask[:, int(shape == "eeg_hidden")].any()
            )
    for percent in (25, 50):
        tasks = [task_at(i, "site", percent, "true_pair") for i in range(16)]
        assert sum(x for x, _ in tasks) == 8
        assert tasks.count((True, "hb_hidden")) == tasks.count((True, "eeg_hidden")) == percent // 25


def test_isolation_and_prior():
    rng = np.random.default_rng(17)
    books = [torch.randn(k, 128) for k in (19, 19, 11, 11)]
    net = FrozenCodeReader(
        books, range(4), dict(width=16, heads=2, layers=1, feedforward=32, dropout=0)
    ).eval()
    x = torch.stack([torch.randint(len(b), (3, 4, 15)) for b in books], 1)
    xy = torch.randn(3, 2, 4, 2)
    hidden = torch.from_numpy(search_mask(3, "scatter", rng))
    y = x.clone()
    y[:, 2:] = 0
    expected = net.encode(x, xy, hidden, isolated=True)
    torch.testing.assert_close(
        net.encode(y, xy, hidden, isolated=True)[:, :2], expected[:, :2], rtol=0, atol=0
    )
    torch.testing.assert_close(
        net.encode(x.masked_fill(hidden.repeat_interleave(2, 1), -1), xy, hidden, isolated=True),
        expected,
        rtol=0,
        atol=0,
    )
    lp = log_priors(x, [0, 1], [len(b) for b in books])
    set_prior_head(net, lp)
    for k, head in enumerate(net.output):
        torch.testing.assert_close(head(expected[0, k, 0, 0]).log_softmax(0), lp[k])
    assert all(not b.weight.requires_grad for b in net.books)
    assert torch.isfinite(net.objective(x, xy, hidden, isolated=True))


def test_delayed_signal_noise_and_promotion():
    rng = np.random.default_rng(1126)
    x = rng.normal(size=(100, 40))
    noise = rng.normal(size=x.shape)
    y = np.roll(x, 3, axis=1) + noise * 0.2

    def association(source, target):
        return [
            np.corrcoef(source[:, :-lag].ravel(), target[:, lag:].ravel())[0, 1] for lag in (1, 2, 3, 4, 5)
        ]

    assert np.argmax(association(x, y)) == 2 and max(map(abs, association(x, noise))) < 0.07
    from src.foundation.aligned_masked import compose

    ids = torch.arange(3 * 4 * 4 * 15).reshape(3, 4, 4, 15)
    xy = torch.randn(3, 2, 4, 2)
    got, loc = compose(ids, xy, [0, 1], [1, 0])
    assert torch.equal(got[:, 2:], ids[[1, 0], 2:]) and torch.equal(loc[:, 1], xy[[1, 0], 1])
    s = dict(
        local={m: dict(gain=1.0) for m in ("eeg", "hb")},
        cross={m: dict(prior=2.0, wrong=1.0, true=1.0) for m in ("eeg", "hb")},
    )
    out = nominees({("c", f, "true_pair"): s for f in (0, 1)}, {"c": {"width": 128}}, "A", {})
    assert len(out["local"]) == 1 and out["cross"] == []


def test_full_search_uses_every_fitting_window_and_disjoint_native_support():
    import pandas as pd
    import pytest
    from src.foundation.aligned_masked_search import coverage_draws, check_disjoint_windows

    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/configs/physiology_semantic_tokenizer/aligned_masked_search_full_v3.yaml"
    )
    cfg = yaml.safe_load(path.read_text())
    for split in ("train", "validation"):
        assert sum(cfg["full"]["by_dataset"][split].values()) == cfg["full"][split]
    ids = np.arange(23, 53)
    draws = coverage_draws(ids, 100, np.ones(len(ids)) / len(ids), np.random.default_rng(17))
    assert set(draws[: len(ids)]) == set(draws) == set(ids)
    with pytest.raises(ValueError):
        coverage_draws(ids, 10, None, np.random.default_rng(17))
    p = pd.DataFrame(
        dict(dataset=["d"] * 3, subject=["s"] * 3, record=["a", "a", "b"], grid_start_sample=[6000, 0, 0])
    )
    check_disjoint_windows(p)
    p.loc[0, "grid_start_sample"] -= 1
    with pytest.raises(ValueError):
        check_disjoint_windows(p)
