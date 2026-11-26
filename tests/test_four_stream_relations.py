"""Synthetic forward/backward checks. No measurements or checkpoints are accessed."""

import pytest
import torch

from src.foundation.four_stream_relations import (
    ComponentReadout,
    HbHistoryBaseline,
    ResponseResidual,
    check_history_support,
)


def fixture():
    gen = torch.Generator().manual_seed(17)
    books = [torch.randn(k, 128, generator=gen) for k in (20, 21, 12, 13)]
    ids = torch.stack([torch.randint(len(b), (3, 4, 15), generator=gen) for b in books], 1)
    xy = torch.randn(3, 2, 4, 2, generator=gen)
    return books, ids, xy


def test_a_nested_residual_training_budget_and_frozen_hb_anchor():
    books, ids, xy = fixture()
    e, h = ids[:, :2], ids[:, 2:]
    ex, hx = xy[:, 0], xy[:, 1]
    target = torch.zeros(3, 4, 3, 2)
    base = HbHistoryBaseline(books[2:], 8, 3)
    opt = torch.optim.Adam(base.parameters(), lr=0.001)
    (base(h, hx) - target).square().mean().backward()
    opt.step()
    base.requires_grad_(False)
    base_prediction = base(h, hx)
    counts = []
    for source_books, src, loc in [(books[2:], h, hx), (books[:2], e, ex)]:
        model = ResponseResidual(books[2:], source_books, 8, 3)
        counts.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
        residual, features = model(h, hx, src, loc)
        assert residual.shape == target.shape and features.shape == (3, 4, 8)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        (base_prediction + residual - target).square().mean().backward()
        opt.step()
        assert all(m.words.weight.grad is None for m in [*model.hb, *model.source])
        assert model.source[0].lags.grad.abs().sum() > 0
        assert not torch.equal(residual, model(h, hx, src.roll(1, 0), loc.roll(1, 0))[0])
    assert counts[0] == counts[1]
    torch.testing.assert_close(base(h, hx), base_prediction)


@pytest.mark.parametrize("which", [0, 1])
def test_b_additivity_and_interaction_are_separately_observable(which):
    books, ids, xy = fixture()
    sl = slice(2 * which, 2 * which + 2)
    pair = ids[:, sl]
    loc = xy[:, which]
    for rank in [0, 3]:
        model = ComponentReadout(books[sl], 8, 2, rank=rank)

        def predict(s, o):
            value, features = model(torch.stack([s, o], 1), loc)
            assert features.shape[-1] == 16 + rank
            return value

        s, o = pair.unbind(1)
        u, v = s.roll(1, 0), o.roll(1, 0)
        diff = predict(s, o) - predict(u, o) - predict(s, v) + predict(u, v)
        if rank:
            assert diff.abs().max() > 1e-5
            diff.square().mean().backward()
            assert model.left.weight.grad.abs().sum() > 0
        else:
            torch.testing.assert_close(diff, torch.zeros_like(diff), atol=1e-6, rtol=0)


def test_cutoff_applies_to_entire_input_chain_and_both_native_clocks():
    start = torch.tensor([[-30.0, -30.0]])
    stop = torch.zeros_like(start)
    cutoff = torch.zeros(1)
    targets = torch.tensor([[0.0, 2.0, 6.0]])
    check_history_support(start, stop, cutoff, targets, targets + 2)
    for starts, ends, future in [
        (start + 2, stop, targets),
        (start, torch.tensor([[0.0, 90.0]]), targets),
        (start, stop, targets - 1),
        (start, stop * float("nan"), targets),
    ]:
        with pytest.raises(ValueError):
            check_history_support(starts, ends, cutoff, future, targets + 2)


def test_readers_reject_combined_or_noninteger_id_streams():
    books, ids, xy = fixture()
    model = ComponentReadout(books[:2], 8, 1)
    with pytest.raises(ValueError, match="one modality"):
        model(ids, xy[:, 0])
    with pytest.raises(ValueError, match="integer"):
        model(ids[:, :2].float(), xy[:, 0])
