import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.foundation.typed_span import LagPath, TypedSpanReader, observed_view, span_mask, view_codes
from src.tokenizers.dual_component_tokenizer import GlobalLatentComponentTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/scripts"))
import run_typed_span as run


def tiny(width):
    model = GlobalLatentComponentTokenizer(
        dict(
            input_dim=width,
            patches=15,
            width=8,
            heads=2,
            latent_dim=8,
            encoder_layers=1,
            decoder_layers=1,
            kmeans_iters=1,
            geometry_each_forward=False,
        ),
        codebook_size=5,
    )
    model.initialize_codebooks(torch.randn(2, 15, width))
    return model.eval()


def test_schedule_and_fold():
    run.validate()
    schedule = [run.task(i) for i in range(24)]
    for mod in (0, 1):
        for src in [(0,), (1,), (0, 1)]:
            lengths = [n for m, n, s in schedule if m == mod and s == src]
            assert lengths == [3 if mod == 0 else 6] * 3 + [15]
    m = pd.DataFrame(dict(split=["train"] * 6, condition=["x"] * 6, subject=list("aabbcc")))
    d = run.assignment(m)
    assert all(m.subject != m.subject.to_numpy()[d])
    with pytest.raises(ValueError):
        run.assignment(m.iloc[:2])


@pytest.mark.parametrize("mod,width", [(0, 400), (1, 40)])
def test_hidden_substitution_and_ste(mod, width):
    torch.manual_seed(511)
    x = torch.randn(3, 1, 15, width)
    mask = span_mask(torch.tensor([0, 1, 8]), 6, 1)
    net = tiny(width)
    gain = torch.ones(1)
    old = copy.deepcopy(net.state_dict())

    def encode(v):
        return view_codes(net, observed_view(v, mask, mod, gain))

    values, ids = encode(x)
    altered, alt_ids = encode(torch.where(mask[..., None], torch.randn_like(x) * 100, x))
    for j, b in enumerate("SO"):
        assert torch.equal(values[j], altered[j]) and torch.equal(ids[j], alt_ids[j])
        table = net.branches[b]["quantizer"].codebook
        torch.testing.assert_close(values[j], table[ids[j]], atol=2e-6, rtol=1e-5)
    sum(v.square().sum() for v in values).backward()
    assert all(net.branches[b]["to_latent"].weight.grad.abs().sum() > 0 for b in "SO")
    assert all(torch.equal(v, old[k]) for k, v in net.state_dict().items())
    assert observed_view(x, torch.ones_like(mask), mod, gain).count_nonzero() == 0


def test_time_support_and_routing():
    torch.manual_seed(12)
    p = LagPath(8, [-2, 12], True)
    target = torch.randn(1, 1, 15, 8)
    source = torch.randn_like(target, requires_grad=True)
    for to_hb, slot in [(True, 0), (False, 14)]:
        grad = torch.autograd.grad(p(target, source, to_hb, True)[..., slot, :].sum(), source)[0]
        allowed = p.allowed if to_hb else p.allowed.T
        assert grad[..., ~allowed[slot], :].count_nonzero() == 0
        assert grad[..., allowed[slot], :].abs().sum() > 0
    reader = TypedSpanReader([5] * 4, 8, 8, run.P["paths"]["support_seconds"])
    vals = [source, source * 2]
    mask = torch.ones(1, 1, 15, dtype=torch.bool)
    xy = torch.zeros(1, 2, 1, 2)
    none = reader(vals, vals, xy, mask, 1, (), "joint")
    so = reader(vals, vals, xy, mask, 1, (0, 1), "ss_only")
    torch.testing.assert_close(none[1], so[1])
    assert not torch.equal(none[0], so[0])


def test_full_synthetic_update(monkeypatch):
    torch.manual_seed(19)
    # Keep the production tokenizer/STE and 30s grid while reducing vocabulary/reader size.
    monkeypatch.setattr(run, "SIZES", [5] * 4)
    monkeypatch.setitem(run.TR, "reader_width", 8)
    monkeypatch.setattr(
        run,
        "TypedSpanReader",
        lambda vocabs, dim, width, supports: TypedSpanReader(vocabs, 8, width, supports),
    )
    nets = [tiny(w) for w in (400, 40)]
    nets[0].spectral_head = torch.nn.Linear(8, 4)
    model = run.Model("joint", nets)
    model.local = torch.nn.ModuleList(
        [torch.nn.Identity(), torch.nn.Linear(8, 4), torch.nn.Linear(8, 40), torch.nn.Linear(8, 40)]
    )
    raw = [torch.randn(2, 4, 15, w) for w in (400, 40)]
    data = dict(
        raw=raw,
        xy=torch.randn(2, 2, 4, 2),
        gain=torch.ones(4),
        scale=torch.ones(4),
        anc=[(v, 0.4 * v) for v in raw],
        var=[[torch.ones(3, 4, c), torch.ones(3, 4, c)] for c in (1, 2)],
    )
    ix = np.arange(2)
    empty = torch.zeros(2, 4, 15, dtype=torch.bool)
    stats = [(0, 1)] * 4
    for mod in (0, 1):
        mask = run.span_mask(torch.tensor([0, 4]), 6, 4)
        src = run.codes(nets[1 - mod], data, 1 - mod, ix, empty)
        tgt = run.codes(nets[mod], data, mod, ix, mask)
        labels = run.codes(nets[mod], data, mod, ix, empty)[1]
        z = model.predict(src, tgt, data["xy"], mask, mod, (0, 1))
        span = sum(torch.nn.functional.cross_entropy(o[mask], y[mask]) for o, y in zip(z, labels))
        src_grads = torch.autograd.grad(
            span, [nets[1 - mod].branches[b]["to_latent"].weight for b in "SO"], retain_graph=True
        )
        assert all(g.abs().sum() > 0 for g in src_grads)
        aux, metrics = run.anchor_loss(model, data, ix, stats, True)
        assert metrics.shape == (2, 6)
        (span + aux).backward()
        torch.optim.AdamW(model.parameters()).step()
        model.zero_grad()


def test_frozen_control_hides_source_branch():
    from src.foundation.aligned_masked import FrozenCodeReader

    reader = FrozenCodeReader(
        [torch.randn(7, 128) for _ in range(4)],
        range(4),
        dict(width=8, heads=2, feedforward=16, layers=1, dropout=0.0),
    ).eval()
    ids = torch.randint(7, (2, 4, 1, 15))
    hide = torch.zeros_like(ids, dtype=torch.bool)
    hide[:, 1] = True
    hide[:, 2:, :, 3:8] = True
    xy = torch.rand(2, 2, 1, 2)
    perturbed = torch.where(hide, (ids + 1) % 7, ids)
    a = reader.encode(ids, xy, None, stream_hidden=hide)
    b = reader.encode(perturbed, xy, None, stream_hidden=hide)
    assert torch.equal(a, b)
