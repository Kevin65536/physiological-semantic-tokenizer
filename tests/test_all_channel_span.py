import sys
from pathlib import Path

import torch

from src.foundation.all_channel_span import NativeSpanReader, hidden_sites, masked_wave


def test_rectangular_native_set_padding_permutation_and_gradients():
    torch.manual_seed(19)
    r = NativeSpanReader([11, 11, 7, 7], 16, [(-2, 12)] * 4, 4, 0.75)
    sites = (3, 5)
    codes = [[torch.randn(2, c, 15, 128) for _ in range(2)] for c in sites]
    xyz = [torch.randn(2, c, 3) * 0.1 for c in sites]
    xy = [v[..., :2] for v in xyz]
    valid = [torch.ones(2, c, 15, dtype=torch.bool) for c in sites]
    valid[0][:, -1] = False
    xyz[0][:, -1] = torch.nan
    hidden = torch.zeros_like(valid[1])
    hidden[:, :, :4] = True
    z = r.encode_target(codes, xyz, xy, valid, hidden, 1)
    perm = torch.tensor([1, 0])
    v = r.encode_target([[x[:, perm] for x in codes[0]], codes[1]], [xyz[0][:, perm], xyz[1]],
                        [xy[0][:, perm], xy[1]], [valid[0][:, perm], valid[1]], hidden, 1)
    for a, b in zip(z, v):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    sum(r.output[2+i](a[hidden]).square().mean() for i, a in enumerate(z)).backward()
    assert all(p.gate.grad.isfinite() for p in r.paths)
    assert r.paths[0].value.weight.grad.abs().sum() > 0
    missing = [torch.zeros_like(valid[0]), valid[1]]
    full = r.encode_target(codes, xyz, xy, missing, hidden, 1)
    control = r.encode_target(codes, xyz, xy, missing, hidden, 1, False)
    torch.testing.assert_close(full[0], control[0])


def test_raw_mask_reference_and_contract():
    from importlib import import_module
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/scripts"))
    try:
        runner = import_module("run_all_channel_span")
        runner.check()
    finally:
        sys.path.pop(0)
    raw = torch.randn(2, 7, 15, 40)
    valid = torch.ones(2, 7, 15, dtype=torch.bool)
    valid[:, -1] = False
    xyz = torch.randn(2, 7, 3)
    for kind in runner.FAMILIES:
        hide = hidden_sites(valid, xyz, kind, 1, torch.Generator().manual_seed(2))
        assert hide.any() and not (hide & ~valid).any()
        shown = valid & ~hide
        first = masked_wave(raw, shown, 1)
        other = masked_wave(raw.masked_fill(~shown[..., None], 1e4), shown, 1)
        torch.testing.assert_close(first, other)
        assert first[~shown].eq(0).all()
