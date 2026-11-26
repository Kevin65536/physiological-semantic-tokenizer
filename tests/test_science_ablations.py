import numpy as np
import torch
from src.tokenizers.dual_component_tokenizer import GlobalLatentComponentTokenizer
from src.tokenizers.science_ablations import ScienceTokenizer, initialize, objective


def test_integer_levels_and_preencoding_masks():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    cfg = dict(
        input_dim=400,
        patches=15,
        width=16,
        latent_dim=128,
        heads=2,
        encoder_layers=1,
        decoder_layers=1,
        channels=1,
        single_id_spectrum=True,
    )
    for arm, sizes in [("cross_hb_only", [8]), ("dual13_first", [8, 4])]:
        model = ScienceTokenizer(GlobalLatentComponentTokenizer(cfg, codebook_size=8), arm, sizes)
        x = torch.randn(3, 4, 15, 400)
        h = torch.randn(3, 4, 15, 40)
        latent = model.encode(x, h)
        for b, z in latent.items():
            for q in model.books(b):
                initialize(q, z.detach())
                z = z - q.get_embedding(
                    q._assignment_logits(z.detach().reshape(-1, 128)).argmax(1)
                ).reshape_as(z)
        model.eval()
        out = model(x, h)
        replay = model.decode(model.lookup(out["ids"]))
        torch.testing.assert_close(replay["S"], out["S"], rtol=0, atol=0)
        if arm == "cross_hb_only":
            torch.testing.assert_close(model(x + 10, h)["S"], out["S"], rtol=0, atol=0)
        model.train()
        s = x * 0.2
        loss = objective(
            model(x, h),
            x,
            s,
            torch.ones(3, 4, 1, 1),
            torch.ones(3, 4, 1, 1),
            torch.ones(4),
            torch.stack([torch.zeros(4), torch.ones(4)]),
            [0, 0],
            0.01,
        )
        loss.backward()
        assert torch.isfinite(loss)


def test_interaction_null_preserves_main_effects():
    from src.analysis.frozen_science_screen import interaction_blocks

    rng = np.random.default_rng(43)
    a = rng.normal(size=(24, 9))
    tr, ev = np.arange(18), np.arange(18, 24)
    perm = np.roll(tr, 1).tolist() + np.roll(ev, 1).tolist()
    blocks = interaction_blocks(a @ a.T, np.ones((24, 24)), tr, ev, np.array(perm))
    for key in ("interaction_true_aligned", "interaction_wrong_aligned", "interaction_wrong_mismatched"):
        for actual, expected in zip(blocks[key], blocks["additive"]):
            np.testing.assert_allclose(actual, expected)


def test_releasing_both_anchors_keeps_sum_and_drops_branch_derivatives():
    torch.manual_seed(17)
    x = torch.randn(2, 4, 15, 400)
    s = 0.4 * x

    def loss(delta, weights):
        pred = dict(
            S=(s + delta).flatten(0, 1),
            O=(x - s - delta).flatten(0, 1),
            sum=x.flatten(0, 1),
            commitment=x.new_zeros(()),
        )
        return objective(
            pred, x, s, torch.ones(3, 4, 1, 1), torch.ones(3, 4, 1, 1), torch.ones(4), None, weights, 0.0
        )

    noise = torch.randn_like(x)
    torch.testing.assert_close(loss(noise, [0, 0]), loss(0.0, [0, 0]), rtol=0, atol=0)
    assert loss(noise, [0, 1]) > loss(0.0, [0, 1])
    assert loss(noise, [1, 0]) > loss(0.0, [1, 0])


def test_hb_color_derivative_and_spectrum_shape(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/scripts"))
    from run_science_hb_replication import loss_fn

    x = torch.randn(3, 4, 15, 40)
    s = 0.3 * x
    # A different constant offset per color has zero time derivative.
    offsets = torch.tensor([1.0, -1.0]).repeat(20)
    pred = (s + offsets).flatten(0, 1).requires_grad_()
    out = dict(
        S=pred,
        O=(x - s).flatten(0, 1),
        sum=x.flatten(0, 1),
        commitment=x.new_zeros(()),
        spectrum=torch.zeros(12, 15, 2),
    )
    a = torch.ones(3, 4, 1, 1, 2)
    cost = loss_fn(out, x, s, a, a, None, None, None, 0.0)
    torch.testing.assert_close(cost, torch.tensor(0.5), rtol=1e-5, atol=1e-6)
    cost.backward()
    assert torch.isfinite(pred.grad).all()
