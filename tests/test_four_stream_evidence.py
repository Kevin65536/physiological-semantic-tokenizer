import numpy as np
import torch
from src.analysis.four_stream_block_ridge import band_histories, block_predictions, select_blocks
from src.foundation.conditional_slot_readers import LocalInteraction


def test_eta_zero_nesting_and_held_person_selection():
    g = np.random.default_rng(7)
    x, e, c, y = [g.normal(size=(30, 4)) for _ in range(4)]
    tr = np.arange(24)
    dev = np.arange(24, 30)
    p = next(block_predictions(x, e, c, y, tr, dev, [0], [0.1]))[1]
    np.testing.assert_array_equal(p, next(block_predictions(x, -e[::-1], c, y, tr, dev, [0], [0.1]))[1])
    params = (np.repeat(np.arange(10), 3), tr, dev, [0, 0.1, 1], [0.1, 10, 100], 3)
    k, loss = select_blocks(x, e, c, y, *params)
    y[dev] += 1000
    assert select_blocks(x, e, c, y, *params) == (k, loss)


def test_equal_budget_separable_residual_and_two_second_features():
    models = [LocalInteraction(8, 2, additive=a) for a in (True, False)]
    assert sum(p.numel() for p in models[0].parameters()) == sum(p.numel() for p in models[1].parameters())
    x = torch.randn(3, 4, 2, 15, 128)
    for model in models:
        assert not model(x, None).any()
        torch.nn.init.ones_(model.output.weight)

    def diff(model):
        y = x.clone()
        y[:, :, 0] += 1
        z = x.clone()
        z[:, :, 1] -= 2
        w = y.clone()
        w[:, :, 1] -= 2
        return model(w, None) - model(y, None) - model(z, None) + model(x, None)

    assert diff(models[0]).abs().max() < 1e-5 and diff(models[1]).abs().max() > 1e-3
    t = np.arange(400) / 200
    wave = np.broadcast_to(np.sin(2 * np.pi * 10 * t), (1, 4, 15, 400))
    bands = band_histories(wave, [[4, 8], [8, 13], [13, 30]])
    assert bands.argmax(-1).min() == bands.argmax(-1).max() == 1


def test_uncertain_word_votes_stay_in_denominator():
    from src.analysis.hb_word_organization import quadrants, word_census

    z = quadrants(np.array([[-1.0, 2.0], [-0.01, 1.0], [-0.2, -0.5]]), [0.1, 0.1])
    assert z.tolist() == [1, -1, 0]
    d = word_census(
        np.arange(6), np.zeros(6, int), np.repeat([0, 1], 3), np.full(6, 3), np.array([1, 1, -1, 1, -1, -1])
    ).iloc[0]
    assert d.named and not d.majority and not d.unanimous


def test_origin_conditioning_and_unsupported_rows():
    from src.analysis.joint_window_surrogates import conditional, joint_counts, rotations

    s = np.zeros((2, 15), int)
    o = s.copy()
    o[1, 1:] = 1
    counts = joint_counts(s, o)
    p = conditional(counts)
    assert np.isclose(p[0, 0], 14 / 15) and p[1, 1] == 1 and np.isnan(p[2]).all()
    assert np.allclose(conditional(np.stack([counts, counts]))[:, 0, 1], 1 / 15)
    assert rotations([0, 30]) == [1] and rotations([0, 29]) == []
