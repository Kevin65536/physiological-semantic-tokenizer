import numpy as np
import pytest
import torch
from types import SimpleNamespace

from src.tokenizers.hb_parent_distillation import (
    fit_tables,
    parent_probabilities,
    preserve_O_mix_HbO,
    validate_ids,
    decode_parent,
)
from src.tokenizers.dual_component_tokenizer import GlobalLatentComponentTokenizer


def test_single_id_table_recovers_permuted_parent_and_context_generalizes():
    rng = np.random.default_rng(19)
    s = rng.integers(0, 4, (300, 15))
    o = rng.integers(0, 2, (300, 15))
    target = (s + 11).astype(int)
    ids = dict(S=s, O=o)
    fitted = fit_tables(ids, target, np.ones_like(s), 16)
    new_s = rng.integers(0, 4, (80, 15))
    new = dict(S=new_s, O=rng.integers(0, 2, (80, 15)))
    single = parent_probabilities(new, fitted, 0)
    context = parent_probabilities(new, fitted, 1)
    np.testing.assert_allclose(context.sum(-1), 1)
    assert np.mean(context.argmax(-1) == new_s + 11) == 1
    assert np.mean(single.argmax(-1) == new_s + 11) < 0.4


def test_sum_mixture_preserves_O_HbR_quiet_and_closure():
    rng = np.random.default_rng(39)
    o = rng.normal(size=(4, 15, 40))
    total = rng.normal(size=o.shape)
    s = total - o
    total[0] = 0
    s[0] = -o[0]
    current = dict(S=s, O=o, sum=total)
    parent = dict(sum=rng.normal(size=o.shape))
    quiet = np.array([True, False, False, False])
    output = preserve_O_mix_HbO(current, parent, quiet, 0.5)
    np.testing.assert_array_equal(output["O"], o)
    for b in current:
        np.testing.assert_array_equal(output[b][0], current[b][0])
        np.testing.assert_array_equal(output[b][..., 1::2], current[b][..., 1::2])
    np.testing.assert_allclose(output["S"] + output["O"], output["sum"], atol=1e-12)


def test_reserved_pair_and_integer_input_fail_closed():
    ids = dict(S=np.zeros((2, 15), int), O=np.zeros((2, 15), int))
    ids["S"][0] = 128
    with pytest.raises(ValueError, match="complete paired quiet"):
        validate_ids(ids)
    ids["O"][0] = 256
    _, quiet = validate_ids(ids)
    np.testing.assert_array_equal(quiet, [True, False])
    with pytest.raises(ValueError, match="integer"):
        validate_ids({k: v.astype(float) for k, v in ids.items()})


def test_predicted_one_hot_parent_uses_the_static_integer_decoder_exactly():
    torch.manual_seed(15)
    model = GlobalLatentComponentTokenizer(
        dict(
            input_dim=40,
            patches=15,
            width=8,
            latent_dim=4,
            heads=2,
            encoder_layers=1,
            decoder_layers=1,
            dropout=0,
        ),
        codebook_size=128,
    ).eval()
    for branch in model.branches.values():
        branch["quantizer"].initialized.fill_(True)
    ids = dict(S=np.zeros((2, 15), int), O=np.ones((2, 15), int))
    labels = np.tile(np.arange(15), (2, 1))
    p = np.eye(128)[labels]
    hard = decode_parent(SimpleNamespace(model=model), ids, p, "hard")
    expected = decode_parent(SimpleNamespace(model=model), ids, p, "expected_embedding")
    with torch.inference_mode():
        direct = model.decode_ids(
            dict(S=torch.as_tensor(ids["S"]), O=torch.as_tensor(labels))
        )
    for branch in ["S", "O", "sum"]:
        np.testing.assert_allclose(hard[branch], direct[branch].numpy(), atol=1e-7)
        np.testing.assert_allclose(expected[branch], hard[branch], atol=1e-7)
