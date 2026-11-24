import numpy as np

from src.tokenizers.hb_native_id_readout import (
    id_design,
    smooth_basis,
    fit_head,
    apply_head,
    supported_pairs,
)


def test_current_and_context_columns_preserve_physical_patch_identity():
    ids = dict(S=np.tile(np.arange(15), (2, 1)), O=np.tile(np.arange(15) + 20, (2, 1)))
    design, _ = id_design(ids, True, np.array([20, 1 * 256 + 21]))
    assert design.shape == (30, 1160)
    assert design[0, 0] == 1 and design[0, 129 + 20] == 1
    assert design[0, 386 + 128] == 1  # fixed left edge, no wrap from the next crop
    assert design[14, 772 + 128] == 1
    assert design[0, 1158] == 1
    np.testing.assert_allclose(
        smooth_basis().T @ smooth_basis() / 20, np.eye(3), atol=1e-15
    )


def test_direct_ID_fit_recovers_real_residual_and_preserves_fixed_outputs():
    rng = np.random.default_rng(81)
    n = 160
    ids = dict(S=rng.integers(0, 4, (n, 15)), O=rng.integers(0, 5, (n, 15)))
    coef = np.zeros((n, 15, 3))
    coef[:, :, 0] = 0.05 * (ids["S"] - 1.5)
    x = np.zeros((n, 15, 40))
    o = rng.normal(size=x.shape) * 0.01
    prediction = dict(S=x - o, O=o, sum=x)
    fitted = fit_head(
        ids,
        prediction,
        coef,
        np.ones((n, 15)),
        np.ones((n, 15), bool),
        np.arange(n) % 8,
        "current_IDs",
        1.0,
        dict(minimum_pair_patches=10000, minimum_pair_subjects=4, maximum_pairs=256),
    )
    out = apply_head(ids, prediction, fitted)
    target = np.einsum("npk,tk->npt", coef, smooth_basis()).reshape(n, 300)
    recovered = out["sum"].reshape(n, 300, 2)[:, :, 0]
    assert np.mean((recovered - target) ** 2) < 0.15 * np.mean(target**2)
    np.testing.assert_array_equal(out["O"], o)
    np.testing.assert_array_equal(out["S"][..., 1::2], prediction["S"][..., 1::2])
    np.testing.assert_allclose(out["S"] + out["O"], out["sum"], atol=1e-12)


def test_pair_support_excludes_quiet_and_requires_cross_subject_coverage():
    ids = dict(S=np.zeros((20, 15), int), O=np.zeros((20, 15), int))
    ids["S"][-1] = 128
    ids["O"][-1] = 256
    pairs = supported_pairs(
        ids, np.ones((20, 15), bool), np.zeros(20), minimum=10, minimum_subjects=2
    )
    assert len(pairs) == 0
    pairs = supported_pairs(
        ids, np.ones((20, 15), bool), np.arange(20), minimum=10, minimum_subjects=2
    )
    np.testing.assert_array_equal(pairs, [0])


def test_unified_loader_accepts_the_real_native_head_and_pinned_quiet_source(tmp_path):
    import hashlib
    import torch
    from experiments.scripts.apply_joint_tokenizer import load_branch_pair
    from src.tokenizers.dual_component_tokenizer import GlobalLatentComponentTokenizer

    torch.manual_seed(13)
    setting = dict(
        input_dim=40,
        patches=15,
        width=8,
        latent_dim=4,
        heads=2,
        encoder_layers=1,
        decoder_layers=1,
        dropout=0,
        branch_codebook_sizes=dict(S=128, O=256),
    )
    base = GlobalLatentComponentTokenizer(setting, codebook_size=256).eval()
    for branch in base.branches.values():
        branch["quantizer"].initialized.fill_(True)
    ordinary = tmp_path / "ordinary.pt"
    torch.save(
        dict(
            metadata=dict(
                schema="contextual_latent_dictionary_v1",
                modality="hb",
                codebook_size=256,
                branch_codebook_sizes=dict(S=128, O=256),
                decoder_component_transfer_gain=[1.0, 1.0],
            ),
            model_config=setting,
            model_state=base.state_dict(),
        ),
        ordinary,
    )

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    current = tmp_path / "current.pt"
    torch.save(
        dict(
            metadata=dict(
                schema="silent_observation_tokenizer_v1",
                modality="hb",
                codebook_size=257,
                base_codebook_size=256,
                base_codebook_sizes=dict(S=128, O=256),
                branch_codebook_sizes=dict(S=129, O=257),
                output_dtype="float64",
                quiet_threshold_input_sd=0.01,
                source_file=ordinary.name,
                source_sha256=sha(ordinary),
            ),
            level=np.array([0.1, 0.2]),
        ),
        current,
    )
    file = tmp_path / "head.pt"
    torch.save(
        dict(
            metadata=dict(
                schema="hb_native_id_readout_v1",
                source_file=current.name,
                source_sha256=sha(current),
            ),
            fitted=dict(
                family="decoded_waveform",
                coefficient=np.zeros((37, 3)),
                pairs=np.array([], int),
            ),
        ),
        file,
    )
    model = load_branch_pair("hb_native_id_readout", file)
    ids = dict(
        S=np.tile([[0] * 15, [128] * 15], (1, 1)),
        O=np.tile([[0] * 15, [256] * 15], (1, 1)),
    )
    decoded = model.decode_ids(ids)
    original = model.source.decode_ids(ids)
    for branch in ["S", "O", "sum"]:
        np.testing.assert_array_equal(decoded[branch], original[branch])
