"""Synthetic-only tests for independent, local, dual-component vocabularies."""

from copy import deepcopy
import inspect

import pytest
import torch
from torch import nn

from src.tokenizers.dual_component_tokenizer import DualComponentTokenizer, VARIANTS


@pytest.fixture
def cfg():
    torch.set_num_threads(1)
    torch.manual_seed(13)
    return dict(input_dims=dict(eeg=12, hb=8), target_dims=dict(eeg=4, hb=8),
                width=16, latent_dim=3, codebook_size=4,
                ema_decay=0.9, commitment_cost=0.25, kmeans_init=False)


def observations(cfg, patches=5):
    return {m: torch.randn(2, patches, d) for m, d in cfg['input_dims'].items()}


def targets(cfg, patches=5):
    return {m: torch.randn(2, patches, d) for m, d in cfg['target_dims'].items()}


@pytest.mark.parametrize('variant', VARIANTS)
def test_shapes_gradient_and_full_observation_space(cfg, variant):
    model = DualComponentTokenizer(cfg, variant)
    inputs, observed, physical = observations(cfg), targets(cfg), targets(cfg)
    if variant == 'component_compression':
        output = model.forward_components(physical, {m: observed[m] - physical[m] for m in observed})
    else:
        output = model(inputs)
    loss = model.objective(output, observed, physical)
    assert torch.isfinite(loss['total'])
    loss['total'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    for modality, dimension in cfg['target_dims'].items():
        result = output[modality]
        assert result['S'].shape == result['O'].shape == (2, 5, dimension)
        assert torch.equal(result['sum'], result['S'] + result['O'])
        if variant == 'continuous_student':
            assert result['ids_S'] is result['ids_O'] is None
        else:
            assert result['ids_S'].dtype == result['ids_O'].dtype == torch.long
            assert result['ids_S'].shape == result['ids_O'].shape == (2, 5)


def test_native_eeg_input_and_proxy_target_can_have_distinct_dimensions(cfg):
    cfg.update(input_dims=dict(eeg=400 * 6, hb=20 * 2), target_dims=dict(eeg=20, hb=20 * 2))
    model = DualComponentTokenizer(cfg)
    inputs = observations(cfg, patches=10)
    output = model(inputs)
    assert output['eeg']['S'].shape == (2, 10, 20)
    assert output['hb']['S'].shape == (2, 10, 40)
    assert model.branches['eeg']['S'].encoder[0].in_features == 2 * 2400
    with pytest.raises(ValueError, match='target-space'):
        model.objective(output, inputs, inputs)


def test_independent_decoding_and_no_cross_branch_gradient(cfg):
    model = DualComponentTokenizer(cfg).eval()
    output = model(observations(cfg))
    ids = {m: {'S': output[m]['ids_S'], 'O': output[m]['ids_O']} for m in output}
    decoded = model.decode_ids(ids)
    modified = deepcopy(ids)
    for modality in modified:
        modified[modality]['O'] = (modified[modality]['O'] + 1) % cfg['codebook_size']
    changed = model.decode_ids(modified)
    for modality in decoded:
        assert torch.allclose(decoded[modality]['S'], output[modality]['S'])
        assert torch.allclose(decoded[modality]['O'], output[modality]['O'])
        assert torch.equal(decoded[modality]['S'], changed[modality]['S'])
        assert torch.equal(decoded[modality]['sum'], decoded[modality]['S'] + decoded[modality]['O'])
        assert (model.branches[modality]['S'].quantizer.codebook.data_ptr()
                != model.branches[modality]['O'].quantizer.codebook.data_ptr())
    sum(value['S'].square().sum() for value in output.values()).backward()
    for branches in model.branches.values():
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in branches['S'].parameters())
        assert all(p.grad is None for p in branches['O'].parameters())


def test_hidden_raw_values_and_other_patches_cannot_change_visible_ids(cfg):
    model = DualComponentTokenizer(cfg).eval()
    inputs = observations(cfg)
    masks = {m: torch.ones_like(x, dtype=torch.bool) for m, x in inputs.items()}
    for mask in masks.values():
        mask[:, 2] = False
    poisoned = {m: x.clone() for m, x in inputs.items()}
    for modality in poisoned:
        poisoned[modality][:, 2] = float('nan')
    original = model(inputs, masks)
    altered = model(poisoned, masks)
    for modality in original:
        for component in ('S', 'O'):
            key = 'ids_' + component
            assert torch.equal(original[modality][key], altered[modality][key])
            assert (altered[modality][key][:, 2] == -1).all()
    # Even a visible patch cannot influence its neighbors: no convolution,
    # attention, across-patch statistics, or shared normalization is present.
    poisoned = {m: x.clone() for m, x in inputs.items()}
    poisoned['eeg'][:, 2] = 1e8
    all_visible = model(poisoned)
    ordinary = model(inputs)
    visible = torch.tensor([True, True, False, True, True])
    for component in ('S', 'O'):
        assert torch.equal(ordinary['eeg']['ids_' + component][:, visible],
                           all_visible['eeg']['ids_' + component][:, visible])
    assert not any(isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.Conv1d))
                   for module in model.modules())


def test_modality_locality_and_no_teacher_input_in_student_api(cfg):
    model = DualComponentTokenizer(cfg).eval()
    inputs = observations(cfg)
    a = model(inputs)
    b = model({**inputs, 'hb': inputs['hb'] * 500})
    for key in ('ids_S', 'ids_O', 'S', 'O'):
        assert torch.equal(a['eeg'][key], b['eeg'][key])
    assert set(inspect.signature(model.forward).parameters) == {'inputs', 'input_masks'}
    with pytest.raises(ValueError, match='only in component_compression'):
        model.forward_components(targets(cfg), targets(cfg))


def test_component_compression_reads_each_component_separately(cfg):
    model = DualComponentTokenizer(cfg, 'component_compression').eval()
    physical, remaining = targets(cfg), targets(cfg)
    a = model.forward_components(physical, remaining)
    b = model.forward_components(physical, {m: x * 100 for m, x in remaining.items()})
    for modality in a:
        assert torch.equal(a[modality]['S'], b[modality]['S'])
        assert torch.equal(a[modality]['ids_S'], b[modality]['ids_S'])
    with pytest.raises(ValueError, match='forward_components'):
        model(observations(cfg))


def manual_output(S, O):
    return dict(S=S, O=O, sum=S + O,
                valid_S=torch.ones(S.shape[:-1], dtype=torch.bool),
                valid_O=torch.ones(O.shape[:-1], dtype=torch.bool),
                commitment=S.reshape(-1)[:0].sum())


def test_masked_training_scale_reduction_is_equal_modality_mean(cfg):
    cfg.update(input_dims=dict(eeg=2, hb=1), target_dims=dict(eeg=2, hb=1))
    model = DualComponentTokenizer(cfg)
    output = {
        'eeg': manual_output(torch.tensor([[[2., 1e10]]], requires_grad=True),
                             torch.tensor([[[3., -1e10]]], requires_grad=True)),
        'hb': manual_output(torch.tensor([[[4.]]], requires_grad=True),
                            torch.tensor([[[2.]]], requires_grad=True)),
    }
    observed = dict(eeg=torch.tensor([[[5., float('nan')]]]), hb=torch.tensor([[[4.]]]))
    physical = dict(eeg=torch.tensor([[[1., float('nan')]]], requires_grad=True),
                    hb=torch.tensor([[[2.]]], requires_grad=True))
    result = model.objective(output, observed, physical,
                             target_scales=dict(eeg=torch.tensor([2., 7.]), hb=1.))
    assert result['S'].item() == pytest.approx(2.125)
    assert result['O'].item() == pytest.approx(.125)
    assert result['sum'].item() == pytest.approx(2.)
    assert result['total'].item() == pytest.approx(2.25)
    assert result['per_modality']['eeg']['valid_elements'].item() == 1
    result['total'].backward()
    assert physical['eeg'].grad is physical['hb'].grad is None
    assert output['eeg']['S'].grad[0, 0, 1] == 0
    assert output['eeg']['O'].grad[0, 0, 1] == 0


def test_O_target_stays_teacher_remaining_and_sum_is_not_main_loss(cfg):
    model = DualComponentTokenizer(cfg)
    observed, physical = targets(cfg), targets(cfg)
    offset = .8
    exact_sum = {m: manual_output(physical[m] + offset, observed[m] - physical[m] - offset)
                 for m in observed}
    loss = model.objective(exact_sum, observed, physical, beta=0.)
    assert loss['sum'].item() == pytest.approx(0., abs=1e-12)
    assert loss['S'].item() == pytest.approx(offset**2)
    assert loss['O'].item() == pytest.approx(offset**2)
    assert loss['total'].item() == pytest.approx(2 * offset**2)


def test_empty_masks_do_not_initialize_or_age_vocabulary_and_have_zero_loss(cfg):
    cfg['kmeans_init'] = True
    model = DualComponentTokenizer(cfg)
    inputs = observations(cfg)
    masks = {m: torch.zeros_like(x, dtype=torch.bool) for m, x in inputs.items()}
    output = model({m: torch.full_like(x, float('nan')) for m, x in inputs.items()}, masks)
    loss = model.objective(output, targets(cfg), targets(cfg))
    assert loss['total'].item() == 0.
    loss['total'].backward()
    for branches in model.branches.values():
        for branch in branches.values():
            assert not branch.quantizer.initialized.item()
            assert branch.quantizer.update_count.item() == 0
    for result in output.values():
        assert (result['ids_S'] == -1).all() and (result['ids_O'] == -1).all()


def test_eval_never_initializes_or_changes_ema_and_hard_ids_remain_decodable(cfg):
    cfg['kmeans_init'] = True
    model = DualComponentTokenizer(cfg)
    inputs = observations(cfg)
    model.eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with pytest.raises(RuntimeError, match='uninitialized'):
        model(inputs)
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    model.train()
    model(inputs)
    model.eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    output = model(inputs)
    decoded = model.decode_ids({m: {c: output[m]['ids_' + c] for c in ('S', 'O')} for m in output})
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    for modality in output:
        for component in ('S', 'O'):
            assert torch.allclose(output[modality][component], decoded[modality][component], atol=1e-6)


class SquareDecoder(nn.Module):
    def forward(self, latent):
        return latent.square()


def test_probability_waveform_expectation_decodes_before_averaging():
    cfg = dict(input_dims=dict(eeg=2), target_dims=dict(eeg=1), latent_dim=1,
               width=4, codebook_size=2, kmeans_init=False)
    model = DualComponentTokenizer(cfg).eval()
    for branch in model.branches['eeg'].values():
        branch.decoder = SquareDecoder()
        branch.quantizer.codebook.copy_(torch.tensor([[-1.], [1.]]))
    probabilities = dict(eeg=dict(S=torch.tensor([[.5, .5]]), O=torch.tensor([[.5, .5]])))
    expected = model.expected_waveforms(probabilities)
    assert torch.equal(expected['eeg']['S'], torch.ones(1, 1))
    assert torch.equal(expected['eeg']['sum'], torch.full((1, 1), 2.))
    branch = model.branches['eeg']['S']
    incorrectly_decode_mean = branch.decoder(probabilities['eeg']['S'] @ branch.quantizer.codebook)
    assert not torch.equal(expected['eeg']['S'], incorrectly_decode_mean)


def test_ordinary_baseline_matches_bitrate_and_capacity_but_only_fits_sum(cfg):
    main = DualComponentTokenizer(cfg)
    ordinary = DualComponentTokenizer(cfg, 'ordinary_hard')
    assert main.nominal_bits_per_patch() == ordinary.nominal_bits_per_patch() == dict(eeg=4., hb=4.)
    assert sum(p.numel() for p in main.parameters()) == sum(p.numel() for p in ordinary.parameters())
    observed, physical = targets(cfg), targets(cfg)
    output = {m: manual_output(physical[m] + 5, observed[m] - physical[m] - 5) for m in observed}
    loss = ordinary.objective(output, observed)
    assert loss['total'].item() == pytest.approx(0., abs=1e-11)
    assert loss['S'].item() == loss['O'].item() == 0.
    assert main.objective(output, observed, physical)['total'].item() == pytest.approx(50.)


def test_invalid_labels_and_explicit_masks_do_not_count(cfg):
    model = DualComponentTokenizer(cfg, 'continuous_student')
    output = model(observations(cfg))
    observed, physical = targets(cfg), targets(cfg)
    masks = {m: torch.zeros_like(x, dtype=torch.bool) for m, x in observed.items()}
    observed = {m: torch.full_like(x, float('nan')) for m, x in observed.items()}
    loss = model.objective(output, observed, physical, masks)
    assert loss['total'].item() == 0
    assert all(value['valid_elements'] == 0 for value in loss['per_modality'].values())
    loss['total'].backward()


def test_strict_id_probability_and_mask_validation(cfg):
    model = DualComponentTokenizer(cfg).eval()
    output = model(observations(cfg))
    ids = {m: {c: output[m]['ids_' + c] for c in ('S', 'O')} for m in output}
    with pytest.raises(ValueError, match='only S and O'):
        model.decode_ids({**ids, 'eeg': {**ids['eeg'], 'amplitude': torch.ones(2, 5)}})
    with pytest.raises(ValueError, match='integer'):
        model.decode_ids({**ids, 'eeg': {**ids['eeg'], 'S': ids['eeg']['S'].float()}})
    with pytest.raises(ValueError, match='inside'):
        model.decode_ids({**ids, 'eeg': {**ids['eeg'], 'S': ids['eeg']['S'] - 100}})
    inputs = observations(cfg)
    with pytest.raises(ValueError, match='boolean'):
        model(inputs, {m: torch.ones_like(x) for m, x in inputs.items()})
    with pytest.raises(ValueError, match='mask must match'):
        model(inputs, {m: torch.ones(2, 5, 1, dtype=torch.bool) for m in inputs})
    inputs['eeg'][0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite visible'):
        model(inputs)
    probabilities = {m: {c: torch.ones(2, cfg['codebook_size']) for c in ('S', 'O')} for m in output}
    with pytest.raises(ValueError, match='normalized'):
        model.expected_waveforms(probabilities)


def test_training_scale_and_target_support_validation(cfg):
    model = DualComponentTokenizer(cfg)
    output = model(observations(cfg))
    observed, physical = targets(cfg), targets(cfg)
    with pytest.raises(ValueError, match='positive and finite'):
        model.objective(output, observed, physical, target_scales=dict(eeg=0., hb=1.))
    with pytest.raises(ValueError, match='one value per target'):
        model.objective(output, observed, physical, target_scales=dict(eeg=torch.ones(2, 4), hb=1.))
    with pytest.raises(ValueError, match='same modalities'):
        DualComponentTokenizer({**cfg, 'target_dims': dict(eeg=4)})


def test_per_record_frozen_training_scales_broadcast_over_patch_support(cfg):
    model = DualComponentTokenizer(cfg)
    observed, physical = targets(cfg), targets(cfg)
    output = {m: manual_output(physical[m] + 2., observed[m] - physical[m] + 2.) for m in observed}
    scales = {m: torch.tensor([1., 2.]).reshape(2, 1, 1).expand(2, 1, d)
              for m, d in cfg['target_dims'].items()}
    loss = model.objective(output, observed, physical, target_scales=scales, beta=0.)
    assert loss['S'].item() == pytest.approx(2.5)
    assert loss['O'].item() == pytest.approx(2.5)
    assert loss['total'].item() == pytest.approx(5.)
