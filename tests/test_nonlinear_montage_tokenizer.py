"""Full-array messaging, observation masking and standalone integer synthesis."""
import numpy as np
import pytest
import torch

from src.tokenizers.nonlinear_montage_tokenizer import NonlinearMontageTokenizer, load_model


def model(cross=True):
    torch.set_num_threads(1)
    torch.manual_seed(81)
    config = dict(vocabulary=8, latent_dim=8, hidden_width=16, position_dim=3, pool_heads=2,
                  ema_decay=.99, commitment_cost=.25, cross_position=cross)
    metadata = dict(schema='nonlinear_montage_tokenizer_v1', positions={'fixture': ['A', 'B', 'C']})
    return NonlinearMontageTokenizer(config, metadata)


def test_integer_reload_uses_only_ids_and_fixed_layout(tmp_path):
    net = model().eval()
    net.local.transfer.fill_(.5)
    x = torch.randn(1, 3, 15, 400)
    observed = torch.tensor([[True, False, True]])
    with torch.inference_mode():
        result = net(x, 'fixture', observed, quantize=True)
        for b in 'SO':assert torch.equal(result['ids'][b][:, 1], torch.zeros((1, 15), dtype=torch.long))
    path = tmp_path/'model.pt'
    torch.save(dict(model_config=net.config, metadata=net.metadata, model_state=net.state_dict()), path)
    loaded = load_model(path)
    loaded.model.encode_continuous = lambda *_: (_ for _ in ()).throw(AssertionError('X used in decoder'))
    replay = loaded.decode_ids({k:v.numpy() for k,v in result['ids'].items()}, 'fixture')
    for b in ('S','O','sum'):np.testing.assert_array_equal(replay[b],result[b].numpy())
    with pytest.raises(ValueError,match='integer'):
        loaded.decode_ids({k:v.numpy().astype(float) for k,v in result['ids'].items()},'fixture')


def test_cross_position_messages_reach_other_sites_and_masked_sites_are_excluded():
    local, field = model(False), model(True)
    field.load_state_dict(local.state_dict())
    for net in (local,field):
        net.local.transfer.fill_(.5)
        for branch in 'SO':
            torch.nn.init.normal_(net.encoder_message[branch].update[-1].weight,std=.05)
    # Use identical trained-like messaging weights in the matched arms.
    field.load_state_dict(local.state_dict())
    x=torch.randn(1,3,15,400);seen=torch.ones((1,3),dtype=torch.bool)
    changed=x.clone();changed[:,1]*=4
    with torch.inference_mode():
        a=local.encode_continuous(x,'fixture',seen)['S'];b=local.encode_continuous(changed,'fixture',seen)['S']
        np.testing.assert_array_equal(a[:,0],b[:,0])
        a=field.encode_continuous(x,'fixture',seen)['S'];b=field.encode_continuous(changed,'fixture',seen)['S']
        assert not torch.equal(a[:,0],b[:,0])
        seen[:,1]=False
        a=field.encode_continuous(x,'fixture',seen)['S'];b=field.encode_continuous(changed,'fixture',seen)['S']
        np.testing.assert_array_equal(a[:,0],b[:,0])
    assert sum(p.numel() for p in local.parameters()) == sum(p.numel() for p in field.parameters())


def test_continuous_and_hard_training_losses_have_finite_masked_gradients():
    from experiments.scripts.run_nonlinear_montage_tokenizer import loss_for
    net = model().train()
    net.local.transfer.fill_(.5)
    shape = (1, 3, 15, 400)
    batch = {k: torch.randn(shape) for k in ('x', 's', 'o')}
    batch.update({k: torch.ones(shape) for k in ('vx', 'vs', 'vo')})
    batch['mask'] = torch.ones(shape, dtype=torch.bool)
    batch['mask'][:, 1, :, 200:] = False
    batch['observed'] = torch.tensor([[True, True, False]])
    batch.update({k: torch.randn(1, 3, 15, 3) for k in ('attr_s', 'attr_o')})
    cfg = {'training': {'loss_weights': dict(sum=.5, band=.1, envelope=.1, attribute=.05)}}
    for hard in (False, True):
        prediction = net(batch['x'], 'fixture', batch['observed'], quantize=hard,
                         valid_mask=batch['mask'].any(-1) & batch['observed'][..., None])
        loss, _ = loss_for(net, prediction, batch, cfg)
        assert torch.isfinite(loss)
        net.zero_grad(); loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_registered_training_schedule_excludes_held_subjects_and_is_reproducible():
    from pathlib import Path
    import yaml
    from experiments.scripts.run_nonlinear_montage_tokenizer import schedule, validate, rank
    cfg = yaml.safe_load((Path(__file__).parents[1]/'experiments/configs/physiology_semantic_tokenizer/dual_component_nonlinear_montage_v1.yaml').read_text())
    validate(cfg)
    assert len(rank(cfg)) == 64
    panel = {'cases': [dict(dataset=d, subject=s, held=s == 'held', crop={'row': i})
                       for i, (d, s) in enumerate((('A', 'fit'), ('A', 'held'), ('B', 'fit'), ('B', 'held')))]}
    order = schedule(panel, cfg)
    assert order == schedule(panel, cfg)
    assert all(len(take) == cfg['training']['batch_crops'] and all(not panel['cases'][i]['held'] and panel['cases'][i]['dataset'] == dataset for i in take)
               for dataset, take in order)
    cfg['data']['evaluation_splits'] = ['validation']
    with pytest.raises(ValueError, match='training-only'):
        validate(cfg)
