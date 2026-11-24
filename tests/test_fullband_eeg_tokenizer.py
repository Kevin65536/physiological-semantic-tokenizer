import numpy as np
import pytest
import torch

from src.tokenizers.fullband_eeg_tokenizer import (
    FullbandEEGTokenizer, FullbandEEGInference, fullband_loss, load_model,
)


def model():
    torch.set_num_threads(1)
    setting = dict(codebook_size=8, latent_dim=16, hidden_width=16, ema_decay=.99, commitment_cost=.05)
    network = FullbandEEGTokenizer(setting)
    network.transfer[0].fill_(.4)
    network.transfer[1].fill_(.6)
    return network.eval()


def test_saved_integer_replay_uses_only_frozen_codebook_and_ids(tmp_path):
    network = model()
    wrapped = FullbandEEGInference(network, dict(schema='fullband_eeg_tokenizer_v1'))
    x = np.random.default_rng(13).normal(size=(2, 15, 400)).astype(np.float32)
    result = wrapped(x)
    torch.save(dict(metadata=wrapped.metadata, model_config=network.setting,
                    model_state=network.state_dict()), tmp_path/'model.pt')
    np.savez(tmp_path/'ids.npz', **result['ids'])
    with np.load(tmp_path/'ids.npz') as ids:
        replay = load_model(tmp_path/'model.pt').decode_ids(dict(ids))
    for branch in ('S', 'O', 'sum'):
        np.testing.assert_array_equal(result[branch], replay[branch])
    with pytest.raises(ValueError, match='integer'):
        wrapped.decode_ids({b: values.astype(float) for b, values in result['ids'].items()})
    with pytest.raises(ValueError, match='only S/O'):
        wrapped.decode_ids(dict(result['ids'], latent=np.ones(2)))


def test_full_sample_decoder_retains_high_frequency_and_train_loss_handles_masked_nan():
    network = model()
    wave = torch.sin(2*torch.pi*30*torch.arange(400)/200)
    with torch.no_grad():
        for b in ('S', 'O'):
            network.branches[b]['output'].weight.zero_()
            network.branches[b]['output'].bias.copy_(wave)
    latent = {b: torch.zeros(2, 15, 16) for b in ('S', 'O')}
    out = network.decode_latents(latent)
    torch.testing.assert_close(out['S'][0, 0], wave)
    mask = torch.ones(2, 15, 400, dtype=torch.bool)
    mask[:, -1, -20:] = False
    target = out['S'].detach().clone()
    target[~mask] = float('nan')
    batch = dict(x=target*2, s=target, o=target, mask=mask)
    for key in ('x', 's', 'o'):
        batch['v'+key] = torch.ones_like(target)
    out['commitment'] = torch.zeros(())
    loss, _ = fullband_loss(network, out, batch, dict(sum=.5, band=.2, envelope=.2))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in network.parameters())
