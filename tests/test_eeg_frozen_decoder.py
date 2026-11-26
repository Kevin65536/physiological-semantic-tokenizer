from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/scripts'))
import run_eeg_frozen_decoder as suite
from src.tokenizers.dual_component_tokenizer import GlobalLatentComponentTokenizer


@pytest.mark.parametrize('arm', ['continued', 'expanded'])
def test_refit_preserves_initial_waveform_ids_and_all_assignment_state(arm):
    suite.check()
    cfg = dict(input_dim=400, patches=15, width=16, latent_dim=8, heads=2, encoder_layers=2,
               decoder_layers=2, waveform_io=True, channels=1, sample_strides=[5,4,5],
               local_width=4, single_id_spectrum=True, geometry_each_forward=False)
    model = GlobalLatentComponentTokenizer(cfg, codebook_size=8).eval()
    for branch in model.branches.values():
        branch['quantizer'].initialized.fill_(True)
    x = torch.randn(2, 15, 400)
    with torch.inference_mode():
        before = model(x)
    model.prepare_decoder_refit(suite.PLAN['arms'][arm])
    tensors = {k:v.clone() for k,v in model.state_dict().items()}
    with torch.no_grad():
        replay = model.forward_frozen_ids(x)
        for k in ('S', 'O', 'sum'):
            torch.testing.assert_close(replay[k], before[k], rtol=0, atol=0)
    batch = dict(x=x, s=x*.4, o=x*.6, mask=torch.ones_like(x, dtype=torch.bool))
    for k in 'xso':
        batch['v'+k] = torch.ones_like(x)
        batch['dv'+k] = torch.ones(len(x), 1)
    suite.update(model, batch, dict(sum_weight=.5, derivative_weight=.1), suite.optimizer_for(model))
    with torch.no_grad():
        after = model.forward_frozen_ids(x)
    trainable = {k for k,p in model.named_parameters() if p.requires_grad}
    for k,v in model.state_dict().items():
        if k not in trainable:
            torch.testing.assert_close(tensors[k], v, atol=0, rtol=0)
    for k in 'SO':
        assert len(model.branches[k]['decoder'].layers) == suite.PLAN['arms'][arm]
        assert torch.equal(after['ids'][k], before['ids'][k])
        assert not torch.equal(after[k], before[k])
    with pytest.raises(ValueError):
        model.decode_ids({k: after['ids'][k].float() for k in 'SO'})
