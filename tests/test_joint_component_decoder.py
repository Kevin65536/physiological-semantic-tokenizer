"""Frozen IDs, matched capacity, foreign-ID ablations, and branch-only replay."""
import numpy as np
import pytest
import torch

from src.tokenizers.dual_component_tokenizer import GlobalLatentComponentTokenizer
from src.tokenizers.joint_component_decoder import JointComponentDecoder, JointComponentTokenizer, load_model


def models():
    torch.set_num_threads(1)
    setting = dict(input_dim=40,channels=2,patches=15,width=8,latent_dim=4,heads=2,
        encoder_layers=1,decoder_layers=1,waveform_io=True,dropout=0.,sample_strides=[5],
        local_width=4,hb_local_shape=True,context_ratio=.25,context_norm_epsilon=1e-8)
    torch.manual_seed(3)
    source = GlobalLatentComponentTokenizer(setting,codebook_size=8,hard=True).eval()
    for branch in source.branches.values():
        branch['quantizer'].initialized.fill_(True)
    paired = {}
    for routing in ('branch_only','joint'):
        paired[routing] = JointComponentDecoder(setting,routing=routing,codebook_size=8).eval()
        paired[routing].load_state_dict(source.state_dict(),strict=False)
    return setting,source,paired


def test_source_initial_predictions_and_encoder_ids_remain_exact():
    _,source,paired = models()
    values = torch.randn(2,15,40)
    with torch.no_grad():
        original = source(values)
        for model in paired.values():
            ids = model.encode_ids(values)
            prediction = model.decode_ids(ids)
            for branch in ('S','O'):
                torch.testing.assert_close(ids[branch],original['ids'][branch],rtol=0,atol=0)
            for branch in ('S','O','sum'):
                torch.testing.assert_close(prediction[branch],original[branch],rtol=0,atol=0)
    assert [sum(p.numel() for p in model.parameters() if p.requires_grad) for model in paired.values()][0] == \
           [sum(p.numel() for p in model.parameters() if p.requires_grad) for model in paired.values()][1]
    model = paired['joint'].train()
    fixed = {key:value.clone() for key,value in model.state_dict().items()
             if '.quantizer.' in key or '.encoder.' in key or '.input.' in key or '.to_latent.' in key}
    loss = model.decode_ids(original['ids'])['sum'].square().mean()
    loss.backward()
    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001).step()
    for key,value in fixed.items():
        torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
    with torch.no_grad():
        for branch,value in model.encode_ids(values).items():
            torch.testing.assert_close(value,original['ids'][branch],rtol=0,atol=0)


def test_foreign_ids_only_affect_joint_decoder_and_saved_replay(tmp_path):
    setting,_,paired = models()
    ids = dict(S=torch.zeros(2,15,dtype=torch.long),O=torch.ones(2,15,dtype=torch.long))
    altered = dict(ids,O=torch.full_like(ids['O'],2))
    for model in paired.values():
        with torch.no_grad():
            model.branches['S']['cross_from_latent'].weight.fill_(.3)
    control = paired['branch_only']
    with torch.no_grad():
        torch.testing.assert_close(control.decode_ids(ids)['S'],control.decode_ids(altered)['S'],rtol=0,atol=0)
        for branch in ('S','O'):
            torch.testing.assert_close(control.decode_branch_ids(branch,ids[branch]),control.decode_ids(ids)[branch],rtol=0,atol=0)
        assert not torch.equal(paired['joint'].decode_ids(ids)['S'],paired['joint'].decode_ids(altered)['S'])
    metadata = dict(schema='joint_component_decoder_v1',routing='joint',codebook_size=8)
    wrapper = JointComponentTokenizer(paired['joint'],metadata)
    path = tmp_path/'model.pt'
    torch.save(dict(metadata=metadata,model_config=setting,model_state=paired['joint'].state_dict()),path)
    codes = {key:value.numpy() for key,value in ids.items()}
    np.savez(tmp_path/'ids.npz',**codes)
    with np.load(tmp_path/'ids.npz') as stored:
        replay = load_model(path).decode_ids(dict(stored))
    for key,value in wrapper.decode_ids(codes).items():
        np.testing.assert_array_equal(value,replay[key])
    with pytest.raises(ValueError,match='integer'):
        wrapper.decode_ids({key:value.astype(float) for key,value in codes.items()})
    with pytest.raises(ValueError,match='declared'):
        wrapper.decode_ids(dict(codes,unencoded_scale=np.ones(2)))


def test_reviewed_config_preserves_keys_through_resolved_json():
    import json
    from pathlib import Path
    import yaml
    config = yaml.safe_load((Path(__file__).resolve().parents[1]/'experiments/configs/physiology_semantic_tokenizer/dual_component_joint_cross_decoder_v2.yaml').read_text())
    assert json.loads(json.dumps(config)) == config
    assert isinstance(config['hypothesis']['null_hypothesis'],str)
    assert config['source']['step'] == 2000
    assert config['data']['allowed_splits'] == ['train','validation']
