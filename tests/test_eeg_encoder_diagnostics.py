from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/scripts'))
import run_eeg_encoder_diagnostics as suite
from run_dual_component_tokenizer import _base_forward, _base_training_model


def model(arch):
    torch.manual_seed(17)
    cfg = suite.settings(arch, synthetic=True)
    return _base_training_model(cfg, suite.task_for(arch,'continuous')).eval()


@pytest.mark.parametrize('arch', ['privileged','wide','deep'])
def test_only_requested_input_or_encoder_changes(arch):
    base, candidate = model('observed'), model(arch)
    expected = suite.ARCHS[arch]
    for branch in 'SO':
        b,c = base.branches[branch],candidate.branches[branch]
        assert len(c['encoder'].layers) == expected['encoder_layers']
        assert c['input'].project.in_features == 4*expected['input_width']
        assert len(c['decoder'].layers)==2 and c['output'].local_width==32
        for k,v in b.state_dict().items():
            if arch=='wide' and k.startswith('input.'):
                continue
            torch.testing.assert_close(v,c.state_dict()[k],rtol=0,atol=0)
    x=torch.randn(2,15,400)
    batch=dict(x=x, s=x*.3, o=x*.7)
    with torch.inference_mode():
        out=_base_forward(candidate,batch)
        changed=_base_forward(candidate,dict(batch,x=x+2))
        assert torch.equal(out['sum'],changed['sum']) == (arch=='privileged')
        changed=_base_forward(candidate,dict(batch,s=batch['s']+2))
        assert torch.equal(out['S'],changed['S']) == (arch!='privileged')
        assert torch.equal(out['O'],changed['O'])
    with pytest.raises(ValueError):
        suite.task_for('privileged','hard')


def test_development_monitor_uses_validation_and_survives_warmup(tmp_path):
    from test_dual_component_base_g1 import prepared
    from run_dual_component_tokenizer import train_base_tokenizer
    from src.data.dual_component_tokenizer import base_tokenizer_examples, base_tokenizer_training_data
    cfg, base = prepared(tmp_path)
    cfg['architectures']['waveform']['target_input'] = True
    cfg['evaluation']['steps'] = []
    rows = base_tokenizer_training_data(cfg,tmp_path)[0]['rows']
    probe = base_tokenizer_examples(rows,'eeg','validation')[:2]
    assert len(probe)==2
    for candidate in ('warmup','continuous'):
        task=dict(modality='eeg',architecture='waveform',candidate=candidate)
        result=train_base_tokenizer(cfg,tmp_path,base,task,'cpu',development_probe=probe)
        assert result['signature']['development_probe']==suite.identity(probe)
        assert all(len(row['development_nmse_xso'])==3 for row in result['history'])
    wrong=base_tokenizer_examples(rows,'eeg','train')[:1]
    with pytest.raises(ValueError,match='admitted development'):
        train_base_tokenizer(cfg,tmp_path,base,task,'cpu',development_probe=wrong)


def test_privileged_terminal_scoring_uses_teacher_and_frozen_example_identity(tmp_path):
    import numpy as np
    from test_dual_component_base_g1 import prepared
    from src.data.dual_component_tokenizer import base_tokenizer_examples, base_tokenizer_training_data, base_tokenizer_batch
    from src.analysis.dual_component_tokenizer import base_training_metrics
    from score_single_vq_spectral_location import extract
    cfg, _ = prepared(tmp_path)
    cfg['architectures']['waveform']['target_input'] = True
    rows, scales, *_ = base_tokenizer_training_data(cfg,tmp_path)
    rows=rows['rows']
    ex=np.array(base_tokenizer_examples(rows,'eeg','validation')[:2])
    ds=rows[ex[0,0]]['dataset']
    assert all(rows[r]['dataset']==ds for r,t,p in ex)
    sites={(ds,rows[r]['positions'][p]):0 for r,t,p in ex}
    net=_base_training_model(cfg,suite.task_for('privileged','continuous')).eval()
    dest=tmp_path/'scores'
    r,t,p=ex[0]
    case={(r,t,rows[r]['positions'][p]):'fixed'}
    calibration={ds:dict(mean=[0.]*4,sd=[1.]*4)}
    with torch.inference_mode():
        extract(net,ex,ds,'validation',rows,scales,calibration,sites,case,dest)
        raw=base_tokenizer_batch(rows,ex,scales)
        s,o=[torch.as_tensor(np.where(raw['mask'],raw[k],0),dtype=torch.float32) for k in 'so']
        out=net.forward_components(s,o)
        m=base_training_metrics({k:out[k].numpy() for k in ('S','O','sum')},raw,1)
    with np.load(dest/(ds+'_validation.npz')) as z:
        assert 'id' not in z
        np.testing.assert_array_equal(z['examples'],ex)
        np.testing.assert_array_equal(z['nmse'],m['nmse'])
    with np.load(dest/'fixed_waveforms.npz') as z:
        np.testing.assert_array_equal(z['reconstructed_S'],out['S'][0].numpy())
