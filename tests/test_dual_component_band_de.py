import numpy as np
import pytest
import torch

from src.analysis.dual_component_tokenizer import eeg_balanced_distortion
from test_dual_component_base_g1 import ROOT, runner, prepared


def contract():
    return runner.read_config(ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_band_de_v1.yaml')


def test_complex_phase_mask_and_scale_floor(tmp_path):
    cfg,_=prepared(tmp_path)
    cfg['band_de']=contract()['band_de']
    manifest,scales,audit,_=runner._base_training_data(cfg,tmp_path)
    examples=audit['eeg']['initialization']
    result=runner._de_scales(cfg,manifest['rows'],examples,scales)
    assert np.asarray(result['scales']).shape==(3,5)
    assert np.min(result['scales'])>=cfg['band_de']['band_power_floor']
    bad=[(next(i for i,r in enumerate(manifest['rows']) if r['split']=='validation'),0,0)]
    with pytest.raises(ValueError,match='training record'):
        runner._de_scales(cfg,manifest['rows'],bad,scales)
    y=torch.sin(2*torch.pi*35*torch.arange(400)/200).expand(2,15,400).clone()
    pred=(-y).requires_grad_()
    mask=torch.ones_like(y,dtype=torch.bool)
    mask[:,0,0]=False
    batch=dict(x=y,s=y,o=y,mask=mask,**{k:torch.ones_like(y) for k in ['vx','vs','vo']})
    def loss(z):return eeg_balanced_distortion(dict(S=z,O=z,sum=z),batch,cfg['band_de'],torch.ones(3,5))
    assert loss(y)==0 and loss(pred)>0
    loss(pred).backward()
    assert torch.isfinite(pred.grad).all() and pred.grad[:,0].abs().sum()==0


def test_joint_gate_checks_each_band_and_component():
    base=dict(refed_db=[-35,-42],component_nmse=np.array([.15,.30,.15]),macro_nmse=.20,correlation=np.array([.9,.8,.9]))
    new=dict(base,refed_db=[-34,-37])
    limits=contract()['band_de']['gate']
    assert runner._de_gate(base,new,limits)['passed']
    assert not runner._de_gate(base,dict(new,refed_db=[-35,-37]),limits)['passed']
    assert not runner._de_gate(base,dict(new,component_nmse=[.17,.30,.15]),limits)['passed']


def test_new_hard_atlas_uses_heldout_real_waves_on_existing_panel(tmp_path):
    from collections import OrderedDict
    source,out=prepared(tmp_path)
    manifest,scales,audit,_=runner._base_training_data(source,tmp_path)
    task=dict(modality='eeg',architecture='waveform',candidate='vq128')
    model=runner._base_training_model(source,task)
    runner._base_initialize(model,manifest['rows'],audit['eeg']['initialization'],scales,'cpu',OrderedDict(),source)
    model.eval()
    job=tmp_path/'frozen'
    runner.evaluate_base_training(source,model,manifest['rows'],scales,'eeg','cpu',job/'evaluation/step_00008',OrderedDict())
    pattern=runner.read_config(ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_pattern_a_v1.yaml')
    pattern.update(reference_step=8,codebook_size=8)
    runner._pattern_scores(pattern,source,manifest['rows'],scales,model,job,out,'eeg')
    record=runner.read_json(out/'summary.json')
    with np.load(record['validation_gallery']) as f:
        assert f['frequency'][0].sum()==record['coverage']['validation']['complete_patches']
