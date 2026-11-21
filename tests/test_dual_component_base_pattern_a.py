import numpy as np
import torch
from test_dual_component_base_g1 import ROOT, prepared, runner
from src.analysis.tokenizer_pattern_audit import attributes, match_training_patterns


def test_attribute_units_alignment_and_relabelled_patterns():
    t=(np.arange(20)-9.5)/10
    wave=np.stack([3+2*t+5*t*t,-1-t],axis=-1).reshape(1,40)
    y,fields=attributes(wave,'hb')
    for field,wanted in [('HbO_slope',2),('HbR_slope',-1),('HbO_curvature',10),('HbR_curvature',0)]:
        a,b=fields[field]
        np.testing.assert_allclose(y[:,a:b],wanted,atol=1e-12)
    parts=np.stack([np.sin(np.arange(400)*2*np.pi*f/200) for f in (3,10,40)],axis=0)
    y,fields=attributes(parts,'eeg')
    np.testing.assert_allclose(y[1,fields['power_8_13'][0]],.5,atol=1e-5)
    i,j,cost=match_training_patterns(y,y[[2,0,1]],fields)
    np.testing.assert_array_equal(j,[1,2,0])
    np.testing.assert_allclose(cost,0)


def test_pattern_a_measured_entry_on_small_development_fixture(tmp_path,monkeypatch):
    source,out=prepared(tmp_path)
    task=dict(modality='hb',architecture='waveform',candidate='vq128')
    source['diagnostics']=dict(reference_step=8)
    manifest,scales,audit,_=runner._base_training_data(source,tmp_path)
    model=runner._base_training_model(source,task)
    from collections import OrderedDict
    runner._base_initialize(model,manifest['rows'],audit['hb']['initialization'],scales,'cpu',OrderedDict(),source)
    model.eval()
    job=tmp_path/'frozen'
    evaluation=job/'evaluation/step_00008'
    runner.evaluate_base_training(source,model,manifest['rows'],scales,'hb','cpu',evaluation,OrderedDict())
    folder=tmp_path/'d0/D0/G1_hb_vq128'
    runner.write_json(folder/'summary.json',dict(source=str(evaluation)))
    np.savez(folder/'validation_same_ID.npz',example=[1])
    cfg=runner.read_config(ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_pattern_a_v1.yaml')
    cfg.update(d0_run='d0',reference_step=8,codebook_size=8)
    monkeypatch.setattr(runner,'read_config',lambda _:source)
    monkeypatch.setattr(runner,'_base_diagnostic_source',lambda *args:(model,job))
    runner._pattern_a_model(cfg,tmp_path,out,'G1_hb')
    result=runner.read_json(out/'G1_hb/summary.json')
    assert result['coverage']['validation']['complete_patches']>0
    assert len(result['branches']['O']['attribute_mse']['macro'])==4
    atlas=runner.read_json(out/'G1_hb/O_atlas.json')
    assert sum(atlas['support']['validation']['patches'])==result['coverage']['validation']['complete_patches']
    assert len(atlas['by_dataset']['patches'][0])==8
