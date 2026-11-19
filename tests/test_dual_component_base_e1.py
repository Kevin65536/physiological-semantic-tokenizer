"""The E1 comparison changes the internal grid while keeping the external interface."""
import numpy as np
import torch

from test_dual_component_base_g1 import ROOT, runner


CONFIG=ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_e1_v1.yaml'


def test_e1_internal_grid_and_matched_convolutions():
    cfg=runner.read_config(CONFIG)
    assert len(cfg['tasks'])==1 and cfg['training']['seed']==17
    spec=cfg['tasks'][0]
    model=runner._base_training_model(cfg,spec)
    reference=runner.read_config(ROOT/cfg['e1']['baseline']['contract'])
    baseline=runner._base_training_model(reference,dict(spec,architecture='waveform'))
    x=torch.randn(2,15,400)
    for m,steps in [(model,8),(baseline,4)]:
        branch=m.branches['S']
        encoded=branch['input'].features(x.reshape(2,1,6000))
        assert encoded.shape==(2,32,15*steps)
        assert branch['input'].project.in_features == branch['output'].project.out_features == 32*steps
        pred=m(x)
        assert not m.hard and pred['latent']['S'].shape==(2,15,64)
        assert pred['sum'].shape==x.shape and torch.isfinite(pred['sum']).all()
    for part,sub in (('input','features'),('output','synthesis')):
        def kernels(m):
            return [v.kernel_size for v in getattr(m.branches['S'][part],sub) if hasattr(v,'kernel_size')]
        assert kernels(model)==kernels(baseline)


def test_e1_report_checks_matched_support_and_both_frequency_bands(tmp_path):
    cfg=runner.read_config(CONFIG)
    out=tmp_path/'run'
    baseline=tmp_path/'g1'
    job=out/'models/eeg__waveform8__continuous'
    runner.write_json(out/'plan.json',dict(baseline_job='g1'))
    runner.write_json(job/'summary.json',dict(task_id=job.name,status='completed'))
    for i,folder in enumerate((baseline,job)):
        dest=folder/'evaluation/step_10000'
        runner.write_json(dest/'validation.json',dict(selection_score=.25-.02*i,inputs=1,metrics={
            'band_13_30_log10_power_ratio':dict(by_dataset=dict(refed=[[-3+i]])),
            'band_30_45_log10_power_ratio':dict(by_dataset=dict(refed=[[-3-i]]))}))
        np.savez(dest/'validation.npz',examples=[[0,0,0]])
    value=runner.summarize_base_training(cfg,out,tmp_path)['E1']
    assert value['delta_macro_nmse']<0 and not value['joint_direction_met']
    assert value['REFED_X']['13_30']['change_in_absolute_dB']==-10
