from copy import deepcopy

import pytest
import torch

from test_dual_component_base_g1 import ROOT, runner


CFG = ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_diagnostics_v1.yaml'


def test_target_input_is_branch_local_and_shared_across_training_and_evaluation():
    cfg = runner.read_config(CFG)
    torch.manual_seed(17)
    for task in cfg['tasks'][:2]:
        model = runner._base_training_model(cfg, task).eval()
        y = torch.randn(2, 15, model.input_dim)
        batch = dict(s=y, o=y, x=3*y)
        with torch.no_grad():
            baseline = model(y)
            output = runner._base_forward(model, batch)
            torch.testing.assert_close(output['S'], baseline['S'])
            torch.testing.assert_close(output['O'], baseline['O'])
            changed = runner._base_forward(model, dict(batch, x=-y, o=-y))
            torch.testing.assert_close(changed['S'], output['S'])
            assert not torch.equal(changed['O'], output['O'])


def test_h1_same_new_decoder_source_ids_and_no_ema_updates():
    cfg = runner.read_config(CFG)
    x = torch.randn(12,15,40)
    heads = []
    for task in cfg['tasks'][2:]:
        torch.manual_seed(17)
        model = runner._base_training_model(cfg, task)
        source = deepcopy(model)
        source.initialize_codebooks(x)
        source.eval()
        model.freeze_assignments(source)
        heads.append({k:p.clone() for k,p in model.named_parameters() if p.requires_grad})
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.001)
        before = {k:v.clone() for k,v in model.state_dict().items() if k not in heads[-1]}
        ids = source(x)['ids']
        model.train()
        out = model.forward_frozen_ids(x)
        out['sum'].square().mean().backward()
        optimizer.step()
        assert all(torch.equal(ids[b],out['ids'][b]) for b in ids)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
        assert any(not torch.equal(v,model.state_dict()[k]) for k,v in heads[-1].items())
        assert all(k.split('.')[2] in ('from_latent','decoder','output') for k in heads[-1])
    assert all(torch.equal(v,heads[1][k]) for k,v in heads[0].items())


def test_shape_baselines_use_heldout_support_and_weighted_training_only():
    import numpy as np
    from src.analysis.tokenizer_shape_diagnostics import ShapeReadout, compare_readouts, training_weights
    rows = [dict(dataset='d',subject=s,record=s,parent=s) for s in ('a','b')]
    examples = [(0,0,0),(0,0,1),(1,0,0)]
    weights = training_weights(examples,rows)
    np.testing.assert_allclose(weights,[.25,.25,.5])
    ids = {b:np.tile([0,1,0]*5,(3,1)) for b in 'SO'}
    sig = np.where(ids['S'][:,:,None]==0,[0.,2.],[2.,0.])
    bat = dict(s=sig,o=sig,mask=np.ones_like(sig,bool),vs=np.ones_like(sig),vo=np.ones_like(sig))
    reader = ShapeReadout(2,2,1)
    reader.update(bat,ids,['p']*3,weights)
    scores,coverage = reader.update(bat,ids,['p']*3,identities=np.array(examples))
    np.testing.assert_allclose(scores[:,:,2:],0,atol=1e-12)
    np.testing.assert_allclose(coverage,1)
    summary = compare_readouts(scores,coverage,examples,rows)
    np.testing.assert_allclose(summary['gain_macro'],1)
    assert len(summary['gains_by_subject'])==2


def test_frozen_source_and_d0_end_to_end_on_small_g0(tmp_path, monkeypatch):
    import yaml
    from test_dual_component_base_g1 import prepared

    old, out = prepared(tmp_path)
    old['evaluation']['integer_replay_check'] = 'disabled_by_user'
    task = dict(modality='hb',architecture='waveform',candidate='vq128')
    old['tasks'] = [task]
    path = tmp_path/'base.yaml'
    path.write_text(yaml.safe_dump(old))
    baseline = tmp_path/'baseline'
    runner.freeze_contract(baseline,old,tmp_path)
    runner.train_base_tokenizer(old,tmp_path,baseline,dict(task,candidate='warmup'),'cpu')
    runner.train_base_tokenizer(old,tmp_path,baseline,task,'cpu')
    cfg = deepcopy(old)
    cfg['diagnostics'] = dict(reference_step=8,context_hit_relative_tolerance=1.e-5)
    cfg['sources'] = dict(G1=dict(run='baseline',contract='base.yaml',architecture='waveform'))
    cfg['architectures']['frozen_g1'] = dict(old['architectures']['waveform'],frozen_source='G1')
    monkeypatch.setattr(runner,'read_config',lambda path:yaml.safe_load(path.read_text()))
    result = runner.train_base_tokenizer(cfg,tmp_path,out,dict(task,architecture='frozen_g1'),'cpu')
    assert result['new_training_steps']==8
    proof=runner.read_json(out/'models/hb__frozen_g1__vq128/assignment_verification.json')
    assert proof['status']=='passed' and proof['identical_ids']['validation']['S']>0
    summary=runner.base_d0_model(cfg,tmp_path,out,'G1','hb','vq128','cpu')
    assert summary['status']=='completed'
    folder=out/'D0/G1_hb_vq128'
    assert (folder/'validation_patterns.png').stat().st_size>0
    assert (folder/'signed_residuals.png').stat().st_size>0
    assert len(runner.read_json(folder/'shape_readout.json')['gains_by_subject'])==summary['subjects']

    original_metrics=runner.read_json(folder/'reconstruction.json')['metrics']
    recovered=runner._base_d0_recover_boundary(cfg,tmp_path,folder,'G1','vq128','cpu')
    import numpy as np
    np.testing.assert_allclose(recovered['macro'],original_metrics['boundary_jump_nmse']['macro'])
