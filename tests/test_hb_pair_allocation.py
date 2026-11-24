import hashlib
import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits
from src.tokenizers.hb_pair_allocation import optimal_delta,design,sufficient_statistics,fit_from_statistics,apply_pair,load_model,fixed_color_gains,fit_signed_gains,spectral_design,spectral_statistics,fit_spectral,apply_spectral


def fixture(n=24):
    rng=np.random.default_rng(8551)
    ids={b:rng.integers(0,size,(n,15)) for b,size in [('S',128),('O',256)]}
    ids['S'][-1]=128;ids['O'][-1]=256
    s=rng.normal(size=(n,15,40));o=rng.normal(size=s.shape)
    prediction=dict(S=s,O=o,sum=s+o)
    bank=dict(s=rng.normal(size=s.shape),o=rng.normal(size=s.shape),mask=rng.random(s.shape)>.1,
              vs=np.tile([.4,1.7],(n,1)),vo=np.tile([1.3,.6],(n,1)))
    return ids,prediction,bank


def test_weighted_optimum_and_native_coordinate_are_exact():
    _,p,b=fixture(6);delta=optimal_delta(p,b);mask=b['mask'].reshape(6,300,2)
    y={k:np.asarray(b[k]).reshape(6,300,2) for k in ['s','o']}
    s,o=[p[k].reshape(6,300,2) for k in 'SO']
    cost=lambda d:((s-d-y['s'])**2/b['vs'][:,None]+(o+d-y['o'])**2/b['vo'][:,None])[mask].sum()
    assert cost(delta)<cost(delta+.01) and cost(delta)<cost(delta-.01)
    scale=np.linspace(.1,5,6)
    native={k:v.copy() for k,v in b.items()}
    for k in ['s','o']:native[k]*=scale[:,None,None]
    for k in ['vs','vo']:native[k]*=scale[:,None]**2
    np.testing.assert_allclose(optimal_delta({k:v*scale[:,None,None] for k,v in p.items()},native),delta*scale[:,None,None],atol=1e-14)


def test_final_smoothing_is_inside_masked_joint_fit_and_sum_quiet_are_preserved():
    ids,p,b=fixture();a=design(p,'window')
    np.testing.assert_array_equal(a[:,:,:111],design(p,'local'))
    coefficient=np.zeros((147,2));coefficient[0]=[.15,-.2];coefficient[42]=[.05,.04]
    law=(a@coefficient).reshape(24,15,40)
    b['s']=p['S']-law;b['o']=p['O']+law
    with threadpool_limits(1):stats=sufficient_statistics(ids,p,b,np.ones(24)/24,batch_size=8)
    fitted=fit_from_statistics(stats,'window',1e-7)
    out=apply_pair(ids,p,fitted)
    valid=b['mask'].copy();valid[-1]=False
    assert np.mean((out['O'][valid]-b['o'][valid])**2)<1e-12
    np.testing.assert_array_equal(out['sum'],p['sum'])
    for k in p:np.testing.assert_array_equal(out[k][-1],p[k][-1])
    np.testing.assert_allclose(out['S']+out['O'],p['sum'],atol=1e-14)


@pytest.mark.parametrize('with_dataset',[False,True])
def test_saved_decoder_accepts_only_integer_source_and_frozen_model(tmp_path,monkeypatch,with_dataset):
    import src.tokenizers.hb_pair_allocation as module
    ids,p,_=fixture(3)
    class Source:
        def decode_ids(self,values):
            from src.tokenizers.hb_parent_distillation import validate_ids
            validate_ids(values)
            return {k:v.copy() for k,v in p.items()}
        def encode(self,*_):raise AssertionError('decoder requested X')
    source=tmp_path/'native.pt';source.write_bytes(b'frozen integer source fixture')
    monkeypatch.setattr(module,'load_native',lambda _:Source())
    fitted=dict(family='local',coefficient=np.ones((111,2))*.001)
    if with_dataset:fitted['dataset_color_gains']={'fixture':[-.5,.3]}
    path=tmp_path/'allocation.pt'
    torch.save(dict(metadata=dict(schema='hb_pair_allocation_gain_v1' if with_dataset else 'hb_pair_allocation_v1',source_file=source.name,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),fitted=fitted),path)
    model=load_model(path);expected=model.decode_ids(ids,dataset='fixture')
    np.savez(tmp_path/'ids.npz',**ids)
    with np.load(tmp_path/'ids.npz') as disk:actual=load_model(path).decode_ids(dict(disk),dataset='fixture')
    for k in expected:np.testing.assert_array_equal(expected[k],actual[k])
    source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity'):load_model(path)


def test_registered_111_outer_folds_and_no_continuous_side_input_contract():
    from pathlib import Path
    import yaml
    from experiments.scripts.run_hb_native_id_readout import validate_pair_config
    cfg=yaml.safe_load((Path(__file__).parents[1]/'experiments/configs/physiology_semantic_tokenizer/dual_component_hb_pair_allocation_v1.yaml').read_text())
    validate_pair_config(cfg)
    assert cfg['model']['candidates']==4 and cfg['model']['scalar_coefficients']==dict(local=222,window=294)
    cfg['data']['selection_splits']=['validation']
    with pytest.raises(ValueError,match='train-first'):validate_pair_config(cfg)


def test_signed_dataset_gain_is_integer_only_and_uses_required_fixed_metadata():
    ids,p,b=fixture(8);fitted=dict(family='window',coefficient=np.ones((147,2))*.001)
    plain=apply_pair(ids,p,fitted);delta=plain['O']-p['O'];labels=np.array(['a']*4+['b']*4)
    law=np.array([[.3,-.7]]*4+[[-.5,.2]]*4)
    target=delta.reshape(8,300,2)*law[:,None]
    fit=fit_signed_gains(delta,target,b,np.ones(8)/8,labels,'dataset')
    np.testing.assert_allclose(fit['a']['gain'],[.3,-.7],atol=1e-14)
    np.testing.assert_allclose(fit['b']['gain'],[-.5,.2],atol=1e-14)
    fitted['dataset_color_gains']={k:v['gain'] for k,v in fit.items()}
    with pytest.raises(ValueError,match='dataset metadata'):apply_pair(ids,p,fitted)
    with pytest.raises(ValueError,match='unknown'):fixed_color_gains(fitted,['bad']*8,8)
    result=apply_pair(ids,p,fitted,dataset=labels)
    np.testing.assert_allclose(result['O'],p['O']+target.reshape(8,15,40),atol=2e-15)
    np.testing.assert_array_equal(result['sum'],p['sum'])
    for k in result:np.testing.assert_array_equal(result[k][-1],p[k][-1])


def test_whole30s_spectral_groups_retain_only_declared_input_bands():
    from scipy.fft import dct
    ids,p,_=fixture(4);a=spectral_design(p)
    assert a.shape==(4,300,12)
    for band,(lo,hi) in enumerate([(0,1),(1,3),(3,13)]):
        frequency=dct(a[:,:,4*band:4*(band+1)],axis=1,norm='ortho')
        outside=np.ones(300,bool);outside[lo:hi]=False
        np.testing.assert_allclose(frequency[:,outside],0,atol=2e-15)
    original=np.stack([p[b].reshape(4,300,2) for b in 'SO'],2).reshape(4,300,4)
    np.testing.assert_allclose(a[:,:,:4],np.broadcast_to(original.mean(1)[:,None,:],(4,300,4)),atol=2e-16)


def test_wholewindow_dataset_fit_uses_masked_native_joint_loss_and_keeps_quiet():
    ids,p,b=fixture(32);labels=np.array(['a']*16+['b']*16);a=spectral_design(p)
    rng=np.random.default_rng(77);law={'a':rng.normal(0,.1,(12,2)),'b':rng.normal(0,.1,(12,2))}
    delta=np.stack([a[i]@law[str(d)] for i,d in enumerate(labels)]);delta[-1]=0
    b['s']=p['S']-delta.reshape(32,15,40);b['o']=p['O']+delta.reshape(32,15,40)
    with threadpool_limits(1):stats=spectral_statistics(ids,p,b,np.ones(32)/32,labels,batch_size=9)
    fitted=fit_spectral(stats,'dataset',1e-12);prediction=apply_spectral(ids,p,fitted,dataset=labels)
    for k,target in [('S','s'),('O','o')]:np.testing.assert_allclose(prediction[k][b['mask']],b[target][b['mask']],atol=1e-11)
    np.testing.assert_array_equal(prediction['sum'],p['sum'])
    for k in prediction:np.testing.assert_array_equal(prediction[k][-1],p[k][-1])
    with pytest.raises(ValueError,match='dataset'):apply_spectral(ids,p,fitted)
    with pytest.raises(ValueError,match='dataset'):apply_spectral(ids,p,fitted,dataset=['bad']*32)
    scale=np.linspace(.2,3,32);native={k:v.copy() for k,v in b.items()}
    for k in ['s','o']:native[k]*=scale[:,None,None]
    for k in ['vs','vo']:native[k]*=scale[:,None]**2
    with threadpool_limits(1):transformed=spectral_statistics(ids,{k:v*scale[:,None,None] for k,v in p.items()},native,np.ones(32)/32,labels,batch_size=9)
    fitted_native=fit_spectral(transformed,'dataset',1e-12)
    for d in law:np.testing.assert_allclose(fitted_native['coefficients'][d],fitted['coefficients'][d],atol=1e-13)


@pytest.mark.parametrize('schema',['hb_spectral_allocation_v1','hb_spectral_protected_v1'])
def test_serialized_spectral_decoder_uses_only_integers_and_fixed_dataset(tmp_path,monkeypatch,schema):
    import src.tokenizers.hb_pair_allocation as module
    ids,p,_=fixture(3)
    class Source:
        def decode_ids(self,values):
            from src.tokenizers.hb_parent_distillation import validate_ids
            validate_ids(values);return {k:v.copy() for k,v in p.items()}
        def encode(self,*_):raise AssertionError('spectral decoder requested X')
    source=tmp_path/'native.pt';source.write_bytes(b'frozen integer-only source')
    monkeypatch.setattr(module,'load_native',lambda _:Source())
    fitted=dict(scope='dataset',coefficients={'fixture':np.full((12,2),.01)},groups=[[0,1],[1,3],[3,13]])
    path=tmp_path/'spectral.pt';torch.save(dict(metadata=dict(schema=schema,source_file=source.name,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),fitted=fitted),path)
    first=load_model(path).decode_ids(ids,dataset='fixture');np.savez(tmp_path/'ids.npz',**ids)
    with np.load(tmp_path/'ids.npz') as disk:second=load_model(path).decode_ids(dict(disk),dataset='fixture')
    for b in first:np.testing.assert_array_equal(first[b],second[b])
    with pytest.raises(ValueError,match='dataset'):load_model(path).decode_ids(ids)


def test_spectral_contract_rejects_unsupported_frequencies_and_split_changes():
    from pathlib import Path
    import yaml
    from experiments.scripts.run_hb_native_id_readout import validate_spectral_config
    cfg=yaml.safe_load((Path(__file__).parents[1]/'experiments/configs/physiology_semantic_tokenizer/dual_component_hb_spectral_allocation_v1.yaml').read_text())
    validate_spectral_config(cfg);cfg['model']['groups_DCT_indices'][-1][1]=25
    with pytest.raises(ValueError,match='spectral contract'):validate_spectral_config(cfg)


def test_protected_contract_rejects_scope_expansion_before_data():
    from pathlib import Path
    from copy import deepcopy
    import yaml
    from experiments.scripts.run_hb_native_id_readout import validate_protected_config
    cfg=yaml.safe_load((Path(__file__).parents[1]/'experiments/configs/physiology_semantic_tokenizer/dual_component_hb_spectral_protected_v1.yaml').read_text())
    validate_protected_config(cfg)
    for section,key,value in [('data','allowed_splits',['train','validation']),('data','confirmation_access','allowed'),('fitting','gain_bounds',[-1,1]),('model','extra_integers',1)]:
        altered=deepcopy(cfg);altered[section][key]=value
        with pytest.raises(ValueError,match='protected spectral contract'):validate_protected_config(altered)


def test_protection_moments_match_original_masked_hierarchy_and_native_coordinates():
    from src.tokenizers.hb_pair_allocation import spectral_group_delta,protection_moments,protection_values
    from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics,base_training_aggregate
    from src.analysis.tokenizer_shape_diagnostics import training_weights
    ids,p,b=fixture(12);b['mask'][:]=True;b['mask'][0,:2]=False
    b.update(x=b['s']+b['o'],vx=b['vs']+b['vo'],dvx=np.ones((12,2)),dvs=np.ones((12,2)),dvo=np.ones((12,2)))
    expanded={k:(np.broadcast_to(np.tile(v,(1,20))[:,None],p['S'].shape) if k.startswith('v') or k.startswith('dv') else v) for k,v in b.items()}
    rows=[dict(dataset='d',subject=str(i//2),record=str(i//2),parent=i) for i in range(6)]
    examples=np.array([(i//2,0,i%2) for i in range(12)]);weights=training_weights(examples,rows)
    coef=np.arange(24).reshape(12,2)/300;fitted=dict(scope='dataset',coefficients={'d':coef},groups=[[0,1],[1,3],[3,13]])
    labels=np.full(12,'d');directions=spectral_group_delta(ids,p,fitted,labels);gain=np.array([.2,.6,.8])
    change=(directions@gain).reshape(12,15,40);expected=dict(S=p['S']-change,O=p['O']+change,sum=p['sum'])
    score=joint_reconstruction_metrics(expected,expanded,2,np.ones(12));aggregate=base_training_aggregate(score['metrics'],examples,rows)
    for color in range(2):
        m=protection_moments(p,b,directions,color,weights,lambda finite:training_weights(examples[finite],rows));v=protection_values(m,gain)
        for component,branch in [(1,'S'),(2,'O')]:
            for name,field in [('rms','rms_gain'),('slope','slope_correlation'),('curvature','curvature_correlation')]:
                np.testing.assert_allclose(v[branch+'_'+name],aggregate['metrics'][field]['macro'][component][color],atol=1e-12)
        np.testing.assert_allclose(v['joint_loss'],np.asarray(aggregate['metrics']['nmse']['macro'])[1:,color].sum(),atol=1e-12)
    scale=3.4;native={k:(v*scale**2 if k.startswith('v') or k.startswith('dv') else v*scale if k in ('x','s','o') else v) for k,v in b.items()}
    m2=protection_moments({k:v*scale for k,v in p.items()},native,directions*scale,0,weights)
    np.testing.assert_allclose(protection_values(m2,gain)['joint_loss'],protection_values(protection_moments(p,b,directions,0,weights),gain)['joint_loss'],atol=1e-12)


def test_single_protected_solver_recovers_feasible_law():
    from src.tokenizers.hb_pair_allocation import protection_moments,protection_values,fit_protected_gains
    _,p,b=fixture(16);b['mask'][:]=True
    rng=np.random.default_rng(121);directions=rng.normal(size=(16,300,2,3))*.04
    delta=directions@np.array([.2,.4,.6]);b['s']=p['S']-delta.reshape(16,15,40);b['o']=p['O']+delta.reshape(16,15,40)
    m=protection_moments(p,b,directions,0,np.ones(16)/16);fit=fit_protected_gains(m)
    assert np.all(np.asarray(fit['gain'])>=0) and np.all(np.asarray(fit['gain'])<=1)
    assert fit['minimum_constraint']>=-1e-9
    assert fit['protected']['joint_loss']<.02*fit['baseline']['joint_loss']
    np.testing.assert_allclose(fit['gain'],[.2,.4,.6],atol=2e-4)


def test_protected_solver_stops_at_original_amplitude_constraint_when_joint_loss_wants_more():
    from src.tokenizers.hb_pair_allocation import protection_moments,protection_values,fit_protected_gains
    n=8;t=np.arange(300)/10;wave=np.sin(t[None,:,None]+np.arange(n)[:,None,None]*.3)*np.array([1.,.7])[None,None]
    p=dict(S=(2*wave).reshape(n,15,40),O=wave.reshape(n,15,40));p['sum']=p['S']+p['O']
    b=dict(s=(2.5*wave).reshape(n,15,40),o=p['O'].copy(),mask=np.ones((n,15,40),bool),vs=np.ones((n,2))*.001,vo=np.ones((n,2)))
    d=np.zeros((n,300,2,3));d[:,:,:,0]=-.5*wave
    m=protection_moments(p,b,d,0,np.ones(n)/n);fit=fit_protected_gains(m)
    assert protection_values(m,[1,0,0])['joint_loss']<fit['protected']['joint_loss']
    assert protection_values(m,[1,0,0])['O_rms']==pytest.approx(.5)
    assert fit['gain'][0]==pytest.approx(.5,abs=1e-7)
    assert fit['protected']['O_rms']==pytest.approx(.75,abs=1e-8)
    assert fit['minimum_constraint']>=-1e-9


def test_non_circular_same_frequency_quadrature_adds_real_phase_freedom():
    from src.tokenizers.hb_pair_allocation import phase_design
    n=300;q=np.arange(n)+.5;k=4;wave=np.cos(np.pi*k*q/n)
    z=np.zeros((1,300,2));z[0,:,0]=wave
    p=dict(S=z.reshape(1,15,40),O=np.zeros((1,15,40)));p['sum']=p['S']
    odd=phase_design(p,'odd')[0,:,4];expected=np.sin(np.pi*k*q/n)
    np.testing.assert_allclose(odd,expected,atol=3e-15)
    assert abs(odd@wave)<1e-12
    assert np.linalg.matrix_rank(np.column_stack((wave,odd)),tol=1e-10)==2
    virtual=np.sqrt(2/n)*np.sin(np.pi*np.array([0,n])[:,None]*np.arange(1,13)[None]/n)
    assert abs(virtual).max()<np.finfo(float).eps*13*np.pi*np.sqrt(2/n)
    assert odd[0]==pytest.approx(-odd[-1])
    # A half-cycle sine is positive throughout the interior, with zero virtual edges;
    # it is not a cyclic shift that wraps the two ends of the visible window.
    z[0,:,0]=np.cos(np.pi*q/n);p['S']=z.reshape(1,15,40);p['sum']=p['S']
    half=phase_design(p,'odd')[0,:,0]
    assert np.all(half>0) and half[0]==pytest.approx(half[-1])


def test_target_informed_phase_oracle_recovers_known_shift_and_preserves_mask_sum_quiet():
    from src.tokenizers.hb_pair_allocation import phase_design,phase_oracle
    ids,p,b=fixture(6);b['mask'][:]=True;b['mask'][0,:3]=False
    a=phase_design(p,'odd');coef=np.arange(16).reshape(8,2)/30;delta=(a@coef).reshape(6,15,40)
    b['s']=p['S']+delta;b['o']=p['O']-delta
    out,fitted,rank=phase_oracle(ids,p,b,'odd',batch_size=3)
    for branch,key in [('S','s'),('O','o')]:
        np.testing.assert_allclose(out[branch][:-1][b['mask'][:-1]],b[key][:-1][b['mask'][:-1]],atol=1e-13)
        np.testing.assert_array_equal(out[branch][-1],p[branch][-1])
    np.testing.assert_array_equal(out['sum'],p['sum']);np.testing.assert_allclose(out['S']+out['O'],p['sum'],atol=1e-14)
    assert rank.shape==(6,2) and np.all(rank[-1]==0) and np.max(rank)<=6
    np.testing.assert_allclose(np.einsum('ntg,ncg->ntc',a,fitted).reshape(6,15,40)[:-1],delta[:-1],atol=1e-13)


def test_phase_contract_rejects_circular_or_expanded_scope_before_arrays():
    from pathlib import Path
    from copy import deepcopy
    import yaml
    from experiments.scripts.run_hb_native_id_readout import validate_phase_config
    cfg=yaml.safe_load((Path(__file__).parents[1]/'experiments/configs/physiology_semantic_tokenizer/dual_component_hb_phase_companion_v1.yaml').read_text())
    validate_phase_config(cfg)
    for section,key,value in [('data','allowed_splits',['train','validation']),('data','confirmation_access','allowed'),('model','virtual_edges_q',[0,299]),('oracle','numerical_rank_relative_cutoff',1e-3)]:
        altered=deepcopy(cfg);altered[section][key]=value
        with pytest.raises(ValueError,match='phase reachability contract'):validate_phase_config(altered)


def test_fit_only_phase_statistics_match_masked_joint_objective_and_native_scaling():
    from src.tokenizers.hb_pair_allocation import phase_statistics,fit_phase,apply_phase,phase_design
    ids,p,b=fixture(12);labels=np.array(['a']*6+['b']*6);weights=np.ones(12)/12
    coef=np.arange(32).reshape(16,2)/100;delta=(phase_design(p,'even_plus_odd')@coef).reshape(12,15,40)
    b['s']=p['S']+delta;b['o']=p['O']-delta
    stats=phase_statistics(ids,p,b,weights,labels,batch_size=5);fit=fit_phase(stats);out=apply_phase(ids,p,fit,labels)
    a=phase_design(p,'even_plus_odd');target=-optimal_delta(p,b);mask=b['mask'].reshape(12,300,2)
    w=weights[:,None,None]*mask/np.maximum(mask.sum(1)[:,None],1)*(1/b['vs']+1/b['vo'])[:,None]
    w[-1]=0
    for c in range(2):
        np.testing.assert_allclose(stats['a']['right'][c],np.einsum('ntg,nt,nt->g',a[:6],w[:6,:,c],target[:6,:,c]),atol=1e-15)
    np.testing.assert_array_equal(out['sum'],p['sum'])
    for branch in p:np.testing.assert_array_equal(out[branch][-1],p[branch][-1])
    np.testing.assert_allclose(out['S']+out['O'],p['sum'],atol=1e-14)
    scale=3.;native={k:(v*scale if k in ['s','o'] else v*scale**2 if k in ['vs','vo'] else v) for k,v in b.items()}
    native_prediction={k:v*scale for k,v in p.items()};native_fit=fit_phase(phase_statistics(ids,native_prediction,native,weights,labels))
    for d in labels:np.testing.assert_allclose(native_fit['coefficients'][d],fit['coefficients'][d],atol=1e-13)


def test_phase_frozen_checkpoint_replays_ids_without_truth_or_continuous_side_input(tmp_path,monkeypatch):
    import src.tokenizers.hb_pair_allocation as module
    ids,p,_=fixture(4)
    class Source:
        def decode_ids(self,values):return {k:v.copy() for k,v in p.items()}
        def encode(self,*_):raise AssertionError('X requested')
    source=tmp_path/'native.pt';source.write_bytes(b'frozen source')
    monkeypatch.setattr(module,'load_native',lambda _:Source())
    protected=dict(scope='dataset',coefficients={'fixture':np.zeros((12,2))},groups=[[0,1],[1,3],[3,13]])
    phase=dict(family='whole_window_even_odd_phase',coefficients={'fixture':np.ones((16,2))*.01})
    path=tmp_path/'phase.pt';torch.save(dict(metadata=dict(schema='hb_phase_fit_v1',source_file=source.name,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),fitted=phase,protected_fitted=protected),path)
    expected=load_model(path).decode_ids(ids,dataset='fixture');np.savez(tmp_path/'ids.npz',**ids)
    with np.load(tmp_path/'ids.npz') as f:actual=load_model(path).decode_ids(dict(f),dataset='fixture')
    for b in p:np.testing.assert_array_equal(expected[b],actual[b])
    with pytest.raises(ValueError,match='dataset'):load_model(path).decode_ids(ids)


def test_phase_fit_scope_and_four_direction_original_protection():
    from pathlib import Path
    from copy import deepcopy
    import yaml
    from experiments.scripts.run_hb_native_id_readout import validate_phase_fit_config
    from src.tokenizers.hb_pair_allocation import protection_moments,protection_values,fit_protected_gains
    cfg=yaml.safe_load((Path(__file__).parents[1]/'experiments/configs/physiology_semantic_tokenizer/dual_component_hb_phase_fit_v1.yaml').read_text());validate_phase_fit_config(cfg)
    for section,key,value in [('data','allowed_splits',['validation']),('fitting','ridge',.1),('fitting','candidates',2)]:
        bad=deepcopy(cfg);bad[section][key]=value
        with pytest.raises(ValueError,match='phase fit contract'):validate_phase_fit_config(bad)
    _,p,b=fixture(6);rng=np.random.default_rng(6);d=rng.normal(size=(6,300,2,4))*.01
    m=protection_moments(p,b,d,0,np.ones(6)/6);reference=protection_values(m,np.zeros(4));fit=fit_protected_gains(m,protection_reference=reference)
    assert len(fit['gain'])==4 and fit['minimum_constraint']>=-1e-9
