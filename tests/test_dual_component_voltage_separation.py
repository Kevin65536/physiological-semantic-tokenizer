"""Synthetic numerical contracts for the spatial/temporal voltage frontend."""
from pathlib import Path
import numpy as np
import pytest
import yaml
from src.data.dual_component_tokenizer import (
    voltage_separation_plan,read_voltage_separation_input,fit_state_guided_subspace,
    fit_neighbor_predictor,apply_voltage_separation,phase_shuffled_states,separation_probe,
)


@pytest.fixture
def spec():
    root=Path(__file__).resolve().parents[1]
    return yaml.safe_load((root/'experiments/configs/physiology_semantic_tokenizer/dual_component_voltage_separation_v1.yaml').read_text())['separation']


def fixture_items():
    rng=np.random.default_rng(12);fit=[];selection=[]
    for phase,items in [('fit',fit),('select',selection)]:
        for _ in range(3):
            z=rng.normal(size=(45,3));cov=[]
            for row in z:
                diagonal=np.exp(.2*np.r_[row,0.,0.,0.])
                cov.append(np.diag(diagonal))
            full=np.asarray(cov);bands=np.repeat(full[:,None],5,axis=1)
            items.append(dict(phase=phase,cov=dict(full=full,band=bands,time_s=np.arange(45)*2+1),states=dict(physical=z)))
    return fit,selection


def test_projection_has_native_voltage_identities_and_retains_signed_coupling(spec):
    fit,select=fixture_items();model=fit_state_guided_subspace(fit,select,3,'physical',spec)
    np.testing.assert_allclose(model['P']@model['P'],model['P'],atol=1e-12)
    np.testing.assert_allclose(model['W'].T@model['P'],model['W'].T,atol=1e-12)
    flipped=[dict(item,states=dict(physical=-item['states']['physical'])) for item in fit]
    flipped_select=[dict(item,states=dict(physical=-item['states']['physical'])) for item in select]
    other=fit_state_guided_subspace(flipped,flipped_select,3,'physical',spec)
    np.testing.assert_allclose(other['P'],model['P'],atol=1e-10)


def test_blind_spot_excludes_current_sample_and_preserves_constants(spec):
    rng=np.random.default_rng(11);x=rng.normal(size=(18000,6))
    fit,select=fixture_items();model=fit_state_guided_subspace(fit,select,3,'physical',spec)
    model['predictor']=fit_neighbor_predictor([x],model,spec)
    np.testing.assert_allclose(model['predictor'].sum(1),1,atol=1e-12)
    model['innovation_fraction']=1.
    first=apply_voltage_separation(x,model,spec);changed=x.copy();changed[8000]+=100
    second=apply_voltage_separation(changed,model,spec)
    np.testing.assert_allclose(first['s'][8000],second['s'][8000],atol=1e-12)
    np.testing.assert_allclose(first['s']+first['o'],x,atol=1e-14)
    np.testing.assert_allclose(first['o_spatial']+first['o_temporal'],first['o'],atol=1e-14)
    constant=np.ones_like(x)*rng.normal(size=6)
    a=apply_voltage_separation(constant,model,spec,innovation_fraction=0)
    b=apply_voltage_separation(constant,model,spec,innovation_fraction=1)
    np.testing.assert_allclose(a['s'],b['s'],atol=1e-12)
    negative=apply_voltage_separation(-x,model,spec)
    np.testing.assert_allclose(first['s'],-negative['s'],atol=1e-12)


def test_joint_phase_sham_preserves_spectrum_but_changes_correspondence():
    rng=np.random.default_rng(5);z=rng.normal(size=(900,3));other=phase_shuffled_states(z,19)
    a=np.fft.rfft(z,axis=0);b=np.fft.rfft(other,axis=0)
    np.testing.assert_allclose(a[:,:,None]*a[:,None,:].conj(),b[:,:,None]*b[:,None,:].conj(),atol=1e-10)
    assert np.linalg.norm(z-other)>10


def test_read_scope_rejects_eog_training_and_test_before_files():
    entry=dict(id='forbidden',split='train')
    plan=dict(schema='voltage_separation_panel_v1',protected_data='forbidden',rows=[entry])
    with pytest.raises(ValueError,match='validation-only'):
        read_voltage_separation_input(plan,entry,include_eog=True)
    entry['split']='test'
    with pytest.raises(ValueError):read_voltage_separation_input(plan,entry)


def test_separation_contract_rejects_protected_scope_before_files(tmp_path,spec):
    bad=dict(spec,protected_data='allowed')
    with pytest.raises(ValueError,match='public development'):
        voltage_separation_plan(tmp_path,dict(separation=bad))


def test_known_fine_and_neural_probes_have_real_nonzero_voltage(spec):
    channels=['Fp1','AFF5h','AFz','F1','FC5','FC1','C3','Cz','CP5','CP1','Fp2','FC2','FC6','C4','CP2','CP6']
    for kind in ('ocular','fine','neural10','neural35','neural55'):
        value=separation_probe(channels,12,kind,spec,rms=3.)
        assert value.shape==(18000,16)
        assert np.sqrt(np.mean(value[6000:10000]**2))==pytest.approx(3.)
        if kind.startswith('neural'):assert np.std(value[6000:10000,channels.index('Cz')])>1


def test_hard_branch_transfer_uses_actual_O_instead_of_assuming_closure():
    from src.analysis.dual_component_tokenizer import separation_change_metrics
    n=np.ones((18000,2));zero=np.zeros_like(n)
    result=separation_change_metrics(zero,zero,n,'ocular',base_o=zero,changed_o=zero)
    assert result['S_signed_gain']==0
    assert result['O_signed_gain']==0
    assert result['O_change_NMSE']==1
    closed=separation_change_metrics(zero,zero,n,'ocular')
    assert closed['O_signed_gain']==1
    assert closed['O_change_NMSE']==0


def test_dictionary_hard_decode_preserves_declared_order_without_raw_signal():
    from src.tokenizers.dual_component_tokenizer import WaveformDictionary
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'experiments/scripts/run_dual_component_tokenizer.py'
    module_spec=importlib.util.spec_from_file_location('separation_runner',path)
    runner=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(runner)
    rng=np.random.default_rng(17);x=rng.normal(size=(1000,3));scale=np.array([.5,2.,4.])
    book=WaveformDictionary(size=8,levels=2,max_iter=5).fit(runner._separation_words(x,scale))
    prediction,ids=runner._separation_hard(x,scale,book,2)
    independent=WaveformDictionary.from_state(book.state()).decode(ids)
    independent=independent.transpose(0,2,1).reshape(x.shape)*scale
    np.testing.assert_array_equal(independent,prediction)
    assert ids.shape==(50,3,2)
