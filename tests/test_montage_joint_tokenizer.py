"""Full-montage gradients and independently reloaded hard integer synthesis."""
import numpy as np
import pytest
import torch

from src.tokenizers.montage_joint_tokenizer import MontageJointTokenizer
from src.tokenizers.spatial_waveform_dictionary import SpatialWaveformDictionary, joint_assignment_features, load_model


def fixture_model(count=10,size=8):
    rng=np.random.default_rng(17)
    metadata=dict(schema='spatial_waveform_dictionary_v1',positions={'fixture':[str(i) for i in range(count)]},
        codebook_size=size,output_projection_hz=[1,45],semantic_distance_fraction=.7,top_m=min(5,size-1),temperature=.1)
    atoms={(b,k):np.r_[np.zeros((1,400)),rng.normal(0,.2,(size-1,400))] for b in ('S','O') for k in range(5)}
    weights={(b,k):np.full(joint_assignment_features(atoms[b,k][1:],k).shape[-1],.05) for b in ('S','O') for k in range(5)}
    centers={(b,k):joint_assignment_features(atoms[b,k][1:],k)*weights[b,k] for b in ('S','O') for k in range(5)}
    spatial={('fixture',b,k):rng.normal(0,.2,(len(np.flatnonzero(np.arange(count)%5==k)),count)) for b in ('S','O') for k in range(5)}
    source=SpatialWaveformDictionary(metadata,{b:np.full(3001,.5) for b in ('S','O')},centers,atoms,spatial,feature_weights=weights)
    return source,MontageJointTokenizer(source),rng


def test_initialization_reproduces_full_thirty_second_spatial_decoder():
    torch.set_num_threads(1)
    source,model,rng=fixture_model()
    values=rng.normal(size=(1,10,15,400))
    observed=np.ones((1,10),bool)
    for missing in (False,True):
        if missing:observed[:,np.arange(10)%5==2]=False
        ids=source.encode(values,'fixture',observed)
        learned=model.encode(values,'fixture',observed)
        for b in ('S','O'):np.testing.assert_array_equal(ids[b],learned[b])
        original=source.decode_ids(ids,'fixture')
        actual=model.decode_ids(ids,'fixture')
        for b in ('S','O','sum'):np.testing.assert_allclose(actual[b],original[b],atol=3e-7,rtol=2e-5)


def test_full_montage_loss_reaches_atoms_assignments_and_cross_branch_weights(tmp_path):
    torch.set_num_threads(1)
    _,model,rng=fixture_model()
    values=rng.normal(size=(1,10,15,400))
    features,candidates=model.training_features(values,'fixture')
    observed=torch.ones(10,dtype=torch.bool)
    atoms,_,ids=model.network.assign(features,candidates,observed,'fixture')
    output=model.network.decode(atoms,'fixture',observed=observed)
    loss=(output-torch.tensor(rng.normal(size=output.shape),dtype=torch.float32)).square().mean()
    loss.backward()
    for group in ('atoms','centers','bias','synthesis'):
        assert sum(float(p.grad.abs().sum()) for p in getattr(model.network,group).values())>0
    assert model.network.synthesis['fixture_1'].grad[:10,10:].abs().sum()>0
    codes={b:ids[b].detach().numpy()[None].astype(np.int32) for b in ('S','O')}
    decoded=model.decode_ids(codes,'fixture')
    np.testing.assert_allclose(output.detach().numpy(),np.stack((decoded['S'][0],decoded['O'][0])),atol=0,rtol=0)
    path=tmp_path/'model.npz';model.save(path)
    restored=load_model(path)
    restored.predict_continuous=lambda *_:(_ for _ in ()).throw(AssertionError('decoder read observations'))
    for b in ('S','O','sum'):np.testing.assert_array_equal(decoded[b],restored.decode_ids(codes,'fixture')[b])
    with pytest.raises(ValueError):restored.decode_ids({b:v.astype(float) for b,v in codes.items()},'fixture')
    invalid={b:v.copy() for b,v in codes.items()};invalid['S'][0,0,0]=8
    with pytest.raises(ValueError):restored.decode_ids(invalid,'fixture')


def test_differentiable_energy_coordinates_match_owning_attribute_computation():
    from experiments.scripts.run_joint_montage_tokenizer import log_attributes
    from src.tokenizers.spectral_waveform_dictionary import raw_features
    rng=np.random.default_rng(6)
    values=rng.normal(size=(19,400))
    actual=log_attributes(torch.from_numpy(values)).numpy()
    expected=raw_features(values,None,None,'theta_alpha_rms')
    np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-10)


def test_global_analysis_has_cross_electrode_gradients_and_requires_full_context(tmp_path):
    torch.set_num_threads(1)
    source,_,rng=fixture_model()
    metadata=dict(source.metadata,global_analysis=True)
    model=MontageJointTokenizer(source,metadata)
    values=rng.normal(size=(1,10,15,400))
    observed=np.ones((1,10),bool)
    source_ids=source.encode(values,'fixture',observed)
    actual_ids=model.encode(values,'fixture',observed)
    for b in ('S','O'):np.testing.assert_array_equal(actual_ids[b],source_ids[b])
    features,candidates=model.training_features(values,'fixture')
    atoms,_,ids=model.network.assign(features,candidates,torch.from_numpy(observed[0]),'fixture')
    output=model.network.decode(atoms,'fixture')
    (output-torch.randn_like(output)).square().mean().backward()
    for p in model.network.analysis.values():
        assert p.grad.abs().sum()>0
        assert torch.count_nonzero(p.grad)>torch.count_nonzero(p)
    with torch.no_grad():
        for p in model.network.analysis.values():p.add_(torch.randn_like(p)*.03)
    atoms,_,learned_ids=model.network.assign(features,candidates,torch.from_numpy(observed[0]),'fixture')
    hard_output=model.network.decode(atoms,'fixture').detach().numpy()
    integer_output=model.decode_ids({b:v.detach().numpy()[None] for b,v in learned_ids.items()},'fixture')
    np.testing.assert_array_equal(hard_output,np.stack((integer_output['S'][0],integer_output['O'][0])))
    actual_ids=model.encode(values,'fixture',observed)
    path=tmp_path/'global.npz';model.save(path)
    restored=load_model(path)
    for b in ('S','O'):np.testing.assert_array_equal(actual_ids[b],restored.encode(values,'fixture',observed)[b])
    with pytest.raises(ValueError):model.encode_positions(values[:,:1],'fixture',np.array([[0]]))
    observed[:,1]=False
    hidden_a=values.copy();hidden_b=values.copy();hidden_a[:,1]=100.;hidden_b[:,1]=-100.
    a=model.encode(hidden_a,'fixture',observed);b=model.encode(hidden_b,'fixture',observed)
    for branch in ('S','O'):np.testing.assert_array_equal(a[branch],b[branch])


def test_masked_complex_phase_statistics_equal_direct_time_loss():
    from src.tokenizers.montage_joint_tokenizer import phase_basis_statistics,phase_joint_quadratic,phase_coefficient
    rng=np.random.default_rng(51);waves=rng.normal(size=(2,6000));targets=rng.normal(size=(3,6000));variance=np.array([1.,.3,.7])
    vector=rng.normal(size=176)*.2;coefficient=phase_coefficient(vector)
    spectrum=np.fft.rfft(waves,axis=-1,norm='ortho').T
    output=np.fft.irfft(np.einsum('fi,fio->fo',spectrum,coefficient),n=6000,axis=0,norm='ortho').T
    prediction=np.stack((output.sum(0),output[0],output[1]))
    for invalid in (0,20,450):
        mask=np.ones(6000,bool);mask[:invalid]=False
        gram,cross,energy=phase_basis_statistics(waves,targets,mask)
        matrix,linear,constant=phase_joint_quadratic(gram,cross,energy,variance)
        direct=np.sum(np.mean((prediction[:,mask]-targets[:,mask])**2,axis=1)/variance)
        np.testing.assert_allclose(vector@matrix@vector-2*linear@vector+constant,direct,atol=1e-11,rtol=1e-11)


def test_frozen_wiener_phase_decoder_reads_integer_sum_and_reloads(tmp_path):
    _,model,rng=fixture_model();torch.set_num_threads(1)
    model.transfer={b:v.reshape(3001,1,1) for b,v in model.transfer.items()}
    values=rng.normal(size=(1,10,15,400));ids=model.encode(values,'fixture',np.ones((1,10),bool));original=model.decode_ids(ids,'fixture')
    coefficient=np.empty((3001,2,2),complex)
    for output,b in enumerate(('S','O')):coefficient[:,:,output]=np.asarray(model.transfer[b]).reshape(3001,1)
    model.phase_readout={'fixture':coefficient};model.metadata.update(schema='montage_joint_phase_readout_v1',integer_phase_readout='wiener_integer_sum')
    actual=model.decode_ids(ids,'fixture')
    for b in ('S','O'):
        expected=np.fft.irfft(np.fft.rfft(original['sum'].reshape(1,10,6000),norm='ortho')*model.transfer[b].reshape(3001),n=6000,norm='ortho')
        np.testing.assert_array_equal(actual[b].reshape(1,10,6000),expected)
    path=tmp_path/'phase.npz';model.save(path);restored=load_model(path)
    restored.predict_continuous=lambda *_:(_ for _ in ()).throw(AssertionError('phase decoder read X'))
    for b in ('S','O','sum'):np.testing.assert_array_equal(actual[b],restored.decode_ids(ids,'fixture')[b])


def test_cross_electrode_integer_phase_readout_and_strong_ridge(tmp_path):
    from src.tokenizers.montage_joint_tokenizer import lowrank_spatial_residual
    _,model,rng=fixture_model();torch.set_num_threads(1)
    ids=model.encode(rng.normal(size=(1,10,15,400)),'fixture',np.ones((1,10),bool))
    coefficient=np.tile(np.eye(20,dtype=complex),(22,1,1));coefficient[:,3,0]+=.25j
    model.phase_readout={'fixture':coefficient};model.metadata.update(schema='montage_joint_phase_readout_v1',integer_phase_readout='cross_electrode_complex_residual',phase_bin_hz=2)
    result=model.decode_ids(ids,'fixture');path=tmp_path/'cross.npz';model.save(path);restored=load_model(path)
    restored.predict_continuous=lambda *_:(_ for _ in ()).throw(AssertionError('cross decoder read X'))
    for b in ('S','O','sum'):np.testing.assert_array_equal(result[b],restored.decode_ids(ids,'fixture')[b])
    z=rng.normal(size=(100,6))+1j*rng.normal(size=(100,6));z[:,5]=z[:,0];cov=z.conj().T@z/100;cross=cov@np.eye(6)
    left,right,info=lowrank_spatial_residual(cov,cross,np.zeros((6,6)),4,10.)
    assert np.isfinite(left@right).all() and info['retained_regularized_condition']<2 and info['effective_rank']<4
