"""Independent binary transport and masked denominator checks without measured data."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('cross_review',Path(__file__).parents[1]/'experiments/scripts/audit_cross_branch_sampling.py')
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)


def test_independent_13bit_accumulator_matches_bit_definition():
    values=np.arange(5116,dtype=np.uint16)
    payload=review.pack_words(values)
    expected=np.packbits(((values[:,None]>>np.arange(13))&1).astype(np.uint8).reshape(-1),bitorder='little').tobytes()
    assert payload==expected
    np.testing.assert_array_equal(review.unpack_words(payload,values.shape),values)
    for bad in ([5116],[-1],[1.]):
        with pytest.raises(ValueError):review.pack_words(np.array(bad))
    with pytest.raises(ValueError):review.unpack_words(b'\xff\x1f',(1,))
    with pytest.raises(ValueError):review.unpack_words(b'\x00\x80',(1,))


def test_physical_masked_MSE_uses_fixed_target_variance_and_original_units():
    batch={k:np.full((1,15,400),v) for k,v in [('x',2.),('s',.5),('o',1.5)]}
    batch['mask']=np.ones((1,15,400),bool);batch['mask'][0,0,0]=False
    for k,v in [('x',4.),('s',.25),('o',2.25)]:batch['v'+k]=np.full((1,15,400),v)
    prediction={k:np.full((1,15,400),v) for k,v in [('sum',4.),('S',1.),('O',3.)]}
    for p in prediction.values():p[0,0,0]=1e10
    score=review.physical_metrics(prediction,batch,np.array([3.]))
    np.testing.assert_array_equal(score['nmse'],np.ones((1,3,1)))
    np.testing.assert_array_equal(score['rmse_native'][0,:,0],[6.,1.5,4.5])


def test_pure_word_decode_closes_sum_and_rejects_float_IDs():
    from src.tokenizers.cross_branch_sampling import decode_words
    wave=np.sin(np.arange(400)*2*np.pi/80).astype(np.float32)[None]
    books={b:[dict(balanced_atoms=wave,balanced_gain=np.ones(1)) for _ in range(5)] for b in 'SO'}
    sent={'S':np.array([[1]*15,[1024]*15],np.uint16),'O':np.array([[1024]*15,[1]*15],np.uint16)}
    cov=np.repeat(np.eye(4)[None],5,axis=0);cov[:,:2,2:]=np.eye(2)*.3;cov[:,2:,:2]=np.eye(2)*.3
    result=decode_words(sent,books,cov,np.ones((5,4)),np.full((5,4),.1))
    np.testing.assert_array_equal(result['sum'],result['S']+result['O'])
    assert all(v.shape==(2,15,400) and np.isfinite(v).all() for v in result.values())
    with pytest.raises(ValueError):decode_words({b:v.astype(float) for b,v in sent.items()},books,cov,np.ones((5,4)),np.zeros((5,4)))


def test_independent_metrics_match_original_contract_on_masked_random_waveforms():
    from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics
    rng=np.random.default_rng(41);batch={k:rng.normal(size=(3,15,400)).astype(np.float32) for k in ('x','s','o')}
    batch['mask']=rng.uniform(size=(3,15,400))>.15
    for i,k in enumerate(('x','s','o')):
        batch['v'+k]=np.full((3,15,400),.7+i);batch['dv'+k]=np.full((3,15,400),.1+i)
    pred={k:rng.normal(size=(3,15,400)) for k in ('sum','S','O')};scale=np.array([1.3,2.8,.43])
    expected=joint_reconstruction_metrics(pred,batch,1,scale)['metrics']
    for k,v in review.physical_metrics(pred,batch,scale).items():np.testing.assert_array_equal(v,expected[k])
