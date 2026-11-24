import numpy as np
import pytest
from src.tokenizers.stratified_source_dictionary import (
    fit_tree,route,fit_book,assign_book,StratifiedSourceDictionary,load_model,solve_readout,pack_ids,unpack_ids,
)


def test_tree_routing_is_hard_deterministic_and_balanced_on_training_only():
    rng=np.random.default_rng(8);features=rng.normal(size=(4096,3))
    tree,groups=fit_tree(features)
    assert len(groups)==8 and max(map(len,groups))-min(map(len,groups))<=2
    ids=route(features,tree)
    for i,indices in enumerate(groups):np.testing.assert_array_equal(ids[indices],i)
    np.testing.assert_array_equal(route(features[:20],tree),ids[:20])


def test_real_final_ID_is_a_single_frozen_atom_and_integer_transport_replays(tmp_path):
    rng=np.random.default_rng(4);t=np.arange(400)/200
    q=rng.uniform(.3,3,(1200,1))*np.sin(2*np.pi*rng.choice([6,10],1200)[:,None]*t+rng.uniform(0,6.28,(1200,1)))
    book=fit_book(q,0,max_iter=2)
    ids=assign_book(q,0,book)
    assert ids.min()>=1 and ids.max()<=511 and book['atoms'].shape==(512,400)
    # Each integer directly selects one immutable waveform; no group/phase/amplitude side stream.
    books={(b,k):fit_book(q,k,max_iter=1) for b in 'SO' for k in range(5)}
    metadata=dict(schema='stratified_source_dictionary_v1',codebook_size=512,positions={'fixture':list(map(str,range(10)))})
    weights={('fixture',b,k):rng.normal(size=(4,10)) for b in 'SO' for k in range(5)}
    model=StratifiedSourceDictionary(metadata,books,weights)
    integers={b:rng.integers(0,512,(1,10,15),dtype=np.int16) for b in 'SO'}
    integers['S'][:,0]=0
    np.testing.assert_array_equal(model.atom_carriers(integers,'fixture')['S'][:,0],0)
    model.save(tmp_path/'model.npz');replay=load_model(tmp_path/'model.npz')
    for b,v in model.decode_ids(integers,'fixture').items():np.testing.assert_array_equal(replay.decode_ids(integers,'fixture')[b],v)
    for b in 'SO':np.testing.assert_array_equal(unpack_ids(pack_ids(integers[b].reshape(-1)),integers[b].size),integers[b].reshape(-1))
    with pytest.raises(ValueError):model.decode_ids({b:v.astype(float) for b,v in integers.items()},'fixture')
    with pytest.raises(ValueError):unpack_ids(b'\xff',1)


def test_joint_true_S_O_X_readout_matches_explicit_target_metric():
    rng=np.random.default_rng(2);x=rng.normal(size=(500,4));truth=rng.normal(size=(500,3));variance=np.array([[.3,2.,1.]])
    gram=x.T@x/500;cross=x.T@truth/500
    weights=solve_readout(gram[None],cross[None],variance,0)[:,0]
    vs,vo,vx=variance[0];h=np.array([[1/vs+1/vx,1/vx],[1/vx,1/vo+1/vx]])
    grad=gram@weights@h-np.column_stack((cross[:,0]/vs+cross[:,2]/vx,cross[:,1]/vo+cross[:,2]/vx))
    np.testing.assert_allclose(grad,0,atol=1e-10)


def test_experiment_contract_rejects_nontrain_before_any_data():
    from pathlib import Path
    import yaml
    from experiments.scripts.run_stratified_source_atoms import validate
    cfg=yaml.safe_load((Path(__file__).parents[1]/'experiments/configs/physiology_semantic_tokenizer/dual_component_stratified_source_atoms_v1.yaml').read_text())
    validate(cfg);cfg['data']['allowed_splits'].append('validation')
    with pytest.raises(ValueError):validate(cfg)
