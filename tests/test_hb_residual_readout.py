import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.tokenizers.hb_residual_readout import HbResidualReadout,apply_readout,features,load_model


class Source:
    silent_ids = dict(S=128,O=256)
    def encode(self,values):
        return {key:np.full((len(values),15),v,dtype=np.int64) for key,v in [('S',1),('O',2)]}
    def decode_ids(self,ids):
        n=len(ids['S']);time=np.arange(300)/10
        s=np.broadcast_to(np.stack([np.sin(time),.4*np.cos(time)],-1),(n,300,2)).copy()
        o=np.broadcast_to(np.stack([.2*np.cos(time),-.1*np.sin(time)],-1),(n,300,2)).copy()
        quiet=(ids['S']==128).all(1)&(ids['O']==256).all(1)
        s[quiet]=[.003,.002];o[quiet]=-s[quiet]
        return dict(S=s.reshape(n,15,40),O=o.reshape(n,15,40),sum=(s+o).reshape(n,15,40))


def fitted():
    return dict(feature_set='paired_so_context',grouping='global',context_lag_samples=5,
                coefficients={'*':[.01,.02,-.01,.03,.02,0.,0.,0.]})


def test_exact_component_color_and_reserved_zero_invariants(tmp_path,monkeypatch):
    source=Source();ids=dict(S=np.array([[1]*15,[128]*15]),O=np.array([[2]*15,[256]*15]))
    before=source.decode_ids(ids)
    model=HbResidualReadout(source,{},fitted());after=model.decode_ids(ids)
    np.testing.assert_array_equal(after['S'],before['S'])
    np.testing.assert_array_equal(after['O'].reshape(2,300,2)[:,:,1],before['O'].reshape(2,300,2)[:,:,1])
    np.testing.assert_array_equal(after['sum'][1],np.zeros((15,40)))
    np.testing.assert_array_equal(after['O'][1],before['O'][1])
    np.testing.assert_allclose(after['S']+after['O'],after['sum'],atol=1e-12,rtol=1e-12)
    assert not np.array_equal(after['sum'][0],before['sum'][0])
    np.testing.assert_array_equal(model.encode(np.ones((2,15,40)))['S'],source.encode(np.ones((2,15,40)))['S'])
    dummy=tmp_path/'base.pt';dummy.write_bytes(b'frozen-source-fixture')
    metadata=dict(schema='hb_x_residual_v1',source_file='base.pt',source_sha256=hashlib.sha256(dummy.read_bytes()).hexdigest())
    checkpoint=tmp_path/'residual.pt';torch.save(dict(metadata=metadata,fitted=fitted()),checkpoint)
    monkeypatch.setattr('src.tokenizers.hb_residual_readout.load_source',lambda _:source)
    for key,value in load_model(checkpoint).decode_ids(ids).items():np.testing.assert_array_equal(value,after[key])
    dummy.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):load_model(checkpoint)


def test_fixed_context_and_dataset_requirements():
    prediction=dict(S=np.ones((2,15,40)),O=-np.ones((2,15,40)),sum=np.zeros((2,15,40)))
    f=features(prediction,'paired_so_context',5)
    np.testing.assert_array_equal(f[:,:,4:],0.)
    params=dict(fitted(),grouping='dataset',coefficients={'known':[0.]*8})
    with pytest.raises(ValueError,match='identities'):HbResidualReadout(Source(),{},params).decode_ids(Source().encode(np.ones((2,15,40))))
    with pytest.raises(ValueError,match='outside'):apply_readout(prediction,[False,False],['other','other'],params)
    config=yaml.safe_load((Path(__file__).resolve().parents[1]/'experiments/configs/physiology_semantic_tokenizer/joint_hb_x_residual_v1.yaml').read_text())
    assert json.loads(json.dumps(config))==config
    assert config['resources']['total_experiment_CPU_budget_seconds']==2700


def test_training_ridge_recovers_known_masked_residual(monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments/scripts'))
    from fit_hb_x_residual import fit_linear
    rng=np.random.default_rng(71)
    f=rng.normal(size=(12,300,2));beta=np.array([.13,-.07])
    x=np.zeros((12,300,2));x[:,:,0]=f@beta
    mask=np.ones_like(x,dtype=bool);mask[:,::7]=False
    batch=dict(x=x.reshape(12,15,40),prediction_sum=np.zeros((12,15,40)),mask=mask.reshape(12,15,40),vx=np.ones((12,1,2)))
    rows=[dict(dataset='d',subject=str(i//2),record=str(i),parent=str(i)) for i in range(12)]
    examples=np.array([(i,0,0) for i in range(12)])
    coefficient,_=fit_linear(f,batch,examples,rows,'global',1e-9,np.arange(12))
    np.testing.assert_allclose(coefficient['*'],beta,rtol=1e-7)
