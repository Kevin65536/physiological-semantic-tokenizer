from pathlib import Path
import sys
import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/scripts'))
import run_single_vq_capacity as suite
from run_dual_component_tokenizer import _base_training_model


@pytest.mark.parametrize('mod,width,k', [('eeg',400,4096),('hb',40,1024)])
def test_capacity_contract_and_single_id_replay(mod,width,k,monkeypatch):
    source=suite.ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml'
    monkeypatch.setattr(suite,'configuration',lambda:yaml.safe_load(source.read_text()))
    cfg=suite.resolved(mod,128,17)
    assert cfg['data']['allowed_splits']==['train','validation']
    assert cfg['data']['test_access']==cfg['data']['protected_data']=='forbidden'
    assert suite.PLAN['scope']['confirmation']=='forbidden'
    assert cfg['architectures']['waveform']['width']==256
    assert suite.PLAN['interface']['patches']==15
    task=suite.model_task(mod,k)
    model=_base_training_model(cfg,task)
    for branch in model.branches.values():
        q=branch['quantizer']
        assert (q.codebook_size,q.embedding_dim)==(k,128)
        assert q.dead_code_threshold==pytest.approx(.1*64*15/k)
        assert not q.geometry_each_forward
        q.initialized.fill_(True)
    rng=np.random.default_rng(17)
    x=rng.normal(size=(2,15,width)).astype('float32')
    model.eval()
    pred,ids=suite.neural_predict(model,{'x':x})
    assert set(ids)=={'S','O'} and all(v.shape==(2,15) and v.dtype.kind=='i' for v in ids.values())
    assert pred['sum'].shape==x.shape
    result=model(torch.from_numpy(x))
    assert 'effective_rank' not in result['health']['S']
    with pytest.raises(ValueError,match='capacity outside'):
        _base_training_model(cfg,suite.model_task(mod,999))


def test_geometry_matches_dense():
    from src.tokenizers.ema_vector_quantizer import EMAVectorQuantizer
    q=EMAVectorQuantizer(513,16)
    z=torch.nn.functional.normalize(q.codebook,dim=-1)
    cosine=z@z.T
    cosine.fill_diagonal_(-1.)
    torch.testing.assert_close(q.geometry()['nearest_neighbor_cosine'],cosine.max(1).values.mean())


def test_dependency_queue_overlaps_scoring_and_reuses_warmup(tmp_path, monkeypatch):
    from collections import Counter
    from threading import Event
    trained = Event()
    calls = []
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-A,GPU-B')
    monkeypatch.delenv('VQ_RESOURCE_POLICY',raising=False)
    monkeypatch.setattr(suite,'OUT',tmp_path)
    def fake(action,arm,m,d,k,s,device):
        calls.append((action,arm,m,d,k,s))
        if action == 'score':
            if arm == 'original' and m == 'eeg':
                assert trained.wait(15)
            name = suite.candidate(m,d,k,s) if arm=='candidate' else f'{arm}_{m}'
            folder = tmp_path/'evaluation'/name
            suite.write_json(folder/f'{m}_summary.json', dict(selection_score=2 if arm=='original' else 1,
                             S_single_word_gain={'a':1},O_single_word_gain={'a':1}))
            suite.write_json(folder/'completion.json',dict(status='completed'))
        else:
            if action=='train':
                assert suite.training_file('warmup',m,d,k,s,'summary.json').is_file()
                if k: trained.set()
            suite.write_json(suite.training_file(action,m,d,k,s,'summary.json'),dict(status='completed'))
    monkeypatch.setattr(suite,'call',fake)
    suite.controller()
    assert all(v==1 for v in Counter(calls).values())
    assert len([j for j in calls if j[0]=='warmup'])==7
    decision=suite.read_json(tmp_path/'selection.json')
    assert decision['eeg_dimension']==64
    assert decision['replication']==[['eeg',64,1024,43],['hb',64,256,43]]
    before=len(calls)
    suite.controller()
    assert len(calls)==before
