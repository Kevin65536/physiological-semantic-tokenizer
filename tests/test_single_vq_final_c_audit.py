"""New context-pairing and emitted-ID replay invariants."""
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/scripts'))
from single_vq_final_c_probes import context_pairs
from run_single_vq_final_c_audit import CONTRACT


def test_context_requires_disjoint_complete_same_record_and_site():
    rows=[dict(dataset='d',subject='p',record='r',positions=['F3','F4'],
               native_start_sample=0,native_rate_hz=200)]
    ex=np.array([[0,0,0],[0,29,0],[0,30,0],[0,30,1],[0,90,1]])
    valid=np.ones((len(ex),15),bool)
    valid[-1,0]=False
    pairs,den=context_pairs(ex,valid,rows)
    assert sorted(map(tuple,pairs))==[(0,2),(2,0)]
    assert den==[dict(dataset='d',subject='p',complete=4,eligible=2,used=2)]


def test_c_contract_json_roundtrip_and_tensor_shape():
    import json
    config=yaml.safe_load(CONTRACT.read_text())
    assert json.loads(json.dumps(config))==config and 'null' in config
    assert config['models']['eeg']==dict(d=64,K=8192,seeds=[17,43],arm='C')
    assert config['shape']==dict(eeg=[15,400],hb=[15,40],token_seconds=2)
    assert config['scope']['confirmation']==config['scope']['test']=='forbidden'
