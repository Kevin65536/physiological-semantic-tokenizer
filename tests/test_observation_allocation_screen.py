"""Allocation follow-up must validate the parent support and mask before claiming a pair."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.scripts import run_observation_detail_allocation as a


def test_configuration_scope_and_synthetic_replay(tmp_path):
    c=a.configuration(a.CONFIG); assert c==json.loads(json.dumps(c))
    c['parent']='forbidden/test'; p=tmp_path/'bad.yaml';p.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError): a.configuration(p)
    a.synthetic()


@pytest.mark.parametrize('break_mask',[False,True])
def test_independent_ordered_panel_replay(tmp_path,monkeypatch,break_mask):
    row=SimpleNamespace(dataset='d',subject='01',record='r',parent='p',start_s=0,position='p1',
        modality='hb',split='validation',scale=2.,native_start_sample=41,vx=(1.,2.))
    good=np.arange(15)!=0
    class P:
        rows=[row]
        def read(self,i):return dict(available=True,patch_scorable=good,x=np.zeros((15,20,2)))
    monkeypatch.setattr(a,'build_panel',lambda *args,**kwargs:P())
    cfg={'data':{'g0':str(a.G0_ROOT),'splits':['train','validation'],'raw_reads':False,'test_access':'forbidden','panel':{},'panel_seed':29}}
    (tmp_path/'resolved_config.json').write_text(json.dumps(cfg));r=vars(row).copy()
    r.update(valid_bits='1'*15 if break_mask else ''.join(map(str,good.astype(int))),included=True)
    pd.DataFrame([r]).to_csv(tmp_path/'panel.csv',index=False)
    if break_mask:
        with pytest.raises(ValueError,match='mask'):a.replay_panel(tmp_path)
    else:
        g,_=a.replay_panel(tmp_path); f,x,v=g['hb','validation']; assert f.iloc[0].subject=='01'
        assert f.patch.tolist()==list(range(1,15));assert x.shape==(14,2,20) and v.shape==(14,2)
