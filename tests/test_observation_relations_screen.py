"""No measured reads: scope, local clocks, features and word payload only."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.scripts.run_observation_relations import CONFIG,config
from src.data.observation_relation_inputs import audit_clock,features
from src.analysis.observation_tokenizer_evaluation import KEYS
from src.tokenizers.factorized_observation import FactorizedObservationTokenizer
from src.analysis.observation_word_limits import frozen_word_collisions


def test_declared_word_limits_on_synthetic_dictionaries():
    g=np.random.default_rng(59)
    cs={m:FactorizedObservationTokenizer(m,n_patterns=16,seed=17).fit(g.normal(size=(80,c,t)),partition='train') for m,c,t in [('eeg',1,400),('hb',2,20)]}
    res=frozen_word_collisions(cs)['checks']
    assert res['eeg']['peak_hz']==[8.5,8.5,12.0]
    np.testing.assert_allclose(res['eeg']['waveform_correlation_vs_first'],[1,-1,0],atol=1e-12)
    np.testing.assert_allclose(res['hb']['detrended_color_correlation'],[1,-1],atol=1e-12)


def test_no_test_refit_or_other_clocks(tmp_path):
    import json
    c=config(CONFIG);assert c['k']==16 and c==json.loads(json.dumps(c)) and 'null_hypothesis' in c
    for k,v in [('test_access','unlocked'),('splits',['train','validation','test']),('refit_tokenizer',True),('coupling_primary_datasets',['refed'])]:
        p=tmp_path/'cfg.yaml';p.write_text(yaml.safe_dump(dict(c,**{k:v})))
        with pytest.raises(ValueError):config(p)


def test_native_offsets_not_lags_and_visual_probe_cannot_be_inferred_by_position():
    a=dict(dataset='visual_cognitive_motivation',subject='S07',record='S07',split='validation',modality='hb',position='Probe2:CH18',native_start_sample=2100,native_rate_hz=500)
    members=tuple((f"{a['dataset']}/S07/S07_Probe{i}",'w01',4.2009,25.) for i in (2,1)); a['member_clocks']=members
    assert audit_clock(SimpleNamespace(**a))[1]==pytest.approx(20.7991)
    for bad in [dict(member_clocks=[(*members[0][:3],25.2),members[1]]),dict(position='CH18'),dict(native_start_sample=2098),dict(split='test')]:
        with pytest.raises(ValueError):audit_clock(SimpleNamespace(**dict(a,**bad)))


@pytest.mark.parametrize('mod,channels,t',[('eeg',1,400),('hb',2,20)])
def test_feature_rows_preserve_clock_and_ac_rms(mod,channels,t):
    gen=np.random.default_rng(123);x=gen.normal(size=(64,channels,t)).astype('f');z=x[:8]+2
    c,b=[FactorizedObservationTokenizer(mod,n_patterns=16,seed=17,routing=r).fit(x,partition='train') for r in ('pattern','waveform')]
    frame=pd.DataFrame([dict(zip(KEYS,['d','s','r','p',0,'site']),patch=i) for i in range(8)])
    a=features(frame,z,mod,c,b);np.testing.assert_array_equal(a.pattern,c.encode(z,mode='prototype')['pattern']);np.testing.assert_allclose(a.rms,np.sqrt((z.astype(float).var(axis=-1)).mean(axis=-1)));assert len(a)==8 and set(frame)<=set(a)
    np.testing.assert_allclose(a[['alpha','beta']].to_numpy() if mod=='eeg' else a[['hbo','hbr','so','sr']].to_numpy(),c.pattern_features(z)[:,2:4] if mod=='eeg' else c.pattern_features(z))
    with pytest.raises(ValueError):features(frame.assign(patch=0),z,mod,c,b)
    with pytest.raises(ValueError):features(frame,z[:4],mod,c,b)
