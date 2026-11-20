"""A consumer gets a nominal ID stream, with optional metered waveform bytes."""
import hashlib
import json

import joblib
import numpy as np
import pytest

from src.tokenizers.observation_baseline import ObservationBaseline
from src.tokenizers.factorized_observation import FactorizedObservationTokenizer
from src.tokenizers.gain_shape_observation import GainShapeObservation
from src.tokenizers.adaptive_observation_detail import AdaptiveObservationDetail


def mock_bundle(root):
    rng=np.random.default_rng(6);scales={};models={}; xout={}; files=[]
    for mod,c,t,b in [('eeg',1,400,256),('hb',2,20,32)]:
        x=rng.normal(size=(40,c,t)); r=FactorizedObservationTokenizer(mod, seed=17).fit(x); xout[mod]=x[:3]
        model=(GainShapeObservation if mod=='eeg' else AdaptiveObservationDetail)(r,b).fit(x)
        md=models[mod]={'router':mod+'r.joblib','model':mod+'m.joblib','total_bits_per_patch':b};scales[mod]={'d':{'p':{'scale':2}}}
        for k,obj in [('router',r),('model',model)]: joblib.dump(obj,root/md[k]);files.append(md[k])
    (root/'training_x_scales.json').write_text(json.dumps(scales));files.append('training_x_scales.json')
    m=dict(schema='observation_baseline_bundle_v1',modalities=models,sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files})
    (root/'manifest.json').write_text(json.dumps(m));return xout


def test_observation_stream_and_wire(tmp_path):
    x=mock_bundle(tmp_path); b=ObservationBaseline(tmp_path)
    for mod,z in x.items():
        ids=b.encode(z,mod); c=b.encode(z,mod,detail=True);np.testing.assert_array_equal(ids['pattern'],c['pattern'])
        np.testing.assert_array_equal(b.scale_g0(z,mod,'d','p'),z.astype('float32')/2)
        for high,codes in [(False,ids),(True,c)]:
            assert all(a.dtype.kind in 'iu' for a in codes.values()); w=b.pack(codes,mod,detail=high)
            assert len(w)==(len(z)*(b.manifest['modalities'][mod]['total_bits_per_patch'] if high else 4)+7)//8
            np.testing.assert_array_equal(b.decode(b.unpack(w,len(z),mod,detail=high),mod,detail=high), b.decode(codes,mod,detail=high))
    with pytest.raises(ValueError): b.encode(next(iter(x.values())),'wrong')
    with pytest.raises(ValueError): b.encode(np.full((1,2,20),np.nan),'hb')


def test_damaged_bundle_rejected(tmp_path):
    mock_bundle(tmp_path); (tmp_path/'eegr.joblib').write_bytes(b'damaged')
    with pytest.raises(ValueError,match='integrity'):ObservationBaseline(tmp_path)
