"""Small, data-free portable-state and four-branch public-interface checks."""
import hashlib
import json

import numpy as np
import pytest

from experiments.scripts.apply_joint_tokenizer import apply
from src.tokenizers.band_exchange_words import BandExchangeEEG, load_model
from src.tokenizers.eeg_transient_gain import BoundedTransientSource
from src.tokenizers.supervised_waveform_dictionary import SupervisedWaveformDictionary


def test_portable_native_model_and_four_branch_units(tmp_path):
    wave=np.sin(np.arange(400)*2*np.pi/100).astype(np.float32)
    book=dict(mean=np.zeros(400,np.float32),basis=np.ones((1,400),np.float32)/20,
              wave_scale=np.float32(1),property_mean=np.zeros(3,np.float32),property_scale=np.ones(3,np.float32),
              balanced_centers=np.array([[0,0,0,0],[1,1,1,1]],np.float32),
              balanced_atoms=np.stack([wave,2*wave]),balanced_gain=np.ones(2,np.float32))
    meta=dict(schema='band_exchange_eeg_v1',modality='eeg',positions={'study':['A','B']},
              layout='exchange',assignment='balanced',synthesis='energy',codebook_size=5116)
    beta=np.zeros((5,2,13));beta[:,:,0]=1
    smeta=dict(schema='eeg_bounded_transient_source_v1',positions={'study':['A','B']},transient=True,gain_bounds=[0,3])
    source=BoundedTransientSource({b:np.ones(3001)*.5 for b in 'SO'},
        {'study':dict(background=np.ones((5,2)),beta=beta)},smeta)
    model=BandExchangeEEG(meta,{b:[book]*5 for b in 'SO'},
        {'study':{b:np.repeat(np.eye(2)[None],5,axis=0) for b in 'SO'}},
        {'study':source},{'study':{b:np.ones(2) for b in 'SO'}})
    path=tmp_path/'eeg.npz';model.save(path);loaded=load_model(path)
    x=np.tile(wave,15).reshape(1,1,15,400).repeat(2,axis=1);seen=np.array([[True,False]])
    ids=model.encode(x,'study',seen);new=loaded.encode(x,'study',seen)
    for b in 'SO':np.testing.assert_array_equal(ids[b],new[b])
    for b,v in model.decode_ids(ids,'study').items():
        np.testing.assert_array_equal(v,loaded.decode_ids(ids,'study')[b])
    with pytest.raises(ValueError):loaded.decode_ids({b:v.astype(float) for b,v in ids.items()},'study')
    z=np.arange(40,dtype=float)
    hb=SupervisedWaveformDictionary(dict(schema='supervised_waveform_dictionary_v1',modality='hb',codebook_size=2),
        {b:np.zeros((151,2,2)) for b in 'SO'}, {b:np.zeros(40) for b in 'SO'},
        {b:np.eye(1,40) for b in 'SO'}, {b:np.stack([z,z+1]) for b in 'SO'})
    hp=tmp_path/'hb.npz';hb.save(hp)
    scales=tmp_path/'scales.json';scales.write_text(json.dumps({'positions':{
        'eeg':{'study':{'A':{'scale':3},'B':{'scale':4}}},'hb':{'study':{'H':{'scale':2}}}}}))
    def entry(p):return dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    bundle=tmp_path/'bundle.json';bundle.write_text(json.dumps(dict(schema='joint_tokenizer_inference_v1',
        eeg=dict(kind='band_exchange_eeg',**entry(path)),hb=dict(kind='dictionary',**entry(hp)),scales=entry(scales))))
    raw=dict(eeg_dataset=np.array(['study','study']),eeg_position=np.array(['B','A']),eeg_crop=np.array([0,0]),
        eeg_observed=np.array([False,True]),eeg_x=np.stack([np.full((6000,1),np.nan),x[0,0].reshape(6000,1)*3]),
        hb_dataset=np.array(['study']),hb_position=np.array(['H']),hb_x=np.zeros((1,300,2),np.float32))
    encoded=apply(bundle,'encode',raw);decoded=apply(bundle,'decode',encoded)
    native=loaded.decode_ids(ids,'study')
    for b in ('S','O','sum'):
        np.testing.assert_array_equal(decoded['eeg_'+b][1,:,0],native[b][0,0].reshape(6000)*3)
    with pytest.raises(ValueError,match='exactly'):
        apply(bundle,'decode',dict(encoded,window_scale=np.ones(1)))
