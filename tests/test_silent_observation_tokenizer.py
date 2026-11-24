import numpy as np
import pytest

from src.tokenizers.silent_observation_tokenizer import SilentObservationTokenizer


class Source:
    metadata = {'modality':'hb','codebook_size':4}
    def encode(self,values):
        return {branch:np.ones(values.shape[:2],dtype=np.int64) for branch in ('S','O')}
    def decode_ids(self,ids):
        s=np.repeat(ids['S'][...,None].astype(np.float32),40,axis=-1)*.1
        o=np.repeat(ids['O'][...,None].astype(np.float32),40,axis=-1)*-.08
        return dict(S=s,O=o,sum=s+o)


def model(level=(0.,0.)):
    return SilentObservationTokenizer(Source(),dict(modality='hb',codebook_size=5,base_codebook_size=4,
        quiet_threshold_input_sd=.001,output_dtype='float32'),level)


def test_quiet_decision_is_encoded_and_ordinary_outputs_stay_bitwise_equal():
    tokenizer=model()
    values=np.zeros((2,15,40),np.float32)
    values[1]=1
    ordinary_ids=tokenizer.source.encode(values)
    ordinary=tokenizer.source.decode_ids(ordinary_ids)
    ids=tokenizer.encode(values)
    assert (ids['S'][0]==4).all() and (ids['O'][0]==4).all()
    decoded=tokenizer.decode_ids(ids)
    for key in ('S','O','sum'):
        np.testing.assert_array_equal(decoded[key][1],ordinary[key][1])
        assert decoded[key].dtype==ordinary[key].dtype
        assert not decoded[key][0].any()


def test_train_fitted_levels_form_an_exact_zero_sum_pair():
    tokenizer=model((.2,-.3))
    ids={branch:np.full((1,15),4,np.int32) for branch in ('S','O')}
    decoded=tokenizer.decode_ids(ids)
    np.testing.assert_array_equal(decoded['S'],-decoded['O'])
    np.testing.assert_array_equal(decoded['sum'],decoded['S']+decoded['O'])


def test_decoder_rejects_partial_or_unpaired_reserved_sequences():
    tokenizer=model()
    for left,right in [(np.full((1,15),4),np.zeros((1,15),int)),
                       (np.r_[4,np.zeros(14,int)][None],np.r_[4,np.zeros(14,int)][None])]:
        with pytest.raises(ValueError):
            tokenizer.decode_ids({'S':left,'O':right})


def test_distinct_branch_reserved_atoms_preserve_original_payload():
    tokenizer=SilentObservationTokenizer(Source(),dict(modality='hb',codebook_size=9,base_codebook_size=8,
        base_codebook_sizes=dict(S=4,O=8),branch_codebook_sizes=dict(S=5,O=9),
        quiet_threshold_input_sd=.001,output_dtype='float32'),(.2,-.1))
    observed=np.zeros((2,15,40),np.float32);observed[1]=1
    ordinary=tokenizer.source(observed) if callable(tokenizer.source) else tokenizer.source.decode_ids(tokenizer.source.encode(observed))
    ids=tokenizer.encode(observed)
    assert (ids['S'][0]==4).all() and (ids['O'][0]==8).all()
    result=tokenizer.decode_ids(ids)
    np.testing.assert_array_equal(result['S'][0],-result['O'][0])
    for key in ('S','O','sum'):
        np.testing.assert_array_equal(result[key][1],ordinary[key][1])
