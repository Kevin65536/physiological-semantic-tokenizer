"""Transport tests independent of measured cache and fitted words."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('band_replay',Path(__file__).parents[1]/'experiments/scripts/audit_band_exchange_replay.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_13bit_exact_values_and_cross_byte_boundaries():
    values=np.array([[0,1,1023,1024,2046,4095,5115],[5115,0,5114,13,26,52,104]],np.uint16)
    payload=module.pack_ids(values)
    assert len(payload)==(values.size*13+7)//8
    np.testing.assert_array_equal(module.unpack_ids(payload,values.shape),values)
    assert payload[:2]==bytes([0,32])  # second word's low bit begins at bit13


def test_13bit_reserved_words_length_and_padding_rejected():
    for value in ([-1],[5116],[8191],[.0]):
        with pytest.raises(ValueError):module.pack_ids(np.asarray(value))
    with pytest.raises(ValueError):module.unpack_ids(b'\xff\x1f',(1,))
    with pytest.raises(ValueError):module.unpack_ids(b'\x00\x80',(1,))
    with pytest.raises(ValueError):module.unpack_ids(b'\x00',(1,))


def test_13bit_all_alphabet_and_empty_roundtrip():
    x=np.arange(5116,dtype=np.uint16)
    np.testing.assert_array_equal(module.unpack_ids(module.pack_ids(x),x.shape),x)
    np.testing.assert_array_equal(module.unpack_ids(module.pack_ids(x[:0]),(0,)),x[:0])


def test_allocated_decoder_source_receives_only_integer_reconstruction():
    from src.tokenizers.band_exchange_words import decode_native
    atom=np.sin(np.arange(400)*2*np.pi/40).astype(np.float32)[None]
    books={b:[dict(balanced_atoms=atom,balanced_gain=np.ones(1,np.float32)) for _ in range(5)] for b in 'SO'}
    cov={b:np.repeat(np.eye(2)[None],5,axis=0) for b in 'SO'}
    ids={b:np.array([np.ones(15),np.zeros(15)],np.uint16) for b in 'SO'}
    class Source:
        def predict_continuous(self,x,dataset,observed):
            np.testing.assert_array_equal(x[0],energy['sum'])
            np.testing.assert_array_equal(observed,[[True,False]])
            return {'S':x*.25}
    energy=decode_native(ids,'synthetic',books,cov,Source(),'balanced','energy')
    allocated=decode_native(ids,'synthetic',books,cov,Source(),'balanced','allocated')
    np.testing.assert_array_equal(energy['sum'],allocated['sum'])
    np.testing.assert_array_equal(allocated['S']+allocated['O'],allocated['sum'])


def test_resumed_resource_budget_includes_prior_service(monkeypatch):
    import subprocess
    monkeypatch.setenv('BAND_REPLAY_UNIT','synthetic-unit')
    monkeypatch.setenv('BAND_REPLAY_PRIOR_CPU','604.525')
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**k:'280000000000')
    with pytest.raises(RuntimeError,match='budget exhausted'):module.budget()
    monkeypatch.setenv('BAND_REPLAY_PRIOR_CPU','100')
    module.budget()
