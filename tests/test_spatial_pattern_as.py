"""AS pattern evidence must use full-montage integers and exact saved identities."""
from pathlib import Path
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest


def owner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments/scripts'))
    return importlib.import_module('audit_spatial_tokenizer_patterns')


def test_as_local_member_keeps_unselected_electrode_context(monkeypatch):
    audit = owner(monkeypatch)
    class Model:
        metadata = dict(encoder_rule='observation_only_analysis_by_synthesis_v1', positions={'fixture':['A','B']})
        def encode(self, values, dataset, observed):
            assert values.shape == (1,2,15,400)
            assert values[0,1].mean() == 9
            return {b: np.full((1,2,15),int(values[0,1].mean()),np.int32) for b in 'SO'}
    fields = {'x':np.column_stack([np.ones(6000),np.full(6000,9.)])}
    monkeypatch.setattr(audit,'base_tokenizer_joint_eeg',lambda *a,**kw:(fields,np.ones(2,bool),np.ones(2,bool)))
    rows=[dict(dataset='fixture',split='train',positions=['A','B'],inputs={'score':[dict(start_s=0,positions=[0,1])]})]
    ids,bands=audit.local_ids(Model(),rows,np.array([[0,0,0]]),{'x':np.ones((1,15,400))},scales={})
    np.testing.assert_array_equal(ids['S'],np.full((1,15),9))
    assert bands.tolist()==[0]


def test_saved_as_screen_ids_map_native_names_and_fail_changed_identity(tmp_path,monkeypatch):
    audit=owner(monkeypatch)
    args=SimpleNamespace(model=tmp_path/'encoder.json')
    metadata=[dict(crop=0,row=0,start_s=30,dataset='fixture',subject='s',native_positions=['B','A'])]
    (tmp_path/'integer_metadata.json').write_text(json.dumps(metadata))
    values=np.array([[[2]*15,[7]*15]],np.int32)
    np.savez_compressed(tmp_path/'integer_ids.npz',S=values)  # Missing native crop key fails closed.
    model=SimpleNamespace(metadata={'positions':{'fixture':['B','A']}},codebook_size=8)
    rows=[dict(dataset='fixture',subject='s',positions=['A','B'])]
    with pytest.raises(KeyError):audit.native_screen_ids(args,rows,model,np.array([[0,30,0]]))
    np.savez_compressed(tmp_path/'integer_ids.npz',S_0=values,O_0=values)
    result,_=audit.native_screen_ids(args,rows,model,np.array([[0,30,0],[0,30,1]]))
    np.testing.assert_array_equal(result['S'][:,0],[7,2])
    with pytest.raises(ValueError,match='exactly cover'):
        audit.native_screen_ids(args,rows,model,np.array([[0,60,0]]))


def observation_fixture(tmp_path):
    values = np.arange(18000 * 2, dtype=np.float32).reshape(18000, 2, 1)
    observed = np.ones(values.shape, bool)
    observed[45 * 200, 1] = False
    np.save(tmp_path / 'x.npy', values)
    np.save(tmp_path / 'observation_valid.npy', observed)
    row = dict(dataset='fixture', split='validation', modality='eeg', positions=['A', 'B'],
        folder=str(tmp_path), inputs={'score': [dict(start_s=s, positions=[0, 1]) for s in [0, 30, 60]]})
    scales = {'positions': {'eeg': {'fixture': {'A': {'scale': 3.1}, 'B': {'scale': 7.3}}}}}
    return [row], scales, values, observed


def test_natural_eeg_reads_actual_added_two_seconds_not_adjacent_parent_flag(tmp_path):
    from src.data.dual_component_tokenizer import base_tokenizer_joint_eeg_context
    rows, scales, values, observed = observation_fixture(tmp_path)
    names = ['B', 'A']
    crop = dict(row=0, start_s=0, positions=[0, 1])
    original, available, raw = base_tokenizer_joint_eeg_context(rows, crop, scales, names)
    shifted, shifted_available, changed = base_tokenizer_joint_eeg_context(rows, crop, scales, names, 1)
    assert shifted_available.all()  # The adjacent parent's missing point is at 45s, outside [2,32).
    np.testing.assert_array_equal(raw['raw_normalized_x'][2800:3200], changed['raw_normalized_x'][2400:2800])
    np.testing.assert_array_equal(shifted[:, 0], values[400:6400, 1, 0] / 7.3)
    assert changed['center_patch'] == 6 and changed['physical_center_start_s'] == 14
    # No S/O/target-mask fixture exists: only observation files are read.
    observed[31 * 200, 1] = False
    np.save(tmp_path / 'observation_valid.npy', observed)
    shifted, shifted_available, changed = base_tokenizer_joint_eeg_context(rows, crop, scales, names, 1)
    assert shifted_available.tolist() == [False, True]
    np.testing.assert_array_equal(shifted[:, 0], 0)
    np.testing.assert_array_equal(raw['raw_normalized_x'][2800:3200], changed['raw_normalized_x'][2400:2800])


def test_natural_eeg_backward_alignment_and_split_guard_before_arrays(tmp_path):
    from src.data.dual_component_tokenizer import base_tokenizer_joint_eeg_context
    rows, scales, _, observed = observation_fixture(tmp_path)
    observed[45 * 200, 1] = True
    observed[0, 1] = False
    np.save(tmp_path / 'observation_valid.npy', observed)
    crop = dict(row=0, start_s=30, positions=[0, 1])
    _, _, original = base_tokenizer_joint_eeg_context(rows, crop, scales, ['A', 'B'])
    _, available, shifted = base_tokenizer_joint_eeg_context(rows, crop, scales, ['A', 'B'], -1)
    assert available.all()
    assert shifted['center_patch'] == 8
    np.testing.assert_array_equal(original['raw_normalized_x'][2800:3200], shifted['raw_normalized_x'][3200:3600])
    rows[0]['split'] = 'test'
    rows[0]['folder'] = '/does/not/exist'
    with pytest.raises(ValueError, match='scope'):
        base_tokenizer_joint_eeg_context(rows, crop, scales, ['A', 'B'], -1)
