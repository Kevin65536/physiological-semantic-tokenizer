import numpy as np
import pytest

from src.data import window_exclusions as qc
from src.data.dual_component_tokenizer import (
    base_tokenizer_joint_eeg, base_tokenizer_joint_hb, base_tokenizer_batch,
)


@pytest.fixture
def flagged(monkeypatch):
    # Same synchronized 30 s window. Hb record clock is offset by 40 seconds.
    intervals = [('eeg', 'P7', 6000, 12000, 6000),
                 ('hb', 'Probe1:A', 6000, 12000, 14000)]
    monkeypatch.setattr(qc, '_by_record', lambda: {('ds', 'train', 's', 'r'): intervals})


def test_paired_windows_use_separate_clocks_and_share_the_visual_event(flagged):
    def row(eeg, hb, probe):
        return dict(dataset='ds', split='train', subject='s', join_key=probe, dependency_id='shared',
                    eeg_start_s=eeg, hb_start_s=hb, duration_s=2)
    samples = [row(28, 68, 'P1'), row(28, 68.1, 'P2'), row(60, 100, 'P1'),
               row(29, 0, 'P1'), row(27, 65, 'P1')]
    clean = qc.filter_synchronized_samples(samples, {'P1': 'r', 'P2': 'r'})
    assert clean == [samples[2], samples[4]]
    assert not qc.paired_window_excluded('ds', 'validation', 's', 'r', 30, 70, 30)


def test_so_masks_only_the_intersection_and_preserves_other_positions(flagged):
    # Parent starts before the mark and ends within it. Relative [10, 40) clips to [10, 20).
    meta = dict(dataset='ds', split='train', subject='s', EEG_channels=['P7', 'Cz'],
                fNIRS_positions=['Probe1:A', 'Probe2:A'],
                source=dict(native_record='r', start_sample=4000, samples=4000))
    a = {}
    for pre, shape in [('eeg', (4000, 2)), ('fnirs', (200, 2, 2))]:
        for k in ('x', 's', 'o', 'observation_valid', 'teacher_valid'):
            a[pre+'_'+k] = np.ones(shape, dtype=bool if k.endswith('valid') else float)
    qc.apply_to_so(meta, a)
    for k, values in a.items():
        n = len(values)//2
        assert values[:n].all() and not values[n:, 0].any() and values[:, 1].all()


def test_prepared_multichannel_auxiliary_observations_follow_local_flags(tmp_path):
    for modality, rate, colors in [('eeg', 200, 1), ('hb', 10, 2)]:
        for field in ('x', 'observation_valid'):
            np.save(tmp_path/(field+'.npy'), np.ones((30*rate, 2, colors),
                    dtype=bool if field=='observation_valid' else np.float32))
        row = dict(split='train', modality=modality, positions=['A','B'], folder=str(tmp_path),
                   inputs={'score':[dict(start_s=0,positions=[1])]}, qc_spans=[(0, 5999, 6000)])
        scales = {'positions':{modality:{'ds':{n:dict(scale=1) for n in row['positions']}}}}
        row['dataset']='ds'
        crop = dict(row=0,start_s=0,positions=[])
        if modality=='hb':
            data, mask=base_tokenizer_joint_hb([row],crop,scales)
            assert not mask[:,0].any() and mask[:,1].all()
        else:
            fields,mask,_=base_tokenizer_joint_eeg([row],crop,scales,row['positions'])
            data=fields['x']
            assert mask.tolist()==[False,True]
        assert not data[:,0].any() and data[:,1].all()
        with pytest.raises(ValueError, match='outside frozen prepared support'):
            base_tokenizer_batch([row], [(0,0,0)], scales)
        assert qc.blocked_positions([(0,6000,12000)], 0) == set()
