import numpy as np
import json
import pytest

from experiments.build_clean_eeg_fnirs_cache import _pair_single_trial_wavelengths


def test_public_subjects_have_matching_signal_event_geometry_coverage(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments import build_clean_eeg_fnirs_cache as signal
    from experiments import build_clean_event_index as event
    from experiments import build_clean_channel_geometry as geometry

    for modality in ('EEG', 'NIRS'):
        for subject in (1, 23, 24, 29):
            folder = tmp_path / f'{modality}_01-29' / f'subject {subject:02d}'
            folder.mkdir(parents=True)
            (folder / 'mnt.mat').touch()
    session = SimpleNamespace(x=np.ones((100, 2)), fs=10.,
                              clab=['CH1lowWL', 'CH1highWL'], yUnit='V')
    monkeypatch.setattr(signal, '_mat_payload', lambda *args: np.array([session], dtype=object))
    marker_reads = []
    def empty_markers(path, key):
        marker_reads.append(path)
        return np.array([], dtype=object)
    monkeypatch.setattr(event, '_mat_payload', empty_markers)
    monkeypatch.setattr(geometry, '_rel', str)
    monkeypatch.setattr(geometry, 'records_from_mnt',
                        lambda path, **kwargs: [(kwargs['subject'], kwargs['modality'])])

    signals = list(signal.iter_single_trial(tmp_path, 29, 6, 0))
    event.iter_single_trial(tmp_path, 29, 6)
    positions = list(geometry.iter_single_trial(tmp_path))
    expected = {'subject 01', 'subject 23', 'subject 24', 'subject 29'}
    assert {row.subject for row in signals} == expected
    assert {path.parent.name for path in marker_reads} == expected
    assert set(positions) == {(subject, modality) for subject in expected for modality in ('eeg', 'fnirs')}
    assert len(list(signal.iter_single_trial(tmp_path, 29, 6, 0, subject_ids=[29]))) == 1
    marker_reads.clear()
    event.iter_single_trial(tmp_path, 29, 6, subject_ids=[29])
    assert {path.parent.name for path in marker_reads} == {'subject 29'}
    for subjects in ([0], [30]):
        with pytest.raises(ValueError, match='between 1 and 29'):
            list(signal.iter_single_trial(tmp_path, 29, 6, 0, subject_ids=subjects))
        with pytest.raises(ValueError, match='between 1 and 29'):
            event.iter_single_trial(tmp_path, 29, 6, subject_ids=subjects)


def test_single_trial_homer2_pair_labels_drop_wavelength_suffixes():
    values = np.arange(20, dtype=np.float64).reshape(5, 4)
    labels = ["AF7Fp1lowWL", "AF3Fp1lowWL", "AF7Fp1highWL", "AF3Fp1highWL"]

    paired, pair_labels = _pair_single_trial_wavelengths(values, labels)

    assert paired.shape == (5, 2, 2)
    assert pair_labels == ("AF7Fp1", "AF3Fp1")


def test_bounded_simultaneous_signal_and_event_builders_select_same_record(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from experiments import build_clean_eeg_fnirs_cache as signal
    from experiments import build_clean_event_index as event
    for modality in ('EEG', 'NIRS'):
        folder = tmp_path/f'VP001-{modality}'
        folder.mkdir()
        for task in ('nback', 'dsr', 'wg'):
            (folder/f'mrk_{task}.mat').touch()
            (folder/f'cnt_{task}.mat').touch()
    chromophore = SimpleNamespace(x=np.ones((100, 2)), fs=10., clab=['CH1', 'CH2'], yUnit='mmol/L')
    monkeypatch.setattr(signal, '_mat_payload', lambda _: SimpleNamespace(oxy=chromophore, deoxy=chromophore))
    monkeypatch.setattr(event, '_mat_payload', lambda _: None)
    monkeypatch.setattr(event, '_rel', str)
    monkeypatch.setattr(event, '_simultaneous_dsr_events', lambda **kw: ([], kw['record_id']))
    records = list(signal.iter_simultaneous(tmp_path, 1, 1, 0))
    _, reports = event.iter_simultaneous(tmp_path, 1, 1)
    assert [r.record_id for r in records] == reports == ['cnt_dsr']


def test_complete_measurement_cache_roundtrip_never_reopens_native(tmp_path, monkeypatch):
    from experiments import build_clean_eeg_fnirs_cache as builder
    import src.data.unified_physiology as unified
    from src.data.clean_physiology_cache import MEASUREMENT_CACHE_STORAGE, CLEAN_CACHE_SCHEMA
    from src.data.event_alignment import EVENT_ALIGNMENT_SCHEMA
    from src.data.fnirs_standardization import DATASET_FNIRS_CONTRACTS
    native_path = tmp_path/'source.mat'
    native_path.write_bytes(b'synthetic source identity')
    t = np.arange(1000)/10
    hb = np.column_stack((.004*np.sin(t), -.001*np.sin(t)))
    record = builder.CleanInputRecord(dataset_id='refed', subject='1', record_id='video_1_hbo_hbr',
        source_paths=(native_path,), values=hb, homer2_input=hb, sample_rate_hz=10.,
        contract=DATASET_FNIRS_CONTRACTS['refed']['hbo_hbr'], entry_stage='chromophore', wavelengths_nm=(),
        channel_names=('CH1_HbO','CH1_HbR'), homer2_channel_names=('CH1_HbO','CH1_HbR'),
        metadata={'metadata_unit':'unknown'})
    eeg = np.sin(np.arange(20000)[:,None]/200*2*np.pi*10)*np.arange(1,7)[None,:]
    monkeypatch.setattr(unified,'load_native_eeg_record',lambda *a: unified.NativeEEGRecord(
        eeg,200.,tuple(f'E{i}' for i in range(6)),'uV',native_path,unit_evidence='synthetic'))
    root = tmp_path/'cache'
    row = builder.build_record(record, root, False, storage=MEASUREMENT_CACHE_STORAGE)
    (root/'cache_manifest.json').write_text(json.dumps(dict(schema=CLEAN_CACHE_SCHEMA,records=[row])))
    (root/'event_index').mkdir()
    (root/'event_index/event_manifest.json').write_text(json.dumps(dict(event_alignment_schema=EVENT_ALIGNMENT_SCHEMA)))
    monkeypatch.setattr(unified,'load_native_eeg_record',lambda *a: (_ for _ in ()).throw(AssertionError('native reread')))
    ds = unified.UnifiedPhysiologyWindowDataset(root,dataset_ids=['refed'])
    payload = ds._load_canonical_record(ds.index.records[0])
    assert isinstance(payload['eeg'],np.memmap) and isinstance(payload['fnirs'],np.memmap)
    assert payload['eeg'].dtype == payload['fnirs'].dtype == np.float64
    np.testing.assert_allclose(np.std(payload['fnirs'],axis=0)[0]/np.std(payload['fnirs'],axis=0)[1],4.,rtol=1e-8)
    assert set(row['arrays']) == {'eeg','fnirs','eeg_supported_channels','fnirs_supported_channels','eeg_bad_channel_mask'}


def test_bounded_parallel_builder_preserves_record_identity(tmp_path):
    from experiments import build_clean_eeg_fnirs_cache as builder
    from src.data.fnirs_standardization import DATASET_FNIRS_CONTRACTS
    source=tmp_path/'synthetic_source';source.write_bytes(b'fixture')
    values=np.column_stack((np.sin(np.arange(1000)/10),-.25*np.sin(np.arange(1000)/10)))
    records=[builder.CleanInputRecord(dataset_id='refed',subject=str(i),record_id='video_1_hbo_hbr',
        source_paths=(source,),values=values,homer2_input=values,sample_rate_hz=10.,
        contract=DATASET_FNIRS_CONTRACTS['refed']['hbo_hbr'],entry_stage='chromophore',wavelengths_nm=(),
        channel_names=('CH1_HbO','CH1_HbR'),homer2_channel_names=('CH1_HbO','CH1_HbR'),metadata={}) for i in (1,2)]
    results=list(builder.build_records(iter(records),tmp_path/'out',workers=2,storage='legacy_npz'))
    assert {r['join_key'] for r,_ in results} == {'refed|1|video_1','refed|2|video_1'}
    for row,_ in results:
        with np.load(row['record_npz']) as arrays:
            assert arrays['homer2_aligned_fnirs'].dtype == np.float64
