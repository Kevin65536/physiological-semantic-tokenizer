"""Integer-only spectral atoms and training-conditional energy calibration."""
import numpy as np
import pytest

from src.tokenizers.spectral_waveform_dictionary import (
    SpectralWaveformDictionary, band_components, calibrated_atoms, load_model, raw_features,
)


def model():
    rng = np.random.default_rng(2)
    transfer = {branch: np.full((3001, 1, 1), gain, dtype=complex)
                for branch, gain in [('S', .6), ('O', .4)]}
    means = {branch: np.zeros(400) for branch in ('S', 'O')}
    bases = {branch: np.eye(400)[:8] for branch in ('S', 'O')}
    atoms = {branch: rng.normal(size=(4, 400)) for branch in ('S', 'O')}
    feature_mean = {branch: np.zeros(48) for branch in ('S', 'O')}
    feature_scale = {branch: np.ones(48) for branch in ('S', 'O')}
    centers = {branch: raw_features(atoms[branch], means[branch], bases[branch]) for branch in ('S', 'O')}
    return SpectralWaveformDictionary(dict(schema='spectral_waveform_dictionary_v1', modality='eeg', codebook_size=4),
        transfer, means, bases, atoms, feature_mean, feature_scale, centers)


def test_integer_payload_frozen_reload_and_X_only_continuous_sum(tmp_path):
    fitted = model()
    x = np.random.default_rng(12).normal(size=(2, 15, 400))
    np.testing.assert_allclose(fitted.predict_continuous(x)['sum'], x, atol=1e-12)
    output = fitted(x)
    fitted.save(tmp_path / 'model.npz')
    np.savez(tmp_path / 'ids.npz', **output['ids'])
    with np.load(tmp_path / 'ids.npz') as stored:
        replay = load_model(tmp_path / 'model.npz').decode_ids(dict(stored))
    for branch in ('S', 'O', 'sum'):
        np.testing.assert_array_equal(output[branch], replay[branch])
    with pytest.raises(ValueError, match='integer'):
        fitted.decode_ids({branch: ids.astype(float) for branch, ids in output['ids'].items()})


def test_fixed_training_phases_match_conditional_band_energy():
    time = np.arange(400) / 200
    wave = np.sin(2 * np.pi * 6 * time) + .3 * np.sin(2 * np.pi * 10 * time)
    members = np.stack((wave, -wave, 2 * wave, -2 * wave))
    labels = np.array([0, 0, 1, 1])
    means = np.zeros((2, 400))
    calibrated, energy, fallback = calibrated_atoms(members, labels, means, members[[0, 2]], 2)
    np.testing.assert_allclose(np.mean(band_components(calibrated) ** 2, axis=-1), energy, atol=1e-12)
    assert fallback.all()
    np.testing.assert_allclose(calibrated, members[[0, 2]], atol=1e-12)


def test_quarter_second_energy_features_retain_intra_patch_timing():
    time = np.arange(400) / 200
    values = np.sin(2 * np.pi * 10 * time)[None] * np.r_[np.ones(200), np.full(200, 3.)]
    features = raw_features(values, np.zeros(400), np.eye(400)[:8])
    assert features.shape == (1, 48)
    alpha = features[0, 16:24]
    assert alpha[4:].mean() > alpha[:4].mean()


def test_branch_fit_preserves_training_atoms_and_resumes_exactly(tmp_path):
    from experiments.scripts.run_joint_spectral_dictionary import fit_branch
    from threadpoolctl import threadpool_limits
    rng = np.random.default_rng(5)
    prediction = rng.normal(size=(128, 400))
    target = prediction + .1 * rng.normal(size=prediction.shape)
    cfg = dict(model=dict(waveform_pca_dim=8, codebook_size=4, dynamics_weight=1., waveform_weight=2.),
        training=dict(seed=29, kmeans_n_init=1, kmeans_max_iter=2, kmeans_batch_size=32,
                      kmeans_max_no_improvement=2, kmeans_reassignment_ratio=0.))
    with threadpool_limits(2):
        _, fitted = fit_branch('S', prediction, target, cfg, tmp_path)
        _, resumed = fit_branch('S', prediction, target, cfg, tmp_path)
    assert fitted['training_count'].sum() == len(prediction)
    for key, value in fitted.items():
        np.testing.assert_array_equal(value, resumed[key])
    with pytest.raises(ValueError, match='signature changed'):
        fit_branch('S', prediction, target, cfg, tmp_path, dict(source_sha256='changed'))


def test_training_support_rejects_validation_before_signal_reads(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import yaml
    from experiments.scripts import run_joint_spectral_dictionary as runner
    cfg = yaml.safe_load((runner.ROOT / 'experiments/configs/physiology_semantic_tokenizer/dual_component_joint_spectral_v1.yaml').read_text())
    support = tmp_path / 'support.json'
    support.write_text(json.dumps(dict(seed=29, examples=[[0, 0, 0]] * 4096)))
    cfg['source']['support'] = str(support)
    manifest = dict(rows=[dict(split='validation', modality='eeg')])
    monkeypatch.setattr(runner, 'base_tokenizer_training_data', lambda *args: (manifest, {}, {}, {}))
    monkeypatch.setattr(runner, 'load_source', lambda *args: SimpleNamespace(metadata=dict(input_identity=runner.identity(manifest))))
    monkeypatch.setattr(runner, 'base_tokenizer_batch', lambda *args, **kwargs: pytest.fail('unexpected signal read'))
    with pytest.raises(ValueError, match='train-only EEG support before signal reads'):
        runner.run(cfg, tmp_path / 'out')


def test_semantic_coordinates_match_pattern_statistic_owner():
    from src.analysis.tokenizer_pattern_audit import attributes
    values = np.random.default_rng(2).normal(size=(4, 400))
    coordinates = raw_features(values, np.zeros(400), np.empty((0, 400)), 'theta_alpha_rms')
    statistics, fields = attributes(values, 'eeg')
    expected = np.stack([statistics[:, fields[name][0]:fields[name][1]].mean(-1)
                         for name in ('power_4_8', 'power_8_13', 'rms_envelope')], axis=-1)
    np.testing.assert_allclose(np.exp(coordinates), expected, atol=1e-12)
    joint = raw_features(values,np.zeros(400),np.eye(400)[:8],'semantic_waveform')
    np.testing.assert_array_equal(joint[:,:3],coordinates)
    np.testing.assert_array_equal(joint[:,3:],values[:,:8])


def test_completed_training_preserves_bytes_and_binds_source_identity(tmp_path, monkeypatch):
    import json
    import yaml
    from experiments.scripts import run_joint_spectral_dictionary as runner
    from src.tokenizers.supervised_waveform_dictionary import SupervisedWaveformDictionary
    cfg = yaml.safe_load((runner.ROOT / 'experiments/configs/physiology_semantic_tokenizer/dual_component_joint_spectral_semantic_v2.yaml').read_text())
    cfg['training'].update(windows=2, kmeans_max_iter=2, kmeans_batch_size=16)
    cfg['model']['codebook_size'] = 4
    cfg['source'] = dict(model=str(tmp_path / 'source.npz'), support=str(tmp_path / 'support.json'))
    manifest = dict(cfg_identity='g0', scales_identity='scales', rows=[dict(split='train', modality='eeg')])
    fitted = model()
    fitted.metadata.update(schema='supervised_waveform_dictionary_v1', input_identity=runner.identity(manifest))
    SupervisedWaveformDictionary(fitted.metadata, fitted.transfer, fitted.means, fitted.bases, fitted.atoms).save(cfg['source']['model'])
    (tmp_path / 'support.json').write_text(json.dumps(dict(seed=29, examples=[[0, 0, 0], [0, 0, 0]])))
    monkeypatch.setattr(runner, 'base_tokenizer_training_data', lambda *args: (manifest, {}, {}, {}))
    x = np.random.default_rng(1).normal(size=(2, 15, 400))
    batch = dict(x=x, s=x*.6, o=x*.4, mask=np.ones_like(x, dtype=bool))
    monkeypatch.setattr(runner, 'base_tokenizer_batch', lambda *args, **kwargs: batch)
    out = tmp_path / 'run'
    runner.run(cfg, out)
    paths = [out / 'summary.json', *sorted((out / 'eeg').glob('k*.npz'))]
    before = {str(path): path.read_bytes() for path in paths}
    monkeypatch.setattr(runner, 'base_tokenizer_batch', lambda *args, **kwargs: pytest.fail('completed run read signals'))
    runner.run(cfg, out)
    assert {str(path): path.read_bytes() for path in paths} == before
    with (tmp_path / 'source.npz').open('ab') as stream:
        stream.write(b'changed-source-bytes')
    with pytest.raises(ValueError, match='identity or training support changed'):
        runner.run(cfg, out)


def test_component_CV_uses_training_subjects_and_preserves_completed_evidence(tmp_path, monkeypatch):
    import json
    import yaml
    from experiments.scripts import run_joint_spectral_dictionary as runner
    from src.tokenizers.supervised_waveform_dictionary import SupervisedWaveformDictionary
    cfg=yaml.safe_load((runner.ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_joint_spectral_joint_v3.yaml').read_text())
    cfg['training'].update(windows=12,kmeans_max_iter=2,kmeans_batch_size=64)
    cfg['pareto'].update(codebook_sizes=[4],waveform_weights=[1.],waveform_pca_dim=4)
    cfg['source']=dict(model=str(tmp_path/'source.npz'),support=str(tmp_path/'support.json'))
    rows=[dict(dataset='d',subject=str(n//2),record=str(n),parent=str(n),positions=['p'],split='train',modality='eeg') for n in range(12)]
    manifest=dict(cfg_identity='g0',scales_identity='scales',rows=rows)
    fitted=model();fitted.metadata.update(schema='supervised_waveform_dictionary_v1',input_identity=runner.identity(manifest))
    SupervisedWaveformDictionary(fitted.metadata,fitted.transfer,fitted.means,fitted.bases,fitted.atoms).save(cfg['source']['model'])
    (tmp_path/'support.json').write_text(json.dumps(dict(seed=29,examples=[[n,0,0] for n in range(12)])))
    monkeypatch.setattr(runner,'base_tokenizer_training_data',lambda *args:(manifest,{},{},{}))
    x=np.random.default_rng(12).normal(size=(12,15,400))
    batch=dict(x=x,s=x*.6,o=x*.4,mask=np.ones_like(x,dtype=bool),vx=np.ones_like(x),vs=np.ones_like(x),vo=np.ones_like(x))
    monkeypatch.setattr(runner,'base_tokenizer_batch',lambda *args,**kwargs:batch)
    out=tmp_path/'cv';result=runner.diagnose_pareto(cfg,out)
    assert result['validation_examples_read']==0 and result['confirm_examples_read']==0
    assert len(result['results'])==4
    assert result['results'][0]['heldout_subjects']==1
    before=(out/'summary.json').read_bytes()
    monkeypatch.setattr(runner,'base_tokenizer_batch',lambda *args,**kwargs:pytest.fail('completed CV read signals'))
    assert runner.diagnose_pareto(cfg,out)==result
    assert (out/'summary.json').read_bytes()==before


def test_rejected_duplicate_owner_preserves_running_status(tmp_path, monkeypatch):
    from experiments.scripts import run_joint_spectral_dictionary as runner
    monkeypatch.setattr('sys.argv',['runner','--output',str(tmp_path)])
    monkeypatch.setattr(runner,'run',lambda *args:pytest.fail('duplicate producer ran'))
    with runner.owner_lock(tmp_path):
        with pytest.raises(RuntimeError,match='another owner'):
            runner.main()
    assert not (tmp_path/'failure.json').exists()
