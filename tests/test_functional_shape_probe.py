"""Local conditional shape tokens and forecast probes on synthetic data only."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

import src.tokenizers.functional_tokenizer as tokenizers
import src.analysis.functional_tokenizer as analysis


REGIONS = ['prefrontal', 'left_motor', 'right_motor']
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location(
        'functional_shape_runner', ROOT / 'experiments/scripts/run_functional_tokenizer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def training_patches():
    rng = np.random.default_rng(12)
    observations = rng.normal(size=(96, 8, 2))
    observations += rng.normal(size=(96, 1, 2))
    conditional = .65 * observations + .1 * observations[:, ::-1]
    return observations, conditional, np.arange(64), np.arange(64, 80)


@pytest.fixture(scope='module')
def shape_model():
    observations, conditional, train, validation = training_patches()
    model = tokenizers.ConditionalShapeTokenizer(codebook_size=8, seed=17).fit(
        observations, conditional, train, validation, [.01, 1.])
    return model


def rows_for_trials(trials_per_split=2):
    return [dict(id=f'{split}_{trial}/{region}', trial_id=f'{split}_{trial}',
                 dataset='eeg_fnirs_single_trial', subject=split, record=f'record{trial % 2}',
                 session='session', task='motor_imagery', condition='LMI', split=split,
                 anchor=dict(region=region))
            for split in ('train', 'validation', 'test')
            for trial in range(trials_per_split) for region in REGIONS]


def forecast_cfg():
    return dict(data=dict(regions=REGIONS), evaluation=dict(ridge_alpha=[.01]))


class RestrictedPacket(dict):
    """Fail when a consumer touches fields outside its declared information path."""
    def __getitem__(self, key):
        if key not in self:
            raise AssertionError(f'forbidden packet read: {key}')
        return super().__getitem__(key)


def test_positive_joint_RMS_preserves_signed_relative_chromophore_scale():
    wave = np.array([-3., -2., -1., 0., 0., 1., 2., 3.])
    values = np.stack((wave + 4., -3 * wave - 2.), axis=-1)[None]
    mu, amplitude, shape = tokenizers.ConditionalShapeTokenizer.decompose(values, 1e-6)
    np.testing.assert_allclose(mu, [[4., -2.]])
    np.testing.assert_allclose(amplitude, [np.sqrt(np.mean((values - mu[:, None]) ** 2))])
    assert amplitude[0] > 0
    np.testing.assert_allclose(shape.mean(axis=-2), 0, atol=1e-14)
    np.testing.assert_allclose(np.mean(shape ** 2, axis=(-2, -1)), 1.)
    np.testing.assert_allclose(shape[..., 1], -3 * shape[..., 0])
    np.testing.assert_allclose(mu[:, None] + amplitude[:, None, None] * shape, values)
    neg_mu, neg_amplitude, neg_shape = tokenizers.ConditionalShapeTokenizer.decompose(-values, 1e-6)
    np.testing.assert_allclose(neg_mu, -mu)
    np.testing.assert_allclose(neg_amplitude, amplitude)
    np.testing.assert_allclose(neg_shape, -shape)


def test_flat_patch_has_explicit_floor_and_zero_shape():
    values = np.broadcast_to(np.array([7., -11.]), (4, 8, 2)).copy()
    mu, amplitude, shape = tokenizers.ConditionalShapeTokenizer.decompose(values, 1e-6)
    np.testing.assert_array_equal(mu, np.tile([7., -11.], (4, 1)))
    np.testing.assert_array_equal(amplitude, np.full(4, 1e-6))
    np.testing.assert_array_equal(shape, np.zeros_like(values))
    near_flat = values.copy()
    near_flat[:, 0, 0] += 1e-9
    _, tiny_amplitude, tiny_shape = tokenizers.ConditionalShapeTokenizer.decompose(near_flat, 1e-6)
    np.testing.assert_array_equal(tiny_amplitude, np.full(4, 1e-6))
    np.testing.assert_array_equal(tiny_shape, np.zeros_like(values))


def test_packet_is_local_integer_shape_plus_explicit_physical_sidecars(shape_model):
    observations = np.random.default_rng(5).normal(size=(3, 10, 8, 2))
    packet = shape_model.encode(observations)
    assert set(packet) == {'c_ids', 'r_ids', 'c_mu', 'r_mu', 'c_amplitude', 'r_amplitude'}
    for component in ('c', 'r'):
        assert packet[component + '_ids'].shape == (3, 10)
        assert packet[component + '_ids'].dtype == np.int16
        assert packet[component + '_mu'].shape == (3, 10, 2)
        assert packet[component + '_mu'].dtype == np.float32
        assert packet[component + '_amplitude'].shape == (3, 10)
        assert packet[component + '_amplitude'].dtype == np.float32
        assert np.all(packet[component + '_amplitude'] > 0)
    changed = observations.copy()
    changed[:, 5:] = np.random.default_rng(7).normal(size=(3, 5, 8, 2)) * 1e5
    changed_packet = shape_model.encode(changed)
    for name in packet:
        np.testing.assert_array_equal(packet[name][:, :5], changed_packet[name][:, :5])
    components = shape_model.predict_components(observations)
    changed_components = shape_model.predict_components(changed)
    for name in ('c', 'r'):
        np.testing.assert_array_equal(components[name][:, :5], changed_components[name][:, :5])
    np.testing.assert_array_equal(
        shape_model.shape_features(RestrictedPacket(), 'continuous', components)[:, :5],
        shape_model.shape_features(RestrictedPacket(), 'continuous', changed_components)[:, :5])
    for representation in ('id_only', 'sidecar_only', 'hybrid'):
        original = shape_model.shape_features(packet, representation)
        poisoned = shape_model.shape_features(changed_packet, representation)
        np.testing.assert_array_equal(original[:, :5], poisoned[:, :5])
    # Encoding a single patch agrees with encoding its containing window.
    single = shape_model.encode(observations[1, 4])
    for name in packet:
        np.testing.assert_array_equal(single[name], packet[name][1, 4])


def test_id_only_cannot_access_sidecars_or_original_components(shape_model):
    observations = training_patches()[0][:5]
    packet = shape_model.encode(observations)
    ids = RestrictedPacket({key: value for key, value in packet.items() if key.endswith('_ids')})
    only_ids = shape_model.shape_features(ids, 'id_only')
    expected = shape_model.shape_features(packet, 'id_only')
    np.testing.assert_array_equal(only_ids, expected)
    poison = {key: np.full_like(value, np.nan) if not key.endswith('_ids') else value
              for key, value in packet.items()}
    components = {key: np.full_like(observations, np.nan) for key in ('c', 'r')}
    np.testing.assert_array_equal(shape_model.shape_features(poison, 'id_only', components), expected)


def test_sidecar_only_cannot_access_ids_and_continuous_requires_components(shape_model):
    observations = training_patches()[0][:5]
    packet = shape_model.encode(observations)
    sidecars = RestrictedPacket({key: value for key, value in packet.items() if not key.endswith('_ids')})
    expected = shape_model.shape_features(packet, 'sidecar_only')
    np.testing.assert_array_equal(shape_model.shape_features(sidecars, 'sidecar_only'), expected)
    poisoned_ids = dict(packet, c_ids=np.full((5,), -999), r_ids=np.full((5,), -999))
    np.testing.assert_array_equal(shape_model.shape_features(poisoned_ids, 'sidecar_only'), expected)
    components = shape_model.predict_components(observations)
    continuous = shape_model.shape_features(RestrictedPacket(), 'continuous', components)
    np.testing.assert_allclose(continuous, np.concatenate(
        [components[name].reshape(5, -1) for name in ('c', 'r')], axis=-1))
    with pytest.raises(ValueError):
        shape_model.shape_features(packet, 'continuous')


def test_component_decomposition_reconstructs_observations_and_reload_preserves_export(shape_model, tmp_path):
    observations = training_patches()[0][80:]
    components = shape_model.predict_components(observations)
    assert set(components) == {'c', 'r'}
    np.testing.assert_allclose(components['c'] + components['r'], observations, atol=1e-12)
    packet = shape_model.encode(observations)
    decoded = shape_model.decode(packet)
    dictionary_shapes = shape_model.shape_features(packet, 'id_only').reshape(len(observations), 2, 8, 2)
    for index, component in enumerate(('c', 'r')):
        assert decoded[component].shape == observations.shape
        assert np.isfinite(decoded[component]).all()
        np.testing.assert_allclose(decoded[component].mean(axis=-2), packet[component + '_mu'], atol=1e-6)
        expected = (packet[component + '_mu'][:, None] +
                    packet[component + '_amplitude'][:, None, None] * dictionary_shapes[:, index])
        np.testing.assert_array_equal(decoded[component], expected)
    reconstructed_features = np.concatenate(
        [decoded[component].reshape(len(observations), -1) for component in ('c', 'r')], axis=-1)
    np.testing.assert_array_equal(shape_model.shape_features(packet, 'hybrid'), reconstructed_features)
    arrays = shape_model.artifact_arrays()
    assert arrays and all(np.asarray(value).dtype.kind in 'biuf' for value in arrays.values())
    path = tmp_path / 'shape_model.npz'
    np.savez(path, **arrays)
    with np.load(path, allow_pickle=False) as exported:
        restored = tokenizers.ConditionalShapeTokenizer.from_artifact({key: exported[key] for key in exported.files})
    for name, value in packet.items():
        np.testing.assert_array_equal(restored.encode(observations)[name], value)
    for representation in ('id_only', 'sidecar_only', 'hybrid'):
        np.testing.assert_array_equal(restored.shape_features(packet, representation),
                                      shape_model.shape_features(packet, representation))
    for component in ('c', 'r'):
        np.testing.assert_array_equal(restored.decode(packet)[component], decoded[component])


@pytest.mark.parametrize('corruption', ['float_ids', 'negative_ids', 'large_ids', 'zero_amplitude',
                                        'nan_mu', 'misaligned_sidecars'])
def test_packet_consumers_reject_corruption_on_their_own_information_path(shape_model, corruption):
    packet = shape_model.encode(training_patches()[0][:5])
    representation = 'id_only'
    if corruption == 'float_ids':
        packet['c_ids'] = packet['c_ids'].astype(float)
    elif corruption == 'negative_ids':
        packet['c_ids'][0] = -1
    elif corruption == 'large_ids':
        packet['r_ids'][0] = 8
    else:
        representation = 'sidecar_only'
        if corruption == 'zero_amplitude':
            packet['r_amplitude'][0] = 0
        elif corruption == 'nan_mu':
            packet['c_mu'][0, 0] = np.nan
        else:
            packet['c_mu'] = packet['c_mu'][:-1]
    with pytest.raises(ValueError):
        shape_model.shape_features(packet, representation)


def test_single_channel_EEG_uses_the_same_shape_and_explicit_scale_contract():
    observations, conditional, train, validation = training_patches()
    observations, conditional = observations[..., :1], conditional[..., :1]
    model = tokenizers.ConditionalShapeTokenizer(codebook_size=4, seed=17).fit(
        observations, conditional, train, validation, [.01])
    packet = model.encode(observations[:5])
    assert packet['c_ids'].shape == (5,) and packet['c_mu'].shape == (5, 1)
    assert model.shape_features(packet, 'id_only').shape == (5, 16)
    assert model.shape_features(packet, 'sidecar_only').shape == (5, 4)
    assert model.shape_features(packet, 'hybrid').shape == (5, 16)
    assert model.decode(packet)['c'].shape == (5, 8, 1)


def test_fit_rejects_overlapping_train_validation_patches():
    observations, conditional, train, _ = training_patches()
    with pytest.raises(ValueError, match='disjoint'):
        tokenizers.ConditionalShapeTokenizer(codebook_size=4, seed=17).fit(
            observations, conditional, train, np.array([0, 65, 66]), [.01])


def test_separate_numeric_hard_and_scale_exports_reload_with_identity_join(shape_model, tmp_path):
    observations = training_patches()[0][:20].reshape(2, 10, 8, 2)
    packet = shape_model.encode(observations)
    artifacts = shape_model.artifact_arrays()
    identities = np.array(['second', 'first'])
    hard = {key: packet[key] for key in ('c_ids', 'r_ids')}
    hard.update({key: artifacts[key] for key in ('c_centers', 'r_centers')})
    scales = {key: value for key, value in packet.items() if not key.endswith('_ids')}
    np.savez(tmp_path / 'hard.npz', identities=identities, **hard)
    np.savez(tmp_path / 'scales.npz', identities=identities, **scales)
    for filename, reader, representation in (
            ('hard.npz', analysis.read_shape_tokens, 'id_only'),
            ('scales.npz', analysis.read_shape_sidecars, 'sidecar_only')):
        loaded_ids, features = reader(tmp_path / filename)
        joined = analysis.join_export(['first', 'second'], loaded_ids, {'features': features})['features']
        np.testing.assert_array_equal(joined, shape_model.shape_features(packet, representation)[::-1])


@pytest.mark.parametrize('side_channel', ['c_mu', 'c_amplitude', 'continuous', 'ridge_coefficients'])
def test_on_disk_shape_hard_consumer_rejects_any_non_id_channel(shape_model, tmp_path, side_channel):
    packet = shape_model.encode(training_patches()[0][:20].reshape(2, 10, 8, 2))
    artifacts = shape_model.artifact_arrays()
    hard = {key: packet[key] for key in ('c_ids', 'r_ids')}
    hard.update({key: artifacts[key] for key in ('c_centers', 'r_centers')})
    hard[side_channel] = np.zeros((2, 10))
    path = tmp_path / 'contaminated_hard.npz'
    np.savez(path, identities=np.array(['one', 'two']), **hard)
    with pytest.raises(ValueError, match='only IDs'):
        analysis.read_shape_tokens(path)


def test_on_disk_scale_consumer_rejects_ids(shape_model, tmp_path):
    packet = shape_model.encode(training_patches()[0][:20].reshape(2, 10, 8, 2))
    scales = {key: value for key, value in packet.items() if not key.endswith('_ids')}
    path = tmp_path / 'contaminated_scales.npz'
    np.savez(path, identities=np.array(['one', 'two']), c_ids=packet['c_ids'], **scales)
    with pytest.raises(ValueError, match='only means'):
        analysis.read_shape_sidecars(path)


def test_fitting_and_codebooks_ignore_test_teacher_and_test_observation_values(shape_model):
    observations, conditional, train, validation = training_patches()
    observations[80:] *= 1e10
    conditional[80:] = -1e15
    poisoned = tokenizers.ConditionalShapeTokenizer(codebook_size=8, seed=17).fit(
        observations, conditional, train, validation, [.01, 1.])
    baseline = shape_model.artifact_arrays()
    modified = poisoned.artifact_arrays()
    assert set(baseline) == set(modified)
    for name in baseline:
        np.testing.assert_array_equal(baseline[name], modified[name], err_msg=name)


def test_matched_smooth_inverse_undoes_training_scales_and_correct_X_T_order():
    rng = np.random.default_rng(21)
    original = rng.normal(size=(4, 80, 3))
    original[..., 1] *= 3
    original[..., 2] *= .2
    coordinates = np.stack((original[..., 0], .35 * original[..., 1] - .65 * original[..., 2],
                            original[..., 1] + original[..., 2]), axis=-1)
    manifest = dict(target_mean=[7., -3., 2.], target_scale=[2., 5., 11.])
    standardized = (coordinates - np.array(manifest['target_mean'])) / np.array(manifest['target_scale'])
    recovered = analysis.shape_smooth_coordinates(standardized, manifest)
    np.testing.assert_allclose(recovered, original, atol=1e-12)


@pytest.mark.parametrize('context', ['own_current', 'own_history', 'spatial_history'])
def test_forecast_context_uses_current_and_past_interior_patches_without_crossing_windows(context, monkeypatch):
    rows = rows_for_trials()
    phase = np.arange(10, dtype=float)
    features = np.array([1000 * row + phase for row in range(len(rows))])[..., None]
    hbo = np.array([10000 * row + phase ** 2 for row in range(len(rows))])
    hbr = np.array([-5000 * row + 3 * phase ** 2 for row in range(len(rows))])
    hb = np.repeat(np.stack((hbo, hbr), axis=-1)[:, :, None, :], 8, axis=2)
    captured = []
    original = analysis.relation_patch_readout

    def capture(meta, x, target, cfg, names):
        captured.append((meta, x.copy(), target.copy(), names))
        return original(meta, x, target, cfg, names)

    monkeypatch.setattr(analysis, 'relation_patch_readout', capture)
    analysis.shape_forecast_readout(rows, features, hb, forecast_cfg(), context)
    assert len(captured) == 3
    row_index = {row['id']: i for i, row in enumerate(rows)}
    trial_region = {(row['trial_id'], row['anchor']['region']): i for i, row in enumerate(rows)}
    for meta, x, target, names in captured:
        assert {row['anchor']['region'] for row in meta} <= set(REGIONS)
        assert len({row['anchor']['region'] for row in meta}) == 1
        indices = np.array([row_index[row['id']] for row in meta])
        expected = features[indices, 1:9]
        if context == 'own_history':
            expected = np.concatenate((features[indices, :8], expected), axis=-1)
        elif context == 'spatial_history':
            by_region = []
            for region in REGIONS:
                region_indices = [trial_region[(row['trial_id'], region)] for row in meta]
                by_region.extend((features[region_indices, :8], features[region_indices, 1:9]))
            expected = np.concatenate(by_region, axis=-1)
        np.testing.assert_array_equal(x, expected)
        coordinate = np.stack((.35 * hbo[indices] - .65 * hbr[indices], hbo[indices] + hbr[indices]), axis=-1)
        np.testing.assert_allclose(target, coordinate[:, 2:10] - coordinate[:, 1:9])
        assert names == ['X', 'T'] and x.shape[1] == target.shape[1] == 8


def lagged_cross_region_fixture():
    rows = rows_for_trials(18)
    rng = np.random.default_rng(83)
    features = rng.normal(size=(len(rows), 10, 2))
    hb = np.zeros((len(rows), 10, 8, 2))
    for offset in range(0, len(rows), 3):
        for region in range(3):
            own = features[offset + region]
            other = features[offset + (region + 1) % 3]
            delta_X = own[1:, 0] + .5 * own[:-1, 1] + 2.5 * other[:-1, 0]
            delta_T = .7 * own[1:, 1] - 1.2 * own[:-1, 0] + 1.5 * other[:-1, 1]
            X = np.zeros(10)
            T = np.zeros(10)
            # d[t] is visible-history -> next-patch change for current t=1..8.
            X[2:] = np.cumsum(delta_X[:-1])
            T[2:] = np.cumsum(delta_T[:-1])
            hbo, hbr = X + .65 * T, .35 * T - X
            hb[offset + region] = np.repeat(np.stack((hbo, hbr), axis=-1)[:, None], 8, axis=1)
    return rows, features, hb


def test_spatial_history_recovers_known_cross_region_lagged_increment():
    rows, features, hb = lagged_cross_region_fixture()
    scores = {context: analysis.shape_forecast_readout(rows, features, hb, forecast_cfg(), context)
              for context in ('own_current', 'own_history', 'spatial_history')}
    summary = {context: next(r for r in records if r['coordinate'] == 'coordinate_mean')
               for context, records in scores.items()}
    assert summary['spatial_history']['NRMSE'] < .002
    assert summary['spatial_history']['NRMSE'] < .01 * summary['own_history']['NRMSE']
    assert summary['own_history']['NRMSE'] < summary['own_current']['NRMSE']
    assert summary['spatial_history']['NRMSE'] < summary['spatial_history']['persistence_NRMSE']
    for score in summary.values():
        assert score['subject'] == 'test'
        assert score['trials'] == 18 and score['records'] == 2
        assert score['patches_per_regional_window'] == 8
        assert set(score['selected_alpha_by_region']) == set(REGIONS)


def test_forecast_target_scale_and_alpha_ignore_future_test_extremes():
    rows, features, hb = lagged_cross_region_fixture()
    before = analysis.shape_forecast_readout(rows, features, hb, forecast_cfg(), 'spatial_history')
    test = np.array([row['split'] == 'test' for row in rows])
    features[test] *= 1e8
    hb[test] *= 1e12
    after = analysis.shape_forecast_readout(rows, features, hb, forecast_cfg(), 'spatial_history')
    for original, poisoned in zip(before, after):
        assert original['selected_alpha_by_region'] == poisoned['selected_alpha_by_region']
        assert original['target_training_SD_by_region'] == poisoned['target_training_SD_by_region']


def test_spatial_forecast_joins_regions_by_native_identity_instead_of_array_order():
    rows, features, hb = lagged_cross_region_fixture()
    original = analysis.shape_forecast_readout(rows, features, hb, forecast_cfg(), 'spatial_history')
    order = np.random.default_rng(104).permutation(len(rows))
    shuffled = analysis.shape_forecast_readout([rows[i] for i in order], features[order], hb[order],
                                               forecast_cfg(), 'spatial_history')
    for before, after in zip(original, shuffled):
        assert (before['subject'], before['coordinate']) == (after['subject'], after['coordinate'])
        np.testing.assert_allclose(before['MSE'], after['MSE'], rtol=1e-8, atol=1e-14)
        assert before['trials'] == after['trials'] == 18


def test_forecast_rejects_incomplete_regions_and_subject_split_overlap():
    rows, features, hb = lagged_cross_region_fixture()
    with pytest.raises(ValueError):
        analysis.shape_forecast_readout(rows[:-1], features[:-1], hb[:-1], forecast_cfg(), 'spatial_history')
    invalid = deepcopy(rows)
    for row in invalid:
        row['subject'] = 'same'
    with pytest.raises(ValueError, match='subjects overlap'):
        analysis.shape_forecast_readout(invalid, features, hb, forecast_cfg(), 'own_history')


@pytest.fixture
def synthetic_parent(tmp_path, runner):
    cfg = yaml.safe_load((ROOT / 'experiments/configs/physiology_semantic_tokenizer/functional_shape_probe_v1.yaml').read_text())
    parent_cfg = yaml.safe_load((ROOT / 'experiments/configs/physiology_semantic_tokenizer/functional_tokenizer_v1.yaml').read_text())
    parent, out = tmp_path / 'parent', tmp_path / 'outputs/probe'
    cfg['parent_run'], cfg['artifact_root'] = str(parent), str(out.parent)
    cfg['evaluation']['ridge_alpha'] = [.1]
    rows = rows_for_trials(4)
    manifest = dict(schema='functional_training_cache_v1', identity='temporary-shape-fixture',
                    rows=rows, target_mean=[0., 0., 0.], target_scale=[1., 1., 1.],
                    planned_trials=13, retained_trials=12,
                    failed_trials=[dict(trial_id='unavailable', reason='synthetic incomplete trial')])
    runner.write(parent / 'summary.json', dict(status='completed'))
    runner.write(parent / 'resolved_config.json', dict(config=parent_cfg))
    runner.write(parent / 'cache/manifest.json', manifest)
    observations = np.random.default_rng(53).normal(size=(len(rows), 80, 3))
    physical = .7 * observations
    smooth = np.stack((physical[..., 0], .35 * physical[..., 1] - .65 * physical[..., 2],
                       physical[..., 1] + physical[..., 2]), axis=-1)
    for name, array in dict(u=observations, physical=physical, smooth=smooth,
                            hb=observations[..., 1:].reshape(len(rows), 10, 16)).items():
        np.save(parent / 'cache' / (name + '.npy'), array)
    return cfg, parent, out, manifest


@pytest.mark.parametrize('pilot', [False, True])
def test_shape_runner_executes_frozen_synthetic_task_set_and_preserves_parent(synthetic_parent, runner, monkeypatch, pilot):
    cfg, parent, out, manifest = synthetic_parent
    before = {path.relative_to(parent): path.read_bytes() for path in parent.rglob('*') if path.is_file()}

    def inline_tasks(output, stage, jobs, worker, workers):
        results = {}
        for key, payload in jobs:
            result = worker(payload)
            runner.write(output / 'tasks' / stage / (key + '.json'), result)
            results[key] = result
        return results

    monkeypatch.setattr(runner, 'bounded_tasks', inline_tasks)
    result = runner.run_shape_probe(cfg, parent, out, workers=1, pilot=pilot)
    count = 1 if pilot else 6
    assert result['status'] == 'completed' and not result['failures']
    assert result['planned_full_tasks'] == 6
    assert result['attempted_tasks'] == result['completed_tasks'] == count
    assert result['parent_support'] == {key: manifest[key] for key in ('planned_trials', 'retained_trials', 'failed_trials')}
    assert result['success_subset_only'] is True and result['full_denominator_claim'] is False
    assert (out / 'resolved_config.json').is_file()
    assert len(list((out / 'tasks/shape_probe').glob('*.json'))) == count
    expected = dict(hard={'c_ids', 'r_ids', 'c_centers', 'r_centers', 'identities'},
                    sidecars={'c_mu', 'r_mu', 'c_amplitude', 'r_amplitude', 'identities'},
                    continuous={'c', 'r', 'identities'})
    for task in sorted((out / 'models').iterdir()):
        for modality in ('eeg', 'hb'):
            for filename, fields in expected.items():
                with np.load(task / modality / (filename + '.npz'), allow_pickle=False) as arrays:
                    assert set(arrays.files) == fields
            with np.load(task / modality / 'tokenizer.npz', allow_pickle=False) as arrays:
                assert arrays['codebook_size'] == 16
                assert arrays['training_patches'] >= 16
    for score in result['forecast_scores']:
        assert score['MSE'] == pytest.approx(np.mean([r['MSE'] for r in score['region_scores'].values()]))
        assert score['NRMSE'] ** 2 == pytest.approx(score['MSE'])
        if score['representation'] == 'persistence':
            assert score['selected_alpha_by_region'] is None
            assert score['MSE'] == score['persistence_MSE']
            for region_score in score['region_scores'].values():
                assert region_score['MSE'] == region_score['persistence_MSE']
                assert region_score['NRMSE'] == region_score['persistence_NRMSE']
                assert region_score['ridge_alpha'] is None
    assert before == {path.relative_to(parent): path.read_bytes() for path in parent.rglob('*') if path.is_file()}
    # The relationship stage consumes the completed, frozen producer artifacts.
    shape_before = {path.relative_to(out): path.read_bytes() for path in out.rglob('*') if path.is_file()}
    relation_cfg = yaml.safe_load((ROOT / 'experiments/configs/physiology_semantic_tokenizer/functional_shape_relations_v1.yaml').read_text())
    relation_cfg.update(parent_run=str(parent), shape_parent_run=str(out), artifact_root=str(out.parent))
    relation_cfg['evaluation']['ridge_alpha'] = [.1]
    relation_out = out.parent / 'relationships'
    relationships = runner.run_shape_relations(relation_cfg, parent, out, relation_out, 1, pilot=pilot)
    assert relationships['status'] == 'completed' and not relationships['failures']
    assert relationships['planned_full_tasks'] == 6
    assert relationships['attempted_tasks'] == relationships['completed_tasks'] == count
    assert relationships['common_support_consistent'] is True
    assert relationships['repeated_direct_baseline_consistent'] is True
    assert relationships['common_support']['planned_trials'] == relationships['common_support']['eligible_trials'] == 12
    assert {score['variant'] for score in relationships['forecast_scores']} == {'real', 'wrong_previous', 'wrong_other_ROIs'}
    for path in (relation_out / 'tasks/shape_relations').glob('*.json'):
        record = runner.read(path)
        assert record['encoder_fit'] is record['teacher_read'] is record['raw_data_read'] is False
    assert shape_before == {path.relative_to(out): path.read_bytes() for path in out.rglob('*') if path.is_file()}
    assert before == {path.relative_to(parent): path.read_bytes() for path in parent.rglob('*') if path.is_file()}


def valid_relation_donors(rows):
    meta, _ = analysis.relation_trial_layout(rows, REGIONS)
    groups = {}
    strata = ('dataset', 'subject', 'record', 'condition', 'split')
    for index, row in enumerate(meta):
        groups.setdefault(tuple(row[key] for key in strata), []).append(index)
    donors = np.empty((len(meta), 3), int)
    for group in groups.values():
        assert len(group) >= 2
        for position, index in enumerate(group):
            for region in range(3):
                donors[index, region] = group[(position + 1 + region % (len(group) - 1)) % len(group)]
    return meta, donors


@pytest.mark.parametrize('context,perturbation', [('own_history', 'wrong_previous'),
                                                 ('spatial_history', 'wrong_other_ROIs')])
def test_relationship_surrogates_preserve_target_and_phase_and_share_other_ROI_donor(context, perturbation, monkeypatch):
    rows = rows_for_trials(6)
    meta, donors = valid_relation_donors(rows)
    features = np.array([1000 * row + np.arange(10) for row in range(len(rows))], float)[..., None]
    values = np.stack((features[..., 0] ** 2, -2 * features[..., 0] ** 2), axis=-1)
    hb = np.repeat(values[:, :, None], 8, axis=2)
    before_features, before_hb = features.copy(), hb.copy()
    captures = []
    original = analysis.relation_patch_readout

    def capture(region_rows, x, y, cfg, names):
        captures.append((region_rows, x.copy(), y.copy()))
        return original(region_rows, x, y, cfg, names)

    monkeypatch.setattr(analysis, 'relation_patch_readout', capture)
    analysis.shape_forecast_readout(rows, features, hb, forecast_cfg(), context,
                                    donors=donors, perturbation=perturbation)
    row_lookup = {(row['trial_id'], row['anchor']['region']): index for index, row in enumerate(rows)}
    trial_lookup = {row['trial_id']: index for index, row in enumerate(meta)}
    assert len(captures) == 3
    for region_rows, x, y in captures:
        target_region = region_rows[0]['anchor']['region']
        region_index = REGIONS.index(target_region)
        expected = []
        for row in region_rows:
            trial_index = trial_lookup[row['trial_id']]
            donor_trial = meta[donors[trial_index, region_index]]['trial_id']
            own = row_lookup[(row['trial_id'], target_region)]
            if perturbation == 'wrong_previous':
                donor = row_lookup[(donor_trial, target_region)]
                expected.append(np.concatenate((features[donor, :8], features[own, 1:9]), axis=-1))
            else:
                parts = []
                for region in REGIONS:
                    source_trial = row['trial_id'] if region == target_region else donor_trial
                    index = row_lookup[(source_trial, region)]
                    parts.extend((features[index, :8], features[index, 1:9]))
                expected.append(np.concatenate(parts, axis=-1))
        np.testing.assert_array_equal(x, np.stack(expected))
        take = [row_lookup[(row['trial_id'], target_region)] for row in region_rows]
        coordinate = np.stack((.35 * values[take, :, 0] - .65 * values[take, :, 1],
                                values[take].sum(axis=-1)), axis=-1)
        np.testing.assert_allclose(y, coordinate[:, 2:10] - coordinate[:, 1:9])
    np.testing.assert_array_equal(features, before_features)
    np.testing.assert_array_equal(hb, before_hb)


@pytest.mark.parametrize('violation', ['self', 'other_subject_split'])
def test_relationship_readout_rejects_self_or_cross_stratum_donor(violation):
    rows = rows_for_trials(6)
    meta, donors = valid_relation_donors(rows)
    donors[0, 0] = 0 if violation == 'self' else next(
        index for index, row in enumerate(meta) if row['split'] != meta[0]['split'])
    with pytest.raises(ValueError):
        analysis.shape_forecast_readout(rows, np.zeros((len(rows), 10, 1)),
            np.zeros((len(rows), 10, 8, 2)), forecast_cfg(), 'own_history',
            donors=donors, perturbation='wrong_previous')


def test_all_relationship_variants_share_eligibility_targets_and_fixed_cross_model_donors(monkeypatch):
    rows = rows_for_trials(4)
    singleton = deepcopy(rows[-3:])
    for row in singleton:
        row.update(id='singleton/' + row['anchor']['region'], trial_id='singleton', record='singleton')
    rows.extend(singleton)
    rng = np.random.default_rng(30)
    features = rng.normal(size=(len(rows), 10, 2))
    hb = rng.normal(size=(len(rows), 10, 8, 2))
    calls = []

    def capture(selected, x, target, cfg, context, **kwargs):
        calls.append((selected, x.copy(), target.copy(), context, kwargs))
        return []

    monkeypatch.setattr(analysis, 'shape_forecast_readout', capture)
    cfg = forecast_cfg()
    cfg['seed'] = 20261025
    _, failures, support = analysis.shape_relationship_probes(
        rows, {'hybrid': features, 'direct_observation': features * 2}, hb, cfg)
    assert not failures and len(calls) == 8
    assert (support['planned_trials'], support['eligible_trials'], support['excluded_trials']) == (13, 12, 1)
    assert 'singleton' not in support['eligible_trial_ids']
    assert support['interpretation'] == 'descriptive_relationship_perturbation_not_randomization_or_causal_test'
    populations = [[row['id'] for row in call[0]] for call in calls]
    assert all(population == populations[0] for population in populations)
    expected_target = {row['id']: hb[index] for index, row in enumerate(rows)}
    donor_maps = []
    for selected, _, target, _, kwargs in calls:
        np.testing.assert_array_equal(target, np.stack([expected_target[row['id']] for row in selected]))
        if kwargs.get('perturbation'):
            donor_maps.append(kwargs['donors'])
    assert len(donor_maps) == 4
    for donor_map in donor_maps[1:]:
        np.testing.assert_array_equal(donor_map, donor_maps[0])
    native, _ = analysis.relation_trial_layout(calls[0][0], REGIONS)
    assert support['eligible_trial_ids'] == [row['trial_id'] for row in native]
    for target, row in enumerate(native):
        for region, donor in enumerate(donor_maps[0][target]):
            assert row['trial_id'] != native[donor]['trial_id']
            assert support['donor_trial_ids'][target][region] == native[donor]['trial_id']
            assert all(row[key] == native[donor][key]
                       for key in ('dataset', 'subject', 'record', 'condition', 'split'))
    # The two feature representations keep the same trial identity and differ only by the declared scaling.
    first_model = {context + str(kwargs.get('perturbation')): x for _, x, _, context, kwargs in calls[:4]}
    second_model = {context + str(kwargs.get('perturbation')): x for _, x, _, context, kwargs in calls[4:]}
    assert first_model.keys() == second_model.keys()
    for key in first_model:
        np.testing.assert_array_equal(second_model[key], 2 * first_model[key])


def test_true_temporal_and_spatial_dependencies_outperform_wrong_pairs():
    rows, features, hb = lagged_cross_region_fixture()
    cfg = forecast_cfg()
    cfg['seed'] = 20261025
    scores, failures, support = analysis.shape_relationship_probes(rows, {'synthetic': features}, hb, cfg)
    assert not failures
    assert support['planned_trials'] == support['eligible_trials'] == 54
    outcomes = {(score['context'], score['variant']): score['NRMSE']
                for score in scores if score['coordinate'] == 'coordinate_mean'}
    assert outcomes[('own_history', 'real')] < outcomes[('own_history', 'wrong_previous')]
    assert outcomes[('spatial_history', 'real')] < .01 * outcomes[('spatial_history', 'wrong_other_ROIs')]
    for score in scores:
        assert score['MSE'] == pytest.approx(np.mean([region['MSE'] for region in score['region_scores'].values()]))


def test_persistence_schema_changes_every_score_level_without_mutating_direct_scores():
    rows, features, hb = lagged_cross_region_fixture()
    direct = analysis.shape_forecast_readout(rows, features, hb, forecast_cfg(), 'own_current')
    before = deepcopy(direct)
    persistence = analysis.shape_persistence_scores(direct)
    assert direct == before
    assert len(persistence) == len(direct)
    for score in persistence:
        assert score['MSE'] == score['persistence_MSE']
        assert score['NRMSE'] == score['persistence_NRMSE']
        assert score['selected_alpha_by_region'] is None
        for nested in score['region_scores'].values():
            assert nested['MSE'] == nested['persistence_MSE']
            assert nested['NRMSE'] == nested['persistence_NRMSE']
            assert nested['ridge_alpha'] is None
        assert score['MSE'] == pytest.approx(np.mean([nested['MSE'] for nested in score['region_scores'].values()]))
