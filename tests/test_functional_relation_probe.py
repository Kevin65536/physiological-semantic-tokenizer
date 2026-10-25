"""Frozen relation probe contracts, using temporary synthetic exports only."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import src.analysis.functional_tokenizer as analysis


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'experiments/configs/physiology_semantic_tokenizer/functional_relation_probe_v1.yaml'
PARENT_CONFIG = ROOT / 'experiments/configs/physiology_semantic_tokenizer/functional_tokenizer_v1.yaml'
REGIONS = ['prefrontal', 'left_motor', 'right_motor']
STRATA = ('dataset', 'subject', 'record', 'condition', 'split')


@pytest.fixture
def cfg():
    result = yaml.safe_load(CONFIG.read_text())
    parent = yaml.safe_load(PARENT_CONFIG.read_text())
    result['data'] = parent['data']
    result['data']['tasks'] = ['motor_imagery']
    result['evaluation']['probe_C'] = [1.0]
    return result


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location(
        'functional_relation_runner', ROOT / 'experiments/scripts/run_functional_tokenizer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_rows(trial, *, split='train', subject=None, record='record', condition='LMI', **extra):
    metadata = dict(trial_id=trial, dataset='eeg_fnirs_single_trial',
                    subject=subject or split, record=record, session='session',
                    task='motor_imagery', condition=condition, split=split)
    metadata.update(extra)
    return [dict(metadata, id=f'{trial}/{region}', anchor=dict(region=region)) for region in REGIONS]


def classification_fixture(task='motor_imagery'):
    labels = ['LMI', 'RMI'] if task == 'motor_imagery' else ['0-back session', '2-back session', '3-back session']
    rows = []
    for split in ('train', 'validation', 'test'):
        for subject in range(2):
            for label in labels:
                for trial in range(3):
                    rows.extend(native_rows(f'{split}_{subject}_{label}_{trial}', split=split,
                                            subject=f'{split}_{subject}', condition=label, task=task))
    signal = np.array([labels.index(r['condition']) - (len(labels) - 1) / 2 for r in rows])
    features = {m: np.broadcast_to(signal[:, None, None] * 8, (len(rows), 10, 2)).copy()
                for m in ('eeg', 'hb')}
    return rows, features


def hard_export(path, count=2):
    ids = np.arange(count * 10).reshape(count, 10).astype(np.int16)
    values = dict(ids_eeg=ids, ids_hb=ids[:, ::-1],
                  codebook_eeg=np.arange(128 * 16).reshape(128, 16).astype(float),
                  codebook_hb=-np.arange(128 * 16).reshape(128, 16).astype(float),
                  identities=np.array([f'row{i}' for i in range(count)][::-1]))
    np.savez(path, **values)
    return values


def test_integer_ids_and_vectors_follow_the_same_identity_reorder(tmp_path):
    path = tmp_path / 'hard.npz'
    exported = hard_export(path)
    identities, vectors, ids = analysis.read_hard_tokens(path, return_ids=True)
    expected = ['row0', 'row1']
    vectors = analysis.join_export(expected, identities, vectors)
    ids = analysis.join_export(expected, identities, ids)
    for modality in ('eeg', 'hb'):
        np.testing.assert_array_equal(ids[modality], exported['ids_' + modality][::-1])
        np.testing.assert_array_equal(vectors[modality], exported['codebook_' + modality][ids[modality]])
        assert ids[modality].dtype.kind in 'iu'
    with pytest.raises(ValueError, match='population mismatch'):
        analysis.join_export(['row0', 'missing'], identities, vectors)


@pytest.mark.parametrize('side_channel', ['eeg_pre', 'soft_assignment', 'observation', 'confidence'])
def test_hard_export_rejects_any_non_id_side_channel(tmp_path, side_channel):
    path = tmp_path / 'hard.npz'
    values = hard_export(path)
    np.savez(path, **values, **{side_channel: np.zeros((2, 10, 16))})
    with pytest.raises(ValueError, match='only IDs'):
        analysis.read_hard_tokens(path, return_ids=True)


@pytest.mark.parametrize('corruption', ['float_ids', 'negative_ids', 'out_of_range', 'duplicate_identity', 'nan_book'])
def test_hard_export_rejects_invalid_ids_or_lookup(tmp_path, corruption):
    path = tmp_path / 'hard.npz'
    values = hard_export(path)
    if corruption == 'float_ids':
        values['ids_eeg'] = values['ids_eeg'].astype(float)
    elif corruption == 'negative_ids':
        values['ids_eeg'][0, 0] = -1
    elif corruption == 'out_of_range':
        values['ids_hb'][0, 0] = 128
    elif corruption == 'duplicate_identity':
        values['identities'][:] = 'same'
    else:
        values['codebook_hb'][0, 0] = np.nan
    np.savez(path, **values)
    with pytest.raises(ValueError):
        analysis.read_hard_tokens(path, return_ids=True)


def test_native_layout_joins_by_identity_with_complete_unique_regions():
    rows = native_rows('b') + native_rows('a')
    rows = [rows[i] for i in [5, 0, 4, 2, 3, 1]]
    metadata, indices = analysis.relation_trial_layout(rows, REGIONS)
    assert [r['trial_id'] for r in metadata] == ['a', 'b']
    for meta, group in zip(metadata, indices):
        assert [rows[i]['anchor']['region'] for i in group] == REGIONS
        assert {rows[i]['trial_id'] for i in group} == {meta['trial_id']}
    for bad in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(ValueError, match='complete unique'):
            analysis.relation_trial_layout(bad, REGIONS)


@pytest.mark.parametrize('field,value', [('subject', 'other'), ('split', 'test'),
                                        ('condition', 'RMI'), ('record', 'other')])
def test_regional_replicas_cannot_disagree_about_native_group(field, value):
    rows = native_rows('a')
    rows[1][field] = value
    with pytest.raises(ValueError, match='metadata disagree'):
        analysis.relation_trial_layout(rows, REGIONS)


def test_patch_targets_use_native_X_T_coordinates_and_eight_sample_means():
    time = np.arange(80, dtype=float)
    values = np.stack((time ** 2, 2 * time + 3, -time + 7), axis=-1)[None]
    target = analysis.relation_patch_targets(values)
    assert target.shape == (1, 10, 3)
    for patch in (0, 4, 9):
        samples = slice(patch * 8, (patch + 1) * 8)
        eeg, hbo, hbr = values[0, samples].mean(axis=0)
        np.testing.assert_allclose(target[0, patch], [eeg, .35 * hbo - .65 * hbr, hbo + hbr])
    with pytest.raises(ValueError):
        analysis.relation_patch_targets(values[:, :-1])
    values[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        analysis.relation_patch_targets(values)


def test_errors_average_regions_trials_records_then_subjects():
    rows = native_rows('a', subject='one', record='short')
    errors = [0., 0., 12.]
    for trial in ('b', 'c', 'd'):
        rows.extend(native_rows(trial, subject='one', record='long'))
        errors.extend([16.] * 3)
    rows.extend(native_rows('e', subject='two'))
    errors.extend([1.] * 3)
    scores = {r['subject']: r for r in analysis.relation_subject_mse(rows, errors)}
    # Record means are 4 and 16, despite one versus three native trials.
    assert scores['one']['MSE'] == 10
    assert scores['one']['NRMSE'] == pytest.approx(np.sqrt(10))
    assert (scores['one']['records'], scores['one']['trials']) == (2, 4)
    assert scores['two']['MSE'] == 1
    assert np.mean([r['MSE'] for r in scores.values()]) == 5.5


def test_patch_ridge_recovers_scaled_coordinates_and_keeps_training_statistics(cfg):
    rows = [r for split in ('train', 'validation', 'test') for trial in range(3)
            for r in native_rows(f'{split}_{trial}', split=split)]
    rng = np.random.default_rng(2)
    x = rng.normal(size=(len(rows), 10, 2))
    y = np.stack((3 * x[..., 0] - 2 * x[..., 1],
                  100 * (x[..., 0] + .1 * x[..., 1])), axis=-1)
    scores = analysis.relation_patch_readout(rows, x, y, cfg, ['X', 'T'])
    assert len(scores) == 3
    assert max(r['NRMSE'] for r in scores) < .01
    assert all(r['NRMSE'] < r['mean_baseline_NRMSE'] for r in scores)
    train = np.array([r['split'] == 'train' for r in rows])
    expected_sd = y[train].reshape(-1, 2).std(0)
    np.testing.assert_allclose(list(scores[0]['target_training_SD'].values()), expected_sd)
    by_coordinate = {r['coordinate']: r for r in scores}
    assert by_coordinate['coordinate_mean']['MSE'] == pytest.approx(
        (by_coordinate['X']['MSE'] + by_coordinate['T']['MSE']) / 2)
    test = np.array([r['split'] == 'test' for r in rows])
    poisoned_x, poisoned_y = x.copy(), y.copy()
    poisoned_x[test] *= 1e8
    poisoned_y[test] += 1e12
    changed = analysis.relation_patch_readout(rows, poisoned_x, poisoned_y, cfg, ['X', 'T'])
    for before, after in zip(scores, changed):
        assert before['target_training_SD'] == after['target_training_SD']
        assert before['ridge_alpha'] == after['ridge_alpha']
        assert before['validation_MSE'] == after['validation_MSE']


def test_patch_ridge_rejects_subject_overlap_across_partitions(cfg):
    rows = [r for split in ('train', 'validation', 'test')
            for r in native_rows(split, split=split, subject='same')]
    with pytest.raises(ValueError, match='subjects overlap'):
        analysis.relation_patch_readout(rows, np.ones((9, 10, 2)),
                                       np.ones((9, 10, 1)), cfg, ['EEG'])


def test_donors_are_distinct_native_trials_in_exact_strata():
    base = native_rows('base')[0]
    meta = []
    for index, changed in enumerate([{}] + [{key: 'different_' + key} for key in STRATA]):
        for trial in range(3):
            meta.append(dict(base, **changed, trial_id=f'{index}_{trial}'))
    meta.append(dict(base, subject='singleton', trial_id='excluded'))
    eligible, spatial, temporal = analysis.relation_null_donors(meta, REGIONS, 31)
    assert eligible.tolist() == list(range(18))
    assert spatial.shape == (18, 3)
    assert temporal.shape == (18, 3, 10)
    kept = [meta[i] for i in eligible]
    for target, row in enumerate(kept):
        for donor in np.concatenate((spatial[target].ravel(), temporal[target].ravel())):
            assert kept[donor]['trial_id'] != row['trial_id']
            assert tuple(kept[donor][k] for k in STRATA) == tuple(row[k] for k in STRATA)
    again = analysis.relation_null_donors(meta, REGIONS, 31)
    for first, second in zip((eligible, spatial, temporal), again):
        np.testing.assert_array_equal(first, second)


def test_feature_blocks_separate_occupancy_order_and_directional_relations():
    phase = np.arange(10, dtype=float)
    values = np.stack((phase, 2 * phase + 1, phase ** 2), axis=0)[None, ..., None]
    ids = np.tile(np.arange(10)[None, None, :], (1, 3, 1)).astype(int)
    blocks = analysis.relation_feature_blocks(values, ids)
    reverse = analysis.relation_feature_blocks(values[:, :, ::-1], ids[:, :, ::-1])
    np.testing.assert_allclose(blocks['marginal'], reverse['marginal'])
    assert blocks['marginal'].shape == (1, 3 * 2 + 3 * 128)
    occupancy = blocks['marginal'][0, 6:].reshape(3, 128)
    np.testing.assert_allclose(occupancy[:, :10], .1)
    assert np.all(occupancy[:, 10:] == 0)
    assert not np.array_equal(blocks['temporal'], reverse['temporal'])
    # The first pair keeps synchronous, left->right and right->left terms.
    np.testing.assert_allclose(blocks['spatial'][0, :3], [
        np.mean(phase * (2 * phase + 1)),
        np.mean(phase[:-1] * (2 * phase[1:] + 1)),
        np.mean((2 * phase[:-1] + 1) * phase[1:])])
    assert blocks['spatial'][0, 1] != blocks['spatial'][0, 2]
    bad_ids = ids.astype(float)
    with pytest.raises(ValueError, match='integer IDs'):
        analysis.relation_feature_blocks(values, bad_ids)


def test_surrogates_keep_phase_and_whole_regions_and_only_replace_relation_blocks(cfg, monkeypatch):
    rows = [r for trial in range(4) for r in native_rows(f't{trial}')]
    rows.extend(native_rows('excluded', subject='singleton'))
    # Every scalar identifies a native trial, region and patch independently.
    values = np.array([[[10000 * (i // 3) + 100 * (i % 3) + phase]
                        for phase in range(10)] for i in range(len(rows))], float)
    block_inputs, probes = [], []
    original = analysis.relation_feature_blocks

    def capture_blocks(z, ids=None, **kwargs):
        block_inputs.append(np.array(z))
        return original(z, ids, **kwargs)

    def capture_probe(meta, x, config, *, seed):
        probes.append((meta, np.array(x)))
        return [], []

    monkeypatch.setattr(analysis, 'relation_feature_blocks', capture_blocks)
    monkeypatch.setattr(analysis, 'classification_probe', capture_probe)
    _, _, support = analysis.relation_classification(rows, {'eeg': values, 'hb': values}, None, cfg, seed=11)
    assert (support['planned_trials'], support['eligible_trials'], support['excluded_trials']) == (5, 4, 1)
    assert len(probes) == 15
    populations = [{r['trial_id'] for r in meta} for meta, _ in probes]
    assert all(population == {'t0', 't1', 't2', 't3'} for population in populations)
    for offset in (0, 3):
        real, temporal, spatial = block_inputs[offset:offset + 3]
        assert real.shape == temporal.shape == spatial.shape == (4, 3, 10, 1)
        for target in range(4):
            for region in range(3):
                # Spatial surrogates copy one entire donor trajectory.
                matches = [donor for donor in range(4)
                           if np.array_equal(spatial[target, region], real[donor, region])]
                assert len(matches) == 1 and matches[0] != target
                # Temporal donor identity may change, but never phase or region.
                for phase in range(10):
                    matches = [donor for donor in range(4)
                               if temporal[target, region, phase, 0] == real[donor, region, phase, 0]]
                    assert len(matches) == 1 and matches[0] != target
    marginal_width = original(block_inputs[0])['marginal'].shape[1]
    temporal_width = original(block_inputs[0])['temporal'].shape[1]
    for offset, width_multiplier in ((0, 1), (5, 1), (10, 2)):
        candidates = [x for _, x in probes[offset:offset + 5]]
        width = marginal_width * width_multiplier
        for candidate in candidates[1:]:
            np.testing.assert_array_equal(candidate[:, :width], candidates[0])
        # The spatial null keeps both the true marginal and true temporal block.
        prefix = width + temporal_width * width_multiplier
        np.testing.assert_array_equal(candidates[4][:, :prefix], candidates[1])


@pytest.mark.parametrize('task', ['motor_imagery', 'nback'])
def test_relationship_classification_recovers_signal_with_native_trial_denominators(cfg, task):
    cfg['data']['tasks'] = [task]
    cfg['evaluation']['probe_C'] = [10.0]
    rows, features = classification_fixture(task)
    records, failures, support = analysis.relation_classification(rows, features, None, cfg, seed=23)
    assert not failures
    class_count = len(cfg['data']['task_labels'][task])
    assert support['planned_trials'] == support['eligible_trials'] == 18 * class_count
    assert support['excluded_trials'] == 0
    assert len(records) == 15 * 2  # modalities x variants x test subjects
    assert all(r['balanced_accuracy'] == 1 for r in records)
    assert all(r['training_labels'] == 6 * class_count and r['validation_labels'] == 6 * class_count for r in records)
    assert all(r['trials'] == 3 * class_count and r['records'] == 1 for r in records)
    assert {r['variant'] for r in records} == set(cfg['relations']['comparisons'])


def test_relationship_classification_rejects_subject_overlap(cfg):
    rows, features = classification_fixture()
    for row in rows:
        row['subject'] = 'same_subject'
    with pytest.raises(ValueError, match='subjects overlap'):
        analysis.relation_classification(rows, features, None, cfg, seed=23)


def test_no_eligible_trials_returns_failure_and_full_planned_denominator(cfg):
    rows = native_rows('one')
    records, failures, support = analysis.relation_classification(
        rows, {'eeg': np.zeros((3, 10, 2)), 'hb': np.zeros((3, 10, 2))}, None, cfg, seed=1)
    assert not records
    assert failures == [dict(reason='no_null_eligible_trials')]
    assert (support['planned_trials'], support['eligible_trials'], support['excluded_trials']) == (1, 0, 1)


def test_missing_training_class_is_failure_for_every_candidate_without_dropping_denominator(cfg):
    rows, features = classification_fixture()
    keep = np.array([i for i, row in enumerate(rows)
                     if row['split'] != 'train' or row['condition'] == 'LMI'])
    rows = [rows[i] for i in keep]
    features = {modality: value[keep] for modality, value in features.items()}
    records, failures, support = analysis.relation_classification(rows, features, None, cfg, seed=1)
    assert not records
    assert len(failures) == 15
    assert all(r['reason'] == 'label_budget_missing_class_or_support' for r in failures)
    assert support['planned_trials'] == support['eligible_trials'] == 30
    assert support['excluded_trials'] == 0


def test_training_projection_unchanged_by_extreme_test_features():
    rng = np.random.default_rng(83)
    x = rng.normal(size=(24, 12))
    training = np.arange(12)
    original, metadata = analysis.fixed_features(x, training, 8, seed=4)
    x[16:] = 1e20
    poisoned, changed_metadata = analysis.fixed_features(x, training, 8, seed=4)
    np.testing.assert_array_equal(original[:16], poisoned[:16])
    assert metadata == changed_metadata
    assert metadata['fit_samples'] == 12


def test_worker_differences_use_same_interior_eight_patches_without_crossing_windows(cfg, runner, tmp_path, monkeypatch):
    parent = tmp_path / 'parent'
    rows = native_rows('first') + native_rows('second')
    runner.write(parent / 'cache/manifest.json', dict(rows=rows))
    export = parent / 'models/QT__seed17/latest_export/hard.npz'
    export.parent.mkdir(parents=True)
    values = hard_export(export, len(rows))
    values['identities'] = np.array([row['id'] for row in rows][::-1])
    np.savez(export, **values)
    patch = np.arange(10, dtype=float) ** 2
    raw = np.empty((len(rows), 80, 3))
    for i in range(len(rows)):
        raw[i] = np.repeat(np.stack((patch + i * 10000,
                                    2 * patch + i * 5000, -patch), axis=-1), 8, axis=0)
    np.save(parent / 'cache/u.npy', raw)
    np.save(parent / 'cache/physical.npy', raw * 3)
    captured = []

    def capture(meta, x, target, config, coordinate_names):
        assert meta == rows
        captured.append((x.copy(), target.copy(), coordinate_names))
        return []

    monkeypatch.setattr(runner, 'relation_patch_readout', capture)
    monkeypatch.setattr(runner, 'relation_classification', lambda *a, **k: ([], [], {}))
    result = runner.relation_probe_worker((cfg, str(parent), 'QT', 17))
    assert len(captured) == 12
    for offset in (0, 3, 6, 9):
        level_x, level_y, _ = captured[offset]
        current_x, current_y, _ = captured[offset + 1]
        neighborhood_x, neighborhood_y, _ = captured[offset + 2]
        assert level_x.shape == (6, 10, 16)
        assert current_x.shape == (6, 8, 16)
        assert neighborhood_x.shape == (6, 8, 48)
        np.testing.assert_array_equal(current_x, level_x[:, 1:9])
        np.testing.assert_array_equal(neighborhood_x, np.concatenate(
            (level_x[:, :8], level_x[:, 1:9], level_x[:, 2:10]), axis=-1))
        np.testing.assert_allclose(current_y, level_y[:, 2:10] - level_y[:, 1:9])
        np.testing.assert_array_equal(current_y, neighborhood_y)
    assert result['tokenizer_fit'] is False and result['raw_data_read'] is False


@pytest.fixture
def frozen_parent(tmp_path, runner):
    parent = tmp_path / 'parent'
    output_root = tmp_path / 'exports'
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg['parent_run'], cfg['artifact_root'] = str(parent), str(output_root)
    parent_cfg = yaml.safe_load(PARENT_CONFIG.read_text())
    manifest = dict(schema='functional_training_cache_v1', identity='synthetic-v1',
                    rows=native_rows('only'), planned_trials=2, retained_trials=1,
                    failed_trials=[dict(trial_id='failed', reason='synthetic failure')])
    runner.write(parent / 'summary.json', dict(status='completed'))
    runner.write(parent / 'resolved_config.json', dict(config=parent_cfg))
    runner.write(parent / 'cache/manifest.json', manifest)
    return cfg, parent, output_root / 'probe', manifest


def test_runner_retains_parent_denominator_worker_failures_and_parent_bytes(frozen_parent, runner, monkeypatch):
    cfg, parent, out, manifest = frozen_parent
    before = {p.relative_to(parent): p.read_bytes() for p in parent.rglob('*') if p.is_file()}

    def tasks(output, stage, jobs, worker, workers):
        assert len(jobs) == 9 and workers == 2
        results = {}
        for key, payload in jobs:
            _, source, arm, seed = payload
            assert source == str(parent)
            if key == 'QT__seed29':
                results[key] = dict(status='failed', reason='deliberate synthetic failure')
            else:
                results[key] = dict(status='completed', arm=arm, seed=seed,
                                    semantic_readouts=[], classification=[], failures=[])
        return results

    monkeypatch.setattr(runner, 'bounded_tasks', tasks)
    result = runner.run_relation_probe(cfg, parent, out, 2)
    assert (result['planned_full_tasks'], result['attempted_tasks'], result['completed_tasks']) == (9, 9, 8)
    assert result['status'] != 'completed'
    assert result['parent_support'] == {k: manifest[k] for k in ('planned_trials', 'retained_trials', 'failed_trials')}
    assert any(r['task'] == 'QT__seed29' and 'deliberate' in r['reason'] for r in result['failures'])
    assert result['success_subset_only'] is True
    assert result['full_denominator_claim'] is False
    assert before == {p.relative_to(parent): p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    assert json.loads((out / 'summary.json').read_text())['completed_tasks'] == 8


def test_runner_accepts_supervisor_metadata_and_identical_resume(frozen_parent, runner, monkeypatch):
    cfg, parent, out, _ = frozen_parent
    out.mkdir(parents=True)
    metadata = ('launch.json', 'resources.json', 'process.log', 'source_snapshot.tar.gz')
    for filename in metadata:
        (out / filename).write_bytes(b'synthetic supervisor evidence')

    def tasks(output, stage, jobs, worker, workers):
        assert len(jobs) == workers == 1
        key, payload = jobs[0]
        _, _, arm, seed = payload
        return {key: dict(status='completed', arm=arm, seed=seed,
                          semantic_readouts=[], classification=[], failures=[])}

    monkeypatch.setattr(runner, 'bounded_tasks', tasks)
    for _ in range(2):
        result = runner.run_relation_probe(cfg, parent, out, 1, pilot=True)
        assert result['planned_full_tasks'] == 9 and result['attempted_tasks'] == 1
        assert result['completed_tasks'] == 1 and result['pilot'] is True
    for filename in metadata:
        assert (out / filename).read_bytes() == b'synthetic supervisor evidence'


@pytest.mark.parametrize('field,value', [
    (('schema',), 'unknown'), (('protected_data',), 'allowed'), (('arms',), ['QT']),
    (('seeds',), [17]), (('tensor', 'patches'), 9), (('tensor', 'patch_seconds'), 1),
    (('tensor', 'changes_scored_patch_indices'), list(range(9))),
    (('relations', 'minimum_trials_per_stratum'), 1),
    (('relations', 'donor_strata'), ['condition', 'split']),
    (('evaluation', 'probe_dimensions'), 64), (('evaluation', 'ridge_alpha'), [0]),
    (('evaluation', 'probe_C'), [-1]), (('evaluation', 'label_fractions'), [.1]),
])
def test_runner_rejects_scope_tensor_and_probe_contract_drift(frozen_parent, runner, field, value):
    cfg, parent, out, _ = frozen_parent
    destination = cfg
    for part in field[:-1]:
        destination = destination[part]
    destination[field[-1]] = value
    with pytest.raises(ValueError):
        runner.run_relation_probe(cfg, parent, out, 1)
    assert not out.exists()


@pytest.mark.parametrize('boundary', ['same', 'child', 'ancestor', 'outside_root', 'wrong_parent',
                                      'incomplete_parent', 'protected_parent', 'protected_manifest',
                                      'nonempty_output', 'immutable_output', 'workers'])
def test_runner_enforces_parent_output_and_resume_boundaries(frozen_parent, runner, boundary):
    cfg, parent, out, manifest = frozen_parent
    workers = 1
    if boundary == 'same':
        out = parent
    elif boundary == 'child':
        out = parent / 'new_probe'
    elif boundary == 'ancestor':
        out = parent.parent
    elif boundary == 'outside_root':
        out = parent.parent / 'elsewhere'
    elif boundary == 'wrong_parent':
        cfg['parent_run'] = str(parent.parent / 'different')
    elif boundary == 'incomplete_parent':
        runner.write(parent / 'summary.json', dict(status='running'))
    elif boundary == 'protected_parent':
        source = runner.read(parent / 'resolved_config.json')
        source['config']['data']['protected_data'] = 'allowed'
        runner.write(parent / 'resolved_config.json', source)
    elif boundary == 'protected_manifest':
        manifest['rows'][0]['dataset'] = 'sealed_test'
        runner.write(parent / 'cache/manifest.json', manifest)
    elif boundary == 'nonempty_output':
        out.mkdir(parents=True)
        (out / 'unrelated.txt').write_text('retained unrelated work')
    elif boundary == 'immutable_output':
        runner.write(out / 'resolved_config.json', {'different': 'existing contract'})
    else:
        workers = cfg['resources']['max_cpu_workers'] + 1
    with pytest.raises(ValueError):
        runner.run_relation_probe(cfg, parent, out, workers)
