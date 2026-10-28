"""Synthetic frozen-vocabulary evidence and masked-input boundary checks."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from src.analysis.dual_component_tokenizer import (
    fit_matched_smoothing, reconstruction_rows, subject_aggregate,
    paired_subject_bootstrap, codebook_health, align_codewords,
    matched_donor_tokens, run_masked_pilot, analyze_suite,
)
from src.foundation.dual_component_masked import (
    DualComponentMaskedModel, block_masks, own_context,
    expected_waveforms, fit_token_baselines, baseline_probabilities,
)


def test_masked_ids_are_removed_before_embedding_and_teacher_has_no_input():
    torch.set_num_threads(1); torch.manual_seed(2)
    model = DualComponentMaskedModel({'eeg':4, 'hb':4}, patches=4, width=8,
                                     layers=1, heads=2, dropout=0).eval()
    ids = torch.randint(0, 4, (2, 4, 2, 2)); visible = torch.ones_like(ids, dtype=torch.bool)
    visible[:, 1:3] = False
    poisoned = ids.clone(); poisoned[~visible] = 123456
    with torch.no_grad():
        assert torch.equal(model(ids, visible), model(poisoned, visible))
    with pytest.raises(TypeError): model(ids, visible, teacher=torch.randn(2, 4, 2))
    with pytest.raises(ValueError): model(ids.float(), visible)


def test_original_support_joint_mask_and_own_context():
    valid = np.ones((3, 10, 2, 2), bool)
    visible, target = block_masks(valid, rng=np.random.default_rng(4), target_modality=1,
                                   block_patches=2, task='time')
    assert np.array_equal(visible[..., 0], visible[..., 1])
    assert np.array_equal(visible[:, :, 0], visible[:, :, 1])
    assert target.sum() == 3 * 2 * 2
    own = own_context(visible, 1)
    assert not own[:, :, 0].any()
    assert np.array_equal(own[:, :, 1], visible[:, :, 1])
    diagnostic, _ = block_masks(valid, rng=np.random.default_rng(4), target_modality=1,
                                 block_patches=2, task='modality', single_branch=0)
    assert diagnostic[:, :, 1, 1].all()


def test_probability_expectation_decodes_codewords_before_average():
    codes = np.array([[-2.], [1.]])
    probabilities = np.array([[.5, .5]])
    decoder_output = codes ** 2
    assert expected_waveforms(probabilities, decoder_output)[0, 0] == pytest.approx(2.5)
    assert not np.allclose(expected_waveforms(probabilities, decoder_output), (probabilities @ codes) ** 2)
    with pytest.raises(ValueError): expected_waveforms(np.array([[.2, .2]]), decoder_output)


def test_marginal_and_metadata_baselines_are_train_only_and_label_blind():
    ids = np.zeros((3, 4, 2, 2), dtype=int); ids[-1] = 3
    rows = [dict(split=s, dataset='d', roi='left', condition='secret') for s in ('train', 'train', 'test')]
    before = fit_token_baselines(ids, rows, 4)
    changed = ids.copy(); changed[-1] = 1
    rows[-1]['condition'] = 'other'; rows[0]['condition'] = 'also ignored'
    after = fit_token_baselines(changed, rows, 4)
    assert np.array_equal(before['marginal'], after['marginal'])
    assert np.array_equal(baseline_probabilities(before, rows, metadata=True), baseline_probabilities(after, rows, metadata=True))
    assert before['marginal'][0, 0, 0] > before['marginal'][0, 0, 3]


def test_smooth_kernel_selection_does_not_see_evaluation():
    rng = np.random.default_rng(3)
    observed = rng.normal(size=(5, 3, 8)); physical = observed.copy()
    training = np.array([1, 1, 1, 0, 0], bool)
    _, first = fit_matched_smoothing(observed, physical, training, [0, 1, 2], channels=2)
    physical[-2:] = 1e8; observed[-2:] = rng.normal(size=(2, 3, 8)) * 100
    _, second = fit_matched_smoothing(observed, physical, training, [0, 1, 2], channels=2)
    assert first['sigmas_samples'] == second['sigmas_samples']
    assert first['sigmas_samples'] == [0., 0.]


def test_record_subject_hierarchy_and_seed_not_independent_units():
    rows = [dict(dataset='d', subject='s1', record='r1', seed=1, NMSE=v) for v in [0., 0., 0.]]
    rows.append(dict(dataset='d', subject='s1', record='r2', seed=1, NMSE=4.))
    aggregate = subject_aggregate(rows)
    assert aggregate[0]['NMSE'] == pytest.approx(2.)
    a = [dict(dataset='d', subject=s, seed=k, NMSE=v) for s, v in [('a', 1), ('b', 3)] for k in range(3)]
    b = [dict(r, NMSE=r['NMSE'] + 1) for r in a]
    effect = paired_subject_bootstrap(a, b, samples=30)
    assert effect['subjects'] == 2
    assert effect['mean_difference'] == -1
    assert effect['CI95'] == [-1, -1]


def test_channel_scores_preserve_HbO_HbR_and_common_training_SD():
    rows = [dict(dataset='d', subject='s', record='r', split='test')]
    target = {'eeg':np.zeros((1, 2, 2)), 'hb':np.zeros((1, 2, 4))}
    predictions = {}
    for m in target:
        predictions[m + '_S'] = np.ones_like(target[m])
        predictions[m + '_O'] = np.zeros_like(target[m])
        predictions[m + '_sum'] = predictions[m + '_S']
    predictions['hb_S'][..., 1::2] = 2
    scores = reconstruction_rows(predictions, target, target, target, rows, variant='hard_student', seed=1)
    assert next(r['NMSE'] for r in scores if r['component'] == 'S' and r['signal'] == 'HbO') == 1
    assert next(r['NMSE'] for r in scores if r['component'] == 'S' and r['signal'] == 'HbR') == 4


def test_codebook_coverage_and_waveform_alignment():
    ids = np.array([[0, 0], [0, 1], [2, 2]])
    rows = [dict(split='train', subject='a', record='r1'), dict(split='train', subject='a', record='r1'),
            dict(split='test', subject='b', record='r2')]
    health = codebook_health(ids, np.arange(8).reshape(4, 2), rows,
                            modality='eeg', branch='S', variant='hard_student', seed=1)
    zero = next(r for r in health if r['split'] == 'train' and r['code'] == 0)
    assert zero['subjects'] == 1 and zero['assignments'] == 3 and zero['active_codes'] == 2
    reference = np.arange(8).reshape(4, 2); permutation = np.array([2, 0, 3, 1])
    mapping, effect = align_codewords(reference, reference[permutation])
    assert np.array_equal(mapping, permutation)
    assert effect['mean_prototype_NMSE'] == 0


def test_donors_never_cross_subject_or_evaluation_partition():
    ids = np.zeros((4, 4, 2, 2), dtype=int)
    ids[1, :, 1] = 1; ids[2:, :, 1] = 2
    rows = [dict(dataset='d', roi='r', subject='a', split='validation'),
            dict(dataset='d', roi='r', subject='b', split='validation'),
            dict(dataset='d', roi='r', subject='c', split='test'),
            dict(dataset='d', roi='r', subject='d', split='test')]
    decoded = {(m, c):np.ones((4, 2)) for m in range(2) for c in range(2)}
    result, available, _ = matched_donor_tokens(ids, rows, decoded, source_modality=1,
                                                training_ids=ids[:2], training_rows=rows[:2])
    assert available.all()
    assert np.array_equal(result[0, :, 1], ids[1, :, 1])
    assert np.array_equal(result[1, :, 1], ids[0, :, 1])


def _synthetic_run(tmp_path):
    rng = np.random.default_rng(9); cache = tmp_path / 'cache'; cache.mkdir()
    rows = [dict(id=str(i), dataset='d', roi='r', subject=f's{i}', record=f'r{i}',
                 split='train' if i < 4 else 'validation' if i < 6 else 'test') for i in range(8)]
    arrays = {}
    for m, d in [('eeg', 4), ('hb', 8)]:
        s = rng.normal(size=(8, 4, d)).astype('float32')
        e = rng.normal(size=(8, 4, d)).astype('float32')
        for name, value in [('s', s), ('e', e), ('u', s + e)]:
            key = m + '_' + name; arrays[key] = value; np.save(cache / (key + '.npy'), value)
    manifest = dict(rows=rows, patches=4, planned_windows=8,
                    arrays={k:k + '.npy' for k in arrays}, target_scaling='observation_training_sd')
    (cache / 'manifest.json').write_text(json.dumps(manifest))
    job = tmp_path / 'jobs' / 'hard_student' / 'seed_17'; job.mkdir(parents=True)
    predictions = {}; codewords = {}
    for m, d in [('eeg', 4), ('hb', 8)]:
        predictions[m + '_sum'] = arrays[m + '_u']
        for branch, name in [('S', 's'), ('O', 'e')]:
            predictions[m + '_' + branch] = arrays[m + '_' + name]
            predictions[m + '_ids_' + branch] = rng.integers(0, 4, size=(8, 4))
            codewords[m + '_' + branch] = rng.normal(size=(4, d)).astype('float32')
    np.savez(job / 'predictions.npz', **predictions); np.savez(job / 'codewords.npz', **codewords)
    (job / 'summary.json').write_text(json.dumps(dict(status='completed', step=2)))
    cfg = dict(seed=9, variants=['hard_student'], seeds=[17], training=dict(steps=2),
        model=dict(codebook_size=4), masked=dict(steps=2, batch_size=3, block_patches=1,
            width=8, layers=1, heads=2, dropout=0, checkpoint_every=1, routes=['hard_student'], seeds=[17]),
        evaluation=dict(bootstrap_repeats=10))
    return cfg


def test_tiny_masked_pipeline_and_evidence_summary(tmp_path):
    torch.set_num_threads(1)
    cfg = _synthetic_run(tmp_path)
    result = run_masked_pilot(cfg, tmp_path, seed=17)
    assert result['status'] == 'completed' and result['teacher_inputs'] is False
    assert result['cross_vocabulary_CE_ranking_allowed'] is False
    assert (tmp_path / 'masked/hard_student/seed_17/trace_hb.png').exists()
    analysis = analyze_suite(cfg, tmp_path)
    assert analysis['masked']['status'] == 'completed'
    assert analysis['teacher_sensitivity']['status'] == 'unavailable'
    assert analysis['quality_screen']['status'] == 'diagnostic_only'
    assert all('dataset' in row for row in analysis['paired_comparisons'])


def test_nonwrapping_shift_never_reveals_original_hidden_tokens():
    from src.foundation.dual_component_masked import nonwrapping_source_shift
    ids = np.arange(10 * 2 * 2).reshape(1, 10, 2, 2) % 4
    visible, _ = block_masks(ids >= 0, rng=np.random.default_rng(1), target_modality=0,
                              block_patches=2, task='time')
    poison = ids.copy(); poison[~visible] = 123456
    first, first_visible = nonwrapping_source_shift(ids, visible, 1, 3)
    second, second_visible = nonwrapping_source_shift(poison, visible, 1, 3)
    assert np.array_equal(first_visible, second_visible)
    assert np.array_equal(first[first_visible], second[second_visible])
    assert not first_visible[0, 7:9, 1].any()


def test_actual_anchor_region_partitions_metadata_baseline():
    ids = np.zeros((3, 4, 2, 2), dtype=int); ids[1] = 1
    rows = [dict(split='train', dataset='d', anchor=dict(region='left')),
            dict(split='train', dataset='d', anchor=dict(region='right')),
            dict(split='test', dataset='d', anchor=dict(region='right'))]
    baseline = fit_token_baselines(ids, rows, 4)
    p = baseline_probabilities(baseline, rows, metadata=True)
    assert p[0, 0, 0, 0, 0] > p[0, 0, 0, 0, 1]
    assert p[2, 0, 0, 0, 1] > p[2, 0, 0, 0, 0]


def test_uncompleted_export_is_not_analysis_evidence(tmp_path):
    cfg = _synthetic_run(tmp_path)
    record = tmp_path / 'jobs/hard_student/seed_17/summary.json'
    record.write_text(json.dumps(dict(status='running', step=1)))
    result = analyze_suite(cfg, tmp_path)
    assert result['status'] == 'partial'
    assert len(result['missing_jobs']) == 1
    assert not any(r.get('first') == 'hard_student' and r.get('subjects', 0) > 0 for r in result['paired_comparisons'])
    with pytest.raises(ValueError, match='completed fixed-budget'):
        run_masked_pilot(cfg, tmp_path, seed=17)


def test_smooth_analysis_scores_own_targets_and_cross_vocabulary_only_sum(tmp_path):
    import csv
    import shutil
    cfg = _synthetic_run(tmp_path)
    cfg['variants'] = ['hard_student', 'smooth_hard', 'ordinary_hard']
    source = tmp_path / 'jobs/hard_student/seed_17'
    smooth_job = tmp_path / 'jobs/smooth_hard/seed_17'
    ordinary_job = tmp_path / 'jobs/ordinary_hard/seed_17'
    shutil.copytree(source, smooth_job); shutil.copytree(source, ordinary_job)
    with np.load(source / 'predictions.npz') as data:
        predictions = {key:data[key].copy() for key in data.files}
    smoothing = dict(arrays={})
    for modality in ('eeg', 'hb'):
        observed = np.load(tmp_path / 'cache' / (modality + '_u.npy'))
        # Distinct from Tak labels: an exact own-target reconstruction must score zero.
        smooth_s = np.full_like(observed, 10.)
        smooth_o = observed - smooth_s
        for name, value in [('s', smooth_s), ('e', smooth_o)]:
            filename = 'smooth_' + modality + '_' + name + '.npy'
            np.save(tmp_path / 'cache' / filename, value)
            smoothing['arrays'][modality + '_' + name] = filename
        predictions[modality + '_S'] = smooth_s
        predictions[modality + '_O'] = smooth_o
        predictions[modality + '_sum'] = observed
    np.savez(smooth_job / 'predictions.npz', **predictions)
    (tmp_path / 'cache/smoothing.json').write_text(json.dumps(smoothing))
    result = analyze_suite(cfg, tmp_path)
    with (tmp_path / 'analysis/reconstruction_subjects.csv').open() as stream:
        scores = list(csv.DictReader(stream))
    smooth = [row for row in scores if row['variant'] == 'smooth_hard' and row['component'] in ('S', 'O')]
    assert smooth and all(float(row['NMSE']) == 0 for row in smooth)
    assert all(row['target_family'] == 'matched_smoothing' for row in smooth)
    assert all(row['component'] == 'sum' for row in scores if row['variant'] == 'ordinary_hard')
    cross = [row for row in result['paired_comparisons'] if row['second'] in ('smooth_hard', 'ordinary_hard')]
    assert cross and all(row['component'] == 'sum' and row['target_family'] == 'processed_observation' for row in cross)
    assert result['physical_compatibility']['status'] == 'not_evaluated'


def test_signed_boundary_error_detects_opposite_slopes_with_equal_roughness():
    from src.analysis.dual_component_tokenizer import _score_probabilities
    ids = np.zeros((1, 2, 2, 2), dtype=int); ids[0, 1, 0, 0] = 1
    probability = np.eye(2)[ids]
    target = np.ones_like(ids, bool)
    decoded = {(m, c):np.zeros((2, 2)) for m in range(2) for c in range(2)}
    decoded[(0, 0)][1] = -1
    physical = np.array([[[0., 0.], [1., 1.]]])
    arrays = dict(eeg_s=physical, eeg_e=np.zeros_like(physical), eeg_u=physical)
    rows = [dict(dataset='d', subject='s', record='r', split='test')]
    scores, _ = _score_probabilities(probability, ids, target, rows, decoded, arrays,
        probability, probability, route='hard_student', seed=17, task='time', context='paired', target_modality=0)
    excess = next(row['value'] for row in scores if row['metric'] == 'boundary_jump_excess_over_teacher_trainingSD')
    signed_error = next(row['value'] for row in scores if row['metric'] == 'boundary_jump_error_MSE_trainingSD2')
    assert excess == 0
    assert signed_error == 4
