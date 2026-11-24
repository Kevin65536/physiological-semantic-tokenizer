#!/usr/bin/env python3
"""Select bounded ID-only local EEG gains by calibration-subject cross-validation."""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '4'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from threadpoolctl import threadpool_limits
import yaml

from experiments.scripts.calibrate_spatial_analysis_synthesis import weights
from experiments.scripts.run_dual_component_tokenizer import identity, owner_lock
from src.analysis.dual_component_tokenizer import base_training_aggregate, joint_reconstruction_metrics, joint_reconstruction_summary
from src.analysis.spatial_tokenizer import write
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_training_data
from src.tokenizers.spatial_local_gain import LocalEnergyGainDecoder, filtered_bank, fixed_prediction, integer_features, load_decoder, local_prediction, patch_rms
from src.tokenizers.spatial_waveform_dictionary import load_model

KEYS = ('nmse', 'rms_gain', 'band_4_8_db', 'band_4_8_abs_db', 'band_8_13_db', 'band_8_13_abs_db', 'local_rms_correlation')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_check(cfg):
    if time.process_time() > cfg['resources']['CPU_soft_stop_seconds']:
        raise TimeoutError('predeclared 2600 CPU-second soft budget exhausted')


def minimal_batch(batch):
    result = {key: batch[key] for key in ('x', 's', 'o', 'mask')}
    for prefix in ('v', 'dv'):
        for field in ('x', 's', 'o'):
            result[prefix + field] = batch[prefix + field][:, :1, :1]
    return result


def audit_case(case, prediction):
    selected = case['selected']
    result = {name: prediction[branch, selected].reshape(len(selected), 15, 400)
              for branch, name in enumerate(('S', 'O'))}
    result['sum'] = result['S'] + result['O']
    metrics = joint_reconstruction_metrics(result, case['batch'], 1, case['scales'])['metrics']
    return {key: metrics[key] for key in KEYS}


def aggregate(cases, metrics, rows):
    examples = np.concatenate([case['examples'] for case in cases])
    values = {key: np.concatenate([record[key] for record in metrics]) for key in KEYS}
    return base_training_aggregate(values, examples.tolist(), rows)


def critical_values(rms):
    return rms[:, :, :, (1, 2)].reshape(*rms.shape[:2], 4)


def prepare(cfg, out, rows, scales):
    source_root = ROOT / cfg['source']['fixed_gain_run']
    panel = json.loads((source_root / 'training_panel.json').read_text())
    cv = json.loads((source_root / 'training_subject_CV.json').read_text())
    fixed = json.loads((source_root / 'frozen_gains.json').read_text())['arms']['energy_constrained']
    folds = {}
    fold_gain = {}
    for record in cv:
        if record['arm'] != 'energy_constrained':
            continue
        fold_gain[record['dataset'], record['fold']] = record['gain']
        for subject in record['held_subjects']:
            folds[record['dataset'], subject] = record['fold']
    model = load_model(ROOT / cfg['source']['decoder'])
    cases = []
    cache = OrderedDict()
    for ordinal, crop in enumerate(panel):
        cpu_check(cfg)
        row = rows[crop['row']]
        if row['split'] != 'train' or row['modality'] != 'eeg':
            raise ValueError('fixed calibration training boundary before arrays')
        with np.load(source_root / 'train_statistics' / f'crop_{ordinal:04d}.npz') as stored:
            codes = {branch: stored[branch].copy() for branch in ('S', 'O')}
            examples = stored['examples'].copy()
        expected = np.array([(crop['row'], crop['start_s'], p) for p in crop['positions']])
        np.testing.assert_array_equal(examples, expected)
        dataset = row['dataset']
        names = model.metadata['positions'][dataset]
        prediction = model.decode_ids(codes, dataset)
        wave = np.stack([prediction[branch][0].reshape(len(names), 6000) for branch in ('S', 'O')])
        bands = filtered_bank(wave)
        features, rms = integer_features(wave, codes)
        selected = np.array([next(j for j, name in enumerate(names) if name.upper() == row['positions'][p].upper()) for p in crop['positions']])
        batch = minimal_batch(base_tokenizer_batch(rows, examples.tolist(), scales, array_cache=cache, cache_limit=8))
        valid = batch['mask'].all(-1)
        target_rms = np.stack([patch_rms(np.where(batch['mask'], batch[field], 0.)).reshape(len(examples), 15, 5)
                               for field in ('s', 'o')], axis=2)
        fold = folds[dataset, row['subject']]
        gain = fold_gain[dataset, fold]
        baseline = fixed_prediction(wave, bands, gain)
        baseline_rms = np.stack([patch_rms(baseline[branch]).reshape(len(names), 15, 5) for branch in range(2)], axis=2)
        original_scales = np.array([scales['positions']['eeg'][dataset][row['positions'][p]]['scale'] for p in crop['positions']])
        case = dict(dataset=dataset, subject=row['subject'], fold=fold, examples=examples, selected=selected,
                    wave=wave, bands=bands, features=features, batch=batch, valid=valid, target_rms=target_rms,
                    baseline_rms=baseline_rms, fixed_gain=np.asarray(gain), scales=original_scales)
        case['baseline_metrics'] = audit_case(case, baseline)
        cases.append(case)
        write(out / 'progress.json', dict(stage='prepare_train', completed=ordinal + 1, planned=len(panel), CPU_seconds=time.process_time()))
    write(out / 'training_support.json', dict(crops=len(cases), position_inputs=sum(len(case['examples']) for case in cases),
        complete_patches=sum(int(case['valid'].sum()) for case in cases), planned_patches=sum(case['valid'].size for case in cases),
        subjects={dataset: sorted({case['subject'] for case in cases if case['dataset'] == dataset}) for dataset in sorted(fixed)},
        source_encoder_exposure='AS books and Wiener were already fitted on original train subjects, including calibration-CV held subjects',
        CV_scope='only local-gain regressor and its feature scales are fit on disjoint calibration subjects',
        screen_opened=False, test_opened=False, confirmation_opened=False))
    return cases, fixed, fold_gain, model


def regression(cases, rows, penalty, full_gain=None):
    examples = np.concatenate([case['examples'] for case in cases])
    position_weight = weights(examples, rows)
    X, Y, W = [], [], []
    offset = 0
    for case in cases:
        valid = case['valid']
        features = case['features'][case['selected']]
        if full_gain is None:
            baseline_rms = case['baseline_rms'][case['selected']]
        else:
            baseline = fixed_prediction(case['wave'], case['bands'], full_gain)
            baseline_rms = np.stack([patch_rms(baseline[branch]).reshape(len(case['wave'][0]), 15, 5) for branch in range(2)], axis=2)[case['selected']]
        target = np.log(np.maximum(critical_values(case['target_rms']), 1e-8)) - np.log(np.maximum(critical_values(baseline_rms), 1e-8))
        weight = position_weight[offset:offset + len(valid)] / np.maximum(valid.sum(1), 1)
        offset += len(valid)
        X.append(features[valid])
        Y.append(target[valid])
        W.append(np.broadcast_to(weight[:, None], valid.shape)[valid])
    X, Y, W = np.concatenate(X), np.concatenate(Y), np.concatenate(W)
    if not np.isfinite(X).all() or not np.isfinite(Y).all() or not len(X):
        raise ValueError('finite complete training-patch regression support required')
    W /= W.sum()
    mean = np.sum(X * W[:, None], axis=0)
    scale = np.sqrt(np.sum((X - mean) ** 2 * W[:, None], axis=0)).clip(1e-6)
    standard = (X - mean) / scale
    intercept = np.sum(Y * W[:, None], axis=0)
    weighted = W * len(X)
    matrix = (standard * weighted[:, None]).T @ standard + penalty * np.eye(X.shape[1])
    linear = (standard * weighted[:, None]).T @ (Y - intercept)
    coefficient = np.linalg.solve(matrix, linear).T
    return dict(mean=mean, scale=scale, coefficient=coefficient, intercept=intercept)


def selection(candidate, baseline, fold_candidates, fold_baselines, cfg):
    setting = cfg['selection']
    current = candidate['metrics']
    reference = baseline['metrics']
    cn = np.asarray(current['nmse']['macro']).ravel()
    bn = np.asarray(reference['nmse']['macro']).ravel()
    thresholds = np.array([setting['guard_X_O_NMSE_max_vs_matched_fixed_CV'], setting['guard_S_NMSE_max_vs_matched_fixed_CV'], setting['guard_X_O_NMSE_max_vs_matched_fixed_CV']])
    nmse_guard = bool(np.all(cn <= bn * thresholds))
    def energy(summary, dataset=None):
        section = 'macro' if dataset is None else 'by_dataset'
        values = []
        for key in ('band_4_8_abs_db', 'band_8_13_abs_db'):
            item = summary['metrics'][key][section] if dataset is None else summary['metrics'][key][section][dataset]
            values.append(float(np.asarray(item).ravel()[1]))
        return float(np.mean(values))
    improvement = 1 - energy(candidate) / max(energy(baseline), 1e-12)
    improved_datasets = [dataset for dataset in current['nmse']['by_dataset'] if energy(candidate, dataset) < energy(baseline, dataset)]
    repeated = []
    for candidate_fold, baseline_fold in zip(fold_candidates, fold_baselines):
        c = np.asarray(candidate_fold['metrics']['nmse']['macro']).ravel()[1]
        b = np.asarray(baseline_fold['metrics']['nmse']['macro']).ravel()[1]
        repeated.append(energy(candidate_fold) < energy(baseline_fold) and c <= b * thresholds[1])
    cr = float(np.asarray(current['rms_gain']['macro']).ravel()[1])
    br = float(np.asarray(reference['rms_gain']['macro']).ravel()[1])
    gain_guard = abs(cr - 1) <= abs(br - 1)
    eligible = (nmse_guard and gain_guard and improvement >= setting['required_global_S_theta_alpha_mean_abs_dB_relative_improvement']
                and len(improved_datasets) >= setting['required_datasets_with_positive_S_energy_improvement']
                and sum(repeated) >= setting['required_folds_with_positive_S_energy_and_S_NMSE_guard'])
    band = [float(np.asarray(current[key]['macro']).ravel()[1]) for key in ('band_4_8_abs_db', 'band_8_13_abs_db')]
    violation = max(0., .75 - cr) / .75 + sum(max(0., value - 3.0103) / 3.0103 for value in band)
    return dict(eligible=bool(eligible), nmse_guard=nmse_guard, RMS_distance_guard=bool(gain_guard),
                S_energy_relative_improvement=float(improvement), improved_datasets=improved_datasets,
                repeated_folds=int(sum(repeated)), fold_replication=list(map(bool, repeated)),
                joint_normalized_gate_violation=float(violation), macro_NMSE=float(np.mean(cn)))


def cross_validation(cases, rows, cfg, out):
    baseline = aggregate(cases, [case['baseline_metrics'] for case in cases], rows)
    fold_baselines = [aggregate([case for case in cases if case['fold'] == fold],
                               [case['baseline_metrics'] for case in cases if case['fold'] == fold], rows) for fold in range(4)]
    results = []
    for penalty in cfg['training']['ridge_penalties']:
        fits = {}
        for dataset in sorted({case['dataset'] for case in cases}):
            for fold in range(4):
                cpu_check(cfg)
                train = [case for case in cases if case['dataset'] == dataset and case['fold'] != fold]
                fits[dataset, fold] = regression(train, rows, penalty)
        for strength in cfg['training']['strengths']:
            records = []
            for case in cases:
                cpu_check(cfg)
                prediction = local_prediction(case['wave'], case['bands'], case['features'], fits[case['dataset'], case['fold']],
                                              case['fixed_gain'], strength, cfg['synthesis'])
                records.append(audit_case(case, prediction))
            summary = aggregate(cases, records, rows)
            fold_summaries = [aggregate([case for case in cases if case['fold'] == fold],
                                       [record for case, record in zip(cases, records) if case['fold'] == fold], rows) for fold in range(4)]
            decision = selection(summary, baseline, fold_summaries, fold_baselines, cfg)
            result = dict(penalty=penalty, strength=strength, decision=decision, held_subject_metrics=summary, fold_metrics=fold_summaries)
            results.append(result)
            write(out / 'CV_progress.json', dict(completed=results, planned=6, CPU_seconds=time.process_time(), screen_opened=False))
    write(out / 'CV_baseline.json', dict(all_subjects=baseline, folds=fold_baselines))
    write(out / 'CV_results.json', results)
    eligible = [record for record in results if record['decision']['eligible']]
    pool = eligible or results
    chosen = min(pool, key=lambda record: (record['decision']['joint_normalized_gate_violation'], record['decision']['macro_NMSE'], -record['penalty'], record['strength']))
    return chosen, bool(eligible)


def screen_fixed_ids(model_path, cfg, rows, scales, out):
    # This function is reached only after the predeclared training-CV admission.
    source = ROOT / cfg['source']['screen_ids']
    metadata = json.loads(source.with_name('integer_metadata.json').read_text())
    with np.load(source, allow_pickle=False) as stored:
        ids = {key: stored[key].copy() for key in stored.files}
    panel = ROOT / 'experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261123_joint_baseline_v1/panels.npz'
    panel_identity = json.loads(panel.with_name('panel.json').read_text())
    if digest(panel) != panel_identity['panel_sha256']:
        raise ValueError('frozen screening panel changed')
    checked = load_decoder(model_path)
    if (panel_identity['cfg_identity'] != checked.metadata['input_identity']
            or panel_identity['scales_identity'] != checked.metadata['scales_identity']):
        raise ValueError('local model and frozen screen coordinate identity differ')
    with np.load(panel) as stored:
        examples = stored['eeg_screen'].copy()
    grouped = defaultdict(list)
    for ordinal, (i, start, position) in enumerate(examples):
        if rows[int(i)]['split'] != 'validation':
            raise ValueError('frozen development screening scope required')
        grouped[int(i), int(start)].append((ordinal, int(position)))
    expected = [(record['row'], record['start_s']) for record in metadata]
    if list(grouped) != expected:
        raise ValueError('saved integer and fixed screening crop identities differ')
    np.savez_compressed(out / 'integer_ids.npz', **ids)
    write(out / 'integer_metadata.json', metadata)
    results = []
    support_results = []
    order = []
    for replay_pass in range(2):
        model = load_decoder(model_path)
        if replay_pass:
            with np.load(out / 'integer_ids.npz', allow_pickle=False) as stored:
                ids = {key: stored[key].copy() for key in stored.files}
        metric_parts = defaultdict(list)
        support_parts = defaultdict(list)
        cache = OrderedDict()
        for crop, ((i, start), selected) in enumerate(grouped.items()):
            cpu_check(cfg)
            row = rows[i]
            native = model.source.metadata['positions'][row['dataset']]
            if metadata[crop]['native_positions'] != native:
                raise ValueError('saved integer native electrode order changed')
            codes = {branch: ids[f'{branch}_{crop}'] for branch in ('S', 'O')}
            prediction = model.decode_ids(codes, row['dataset'])
            names = [next(j for j, name in enumerate(native) if name.upper() == row['positions'][p].upper()) for _, p in selected]
            batch = base_tokenizer_batch(rows, [(i, start, p) for _, p in selected], scales, array_cache=cache, cache_limit=8)
            original_scales = np.array([scales['positions']['eeg'][row['dataset']][row['positions'][p]]['scale'] for _, p in selected])
            selected_prediction = {key: value[0, names] for key, value in prediction.items()}
            audited = joint_reconstruction_metrics(selected_prediction, batch, 1, original_scales)
            for key, value in audited['metrics'].items():
                metric_parts[key].append(value)
            for key, value in audited['support'].items():
                support_parts[key].append(value)
            if not replay_pass:
                order.extend(ordinal for ordinal, _ in selected)
            write(out / 'screen_progress.json', dict(replay_pass=replay_pass, completed=crop + 1, planned=len(grouped), CPU_seconds=time.process_time()))
        results.append({key: np.concatenate(value)[np.argsort(order)] for key, value in metric_parts.items()})
        support_results.append({key: np.concatenate(value)[np.argsort(order)] for key, value in support_parts.items()})
    for key in results[0]:
        np.testing.assert_allclose(results[0][key], results[1][key], rtol=0, atol=0, equal_nan=True)
    for key in support_results[0]:
        np.testing.assert_array_equal(support_results[0][key], support_results[1][key])
    np.savez_compressed(out / 'metrics.npz', examples=examples, **results[1], **{'support_' + key: value for key, value in support_results[1].items()})
    summary = joint_reconstruction_summary(results[1], support_results[1], examples.tolist(), rows)
    summary.update(status='completed', modality='eeg', role='screen', checkpoint=str(model_path),
                   integer_replay=dict(status='passed', independent_checkpoint_reload=True, metric_max_abs_difference=0.),
                   same_saved_AS_integers=True)
    write(out / 'screen_summary.json', summary)


def run(args, cfg):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with owner_lock(out):
        manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
        source_root = ROOT / cfg['source']['fixed_gain_run']
        old_identity = json.loads((source_root / 'resolved_config.json').read_text())
        if (old_identity['input_identity'] != identity(manifest) or old_identity['scales_identity'] != manifest['scales_identity']
                or old_identity['source_decoder_sha256'] != digest(ROOT / cfg['source']['decoder'])
                or old_identity['source_encoder_sha256'] != digest(ROOT / cfg['source']['encoder_bundle'])):
            raise ValueError('retained training integers and current G0/source identity drift')
        source_metadata = load_model(ROOT / cfg['source']['decoder']).metadata
        if source_metadata['input_identity'] != manifest['cfg_identity'] or source_metadata['scales_identity'] != manifest['scales_identity']:
            raise ValueError('source decoder and current G0 coordinate drift')
        signature = dict(config=identity(cfg), input=identity(manifest), scales=manifest['scales_identity'],
                         source_decoder_sha256=digest(ROOT / cfg['source']['decoder']),
                         training_panel_sha256=digest(source_root / 'training_panel.json'),
                         fixed_gains_sha256=digest(source_root / 'frozen_gains.json'),
                         training_CV_sha256=digest(source_root / 'training_subject_CV.json'))
        resolved = dict(config=cfg, signature=signature)
        if (out / 'resolved_config.json').exists() and json.loads((out / 'resolved_config.json').read_text()) != resolved:
            raise ValueError('local-gain retained input identity changed')
        if (out / 'summary.json').exists():
            return
        write(out / 'resolved_config.json', resolved)
        cases, fixed, _, source = prepare(cfg, out, manifest['rows'], scales)
        chosen, admitted = cross_validation(cases, manifest['rows'], cfg, out)
        write(out / 'choice.json', dict(status='frozen_before_screen', chosen=chosen, admitted=admitted,
                                      training_CPU_seconds=time.process_time(), screen_opened=False))
        states = {}
        fixed_gains = {}
        for dataset in sorted(fixed):
            selected = [case for case in cases if case['dataset'] == dataset]
            fixed_gains[dataset] = fixed[dataset]['gain']
            states[dataset] = regression(selected, manifest['rows'], chosen['penalty'], fixed_gains[dataset])
        metadata = dict(schema='spatial_AS_local_gain_v1', modality='eeg', codebook_size=8192,
                        source_decoder=str((ROOT / cfg['source']['decoder']).resolve()),
                        source_decoder_sha256=signature['source_decoder_sha256'], input_identity=manifest['cfg_identity'],
                        scales_identity=manifest['scales_identity'], signature=signature,
                        decoder_inputs='S/O integer arrays and frozen native metadata only', source_encoder_exposure='previous original train fitting',
                        selection='four calibration-subject CV folds; original historical validation exposure retained')
        model = LocalEnergyGainDecoder(source, states, fixed_gains, cfg['synthesis'], chosen['strength'], metadata)
        model_path = out / ('model.npz' if admitted else 'diagnostic_model.npz')
        model.save(model_path)
        if admitted:
            del cases
            screen_fixed_ids(model_path, cfg, manifest['rows'], scales, out)
        write(out / 'summary.json', dict(status='completed' if admitted else 'CV_rejected', admitted_to_screen=admitted,
            selected_penalty=chosen['penalty'], selected_strength=chosen['strength'], CV_decision=chosen['decision'],
            model=str(model_path), CPU_seconds=time.process_time(), screen_opened=admitted, confirmation_opened=False, test_opened=False,
            transmitted_ID_count_and_rate='unchanged from frozen AS source', decoder_per_window_floating_side_information=0))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'experiments/configs/physiology_semantic_tokenizer/dual_component_spatial_as_local_gain_v1.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if cfg['schema'] != 'dual_component_spatial_as_local_gain_v1' or cfg['training']['validation_fit'] != 'forbidden':
        raise ValueError('train-only local-gain calibration contract required')
    resource.setrlimit(resource.RLIMIT_CPU, (cfg['resources']['CPU_budget_seconds'], cfg['resources']['CPU_budget_seconds'] + 10))
    def stop(signum, frame):
        raise TimeoutError('45 CPU-minute budget reached')
    signal.signal(signal.SIGXCPU, stop)
    try:
        with threadpool_limits(cfg['resources']['numerical_threads']):
            run(args, cfg)
    except BaseException as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write(args.output / 'failure.json', dict(type=type(error).__name__, message=str(error), CPU_seconds=time.process_time()))
        raise


if __name__ == '__main__':
    main()
