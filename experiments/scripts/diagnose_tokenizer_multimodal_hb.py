"""Train-subject nested CV of local and observed multimodal Hb component prediction."""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '2')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from threadpoolctl import threadpool_limits
import yaml

from src.data.dual_component_tokenizer import (
    base_tokenizer_batch, base_tokenizer_joint_eeg, base_tokenizer_joint_hb,
    base_tokenizer_training_data,
)
from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics, joint_reconstruction_summary
from src.analysis.dual_component_tokenizer import base_training_aggregate


def write(path, value):
    def encode(item):
        if isinstance(item, np.ndarray):
            return encode(item.tolist())
        if isinstance(item, np.generic):
            return encode(item.item())
        if isinstance(item, dict):
            return {key: encode(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [encode(value) for value in item]
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    path = Path(path)
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(encode(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def make_plan(rows, cfg):
    setting = cfg['diagnostic']
    records = defaultdict(list)
    eeg_lookup = {row['parent']: i for i, row in enumerate(rows)
                  if row['modality'] == 'eeg' and row['split'] == 'train'}
    for index, row in enumerate(rows):
        if row['modality'] != 'hb' or row['split'] != 'train':
            continue
        for item in row['inputs']['score']:
            positions = sorted(item['positions'], key=lambda p: digest(
                (setting['panel_seed'], row['parent'], item['start_s'], row['positions'][p])))
            if positions:
                records[row['dataset'], row['subject'], row['record']].append(dict(
                    row=index, start_s=item['start_s'], positions=positions[:setting['target_positions_per_window']],
                    eeg_row=eeg_lookup.get(row['parent'])))
    subjects = defaultdict(list)
    for key, values in sorted(records.items()):
        selected = sorted(values, key=lambda value: digest((setting['panel_seed'], key, value)))
        subjects[key[:2]].extend(selected[:setting['windows_per_record']])
    datasets = defaultdict(dict)
    for (dataset, subject), values in sorted(subjects.items()):
        datasets[dataset][subject] = sorted(values, key=lambda value: digest(
            (setting['panel_seed'], dataset, subject, value)))[:setting['windows_per_subject']]
    folds = {dataset: {subject: i % setting['folds'] for i, subject in enumerate(sorted(
        values, key=lambda subject: digest((setting['panel_seed'], dataset, subject))))}
        for dataset, values in datasets.items()}
    return dict(datasets=dict(datasets), subject_folds=folds, numerical_read_splits=['train'])


def lag_features(values, valid, seconds, rate=10):
    """Feature at t uses X(t-lag); zero and validity indicators stay within this crop."""
    values = np.asarray(values, float)
    valid = np.asarray(valid, bool)
    if values.shape != valid.shape or values.shape[0] != 300:
        raise ValueError('matched 30-s 10-Hz features and observation masks required')
    result, observed = [], []
    for lag in seconds:
        shift = int(round(lag * rate))
        x = np.zeros_like(values)
        mask = np.zeros_like(valid)
        if abs(shift) >= len(values):
            raise ValueError('lag must leave visible support')
        if shift > 0:
            x[shift:], mask[shift:] = values[:-shift], valid[:-shift]
        elif shift < 0:
            x[:shift], mask[:shift] = values[-shift:], valid[-shift:]
        else:
            x[:], mask[:] = values, valid
        result.append(np.where(mask, x, 0.))
        observed.append(mask.astype(float))
    return np.concatenate(result + observed, axis=1)


def eeg_power_summary(values, observed, bands):
    if values.shape[0] != 6000 or observed.shape != (values.shape[1],):
        raise ValueError('30-s EEG and observation-complete channel flags required')
    result = np.zeros((300, len(bands)))
    if not observed.any():
        return result, np.zeros_like(result, bool)
    wave = values[:, observed]
    transformed = np.fft.rfft(wave, axis=0)
    frequency = np.fft.rfftfreq(6000, 1 / 200)
    for index, (low, high) in enumerate(bands):
        selected = (frequency >= low) & (frequency < high if high < 45 else frequency <= high)
        filtered = np.fft.irfft(transformed * selected[:, None], n=6000, axis=0)
        power = np.mean(filtered.reshape(300, 20, -1) ** 2, axis=1)
        result[:, index] = np.log1p(power).mean(axis=1)
    return result, np.ones_like(result, bool)


def arm_columns(setting):
    width = len(setting['lag_seconds']) * 2
    local = np.arange(2 * width)
    other = np.arange(2 * width, 4 * width)
    eeg = np.arange(4 * width, 9 * width)
    return dict(local_hb=local, local_plus_other_hb=np.r_[local, other],
                local_plus_eeg=np.r_[local, eeg], local_plus_other_hb_eeg=np.r_[local, other, eeg])


def prepare(dataset, subjects, rows, scales, setting, output):
    examples, features, targets, masks, variances, input_scales, availability = [], [], [], [], [], [], []
    cache = OrderedDict()
    for subject, crops in subjects.items():
        for crop in crops:
            row = rows[crop['row']]
            assert row['split'] == 'train'
            values, observed = base_tokenizer_joint_hb(rows, crop, scales)
            eeg_value = np.zeros((300, 5))
            eeg_valid = np.zeros((300, 5), bool)
            eeg_channels = 0
            eeg_reason = 'no_paired_EEG_G0_parent'
            if crop['eeg_row'] is not None:
                eeg_row = rows[crop['eeg_row']]
                assert eeg_row['split'] == 'train' and eeg_row['parent'] == row['parent']
                admitted = any(item['start_s'] == crop['start_s']
                               for group in eeg_row['inputs'].values() for item in group)
                if admitted:
                    eeg_fields, eeg_observed, _ = base_tokenizer_joint_eeg(rows,
                        dict(row=crop['eeg_row'], start_s=crop['start_s'], positions=[]),
                        scales, eeg_row['positions'])
                    eeg_value, eeg_valid = eeg_power_summary(eeg_fields['x'], eeg_observed, setting['bands_hz'])
                    eeg_channels = int(eeg_observed.sum())
                    eeg_reason = 'available' if eeg_channels else 'no_observation_complete_scaled_EEG_electrode'
                else:
                    eeg_reason = 'no_matching_admitted_EEG_crop'
            crop_examples = [(crop['row'], crop['start_s'], p) for p in crop['positions']]
            batch = base_tokenizer_batch(rows, crop_examples, scales, array_cache=cache, cache_limit=8)
            for i, position in enumerate(crop['positions']):
                local, local_valid = values[:, position], observed[:, position]
                other_mask = observed.copy()
                other_mask[:, position] = False
                count = other_mask.sum(axis=1)
                other = np.divide(np.where(other_mask, values, 0).sum(axis=1), count,
                                  out=np.zeros_like(local), where=count > 0)
                parts = [lag_features(v, mask, setting['lag_seconds']) for v, mask in
                         [(local, local_valid), (other, count > 0), (eeg_value, eeg_valid)]]
                features.append(np.concatenate(parts, axis=1).astype(np.float32))
                examples.append(crop_examples[i])
                targets.append(np.stack([batch[k][i] for k in 'xso']))
                masks.append(batch['mask'][i])
                variances.append(np.stack([batch['v' + k][i, 0, :2] for k in 'xso']))
                input_scales.append(scales['positions']['hb'][dataset][row['positions'][position]]['scale'])
                availability.append(dict(subject=subject, row=crop['row'], start_s=crop['start_s'],
                    position=row['positions'][position], eeg_observed_electrodes=eeg_channels,
                    eeg_reason=eeg_reason, other_Hb_positions_min=int(count.min()),
                    local_observed_points=int(local_valid.sum()), target_scored_points=int(batch['mask'][i].sum())))
            write(output / 'progress.json', dict(stage='prepare', dataset=dataset, subject=subject,
                                                prepared_position_windows=len(examples)))
    arrays = dict(features=np.stack(features), targets=np.stack(targets), mask=np.stack(masks),
                  variances=np.stack(variances), input_scales=np.asarray(input_scales), examples=np.asarray(examples))
    np.savez_compressed(output / 'prepared.npz', **arrays)
    write(output / 'observation_support.json', availability)
    return arrays


def moment_matrix(features, target, valid):
    values = np.c_[features[valid], target[valid], np.ones(int(valid.sum()))]
    return values.T @ values / len(values)


def subject_moments(arrays, rows):
    records = {}
    counts = Counter()
    for index, (row_index, start, position) in enumerate(arrays['examples']):
        row = rows[row_index]
        assert row['split'] == 'train'
        mask = arrays['mask'][index].reshape(300, 2)
        target = arrays['targets'][index, 1:].reshape(2, 300, 2)
        for color in range(2):
            valid = mask[:, color]
            if not valid.any():
                continue
            key = row['subject'], row['positions'][position], row['record'], color
            moment = moment_matrix(arrays['features'][index].astype(float), target[:, :, color].T, valid)
            records[key] = records.get(key, np.zeros_like(moment)) + moment
            counts[key] += 1
    subjects = defaultdict(list)
    for (subject, position, record, color), value in records.items():
        subjects[subject, position, color].append(value / counts[subject, position, record, color])
    return {key: np.mean(values, axis=0) for key, values in subjects.items()}


def training_moment(stats, subjects, position, color):
    matched = [stats[subject, position, color] for subject in subjects if (subject, position, color) in stats]
    if matched:
        return np.mean(matched, axis=0), False
    pooled = []
    for subject in subjects:
        values = [value for (s, p, c), value in stats.items() if s == subject and c == color]
        if values:
            pooled.append(np.mean(values, axis=0))
    if not pooled:
        raise ValueError('training fold has no scored Hb targets')
    return np.mean(pooled, axis=0), True


def ridge_path(moment, columns, fractions):
    full_width = len(moment) - 3
    indices = np.r_[columns, full_width, full_width + 1, full_width + 2]
    selected = moment[np.ix_(indices, indices)]
    width = len(columns)
    mx, my = selected[:width, -1], selected[width:width + 2, -1]
    gram = selected[:width, :width] - np.outer(mx, mx)
    cross = selected[:width, width:width + 2] - np.outer(mx, my)
    sd = np.sqrt(np.maximum(np.diag(gram), 1e-12))
    normalized = gram / sd[:, None] / sd[None]
    eigenvalues, eigenvectors = np.linalg.eigh((normalized + normalized.T) / 2)
    projected = eigenvectors.T @ (cross / sd[:, None])
    result = []
    for strength in fractions:
        beta = (eigenvectors @ (projected / (np.maximum(eigenvalues, 0) + strength)[:, None])) / sd[:, None]
        result.append((beta, my - mx @ beta))
    return result


def prediction_mse(moment, columns, model):
    width = len(moment) - 3
    beta, bias = model
    gram = moment[np.ix_(columns, columns)]
    cross = moment[np.ix_(columns, [width, width + 1])]
    mx, my = moment[columns, -1], moment[width:width + 2, -1]
    second = np.diag(moment)[width:width + 2]
    value = (second + np.einsum('ik,ij,jk->k', beta, gram, beta) + bias**2
             + 2 * bias * (mx @ beta) - 2 * np.sum(beta * cross, axis=0) - 2 * bias * my)
    return np.maximum(value, 0)


def fit_cv(dataset, arrays, rows, scales, setting, folds, output):
    started = time.monotonic()
    stats = subject_moments(arrays, rows)
    columns_by_arm = arm_columns(setting)
    positions = sorted({rows[i]['positions'][p] for i, _, p in arrays['examples']})
    subjects = sorted(folds)
    predictions = {arm: np.zeros((len(arrays['examples']), 2, 300, 2), np.float64) for arm in columns_by_arm}
    model_arrays = {}
    selections = []
    references = scales['positions']['hb'][dataset]
    for outer in range(setting['folds']):
        training = [subject for subject in subjects if folds[subject] != outer]
        held = [subject for subject in subjects if folds[subject] == outer]
        assert set(training).isdisjoint(held)
        for arm, columns in columns_by_arm.items():
            scores = [[] for _ in setting['ridge_fractions']]
            for inner in sorted(set(folds.values()) - {outer}):
                inner_train = [subject for subject in training if folds[subject] != inner]
                inner_held = [subject for subject in training if folds[subject] == inner]
                assert set(inner_train).isdisjoint(inner_held)
                inner_scores = [defaultdict(list) for _ in setting['ridge_fractions']]
                for position in positions:
                    for color in range(2):
                        moment, _ = training_moment(stats, inner_train, position, color)
                        models = ridge_path(moment, columns, setting['ridge_fractions'])
                        variance = np.asarray([references[position]['target_variance'][key][color] for key in 'so'])
                        for subject in inner_held:
                            valid_moment = stats.get((subject, position, color))
                            if valid_moment is None:
                                continue
                            for index, model in enumerate(models):
                                inner_scores[index][subject].append(float(np.mean(prediction_mse(
                                    valid_moment, columns, model) / variance)))
                for index, by_subject in enumerate(inner_scores):
                    scores[index].extend(float(np.mean(value)) for value in by_subject.values())
            selected = int(np.argmin([np.mean(value) for value in scores]))
            strength = setting['ridge_fractions'][selected]
            selections.append(dict(outer_fold=outer, arm=arm, training_subjects=training, held_subjects=held,
                inner_subject_site_mean_scores=[float(np.mean(value)) for value in scores], selected_ridge_fraction=strength))
            coefficients, biases, fallbacks = [], [], []
            for position in positions:
                color_beta, color_bias = [], []
                for color in range(2):
                    moment, fallback = training_moment(stats, training, position, color)
                    model = ridge_path(moment, columns, [strength])[0]
                    beta, bias = model
                    color_beta.append(beta)
                    color_bias.append(bias)
                    fallbacks.append(dict(position=position, color=color, pooled_fallback=fallback))
                    indices = [i for i, (row_index, _, p) in enumerate(arrays['examples'])
                               if rows[row_index]['subject'] in held and rows[row_index]['positions'][p] == position]
                    if indices:
                        features = arrays['features'][indices][:, :, columns].astype(float)
                        values = features @ beta + bias
                        predictions[arm][indices, :, :, color] = values.transpose(0, 2, 1)
                coefficients.append(color_beta)
                biases.append(color_bias)
            model_arrays[f'{arm}_fold{outer}_coefficient'] = np.asarray(coefficients)
            model_arrays[f'{arm}_fold{outer}_bias'] = np.asarray(biases)
            model_arrays[f'{arm}_columns'] = columns
            selections[-1]['position_fallbacks'] = fallbacks
            write(output / 'progress.json', dict(stage='nested_cv', dataset=dataset, outer_fold=outer,
                arm=arm, elapsed_seconds=time.monotonic() - started))
    model_arrays['metadata'] = np.asarray(json.dumps(dict(schema='multimodal_hb_ridge_cv_v1', dataset=dataset,
        subject_folds=folds, positions=positions, context_seconds=30, input='observed_X_only',
        output='continuous_component_diagnostic', ridge_selection=selections), ensure_ascii=False))
    np.savez_compressed(output / 'models.npz', **model_arrays)
    np.savez_compressed(output / 'oof_predictions.npz', **predictions)
    write(output / 'fit_selection.json', selections)
    return predictions


def evaluate(arrays, predictions, rows, output):
    batch = {key: arrays['targets'][:, index] for index, key in enumerate('xso')}
    batch['mask'] = arrays['mask']
    batch['v' + 'x'] = np.tile(arrays['variances'][:, 0, None, :], (1, 15, 20))
    for index, key in enumerate('so', start=1):
        batch['v' + key] = np.tile(arrays['variances'][:, index, None, :], (1, 15, 20))
    for key in 'xso':
        values = []
        for row_index, _, position in arrays['examples']:
            row = rows[row_index]
            values.append(row['_references'][row['positions'][position]]['difference_variance'][key])
        batch['dv' + key] = np.tile(np.asarray(values)[:, None, :], (1, 15, 20))
    summaries = {}
    for arm, value in predictions.items():
        estimated = dict(S=value[:, 0].reshape(-1, 15, 40), O=value[:, 1].reshape(-1, 15, 40))
        estimated['sum'] = estimated['S'] + estimated['O']
        result = joint_reconstruction_metrics(estimated, batch, 2, arrays['input_scales'])
        summary_record = joint_reconstruction_summary(result['metrics'], result['support'], arrays['examples'], rows)
        summary_record.update(status='completed', modality='hb', role='original_train_subject_OOF', arm=arm)
        summaries[arm] = summary_record
        folder = output / arm
        folder.mkdir(exist_ok=True)
        write(folder / 'summary.json', summary_record)
        np.savez_compressed(folder / 'metrics.npz', examples=arrays['examples'], **result['metrics'])
    return summaries


def dataset_worker(task):
    dataset, subjects, folds, rows, scales, cfg, output = task
    output = Path(output) / dataset
    output.mkdir(exist_ok=True)
    started = time.monotonic()
    with threadpool_limits(cfg['resources']['numerical_threads_per_worker']):
        prepared = output / 'prepared.npz'
        if prepared.exists():
            with np.load(prepared) as archive:
                arrays = {key: archive[key] for key in archive.files}
        else:
            arrays = prepare(dataset, subjects, rows, scales, cfg['diagnostic'], output)
        estimates = output / 'oof_predictions.npz'
        if estimates.exists():
            with np.load(estimates) as archive:
                predictions = {key: archive[key] for key in archive.files}
        else:
            predictions = fit_cv(dataset, arrays, rows, scales, cfg['diagnostic'], folds, output)
        summaries = evaluate(arrays, predictions, rows, output)
    support = read_json(output / 'observation_support.json')
    result = dict(status='completed', dataset=dataset, subject_folds=folds, subjects=len(subjects),
        position_windows=len(arrays['examples']), elapsed_seconds=time.monotonic() - started,
        validation_arrays_read=0, test_arrays_read=0,
        EEG_availability_reasons=dict(Counter(item['eeg_reason'] for item in support)),
        nmse={arm: value['metrics']['nmse']['macro'] for arm, value in summaries.items()})
    write(output / 'summary.json', result)
    return result


def read_json(path):
    return json.loads(Path(path).read_text())


def synthetic():
    rng = np.random.default_rng(20261123)
    values = np.arange(300, dtype=float)[:, None]
    features = lag_features(values, np.ones_like(values, bool), [-1, 0, 1])
    assert features[0, 0] == 10 and features[0, 2] == 0
    assert features[-1, 0] == 0 and features[-1, 2] == 289
    x = rng.normal(size=(6000, 2))
    signal = x[:, :1] * 2 + x[:, 1:] * .7
    y = np.c_[signal, x[:, :1] - signal]
    train = moment_matrix(x[:4000], y[:4000], np.ones(4000, bool))
    test = moment_matrix(x[4000:], y[4000:], np.ones(2000, bool))
    local = ridge_path(train, np.array([0]), [.001])[0]
    augmented = ridge_path(train, np.array([0, 1]), [.001])[0]
    local_mse = prediction_mse(test, np.array([0]), local)
    augmented_mse = prediction_mse(test, np.array([0, 1]), augmented)
    assert (local_mse > .4).all() and (augmented_mse < 1e-5).all()
    direct = np.mean((x[4000:] @ augmented[0] + augmented[1] - y[4000:])**2, axis=0)
    np.testing.assert_allclose(direct, augmented_mse, atol=1e-12)
    independent = rng.normal(size=(6000, 1))
    permuted = np.c_[x[:, 0], independent[:, 0]]
    null_train = moment_matrix(permuted[:4000], y[:4000], np.ones(4000, bool))
    null_test = moment_matrix(permuted[4000:], y[4000:], np.ones(2000, bool))
    null_model = ridge_path(null_train, np.array([0, 1]), [.001])[0]
    assert (prediction_mse(null_test, np.array([0, 1]), null_model) > .4).all()
    return dict(status='passed', signed_lag_and_crop_edges=True, known_auxiliary_signal_recovered=True,
        unrelated_auxiliary_not_falsely_recovered=True, moment_MSE_matches_direct=True,
        local_mse=local_mse, augmented_mse=augmented_mse)


def audit(output, cfg):
    """Only retained training-OOF arrays: paired subject comparisons and unsmoothed plots."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    plan = read_json(output / 'plan.json')
    root_result = read_json(output / 'summary.json')
    if root_result['status'] != 'completed' or root_result['signature'] != plan['signature']:
        raise ValueError('all four planned training-CV datasets must complete before audit')
    manifest, _, _, _ = base_tokenizer_training_data(cfg, ROOT)
    rows = manifest['rows']  # metadata only; signal reads below use retained train arrays exclusively.
    arms = cfg['diagnostic']['arms']
    metric_names = ['nmse', 'nrmse', 'rmse_native', 'bias_native', 'rms_gain', 'correlation',
                    'shape_nmse', 'slope_nrmse', 'curvature_nrmse', 'peak_timing_error_s',
                    'trough_timing_error_s', 'boundary_jump_rmse_native']
    all_subjects = {arm: {} for arm in arms}
    all_datasets = {arm: {} for arm in arms}
    audit_checks, selections, figure_entries = {}, {}, []
    tables, raw_datasets, extra_summaries = [], {}, {}
    figure_dir = output / 'figures'
    figure_dir.mkdir(exist_ok=True)
    colors = dict(local_hb='#0072B2', local_plus_other_hb='#E69F00', local_plus_eeg='#009E73',
                  local_plus_other_hb_eeg='#CC79A7')
    for dataset in sorted(plan['datasets']):
        folder = output / dataset
        with np.load(folder / 'prepared.npz') as source:
            arrays = {key: source[key] for key in source.files}
        with np.load(folder / 'oof_predictions.npz') as source:
            predictions = {key: source[key] for key in source.files}
        examples = arrays['examples']
        if any(rows[int(i)]['split'] != 'train' for i, _, _ in examples):
            raise ValueError('non-training numerical array in diagnostic')
        target = arrays['targets'].reshape(-1, 3, 300, 2)
        mask = arrays['mask'].reshape(-1, 300, 2)
        count = mask.sum(1)
        all_metrics, max_nmse_delta, max_rmse_delta = {}, 0., 0.
        cancellation = {}
        for arm in arms:
            summary_record = read_json(folder / arm / 'summary.json')
            all_datasets[arm][dataset] = summary_record
            by_subject = summary_record['metrics']['nmse']['by_subject']
            all_subjects[arm][dataset] = {v['subject']: np.asarray(v['value']) for v in by_subject}
            with np.load(folder / arm / 'metrics.npz') as source:
                metrics = {key: source[key] for key in source.files if key != 'examples'}
            all_metrics[arm] = metrics
            estimate = np.concatenate([predictions[arm].sum(1)[:, None], predictions[arm]], axis=1)
            errors = estimate - target
            mse = np.where(mask[:, None], errors**2, 0).sum(2) / count[:, None]
            direct_nmse = mse / arrays['variances'][:, [0, 1, 2], :]
            direct_rmse = np.sqrt(mse) * arrays['input_scales'][:, None, None]
            max_nmse_delta = max(max_nmse_delta, float(np.max(abs(direct_nmse - metrics['nmse']))))
            max_rmse_delta = max(max_rmse_delta, float(np.max(abs(direct_rmse - metrics['rmse_native']))))
            component_energy = mse[:, 1] + mse[:, 2]
            cross = np.where(mask, errors[:, 1] * errors[:, 2], 0).sum(1) / count
            cancel = np.divide(-2 * cross, component_energy, out=np.full_like(cross, np.nan), where=component_energy > 0)
            extra = dict(SO_error_cancellation_fraction=np.repeat(cancel[:, None], 3, axis=1))
            cancellation[arm] = base_training_aggregate(dict(nmse=metrics['nmse'], **extra), examples, rows)['metrics']['SO_error_cancellation_fraction']
            for name in metric_names:
                item = summary_record['metrics'][name]
                for component, label in enumerate(['X', 'S', 'O']):
                    for color, pigment in enumerate(['HbO', 'HbR']):
                        tables.append(dict(dataset=dataset, arm=arm, metric=name, component=label,
                            color=pigment, mean=item['macro'][component][color],
                            source=str(folder / arm / 'summary.json')))
        audit_checks[dataset] = dict(independent_NMSE_max_abs_delta=max_nmse_delta,
            independent_native_RMSE_max_abs_delta=max_rmse_delta,
            subjects=len(plan['datasets'][dataset]), physical_crops=sum(map(len, plan['datasets'][dataset].values())),
            position_windows=len(examples), valid_points=int(mask.sum()), unscored_colors=int((count == 0).sum()),
            closure_X_minus_S_minus_O_max_abs=float(np.max(abs(np.where(mask, target[:, 0] - target[:, 1] - target[:, 2], 0)))))
        if max_nmse_delta > 1e-9 or max_rmse_delta > 1e-9:
            raise ValueError('independent score audit failed')
        extra_summaries[dataset] = cancellation
        local_error = all_metrics['local_hb']['nmse'][:, 1:].mean((1, 2))
        combined_error = all_metrics['local_plus_other_hb_eeg']['nmse'][:, 1:].mean((1, 2))
        identity_order = sorted(range(len(examples)), key=lambda i: (rows[examples[i, 0]]['subject'],
            rows[examples[i, 0]]['record'], rows[examples[i, 0]]['parent'], int(examples[i, 1]),
            rows[examples[i, 0]]['positions'][examples[i, 2]]))
        finite_order = np.flatnonzero(np.isfinite(local_error))[np.argsort(local_error[np.isfinite(local_error)], kind='stable')]
        selected = [('first_identity', identity_order[0]), ('local_P90', int(finite_order[int(.9 * (len(finite_order) - 1))])),
                    ('local_worst', int(finite_order[-1])), ('combined_worst', int(np.nanargmax(combined_error)))]
        selections[dataset] = []
        for role, index in selected:
            i, start, pos = (int(v) for v in examples[index])
            row = rows[i]
            scale = arrays['input_scales'][index]
            y = target[index] * scale
            observed = mask[index]
            predicted = {arm: np.concatenate([value[index].sum(0)[None], value[index]], axis=0) * scale
                         for arm, value in predictions.items()}
            patch_error = np.where(observed, (predicted['local_hb'][1:] - y[1:])**2, 0).reshape(2, 15, 20, 2).sum((0, 2, 3))
            zoom_patch = int(np.argmax(patch_error))
            meta = dict(role=role, example=[i, start, pos], index=index, dataset=dataset,
                subject=row['subject'], record=row['record'], parent=row['parent'], position=row['positions'][pos],
                outer_fold=plan['subject_folds'][dataset][row['subject']], original_split='train',
                native_start_sample=row.get('native_start_sample'), native_rate_hz=row.get('native_rate_hz'),
                crop_start_s=start, original_unit=row.get('source_unit'), fixed_G0_scale=scale,
                baseline_mean_SO_nmse=float(local_error[index]), combined_mean_SO_nmse=float(combined_error[index]),
                zoom_patch=zoom_patch, zoom_rule='maximum_local_baseline_SO_native_squared_error_2s_patch_within_selected_crop')
            selections[dataset].append(meta)
            for view in ['full_30s', 'local_centered_2s']:
                fig, axes = plt.subplots(3, 2, figsize=(13.0, 9.1), sharex=True)
                full_time = np.arange(300) / 10
                sample = slice(None) if view == 'full_30s' else slice(zoom_patch * 20, (zoom_patch + 1) * 20)
                t = full_time[sample]
                for component, label in enumerate(['X = S + O', 'S', 'O']):
                    for color, pigment in enumerate(['HbO', 'HbR']):
                        ax = axes[component, color]
                        valid = observed[sample, color]
                        target_line = y[component, sample, color].copy()
                        if view != 'full_30s':
                            target_line -= np.mean(target_line[valid])
                        ax.plot(t, np.where(valid, target_line, np.nan), color='black', linewidth=1.35, label='target')
                        for arm in arms:
                            line = predicted[arm][component, sample, color].copy()
                            if view != 'full_30s':
                                line -= np.mean(line[valid])
                            ax.plot(t, np.where(valid, line, np.nan), color=colors[arm], linewidth=1.05,
                                    alpha=.9, label=arm.replace('local_plus_', '+').replace('_', ' '))
                        for boundary in np.arange(0, 31, 2):
                            if t[0] <= boundary <= t[-1]:
                                ax.axvline(boundary, color='.7', linewidth=.5, linestyle=':')
                        if not valid.all():
                            ax.fill_between(t, 0, 1, where=~valid, transform=ax.get_xaxis_transform(), color='grey', alpha=.25, label='unscored mask')
                        ax.set_title(f'{label} / {pigment}')
                        ax.set_ylabel('subject_normalized_Hb' + (' (demeaned)' if view != 'full_30s' else ''))
                        ax.grid(alpha=.15)
                        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
                axes[-1, 0].set_xlabel('time in visible crop (s)')
                axes[-1, 1].set_xlabel('time in visible crop (s)')
                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9)
                fig.suptitle(f'{dataset} / {row["subject"]} / {row["positions"][pos]} / start {start}s / {role}\n'
                    f'Original train, held subject fold {meta["outer_fold"]}; {view}; shared target/prediction coordinates\n'
                    f'S/O mean NMSE: local {local_error[index]:.3f}, all auxiliary {combined_error[index]:.3f}; no smoothing', fontsize=12)
                fig.tight_layout(rect=(0, .055, 1, .90))
                path = figure_dir / f'{dataset}__{role}__{view}.png'
                fig.savefig(path, dpi=220)
                plt.close(fig)
                figure_entries.append(dict(path=str(path), sample=meta, view=view))
        raw_datasets[dataset] = dict(metrics=all_metrics, examples=examples)
    comparisons = {}
    baseline = 'local_hb'
    baseline_macro = np.mean([np.mean(np.stack(list(values.values())), axis=0)
                              for values in all_subjects[baseline].values()], axis=0)
    rng = np.random.default_rng(cfg['diagnostic']['bootstrap']['seed'])
    for arm in arms[1:]:
        by_dataset, paired = {}, []
        for dataset in sorted(plan['datasets']):
            names = sorted(all_subjects[baseline][dataset])
            a = np.stack([all_subjects[baseline][dataset][name] for name in names])
            b = np.stack([all_subjects[arm][dataset][name] for name in names])
            gain = 1 - b[:, 1:].mean((1, 2)) / a[:, 1:].mean((1, 2))
            by_dataset[dataset] = dict(subjects=names, baseline_NMSE=a.mean(0), candidate_NMSE=b.mean(0),
                SO_relative_gain=1 - b[:, 1:].mean() / a[:, 1:].mean(),
                all_four_SO_colors_improve=bool(np.all(b.mean(0)[1:] < a.mean(0)[1:])),
                maximum_SO_cell_ratio=float(np.max(b.mean(0)[1:] / a.mean(0)[1:])),
                positive_subjects=int((gain > 0).sum()), planned_subjects=len(names))
            paired.extend(dict(dataset=dataset, subject=name, relative_SO_gain=float(value)) for name, value in zip(names, gain))
        candidate_macro = np.mean([np.mean(np.stack(list(values.values())), axis=0)
                                   for values in all_subjects[arm].values()], axis=0)
        bootstrap = []
        for _ in range(cfg['diagnostic']['bootstrap']['replicates']):
            a_groups, b_groups = [], []
            for dataset in sorted(plan['datasets']):
                names = sorted(all_subjects[baseline][dataset])
                a = np.stack([all_subjects[baseline][dataset][name][1:] for name in names])
                b = np.stack([all_subjects[arm][dataset][name][1:] for name in names])
                selected = rng.integers(0, len(names), len(names))
                a_groups.append(a[selected].mean()); b_groups.append(b[selected].mean())
            bootstrap.append(1 - np.mean(b_groups) / np.mean(a_groups))
        interval = np.quantile(bootstrap, [.025, .975])
        relative_gain = 1 - candidate_macro[1:].mean() / baseline_macro[1:].mean()
        positive_fraction = np.mean([value['relative_SO_gain'] > 0 for value in paired])
        rule = cfg['diagnostic']['clear_gain_rule']
        checks = dict(macro_SO_gain_at_least_5_percent=relative_gain >= rule['minimum_macro_SO_relative_improvement'],
            each_SO_color_macro_improves=bool(np.all(candidate_macro[1:] < baseline_macro[1:])),
            at_least_60_percent_subjects_improve=positive_fraction >= rule['minimum_positive_subject_fraction'],
            at_least_three_datasets_all_four_SO_cells_improve=sum(v['all_four_SO_colors_improve'] for v in by_dataset.values()) >= rule['minimum_datasets_all_SO_colors_improve'],
            no_dataset_SO_cell_worsens_over_10_percent=max(v['maximum_SO_cell_ratio'] for v in by_dataset.values()) <= rule['maximum_dataset_SO_color_ratio'],
            paired_subject_bootstrap_lower_bound_positive=interval[0] > rule['paired_subject_bootstrap_lower_bound_relative_improvement'])
        comparisons[arm] = dict(baseline_macro_NMSE=baseline_macro, candidate_macro_NMSE=candidate_macro,
            macro_SO_relative_gain=float(relative_gain), paired_subject_positive_fraction=float(positive_fraction),
            relative_gain_percentile_95_interval=interval, bootstrap_unit='subject_within_dataset_conditional_on_frozen_OOF_fits_no_refitting',
            by_dataset=by_dataset, paired_subjects=paired, preregistered_checks=checks,
            clear_gain_pass=all(checks.values()))
    with (output / 'metric_comparison.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tables[0])); writer.writeheader(); writer.writerows(tables)
    outcome = dict(schema='multimodal_hb_train_CV_audit_v1', status='numerical_completed_visual_review_pending',
        numerical_split='train', validation_arrays_read=0, confirmation_arrays_read=0, test_arrays_read=0,
        baseline_macro_NMSE=baseline_macro, comparisons=comparisons, independent_checks=audit_checks,
        error_cancellation=extra_summaries, examples=selections, figures=figure_entries,
        units='subject_normalized_Hb_preexisting_G0_preparation',
        normalization='NMSE = valid-time MSE / frozen G0 train target variance; native RMSE multiplies frozen train input scale',
        hierarchy='valid time -> position -> physical crop -> record -> subject -> dataset, equal at each aggregation level',
        diagnostic_scope='continuous linear nested ridge/FIR, 30s visible X; neither a discrete candidate nor an exhaustive multimodal upper bound',
        interpretation='Multimodal improvement is supported only if all preregistered checks pass and actual image review supports the gain')
    write(output / 'audit.json', outcome)
    print(json.dumps({arm: dict(SO_relative_gain=value['macro_SO_relative_gain'],
        interval=value['relative_gain_percentile_95_interval'].tolist(), pass_all=value['clear_gain_pass'])
        for arm, value in comparisons.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'experiments/configs/physiology_semantic_tokenizer/dual_component_multimodal_hb_cv_v1.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['plan', 'synthetic', 'run', 'audit'], required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if (cfg['schema'] != 'dual_component_multimodal_hb_cv_v1'
            or cfg['diagnostic']['numerical_read_splits'] != ['train']
            or cfg['diagnostic']['context_seconds'] != 30 or cfg['diagnostic']['folds'] != 3):
        raise ValueError('versioned train-only 30-s subject-CV contract required')
    args.output.mkdir(exist_ok=True, parents=True)
    if args.stage == 'audit':
        audit(args.output, cfg)
        return
    if args.stage == 'synthetic':
        write(args.output / 'synthetic.json', synthetic())
        return
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    rows = manifest['rows']
    selected = make_plan(rows, cfg)
    signature = dict(config=digest(cfg), G0=manifest['cfg_identity'], scales=manifest['scales_identity'], plan=digest(selected))
    if args.stage == 'plan':
        path = args.output / 'plan.json'
        if path.exists() and read_json(path)['signature'] != signature:
            raise ValueError('retained plan identity differs')
        write(path, dict(signature=signature, config=cfg, **selected, measured_arrays_read=0,
            normalization_note='G0 fixed original train statistics include diagnostic CV-held training subjects; all new ridge statistics fit disjoint CV subjects'))
        return
    if read_json(args.output / 'plan.json')['signature'] != signature:
        raise ValueError('freeze matching plan before signal arrays')
    if read_json(args.output / 'synthetic.json')['status'] != 'passed':
        raise ValueError('synthetic validation required before measured arrays')
    for row in rows:
        row['_references'] = scales['positions'][row['modality']][row['dataset']]
    tasks = [(dataset, subjects, selected['subject_folds'][dataset], rows, scales, cfg, args.output)
             for dataset, subjects in sorted(selected['datasets'].items())]
    started = time.monotonic()
    write(args.output / 'controller.json', dict(status='running', signature=signature,
                                               started_unix=time.time(), planned_datasets=[t[0] for t in tasks]))
    results, failures = [], {}
    with ProcessPoolExecutor(max_workers=cfg['resources']['dataset_workers']) as pool:
        futures = {pool.submit(dataset_worker, task): task[0] for task in tasks}
        for future in as_completed(futures):
            dataset = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                failures[dataset] = dict(type=type(error).__name__, message=str(error))
            write(args.output / 'controller.json', dict(status='running', signature=signature,
                completed_datasets=[result['dataset'] for result in results], failures=failures,
                elapsed_seconds=time.monotonic() - started))
    write(args.output / 'summary.json', dict(status='completed' if not failures else 'completed_with_failures',
        signature=signature, datasets={result['dataset']: result for result in results}, failures=failures,
        validation_arrays_read=0, test_arrays_read=0, elapsed_seconds=time.monotonic() - started,
        phase='train_only_continuous_predictability_diagnostic_pending_paired_summary_and_visual_audit'))
    write(args.output / 'controller.json', dict(status='completed' if not failures else 'failed', signature=signature,
        completed_datasets=[result['dataset'] for result in results], failures=failures,
        elapsed_seconds=time.monotonic() - started))


if __name__ == '__main__':
    main()
