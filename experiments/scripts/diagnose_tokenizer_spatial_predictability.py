#!/usr/bin/env python3
"""Train-only subject-held-out spatial predictability on the frozen G0 EEG grid."""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '2')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from threadpoolctl import threadpool_limits
import yaml

from experiments.scripts.run_dual_component_tokenizer import identity
from src.data.dual_component_tokenizer import base_tokenizer_training_data
from src.data.dual_component_tokenizer import base_tokenizer_joint_eeg as read_window


def write(path, value):
    def encode(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path = Path(path)
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value, indent=2, default=encode, allow_nan=False) + '\n')
    temporary.replace(path)


def rank(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def plan(rows, setting):
    records = defaultdict(list)
    for index, row in enumerate(rows):
        if row['modality'] == 'eeg' and row['split'] == 'train':
            for crop in row['inputs']['score']:
                records[row['dataset'], row['subject'], row['record']].append(
                    dict(row=index, start_s=crop['start_s'], positions=crop['positions']))
    subjects = defaultdict(list)
    for key, crops in sorted(records.items()):
        chosen = sorted(crops, key=lambda item: rank((key, item)))[:setting['windows_per_record']]
        subjects[key[:2]].extend(chosen)
    result = defaultdict(dict)
    for (dataset, subject), crops in sorted(subjects.items()):
        result[dataset][subject] = sorted(crops, key=lambda item: rank((dataset, subject, item)))[:setting['windows_per_subject']]
    return dict(result)


def band_values(values, bands):
    spectrum = np.fft.rfft(values, axis=0)
    frequencies = np.fft.rfftfreq(len(values), 1 / 200)
    return [np.fft.irfft(spectrum * ((frequencies >= lo) & (frequencies < hi))[:, None], n=len(values), axis=0)
            for lo, hi in bands]


def fit_other(gram, cross, ridge_fraction):
    """Solve each target with its own electrode column constrained to zero."""
    count = len(cross)
    regularized = gram.copy()
    diagonal = np.arange(count)
    strength = ridge_fraction * np.trace(gram, axis1=1, axis2=2) / count + 1e-12
    regularized[:, diagonal, diagonal] += strength[:, None]
    for target in range(count):
        regularized[target, target, :] = 0
        regularized[target, :, target] = 0
        regularized[target, target, target] = 1
    right = cross.copy()
    right[diagonal, diagonal] = 0
    return np.linalg.solve(regularized, right[..., None])[..., 0].T


def dataset_diagnostic(task):
    dataset, subjects, rows, scales, cfg, output = task
    started = time.monotonic()
    setting = cfg['diagnostic']
    bands = setting['bands_hz']
    names = sorted({name for subject in subjects.values() for crop in subject
                    for name in rows[crop['row']]['positions']}, key=str.upper)
    if len({name.upper() for name in names}) != len(names):
        raise ValueError('case-only native electrode duplicates require canonical identity')
    count = len(names)
    folds = {subject: index % setting['folds'] for index, subject in enumerate(sorted(subjects, key=lambda s: rank((dataset, s))))}
    cache = OrderedDict()
    stats = {}
    support = []
    with threadpool_limits(cfg['resources']['numerical_threads_per_worker']):
        for subject, crops in subjects.items():
            totals = defaultdict(lambda: np.zeros((len(bands), count, count, count)))
            cross = defaultdict(lambda: np.zeros((len(bands), count, count)))
            denominator = np.zeros(count)
            for crop in crops:
                fields, input_valid, valid = read_window(rows, crop, scales, names, cache)
                support.append(dict(dataset=dataset, subject=subject, **crop, observed_electrodes=int(input_valid.sum()),
                                    scored_complete_electrodes=int(valid.sum()), planned_electrodes=len(crop['positions'])))
                denominator += valid
                filtered = {key: band_values(value, bands) for key, value in fields.items()}
                for band in range(len(bands)):
                    xx = filtered['x'][band]
                    for branch in ('x', 's', 'o'):
                        yy = filtered[branch][band]
                        for predictor, values in [('X', xx), ('oracle', yy)]:
                            key = branch + '_' + predictor
                            gram = values.T @ values / len(values)
                            totals[key][band, valid] += gram
                            cross[key][band, valid] += (values.T @ yy / len(values))[:, valid].T
            stats[subject] = dict(gram={key: value / np.maximum(denominator, 1)[None, :, None, None] for key, value in totals.items()},
                                  cross={key: value / np.maximum(denominator, 1)[None, :, None] for key, value in cross.items()})
            write(output / (dataset + '_progress.json'), dict(stage='train_statistics', completed_subjects=len(stats), planned_subjects=len(subjects)))
        models = {}
        for fold in sorted(set(folds.values())):
            training = [subject for subject in subjects if folds[subject] != fold]
            if not training:
                raise ValueError('at least two disjoint subject folds required')
            fitted = {}
            for branch in ('x', 's', 'o'):
                for predictor in ('X', 'oracle'):
                    key = branch + '_' + predictor
                    gram = np.mean([stats[s]['gram'][key] for s in training], axis=0)
                    cross = np.mean([stats[s]['cross'][key] for s in training], axis=0)
                    fitted[key] = np.stack([fit_other(g, c, setting['ridge_fraction']) for g, c in zip(gram, cross)])
                    if predictor == 'X':
                        diag = np.arange(count)
                        fitted[branch + '_local'] = cross[:, diag, diag] / np.maximum(gram[:, diag, diag, diag], 1e-12)
            models[fold] = fitted
        scores = []
        for subject, crops in subjects.items():
            model = models[folds[subject]]
            for crop in crops:
                fields, input_valid, valid = read_window(rows, crop, scales, names, cache)
                filtered = {key: band_values(value, bands) for key, value in fields.items()}
                for branch in ('x', 's', 'o'):
                    for predictor in ('local', 'X', 'oracle'):
                        for band, (lo, hi) in enumerate(bands):
                            truth = filtered[branch][band]
                            if predictor == 'local':
                                prediction = filtered['x'][band] * model[branch + '_local'][band][None]
                            else:
                                values = filtered['x' if predictor == 'X' else branch][band]
                                prediction = values @ model[branch + '_' + predictor][band]
                            error = prediction - truth
                            target_energy = np.mean(truth * truth, axis=0)
                            mse = np.mean(error * error, axis=0)
                            correlation = np.sum(prediction * truth, axis=0) / np.maximum(
                                np.sqrt(np.sum(prediction**2, axis=0) * np.sum(truth**2, axis=0)), 1e-20)
                            gain = np.sqrt(np.mean(prediction**2, axis=0) / np.maximum(target_energy, 1e-20))
                            for position in np.flatnonzero(valid):
                                scores.append(dict(subject=subject, record=rows[crop['row']]['record'], row=crop['row'],
                                    start_s=crop['start_s'], electrode=names[position], branch=branch.upper(), predictor=predictor,
                                    band_hz=[lo, hi], mse=float(mse[position]), target_energy=float(target_energy[position]),
                                    energy_nmse=float(mse[position] / max(target_energy[position], 1e-20)),
                                    correlation=float(correlation[position]), rms_gain=float(gain[position])))
            write(output / (dataset + '_progress.json'), dict(stage='held_subject_evaluation', subject=subject, planned_subjects=len(subjects)))
        write(output / (dataset + '_scores.json'), scores)
        grouped = defaultdict(list)
        for score in scores:
            grouped[score['branch'], score['predictor'], tuple(score['band_hz']), score['subject']].append(score)
        subject_scores = []
        for (branch, predictor, band, subject), values in sorted(grouped.items()):
            by_record = defaultdict(list)
            for value in values:
                by_record[value['record']].append(value)
            summary = {metric: float(np.mean([np.mean([value[metric] for value in record]) for record in by_record.values()]))
                       for metric in ('energy_nmse', 'correlation', 'rms_gain')}
            subject_scores.append(dict(subject=subject, branch=branch, predictor=predictor, band_hz=band, **summary))
        grouped = defaultdict(list)
        for score in subject_scores:
            grouped[score['branch'], score['predictor'], tuple(score['band_hz'])].append(score)
        summary = [dict(branch=key[0], predictor=key[1], band_hz=key[2],
                        **{metric: float(np.mean([value[metric] for value in values])) for metric in ('energy_nmse', 'correlation', 'rms_gain')})
                   for key, values in sorted(grouped.items())]
        result = dict(status='completed', dataset=dataset, native_electrodes=names, subject_folds=folds,
                      support=support, summary=summary, by_subject=subject_scores,
                      planned_windows=len(support), scored_windows=sum(s['scored_complete_electrodes'] > 0 for s in support),
                      elapsed_seconds=time.monotonic() - started, validation_arrays_read=0)
        write(output / (dataset + '_summary.json'), result)
        return result


def synthetic():
    rng = np.random.default_rng(19)
    values = rng.normal(size=(20000, 4))
    values[:, 1] = values[:, 0] * 2
    gram = values.T @ values / len(values)
    grams = np.tile(gram[None], (4, 1, 1))
    weights = fit_other(grams, gram.T, 1e-4)
    np.testing.assert_array_equal(np.diag(weights), 0)
    prediction = values @ weights
    assert np.mean((prediction[:, 0] - values[:, 0])**2) < 1e-6
    assert np.mean((prediction[:, 2] - values[:, 2])**2) > .9
    missing = values.copy()
    missing[:, 1] = 0
    assert np.isfinite(missing @ weights).all()
    repeated_grams = np.tile(grams[None], (2, 1, 1, 1))
    repeated_cross = np.tile(gram.T[None], (2, 1, 1))
    diagonal = np.arange(4)
    np.testing.assert_allclose(repeated_cross[:, diagonal, diagonal] / repeated_grams[:, diagonal, diagonal, diagonal], 1)
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        x = rng.normal(size=(6000, 3, 1)).astype(np.float32)
        for key, value in [('x', x), ('s', x * .5), ('o', x * .5),
                           ('mask', np.ones(x.shape, bool)), ('observation_valid', np.ones(x.shape, bool))]:
            np.save(folder / (key + '.npy'), value)
        row = dict(split='train', modality='eeg', dataset='fixture', rate=200, positions=['A', 'B', 'C'],
                   folder=str(folder), inputs={'score': [dict(start_s=0, positions=[0])]})
        ref = dict(scale=1., target_variance={key: [1.] for key in ('x', 's', 'o')},
                   difference_variance={key: [1.] for key in ('x', 's', 'o')})
        scales = dict(positions={'eeg': {'fixture': {name: ref for name in row['positions']}}})
        fields, observed, scored = read_window([row], dict(row=0, start_s=0, positions=[0]), scales, row['positions'], OrderedDict())
        assert observed.all() and scored.tolist() == [True, False, False]
        np.testing.assert_array_equal(fields['x'], x[:, :, 0])
        for key in ('s', 'o'):
            changed = x.copy()
            changed[:, 1:] = np.nan
            np.save(folder / (key + '.npy'), changed)
        fields_after, _, _ = read_window([row], dict(row=0, start_s=0, positions=[0]), scales, row['positions'], OrderedDict())
        np.testing.assert_array_equal(fields_after['x'], fields['x'])
    return dict(status='passed', target_electrode_contribution='exact_zero', missing_predictor='fixed_zero_contribution',
                correlated_neighbor_recovered=True, independent_neighbor_not_falsely_recovered=True,
                observed_auxiliary_without_teacher_is_retained=True, auxiliary_teacher_changes_do_not_change_input=True)


def render_summary(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    payload = json.loads((output / 'summary.json').read_text())
    fig, axes = plt.subplots(3, 4, figsize=(15, 8), constrained_layout=True)
    labels = dict(eeg_fnirs_single_trial='Single-Trial', refed='REFED', simultaneous_eeg_nirs='Simultaneous',
                  visual_cognitive_motivation='Visual')
    for column, (dataset, values) in enumerate(sorted(payload['datasets'].items())):
        for row, branch in enumerate(('S', 'O', 'X')):
            selected = [value for value in values if value['branch'] == branch]
            predictors = ('local', 'X', 'oracle')
            bands = sorted({tuple(value['band_hz']) for value in selected})
            matrix = np.array([[next(value['energy_nmse'] for value in selected
                                     if value['predictor'] == predictor and tuple(value['band_hz']) == band)
                                for band in bands] for predictor in predictors])
            ax = axes[row, column]
            colored = ax.imshow(matrix, vmin=0, vmax=2.5, cmap='magma_r', aspect='auto')
            for y, x in np.ndindex(matrix.shape):
                ax.text(x, y, f'{matrix[y, x]:.2f}', ha='center', va='center',
                        color='white' if matrix[y, x] > 1.25 else 'black', fontsize=10)
            ax.set_xticks(range(5), [f'{lo}-{hi}' for lo, hi in bands])
            ax.set_yticks(range(3), ['local X', 'other X', 'other target oracle'] if column == 0 else ['', '', ''])
            ax.set_title(f'{labels[dataset]} / {branch}', fontsize=12)
            ax.set_xlabel('Hz')
    fig.colorbar(colored, ax=axes, label='Band MSE / target mean-square', shrink=.85)
    fig.suptitle('Train-only spatial predictability\nTarget electrode excluded from neighbor inputs; five subject-disjoint folds')
    fig.savefig(output / 'spatial_predictability_v2.png', dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'experiments/configs/physiology_semantic_tokenizer/dual_component_spatial_diagnostic_v1.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    cfg = yaml.safe_load(args.config.read_text())
    if (cfg['schema'] != 'dual_component_spatial_diagnostic_v1' or cfg['diagnostic']['read_splits'] != ['train']
            or not cfg['diagnostic']['exclude_target_electrode_from_neighbors']
            or cfg['diagnostic']['input_support'] != 'observation_valid_only'):
        raise ValueError('versioned train-only observed spatial diagnostic required')
    if args.synthetic:
        write(args.output / 'synthetic.json', synthetic())
        return
    if (args.output / 'summary.json').exists():
        raise FileExistsError('retained completed diagnostic')
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    rows = manifest['rows']
    selected = plan(rows, cfg['diagnostic'])
    write(args.output / 'resolved_config.json', dict(config=cfg, config_identity=identity(cfg),
        input_identity=manifest['cfg_identity'], scales_identity=manifest['scales_identity'], plan=selected,
        source_head=os.popen('git rev-parse HEAD').read().strip(), started_unix=time.time(),
        normalization_note='G0 scales use original train population including diagnostic held subjects; predictors fit disjoint subjects'))
    tasks = [(dataset, subjects, rows, scales, cfg, args.output) for dataset, subjects in sorted(selected.items())]
    with ProcessPoolExecutor(max_workers=cfg['resources']['dataset_workers']) as pool:
        results = list(pool.map(dataset_diagnostic, tasks))
    write(args.output / 'summary.json', dict(status='completed', config_identity=identity(cfg),
        datasets={result['dataset']: result['summary'] for result in results},
        planned_windows={result['dataset']: result['planned_windows'] for result in results},
        scored_windows={result['dataset']: result['scored_windows'] for result in results},
        denominator='per-window per-electrode band MSE / target band mean-square, then electrode/window within record, record within subject, subject within dataset',
        validation_arrays_read=0, target='conditional v3 teacher S/O or observed X; same-branch oracle is privileged diagnostic',
        eligibility='all fixed plan failures retained in dataset support tables'))
    render_summary(args.output)


if __name__ == '__main__':
    main()
