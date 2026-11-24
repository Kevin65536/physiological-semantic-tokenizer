"""Fixed-seed paired contrasts on the four terminal development score panels."""

import csv

import numpy as np
import torch

import run_single_vq_dynamic_factorial as study
from run_dual_component_tokenizer import identity, now, read_json, write_json

ARMS = ('continuous_wave', 'continuous_dynamic', 'hard_wave', 'hard_dynamic')
FEATURES = study.PLAN['features']
COMPONENTS = ('X', 'S', 'O')
ENDPOINTS = ('spectral_mse', *FEATURES, 'macro_nmse', *('nmse_' + c for c in COMPONENTS))
CONTRASTS = {
    'continuous_wave_minus_dynamic': (1, -1, 0, 0),
    'hard_wave_minus_dynamic': (0, 0, 1, -1),
    'hard_minus_continuous_benefit': (-1, 1, 1, -1),
    'wave_hard_minus_continuous': (-1, 0, 1, 0),
    'dynamic_hard_minus_continuous': (0, -1, 0, 1),
}


def save_table(name, rows):
    with (study.OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def endpoints(decoded, nmse):
    decoded = np.asarray(decoded).reshape(4)
    nmse = np.asarray(nmse).reshape(3)
    return np.r_[decoded.mean(), decoded, nmse.mean(), nmse]


def person_values(summary, metric):
    return {
        (row['dataset'], row['subject']): np.asarray(row['value']).ravel()
        for row in summary['metrics'][metric]['by_subject']
    }


def paired_means(values, datasets, draws=10000):
    """Resample people jointly across all cells and endpoints within each dataset."""
    rng = np.random.default_rng(study.SEED)
    point, replicates = [], []
    for dataset in sorted(set(datasets)):
        block = values[datasets == dataset]
        point.append(block.mean(axis=0))
        indices = rng.integers(len(block), size=(draws, len(block)))
        replicates.append(block[indices].mean(axis=1))
    return np.mean(point, axis=0), np.mean(replicates, axis=0)


def reconstruction_checks(wave, dynamic):
    limits = study.parent.PLAN['guardrails']

    def macro(summary, name):
        return np.asarray(summary['metrics'][name]['macro']).ravel()

    nmse_relative = macro(dynamic, 'nmse') / macro(wave, 'nmse') - 1
    correlation_delta = macro(dynamic, 'correlation') - macro(wave, 'correlation')
    macro_relative = dynamic['selection_score'] / wave['selection_score'] - 1
    checks = {
        'macro_nmse': bool(macro_relative <= limits['macro_nmse_relative_increase_max']),
        'component_nmse': bool(np.all(nmse_relative <= limits['component_nmse_relative_increase_max'])),
        'component_correlation': bool(np.all(correlation_delta >= -limits['component_correlation_drop_max'])),
    }
    return dict(
        passed=all(checks.values()),
        checks=checks,
        macro_nmse_relative_change=macro_relative,
        component_nmse_relative_change=nmse_relative.tolist(),
        component_correlation_change=correlation_delta.tolist(),
    )


def verify_pairing(paths):
    parent = study.parent
    warm_path = parent.OUT / f'seed_{study.SEED}/models/eeg__waveform__warmup'
    warm = read_json(warm_path / 'summary.json')
    if warm['status'] != 'completed' or warm['completed_step'] != study.PLAN['training']['warmup_steps']:
        raise ValueError('incomplete shared warmup')
    if warm['signature']['config'] != identity(parent.training_cfg(study.SEED)):
        raise ValueError('wrong warmup configuration')
    common_rng = None
    for arm, path in paths.items():
        state = torch.load(path / 'checkpoint.pt', weights_only=False, map_location='cpu')
        train = read_json(path / 'summary.json')
        sig = dict(warm['signature'], task=parent.TASK if arm.startswith('hard') else study.TASK)
        if arm.endswith('dynamic'):
            sig['spectral_objective'] = identity(study.objective('continuous_dynamic'))
        if (
            state['signature'] != sig
            or train['signature'] != sig
            or train['status'] != 'completed'
            or state['step'] != study.PLAN['training']['total_steps']
            or train['completed_step'] != state['step']
        ):
            raise ValueError(f'unmatched terminal training signature for {arm}')
        if common_rng is not None and state['numpy_rng'] != common_rng:
            raise ValueError(f'different terminal training sampler state for {arm}')
        common_rng = state['numpy_rng']
    if read_json(paths['hard_wave'] / 'initialization.json') != read_json(
        paths['hard_dynamic'] / 'initialization.json'
    ):
        raise ValueError('hard initialization panels differ')
    counts = {}
    for ds in parent.capacity.DATASETS:

        def file(arm):
            suffix = '_validation' if arm.startswith('hard') else ''
            return paths[arm] / 'evaluation' / f'{ds}{suffix}.npz'

        with np.load(file('hard_wave')) as ref:
            examples, valid, target = (ref[k] for k in ('examples', 'valid', 'target'))
        if (
            valid.shape != (len(examples), 15)
            or target.shape != (valid.sum(), 4)
            or not np.isfinite(target).all()
        ):
            raise ValueError(f'invalid complete patch support for {ds}')
        for arm in ARMS:
            with np.load(file(arm)) as data:
                np.testing.assert_array_equal(data['examples'], examples)
                np.testing.assert_array_equal(data['valid'], valid)
                np.testing.assert_allclose(data['target'], target, rtol=1e-5, atol=1e-5)
        counts[ds] = dict(positions=len(examples), complete_patches=int(valid.sum()))
    return dict(
        shared_warmup=str(warm_path.relative_to(parent.ROOT)),
        warmup_signature=warm['signature'],
        same_terminal_sampler=True,
        hard_initialization_equal=True,
        matched_development_support=counts,
    )


def summarize():
    parent = study.parent
    paths = {arm: study.OUT / arm for arm in study.ARMS}
    paths.update(hard_wave=parent.folder(study.SEED, 'A'), hard_dynamic=parent.folder(study.SEED, 'B'))
    pairing = verify_pairing(paths)
    results = {arm: read_json(path / 'evaluation/summary.json') for arm, path in paths.items()}
    subjects = sorted(person_values(results[ARMS[0]], 'decoded'))
    datasets = np.array([key[0] for key in subjects])
    panel = pairing['matched_development_support']
    patches = sum(row['complete_patches'] for row in panel.values())
    if (
        len(subjects) != 16
        or set(datasets) != set(panel)
        or sum(v['positions'] for v in panel.values()) != 53476
    ):
        raise ValueError('incomplete development panel')
    arm_rows, subject_rows, dataset_rows, cube = [], [], [], []
    for arm, result in results.items():
        if (
            result['status'] != 'completed'
            or result['inputs'] != 53476
            or result['subjects'] != len(subjects)
            or result.get('development_complete_patches', patches) != patches
        ):
            raise ValueError(f'incomplete scoring for {arm}')
        spectral, nmse, corr = (person_values(result, k) for k in ('decoded', 'nmse', 'correlation'))
        if any(set(table) != set(subjects) for table in (spectral, nmse, corr)):
            raise ValueError('subject pairing changed')
        values = np.array([endpoints(spectral[key], nmse[key]) for key in subjects])
        if not np.isfinite(values).all():
            raise ValueError('nonfinite subject score')
        cube.append(values)
        for (ds, person), row in zip(subjects, values, strict=True):
            subject_rows.append(
                dict(
                    arm=arm,
                    dataset=ds,
                    subject=person,
                    **dict(zip(ENDPOINTS, row, strict=True)),
                    **dict(zip(('correlation_' + c for c in COMPONENTS), corr[ds, person], strict=True)),
                )
            )
        for ds in sorted(set(datasets)):
            dataset_rows.append(
                dict(arm=arm, dataset=ds, **dict(zip(ENDPOINTS, values[datasets == ds].mean(0), strict=True)))
            )
        m = result['metrics']
        macros = endpoints(m['decoded']['macro'], m['nmse']['macro'])
        np.testing.assert_allclose(
            macros, np.mean([values[datasets == ds].mean(0) for ds in set(datasets)], 0)
        )
        arm_rows.append(
            dict(
                arm=arm,
                **dict(zip(ENDPOINTS, macros, strict=True)),
                **dict(
                    zip(
                        ('correlation_' + c for c in COMPONENTS),
                        np.ravel(m['correlation']['macro']),
                        strict=True,
                    )
                ),
                mean_id_gain=result.get('mean_id_gain'),
            )
        )
    point, boot = paired_means(np.stack(cube, axis=1), datasets)
    contrasts = []
    for name, weights in CONTRASTS.items():
        effect = np.einsum('a,am->m', weights, point)
        lo, hi = np.quantile(np.einsum('a,ram->rm', weights, boot), [0.025, 0.975], axis=0)
        for label, value, low, high in zip(ENDPOINTS, effect, lo, hi, strict=True):
            contrasts.append(dict(contrast=name, endpoint=label, estimate=value, lower95=low, upper95=high))
    # The ID readout gain is a ratio of hierarchical MSEs. Keep that denominator inside each bootstrap.
    id_inputs, id_subjects = [], []
    for arm in ARMS[2:]:
        bg, joint = (person_values(results[arm], m) for m in ('background', 'conditional_id'))
        if set(bg) != set(subjects) or set(joint) != set(subjects):
            raise ValueError('integer readout must use the same paired people')
        for ds, person in subjects:
            for i, feature in enumerate(FEATURES):
                id_subjects.append(
                    dict(
                        arm=arm,
                        dataset=ds,
                        subject=person,
                        feature=feature,
                        background_mse=bg[ds, person][i],
                        conditional_id_mse=joint[ds, person][i],
                    )
                )
        id_inputs.append(np.array([[bg[k], joint[k]] for k in subjects]))
    id_point, id_boot = paired_means(np.stack(id_inputs, axis=1), datasets)
    np.testing.assert_allclose(id_point[0, 0], id_point[1, 0], rtol=1e-5, atol=1e-5)
    gains = 1 - id_point[:, 1] / id_point[:, 0]
    estimates = 1 - id_boot[:, :, 1] / id_boot[:, :, 0]
    gains = np.column_stack((gains, gains.mean(-1)))
    estimates = np.concatenate((estimates, estimates.mean(-1, keepdims=True)), axis=-1)
    lo, hi = np.quantile(estimates[:, 1] - estimates[:, 0], [0.025, 0.975], axis=0)
    id_rows = []
    for idx, feature in enumerate([*FEATURES, 'mean']):
        id_rows.append(
            dict(
                feature=feature,
                wave=gains[0, idx],
                dynamic=gains[1, idx],
                delta=gains[1, idx] - gains[0, idx],
                lower95=lo[idx],
                upper95=hi[idx],
            )
        )
    for idx, arm in enumerate(ARMS[2:]):
        np.testing.assert_allclose(gains[idx], [*results[arm]['id_gain'], results[arm]['mean_id_gain']])
    for filename, rows in [
        ('factorial.csv', arm_rows),
        ('contrasts.csv', contrasts),
        ('subject_scores.csv', subject_rows),
        ('dataset_scores.csv', dataset_rows),
        ('hard_id_gain.csv', id_rows),
        ('hard_id_subject_mse.csv', id_subjects),
    ]:
        save_table(filename, rows)
    guardrails = {
        name: reconstruction_checks(results[name + '_wave'], results[name + '_dynamic'])
        for name in ('continuous', 'hard')
    }
    write_json(
        study.OUT / 'summary.json',
        dict(
            status='completed',
            experiment_id=study.PLAN['experiment_id'],
            seed=study.SEED,
            completed_at=now(),
            resolved_config='resolved_config.json',
            planned_arms=4,
            completed_arms=4,
            training_people=60,
            development_people=len(subjects),
            development_people_by_dataset={ds: int((datasets == ds).sum()) for ds in sorted(set(datasets))},
            development_positions=results['hard_wave']['inputs'],
            development_crops=results['hard_wave']['windows'],
            development_records=results['hard_wave']['records'],
            development_complete_patches=patches,
            pairing=pairing,
            inputs={k: str(v.relative_to(parent.ROOT)) for k, v in paths.items()},
            scores=arm_rows,
            primary_contrasts=[r for r in contrasts if r['endpoint'] == 'spectral_mse'],
            guardrails=guardrails,
            hard_id_gain=id_rows,
            id_support={
                a: {
                    k: results[a][k]
                    for k in (
                        'supported_ids',
                        'supported_fraction_vocabulary',
                        'supported_development_tokens',
                        'train_complete_patches',
                        'emitted_development_ids',
                    )
                }
                for a in ARMS[2:]
            },
            interval=study.PLAN['interval'],
            component_order=list(COMPONENTS),
        ),
    )
