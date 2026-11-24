#!/usr/bin/env python3
"""Exact-input collisions and empirical conditional-target floors on public Hb."""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from threadpoolctl import threadpool_limits
import yaml

from src.data.dual_component_tokenizer import (
    base_tokenizer_batch, base_tokenizer_examples, base_tokenizer_training_data,
    base_tokenizer_sample, base_tokenizer_sampling_index,
)
from src.analysis.dual_component_tokenizer import base_training_aggregate, joint_reconstruction_metrics
from src.data.so_supervision_cache import SOSupervisionDataset
from experiments.scripts.run_dual_component_tokenizer import owner_lock


LOCAL = threading.local()


def write(path, payload):
    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)
    path = Path(path)
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=encode) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(values):
    result = np.asarray(values, dtype='<f8').copy()
    if not np.isfinite(result).all():
        raise ValueError('finite observations required')
    result[result == 0] = 0
    return result


def hierarchy_weights(examples, rows):
    """The exact scalar weights of G0's position/crop/record/subject/dataset mean."""
    crops = Counter()
    records, subjects, datasets = defaultdict(set), defaultdict(set), defaultdict(set)
    for index, start, _ in examples:
        row = rows[index]
        crop = (row['dataset'], row['subject'], row['record'], row['parent'], int(start))
        crops[crop] += 1
        records[crop[:3]].add(crop)
        subjects[crop[:2]].add(crop[:3])
        datasets[crop[0]].add(crop[:2])
    result = []
    for index, start, _ in examples:
        row = rows[index]
        crop = (row['dataset'], row['subject'], row['record'], row['parent'], int(start))
        result.append(1 / (len(datasets) * len(datasets[crop[0]]) * len(subjects[crop[:2]])
                           * len(records[crop[:3]]) * crops[crop]))
    return np.asarray(result)


def pointwise_constant(targets, masks, weights):
    """Minimize weighted per-example masked MSE, independently at each time/color."""
    targets, masks, weights = np.asarray(targets), np.asarray(masks, bool), np.asarray(weights, float)
    if targets.ndim != 4 or targets.shape[1] != 2 or targets.shape[2:] != masks.shape[1:]:
        raise ValueError('targets [item,S/O,time,color] and matched masks required')
    if len(weights) != len(targets) or (weights < 0).any():
        raise ValueError('nonnegative matched example weights required')
    count = masks.sum(axis=1)
    importance = np.divide(weights[:, None], count, out=np.zeros_like(count, dtype=float), where=count > 0)
    point_weight = masks[:, None] * importance[:, None, None]
    denominator = point_weight.sum(axis=0)
    total = np.where(masks[:, None], targets, 0.) * point_weight
    mean = np.divide(total.sum(axis=0), denominator, out=np.zeros_like(total[0]), where=denominator > 0)
    return mean, denominator > 0


def normalized_error(targets, masks, prediction, variance):
    error = np.where(masks[:, None], targets - prediction, 0.)
    count = masks.sum(axis=1)
    mse = np.divide(np.square(error).sum(axis=2), count[:, None],
                    out=np.zeros((len(targets), 2, 2)), where=count[:, None] > 0)
    return mse / np.asarray(variance)[None]


def max_shared_spread(targets, masks):
    count = masks.sum(axis=0)
    upper = np.where(masks[:, None], targets, -np.inf).max(axis=0)
    lower = np.where(masks[:, None], targets, np.inf).min(axis=0)
    return float(np.max(np.where(count[None] > 1, upper - lower, 0.), initial=0.))


def fingerprint_batch(payload, rows, scales, cache_limit):
    offset, examples = payload
    if not hasattr(LOCAL, 'cache'):
        LOCAL.cache = OrderedDict()
    batch = base_tokenizer_batch(rows, examples, scales, array_cache=LOCAL.cache, cache_limit=cache_limit)
    records = []
    for local, (index, _, position) in enumerate(examples):
        row = rows[index]
        scale = scales['positions']['hb'][row['dataset']][row['positions'][position]]['scale']
        x = canonical(batch['x'][local])
        mask = batch['mask'][local]
        target = canonical(np.stack([np.where(mask, batch[key][local], 0.) for key in ('s', 'o')]))
        context = json.dumps([row['dataset'], row['positions'][position], float(scale).hex()], separators=(',', ':'))
        records.append((context, hashlib.sha256(x.tobytes()).hexdigest(),
            hashlib.sha256(target.tobytes() + mask.tobytes()).hexdigest(),
            hashlib.sha256(mask.tobytes()).hexdigest(), not bool(np.any(x)),
            mask.reshape(300, 2).sum(axis=0)))
    return offset, records


def scan(examples, rows, scales, workers, cfg):
    batch = cfg['resources']['batch_size']
    payloads = [(offset, examples[offset:offset + batch]) for offset in range(0, len(examples), batch)]
    function = partial(fingerprint_batch, rows=rows, scales=scales,
                       cache_limit=cfg['resources']['mmap_parents_per_worker'])
    if workers == 1:
        for payload in payloads:
            yield function(payload)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(function, payloads)


def example_identity(example, rows):
    index, start, position = (int(value) for value in example)
    row = rows[index]
    return dict(triple=[index, start, position], dataset=row['dataset'], subject=row['subject'],
        record=row['record'], parent=row['parent'], position=row['positions'][position], split=row['split'],
        crop_start_s=start, parent_native_start_sample=row['native_start_sample'], native_rate_hz=row['native_rate_hz'],
        folder=row['folder'], source_unit=row['source_unit'])


def source_trace(example, rows, scales, readers, cache_root):
    item = example_identity(example, rows)
    item['G0_training_scale'] = scales['positions']['hb'][item['dataset']][item['position']]['scale']
    reader = readers[item['split']]
    selected = next(i for i, row in enumerate(reader.windows) if row['key'] == item['parent'])
    metadata_path = Path(reader.windows[selected]['metadata']).resolve()
    if not metadata_path.is_relative_to(cache_root / 'windows'):
        raise ValueError('retained source metadata escapes declared public cache')
    metadata = json.loads(metadata_path.read_text())
    if metadata['split'] != item['split'] or metadata['subject'] != item['subject']:
        raise ValueError('source identity mismatch before source-array access')
    slot = metadata['fNIRS_positions'].index(item['position'])
    task_path = Path(metadata['source_task_paths'][slot]).resolve()
    if not task_path.is_relative_to(cache_root / 'tasks'):
        raise ValueError('teacher task escapes declared public cache')
    task = json.loads(task_path.read_text())
    if (task['identity']['split'] != item['split'] or task['identity']['pair']['id'] != item['position']
            or task['status'] != 'completed' or not task['qualified']):
        raise ValueError('qualified exact source task required before array replay')
    array_path = Path(metadata['array_file']).resolve()
    if array_path != metadata_path.with_suffix('.npz') or not array_path.is_relative_to(cache_root / 'windows'):
        raise ValueError('retained array path must match the declared public teacher window')
    source = reader[selected]['arrays']
    start = item['crop_start_s'] * 10
    take = slice(start, start + 300)
    prepared = {key: np.load(Path(item['folder']) / (key + '.npy'), mmap_mode='r')[take, item['triple'][2]]
                for key in ('x', 's', 'o', 'mask')}
    for key in ('x', 's', 'o'):
        np.testing.assert_array_equal(prepared[key], source['fnirs_' + key][take, slot].astype(np.float32))
    expected_mask = (source['fnirs_observation_valid'][take, slot] & source['fnirs_teacher_valid'][take, slot]
                     & source['source_scoring_mask'][take, None])
    np.testing.assert_array_equal(prepared['mask'], expected_mask)
    with np.load(task_path.with_suffix('.npz'), allow_pickle=False) as task_arrays:
        factor = float(source['fnirs_scale'])
        np.testing.assert_allclose(source['fnirs_s'][:, slot], task_arrays['prediction'][:, 1:] * factor,
                                   rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(source['fnirs_x'][:, slot], task_arrays['target'][:, 1:] * factor,
                                   rtol=1e-12, atol=1e-12)
        task_summary = dict(samples=int(len(task_arrays['target'])),
            observation_columns=['regional_EEG_PC1', 'HbO_at_this_position', 'HbR_at_this_position'],
            full_teacher_EEG_RMS=float(np.sqrt(np.mean(task_arrays['target'][:, 0] ** 2))),
            full_teacher_Hb_RMS=np.sqrt(np.mean(task_arrays['target'][:, 1:] ** 2, axis=0)).tolist())
    return dict(example=item, metadata=str(metadata_path), task=str(task_path), task_sha256=digest(task_path),
        replay='G0 float32 coordinates, masks, and task prediction-to-S exact replay passed',
        source_visibility=task['identity'].get('visibility'), source_subject_adaptation=task['identity'].get('subject_adaptation'),
        teacher_window=task['identity']['window'], teacher_observations=task_summary,
        nearest6_EEG_channels=task['identity']['pair'].get('channel_set_names', {}).get('nearest6'),
        other_Hb_positions_in_this_task=False,
        inference='Teacher optimization used the recorded full source window and EEG projection. This audit does not isolate their causal contribution to any individual S/O difference.')


def run(args, cfg):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    rows = manifest['rows']
    panel_path = ROOT / cfg['panel']['file']
    panel_record = json.loads((panel_path.parent / 'panel.json').read_text())
    if (digest(panel_path) != panel_record['panel_sha256'] or panel_record['cfg_identity'] != manifest['cfg_identity']
            or panel_record['scales_identity'] != manifest['scales_identity']):
        raise ValueError('exact frozen development panel required')
    train = np.asarray(base_tokenizer_examples(rows, 'hb', 'train'), dtype=np.int32)
    with np.load(panel_path, allow_pickle=False) as archive:
        screen = archive[cfg['panel']['role']].copy()
    if len(screen) != 476:
        raise ValueError('exact 476-input development screen required')
    for examples, split in ((train, 'train'), (screen, 'validation')):
        for index, _, _ in examples:
            row = rows[index]
            if row['modality'] != 'hb' or row['split'] != split:
                raise ValueError('permitted Hb split required before arrays')
            if split == 'validation' and row['subject'] == panel_record['confirmation_subjects'][row['dataset']]:
                raise ValueError('confirmation subject excluded before arrays')
    examples = np.concatenate([train, screen])
    roles = np.r_[np.zeros(len(train), np.uint8), np.ones(len(screen), np.uint8)]
    weights = np.r_[hierarchy_weights(train, rows), hierarchy_weights(screen, rows)]
    signature = dict(config=cfg, manifest_sha256=digest(ROOT / cfg['data']['run'] / 'cache/manifest.json'),
        scales_sha256=digest(ROOT / cfg['data']['run'] / 'scales.json'), panel_sha256=digest(panel_path))
    resolved = out / 'resolved_config.json'
    if resolved.exists() and json.loads(resolved.read_text()) != signature:
        raise ValueError('retained diagnostic identity changed')
    write(resolved, signature)
    if (out / 'summary.json').exists():
        return
    np.savez_compressed(out / 'population.npz', examples=examples, roles=roles, hierarchy_weights=weights)
    index_path = out / 'fingerprints.npz'
    if not index_path.exists():
        benchmark = []
        sample = examples[:cfg['resources']['benchmark_examples']]
        list(scan(sample, rows, scales, 1, cfg))
        for workers in cfg['resources']['worker_candidates']:
            began = time.monotonic()
            list(scan(sample, rows, scales, workers, cfg))
            elapsed = time.monotonic() - began
            benchmark.append(dict(workers=workers, examples=len(sample), seconds=elapsed, examples_per_second=len(sample) / elapsed))
        chosen = max(benchmark, key=lambda result: result['examples_per_second'])['workers']
        write(out / 'resources.json', dict(benchmark=benchmark, chosen_workers=chosen,
            numerical_threads=1, reason='maximum measured exact-loader/hash throughput within declared worker candidates'))
        result = []
        began = time.monotonic()
        for offset, records in scan(examples, rows, scales, chosen, cfg):
            result.extend(records)
            if offset % 4096 == 0:
                write(out / 'progress.json', dict(stage='exact_input_hashing', completed=len(result), planned=len(examples), seconds=time.monotonic() - began))
        np.savez_compressed(index_path, contexts=np.array([r[0] for r in result]),
            x_digest=np.array([r[1] for r in result]), target_digest=np.array([r[2] for r in result]),
            mask_digest=np.array([r[3] for r in result]), zero_X=np.array([r[4] for r in result]),
            valid_counts=np.stack([r[5] for r in result]))
    with np.load(index_path, allow_pickle=False) as archive:
        fingerprint = dict(archive)
    if len(fingerprint['contexts']) != len(examples):
        raise ValueError('exact fingerprint population required')
    if not (fingerprint['valid_counts'] > 0).all():
        raise ValueError('unscored colors require an explicit aggregation extension')
    groups = defaultdict(list)
    for n, key in enumerate(zip(fingerprint['contexts'], fingerprint['x_digest'])):
        groups[key].append(n)
    collisions = [members for members in groups.values() if len(members) > 1]
    floors = {role + '_' + level: np.zeros((len(examples), 2, 2))
              for role in ('own_role', 'joint') for level in ('local_X', 'subject', 'subject_record')}
    train_prediction_error = np.zeros((len(examples), 2, 2))
    train_prediction_support = np.zeros((len(examples), 2), dtype=int)
    records = []
    cache = OrderedDict()
    for count, members in enumerate(collisions):
        take = examples[members]
        batch = base_tokenizer_batch(rows, take, scales, array_cache=cache, cache_limit=128)
        x = canonical(batch['x'])
        for current in x[1:]:
            np.testing.assert_array_equal(x[0], current)
        masks = batch['mask'].reshape(len(members), 300, 2)
        targets = np.stack([batch[key].reshape(len(members), 300, 2) for key in ('s', 'o')], axis=1)
        variance = np.stack([batch['v' + key][0, 0, :2] for key in ('s', 'o')])
        spread = max_shared_spread(targets, masks)
        for level in ('local_X', 'subject', 'subject_record'):
            partitions = defaultdict(list)
            for local, n in enumerate(members):
                row = rows[examples[n, 0]]
                key = () if level == 'local_X' else (row['subject'],) if level == 'subject' else (row['subject'], row['record'])
                partitions[key].append(local)
            for selected in partitions.values():
                selected = np.asarray(selected)
                for role in (0, 1):
                    subset = selected[roles[np.asarray(members)[selected]] == role]
                    if not len(subset):
                        continue
                    global_ids = np.asarray(members)[subset]
                    mean, _ = pointwise_constant(targets[subset], masks[subset], weights[global_ids])
                    floors['own_role_' + level][global_ids] = normalized_error(targets[subset], masks[subset], mean, variance)
                global_ids = np.asarray(members)[selected]
                mean, _ = pointwise_constant(targets[selected], masks[selected], weights[global_ids] / 2)
                floors['joint_' + level][global_ids] = normalized_error(targets[selected], masks[selected], mean, variance)
        training = np.flatnonzero(roles[members] == 0)
        validation = np.flatnonzero(roles[members] == 1)
        if len(training) and len(validation):
            mean, supported = pointwise_constant(targets[training], masks[training], weights[np.asarray(members)[training]])
            valid = masks[validation] & supported.all(axis=0)[None]
            global_ids = np.asarray(members)[validation]
            train_prediction_error[global_ids] = normalized_error(targets[validation], valid, mean, variance)
            train_prediction_support[global_ids] = valid.sum(axis=1)
        if spread > 0:
            item = example_identity(examples[members[0]], rows)
            record = dict(group=len(records), dataset=item['dataset'], position=item['position'],
                fixed_scale=scales['positions']['hb'][item['dataset']][item['position']]['scale'],
                x_sha256=str(fingerprint['x_digest'][members[0]]), exact_X_reread_verified=True,
                zero_X=bool(fingerprint['zero_X'][members[0]]), members=members,
                train_inputs=int((roles[members] == 0).sum()), screen_inputs=int((roles[members] == 1).sum()),
                subjects=len({rows[examples[n, 0]]['subject'] for n in members}),
                subject_records=len({(rows[examples[n, 0]]['subject'], rows[examples[n, 0]]['record']) for n in members}),
                distinct_mask_patterns=len({str(fingerprint['mask_digest'][n]) for n in members}),
                max_target_spread_input_coordinate=spread,
                max_X_S_O_closure=float(np.max(np.where(masks, np.abs(x.reshape(len(members), 300, 2) - targets.sum(axis=1)), 0.))),
                joint_oracle_weighted_NMSE_contribution=(floors['joint_local_X'][members] * weights[members, None, None] / 2).sum(axis=0))
            records.append(record)
        if count % 50 == 0:
            write(out / 'progress.json', dict(stage='collision_array_recheck_and_oracle', completed=count + 1,
                planned=len(collisions), conflicting_groups=len(records)))
    np.savez_compressed(out / 'oracle_metrics.npz', examples=examples, roles=roles, **floors,
                        train_prediction_error=train_prediction_error, train_prediction_support=train_prediction_support)
    write(out / 'conflicting_groups.json', records)
    summaries = {}
    for role, label in ((0, 'train'), (1, 'screen')):
        selected = roles == role
        summary_metrics = {name: value[selected] for name, value in floors.items()}
        summary_metrics['nmse'] = floors['own_role_local_X'][selected]
        summaries[label] = base_training_aggregate(summary_metrics, examples[selected], rows)
        summaries[label]['component_order'] = ['S', 'O']
        quantities = summaries[label]['metrics']
        full = np.asarray(quantities['own_role_local_X']['macro'])
        subject = np.asarray(quantities['own_role_subject']['macro'])
        record = np.asarray(quantities['own_role_subject_record']['macro'])
        parts = dict(same_subject_same_record=record, additional_within_subject_across_records=subject - record,
                     additional_across_subjects=full - subject)
        if min(float(np.min(value)) for value in parts.values()) < -1e-10:
            raise AssertionError('nested empirical oracle error increased after adding identity information')
        summaries[label]['ambiguity_NMSE_decomposition'] = {name: np.maximum(value, 0).tolist() for name, value in parts.items()}
    summaries['equal_role_joint'] = {name: (value * weights[:, None, None] / 2).sum(axis=0).tolist()
                                     for name, value in floors.items() if name.startswith('joint_')}
    matching = train_prediction_support[len(train):]
    complete_match = (matching == fingerprint['valid_counts'][len(train):]).all(axis=1)
    summaries['train_fitted_screen_exact_match'] = dict(examples=int((matching > 0).any(axis=1).sum()),
        complete_support_examples=int(complete_match.sum()), planned_screen_examples=len(screen),
        matched_scored_points=int(matching.sum()), planned_scored_points=int(fingerprint['valid_counts'][len(train):].sum()),
        full_panel_error_contribution_on_completely_supported_matches=(train_prediction_error[len(train):]
            * complete_match[:, None, None] * weights[len(train):, None, None]).sum(axis=0).tolist(),
        interpretation='Train-fitted diagnostic on exact input matches. This benchmark error is not a universal statistical lower bound.')
    source_cases = []
    readers = {split: SOSupervisionDataset(ROOT / cfg['source_audit']['so_cache'], split=split) for split in ('train', 'validation')}
    ranked = sorted(records, key=lambda record: -float(np.mean(record['joint_oracle_weighted_NMSE_contribution'])))
    required = [record for record in records if record['dataset'] == 'refed' and record['position'] == 'CH37' and record['zero_X']]
    chosen_groups = list(dict.fromkeys([record['group'] for record in required + ranked[:cfg['source_audit']['max_conflict_groups']]]))
    for group in chosen_groups:
        record = records[group]
        members = record['members']
        selected = [members[0]]
        selected.extend([n for n in members if roles[n] == 1][:1])
        first_row = rows[examples[members[0], 0]]
        selected.extend([n for n in members[1:] if (rows[examples[n, 0]]['subject'], rows[examples[n, 0]]['record'])
                         == (first_row['subject'], first_row['record'])][:1])
        selected.extend(n for n in members if rows[examples[n, 0]]['subject'] != first_row['subject'])
        selected = list(dict.fromkeys(selected))[:cfg['source_audit']['members_per_group']]
        for n in selected:
            source_cases.append(source_trace(examples[n], rows, scales, readers, (ROOT / cfg['source_audit']['so_cache']).resolve()))
        plot_batch = base_tokenizer_batch(rows, examples[members], scales, array_cache=cache, cache_limit=128)
        plot_data = dict(targets=np.stack([plot_batch[key].reshape(len(members), 300, 2) for key in ('s', 'o')], axis=1),
            masks=plot_batch['mask'].reshape(len(members), 300, 2), x=canonical(plot_batch['x']), examples=examples[members])
        collision_plot(out, record, plot_data, roles[members], rows)
    zero_cases = []
    for n in np.flatnonzero(fingerprint['zero_X']):
        batch = base_tokenizer_batch(rows, [examples[n]], scales)
        mask = batch['mask'][0].reshape(300,2)
        value_s = batch['s'][0].reshape(300,2)
        value_o = batch['o'][0].reshape(300,2)
        item = example_identity(examples[n], rows)
        factor = scales['positions']['hb'][item['dataset']][item['position']]['scale']
        counts = mask.sum(0)
        rms = lambda values: np.sqrt(np.where(mask, values**2, 0.).sum(0)/counts)
        zero_cases.append(dict(example=item, role='train' if roles[n] == 0 else 'screen',
            exact_X_zero=True, scored_points_by_color=counts,
            S_RMS_native=rms(value_s)*factor, O_RMS_native=rms(value_o)*factor,
            max_abs_S_plus_O_on_scored_points_native=float(np.max(np.where(mask, abs(value_s+value_o),0.)))*factor,
            exact_training_group_matches=int(sum((fingerprint['contexts'][:len(train)] == fingerprint['contexts'][n])
                & (fingerprint['x_digest'][:len(train)] == fingerprint['x_digest'][n])))))
        source_cases.append(source_trace(examples[n], rows, scales, readers, (ROOT / cfg['source_audit']['so_cache']).resolve()))
    write(out / 'zero_input_cases.json', zero_cases)
    write(out / 'source_cases.json', source_cases)
    failures = audit_training_failures(cfg, rows, scales, readers, out)
    source_cases.extend(failures['source_cases'])
    write(out / 'source_cases.json', source_cases)
    available = Counter(rows[i]['dataset'] for i, _, _ in train)
    unavailable = defaultdict(Counter)
    for row in rows:
        if row['modality'] == 'hb' and row['split'] == 'train':
            for entry in row['unavailable_inputs']['score']:
                for key in ('input_invalid', 'teacher_unavailable'):
                    unavailable[row['dataset']][key] += entry[key]
    baseline_path = ROOT / cfg['source_audit']['screen_baseline_metrics']
    with np.load(baseline_path, allow_pickle=False) as reference:
        np.testing.assert_array_equal(reference['examples'], screen)
        baseline_nmse = reference['nmse'][:, 1:]
    baseline_summary = base_training_aggregate({'nmse': baseline_nmse}, screen, rows)
    conflicted = np.zeros(len(examples), bool)
    for record in records:
        conflicted[record['members']] = True
    baseline_contribution = (baseline_nmse * conflicted[len(train):, None, None] * weights[len(train):, None, None]).sum(0)
    floor = np.asarray(summaries['screen']['metrics']['own_role_local_X']['macro'])
    reference_nmse = np.asarray(baseline_summary['metrics']['nmse']['macro'])
    summaries['screen_baseline_comparison'] = dict(source=str(baseline_path), component_order=['S','O'],
        baseline_macro_NMSE=reference_nmse.tolist(), baseline_error_on_conflicted_inputs=baseline_contribution.tolist(),
        own_role_oracle_floor_fraction_of_baseline_error=np.divide(floor, reference_nmse).tolist())
    summary = dict(status='completed', assumption=cfg['assumption'],
        population=dict(train_available_inputs=len(train), train_available_by_dataset=available,
            train_unavailable_by_dataset=unavailable, screen_inputs=len(screen),
            train_subjects=len({(rows[i]['dataset'], rows[i]['subject']) for i, _, _ in train}),
            confirmation_arrays_read=0, test_arrays_read=0, raw_arrays_read=0),
        exact_matching=dict(context_X_groups=len(groups), duplicate_groups=len(collisions),
            duplicate_inputs=sum(map(len, collisions)), conflicting_groups=len(records),
            conflicting_inputs=sum(len(record['members']) for record in records),
            all_hash_collision_members_reread_and_array_verified=True),
        oracle=summaries, zero_input_cases=zero_cases, source_audited_cases=len(source_cases), training_failures=failures['summary'],
        scope='Empirical exact-input ambiguity for one-position 30s X and the declared fixed metadata. Full multimodal or longer-context encoders have a different observation condition.',
        limitations=['The oracle uses empirical target replicates to diagnose ambiguity and supplies no model training targets.',
            'Only selected collision and training-failure cases were replayed against retained teacher tasks.',
            'Teacher metadata and replay show available inputs and support. Their causal contributions require a separate controlled teacher experiment.'])
    write(out / 'summary.json', summary)


def collision_plot(out, record, data, roles, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scale = record['fixed_scale']
    time_axis = np.arange(300) / 10
    targets, masks = data['targets'], data['masks']
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), layout='constrained')
    chosen = np.linspace(0, len(targets) - 1, min(6, len(targets)), dtype=int)
    for component, label in enumerate(('X', 'S', 'O')):
        values = data['x'].reshape(len(targets), 300, 2) if component == 0 else targets[:, component - 1]
        for color in range(2):
            ax = axes[component, color]
            selected = np.where(masks[:, :, color], values[:, :, color], np.nan) * scale
            low, high = np.nanquantile(selected, [.05, .95], axis=0)
            ax.fill_between(time_axis, low, high, color='#cddbe7', label='all-member 5–95% target interval')
            for index in chosen:
                item = example_identity(data['examples'][index], rows)
                ax.plot(time_axis, selected[index], lw=.9, alpha=.8,
                    label=f"{item['split']} {item['subject']}/{item['record']} +{item['crop_start_s']}s")
            ax.set(xlabel='Time in 30 s input (s)', ylabel=f"{label} {'HbO' if color == 0 else 'HbR'} (cache Hb unit)")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7, loc='upper left')
    fig.suptitle(f"Exact same X, fixed scale | {record['dataset']} {record['position']} | {len(targets)} inputs, {record['subjects']} subjects")
    fig.savefig(out / f"collision_{record['group']:03d}.png", dpi=180)
    plt.close(fig)


def audit_training_failures(cfg, rows, scales, readers, out):
    import torch
    torch.set_num_threads(2)
    from src.tokenizers.contextual_latent_dictionary import load_model
    from experiments.scripts.render_joint_tokenizer_waveforms import curves, waveform_figure
    model_path = ROOT / cfg['source_audit']['failure_model']
    model = load_model(model_path)
    groups = base_tokenizer_sampling_index(rows, 'hb')
    examples = base_tokenizer_sample(np.random.default_rng(cfg['source_audit']['training_monitor_seed']), groups,
                                      cfg['source_audit']['training_monitor_examples'])
    batch = base_tokenizer_batch(rows, examples, scales)
    prediction = model.decode_ids(model.encode(batch['x']))
    scale_values = np.asarray([scales['positions']['hb'][rows[i]['dataset']][rows[i]['positions'][position]]['scale']
                               for i, _, position in examples])
    metrics = joint_reconstruction_metrics(prediction, batch, 2, scale_values)['metrics']
    bias_name = next(name for name in metrics if name.startswith('bias'))
    bias = np.nanmax(abs(metrics[bias_name][:, 2]), axis=1)
    peak = np.nanmax(metrics['peak_timing_error_s'][:, 2], axis=1)
    selected = list(dict.fromkeys([int(np.nanargmax(bias)), int(np.nanargmax(peak))]))
    cases, source_cases = [], []
    arrays = {key: batch[key] for key in ('x', 's', 'o', 'mask')}
    arrays.update({'prediction_' + key: value for key, value in prediction.items()})
    for n in selected:
        item = example_identity(examples[n], rows)
        item.update(modality='hb', position_name=item['position'], scale=float(scale_values[n]), token_seconds=2, interface_eligible=True,
                    selection=['worst_O_bias_or_peak_on_fixed_training_monitor'])
        data, rate, channels = curves(arrays, item, n)
        waveform_figure(data, item, str(model_path), rate, channels, out / f'train_failure_{n:03d}.png', 'Frozen integer decode')
        cases.append(dict(example=item, O_bias=metrics[bias_name][n, 2], O_peak_timing_error_s=metrics['peak_timing_error_s'][n, 2]))
        source_cases.append(source_trace(examples[n], rows, scales, readers, (ROOT / cfg['source_audit']['so_cache']).resolve()))
    return dict(summary=dict(model=str(model_path), model_sha256=digest(model_path), training_monitor_examples=len(examples),
                             cases=cases, bias_field=bias_name), source_cases=source_cases)


def zero_source_record_check(args, cfg):
    """Replay only additional score crops inside an already inspected zero-X source."""
    import torch
    torch.set_num_threads(2)
    from src.tokenizers.contextual_latent_dictionary import load_model

    out = args.output.resolve()
    if (out / 'zero_source_record_check.json').exists():
        return
    main = json.loads((out / 'summary.json').read_text())
    if main['status'] != 'completed':
        raise ValueError('completed bounded primary scan required before source appendix')
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    rows = manifest['rows']
    panel_path = ROOT / cfg['panel']['file']
    panel_record = json.loads((panel_path.parent / 'panel.json').read_text())
    with np.load(panel_path, allow_pickle=False) as archive:
        screen = archive['hb_screen']
    weights = hierarchy_weights(screen, rows)
    baseline_path = ROOT / cfg['source_audit']['screen_baseline_metrics']
    with np.load(baseline_path, allow_pickle=False) as reference:
        np.testing.assert_array_equal(reference['examples'], screen)
        baseline = reference['nmse'].copy()
    baseline_mean = base_training_aggregate({'nmse': baseline}, screen, rows)['metrics']['nmse']['macro']
    sources = json.loads((out / 'source_cases.json').read_text())
    model_path = ROOT / cfg['source_audit']['failure_model']
    model = load_model(model_path)
    findings = []
    for original in main['zero_input_cases']:
        index, _, position = original['example']['triple']
        row = rows[index]
        if (row['split'] != 'validation' or row['subject'] == panel_record['confirmation_subjects'][row['dataset']]
                or not np.any(np.all(screen == np.asarray(original['example']['triple']), axis=1))):
            raise ValueError('source appendix must start from an audited nonconfirmation screen example')
        examples = np.asarray([(index, item['start_s'], position) for item in row['inputs']['score']
                               if position in item['positions']], dtype=np.int32)
        batch = base_tokenizer_batch(rows, examples, scales)
        first = canonical(batch['x'][0])
        identical = np.asarray([np.array_equal(first, canonical(value)) for value in batch['x']])
        selected = np.flatnonzero(identical)
        masks = batch['mask'][selected].reshape(len(selected), 300, 2)
        targets = np.stack([batch[key][selected].reshape(len(selected), 300, 2) for key in ('s','o')], axis=1)
        variance = np.stack([batch['v'+key][0,0,:2] for key in ('s','o')])
        group_weights = hierarchy_weights(examples[selected], rows)
        mean, _ = pointwise_constant(targets, masks, group_weights)
        errors = normalized_error(targets, masks, mean, variance)
        optimum = (errors * group_weights[:,None,None]).sum(0)
        raw_mse = errors * variance[None]
        factor = scales['positions']['hb'][row['dataset']][row['positions'][position]]['scale']
        native_rmse = np.sqrt((raw_mse * group_weights[:,None,None]).sum(0)) * factor
        ids = model.encode(batch['x'][selected])
        for branch in ('S','O'):
            for current in ids[branch][1:]:
                np.testing.assert_array_equal(current, ids[branch][0])
        np.savez_compressed(out / 'zero_source_record_ids.npz', examples=examples[selected], **ids)
        source = next(item for item in sources if item['example']['triple'] == original['example']['triple'])
        task_path = Path(source['task'])
        if not task_path.resolve().is_relative_to((ROOT / cfg['source_audit']['so_cache'] / 'tasks').resolve()):
            raise ValueError('source appendix task path outside declared teacher root')
        task = json.loads(task_path.read_text())
        with np.load(task_path.with_suffix('.npz'), allow_pickle=False) as retained:
            eeg = [retained['target'][start*10:(start+30)*10,0] for _,start,_ in examples]
            eeg_rms = [float(np.sqrt(np.mean(value**2))) for value in eeg]
            eeg_different = [not np.array_equal(eeg[0], value) for value in eeg]
        screen_index = int(np.flatnonzero(np.all(screen == np.asarray(original['example']['triple']),axis=1))[0])
        contribution = baseline[screen_index] * weights[screen_index]
        record = dict(dataset=row['dataset'],subject=row['subject'],record=row['record'],position=row['positions'][position],
            source_task=str(task_path),source_task_sha256=digest(task_path),
            owner_admitted_source_score_examples=examples.tolist(),exact_identical_examples=examples[selected].tolist(),
            windows=len(selected), independent_records=1, outside_primary_screen_examples=[item.tolist() for item in examples[selected]
                if not np.any(np.all(screen == item,axis=1))],
            scored_points_by_color=masks.sum(1),observations_all_zero=not bool(np.any(batch['x'][selected])),
            max_shared_S_O_spread_input_coordinate=max_shared_spread(targets,masks),
            optimum_constant_waveform_NMSE_S_O_by_color=optimum,
            optimum_constant_waveform_RMSE_native_S_O_by_color=native_rmse,
            source_target_variance_S_O_by_color=variance,
            max_abs_S_plus_O_native=float(np.max(np.where(masks,abs(targets.sum(1)),0.))) * factor,
            actual_frozen_model_ID_sequences_identical=True, teacher_regional_EEG_RMS_by_crop=eeg_rms,
            teacher_regional_EEG_differs_from_first_crop=eeg_different,
            teacher_qualified=task['qualified'], teacher_Hb_not_better_than_zero=task.get('Hb_not_better_than_zero'),
            teacher_Hb_zero_baseline=task.get('zero_baseline'),
            singleton_primary_screen_baseline_NMSE_X_S_O_by_color=baseline[screen_index],
            singleton_primary_screen_G0_weight=float(weights[screen_index]),
            singleton_primary_screen_baseline_NMSE_contribution_X_S_O_by_color=contribution,
            singleton_primary_screen_fraction_of_baseline_error_X_S_O_by_color=np.divide(contribution,baseline_mean),
            statement='This appendix isolates repeated zero-X crops of one audited source record. Its floor applies to a deterministic single-position student with the stated fixed metadata. It is separate from the original 216711-input primary population and supplies no whole-population impossibility result.')
        findings.append(record)
        plot_data = dict(targets=targets,masks=masks,x=canonical(batch['x'][selected]),examples=examples[selected])
        plot_record = dict(group=1000,dataset=row['dataset'],position=row['positions'][position],fixed_scale=factor,subjects=1)
        collision_plot(out,plot_record,plot_data,np.ones(len(selected),dtype=int),rows)
    write(out / 'zero_source_record_check.json',dict(status='completed_source_appendix',primary_scan_unchanged=True,
        primary_summary='summary.json',source_only_local_appendix=True,confirmation_arrays_read=0,test_arrays_read=0,
        findings=findings,limitations=['Three crops from one conditional teacher trajectory constitute one source record.',
            'The main train and screen populations retain zero exact collisions. Only this inspected source record is extended.',
            'Multimodal or absolute-clock-conditioned encoders have different observable inputs.']))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default='experiments/configs/physiology_semantic_tokenizer/dual_component_supervision_ambiguity_v1.yaml')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--zero-source-check', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    if (cfg['schema'] != 'dual_component_supervision_ambiguity_v1' or cfg['panel']['role'] != 'hb_screen'
            or cfg['population']['confirmation_reads'] != 'forbidden' or cfg['data']['raw_reads']):
        raise ValueError('public Hb train and development-screen-only diagnostic required')
    with owner_lock(args.output), threadpool_limits(limits=1):
        if args.zero_source_check:
            zero_source_record_check(args,cfg)
        else:
            run(args, cfg)


if __name__ == '__main__':
    main()
