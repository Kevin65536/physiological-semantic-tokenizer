"""Train-subject audits of actual AS integers, coarse maps and sample support."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics, joint_reconstruction_summary
from src.analysis.tokenizer_pattern_audit import SparsePatternAudit, attributes, feature_means
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_training_data
from src.tokenizers.coarsened_spatial_dictionary import CoarsenedSpatialDictionary, supported_remap
from src.tokenizers.spatial_analysis_synthesis import load_encoder
from src.tokenizers.spatial_local_gain import load_decoder
from src.tokenizers.spatial_waveform_dictionary import spectral_features
from audit_joint_tokenizer_patterns import hierarchy, write_json

FIELDS = {'power_4_8': (0, 1), 'power_8_13': (1, 2), 'rms_envelope': (2, 10)}
SAMPLE_SUPPORT_STATE = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def check_budget(cfg):
    if time.process_time() > cfg['resources']['CPU_budget_seconds'] - 30:
        raise RuntimeError('bounded CPU budget reached')


def prepare(cfg, output, rows, scales):
    """Read only the already registered pattern-training population."""
    source = ROOT / cfg['source']['actual_ID_run']
    with np.load(source/'train_IDs.npz', allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    examples = data['examples']
    if any(rows[i]['split'] != 'train' or rows[i]['modality'] != 'eeg' for i, _, _ in examples):
        raise ValueError('train-only actual-ID panel required before signal reads')
    sites = sorted({(r['dataset'], name) for r in rows if r['split'] == 'train' and r['modality'] == 'eeg' for name in r['positions']})
    site_index = {key: j for j, key in enumerate(sites)}
    data['background'] = np.empty((len(examples), 15), np.int32)
    data['valid'] = np.empty((len(examples), 15), bool)
    data['S_attributes'] = np.zeros((len(examples), 15, 10))
    data['O_attributes'] = np.zeros((len(examples), 15, 10))
    cache = OrderedDict()
    for start in range(0, len(examples), 64):
        check_budget(cfg)
        take = examples[start:start + 64]
        batch = base_tokenizer_batch(rows, take, scales, array_cache=cache, cache_limit=64)
        valid = batch['mask'].all(-1)
        loc = np.array([site_index[rows[i]['dataset'], rows[i]['positions'][p]] for i, _, p in take])
        data['background'][start:start + len(take)] = 4 * loc[:, None] + np.searchsorted([.5, 1., 2.], np.sqrt(np.mean(batch['x']**2, axis=-1)), side='right')
        data['valid'][start:start + len(take)] = valid
        for branch in ('S', 'O'):
            raw, slices = attributes(batch[branch.lower()][valid], 'eeg')
            data[branch + '_attributes'][start:start + len(take)][valid] = np.concatenate([raw[:, slice(*slices[k])] for k in FIELDS], axis=1)
        write_json(output/'progress.json', dict(stage='training_attributes', completed=start + len(take), planned=len(examples), CPU_seconds=time.process_time()))
    subjects = sorted({(rows[i]['dataset'], rows[i]['subject']) for i, _, _ in examples})
    unit_index = {key: j for j, key in enumerate(subjects)}
    data['subject'] = np.array([unit_index[rows[i]['dataset'], rows[i]['subject']] for i, _, _ in examples])
    folds = {}
    for dataset in sorted({d for d, _ in subjects}):
        names = sorted([s for d, s in subjects if d == dataset], key=lambda s: rank([cfg['training']['seed'], dataset, s]))
        folds.update({(dataset, name): j % cfg['training']['folds'] for j, name in enumerate(names)})
    data['fold'] = np.array([folds[rows[i]['dataset'], rows[i]['subject']] for i, _, _ in examples])
    np.savez_compressed(output/'training_attributes.npz', **data)
    write_json(output/'training_population.json', dict(inputs=len(examples), complete_patches=int(data['valid'].sum()), subjects=subjects,
        folds=[dict(dataset=d, subject=s, fold=folds[d, s]) for d, s in subjects], background_sites=sites,
        exposure=cfg['training']['original_model_exposure'], validation_arrays_read=0))
    return data, len(sites)


def fit_maps(data, selected, capacity, fraction, carriers):
    maps, lifts = {}, {}
    for branch in ('S', 'O'):
        for band in range(5):
            mask = data['valid'] & selected[:, None] & (data['fixed_band'][:, None] == band)
            fine = data[branch][mask]
            count = np.bincount(fine, minlength=8192)
            targets = data[branch + '_attributes'][mask]
            y = np.column_stack((targets[:, :2], targets[:, 2:].mean(1)))
            totals = np.zeros((8192, 3))
            np.add.at(totals, fine, y)
            means = totals / count[:, None].clip(1)
            support = np.zeros(8192, int)
            for unit in np.unique(data['subject'][selected]):
                unit_mask = mask & (data['subject'][:, None] == unit)
                support[np.unique(data[branch][unit_mask])] += 1
            maps[branch, band], lifts[branch, band] = supported_remap(carriers[branch, band], count, support, means, capacity, fraction)
    return maps, lifts


def local_coarse_ids(data, mapping, capacity):
    result = {}
    for branch in ('S', 'O'):
        value = np.empty(data[branch].shape, np.int32)
        for band in range(5):
            selected = data['fixed_band'] == band
            value[selected] = mapping[branch, band][data[branch][selected]] + band * capacity
        result[branch] = value
    return result


def pattern_fold(data, selected, mapping, capacity, backgrounds):
    coarse = local_coarse_ids(data, mapping, capacity)
    result = {}
    for branch in ('S', 'O'):
        audit = SparsePatternAudit(5 * capacity, 4 * (backgrounds + 1), [], 10, FIELDS)
        fitted = data['valid'] & selected[:, None]
        held = data['valid'] & ~selected[:, None]
        audit.fit(coarse[branch][fitted], data['background'][fitted], data[branch + '_attributes'][fitted])
        bg, _, joint = audit.read(coarse[branch][held], data['background'][held])
        target = data[branch + '_attributes'][held]
        errors = np.zeros((*held.shape, 2, 3))
        errors[held] = np.stack([feature_means((prediction - target)**2, FIELDS) for prediction in (bg, joint)], axis=1)
        selected_errors = errors[~selected].sum(1) / data['valid'][~selected].sum(1).clip(1)[:, None, None]
        selected_errors[~data['valid'][~selected].any(1)] = np.nan
        subjects = np.zeros(5 * capacity, int)
        for unit in np.unique(data['subject'][selected]):
            mask = fitted & (data['subject'][:, None] == unit)
            subjects[np.unique(coarse[branch][mask])] += 1
        strong = (audit.local.count >= 8) & (subjects >= 2)
        mass = float(strong[coarse[branch][held]].mean())
        result[branch] = dict(errors=selected_errors, supported=int(strong[coarse[branch][held]].sum()), total=int(held.sum()), support_mass=mass)
    return result


def native_blocks(cfg):
    lookup = {}
    directory = ROOT/cfg['source']['actual_ID_run']/'native_train_ids'
    for path in sorted(directory.glob('*.npz')):
        with np.load(path, allow_pickle=False) as stored:
            for j, key in enumerate(stored['crops']):
                lookup[tuple(int(v) for v in key)] = (path, j)
    return lookup


def waveform_cases(cfg, data, rows, scales, decoder, output):
    """One predetermined crop per training subject, all decode inputs from saved IDs."""
    lookup = native_blocks(cfg)
    chosen = {}
    for row, start, position in data['examples']:
        unit = rows[row]['dataset'], rows[row]['subject']
        key = int(row), int(start)
        if unit not in chosen or rank([cfg['training']['seed'], *key]) < rank([cfg['training']['seed'], *chosen[unit]]):
            chosen[unit] = key
    cases, cache = [], OrderedDict()
    unit_folds = {(rows[i]['dataset'], rows[i]['subject']): int(fold)
                  for (i, _, _), fold in zip(data['examples'], data['fold'])}
    for unit, key in sorted(chosen.items()):
        i, start = key
        row = rows[i]
        path, ordinal = lookup[key]
        with np.load(path, allow_pickle=False) as stored:
            ids = {b: stored[b][ordinal:ordinal + 1].copy() for b in ('S', 'O')}
        positions = next(item['positions'] for item in row['inputs']['score'] if item['start_s'] == start)
        positions = sorted(positions, key=lambda p: rank([cfg['training']['seed'], i, start, p]))[:4]
        examples = np.array([(i, start, p) for p in sorted(positions)])
        batch = base_tokenizer_batch(rows, examples, scales, array_cache=cache, cache_limit=16)
        names = decoder.metadata['positions'][unit[0]]
        selected = np.array([next(j for j, n in enumerate(names) if n.upper() == row['positions'][p].upper()) for p in sorted(positions)])
        factor = np.array([scales['positions']['eeg'][unit[0]][row['positions'][p]]['scale'] for p in sorted(positions)])
        prediction = decoder.decode_ids(ids, unit[0])
        audited = joint_reconstruction_metrics({k: v[0, selected] for k, v in prediction.items()}, batch, 1, factor)
        fold = unit_folds[unit]
        cases.append(dict(dataset=unit[0], subject=unit[1], fold=fold, examples=examples, ids=ids,
            batch=batch, selected=selected, scales=factor, baseline=audited))
    write_json(output/'waveform_probe.json', dict(cases=[{k: v for k, v in c.items() if k in ('dataset', 'subject', 'fold', 'examples')} for c in cases],
        scope='coarsening-layer held-subject proxy; original encoder and decoder already exposed to these train subjects', read_split='train'))
    return cases


def combined_waveform(cases, values, rows):
    metrics = {key: np.concatenate([value['metrics'][key] for value in values]) for key in values[0]['metrics']}
    support = {key: np.concatenate([value['support'][key] for value in values]) for key in values[0]['support']}
    examples = np.concatenate([case['examples'] for case in cases])
    return joint_reconstruction_summary(metrics, support, examples.tolist(), rows)


def crossfit(cfg, data, backgrounds, cases, rows, encoder, decoder, carriers, output):
    baseline = combined_waveform(cases, [case['baseline'] for case in cases], rows)
    write_json(output/'matched_waveform_baseline.json', baseline)
    records = []
    for positive in cfg['model']['coarse_positive_atoms']:
        capacity = positive + 1
        for fraction in cfg['model']['semantic_distance_fractions']:
            check_budget(cfg)
            key = f'k{positive}_semantic{fraction:g}'
            errors = {b: np.full((len(data['examples']), 2, 3), np.nan) for b in ('S', 'O')}
            counts = {b: [0, 0] for b in ('S', 'O')}
            waveform = [None] * len(cases)
            for fold in range(cfg['training']['folds']):
                selected = data['fold'] != fold
                maps, lifts = fit_maps(data, selected, capacity, fraction, carriers)
                pattern = pattern_fold(data, selected, maps, capacity, backgrounds)
                for branch in ('S', 'O'):
                    errors[branch][~selected] = pattern[branch]['errors']
                    counts[branch][0] += pattern[branch]['supported']
                    counts[branch][1] += pattern[branch]['total']
                metadata = dict(codebook_size=capacity, positions=decoder.metadata['positions'])
                model = CoarsenedSpatialDictionary(decoder, encoder, metadata, maps, lifts)
                for n, case in enumerate(cases):
                    if case['fold'] != fold:
                        continue
                    coarse = model.remap(case['ids'], case['dataset'])
                    prediction = model.decode_ids(coarse, case['dataset'])
                    waveform[n] = joint_reconstruction_metrics({k: v[0, case['selected']] for k, v in prediction.items()}, case['batch'], 1, case['scales'])
                write_json(output/'progress.json', dict(stage='crossfit', candidate=key, completed_fold=fold + 1, CPU_seconds=time.process_time(), validation_arrays_read=0))
            stats = {}
            for branch in ('S', 'O'):
                score = hierarchy(errors[branch], data['examples'], rows)
                mse = np.array(score['macro'])
                gain = 1 - mse[1]/mse[0]
                fraction_subjects = np.mean([np.array(s['value'])[1] < np.array(s['value'])[0] for s in score['by_subject']], axis=0)
                stats[branch] = dict(gain=dict(zip(FIELDS, gain)), positive_subject_fraction=dict(zip(FIELDS, fraction_subjects)),
                    support_mass=counts[branch][0]/counts[branch][1], hierarchy=score)
            wave = combined_waveform(cases, waveform, rows)
            selection = cfg['selection']
            pattern_pass = all(v['support_mass'] >= selection['minimum_training_support_mass']
                and min(v['gain'][k] for k in ('power_4_8', 'power_8_13')) > 0
                and max(v['gain'][k] for k in ('power_4_8', 'power_8_13')) >= selection['primary_one_gain_min']
                and min(v['positive_subject_fraction'][k] for k in ('power_4_8', 'power_8_13')) >= .5 for v in stats.values())
            waveform_pass = bool(np.all(np.array(wave['metrics']['nmse']['macro']) <= selection['held_waveform_component_NMSE_max_ratio'] * np.array(baseline['metrics']['nmse']['macro'])))
            waveform_pass &= all(np.all(np.array(value) <= selection['held_waveform_dataset_component_NMSE_max_ratio'] * np.array(baseline['metrics']['nmse']['by_dataset'][dataset])) for dataset, value in wave['metrics']['nmse']['by_dataset'].items())
            record = dict(candidate=key, positive_atoms=positive, vocabulary=capacity, semantic_fraction=fraction,
                pattern=stats, waveform=wave, pattern_pass=pattern_pass, waveform_pass=waveform_pass,
                eligible=bool(pattern_pass and waveform_pass), primary_fitting_split='train', screen_read=False)
            records.append(record)
            np.savez_compressed(output/(key+'_held_attribute_errors.npz'), examples=data['examples'], **errors)
            write_json(output/'training_crossfit.json', dict(status='in_progress', source=baseline, candidates=records, validation_arrays_read=0))
    write_json(output/'training_crossfit.json', dict(status='completed', source=baseline, candidates=records, validation_arrays_read=0))
    eligible = [r for r in records if r['eligible']]
    selected = min(eligible, key=lambda r: (r['waveform']['selection_score'], r['positive_atoms'], r['semantic_fraction'])) if eligible else None
    return records, selected


def run(args, cfg):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'summary.json').exists():
        raise FileExistsError('retained experiment already finished')
    cpu = cfg['resources']['CPU_budget_seconds']
    resource.setrlimit(resource.RLIMIT_CPU, (cpu - 10, cpu))
    if cfg['data']['test_access'] != 'forbidden' or cfg['data']['protected_data'] != 'forbidden':
        raise ValueError('public train/validation-only boundary required')
    write_json(output/'resolved_config.json', cfg)
    base = yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(base, ROOT)
    encoder = load_encoder(ROOT/cfg['source']['encoder'])
    decoder = load_decoder(ROOT/cfg['source']['decoder'])
    panel = ROOT/cfg['source']['panel_run']
    retained = json.loads((panel/'panel.json').read_text())
    if retained['cfg_identity'] != manifest['cfg_identity'] or retained['scales_identity'] != manifest['scales_identity']:
        raise ValueError('G0 identity changed')
    data, backgrounds = prepare(cfg, output, manifest['rows'], scales)
    with np.load(panel/'panels.npz', allow_pickle=False) as stored:
        np.testing.assert_array_equal(data['examples'], stored['eeg_pattern_train'])
    carriers = {(b, band): spectral_features(decoder.source.atoms[b, band], band) for b in ('S', 'O') for band in range(5)}
    cases = waveform_cases(cfg, data, manifest['rows'], scales, decoder, output)
    records, selected = crossfit(cfg, data, backgrounds, cases, manifest['rows'], encoder, decoder, carriers, output)
    for record in records:
        maps, lifts = fit_maps(data, np.ones(len(data['examples']), bool), record['vocabulary'], record['semantic_fraction'], carriers)
        metadata = dict(schema='coarsened_AS_spatial_dictionary_v1', codebook_size=record['vocabulary'],
            positions=decoder.metadata['positions'], candidate=record['candidate'], fitting_split='train',
            input_identity=manifest['cfg_identity'], scales_identity=manifest['scales_identity'],
            source_encoder=os.path.relpath(ROOT/cfg['source']['encoder'], output),
            source_decoder=os.path.relpath(ROOT/cfg['source']['decoder'], output),
            source_encoder_sha256=digest(ROOT/cfg['source']['encoder']), source_decoder_sha256=digest(ROOT/cfg['source']['decoder']),
            integers_per_branch_native_position_per_30seconds=15, seed=cfg['training']['seed'])
        model = CoarsenedSpatialDictionary(decoder, encoder, metadata, maps, lifts)
        model.save(output/(record['candidate']+'.npz'))
    result = dict(status='train_crossfit_rejected' if selected is None else 'train_crossfit_admitted',
        selected=None if selected is None else selected['candidate'], candidates=len(records),
        CPU_seconds=time.process_time(), screen_opened=False, confirmation_opened=False, test_opened=False,
        source_actual_IDs=cfg['source']['actual_ID_run'], source_exposure=cfg['training']['original_model_exposure'])
    write_json(output/'summary.json', result)
    print(json.dumps(result), flush=True)


def review(args, cfg):
    """Compare fine-ID crossfits and prove retained coarse-only integer replay."""
    import csv
    import gc
    from src.tokenizers.coarsened_spatial_dictionary import load_model

    output = args.output.resolve()
    result = json.loads((output/'summary.json').read_text())
    remaining = int(cfg['resources']['CPU_budget_seconds'] - result['CPU_seconds'])
    if remaining < 60:
        raise RuntimeError('insufficient review CPU budget')
    resource.setrlimit(resource.RLIMIT_CPU, (remaining - 10, remaining))
    if result['status'] != 'train_crossfit_rejected':
        raise ValueError('a rejected training experiment is required for terminal review')
    base = yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text())
    manifest, _, _, _ = base_tokenizer_training_data(base, ROOT)
    rows = manifest['rows']
    with np.load(output/'training_attributes.npz', allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    population = json.loads((output/'training_population.json').read_text())
    identity = {(b, band): np.arange(8192) for b in ('S', 'O') for band in range(5)}
    errors = {b: np.empty((len(data['examples']), 2, 3)) for b in ('S', 'O')}
    for fold in range(cfg['training']['folds']):
        selected = data['fold'] != fold
        current = pattern_fold(data, selected, identity, 8192, len(population['background_sites']))
        for branch in ('S', 'O'):
            errors[branch][~selected] = current[branch]['errors']
    fine = {}
    for branch in ('S', 'O'):
        stats = hierarchy(errors[branch], data['examples'], rows)
        mse = np.asarray(stats['macro'])
        fine[branch] = dict(gain=dict(zip(FIELDS, 1 - mse[1]/mse[0])), hierarchy=stats)
    write_json(output/'fine_ID_training_crossfit.json', fine)
    table = []
    replay = []
    blocks = native_blocks(cfg)
    (first_row, first_start), (block, block_index) = next(iter(sorted(blocks.items())))
    with np.load(block, allow_pickle=False) as saved:
        fine_ids = {b: saved[b][block_index:block_index + 1].copy() for b in ('S', 'O')}
    dataset = rows[first_row]['dataset']
    candidates = json.loads((output/'training_crossfit.json').read_text())['candidates']
    for candidate in candidates:
        model_path = output/(candidate['candidate']+'.npz')
        model = load_model(model_path)
        coarse_ids = model.remap(fine_ids, dataset)
        ids_path = output/(model_path.stem + '_replay_IDs.npz')
        np.savez_compressed(ids_path, **coarse_ids)
        expected = model.decode_ids(coarse_ids, dataset)
        del model
        gc.collect()
        model = load_model(model_path)
        with np.load(ids_path, allow_pickle=False) as saved:
            repeated = model.decode_ids({b: saved[b].copy() for b in ('S', 'O')}, dataset)
        for branch in ('S', 'O', 'sum'):
            np.testing.assert_array_equal(repeated[branch], expected[branch])
        replay.append(dict(candidate=model_path.stem, dataset=dataset, row=first_row, start_s=first_start,
            coarse_integer_count=sum(value.size for value in coarse_ids.values()),
            max_abs_output_difference=0., original_fine_IDs_supplied_to_replay=False))
        local = local_coarse_ids(data, model.mapping, model.codebook_size)
        for branch in ('S', 'O'):
            namespace_count = np.bincount(local[branch][data['valid']], minlength=5*model.codebook_size)
            subject_count = np.zeros_like(namespace_count)
            for unit in np.unique(data['subject']):
                valid = data['valid'] & (data['subject'][:, None] == unit)
                subject_count[np.unique(local[branch][valid])] += 1
            for band in range(5):
                count = namespace_count.reshape(5, -1)[band]
                subjects = subject_count.reshape(5, -1)[band]
                original = data[branch][data['valid'] & (data['fixed_band'][:, None] == band)]
                fine_count = np.bincount(original, minlength=8192)
                atoms = spectral_features(model.source.source.atoms[branch, band], band)
                replaced = atoms[model.representatives[branch, band][model.mapping[branch, band]]]
                error = float(np.sum(fine_count[:, None]*(atoms-replaced)**2)/max(np.sum(fine_count[:, None]*atoms**2), 1e-12))
                for code in range(model.codebook_size):
                    table.append(dict(candidate=model_path.stem, branch=branch, fixed_band=band, coarse_ID=code,
                        representative_fine_ID=int(model.representatives[branch, band][code]),
                        complete_train_patches=int(count[code]), training_subjects=int(subjects[code]),
                        per_band_atom_error_over_original_energy=error))
        del model
        gc.collect()
    with (output/'all_coarse_code_support.csv').open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    write_json(output/'integer_replay_review.json', dict(status='passed', scope='one complete real training montage per candidate; no new encoding', records=replay,
        decoder_inputs='coarse S/O integers, fixed native dataset layout and frozen model only'))
    write_json(output/'completion.json', dict(status='completed_training_rejection', candidates=result['candidates'],
        training_subjects=len(population['subjects']), source_training_inputs=population['inputs'],
        total_CPU_seconds=result['CPU_seconds'] + time.process_time(), budget_CPU_seconds=cfg['resources']['CPU_budget_seconds'],
        screen_read=False, confirmation_read=False, test_read=False,
        conclusion='supported coarse representatives improve support but fail the joint held-training waveform and local-information conditions'))


def additional_training_examples(original, rows, crop_keys, maximum, seed):
    """Extend subject counts only through admitted positions of already encoded crops."""
    original = np.asarray(original)
    seen = {tuple(int(x) for x in row) for row in original}
    if len(seen) != len(original):
        raise ValueError('original score population contains duplicate triples')
    units = {(rows[i]['dataset'], rows[i]['subject']) for i, _, _ in original}
    counts = {unit: 0 for unit in units}
    for i, _, _ in original:
        counts[rows[i]['dataset'], rows[i]['subject']] += 1
    candidates = {unit: [] for unit in units}
    for i, start in sorted(crop_keys):
        row = rows[i]
        if row['split'] != 'train' or row['modality'] != 'eeg':
            raise ValueError('native cache contains a non-training EEG crop')
        unit = row['dataset'], row['subject']
        if unit not in candidates:
            raise ValueError('native cache adds a subject outside the original population')
        support = next(item['positions'] for item in row['inputs']['score'] if item['start_s'] == start)
        for p in support:
            triple = int(i), int(start), int(p)
            if triple not in seen:
                candidates[unit].append(triple)
    chosen = []
    for unit in sorted(units):
        needed = maximum - counts[unit]
        if needed < 0:
            raise ValueError('expanded arm must include every original score input')
        unique = set(candidates[unit])
        selected = sorted(unique, key=lambda t: rank([seed, *unit, *t]))[:needed]
        chosen.extend(selected)
    return np.asarray(sorted(chosen), np.int32).reshape(-1, 3)


def sample_support_readout(bank, original_count, backgrounds, expanded, fold, branch):
    """Fit unchanged categorical group means and score the same held originals."""
    eligible = np.ones(len(bank['examples']), bool) if expanded else np.arange(len(bank['examples'])) < original_count
    fitting = eligible & (bank['fold'] != fold)
    score_rows = np.flatnonzero((np.arange(len(bank['examples'])) < original_count) & (bank['fold'] == fold))
    codes = bank[branch].astype(np.int64) + 8192*bank['fixed_band'][:, None]
    audit = SparsePatternAudit(5*8192, 4*(backgrounds + 1), [], 10, FIELDS)
    fit = bank['valid'] & fitting[:, None]
    audit.fit(codes[fit], bank['background'][fit], bank[branch + '_attributes'][fit])
    subjects = np.zeros(5*8192, int)
    for unit in np.unique(bank['subject'][fitting]):
        mask = fit & (bank['subject'][:, None] == unit)
        subjects[np.unique(codes[mask])] += 1
    strong = (audit.local.count >= 8) & (subjects >= 2)
    errors = np.empty((len(score_rows), 2, 3))
    flags = np.zeros((len(score_rows), 15, 3), bool)
    joint_fit_count = np.zeros((len(score_rows), 15), np.int32)
    background_fit_count = np.zeros((len(score_rows), 15), np.int32)
    for begin in range(0, len(score_rows), 64):
        selected = score_rows[begin:begin + 64]
        valid = bank['valid'][selected]
        ids = codes[selected][valid]
        bg = bank['background'][selected][valid]
        background, _, conditional = audit.read(ids, bg)
        target = bank[branch + '_attributes'][selected][valid]
        losses = np.zeros((*valid.shape, 2, 3))
        losses[valid] = np.stack([feature_means((p - target)**2, FIELDS) for p in (background, conditional)], axis=1)
        mean = losses.sum(1)/valid.sum(1).clip(1)[:, None, None]
        mean[~valid.any(1)] = np.nan
        errors[begin:begin + len(selected)] = mean
        keys = bg*audit.k + ids
        support = np.column_stack((audit.local.count[ids] > 0, strong[ids],
                                   np.fromiter((int(k) in audit.joint.count for k in keys), bool, len(keys))))
        current = flags[begin:begin + len(selected)]
        current[valid] = support
        joint_fit_count[begin:begin + len(selected)][valid] = np.fromiter((audit.joint.count.get(int(k), 0) for k in keys), np.int32, len(keys))
        background_fit_count[begin:begin + len(selected)][valid] = audit.background.count[bg]
    record = dict(fold=fold, fitting_subjects=sorted(np.unique(bank['subject'][fitting]).tolist()),
        held_subjects=sorted(np.unique(bank['subject'][score_rows]).tolist()),
        fitting_inputs=int(fitting.sum()), fitting_complete_patches=int(fit.sum()),
        scored_inputs=len(score_rows), scored_complete_patches=int(bank['valid'][score_rows].sum()))
    if set(record['fitting_subjects']) & set(record['held_subjects']):
        raise AssertionError('held subject entered categorical fitting')
    return score_rows, errors, flags, joint_fit_count, background_fit_count, record


def sample_source_identity(cfg, manifest):
    retained = json.loads((ROOT/cfg['source']['actual_ID_run']/'support_summary.json').read_text())['model_metadata']
    if (retained['input_identity'] != manifest['cfg_identity']
            or retained['scales_identity'] != manifest['scales_identity']):
        raise ValueError('actual integer cache and G0 training coordinates differ')
    return dict(cfg_identity=manifest['cfg_identity'], scales_identity=manifest['scales_identity'],
                actual_ID_cache='same retained AS namespace and G0 input coordinate')


def sample_support_bank(cfg, output, rows, scales):
    original_path = ROOT/cfg['source']['original_bank']
    with np.load(original_path, allow_pickle=False) as saved:
        original = {key: saved[key].copy() for key in saved.files}
    population = json.loads((ROOT/cfg['source']['original_population']).read_text())
    retained = output/'sample_support_bank.npz'
    if retained.exists() and (output/'sample_population.json').exists():
        with np.load(retained, allow_pickle=False) as saved:
            bank = {key: saved[key].copy() for key in saved.files}
        for key, value in original.items():
            np.testing.assert_array_equal(bank[key][:len(value)], value)
        return bank, len(original['examples']), len(population['background_sites'])
    units = [tuple(v) for v in population['subjects']]
    unit_index = {key: j for j, key in enumerate(units)}
    folds = {(item['dataset'], item['subject']): item['fold'] for item in population['folds']}
    lookup = native_blocks(cfg)
    original_crops = {(int(i), int(t)) for i, t, _ in original['examples']}
    if set(lookup) != original_crops:
        raise ValueError('native blocks must exactly match original pattern-training crops')
    extra = additional_training_examples(original['examples'], rows, lookup,
        cfg['sampling']['expanded_max_per_subject'], cfg['sampling']['seed'])
    original_count = len(original['examples'])
    examples = np.concatenate((original['examples'], extra))
    if any(rows[i]['modality'] != 'eeg' or rows[i]['split'] != 'train' for i, _, _ in examples):
        raise ValueError('train boundary before target arrays')
    dtype_fields = {'S_attributes': (15, 10), 'O_attributes': (15, 10), 'background': (15,), 'valid': (15,), 'S': (15,), 'O': (15,), 'fixed_band': (), 'subject': (), 'fold': ()}
    bank = {'examples': examples}
    for key, shape in dtype_fields.items():
        bank[key] = np.empty((len(examples), *shape), dtype=original[key].dtype)
        bank[key][:original_count] = original[key]
    with np.load(ROOT/cfg['source']['vocabulary_checkpoint'], allow_pickle=False) as saved:
        model_metadata = json.loads(str(saved['metadata']))
    if model_metadata['codebook_size'] != 8192:
        raise ValueError('fixed original vocabulary required')
    names = model_metadata['positions']
    native = {d: {name.upper(): p for p, name in enumerate(v)} for d, v in names.items()}
    sites = {tuple(key): n for n, key in enumerate(population['background_sites'])}
    cache, ID_cache = OrderedDict(), OrderedDict()
    for start in range(0, len(extra), cfg['resources']['batch_size']):
        check_budget(cfg)
        take = extra[start:start + cfg['resources']['batch_size']]
        where = slice(original_count + start, original_count + start + len(take))
        for j, (i, t, p) in enumerate(take, original_count + start):
            key = int(i), int(t)
            path, ordinal = lookup[key]
            if path not in ID_cache:
                with np.load(path, allow_pickle=False) as saved:
                    ID_cache[path] = {b: saved[b].copy() for b in ('S', 'O')}
                if len(ID_cache) > 4:
                    ID_cache.popitem(last=False)
            row = rows[i]
            position = native[row['dataset']][row['positions'][p].upper()]
            for branch in ('S', 'O'):
                bank[branch][j] = ID_cache[path][branch][ordinal, position]
            bank['fixed_band'][j] = position % 5
            unit = row['dataset'], row['subject']
            bank['subject'][j] = unit_index[unit]
            bank['fold'][j] = folds[unit]
        batch = base_tokenizer_batch(rows, take, scales, array_cache=cache, cache_limit=64)
        valid = batch['mask'].all(-1)
        loc = np.array([sites[rows[i]['dataset'], rows[i]['positions'][p]] for i, _, p in take])
        bank['valid'][where] = valid
        bank['background'][where] = 4*loc[:, None] + np.searchsorted([.5, 1., 2.], np.sqrt(np.mean(batch['x']**2, axis=-1)), side='right')
        for branch in ('S', 'O'):
            target = bank[branch + '_attributes'][where]
            target[:] = 0
            raw, slices = attributes(batch[branch.lower()][valid], 'eeg')
            target[valid] = np.concatenate([raw[:, slice(*slices[k])] for k in FIELDS], axis=1)
        write_json(output/'progress.json', dict(stage='additional_train_attributes', completed=start + len(take), planned=len(extra), CPU_seconds=time.process_time()))
    for key in original:
        np.testing.assert_array_equal(bank[key][:original_count], original[key])
    np.savez_compressed(output/'sample_support_bank.npz', **bank)
    def describe(inputs):
        return dict(position_windows=len(inputs), exact_duplicates=len(inputs)-len({tuple(v) for v in inputs}),
            physical_crops=len({(int(i), int(t)) for i, t, _ in inputs}),
            records=len({(rows[i]['dataset'], rows[i]['subject'], rows[i]['record']) for i, _, _ in inputs}),
            subjects=len({(rows[i]['dataset'], rows[i]['subject']) for i, _, _ in inputs}))
    counts = []
    for unit in units:
        def count_inputs(a):
            return sum((rows[i]['dataset'], rows[i]['subject']) == unit for i, _, _ in a)
        counts.append(dict(dataset=unit[0], subject=unit[1], fold=folds[unit], original=count_inputs(original['examples']), expanded=count_inputs(examples)))
    record = dict(original=describe(original['examples']), added=describe(extra), expanded=describe(examples), per_subject=counts,
        new_physical_crops=len({(int(i), int(t)) for i, t, _ in extra}-{(int(i), int(t)) for i, t, _ in original['examples']}),
        original_complete_patches=int(original['valid'].sum()), expanded_complete_patches=int(bank['valid'].sum()),
        original_prefix_all_fields_bitwise_equal=True, additional_source='same already encoded native train crops, distinct admitted score positions',
        source_original_bank_sha256=digest(original_path), original_ID_file_sha256=digest(ROOT/cfg['source']['actual_ID_run']/'train_IDs.npz'),
        encoder_sha256=digest(ROOT/cfg['source']['encoder']), decoder_sha256=digest(ROOT/cfg['source']['decoder']),
        validation_arrays_read=0, AS_calls=0)
    write_json(output/'sample_population.json', record)
    return bank, original_count, len(sites)


def sample_support_worker(task):
    arm, expanded, fold, branch = task
    state = SAMPLE_SUPPORT_STATE
    folder = state['output']/'folds'
    folder.mkdir(exist_ok=True)
    stem = f'arm{arm}_fold{fold}_{branch}'
    artifact = folder/(stem+'.npz')
    metadata = folder/(stem+'.json')
    started = time.process_time()
    if artifact.exists() and metadata.exists():
        record = json.loads(metadata.read_text())
        if record['bank_sha256'] != state['bank_sha256']:
            raise ValueError('resumable fold uses a different sample bank')
        with np.load(artifact, allow_pickle=False) as data:
            values = tuple(data[k].copy() for k in ('score_rows', 'errors', 'flags', 'joint_count', 'background_count'))
        return task, (*values, record['counts'])
    with threadpool_limits(limits=1):
        result = sample_support_readout(state['bank'], state['original_count'], state['backgrounds'], expanded, fold, branch)
    rows, errors, flags, joint, background, counts = result
    pending = artifact.with_suffix('.pending.npz')
    np.savez_compressed(pending, score_rows=rows, errors=errors, flags=flags, joint_count=joint, background_count=background)
    pending.replace(artifact)
    write_json(metadata, dict(bank_sha256=state['bank_sha256'], counts=counts, CPU_seconds=time.process_time()-started))
    return task, result


def sample_support_scores(bank, original_count, backgrounds, cfg, rows, output):
    errors = {b: np.full((original_count, 2, 2, 3), np.nan) for b in ('S', 'O')}
    flags = {b: np.zeros((original_count, 2, 15, 3), bool) for b in ('S', 'O')}
    joint_count = {b: np.zeros((original_count, 2, 15), np.int32) for b in ('S', 'O')}
    bg_count = {b: np.zeros((original_count, 2, 15), np.int32) for b in ('S', 'O')}
    folds = []
    tasks = [(arm, expanded, fold, branch) for arm, expanded in enumerate((False, True)) for fold in range(4) for branch in ('S', 'O')]
    SAMPLE_SUPPORT_STATE.update(bank=bank, original_count=original_count, backgrounds=backgrounds, output=output,
                                bank_sha256=digest(output/'sample_support_bank.npz'))
    with ProcessPoolExecutor(max_workers=cfg['resources']['readout_workers'], mp_context=multiprocessing.get_context('fork')) as pool:
        pending = [pool.submit(sample_support_worker, task) for task in tasks]
        for completed in as_completed(pending):
            (arm, _, fold, branch), current_result = completed.result()
            score_rows, current, support, n_joint, n_bg, record = current_result
            errors[branch][score_rows, arm] = current
            flags[branch][score_rows, arm] = support
            joint_count[branch][score_rows, arm] = n_joint
            bg_count[branch][score_rows, arm] = n_bg
            folds.append(dict(arm=arm, branch=branch, **record))
            write_json(output/'progress.json', dict(stage='sample_size_crossfit', completed_groups=len(folds), planned_groups=len(tasks), CPU_seconds=time.process_time()))
    examples = bank['examples'][:original_count]
    score_valid = bank['valid'][:original_count]
    result = {}
    for arm, label in enumerate(('fit256', 'fit1024')):
        branches = {}
        for branch in ('S', 'O'):
            stats = hierarchy(errors[branch][:, arm], examples, rows)
            mse = np.asarray(stats['macro'])
            gain = 1 - mse[1]/mse[0]
            fraction = np.mean([np.asarray(x['value'])[1] < np.asarray(x['value'])[0] for x in stats['by_subject']], axis=0)
            current_flags = flags[branch][:, arm][score_valid]
            mass = current_flags.mean(0)
            passed = bool(min(gain[:2]) > 0 and max(gain) >= .05 and min(fraction[:2]) >= .5 and mass[1] >= .9)
            branches[branch] = dict(hierarchy=stats, gain=dict(zip(FIELDS, gain)), positive_subject_fraction=dict(zip(FIELDS, fraction)),
                ID_seen_mass=float(mass[0]), qualified_ID_mass=float(mass[1]), background_ID_seen_mass=float(mass[2]),
                joint_fit_patch_count_quantiles=np.quantile(joint_count[branch][:, arm][score_valid], [0, .1, .5, .9, 1]),
                background_fit_patch_count_quantiles=np.quantile(bg_count[branch][:, arm][score_valid], [0, .1, .5, .9, 1]),
                numerical_gate_pass=passed)
        result[label] = branches
    comparison = {}
    for branch in ('S', 'O'):
        a = np.asarray(result['fit256'][branch]['hierarchy']['macro'])
        b = np.asarray(result['fit1024'][branch]['hierarchy']['macro'])
        comparison[branch] = dict(background_relative_MSE_change=b[0]/a[0]-1, conditional_relative_MSE_change=b[1]/a[1]-1,
            expanded_joint_gain_vs_original_background=1-b[1]/a[0],
            paired_error_change=hierarchy(errors[branch][:, 1]-errors[branch][:, 0], examples, rows))
    record = dict(arms=result, comparison=comparison, folds=folds,
        readout_order=['background', 'background_plus_ID'], attributes=list(FIELDS),
        score_scope='same 15360 original train position windows in both arms; each subject held out of new readout fit',
        original_model_exposure=cfg['crossfit']['model_exposure'], all_expanded_gates=all(v['numerical_gate_pass'] for v in result['fit1024'].values()))
    np.savez_compressed(output/'sample_support_errors.npz', examples=examples,
        **errors, **{b+'_flags': f for b, f in flags.items()},
        **{b+'_joint_fit_count': f for b, f in joint_count.items()}, **{b+'_background_fit_count': f for b, f in bg_count.items()})
    write_json(output/'sample_support_summary.json', record)
    return record


def run_sample_support(args, cfg):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    resource.setrlimit(resource.RLIMIT_CPU, (cfg['resources']['CPU_budget_seconds'] - 10, cfg['resources']['CPU_budget_seconds']))
    if cfg['schema'] != 'dual_component_actual_id_sample_support_v1' or cfg['data']['allowed_splits'] != ['train']:
        raise ValueError('fixed training-only sample-size diagnostic required')
    if (output/'sample_completion.json').exists():
        raise FileExistsError('retained sample-size audit already completed')
    write_json(output/'resolved_config.json', cfg)
    base = yaml.safe_load((ROOT/cfg['source']['G0_contract']).read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(base, ROOT)
    write_json(output/'source_identity.json', sample_source_identity(cfg, manifest))
    bank, original_count, backgrounds = sample_support_bank(cfg, output, manifest['rows'], scales)
    record = sample_support_scores(bank, original_count, backgrounds, cfg, manifest['rows'], output)
    previous = json.loads((ROOT/cfg['source']['original_CV']).read_text())
    differences = {}
    for branch in ('S', 'O'):
        for level in ('macro', 'by_dataset'):
            current = record['arms']['fit256'][branch]['hierarchy'][level]
            old = previous[branch]['hierarchy'][level]
            if level == 'macro':
                np.testing.assert_array_equal(current, old)
                differences[branch] = float(np.max(np.abs(np.asarray(current)-np.asarray(old))))
            else:
                for dataset in current:
                    np.testing.assert_array_equal(current[dataset], old[dataset])
    write_json(output/'matched_original_CV.json', dict(status='passed', macro_max_abs_difference=differences,
        comparison='same raw categorical means, subject folds, source integers, targets and hierarchy as retained original fine-ID crossfit'))
    write_json(output/'sample_completion.json', dict(status='completed_fixed_sample_size_diagnostic',
        expanded_numeric_pass=record['all_expanded_gates'], CPU_seconds=time.process_time() + os.times().children_user + os.times().children_system,
        screen_read=False, validation_read=False, confirmation_read=False, test_read=False, AS_calls=0))


def review_sample_support(args, cfg):
    """Rebuild the readout in another process from preserved integers and targets."""
    output = args.output.resolve()
    previous = json.loads((output/'sample_completion.json').read_text())
    remaining = int(cfg['resources']['CPU_budget_seconds'] - previous['CPU_seconds'])
    if remaining < 30:
        raise RuntimeError('insufficient remaining independent-review CPU budget')
    resource.setrlimit(resource.RLIMIT_CPU, (remaining - 10, remaining))
    with np.load(output/'sample_support_bank.npz', allow_pickle=False) as saved:
        bank = {key: saved[key].copy() for key in saved.files}
    base = yaml.safe_load((ROOT/cfg['source']['G0_contract']).read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(base, ROOT)
    sample_source_identity(cfg, manifest)
    rows = manifest['rows']
    population = json.loads((ROOT/cfg['source']['original_population']).read_text())
    with np.load(ROOT/cfg['source']['original_bank'], allow_pickle=False) as saved:
        original_count = len(saved['examples'])
        for key in saved.files:
            np.testing.assert_array_equal(bank[key][:original_count], saved[key])
    retained_population = json.loads((output/'sample_population.json').read_text())
    for key in ('encoder', 'decoder'):
        if digest(ROOT/cfg['source'][key]) != retained_population[key+'_sha256']:
            raise ValueError('source tokenizer changed during sample-size diagnostic')
    lookup = native_blocks(cfg)
    with np.load(ROOT/cfg['source']['vocabulary_checkpoint'], allow_pickle=False) as saved:
        names = json.loads(str(saved['metadata']))['positions']
    native = {d: {name.upper(): j for j, name in enumerate(v)} for d, v in names.items()}
    cached = OrderedDict()
    for n, (i, start, p) in enumerate(bank['examples']):
        row = rows[i]
        if row['split'] != 'train' or row['modality'] != 'eeg':
            raise ValueError('review encountered a non-training target')
        path, ordinal = lookup[int(i), int(start)]
        if path not in cached:
            with np.load(path, allow_pickle=False) as saved:
                cached[path] = {b: saved[b].copy() for b in ('S', 'O')}
            if len(cached) > 4:
                cached.popitem(last=False)
        position = native[row['dataset']][row['positions'][p].upper()]
        for branch in ('S', 'O'):
            np.testing.assert_array_equal(bank[branch][n], cached[path][branch][ordinal, position])
    probes = []
    for dataset in sorted(names):
        eligible = [j for j in range(original_count, len(bank['examples'])) if rows[bank['examples'][j, 0]]['dataset'] == dataset]
        probes.extend(sorted(eligible, key=lambda j: rank(['sample-size-independent-truth', cfg['sampling']['seed'], *bank['examples'][j].tolist()]))[:2])
    target = base_tokenizer_batch(rows, bank['examples'][probes], scales, array_cache=OrderedDict(), cache_limit=16)
    valid = target['mask'].all(-1)
    np.testing.assert_array_equal(valid, bank['valid'][probes])
    for branch in ('S', 'O'):
        raw, slices = attributes(target[branch.lower()][valid], 'eeg')
        values = np.concatenate([raw[:, slice(*slices[key])] for key in FIELDS], axis=1)
        np.testing.assert_array_equal(values, bank[branch+'_attributes'][probes][valid])
    verify = output/'verification'
    verify.mkdir(exist_ok=True)
    link = verify/'sample_support_bank.npz'
    if not link.exists():
        link.symlink_to(output/'sample_support_bank.npz')
    elif link.resolve() != output/'sample_support_bank.npz':
        raise ValueError('verification bank identity changed')
    sample_support_scores(bank, original_count, len(population['background_sites']), cfg, rows, verify)
    with np.load(verify/'sample_support_errors.npz', allow_pickle=False) as new, np.load(output/'sample_support_errors.npz', allow_pickle=False) as old:
        if set(new.files) != set(old.files):
            raise AssertionError('independent error archive schema changed')
        for key in old.files:
            np.testing.assert_array_equal(new[key], old[key])
    write_json(output/'independent_review.json', dict(status='passed', integer_position_windows=len(bank['examples']),
        original_position_windows=original_count, all_S_O_native_integer_arrays_bitwise_equal=True,
        independently_reread_target_probes=bank['examples'][probes], errors_and_support_counts_bitwise_equal=True,
        raw_original_bank_prefix_bitwise_equal=True, readout_recomputed_in_separate_process=True,
        total_training_and_review_CPU_seconds=previous['CPU_seconds']+time.process_time()+os.times().children_user+os.times().children_system,
        original_subject_CV_not_an_independent_tokenizer_fit=True, validation_reads=0, AS_calls=0))


def render_sample_support(args, cfg):
    """Render the fixed paired numerical comparison and every dataset/subject row."""
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = args.output.resolve()
    summary = json.loads((output/'sample_support_summary.json').read_text())
    population = json.loads((output/'sample_population.json').read_text())
    with np.load(output/'sample_support_bank.npz', allow_pickle=False) as saved:
        original_count = population['original']['position_windows']
        examples = saved['examples'][:original_count].copy()
        valid = saved['valid'][:original_count].copy()
    base = yaml.safe_load((ROOT/cfg['source']['G0_contract']).read_text())
    manifest, _, _, _ = base_tokenizer_training_data(base, ROOT)
    rows = manifest['rows']
    entries = []
    with np.load(output/'sample_support_errors.npz', allow_pickle=False) as saved:
        for arm_index, arm in enumerate(('fit256', 'fit1024')):
            for branch in ('S', 'O'):
                stats = summary['arms'][arm][branch]['hierarchy']
                groups = [('macro', '', '', stats['macro'])]
                groups.extend(('dataset', d, '', v) for d, v in stats['by_dataset'].items())
                groups.extend(('subject', v['dataset'], v['subject'], v['value']) for v in stats['by_subject'])
                for level, dataset, subject, value in groups:
                    selected = np.array([(not dataset or rows[i]['dataset'] == dataset)
                                         and (not subject or rows[i]['subject'] == subject) for i, _, _ in examples])
                    masks = valid[selected]
                    flags = saved[branch+'_flags'][selected, arm_index][masks]
                    mse = np.asarray(value)
                    for column, attribute in enumerate(FIELDS):
                        entries.append(dict(arm=arm, branch=branch, level=level, dataset=dataset, subject=subject,
                            attribute=attribute, background_MSE=float(mse[0, column]), conditional_MSE=float(mse[1, column]),
                            ID_gain=float(1-mse[1, column]/mse[0, column]),
                            score_position_windows=int(selected.sum()), score_complete_patches=int(masks.sum()),
                            qualified_ID_patch_mass=float(flags[:, 1].mean()), background_ID_seen_patch_mass=float(flags[:, 2].mean())))
    with (output/'dataset_subject_metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    colors = ('#7e9bae', '#b65e36')
    labels = ['Theta 4-8 Hz', 'Alpha 8-13 Hz', 'Local RMS']
    x = np.arange(3)
    for column, branch in enumerate(('S', 'O')):
        ax = axes[0, column]
        for j, arm in enumerate(('fit256', 'fit1024')):
            values = np.array(list(summary['arms'][arm][branch]['gain'].values()))*100
            bars = ax.bar(x + (j-.5)*.34, values, .34, color=colors[j], label=arm)
            ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=3)
        ax.axhline(0, color='#263c47', lw=.8)
        ax.set(title=f'EEG-{branch}: local attribute increment', ylabel='1 - MSE(background + ID) / MSE(background), %',
               xticks=x, xticklabels=labels, ylim=(-25, 3))
        ax.legend(fontsize=9)
    ax = axes[1, 0]
    for j, arm in enumerate(('fit256', 'fit1024')):
        values = [summary['arms'][arm][b][key]*100 for key in ('qualified_ID_mass', 'background_ID_seen_mass') for b in ('S', 'O')]
        bars = ax.bar(np.arange(4) + (j-.5)*.34, values, .34, color=colors[j], label=arm)
        ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=3)
    ax.plot([-.4, 1.4], [90, 90], '--', color='#263c47', lw=1, label='Code-support gate')
    ax.set(title='Same held complete patches', ylabel='Held patch mass, %', ylim=(0, 106), xticks=np.arange(4),
           xticklabels=['S code support', 'O code support', 'S background x ID', 'O background x ID'])
    ax.tick_params(axis='x', rotation=15)
    ax.legend(fontsize=8, loc='lower right')
    ax = axes[1, 1]
    for j, key in enumerate(('background_relative_MSE_change', 'conditional_relative_MSE_change')):
        values = np.concatenate([summary['comparison'][b][key] for b in ('S', 'O')])*100
        bars = ax.bar(np.arange(6) + (j-.5)*.34, values, .34, color=colors[j], label='Background' if j == 0 else 'Background + ID')
        ax.bar_label(bars, fmt='%.1f', fontsize=7, padding=3)
    ax.axhline(0, color='#263c47', lw=.8)
    ax.set(title='1024 versus 256 fitting inputs per subject', ylabel='Relative MSE change, % (negative improves)',
           xticks=np.arange(6), xticklabels=['S theta', 'S alpha', 'S RMS', 'O theta', 'O alpha', 'O RMS'])
    ax.tick_params(axis='x', rotation=15)
    ax.legend(fontsize=9)
    fig.suptitle('Frozen AS integers: paired training-sample support diagnostic\n'
                 '60 original training subjects; four readout folds; same 15,360 held-score inputs', fontsize=13)
    fig.savefig(output/'sample_support_summary.png', dpi=230)
    plt.close(fig)
    write_json(output/'figure_manifest.json', dict(path='sample_support_summary.png',
        source='sample_support_summary.json', raw_group_table='dataset_subject_metrics.csv',
        independent_entities=60, held_original_positions=15360, original_physical_crops=5294,
        added_positions=population['added']['position_windows'], new_physical_crops=population['new_physical_crops'],
        caption='Both arms score exactly the original 256 positions per held subject. Extra fitting positions share the original physical crops. The original tokenizer already used these training subjects. This comparison refits only categorical readout means.',
        visual_review='pending_actual_open'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review', action='store_true', help='Independently replay a completed audit from its retained arrays.')
    parser.add_argument('--sample-support', action='store_true', help='Run the fixed training-subject 256 versus 1024 sample-size diagnostic.')
    parser.add_argument('--render', action='store_true', help='Render a completed sample-support comparison.')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.render and (not args.sample_support or args.review):
        parser.error('--render requires --sample-support and a separate invocation from --review')
    if args.sample_support != (cfg['schema'] == 'dual_component_actual_id_sample_support_v1'):
        parser.error('sample-support flag and configuration schema must agree')
    with threadpool_limits(limits=cfg['resources']['numerical_threads']):
        try:
            if args.sample_support and args.render:
                render_sample_support(args, cfg)
            elif args.sample_support and args.review:
                review_sample_support(args, cfg)
            elif args.sample_support:
                run_sample_support(args, cfg)
            elif args.review:
                review(args, cfg)
            else:
                run(args, cfg)
        except Exception as error:
            write_json(args.output/'failure.json', dict(error=repr(error), CPU_seconds=time.process_time()))
            raise


if __name__ == '__main__':
    main()
