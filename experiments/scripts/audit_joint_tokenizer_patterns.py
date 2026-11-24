"""Frozen-G0 local-token readouts, coverage, natural contexts and member atlases."""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.dual_component_tokenizer import base_training_aggregate
from src.analysis.tokenizer_pattern_audit import (
    GroupTotals, SparsePatternAudit, attributes, feature_means, match_training_patterns,
)
from src.data.dual_component_tokenizer import base_tokenizer_batch


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def pattern_attributes(values, modality):
    result, fields = attributes(values, modality)
    if modality == 'hb':
        wave = values.reshape(-1, 20, 2)
        additions = []
        offset = result.shape[-1]
        for c, color in enumerate(('HbO', 'HbR')):
            centered = wave[:, :, c] - wave[:, :, c].mean(1, keepdims=True)
            fields[color + '_centered'] = (offset, offset + 20)
            additions.append(centered)
            offset += 20
        result = np.concatenate([result, *additions], axis=-1)
    return result, fields


class MemberReservoir:
    """Fixed random priorities give a uniform, reproducible sample within each ID."""
    def __init__(self, seed=817, capacity=32):
        self.rng = np.random.default_rng(seed)
        self.capacity = capacity
        self.members = {}

    def add(self, split, ids, waves, metadata):
        priorities = self.rng.random(len(ids))
        for code in np.unique(ids):
            selected = np.flatnonzero(ids == code)
            take = selected[np.argsort(priorities[selected])[:self.capacity]]
            key = (split, int(code))
            value = (priorities[take], waves[take].astype(np.float32), metadata[take])
            previous = self.members.get(key)
            if previous is not None:
                value = tuple(np.concatenate([a, b]) for a, b in zip(previous, value))
            order = np.argsort(value[0])[:self.capacity]
            self.members[key] = tuple(v[order] for v in value)


def hierarchy(values, examples, rows):
    return base_training_aggregate({'nmse': values}, examples, rows)['metrics']['nmse']


def collect(model, rows, scales, panel, modality, out, *, batch_size=64, atlas_fraction=.9, panel_role='screen',
            max_atlas_pages=None):
    """Fit all readout statistics on the frozen pattern-training panel only."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    datasets = sorted({r['dataset'] for r in rows if r['modality'] == modality})
    sites = sorted({(r['dataset'], p) for r in rows if r['modality'] == modality and r['split'] == 'train'
                    for p in r['positions']})
    site_index = {site: n for n, site in enumerate(sites)}
    bins = np.asarray([.5, 1., 2.])
    width = 400 if modality == 'eeg' else 40
    _, fields = pattern_attributes(np.zeros((1, width)), modality)
    audits = {}
    training_second = {}
    samples = {b: MemberReservoir() for b in 'SO'}
    errors = defaultdict(list)
    code_archive = {}
    coverage = {}
    cache = OrderedDict()
    began = time.monotonic()
    for split in ('train', 'validation'):
        examples = np.asarray(panel[split], dtype=np.int64)
        if len(examples) == 0:
            raise ValueError(f'empty {split} pattern panel')
        expected = 'train' if split == 'train' else 'validation'
        if any(rows[i]['split'] != expected or rows[i]['modality'] != modality for i, _, _ in examples):
            raise ValueError('pattern panel split/modality boundary')
        saved_ids = defaultdict(list)
        complete = 0
        for start in range(0, len(examples), batch_size):
            take = examples[start:start + batch_size]
            batch = base_tokenizer_batch(rows, take, scales, array_cache=cache, cache_limit=128)
            ids = model.encode(batch['x'])
            decoded = model.decode_ids(ids)
            valid = batch['mask'].all(-1)
            complete += int(valid.sum())
            units = [(rows[i]['dataset'], rows[i]['subject'], rows[i]['record']) for i, _, _ in take]
            dataset = np.asarray([datasets.index(u[0]) for u in units])
            location = np.asarray([site_index.get((rows[i]['dataset'], rows[i]['positions'][p]), len(sites))
                                   for i, _, p in take])
            rms = np.sqrt(np.mean(batch['x'] ** 2, axis=-1))
            background = location[:, None] * (len(bins) + 1) + np.searchsorted(bins, rms, side='right')
            metadata = np.concatenate([np.repeat(take, 15, axis=0),
                                       np.tile(np.arange(15), len(take))[:, None]], axis=1)[valid.ravel()]
            for b in 'SO':
                codes = np.asarray(ids[b], dtype=np.int64)
                if codes.shape != valid.shape or (codes < 0).any():
                    raise ValueError('one nonnegative integer per branch per two-second patch required')
                saved_ids[b].append(codes)
                target, _ = pattern_attributes(batch[b.lower()][valid], modality)
                if b not in audits:
                    k = int(model.metadata.get('branch_codebook_sizes',{}).get(b,model.codebook_size))
                    audits[b] = SparsePatternAudit(k, (len(sites) + 1) * 4, datasets, target.shape[-1], fields)
                    training_second[b] = np.zeros(target.shape[-1])
                audit = audits[b]
                if codes.max(initial=0) >= audit.k:
                    raise ValueError('ID outside declared codebook')
                audit.support_batch(codes, valid, units, split)
                selected = codes[valid]
                samples[b].add(split, selected, batch[b.lower()][valid], metadata)
                if split == 'train':
                    audit.fit(selected, background[valid], target)
                    training_second[b] += np.square(target).sum(0)
                else:
                    predicted, _ = pattern_attributes(np.asarray(decoded[b])[valid], modality)
                    error = audit.score(selected, background[valid],
                                        np.broadcast_to(dataset[:, None], valid.shape)[valid], target, predicted)
                    values = np.zeros((*valid.shape, 4, len(fields)))
                    values[valid] = error
                    means = values.sum(1) / valid.sum(1)[:, None, None].clip(1)
                    means[~valid.any(1)] = np.nan
                    errors[b].append(means)
            write_json(out/'progress.json', dict(stage=split, inputs=start + len(take), planned=len(examples),
                                                elapsed_seconds=time.monotonic() - began))
        code_archive[split] = {b: np.concatenate(saved_ids[b]) for b in 'SO'}
        coverage[split] = dict(inputs=len(examples), planned_patches=len(examples) * 15,
                               complete_patches=complete, excluded_partial_or_missing=len(examples) * 15 - complete)
        np.savez_compressed(out/(split + '_IDs.npz'), examples=examples, **code_archive[split])
    summary = dict(status='completed', modality=modality, panel_role=panel_role, attributes=list(fields), coverage=coverage,
                   readouts=['background', 'single_ID', 'background_ID', 'integer_sequence_decoder'],
                   background='dataset/native-position crossed with X RMS bins [0.5,1,2] in frozen train SD',
                   fit='training complete patches only; observed joint means, unseen joint falls back to background',
                   amplitude_coordinate='G0 frozen training-position X SD; no per-window centering or scale input',
                   quantile_intervals='10/50/90 percentiles of uniform fixed-priority reservoirs, at most 32 members per split/ID',
                   atlas_fraction=atlas_fraction, max_atlas_pages=max_atlas_pages, branches={})
    validation_examples = np.asarray(panel['validation'], dtype=np.int64)
    render_tasks = []
    for b, audit in audits.items():
        atlas = audit.atlas(datasets, dict(minimum_train_subjects=2, minimum_train_patches=8))
        scores = hierarchy(np.concatenate(errors[b]), validation_examples, rows)
        macro = np.asarray(scores['macro'])
        gain = 1 - np.divide(macro[2], macro[0], out=np.full(len(fields), np.nan), where=macro[0] > 0)
        subject_gains = []
        for subject in scores['by_subject']:
            value = np.asarray(subject['value'])
            subject_gain = 1 - np.divide(value[2], value[0], out=np.full(len(fields), np.nan), where=value[0] > 0)
            subject_gains.append(dict(dataset=subject['dataset'], subject=subject['subject'],
                                      gain=dict(zip(fields, subject_gain))))
        count_total = max(int(audit.local.count.sum()), 1)
        global_mean = audit.local.total.sum(0) / count_total
        training_variance = feature_means(np.maximum(0, training_second[b] / count_total - global_mean ** 2), fields)
        summary['branches'][b] = dict(attribute_mse=scores,
            additional_ID_gain_over_background=dict(zip(fields, gain)), by_subject_gain=subject_gains,
            positive_subject_fraction={name: np.mean([s['gain'][name] > 0 for s in subject_gains]) for name in fields},
            by_dataset_gain={dataset: dict(zip(fields, 1 - np.divide(np.asarray(value)[2], np.asarray(value)[0],
                out=np.full(len(fields), np.nan), where=np.asarray(value)[0] > 0)))
                for dataset, value in scores['by_dataset'].items()},
            training_attribute_variance=dict(zip(fields, training_variance)),
            validation_mass_supported=1 - atlas['low_support_validation_patches'] / max(atlas['complete_validation_patches'], 1))
        write_json(out/(b + '_atlas.json'), dict(attribute_order=list(fields), **atlas))
        np.savez_compressed(out/(b + '_training_prototypes.npz'),
                            conditional_mean=audit.local.mean(np.arange(audit.k)), count=audit.local.count,
                            attribute_slices=np.asarray(list(fields.values())))
        rank = np.argsort(-audit.local.count, kind='stable')
        cumulative = np.cumsum(audit.local.count[rank]) / max(int(audit.local.count.sum()), 1)
        major_count = min(len(rank), int(np.searchsorted(cumulative, atlas_fraction) + 1))
        main_codes = rank[:major_count]
        rare = rank[(audit.local.count[rank] > 0) & atlas['low_training_support'][rank]]
        selected = list(main_codes)
        selected.extend(int(c) for c in rare[:6] if c not in selected)
        summary['branches'][b].update(major_codes=len(main_codes), major_training_mass=float(cumulative[major_count - 1]),
                                      rare_codes=int(len(rare)), unused_training_codes=int((audit.local.count == 0).sum()))
        export_members(out, b, samples[b])
        write_coverage_csv(out/(b + '_code_coverage.csv'), audit, atlas, rank)
        rendered = selected if max_atlas_pages is None else selected[:max_atlas_pages * 4]
        summary['branches'][b]['atlas_codes_rendered'] = len(rendered)
        summary['branches'][b]['atlas_selected_codes_retained_in_arrays'] = len(selected)
        render_tasks.append((b, rendered, audit, atlas, samples[b]))
    context = natural_context(model, rows, scales, validation_examples, audits, fields, modality, out, cache,
                              batch_size, {b: summary['branches'][b]['training_attribute_variance'] for b in 'SO'})
    for b in 'SO':
        summary['branches'][b]['natural_context'] = context[b]
        summary['branches'][b]['numerical_pattern_gate'] = pattern_gate(summary['branches'][b], modality, panel_role)
    summary['status'] = 'numerical_completed'
    write_json(out/'numerical_summary.json', summary)
    for b, selected, audit, atlas, reservoir in render_tasks:
        write_json(out/'progress.json', dict(stage='render_atlas', branch=b, selected_codes=len(selected),
                                            elapsed_seconds=time.monotonic() - began))
        render_atlas(out, b, modality, selected, audit, atlas, reservoir, fields)
    summary['status'] = 'completed'
    write_json(out/'summary.json', summary)
    write_json(out/'progress.json', dict(stage='completed', elapsed_seconds=time.monotonic() - began))
    return summary


def pattern_gate(branch, modality, panel_role='screen'):
    """Apply the fixed local-attribute criteria, leaving visual review explicit."""
    gain = branch['additional_ID_gain_over_background']
    if modality == 'eeg':
        primary = ['power_4_8', 'power_8_13', 'rms_envelope']
        attribute_pass = (gain['power_4_8'] > 0 and gain['power_8_13'] > 0
                          and max(gain[name] for name in primary) >= .05)
        subject_primary = primary[:2]
    else:
        primary = [color + suffix for color in ('HbO', 'HbR') for suffix in ('_slope', '_centered')]
        attribute_pass = all(gain[color + '_slope'] >= .1 and gain[color + '_centered'] > 0
                             for color in ('HbO', 'HbR'))
        subject_primary = primary
    subject_pass = all(branch['positive_subject_fraction'][name] >= .5 for name in subject_primary)
    context = branch['natural_context'].get('attribute_change_over_training_variance', {})
    context_pass = bool(context) and all(context[name] < .25 for name in primary)
    support_pass = branch['validation_mass_supported'] >= .9
    positive_datasets = {
        name: sum(values[name] > 0 for values in branch['by_dataset_gain'].values()) for name in subject_primary}
    result = dict(panel_role=panel_role, attributes=bool(attribute_pass), subject_coverage=bool(subject_pass),
                natural_context=bool(context_pass), training_support=bool(support_pass),
                numerical_pass=bool(attribute_pass and subject_pass and context_pass and support_pass),
                panel_positive_datasets=positive_datasets,
                visual_audit='pending_image_open_and_adjudication')
    if panel_role == 'confirm':
        result['confirmation_three_datasets_pass'] = all(n >= 3 for n in positive_datasets.values())
        result['numerical_pass'] = result['numerical_pass'] and result['confirmation_three_datasets_pass']
    return result


def export_members(out, branch, reservoir):
    arrays = {}
    for (split, code), (_, waves, info) in reservoir.members.items():
        prefix = f'{split}_{code}'
        arrays[prefix + '_waves'] = waves
        arrays[prefix + '_row_start_position_patch'] = info
    np.savez_compressed(out/(branch + '_atlas_members.npz'), **arrays)


def write_coverage_csv(path, audit, atlas, rank):
    with path.open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['rank', 'ID', 'train_patches', 'train_subjects', 'train_records', 'validation_patches',
                         'validation_subjects', 'validation_records', 'cumulative_training_fraction', 'low_training_support'])
        total = max(int(audit.local.count.sum()), 1)
        cumulative = 0
        for n, code in enumerate(rank, 1):
            cumulative += int(audit.local.count[code])
            support = atlas['support']
            writer.writerow([n, code, *[support[s][key][code] for s in ('train', 'validation')
                                       for key in ('patches', 'subjects', 'records')],
                             cumulative / total, atlas['low_training_support'][code]])


def render_atlas(out, branch, modality, codes, audit, atlas, reservoir, fields):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': ['DejaVu Sans'], 'font.size': 8})
    columns = 2 if modality == 'eeg' else 4
    count = audit.local.count
    total = max(int(count.sum()), 1)
    cumulative = np.cumsum(count[np.argsort(-count, kind='stable')]) / total
    ranking = np.empty(audit.k, dtype=int)
    ranking[np.argsort(-count, kind='stable')] = np.arange(audit.k)
    for first in range(0, len(codes), 4):
        page_codes = codes[first:first + 4]
        fig, axes = plt.subplots(len(page_codes), columns, figsize=(12, 2.55 * len(page_codes)),
                                 squeeze=False, layout='constrained')
        for row, code in enumerate(page_codes):
            prototype = audit.local.mean(np.asarray([code]))[0]
            train = reservoir.members.get(('train', int(code)))
            validation = reservoir.members.get(('validation', int(code)))
            wave = np.empty((0, 400 if modality == 'eeg' else 40)) if validation is None else validation[1]
            if modality == 'eeg':
                x = np.arange(400) / 200
                a, z = fields['voltage']
                display = [wave, None]
                template = [prototype[a:z], None]
                titles = ['Voltage / frozen training X SD', 'Power / frozen training X SD² per Hz']
                if len(wave):
                    centered = wave - wave.mean(1, keepdims=True)
                    window = np.hanning(400)
                    display[1] = abs(np.fft.rfft(centered * window, axis=1)) ** 2 / (200 * (window ** 2).sum()) * 2
                else:
                    display[1] = np.empty((0, 201))
                if train is not None:
                    w = train[1] - train[1].mean(1, keepdims=True)
                    template[1] = (abs(np.fft.rfft(w * np.hanning(400), axis=1)) ** 2 /
                                   (200 * (np.hanning(400) ** 2).sum()) * 2).mean(0)
                else:
                    template[1] = np.zeros(201)
                coordinates = [x, np.fft.rfftfreq(400, 1 / 200)]
            else:
                coordinates = [np.arange(20) / 10] * 4
                display, template = [], []
                for c, color in enumerate(('HbO', 'HbR')):
                    values = wave.reshape(-1, 20, 2)[:, :, c]
                    a, z = fields[color + '_wave']
                    p = prototype[a:z]
                    display.extend([values, values - values.mean(1, keepdims=True)])
                    template.extend([p, p - p.mean()])
                titles = ['HbO / frozen training X SD', 'HbO centered / same SD',
                          'HbR / frozen training X SD', 'HbR centered / same SD']
            for col, (axis, values, training_mean, coordinate) in enumerate(zip(axes[row], display, template, coordinates)):
                if len(values):
                    lo, median, hi = np.quantile(values, [.1, .5, .9], axis=0)
                    axis.fill_between(coordinate, lo, hi, color='#e3a24a', alpha=.25, label='Validation 10–90%')
                    axis.plot(coordinate, values[:3].T, lw=.65, alpha=.55, color='#8d652e')
                    axis.plot(coordinate, median, color='#bf6b20', lw=1.2, label='Validation median')
                training_label = ('Train spectral reservoir mean' if modality == 'eeg' and col == 1
                                  else 'Train conditional mean')
                axis.plot(coordinate, training_mean, color='#236c9e', lw=1.3, label=training_label)
                axis.set(xlabel='Time (s)' if col == 0 or modality == 'hb' else 'Frequency (Hz)', ylabel=titles[col])
                if modality == 'eeg' and col == 1:
                    axis.set_xlim(1, 45)
                axis.axhline(0, color='.75', lw=.5)
                if row == 0 and (col == 0 or modality == 'eeg'):
                    axis.legend(fontsize=6, loc='upper right')
            support = atlas['support']
            caption = (f'{branch} ID {code} | train n={count[code]}, subjects={support["train"]["subjects"][code]} | '
                       f'validation n={support["validation"]["patches"][code]}, subjects={support["validation"]["subjects"][code]} | '
                       f'occupancy={count[code]/total:.2%}, cumulative={cumulative[ranking[code]]:.1%}')
            if atlas['low_training_support'][code]:
                caption += ' | LOW TRAIN SUPPORT'
            axes[row, 0].set_title(caption, loc='left', fontsize=8)
        fig.suptitle(f'{modality.upper()} {branch} single-ID target distributions | complete 2-s patches | '
                     'three uniform reservoir members and training-only template', fontsize=10)
        fig.supxlabel('Axes scale independently across codes. Quantiles use at most 32 uniform reservoir members per split/ID.',
                      fontsize=8)
        fig.savefig(out/f'{branch}_atlas_{first//4 + 1:03d}.png', dpi=200)
        plt.close(fig)


def natural_context(model, rows, scales, examples, audits, fields, modality, out, cache, batch_size,
                    training_variances):
    # Metadata-only, fixed rule. Both parents are on this panel's subject/record/site.
    # Forward adjacent support is preferred, otherwise use the preceding score crop.
    pairs = []
    for n, (i, start, p) in enumerate(examples):
        support = {item['start_s'] for item in rows[i]['inputs']['score'] if p in item['positions']}
        if start + 30 in support:
            pairs.append((n, (i, start + 30, p), 1))
        elif start - 30 in support:
            pairs.append((n, (i, start - 30, p), -1))
    changes = defaultdict(list)
    kept = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        indices = np.asarray([pair[0] for pair in chunk])
        first = base_tokenizer_batch(rows, examples[indices], scales, array_cache=cache, cache_limit=128)
        adjacent = base_tokenizer_batch(rows, [pair[1] for pair in chunk], scales, array_cache=cache, cache_limit=128)
        forward = np.asarray([pair[2] == 1 for pair in chunk])
        shifted = np.empty_like(first['x'])
        shifted[forward] = np.concatenate([first['x'][forward, 1:], adjacent['x'][forward, :1]], axis=1)
        shifted[~forward] = np.concatenate([adjacent['x'][~forward, -1:], first['x'][~forward, :-1]], axis=1)
        original_ids, shifted_ids = model.encode(first['x']), model.encode(shifted)
        valid = first['mask'][:, 7].all(-1)
        kept.extend(indices[valid])
        for b, audit in audits.items():
            old = original_ids[b][:, 7][valid]
            new = shifted_ids[b][np.arange(len(chunk)), np.where(forward, 6, 8)][valid]
            supported = (audit.local.count[old] > 0) & (audit.local.count[new] > 0)
            difference = feature_means((audit.local.mean(old) - audit.local.mean(new)) ** 2, fields)
            difference[~supported] = np.nan
            changes[b].append(np.column_stack([old, new, old != new, supported, difference]))
    result = {}
    for b in 'SO':
        result[b] = dict(planned_adjacent_pairs=len(pairs), complete_central_patches=len(kept),
                         examples_without_adjacent_partner=len(examples) - len(pairs))
        if kept:
            values = np.concatenate(changes[b])
            np.savez_compressed(out/(b + '_natural_context.npz'), examples=examples[kept],
                                original_ID=values[:, 0], shifted_ID=values[:, 1],
                                supported=values[:, 3], attribute_squared_change=values[:, 4:])
            metrics = hierarchy(values[:, 2:], examples[kept], rows)
            result[b]['metrics'] = metrics
            result[b]['column_order'] = ['ID_changed', 'both_IDs_train_supported', *fields]
            variance = np.asarray([training_variances[b][name] for name in fields])
            change = np.asarray(metrics['macro'])[2:]
            normalized = np.divide(change, variance, out=np.full(len(fields), np.inf), where=variance > 0)
            normalized[(variance == 0) & (change == 0)] = 0
            result[b]['attribute_change_over_training_variance'] = dict(zip(fields, normalized))
            result[b]['training_variance_definition'] = 'pooled complete patches in the frozen pattern-training panel'
    return result


def compare_seeds(left, right, out):
    left, right, out = Path(left), Path(right), Path(out)
    lsummary, rsummary = [json.loads((p/'summary.json').read_text()) for p in (left, right)]
    if lsummary['attributes'] != rsummary['attributes'] or lsummary['coverage'] != rsummary['coverage']:
        raise ValueError('seed comparison requires matched attribute and panel support')
    if any(lsummary.get(key) != rsummary.get(key) for key in ('panel_sha256', 'panel', 'modality')):
        raise ValueError('seed comparison requires the same frozen physical panel')
    if lsummary['checkpoint_metadata']['seed'] == rsummary['checkpoint_metadata']['seed']:
        raise ValueError('seed comparison requires independent training seeds')
    if any(lsummary['checkpoint_metadata'].get(key) != rsummary['checkpoint_metadata'].get(key)
           for key in ('schema', 'codebook_size', 'architecture', 'quantization_levels')):
        raise ValueError('seed comparison requires the same model family and integer interface')
    matching_names = (['power_4_8', 'power_8_13', 'rms_envelope'] if lsummary['modality'] == 'eeg' else
                      [color + suffix for color in ('HbO', 'HbR') for suffix in ('_slope', '_centered')])
    result = {}
    for b in 'SO':
        with np.load(left/(b + '_training_prototypes.npz')) as l, np.load(right/(b + '_training_prototypes.npz')) as r:
            left_support = json.loads((left/(b + '_atlas.json')).read_text())['support']['train']['subjects']
            right_support = json.loads((right/(b + '_atlas.json')).read_text())['support']['train']['subjects']
            keep_l = np.flatnonzero((l['count'] >= 20) & (np.asarray(left_support) >= 2))
            keep_r = np.flatnonzero((r['count'] >= 20) & (np.asarray(right_support) >= 2))
            all_fields = dict(zip(lsummary['attributes'], map(tuple, l['attribute_slices'])))
            fields = {name: all_fields[name] for name in matching_names}
            i, j, cost = match_training_patterns(l['conditional_mean'][keep_l], r['conditional_mean'][keep_r], fields)
            write_json(out/(b + '_matched_IDs.json'), dict(left=keep_l[i], right=keep_r[j], normalized_attribute_cost=cost))
            result[b] = dict(matched_codes=len(i), left_supported=len(keep_l), right_supported=len(keep_r),
                             matching_attributes=matching_names,
                             normalized_attribute_cost_quantiles=np.quantile(cost, [0, .5, .9, 1]) if len(cost) else [],
                             left_gain=lsummary['branches'][b]['additional_ID_gain_over_background'],
                             right_gain=rsummary['branches'][b]['additional_ID_gain_over_background'])
            result[b]['permutation_invariant_gain_differences'] = {
                name: result[b]['right_gain'][name] - result[b]['left_gain'][name] for name in matching_names}
    write_json(out/'summary.json', dict(status='completed', left=str(left.resolve()), right=str(right.resolve()),
        panel_role=lsummary.get('panel_role', lsummary.get('panel')), panel_sha256=lsummary.get('panel_sha256'),
        left_seed=lsummary['checkpoint_metadata']['seed'], right_seed=rsummary['checkpoint_metadata']['seed'],
        minimum_training_patches=20, minimum_training_subjects=2, branches=result))


def rerender_retained(source, out, max_atlas_pages=None):
    """Render retained atlas members without signal access or encoder replay."""
    from types import SimpleNamespace
    source, out = Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = source/'numerical_summary.json'
    if not summary_path.exists():
        summary_path = source/'summary.json'
    summary = json.loads(summary_path.read_text())
    rendered = {}
    for branch in 'SO':
        atlas = json.loads((source/(branch + '_atlas.json')).read_text())
        with np.load(source/(branch + '_training_prototypes.npz')) as archive:
            means, counts = archive['conditional_mean'], archive['count']
            fields = dict(zip(summary['attributes'], map(tuple, archive['attribute_slices'])))
        local = GroupTotals(len(counts), means.shape[-1])
        local.count, local.total = counts, means * counts[:, None]
        audit = SimpleNamespace(k=len(counts), local=local)
        reservoir = MemberReservoir()
        with np.load(source/(branch + '_atlas_members.npz')) as archive:
            for key in archive.files:
                if key.endswith('_waves'):
                    prefix = key.removesuffix('_waves')
                    split, code = prefix.rsplit('_', 1)
                    waves = archive[key]
                    info = archive[prefix + '_row_start_position_patch']
                    reservoir.members[(split, int(code))] = (np.zeros(len(waves)), waves, info)
        rank = np.argsort(-counts, kind='stable')
        cumulative = np.cumsum(counts[rank]) / max(int(counts.sum()), 1)
        main = rank[:int(np.searchsorted(cumulative, summary.get('atlas_fraction', .9)) + 1)]
        selected = list(main)
        rare = [int(code) for code in rank if counts[code] > 0 and atlas['low_training_support'][code]]
        selected.extend(code for code in rare[:6] if code not in selected)
        if max_atlas_pages is not None:
            selected = selected[:max_atlas_pages * 4]
        render_atlas(out, branch, summary['modality'], selected, audit, atlas, reservoir, fields)
        rendered[branch] = selected
    write_json(out/'export.json', dict(source=str(source.resolve()), source_summary=str(summary_path.resolve()),
        max_atlas_pages=max_atlas_pages, rendered_codes=rendered,
        labels='per-code axis scaling and train spectral reservoir mean explicitly identified',
        new_signal_reads=0, new_encoder_replays=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--modality', choices=['eeg', 'hb'])
    parser.add_argument('--kind', choices=['dictionary', 'contextual', 'spectral', 'legacy', 'fullband'], default='dictionary')
    parser.add_argument('--legacy-run', type=Path)
    parser.add_argument('--panels', type=Path)
    parser.add_argument('--panel', default='screen', choices=['screen', 'development', 'confirm'])
    parser.add_argument('--config', type=Path, default=ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-atlas-pages', type=int)
    parser.add_argument('--compare', type=Path, nargs=2)
    parser.add_argument('--rerender', type=Path)
    args = parser.parse_args()
    if args.compare:
        compare_seeds(*args.compare, args.output)
        return
    if args.rerender:
        rerender_retained(args.rerender, args.output, args.max_atlas_pages)
        return
    if any(value is None for value in (args.checkpoint, args.modality, args.panels)):
        parser.error('--checkpoint, --modality and --panels are required')
    from run_dual_component_tokenizer import _base_training_model, identity, read_config
    from src.data.dual_component_tokenizer import base_tokenizer_training_data
    cfg = read_config(args.config)
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    panel_record = json.loads((args.panels.parent/'panel.json').read_text())
    if (panel_record['cfg_identity'] != manifest['cfg_identity']
            or panel_record['scales_identity'] != manifest['scales_identity']
            or panel_record['panel_sha256'] != hashlib.sha256(args.panels.read_bytes()).hexdigest()):
        raise ValueError('frozen panel G0 identity mismatch before signal arrays')
    if args.kind == 'legacy':
        import torch
        from src.tokenizers.contextual_latent_dictionary import ContextualLatentDictionary
        if args.legacy_run is None:
            parser.error('--legacy-run required for legacy checkpoint')
        legacy_config = json.loads((args.legacy_run/'resolved_config.json').read_text())['config']
        task = dict(modality=args.modality, architecture='waveform', candidate='vq128')
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        signature = saved['signature']
        if (signature['config'] != identity(legacy_config) or signature['task'] != task
                or signature['input'] != identity(manifest) or signature['scales'] != identity(scales)):
            raise ValueError('legacy checkpoint identity mismatch before signal arrays')
        underlying = _base_training_model(legacy_config, task)
        underlying.load_state_dict(saved['model_state'])
        metadata = dict(schema='legacy_joint_pattern_adapter_v1', modality=args.modality,
                        codebook_size=legacy_config['model']['codebook_size'],
                        cfg_identity=manifest['cfg_identity'], scales_identity=manifest['scales_identity'],
                        seed=legacy_config['training']['seed'], source_signature=signature)
        model = ContextualLatentDictionary(underlying, metadata)
    elif args.kind == 'contextual':
        from src.tokenizers.contextual_latent_dictionary import load_model
    elif args.kind == 'spectral':
        from src.tokenizers.spectral_waveform_dictionary import load_model
    elif args.kind == 'fullband':
        from src.tokenizers.fullband_eeg_tokenizer import load_model
    else:
        from src.tokenizers.supervised_waveform_dictionary import load_model
    if args.kind != 'legacy':
        model = load_model(args.checkpoint)
    metadata = model.metadata
    if metadata['modality'] != args.modality:
        raise ValueError('checkpoint modality mismatch before signal arrays')
    if args.kind == 'dictionary':
        if metadata['input_identity'] != identity(manifest):
            raise ValueError('checkpoint G0 input/scales identity mismatch before signal arrays')
    elif metadata['cfg_identity'] != manifest['cfg_identity'] or metadata['scales_identity'] != manifest['scales_identity']:
        raise ValueError('checkpoint G0 input/scales identity mismatch before signal arrays')
    if not hasattr(model, 'codebook_size'):
        model.codebook_size = int(metadata['codebook_size'])
    with np.load(args.panels) as archive:
        panel = dict(train=archive[args.modality + '_pattern_train'], validation=archive[args.modality + '_' + args.panel])
    if args.max_atlas_pages is not None and args.max_atlas_pages < 1:
        parser.error('--max-atlas-pages must be positive')
    summary = collect(model, manifest['rows'], scales, panel, args.modality, args.output,
                      batch_size=args.batch_size, panel_role=args.panel, max_atlas_pages=args.max_atlas_pages)
    summary.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_metadata=metadata,
                   panels=str(args.panels.resolve()), panel_sha256=panel_record['panel_sha256'], panel=args.panel)
    write_json(args.output/'summary.json', summary)


if __name__ == '__main__':
    main()
