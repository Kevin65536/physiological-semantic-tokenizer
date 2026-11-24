"""Raw single-ID information from the same EEG source in every training fold."""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import csv
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.dual_component_tokenizer import base_training_aggregate
from src.analysis.tokenizer_pattern_audit import SparsePatternAudit, GroupTotals, attributes, feature_means
from src.data.dual_component_tokenizer import base_tokenizer_training_data, base_tokenizer_batch, base_tokenizer_joint_eeg
from src.tokenizers.eeg_transient_gain import load_model as load_source
from src.tokenizers.spatial_analysis_synthesis import load_encoder

FIELDS = {'power_4_8': (0, 1), 'power_8_13': (1, 2), 'rms_envelope': (2, 10)}
STATE = {}


def write(path, value):
    def encode(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=encode)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_families(summary):
    """Frozen waveform-first rule; a failed family stays a mechanism diagnostic."""
    families = summary['families']
    eligible = [name for name, item in families.items() if item['train_waveform_eligible']]
    improvement = {}
    for name, item in families.items():
        old = np.ravel(item['arms']['legacy']['metrics']['nmse']['macro'])
        new = np.ravel(item['arms']['transient']['metrics']['nmse']['macro'])
        if old.shape != (3,) or new.shape != (3,) or not np.isfinite([old, new]).all() or np.any(old <= 0):
            raise ValueError('finite positive original waveform denominators required')
        improvement[name] = float(np.mean(1-new[1:]/old[1:]))
    return dict(families=eligible if eligible else [max(families, key=improvement.get)],
        mode='waveform_admitted' if eligible else 'mechanism_only_waveform_rejection_retained',
        mean_relative_component_improvement=improvement,
        original_waveform_eligibility={name: bool(item['train_waveform_eligible']) for name, item in families.items()})


def selected_scope(summary, cfg):
    chosen = select_families(summary)
    chosen['primary_selected_families'] = chosen['families'].copy()
    chosen['family_roles'] = {name: chosen['mode'] for name in chosen['families']}
    extra = cfg['selection'].get('additional_family')
    if extra:
        if extra not in summary['families']:
            raise ValueError('authorized extra family absent from frozen waveform result')
        chosen['family_roles'][extra] = cfg['selection']['additional_family_role']
        if extra not in chosen['families']:
            chosen['families'].append(extra)
    return chosen


def check_source(metadata, dataset, fold, folds, names, encoder_metadata):
    expected = {s for s, f in folds[dataset].items() if f != fold}
    if (metadata['held_fold'] != fold or set(metadata['fit_subjects']) != expected
            or metadata['positions'].get(dataset) != names
            or metadata['input_identity'] != encoder_metadata['input_identity']
            or metadata['scales_identity'] != encoder_metadata['scales_identity']):
        raise ValueError('same-fold source ancestry, exact fit people and coordinates required')


def cpu_budget(cfg, *, review=False):
    unit = os.environ.get('TOKENIZER_TRANSIENT_WORD_UNIT')
    prior = float(os.environ.get('TOKENIZER_TRANSIENT_WORD_PRIOR_CPU', '0'))
    used = time.process_time() if not unit else int(subprocess.check_output(
        ['systemctl', '--user', 'show', unit, '-p', 'CPUUsageNSec', '--value'], text=True))/1e9
    limit = cfg['resources']['total_CPU_budget_seconds']-20 if review else cfg['resources']['fit_CPU_stop_seconds']
    if prior+used >= limit:
        raise RuntimeError('bounded actual-word CPU budget reached; completed blocks retained')
    return prior+used


def load_inputs(cfg):
    hard = yaml.safe_load((ROOT/cfg['source']['hard_contract']).read_text())
    run = ROOT/hard['data']['run']
    basis = yaml.safe_load((ROOT/hard['data']['basis_config']).read_text())['source']
    manifest, scales, _, _ = base_tokenizer_training_data(yaml.safe_load((ROOT/basis['g0_config']).read_text()), ROOT)
    folds = json.loads((run/hard['data']['folds']).read_text())
    plan = json.loads((ROOT/basis['native_train_plan']).read_text())['plan']
    cases = []
    rows = manifest['rows']
    for dataset in sorted(plan):
        for subject, crop in sorted([(s, c) for s, crops in plan[dataset].items() for c in crops], key=lambda pair: (pair[1]['row'], pair[1]['start_s'])):
            row = rows[crop['row']]
            if (row['split'], row['modality'], row['dataset'], row['subject']) != ('train', 'eeg', dataset, subject):
                raise ValueError('original authorized train population required before arrays')
            cases.append(dict(dataset=dataset, subject=subject, fold=folds[dataset][subject], crop=crop))
    if (hard['data']['allowed_splits'] != ['train'] or cfg['data']['allowed_splits'] != ['train']
            or len(cases) != 400 or len({(c['dataset'], c['subject']) for c in cases}) != 60
            or sum(len(c['crop']['positions']) for c in cases) != 17040):
        raise ValueError('frozen 400/60/17040 train population required')
    return hard, run, manifest, scales, folds, cases


def prepare(cfg, hard, run, manifest, scales, cases, out):
    rows = manifest['rows']
    sites = sorted({(r['dataset'], name) for r in rows if r['split'] == 'train' and r['modality'] == 'eeg' for name in r['positions']})
    site_index = {v: j for j, v in enumerate(sites)}
    names = {}
    for dataset in {c['dataset'] for c in cases}:
        with np.load(run/hard['data']['source_model_run']/f'{dataset}_fold0_transient.npz', allow_pickle=False) as z:
            names[dataset] = json.loads(str(z['metadata']))['positions'][dataset]
    parts = defaultdict(list)
    cache = OrderedDict()
    offset = 0
    for number, c in enumerate(cases):
        cpu_budget(cfg)
        crop, dataset = c['crop'], c['dataset']
        ex = np.asarray([(crop['row'], crop['start_s'], p) for p in crop['positions']])
        batch = base_tokenizer_batch(rows, ex, scales, array_cache=cache, cache_limit=16)
        valid = batch['mask'].all(-1)
        native = np.asarray([{v.upper(): j for j, v in enumerate(names[dataset])}[rows[i]['positions'][p].upper()] for i, _, p in ex])
        background = 4*np.asarray([site_index[dataset, rows[i]['positions'][p]] for i, _, p in ex])[:, None]
        background = background+np.searchsorted([.5, 1., 2.], np.sqrt(np.mean(batch['x']**2, axis=-1)), side='right')
        c.update(first=offset, count=len(ex), native=native.tolist())
        offset += len(ex)
        parts['examples'].append(ex)
        parts['valid'].append(valid)
        parts['background'].append(background)
        parts['fold'].append(np.full(len(ex), c['fold'], np.int16))
        parts['fixed_band'].append(native % 5)
        for b in ('S', 'O'):
            values = np.zeros((*valid.shape, 10))
            raw, slices = attributes(batch[b.lower()][valid], 'eeg')
            values[valid] = np.concatenate([raw[:, slice(*slices[k])] for k in FIELDS], axis=1)
            parts[b+'_attributes'].append(values)
        if number % 25 == 0:
            write(out/'progress.json', dict(stage='original_training_attributes', crops=number+1, planned=400, CPU_seconds=cpu_budget(cfg)))
    bank = {k: np.concatenate(v) for k, v in parts.items()}
    people = sorted({(c['dataset'], c['subject']) for c in cases})
    lookup = {v: j for j, v in enumerate(people)}
    bank['subject'] = np.concatenate([np.full(c['count'], lookup[c['dataset'], c['subject']], np.int16) for c in cases])
    # Original anchored held cache is the population owner; its order may differ.
    for dataset in names:
        with np.load(run/'anchored_carrier/v1'/f'{dataset}_held.npz', allow_pickle=False) as old:
            actual = [tuple(v) for v in bank['examples'] if rows[v[0]]['dataset'] == dataset]
            if sorted(actual) != sorted(map(tuple, old['examples'])):
                raise ValueError('original anchored score population changed')
    np.savez_compressed(out/'training_attributes.npz', **bank)
    write(out/'population.json', dict(cases=cases, subjects=people, background_sites=sites, native_positions=names,
        scoring_positions=len(bank['examples']), complete_patches=int(bank['valid'].sum()),
        cfg_identity=manifest['cfg_identity'], scales_identity=manifest['scales_identity'],
        independent_entities='60 original training subjects; repeated windows remain dependent',
        validation_reads=0, confirmation_reads=0, test_reads=0))
    return bank, names, len(sites)*4


def root_ids(family, dataset, cases, hard, run, directory):
    spec = hard['families'][family]
    expected = [sha(run/spec['encoder']), sha(run/spec['decoder'])]
    result = {}
    for path in sorted((directory/family/dataset).glob('ids_*.npz')):
        with np.load(path, allow_pickle=False) as z:
            if z['identity'].tolist() != expected:
                raise ValueError('root actual IDs have different encoder/decoder identity')
            for j, key in enumerate(z['keys']):
                k = tuple(map(int, key))
                if k in result:
                    raise ValueError('duplicated original full-array integers')
                result[k] = {arm: {b: z[arm+'_'+b][j:j+1].copy() for b in ('S', 'O')} for arm in ('legacy', 'transient')}
    required = {(c['crop']['row'], c['crop']['start_s'], c['fold']) for c in cases if c['dataset'] == dataset}
    if result.keys() != required:
        raise ValueError('incomplete root OOF integers; do not duplicate held encoding')
    return result


def fit_worker(task):
    family, dataset, fold = task
    cfg, hard, run, rows, scales, folds, cases, out = (STATE[k] for k in ('cfg', 'hard', 'run', 'rows', 'scales', 'folds', 'cases', 'out'))
    directory = out/family/dataset/f'fold_{fold}'
    directory.mkdir(parents=True, exist_ok=True)
    source_path = run/hard['data']['source_model_run']/f'{dataset}_fold{fold}_transient.npz'
    source = load_source(source_path)
    encoder_path = run/hard['families'][family]['encoder']
    encoder = load_encoder(encoder_path, component_source=source)
    names = encoder.metadata['positions'][dataset]
    check_source(source.metadata, dataset, fold, folds, names, encoder.metadata)
    source_sha = sha(source_path)
    encoder_sha = sha(encoder_path)
    if encoder.codebook_size != hard['families'][family]['size']:
        raise ValueError('frozen vocabulary changed')
    root = root_ids(family, dataset, cases, hard, run, ROOT/cfg['source']['hard_run'])
    held = [c for c in cases if c['dataset'] == dataset and c['fold'] == fold]
    fit = [c for c in cases if c['dataset'] == dataset and c['fold'] != fold]
    cache = OrderedDict()
    probes = []
    # The sole allowed held recode is a deterministic exact-equivalence probe.
    probe = held[0]
    fields, observed, _ = base_tokenizer_joint_eeg(rows, dict(probe['crop'], positions=[]), scales, names, cache, allowed_splits=('train',))
    values = fields['x'].T.reshape(1, len(names), 15, 400)
    tick = time.process_time()
    measured = encoder.encode(values, dataset, observed[None])
    saved = root[probe['crop']['row'], probe['crop']['start_s'], fold]
    for b in ('S', 'O'):
        np.testing.assert_array_equal(measured[b], saved['transient'][b])
    probes.append(dict(arm='transient', crop=probe['crop'], source_fold=fold,
        native_X_dtype=str(values.dtype), X_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
        observed_sha256=hashlib.sha256(observed.tobytes()).hexdigest(), IDs_exact=True,
        CPU_seconds=time.process_time()-tick))
    if probe is min([c for c in cases if c['dataset'] == dataset], key=lambda c: (c['crop']['row'], c['crop']['start_s'])):
        original = load_encoder(encoder_path)
        old = original.encode(values, dataset, observed[None])
        for b in ('S', 'O'):
            np.testing.assert_array_equal(old[b], saved['legacy'][b])
        probes.append(dict(arm='legacy', crop=probe['crop'], source_fold_independent=True, IDs_exact=True))
        del original
    for start in range(0, len(fit), 8):
        cpu_budget(cfg)
        pairs = fit[start:start+8]
        keys = np.asarray([[c['crop']['row'], c['crop']['start_s'], c['fold']] for c in pairs])
        path = directory/f'fit_{start:04d}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                np.testing.assert_array_equal(z['keys'], keys)
                if str(z['source_sha256']) != source_sha or str(z['encoder_sha256']) != encoder_sha:
                    raise ValueError('resume fit source identity changed')
            continue
        arrays = defaultdict(list)
        for c in pairs:
            if c['subject'] not in source.metadata['fit_subjects'] or c['fold'] == fold:
                raise ValueError('held subject entered fit-source prototype collection')
            fields, observed, _ = base_tokenizer_joint_eeg(rows, dict(c['crop'], positions=[]), scales, names, cache, allowed_splits=('train',))
            values = fields['x'].T.reshape(1, len(names), 15, 400)
            assigned = encoder.encode(values, dataset, observed[None])
            arrays['observed'].append(observed)
            arrays['X_sha256'].append(hashlib.sha256(values.tobytes()).hexdigest())
            for b in ('S', 'O'):
                ids = assigned[b][0]
                if ids.dtype.kind not in 'iu' or ids.shape != (len(names), 15):
                    raise ValueError('actual full-array integer interface required')
                np.testing.assert_array_equal((ids != 0).any(-1), observed)
                arrays[b].append(ids)
        pending = path.with_suffix('.pending.npz')
        np.savez_compressed(pending, keys=keys, source_fold=fold, source_sha256=source_sha,
            encoder_sha256=encoder_sha, **arrays)
        pending.replace(path)
        write(directory/'progress.json', dict(fit_crops=min(start+8, len(fit)), planned=len(fit), CPU_seconds=time.process_time()))
    result = dict(family=family, dataset=dataset, source_fold=fold, fit_crops=len(fit), held_crops=len(held),
        fit_subjects=source.metadata['fit_subjects'], source_sha256=source_sha, encoder_sha256=encoder_sha,
        held_reencoding='only_the_retained_exact_equivalence_probe', probes=probes,
        CPU_seconds=time.process_time())
    write(directory/'completed.json', result)
    return result


def fit_codes(out, family, cases, bank, names, folds, hard, run):
    """Each fold receives its own fit-source integers; legacy is loaded only once."""
    root = {}
    for dataset in names:
        root[dataset] = root_ids(family, dataset, cases, hard, run, ROOT/STATE['cfg']['source']['hard_run'])
    legacy = {b: np.zeros(bank['valid'].shape, np.int32) for b in ('S', 'O')}
    held_transient = {b: np.zeros(bank['valid'].shape, np.int32) for b in ('S', 'O')}
    for c in cases:
        key = c['crop']['row'], c['crop']['start_s'], c['fold']
        sl = slice(c['first'], c['first']+c['count'])
        for b in ('S', 'O'):
            legacy[b][sl] = root[c['dataset']][key]['legacy'][b][0, c['native']]
            held_transient[b][sl] = root[c['dataset']][key]['transient'][b][0, c['native']]
    del root
    yield 'legacy', -1, legacy
    for fold in range(4):
        values = {b: v.copy() for b, v in held_transient.items()}
        records = {}
        for dataset in names:
            source_path = run/hard['data']['source_model_run']/f'{dataset}_fold{fold}_transient.npz'
            expected = sha(source_path)
            for path in sorted((out/family/dataset/f'fold_{fold}').glob('fit_*.npz')):
                with np.load(path, allow_pickle=False) as z:
                    if int(z['source_fold']) != fold or str(z['source_sha256']) != expected:
                        raise ValueError('mixed source folds in training prototype integers')
                    for j, key in enumerate(z['keys']):
                        key = tuple(map(int, key))
                        if key in records:
                            raise ValueError('duplicated fit crop')
                        records[key] = {b: z[b][j].copy() for b in ('S', 'O')}
        for c in cases:
            if c['fold'] == fold:
                continue
            key = c['crop']['row'], c['crop']['start_s'], c['fold']
            sl = slice(c['first'], c['first']+c['count'])
            one = records.pop(key)
            for b in ('S', 'O'):
                values[b][sl] = one[b][c['native']]
        if records:
            raise ValueError('unexpected source-fit crop')
        yield 'transient', fold, values


def raw_readout(bank, codes, fold, branch, capacity, backgrounds, *, fit_mask=None, held_mask=None, include_usage=True):
    fit = bank['fold'] != fold if fit_mask is None else np.asarray(fit_mask, bool)
    held = bank['fold'] == fold if held_mask is None else np.asarray(held_mask, bool)
    valid = bank['valid']
    if set(bank['subject'][fit]) & set(bank['subject'][held]):
        raise ValueError('subject overlap in categorical readout')
    words = codes[branch]+bank['fixed_band'][:, None]*capacity
    if np.any(codes[branch] < 0) or np.any(codes[branch] >= capacity):
        raise ValueError('actual ID outside frozen vocabulary')
    audit = SparsePatternAudit(5*capacity, backgrounds, [], 10, FIELDS)
    mask = valid & fit[:, None]
    audit.fit(words[mask], bank['background'][mask], bank[branch+'_attributes'][mask])
    counts = np.zeros(5*capacity, np.int32)
    for person in np.unique(bank['subject'][fit]):
        seen = valid & (bank['subject'][:, None] == person)
        counts[np.unique(words[seen])] += 1
    hv = valid[held]
    ids, bg = words[held][hv], bank['background'][held][hv]
    background, _, conditional = audit.read(ids, bg)
    target = bank[branch+'_attributes'][held][hv]
    error = np.zeros((*hv.shape, 2, 3))
    error[hv] = np.stack([feature_means((y-target)**2, FIELDS) for y in (background, conditional)], axis=1)
    per_window = error.sum(1)/hv.sum(1).clip(1)[:, None, None]
    per_window[~hv.any(1)] = np.nan
    supported = (audit.local.count[ids] >= 8) & (counts[ids] >= 2)
    indicator = np.zeros(hv.shape, bool)
    indicator[hv] = supported
    mass = indicator.sum(1)/hv.sum(1).clip(1)
    mass[~hv.any(1)] = np.nan
    held_counts = np.bincount(ids, minlength=5*capacity)
    usage = [dict(fold=fold, branch=branch, band=k//capacity, ID=k%capacity,
        fit_complete_patches=int(audit.local.count[k]), fit_subjects=int(counts[k]), held_complete_patches=int(held_counts[k])) for k in range(5*capacity)] if include_usage else []
    support = dict(complete_held_patches=len(ids), qualified_held_patches=int(supported.sum()),
        ID_seen=int((audit.local.count[ids] > 0).sum()),
        background_ID_seen=sum(audit.joint.count.get(int(k), 0) > 0 for k in bg*5*capacity+ids),
        fit_people=sorted(map(int, np.unique(bank['subject'][fit]))), held_people=sorted(map(int, np.unique(bank['subject'][held]))))
    return np.flatnonzero(held), per_window, mass, usage, support


def summarize(cfg, hard, run, manifest, folds, cases, bank, names, backgrounds, selected, out, *, independent=False):
    summaries = {}
    for family in selected:
        capacity = hard['families'][family]['size']
        result = {arm: {b: dict(error=np.zeros((len(bank['examples']), 2, 3)), mass=np.zeros(len(bank['examples'])), counts=[]) for b in ('S', 'O')} for arm in ('legacy', 'transient')}
        usage = []
        for arm, source_fold, codes in fit_codes(out, family, cases, bank, names, folds, hard, run):
            for fold in range(4) if arm == 'legacy' else [source_fold]:
                for b in ('S', 'O'):
                    selected_rows, errors, mass, occupied, count = raw_readout(bank, codes, fold, b, capacity, backgrounds)
                    result[arm][b]['error'][selected_rows] = errors
                    result[arm][b]['mass'][selected_rows] = mass
                    result[arm][b]['counts'].append(count)
                    usage.extend(dict(v, arm=arm) for v in occupied)
        if not independent:
            with (out/f'{family}_all_code_usage.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(usage[0]));writer.writeheader();writer.writerows(usage)
        arms = {}
        for arm, branches in result.items():
            branch_summary = {}
            for b, data in branches.items():
                hierarchy = base_training_aggregate({'nmse': data['error']}, bank['examples'], manifest['rows'])['metrics']['nmse']
                mse = np.asarray(hierarchy['macro'])
                gains = 1-mse[1]/mse[0]
                fraction = np.mean([np.asarray(v['value'])[1] < np.asarray(v['value'])[0] for v in hierarchy['by_subject']], axis=0)
                total = sum(v['complete_held_patches'] for v in data['counts'])
                supported = sum(v['qualified_held_patches'] for v in data['counts'])
                qualified = supported/total
                branch_summary[b] = dict(gain=dict(zip(FIELDS, gains)), positive_subject_fraction=dict(zip(FIELDS, fraction)),
                    hierarchy=hierarchy, support_mass=qualified, support_counts=data['counts'],
                    support_hierarchy=base_training_aggregate({'nmse': data['mass'][:, None, None]}, bank['examples'], manifest['rows'])['metrics']['nmse'],
                    passed=bool(min(gains[:2]) > 0 and max(gains) >= .05 and min(fraction[:2]) >= .5 and qualified >= .9))
            arms[arm] = branch_summary
            path = out/f'{family}_{arm}_readout_errors.npz'
            arrays = {b+'_'+k: v for b, d in branches.items() for k, v in d.items() if k in ('error', 'mass')}
            if independent:
                with np.load(path, allow_pickle=False) as saved:
                    for k, value in arrays.items():
                        np.testing.assert_array_equal(saved[k], value)
            else:
                np.savez_compressed(path, examples=bank['examples'], **arrays)
        summaries[family] = dict(arms=arms, both_transient_branches_passed=all(v['passed'] for v in arms['transient'].values()))
    return summaries


def render_and_stratify(summary, bank, rows, out, hard):
    """Display the fixed raw-loss readout, retaining dataset, role and person failures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'legacy': '#a9562d', 'transient': '#137e7b'}
    aliases = dict(eeg_fnirs_single_trial='ST', refed='REFED', simultaneous_eeg_nirs='Sim', visual_cognitive_motivation='Visual')
    fields = list(FIELDS)
    result, people, sites = {}, [], []
    for family, record in summary['families'].items():
        result[family] = {}
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), layout='constrained')
        for column, branch in enumerate(('S', 'O')):
            for offset, (arm, branches) in enumerate(record['arms'].items()):
                data = branches[branch]
                gains = np.asarray(list(data['gain'].values()))
                axes[0, column].bar(np.arange(3)+(-.18 if offset == 0 else .18), 100*gains, .34, color=colors[arm], label=arm)
            axes[0, column].set(xticks=range(3), xticklabels=['Theta', 'Alpha', 'Local RMS'], ylabel='Error reduction (%)', title=branch+': single-ID error reduction')
            axes[0, column].axhline(0, color='gray', lw=.8)
            axes[0, column].axhline(5, color='gray', ls=':', lw=.8)
            axes[0, column].legend()
            datasets = record['arms']['legacy'][branch]['hierarchy']['by_dataset']
            grid = []
            labels = []
            for dataset in datasets:
                for arm in ('legacy', 'transient'):
                    mse = np.asarray(record['arms'][arm][branch]['hierarchy']['by_dataset'][dataset])
                    grid.append(100*(1-mse[1]/mse[0]));labels.append(aliases.get(dataset, dataset)+' / '+arm)
            limit = max(5, float(np.max(np.abs(grid))))
            im = axes[1, column].imshow(grid, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
            for y, row in enumerate(grid):
                for x, value in enumerate(row):
                    axes[1, column].text(x, y, f'{value:.1f}', ha='center', va='center', color='white' if abs(value) > limit*.55 else 'black', fontsize=8)
            axes[1, column].set(yticks=range(len(labels)), yticklabels=labels, xticks=range(3), xticklabels=['Theta', 'Alpha', 'Local RMS'], title=branch+': dataset error reduction (%)')
            fig.colorbar(im, ax=axes[1, column], label='Reduction (%)')
        for offset, arm in enumerate(('legacy', 'transient')):
            data = record['arms'][arm]
            axes[0, 2].bar(np.arange(2)+(-.18 if offset == 0 else .18), [data[b]['support_mass'] for b in ('S', 'O')], .34, color=colors[arm], label=arm)
            fractions = [data[b]['positive_subject_fraction'][k] for b in ('S', 'O') for k in fields[:2]]
            axes[1, 2].bar(np.arange(4)+(-.18 if offset == 0 else .18), fractions, .34, color=colors[arm], label=arm)
        axes[0, 2].set(xticks=[0, 1], xticklabels=['S', 'O'], ylim=(0, 1.05), title='Held support mass\n>=8 fit patches and >=2 fit people')
        axes[0, 2].axhline(.9, color='black', ls=':', lw=1)
        axes[1, 2].set(xticks=range(4), xticklabels=['S theta', 'S alpha', 'O theta', 'O alpha'], ylim=(0, 1.05), title='Positive theta/alpha gain\nFraction of 60 held people')
        axes[1, 2].axhline(.5, color='black', ls=':', lw=1)
        fig.suptitle(f'{family} | K{hard["families"][family]["size"]} | 400 native train crops, 60 people, four subject folds, 17,040 position-windows\nEach source_f encodes its fit-person prototypes; held integers reused. Frozen G0 coordinate; original raw attribute MSE.\nTheta 4–8 Hz, alpha 8–13 Hz; local RMS: eight 0.25-s blocks. Original waveform verdict retained.', fontsize=11)
        fig.supxlabel('Gain = 1 − MSE(background + actual ID) / MSE(background). Positions -> crop -> record -> subject -> dataset, equal within each level.\nST: EEG-fNIRS single trial; Sim: simultaneous EEG-NIRS; Visual: visual cognitive motivation. Dotted lines: original thresholds.', fontsize=9)
        path = out/f'{family}_raw_single_ID_information_v2.png'
        if path.exists():
            raise FileExistsError('retained figure export; use a new version for a revised figure')
        fig.savefig(path, dpi=200, facecolor='white', bbox_inches='tight');plt.close(fig)
        for arm, branches in record['arms'].items():
            result[family][arm] = {}
            with np.load(out/f'{family}_{arm}_readout_errors.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['examples'], bank['examples'])
                for branch, data in branches.items():
                    error, mass = saved[branch+'_error'], saved[branch+'_mass']
                    by_band = {}
                    for band in range(5):
                        selected = bank['fixed_band'] == band
                        stats = base_training_aggregate({'nmse': error[selected]}, bank['examples'][selected], rows)['metrics']['nmse']
                        mse = np.asarray(stats['macro'])
                        patches = bank['valid'][selected].sum(1)
                        by_band[band] = dict(gain=dict(zip(fields, 1-mse[1]/mse[0])), hierarchy=stats,
                            complete_patches=int(patches.sum()), qualified_patch_mass=float(np.sum(mass[selected]*patches)/patches.sum()))
                    result[family][arm][branch] = by_band
                    for item in data['hierarchy']['by_subject']:
                        mse = np.asarray(item['value'])
                        unit = next(j for j, (d, s) in enumerate(json.loads((out/'population.json').read_text())['subjects']) if (d, s) == (item['dataset'], item['subject']))
                        selected = bank['subject'] == unit
                        n = bank['valid'][selected].sum(1)
                        people.append(dict(family=family, arm=arm, branch=branch, dataset=item['dataset'], subject=item['subject'], windows=int(selected.sum()), complete_patches=int(n.sum()),
                            qualified_patch_mass=float(np.sum(mass[selected]*n)/n.sum()), **{k: float(v) for k, v in zip(fields, 1-mse[1]/mse[0])}))
                    locations = defaultdict(list)
                    for j, (i, start, position) in enumerate(bank['examples']):
                        locations[rows[i]['dataset'], rows[i]['positions'][position]].append(j)
                    for (dataset, native), indices in locations.items():
                        indices = np.asarray(indices)
                        stats = base_training_aggregate({'nmse': error[indices]}, bank['examples'][indices], rows)['metrics']['nmse']
                        mse = np.asarray(stats['macro']);n=bank['valid'][indices].sum(1)
                        sites.append(dict(family=family, arm=arm, branch=branch, dataset=dataset, position=native, fixed_band=int(bank['fixed_band'][indices[0]]), windows=len(indices), complete_patches=int(n.sum()),
                            qualified_patch_mass=float(np.sum(mass[indices]*n)/n.sum()), **{k: float(v) for k, v in zip(fields, 1-mse[1]/mse[0])}))
    for name, records in [('subject_attributes.csv', people), ('native_site_attributes.csv', sites)]:
        if not (out/name).exists():
            with (out/name).open('w') as f:
                writer=csv.DictWriter(f, fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        else:
            with (out/name).open() as f:
                retained=list(csv.DictReader(f))
            if len(retained) != len(records) or any(any(str(row[k]) != original[k] for k in original) for row, original in zip(records, retained)):
                raise AssertionError('figure-only export changed retained stratified values')
    if (out/'frequency_role_attributes.json').exists():
        if json.loads(json.dumps(result, default=lambda a: a.tolist())) != json.loads((out/'frequency_role_attributes.json').read_text()):
            raise AssertionError('figure-only export changed retained frequency-role values')
    else:
        write(out/'frequency_role_attributes.json', result)
    write(out/'render_manifest_v2.json', dict(figures=[str(out/f'{family}_raw_single_ID_information_v2.png') for family in summary['families']],
        subject_rows=len(people), native_site_rows=len(sites), plot_values='exact original raw conditional-readout errors; positive favors ID',
        source='summary.json and *_readout_errors.npz', visual_opening_required=True))


def review_input_identities(cfg, hard, run, population, out):
    """Every repeated physical crop retains one X and observation identity."""
    reference, observed_reference, reports = {}, {}, []
    folds = json.loads((run/hard['data']['folds']).read_text())
    for family in json.loads((out/'selection.json').read_text())['families']:
        enc = json.loads((run/hard['families'][family]['encoder']).read_text())
        count, repeat, integers = 0, defaultdict(int), 0
        for dataset, names in population['native_positions'].items():
            for fold in range(4):
                source_path = run/hard['data']['source_model_run']/f'{dataset}_fold{fold}_transient.npz'
                source = load_source(source_path)
                check_source(source.metadata, dataset, fold, folds, names, enc)
                expected = sha(source_path)
                for path in sorted((out/family/dataset/f'fold_{fold}').glob('fit_*.npz')):
                    with np.load(path, allow_pickle=False) as z:
                        if int(z['source_fold']) != fold or str(z['source_sha256']) != expected or str(z['encoder_sha256']) != sha(run/hard['families'][family]['encoder']):
                            raise ValueError('all-block source and encoder identity changed')
                        for j, key in enumerate(z['keys']):
                            crop = dataset, int(key[0]), int(key[1])
                            if int(key[2]) == fold:
                                raise ValueError('held crop entered the fitting prototype bank')
                            identity, seen = str(z['X_sha256'][j]), z['observed'][j]
                            if crop in reference:
                                if reference[crop] != identity:
                                    raise ValueError('physical full-X changed between source folds or families')
                                np.testing.assert_array_equal(observed_reference[crop], seen)
                            else:
                                reference[crop], observed_reference[crop] = identity, seen.copy()
                            repeat[crop] += 1
                            count += 1
                        integers += z['S'].size+z['O'].size
        if len(repeat) != 400 or set(repeat.values()) != {3}:
            raise ValueError('each physical crop must fit exactly the other three source folds')
        reports.append(dict(family=family, fit_source_crops=count, distinct_physical_crops=len(repeat),
            source_folds_per_physical_crop=3, actual_fit_integers=integers, full_X_float64_hash_exact=True,
            observed_exact=True, same_source_fit_people_verified=True,
            frozen_dictionary_PCA_sha256=enc['source_sha256']))
    write(out/'input_and_source_identity_review.json', dict(reports=reports,
        cross_family_X_and_observed_exact=True, unique_physical_crops=len(reference),
        decoder_input_floating_sidecars=0, scope='stored original 400 train crops only'))


def background_roundoff_review(bank, rows, families, backgrounds, out):
    """Keep the failed bitwise check and explain its original global-mean fallback."""
    result, arrays = {}, {}
    for branch in ('S', 'O'):
        canonical = np.zeros((len(bank['examples']), 3))
        unseen = np.zeros(bank['valid'].shape, bool)
        for fold in range(4):
            fit, held = bank['fold'] != fold, bank['fold'] == fold
            fitting = bank['valid'] & fit[:, None]
            y = bank[branch+'_attributes'][fitting]
            bg = bank['background'][fitting]
            table = GroupTotals(backgrounds, 10);table.add(bg, y)
            hv = bank['valid'][held]
            held_bg = bank['background'][held][hv]
            # The same mathematical fallback mean, summed in original sample
            # order, removes the ID-dependent summation path for this audit only.
            predicted = np.where(table.count[held_bg, None] > 0, table.mean(held_bg), y.mean(0))
            error = np.zeros((*hv.shape, 3))
            error[hv] = feature_means((predicted-bank[branch+'_attributes'][held][hv])**2, FIELDS)
            canonical[held] = error.sum(1)/hv.sum(1).clip(1)[:, None]
            block = np.zeros(hv.shape, bool);block[hv] = table.count[held_bg] == 0
            unseen[held] = block
        arrays[branch+'_common_background_error'] = canonical
        arrays[branch+'_unseen_background_patch'] = unseen
        common = base_training_aggregate({'nmse': canonical[:, None]}, bank['examples'], rows)['metrics']['nmse']
        macro = np.ravel(common['macro'])
        result[branch] = {}
        for family in families:
            with np.load(out/f'{family}_legacy_readout_errors.npz', allow_pickle=False) as old, np.load(out/f'{family}_transient_readout_errors.npz', allow_pickle=False) as new:
                before, after = old[branch+'_error'][:, 0], new[branch+'_error'][:, 0]
                # The assertion is retained as an explicit failed numerical
                # finding; it is never reported as a passed bitwise comparison.
                try:
                    np.testing.assert_array_equal(before, after)
                    exact = True
                except AssertionError:
                    exact = False
                difference = np.abs(after-before)
                affected = np.any(difference != 0, axis=1)
                if np.any(affected & ~unseen.any(1)):
                    raise AssertionError('background changed outside the original unseen-bin global fallback')
                rebased = {}
                for arm, store in [('legacy', old), ('transient', new)]:
                    stats = base_training_aggregate({'nmse': store[branch+'_error']}, bank['examples'], rows)['metrics']['nmse']
                    original = np.asarray(stats['macro'])
                    original_gain = 1-original[1]/original[0]
                    gain = 1-original[1]/macro
                    original_people = np.mean([np.asarray(v['value'])[1] < np.asarray(v['value'])[0] for v in stats['by_subject']], axis=0)
                    canonical_people = {(v['dataset'], v['subject']): np.ravel(v['value']) for v in common['by_subject']}
                    people = np.mean([np.asarray(v['value'])[1] < canonical_people[v['dataset'], v['subject']] for v in stats['by_subject']], axis=0)
                    if (min(original_gain[:2]) > 0, max(original_gain) >= .05) != (min(gain[:2]) > 0, max(gain) >= .05):
                        raise AssertionError('original raw-property gate changed under a common background sum')
                    np.testing.assert_array_equal(people, original_people)
                    rebased[arm] = dict(original_gain=original_gain, common_background_gain= gain,
                        maximum_gain_difference=float(np.max(abs(gain-original_gain))), same_primary_gate_and_subject_fractions=True,
                        maximum_background_error_difference=float(np.max(abs(store[branch+'_error'][:, 0]-canonical))))
                result[branch][family] = dict(bitwise_equal=exact, mismatched_cells=int(np.count_nonzero(difference)),
                    maximum_absolute_difference=float(difference.max()), affected_windows=int(affected.sum()),
                    differences_only_in_original_unseen_background_fallback=True, complete_held_patches_in_unseen_background=int(unseen.sum()),
                    explanation='SparsePatternAudit original global fallback sums local.total grouped by actual ID; ID grouping changes final floating-point reduction order',
                    original_scores_unchanged=True, common_background_denominator_check=rebased)
    np.savez_compressed(out/'common_background_numerical_audit.npz', **arrays)
    write(out/'background_roundoff_review.json', result)
    return result


def role_mix_inner_worker(outer):
    from src.analysis.role_mixed_information import (
        RoleInformationPanel,
        select_fixed_mask,
    )

    bank, cfg = STATE["role_mix_bank"], STATE["cfg"]
    inner_errors, inner_mass = {}, {}
    for family in ["high", "semantic"]:
        codes = STATE["role_mix_codes"][family, outer]
        for branch in "SO":
            error = np.full((len(bank["examples"]), 2, 3), np.nan)
            mass = np.full(len(bank["examples"]), np.nan)
            for inner in range(4):
                if inner == outer:
                    continue
                cpu_budget(cfg)
                fit = (bank["fold"] != outer) & (bank["fold"] != inner)
                held = bank["fold"] == inner
                take, values, support, _, _ = raw_readout(
                    bank,
                    codes,
                    inner,
                    branch,
                    STATE["hard"]["families"][family]["size"],
                    STATE["role_mix_backgrounds"],
                    fit_mask=fit,
                    held_mask=held,
                )
                error[take], mass[take] = values, support
            inner_errors[family, branch], inner_mass[family, branch] = error, mass
    records, selected, feasible = {}, {}, {}
    for branch in "SO":
        panel = RoleInformationPanel(
            inner_errors["high", branch],
            inner_errors["semantic", branch],
            inner_mass["high", branch],
            inner_mass["semantic", branch],
            bank,
            STATE["rows"],
            bank["fold"] != outer,
        )
        records[branch] = {mask: panel.evaluate(mask) for mask in range(32)}
        feasible[branch] = [
            mask for mask, result in records[branch].items() if result["passed"]
        ]
        selected[branch] = select_fixed_mask(
            records[branch], cfg["selection"]["fixed_priority_high_roles"]
        )
    result = dict(
        outer_fold=outer,
        records=records,
        feasible=feasible,
        selected_masks={b: selected[b] if selected[b] is not None else 0 for b in "SO"},
        no_admissible_branch=[b for b in "SO" if selected[b] is None],
        outer_people_seen_in_selection=False,
        source="same outer-source_f actual fit-person integers",
        CPU_seconds=time.process_time(),
    )
    write(STATE["out"] / f"outer{outer}_inner_selection.json", result)
    return result


def run_role_mix_information(args):
    from src.analysis.role_mixed_information import RoleInformationPanel, mix_by_role

    cfg = yaml.safe_load(args.config.read_text())
    if cfg["schema"] != "fixed_role_mixed_spatial_v1" or any(
        cfg["data"][k] != "forbidden"
        for k in ["validation_access", "confirmation_access", "test_access"]
    ):
        raise ValueError("frozen train-only role mixture required")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "information_summary.json").exists():
            raise FileExistsError("retained completed role feasibility result")
        hard, run, manifest, scales, folds, _ = load_inputs(cfg)
        source = ROOT / cfg["source"]["local_information"]
        population = json.loads((source / "population.json").read_text())
        if (
            population["cfg_identity"] != manifest["cfg_identity"]
            or population["scales_identity"] != manifest["scales_identity"]
        ):
            raise ValueError("canonical same-source attributes changed")
        with np.load(source / "training_attributes.npz") as f:
            bank = {k: f[k] for k in f}
        if len(bank["examples"]) != 17040 or len(population["cases"]) != 400:
            raise ValueError("same 400/17040 population required")
        cases, names = population["cases"], population["native_positions"]
        backgrounds = len(population["background_sites"]) * 4
        STATE.update(
            cfg=cfg,
            hard=hard,
            run=run,
            rows=manifest["rows"],
            scales=scales,
            folds=folds,
            cases=cases,
            out=out,
            role_mix_bank=bank,
            role_mix_backgrounds=backgrounds,
        )
        write(
            out / "resolved_config.json",
            dict(
                config=cfg,
                canonical_population_sha256=sha(source / "population.json"),
                canonical_attributes_sha256=sha(source / "training_attributes.npz"),
                readout_source="actual pure source_f fitting codes and original held errors",
            ),
        )
        cache, panels, tables = {}, {}, {}
        background_audit = {}
        prior = json.loads((source / "summary.json").read_text())
        for arm in ["legacy", "transient"]:
            for family in ["high", "semantic"]:
                file = source / f"{family}_{arm}_readout_errors.npz"
                with np.load(file) as f:
                    np.testing.assert_array_equal(f["examples"], bank["examples"])
                    cache[family, arm] = {k: f[k] for k in f if k != "examples"}
            for branch in "SO":
                high, low = cache["high", arm], cache["semantic", arm]
                delta = high[branch + "_error"][:, 0] - low[branch + "_error"][:, 0]
                background_audit[arm + "_" + branch] = dict(
                    max_abs_difference=float(np.nanmax(abs(delta))),
                    reason="unseen background global fallback sums training target totals in dictionary-ID order",
                    targets_and_support="unchanged; finite masks exact",
                    comparison_rtol=1e-12,
                    comparison_atol=1e-14,
                )
                panel = RoleInformationPanel(
                    high[branch + "_error"],
                    low[branch + "_error"],
                    high[branch + "_mass"],
                    low[branch + "_mass"],
                    bank,
                    manifest["rows"],
                )
                panels[arm, branch] = panel
                tables[arm, branch] = {mask: panel.evaluate(mask) for mask in range(32)}
                for family, mask in [("semantic", 0), ("high", 31)]:
                    actual, expected = (
                        tables[arm, branch][mask],
                        prior["families"][family]["arms"][arm][branch],
                    )
                    np.testing.assert_allclose(
                        actual["hierarchy"]["macro"],
                        expected["hierarchy"]["macro"],
                        rtol=1e-12,
                        atol=1e-14,
                    )
                    for key in FIELDS:
                        np.testing.assert_allclose(
                            actual["gain"][key],
                            expected["gain"][key],
                            rtol=1e-12,
                            atol=1e-14,
                        )
                    np.testing.assert_allclose(
                        actual["qualified_mass"],
                        expected["support_mass"],
                        rtol=1e-12,
                        atol=1e-14,
                    )
                # Verify a genuine mixture with the unoptimized original hierarchy.
                for mask in [1, 25]:
                    errors = mix_by_role(
                        high[branch + "_error"],
                        low[branch + "_error"],
                        bank["fixed_band"],
                        mask,
                    )
                    expected = base_training_aggregate(
                        {"nmse": errors}, bank["examples"], manifest["rows"]
                    )["metrics"]["nmse"]
                    np.testing.assert_allclose(
                        tables[arm, branch][mask]["hierarchy"]["macro"],
                        expected["macro"],
                        rtol=1e-12,
                        atol=1e-14,
                    )
        feasible = {
            arm: {
                b: [m for m, record in tables[arm, b].items() if record["passed"]]
                for b in "SO"
            }
            for arm in ["legacy", "transient"]
        }
        write(out / "background_float64_path_audit.json", background_audit)
        write(
            out / "full_OOF_information_sets.json",
            dict(
                scope="descriptive exhaustive stitching; never used to choose nested assignment",
                endpoints="all-low/all-high original gains support and macro exact to 1e-12; masks1/25 checked against original hierarchy",
                feasible_masks=feasible,
                feasible_pairs={
                    arm: [
                        [s, o] for s in feasible[arm]["S"] for o in feasible[arm]["O"]
                    ]
                    for arm in feasible
                },
                branch_records={
                    arm: {b: tables[arm, b] for b in "SO"}
                    for arm in ["legacy", "transient"]
                },
            ),
        )
        with (out / "all_fixed_role_information.csv").open("w") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "source",
                    "branch",
                    "mask",
                    "gain_theta",
                    "gain_alpha",
                    "gain_RMS",
                    "positive_theta_people",
                    "positive_alpha_people",
                    "qualified_mass",
                    "passed",
                ]
            )
            for (arm, branch), records in tables.items():
                for mask, record in records.items():
                    writer.writerow(
                        [
                            arm,
                            branch,
                            mask,
                            *record["gain"].values(),
                            *list(record["positive_subject_fraction"].values())[:2],
                            record["qualified_mass"],
                            record["passed"],
                        ]
                    )
        codes_by_fold = {}
        for family in ["high", "semantic"]:
            for arm, fold, codes in fit_codes(
                source, family, cases, bank, names, folds, hard, run
            ):
                if arm == "transient":
                    codes_by_fold[family, fold] = codes
        STATE["role_mix_codes"] = codes_by_fold
        # Actual categorical fitting must agree at both extremes and a mixed namespace.
        proof = []
        for branch, mask in [("S", 1), ("O", 25)]:
            high, low = codes_by_fold["high", 0], codes_by_fold["semantic", 0]
            mixed = {
                b: mix_by_role(high[b], low[b], bank["fixed_band"], mask) for b in "SO"
            }
            take, error, mass, _, _ = raw_readout(
                bank, mixed, 0, branch, 8192, backgrounds
            )
            expected = mix_by_role(
                cache["high", "transient"][branch + "_error"],
                cache["semantic", "transient"][branch + "_error"],
                bank["fixed_band"],
                mask,
            )[take]
            np.testing.assert_allclose(
                error, expected, rtol=1e-12, atol=1e-14, equal_nan=True
            )
            expected_mass = mix_by_role(
                cache["high", "transient"][branch + "_mass"],
                cache["semantic", "transient"][branch + "_mass"],
                bank["fixed_band"],
                mask,
            )[take]
            np.testing.assert_array_equal(mass, expected_mass)
            proof.append(
                dict(
                    branch=branch,
                    mask=mask,
                    held_fold=0,
                    actual_mixed_conditional_table_and_support="matches role stitching",
                )
            )
        write(out / "categorical_stitching_proof.json", proof)
        with ProcessPoolExecutor(
            max_workers=cfg["resources"]["workers"],
            mp_context=multiprocessing.get_context("fork"),
        ) as pool:
            nested = list(pool.map(role_mix_inner_worker, range(4)))
        assembled, nested_information = {}, {}
        for branch in "SO":
            error = np.full((len(bank["examples"]), 2, 3), np.nan)
            mass = np.full(len(error), np.nan)
            for record in nested:
                held = bank["fold"] == record["outer_fold"]
                mask = record["selected_masks"][branch]
                high, low = cache["high", "transient"], cache["semantic", "transient"]
                error[held] = mix_by_role(
                    high[branch + "_error"],
                    low[branch + "_error"],
                    bank["fixed_band"],
                    mask,
                )[held]
                mass[held] = mix_by_role(
                    high[branch + "_mass"],
                    low[branch + "_mass"],
                    bank["fixed_band"],
                    mask,
                )[held]
            assembled[branch + "_error"], assembled[branch + "_mass"] = error, mass
            nested_information[branch] = RoleInformationPanel(
                error, error, mass, mass, bank, manifest["rows"]
            ).evaluate(0)
        np.savez_compressed(
            out / "nested_selected_readout_errors.npz",
            examples=bank["examples"],
            **assembled,
        )
        secondary = [
            pair
            for pair in cfg["selection"]["structural_secondary_masks"]
            if all(
                tables["transient", b][mask]["passed"] for b, mask in zip("SO", pair)
            )
        ]
        write(
            out / "role_mix_selection.json",
            dict(
                outer=[
                    {
                        k: r[k]
                        for k in [
                            "outer_fold",
                            "selected_masks",
                            "no_admissible_branch",
                        ]
                    }
                    for r in nested
                ],
                nested_outer_information=nested_information,
                outer_fit_selection_supported=not any(
                    r["no_admissible_branch"] for r in nested
                ),
                structural_secondary_pairs=secondary,
                priority=cfg["selection"]["fixed_priority_high_roles"],
                descriptor="outer-fold selection excludes every outer person; source_f and fixed historical books exposure stated in config",
            ),
        )
        write(
            out / "information_summary.json",
            dict(
                status="completed_pattern_stage_waveform_pending",
                feasible_masks=feasible,
                feasible_pair_counts={
                    arm: len(v["S"]) * len(v["O"]) for arm, v in feasible.items()
                },
                nested_selected_masks=[r["selected_masks"] for r in nested],
                nested_outer_information=nested_information,
                outer_fit_selection_supported=not any(
                    r["no_admissible_branch"] for r in nested
                ),
                structural_secondary_pairs=secondary,
                CPU_seconds=cpu_budget(cfg),
                validation_reads=0,
                confirmation_reads=0,
                test_reads=0,
            ),
        )


def predictive_fold(job):
    from src.tokenizers.predictive_word_order import (
        fit_site,
        encode_words,
        decode_words,
        previous_states,
        word_stats,
    )

    fold, codes = job
    bank = STATE["order_bank"]
    out = STATE["order_out"] / f"fold_{fold}"
    out.mkdir(exist_ok=True)
    fit = bank["fold"] != fold
    held = ~fit
    valid = bank["valid"]
    backgrounds = bank["background"]
    sites = backgrounds[:, 0] // 4
    n_background = int(STATE["order_backgrounds"])
    size = 8192
    arms = ("static", "rank", "semantic")
    output = {}
    network = {}
    sent_arrays = {"examples": bank["examples"][held]}
    for b in "SO":
        words = codes[b]
        fields = bank[b + "_attributes"][:, :, :2]
        trained = valid & fit[:, None]
        samples = fields[trained]
        groups = backgrounds[trained]
        group_counts, group_totals = word_stats(groups, samples, n_background)
        fallback = samples.mean(0)
        group_mean = np.where(
            group_counts[:, None] > 0, group_totals / np.maximum(1, group_counts[:, None]), fallback
        )
        properties = np.log1p(fields / np.maximum(1e-14, group_mean[backgrounds]))
        transformed = {arm: np.zeros(words.shape, np.uint16) for arm in arms}
        context = np.zeros(words.shape, np.uint8)
        states = []
        all_tables = {arm: [] for arm in arms}
        role_priors = {}
        for role in range(5):
            take = trained & (bank["fixed_band"][:, None] == role)
            count, totals = word_stats(words[take], properties[take], size)
            mean = (totals + 8 * properties[take].mean(0)) / (count[:, None] + 8)
            role_priors[role] = count, mean
        for site in range(n_background // 4):
            at_site = sites == site
            row = at_site & fit
            role = int(bank["fixed_band"][at_site][0])
            assert np.all(bank["fixed_band"][at_site] == role)
            state, tables = fit_site(
                words[row],
                valid[row],
                properties[row],
                STATE["order_energy"][b, role],
                *role_priors[role],
            )
            states.append(state)
            context[at_site] = previous_states(words[at_site], state)
            for arm, table in tables.items():
                all_tables[arm].append(table)
                transformed[arm][at_site] = encode_words(words[at_site], state, table)
        network[b + "_state"] = np.stack(states)
        for arm in arms:
            table = np.stack(all_tables[arm])
            # Only the inverse dictionary is needed on the decoding side.
            network[b + "_" + arm] = np.argsort(table, axis=-1).astype(np.uint16)
            sent_arrays[b + "_" + arm] = transformed[arm][held]
        support = {}
        for arm in (*arms, "semantic_context"):
            new_bank = bank
            sent = transformed["semantic" if arm == "semantic_context" else arm]
            if arm == "semantic_context":
                new_bank = dict(bank, background=5 * backgrounds + context)
            idx, error, mass, _, coverage = raw_readout(
                new_bank,
                {b: sent},
                fold,
                b,
                size,
                n_background * (5 if arm == "semantic_context" else 1),
                include_usage=False,
            )
            assert np.array_equal(idx, np.flatnonzero(held))
            output[arm + b + "error"] = error
            output[arm + b + "support"] = mass
            support[arm] = coverage
            counts = sent + bank["fixed_band"][:, None].astype(np.int32) * size
            output[arm + b + "fit_count"] = np.bincount(counts[trained], minlength=5 * size)
            output[arm + b + "held_count"] = np.bincount(
                counts[valid & held[:, None]], minlength=5 * size
            )
        write(out / (b + "_support.json"), support)
    model = out / "inverse_tables.npz"
    # Frequent ordering maps compress poorly. Uncompressed uint16 gives a fixed rate and fast audit.
    np.savez(model, **network)
    np.savez_compressed(out / "sent.npz", **sent_arrays)
    np.savez_compressed(out / "metrics.npz", held=np.flatnonzero(held), **output)
    held_sites = sites[held]
    restored = {}
    with np.load(model) as tables, np.load(out / "sent.npz") as transport:
        for b in "SO":
            states = tables[b + "_state"]
            for arm in arms:
                inverse = tables[b + "_" + arm]
                sent = transport[b + "_" + arm]
                recovered = np.zeros_like(sent)
                for site in np.unique(held_sites):
                    idx = held_sites == site
                    recovered[idx] = decode_words(sent[idx], states[site], inverse[site])
                np.testing.assert_array_equal(recovered, codes[b][held])
                if arm == "semantic":
                    restored[b] = recovered
    np.savez_compressed(out / "restored_high.npz", **restored, examples=bank["examples"][held])
    write(
        out / "replay.json",
        dict(
            original_words_per_branch=int(restored["S"].size),
            all_three_transmitted_forms_reversed_exactly=True,
            model_sha256=sha(model),
            fit_subjects=sorted(set(bank["subject"][fit])),
            held_subjects=sorted(set(bank["subject"][held])),
            CPU_seconds=time.process_time(),
        ),
    )
    return fold


def shortlist_fold(job):
    from src.tokenizers.predictive_word_order import compact_words, decode_words
    from src.tokenizers.spatial_waveform_dictionary import load_model
    from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics
    from evaluate_joint_tokenizer import input_scales

    fold, codes = job
    bank, population = STATE["order_bank"], STATE["order_population"]
    out = STATE["order_out"] / f"fold_{fold}"
    out.mkdir(exist_ok=True)
    source = STATE["order_previous"] / f"fold_{fold}"
    original = STATE["order_wave"]
    rows, scales = STATE["order_rows"], STATE["order_scales"]
    words_per_site = bank["background"][:, 0] // 4
    held = bank["fold"] == fold
    arms = ("static", "rank", "semantic")
    tables_out, sent_out, audit_out = {}, {}, {}
    counts = {}
    with np.load(source / "inverse_tables.npz") as z:
        for b in "SO":
            states = z[b + "_state"]
            tables_out[b + "_state"] = states
            for arm in arms:
                representatives = z[b + "_" + arm][:, :, :512].copy()
                tables_out[b + "_" + arm] = representatives
                sent = np.zeros(codes[b].shape, np.uint16)
                restored = np.zeros_like(sent)
                for site in range(len(states)):
                    take = words_per_site == site
                    role = int(bank["fixed_band"][take][0])
                    encoded, recons, _ = compact_words(
                        codes[b][take],
                        states[site],
                        representatives[site],
                        STATE["order_atoms"][b, role],
                    )
                    sent[take], restored[take] = encoded, recons
                tag = arm + "_" + b
                sent_out[tag] = sent[held]
                ok = bank["valid"] & held[:, None]
                counts[tag + "_unchanged_fraction"] = float((restored[ok] == codes[b][ok]).mean())
                _, error, mass, _, count = raw_readout(
                    bank, {b: sent}, fold, b, 512, STATE["order_backgrounds"], include_usage=False
                )
                audit_out[tag + "_error"] = error
                audit_out[tag + "_mass"] = mass
                counts[tag] = count
                expanded = sent + 512 * bank["fixed_band"][:, None].astype(np.int32)
                for key, selected in [("fit", ~held), ("held", held)]:
                    audit_out[tag + "_" + key + "_count"] = np.bincount(
                        expanded[bank["valid"] & selected[:, None]], minlength=2560
                    )
    np.savez_compressed(out / "model.npz", **tables_out)
    np.savez_compressed(out / "sent.npz", examples=bank["examples"][held], **sent_out)
    write(out / "information_support.json", counts)
    del sent_out, tables_out
    idx = np.flatnonzero(held)
    local_idx = np.full(len(held), -1, dtype=np.int32)
    local_idx[idx] = np.arange(len(idx))
    reconstructed = {}
    with np.load(out / "model.npz") as model, np.load(out / "sent.npz") as sentfile:
        np.testing.assert_array_equal(sentfile["examples"], bank["examples"][idx])
        for arm in arms:
            reconstructed[arm] = {}
            for b in "SO":
                states, tables = model[b + "_state"], model[b + "_" + arm]
                sent = sentfile[arm + "_" + b]
                raw = np.zeros_like(sent)
                for site in range(len(states)):
                    take = words_per_site[held] == site
                    raw[take] = decode_words(sent[take], states[site], tables[site])
                reconstructed[arm][b] = raw
    metrics = {arm: {} for arm in arms}
    gallery = {}
    cache = OrderedDict()
    for ds in population["native_positions"]:
        decoder = load_model(original / "held_decoder_manifests" / f"{ds}_fold{fold}.npz")
        for c in population["cases"]:
            if c["dataset"] != ds or c["fold"] != fold:
                continue
            if c["subject"] in decoder.estimator.metadata["fit_subjects"]:
                raise ValueError("changed outer fold")
            which = np.arange(c["first"], c["first"] + c["count"])
            ex = bank["examples"][which]
            local = local_idx[which]
            sites = np.asarray(c["native"])
            ordinal = sites.argsort()
            pred = {}
            for arm in arms:
                original_ids = {b: reconstructed[arm][b][local[ordinal]][None] for b in "SO"}
                waves = decoder.decode_ids(original_ids, ds)
                pred[arm] = {b: v[0, sites] for b, v in waves.items()}
            batch = base_tokenizer_batch(rows, ex, scales, array_cache=cache, cache_limit=8)
            for arm in arms:
                result = joint_reconstruction_metrics(
                    pred[arm], batch, 1, input_scales(rows, ex, scales)
                )
                if result["support"]["nonfinite_prediction_points"].any():
                    raise ValueError("nonfinite voltage")
                for key, v in result["metrics"].items():
                    if key not in metrics[arm]:
                        metrics[arm][key] = np.empty((len(idx), *v.shape[1:]), dtype=v.dtype)
                    metrics[arm][key][local] = v
            for j, value in enumerate(ex):
                pos = STATE["order_example_index"].get(tuple(value))
                if pos is not None:
                    for arm in arms:
                        for b, v in pred[arm].items():
                            gallery[f"{pos}_{arm}_{b}"] = v[j]
    for arm in arms:
        np.savez_compressed(
            out / (arm + "_wave_metrics.npz"), examples=bank["examples"][idx], **metrics[arm]
        )
    np.savez_compressed(out / "gallery.npz", **gallery)
    np.savez_compressed(out / "word_metrics.npz", held=idx, **audit_out)
    write(
        out / "replay.json",
        dict(
            observation_and_teacher_at_decode=False,
            read_disk_compact_IDs=True,
            ID_count=2 * 15 * len(idx),
            scoring_positions=len(idx),
            process_CPU_seconds=time.process_time(),
        ),
    )
    return fold


def summarize_shortlists(out, bank, rows, original):
    ref = json.loads((original / "summary.json").read_text())["arms"]["high_transient_source"][
        "metrics"
    ]
    summaries = {}
    for arm in ("static", "rank", "semantic"):
        metrics = {}
        pattern = {}
        for b in "SO":
            errors = np.empty((len(bank["examples"]), 2, 3))
            qualified, patches = 0, 0
            for fold in range(4):
                folder = out / f"fold_{fold}"
                with np.load(folder / "word_metrics.npz") as z:
                    index = z["held"]
                    errors[index] = z[arm + "_" + b + "_error"]
                count = json.loads((folder / "information_support.json").read_text())[arm + "_" + b]
                qualified += count["qualified_held_patches"]
                patches += count["complete_held_patches"]
                if b != "S":
                    continue
                with np.load(folder / (arm + "_wave_metrics.npz")) as z:
                    np.testing.assert_array_equal(bank["examples"][index], z["examples"])
                    for key in z.files:
                        if key == "examples":
                            continue
                        v = z[key]
                        if key not in metrics:
                            metrics[key] = np.empty((len(errors), *v.shape[1:]), dtype=v.dtype)
                        metrics[key][index] = v
            hierarchy = base_training_aggregate({"nmse": errors}, bank["examples"], rows)["metrics"][
                "nmse"
            ]
            macro = np.asarray(hierarchy["macro"])
            gain = 1 - macro[1] / macro[0]
            people = np.sum(
                [
                    np.asarray(d["value"])[1] < np.asarray(d["value"])[0]
                    for d in hierarchy["by_subject"]
                ],
                axis=0,
            )
            pattern[b] = dict(
                gain=gain,
                positive_people=people,
                support=qualified / patches,
                hierarchy=hierarchy,
                passed=bool(
                    min(gain[:2]) > 0
                    and max(gain) >= 0.05
                    and min(people[:2]) >= 30
                    and qualified / patches >= 0.90
                ),
            )
        waveform = base_training_aggregate(metrics, bank["examples"], rows)
        m = waveform["metrics"]
        nm = np.ravel(m["nmse"]["macro"])
        checks = dict(
            total=nm[0] <= 1.05 * np.ravel(ref["nmse"]["macro"])[0],
            components=np.all(nm[1:] <= 1.1 * np.ravel(ref["nmse"]["macro"])[1:]),
            datasets=all(
                np.all(np.asarray(v) <= 1.1 * np.asarray(ref["nmse"]["by_dataset"][d]))
                for d, v in m["nmse"]["by_dataset"].items()
            ),
            RMS=np.all(
                (np.asarray(m["rms_gain"]["macro"]) >= 0.75)
                & (np.asarray(m["rms_gain"]["macro"]) <= 1.25)
            ),
            theta_alpha=all(
                np.all(abs(np.asarray(m[k]["macro"])) <= 3.0103)
                for k in ("band_4_8_db", "band_8_13_db", "band_4_8_abs_db", "band_8_13_abs_db")
            ),
            envelope=min(np.ravel(m["local_rms_correlation"]["macro"])) >= 0.5,
            correlation=min(np.ravel(m["correlation"]["macro"])) > 0,
        )
        summaries[arm] = dict(
            waveform=waveform,
            checks=checks,
            pattern=pattern,
            passed=all(checks.values()) and all(v["passed"] for v in pattern.values()),
        )
    write(
        out / "summary.json",
        dict(
            arms=summaries,
            positions=17040,
            native_arrays=400,
            subjects=60,
            transmitted_K=512,
            paid_pair_bits_per_site_second=9,
            passed=[k for k, v in summaries.items() if v["passed"]],
            validation_reads=0,
            confirmation_reads=0,
            test_reads=0,
        ),
    )


def run_predictive_order(args, cfg):
    hard_cfg = yaml.safe_load((ROOT / cfg["source"]["word_config"]).read_text())
    compact = cfg["schema"] == "conditional_waveform_shortlist_v1"
    expected_size = cfg["model"].get("K") == 512 if compact else cfg["model"].get("paid_K") == 8192
    if (
        cfg["data"]["allowed_splits"] != ["train"]
        or not expected_size
        or cfg["data"]["confirmation_access"] != "forbidden"
        or cfg["data"]["test_access"] != "forbidden"
    ):
        raise ValueError("original training data only")
    hard, run, manifest, scales, folds, expected_cases = load_inputs(hard_cfg)
    source = ROOT / hard_cfg["source"]["output"]
    population = json.loads((source / "population.json").read_text())
    if (population["cfg_identity"], population["scales_identity"]) != (
        manifest["cfg_identity"],
        manifest["scales_identity"],
    ):
        raise ValueError("changed input identity")
    cases = population["cases"]
    for c, expected in zip(cases, expected_cases, strict=True):
        if (
            c["crop"] != expected["crop"]
            or c["fold"] != expected["fold"]
            or sorted(c["native"]) != list(range(c["count"]))
        ):
            raise ValueError("require original whole training montage and fold")
    with np.load(source / "training_attributes.npz") as z:
        bank = {k: z[k] for k in z}
    for c in cases:
        sl = slice(c["first"], c["first"] + c["count"])
        crop = c["crop"]
        if not np.array_equal(
            bank["examples"][sl], [(crop["row"], crop["start_s"], p) for p in crop["positions"]]
        ):
            raise ValueError("training order changed")
    encoder = run / hard["families"]["high"]["encoder"]
    metadata = json.loads(encoder.read_text())
    book = encoder.parent / metadata["source"]
    if sha(book) != metadata["source_sha256"]:
        raise ValueError("waveform source changed")
    with np.load(book) as z:
        atoms = {(b, role): z[f"atoms_{b}_{role}"] for b in "SO" for role in range(5)}
        energy = {k: np.sqrt(np.mean(np.square(v, dtype=float), axis=1)) for k, v in atoms.items()}
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise FileExistsError("completed order evidence retained")
    with (out / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(
            out / "resolved_config.json",
            dict(
                config=cfg,
                bank_sha256=sha(source / "training_attributes.npz"),
                population_sha256=sha(source / "population.json"),
                atom_sha256=metadata["source_sha256"],
            ),
        )
        STATE.update(
            cfg=hard_cfg,
            order_bank=bank,
            order_out=out,
            order_energy=energy,
            order_backgrounds=4 * len(population["background_sites"]),
        )
        worker = predictive_fold
        if compact:
            prev = run / cfg["source"]["order_run"]
            wave = run / cfg["source"]["waveform_run"]
            if not json.loads((prev / "summary.json").read_text())[
                "original_high_recovered_every_held_integer"
            ]:
                raise ValueError("invalid order model ancestor")
            gallery = json.loads((wave / "training_examples/waveform_examples.json").read_text())[
                "examples"
            ]
            lookup = {(v["row"], v["start_s"], v["position"]): i for i, v in enumerate(gallery)}
            STATE.update(
                order_atoms=atoms,
                order_population=population,
                order_previous=prev,
                order_wave=wave,
                order_rows=manifest["rows"],
                order_scales=scales,
                order_example_index=lookup,
            )
            worker = shortlist_fold
        with ProcessPoolExecutor(
            max_workers=4, mp_context=multiprocessing.get_context("fork")
        ) as pool:
            futures = [
                pool.submit(worker, (fold, codes))
                for arm, fold, codes in fit_codes(
                    source, "high", cases, bank, population["native_positions"], folds, hard, run
                )
                if arm == "transient"
            ]
            for future in futures:
                future.result()
        if compact:
            return summarize_shortlists(out, bank, manifest["rows"], wave)
        prior = json.loads((source / "high_summary.json").read_text())["result"]["arms"]["transient"]
        results = {}
        for arm in ("static", "rank", "semantic", "semantic_context"):
            results[arm] = {}
            for b in "SO":
                errors = np.zeros((len(bank["fold"]), 2, 3))
                qualified = 0
                patches = 0
                occupancy = []
                for fold in range(4):
                    directory = out / f"fold_{fold}"
                    with np.load(directory / "metrics.npz") as z:
                        errors[z["held"]] = z[arm + b + "error"]
                        counts = z[arm + b + "fit_count"].reshape(5, 8192)
                        occupancy.append(np.count_nonzero(counts[:, 1:], axis=1))
                    coverage = json.loads((directory / (b + "_support.json")).read_text())[arm]
                    patches += coverage["complete_held_patches"]
                    qualified += coverage["qualified_held_patches"]
                scores = base_training_aggregate(
                    {"nmse": errors}, bank["examples"], manifest["rows"]
                )["metrics"]["nmse"]
                macro = np.asarray(scores["macro"])
                gain = 1 - macro[1] / macro[0]
                people = np.sum(
                    [
                        np.asarray(row["value"])[1] < np.asarray(row["value"])[0]
                        for row in scores["by_subject"]
                    ],
                    axis=0,
                )
                if arm == "static":
                    np.testing.assert_allclose(
                        gain, list(prior[b]["gain"].values()), atol=1e-12, rtol=1e-12
                    )
                results[arm][b] = dict(
                    gain=gain,
                    positive_people=people,
                    support=qualified / patches,
                    complete_patches=patches,
                    occupied_per_fold_role=occupancy,
                    hierarchy=scores,
                    passed=bool(
                        min(gain[:2]) > 0
                        and max(gain) >= 0.05
                        and min(people[:2]) >= 30
                        and qualified / patches >= 0.9
                    ),
                )
        write(
            out / "summary.json",
            dict(
                status="training_information_scored",
                arms=results,
                input_arm="high_transient_source",
                training_people=60,
                training_crops=400,
                positions=17040,
                original_high_recovered_every_held_integer=True,
                validation_reads=0,
                confirmation_reads=0,
                selected=all(
                    results["semantic"][b]["passed"] and results["semantic_context"][b]["passed"]
                    for b in "SO"
                ),
                parent_waveform_visual="previous difficult-event failures retained by exactly lossless conversion",
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--role-mix", action="store_true")
    args = parser.parse_args()
    if args.role_mix:
        run_role_mix_information(args)
        return
    cfg = yaml.safe_load(args.config.read_text())
    if cfg["schema"] in ("predictive_word_order_v1", "conditional_waveform_shortlist_v1"):
        return run_predictive_order(args, cfg)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if any(
        cfg["data"][k] != "forbidden"
        for k in ("validation_access", "confirmation_access", "test_access")
    ):
        raise ValueError("training-only local-information scope")
    hard, run, manifest, scales, folds, cases = load_inputs(cfg)
    hard_run = ROOT / cfg["source"]["hard_run"]
    # Root's producer must finish; incomplete files never trigger a second producer.
    root_summary = json.loads((hard_run / "summary.json").read_text())
    selection = selected_scope(root_summary, cfg)
    selection["waveform_summary_sha256"] = sha(hard_run / "summary.json")
    if (out / "selection.json").exists():
        if json.loads((out / "selection.json").read_text()) != selection:
            raise ValueError("frozen family selection changed")
    else:
        write(out / "selection.json", selection)
    STATE.update(
        cfg=cfg,
        hard=hard,
        run=run,
        rows=manifest["rows"],
        scales=scales,
        folds=folds,
        cases=cases,
        out=out,
    )
    with threadpool_limits(cfg["resources"]["BLAS_threads"]):
        if args.review or args.render:
            population = json.loads((out / "population.json").read_text())
            cases = population["cases"]
            names = population["native_positions"]
            with np.load(out / "training_attributes.npz", allow_pickle=False) as saved:
                bank = {k: saved[k].copy() for k in saved.files}
            backgrounds = len(population["background_sites"]) * 4
            STATE["cases"] = cases
            if args.render:
                review_input_identities(cfg, hard, run, population, out)
                render_and_stratify(
                    json.loads((out / "summary.json").read_text()), bank, manifest["rows"], out, hard
                )
                return
            repeated = summarize(
                cfg,
                hard,
                run,
                manifest,
                folds,
                cases,
                bank,
                names,
                backgrounds,
                selection["families"],
                out,
                independent=True,
            )
            previous = json.loads((out / "summary.json").read_text())["families"]
            if json.loads(json.dumps(repeated, default=lambda a: a.tolist())) != previous:
                raise AssertionError("independent hierarchy or support changed")
            background_check = background_roundoff_review(
                bank, manifest["rows"], selection["families"], backgrounds, out
            )
            write(
                out / "independent_review.json",
                dict(
                    status="all_errors_and_support_exact",
                    families=selection["families"],
                    physical_source_fold_identity="checked_for_every_fit_integer_block",
                    CPU_seconds=cpu_budget(cfg, review=True),
                    background_bitwise_check={
                        b: {f: v["bitwise_equal"] for f, v in records.items()}
                        for b, records in background_check.items()
                    },
                    original_background_definition_and_gates_preserved=True,
                    additional_bitwise_background_failure_retained="review.log and review_attempt001.json",
                    validation_reads=0,
                    confirmation_reads=0,
                    test_reads=0,
                ),
            )
            return
        if (out / "summary.json").exists():
            raise FileExistsError("retained completed actual-word audit")
        if (out / "resolved_config.json").exists() and json.loads(
            (out / "resolved_config.json").read_text()
        ) != cfg:
            raise ValueError("retained frozen readout configuration changed")
        write(out / "resolved_config.json", cfg)
        bank, names, backgrounds = prepare(cfg, hard, run, manifest, scales, cases, out)
        STATE["cases"] = cases
        complete = []
        result = {}
        for family in selection["families"]:
            tasks = [(family, dataset, fold) for dataset in sorted(names) for fold in range(4)]
            with ProcessPoolExecutor(
                max_workers=cfg["resources"]["workers"],
                mp_context=multiprocessing.get_context("fork"),
            ) as pool:
                pending = {pool.submit(fit_worker, task) for task in tasks}
                current = []
                while pending:
                    cpu_budget(cfg)
                    done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                    current.extend(f.result() for f in done)
                    write(
                        out / "progress.json",
                        dict(
                            stage="same_source_fit_integer_prototypes",
                            family=family,
                            complete_tasks=len(current),
                            planned=len(tasks),
                            CPU_seconds=cpu_budget(cfg),
                        ),
                    )
            complete.extend(current)
            result.update(
                summarize(
                    cfg, hard, run, manifest, folds, cases, bank, names, backgrounds, [family], out
                )
            )
            write(
                out / f"{family}_summary.json",
                dict(
                    family=family,
                    role=selection["family_roles"][family],
                    result=result[family],
                    CPU_seconds=cpu_budget(cfg),
                    original_waveform_eligible=selection["original_waveform_eligibility"][family],
                ),
            )
        write(
            out / "summary.json",
            dict(
                status="completed_same_source_raw_word_information",
                selection=selection,
                families=result,
                scoring_positions=17040,
                crops=400,
                independent_subjects=60,
                worker_results=complete,
                CPU_seconds=cpu_budget(cfg),
                validation_reads=0,
                confirmation_reads=0,
                test_reads=0,
                prototypes="source_f encodes fit people; source_f OOF root integers score held people",
                exposure="frozen legacy dictionaries, PCA/Wiener and G0 scales previously saw all train; transient source and word readout exclude held people",
            ),
        )


if __name__ == "__main__":
    main()
