"""Namespace-correct support and local-attribute audits for spatial EEG tokens."""
from __future__ import annotations
import argparse
from collections import OrderedDict, defaultdict
import csv
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing
import os
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.dual_component_tokenizer import base_tokenizer_training_data, base_tokenizer_batch, base_tokenizer_joint_eeg, base_tokenizer_joint_eeg_context
from src.tokenizers.spatial_waveform_dictionary import BANDS, load_model
from src.tokenizers.spatial_analysis_synthesis import load_encoder
from threadpoolctl import threadpool_limits

AS_STATE = {}


def is_as(model):
    return model.metadata.get("encoder_rule") == "observation_only_analysis_by_synthesis_v1"
from audit_joint_tokenizer_patterns import write_json


def source(args):
    cfg = yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    panel_metadata = json.loads(args.panels.with_name('panel.json').read_text())
    model = load_encoder(args.model) if args.model.suffix == '.json' else load_model(args.model)
    if (model.metadata['input_identity'] != manifest['cfg_identity']
            or model.metadata['scales_identity'] != manifest['scales_identity']
            or panel_metadata['cfg_identity'] != manifest['cfg_identity']
            or panel_metadata['scales_identity'] != manifest['scales_identity']
            or panel_metadata['panel_sha256'] != hashlib.sha256(args.panels.read_bytes()).hexdigest()):
        raise ValueError('frozen model/G0/panel identities required before signal arrays')
    with np.load(args.panels) as stored:
        panels = {split: stored[key].copy() for split, key in [('train','eeg_pattern_train'), ('validation','eeg_screen')]}
    rows = manifest['rows']
    for split, examples in panels.items():
        if any(rows[i]['modality'] != 'eeg' or rows[i]['split'] != split
               or rows[i]['subject'] == panel_metadata['confirmation_subjects'][rows[i]['dataset']]
               for i, _, _ in examples):
            raise ValueError('fixed pattern split before signal arrays')
    return rows, scales, model, panels, panel_metadata


def local_ids(model, rows, examples, batch, scales=None):
    if model.metadata.get('global_analysis') or is_as(model):
        if scales is None:
            raise ValueError('global spatial analysis requires train coordinates and complete native crops')
        result={b:np.zeros((len(examples),15),np.int32) for b in 'SO'}
        bands=np.empty(len(examples),np.int64)
        cache=OrderedDict();encoded={}
        for n,(index,start,p) in enumerate(examples):
            row=rows[index];dataset=row['dataset'];names=model.metadata['positions'][dataset]
            key=int(index),int(start)
            if key not in encoded:
                positions=next(v['positions'] for v in row['inputs']['score'] if v['start_s']==start)
                fields,observed,_=base_tokenizer_joint_eeg(rows,dict(row=int(index),start_s=int(start),positions=positions),
                    scales,names,cache,allowed_splits=(row['split'],))
                encoded[key]=model.encode(fields['x'].T.reshape(1,len(names),15,400),dataset,observed[None])
            native=next(j for j,name in enumerate(names) if name.upper()==row['positions'][p].upper())
            bands[n]=native%5
            for b in 'SO':result[b][n]=encoded[key][b][0,native]
        return result,bands
    result = {branch: np.zeros((len(examples),15),dtype=np.int32) for branch in 'SO'}
    bands = np.empty(len(examples),dtype=np.int64)
    for dataset in sorted({rows[i]['dataset'] for i,_,_ in examples}):
        selection = np.asarray([rows[i]['dataset'] == dataset for i,_,_ in examples])
        native = {name.upper(): n for n,name in enumerate(model.metadata['positions'][dataset])}
        indices = np.asarray([native[rows[i]['positions'][p].upper()] for i,_,p in examples[selection]])
        codes = model.encode_positions(batch['x'][selection,None], dataset, indices[:,None])
        bands[selection] = indices % len(BANDS)
        for branch in 'SO':
            result[branch][selection] = codes[branch][:,0]
    return result, bands


def equivalence(model, rows, scales, examples):
    checked = []
    cache = OrderedDict()
    for dataset in sorted({rows[i]['dataset'] for i,_,_ in examples}):
        example = next(v for v in examples if rows[v[0]]['dataset'] == dataset)
        i,start,p = map(int,example)
        row = rows[i]
        positions = next(item['positions'] for item in row['inputs']['score'] if item['start_s'] == start)
        names = model.metadata['positions'][dataset]
        fields, observed, _ = base_tokenizer_joint_eeg(rows, dict(row=i,start_s=start,positions=positions), scales,
                                                      names, cache, allowed_splits=('train',))
        values = fields['x'].T.reshape(1,len(names),15,400)
        full = model.encode(values,dataset,observed[None])
        batch = base_tokenizer_batch(rows,np.asarray([example]),scales,array_cache=cache,cache_limit=8)
        local,_ = local_ids(model,rows,np.asarray([example]),batch,scales)
        native = {name.upper(): n for n,name in enumerate(names)}[row['positions'][p].upper()]
        for branch in 'SO':
            np.testing.assert_array_equal(local[branch][0],full[branch][0,native])
        checked.append(dict(dataset=dataset,row=i,start_s=start,position=p,native_index=native,exact_integer_match=True))
    return checked



def native_screen_ids(args, rows, model, examples):
    """Extract original AS integers in the exact requested position-window order."""
    archive = args.model.parent/'integer_ids.npz'
    metadata = json.loads(archive.with_name('integer_metadata.json').read_text())
    expected = {(int(i), int(start)) for i, start, _ in examples}
    lookup = {(int(item['row']), int(item['start_s'])): item for item in metadata}
    if len(lookup) != len(metadata) or set(lookup) != expected:
        raise ValueError('saved AS integers must exactly cover frozen screen crops before extraction')
    result = {b: np.empty((len(examples), 15), np.int32) for b in 'SO'}
    band = np.empty(len(examples), np.int64)
    with np.load(archive, allow_pickle=False) as stored:
        for n, (i, start, p) in enumerate(examples):
            item = lookup[int(i), int(start)]
            row = rows[i]
            names = model.metadata['positions'][row['dataset']]
            if (item['native_positions'] != names or item['dataset'] != row['dataset']
                    or item['subject'] != row['subject']):
                raise ValueError('saved AS integer native identity mismatch')
            native = next(j for j, name in enumerate(names) if name.upper() == row['positions'][p].upper())
            band[n] = native % 5
            for b in 'SO':
                value = stored[f'{b}_{item["crop"]}']
                if value.shape != (1, len(names), 15) or value.dtype.kind not in 'iu':
                    raise ValueError('saved AS full-native 15-integer interface mismatch')
                if value.min() < 0 or value.max() >= model.codebook_size:
                    raise ValueError('saved AS integer outside dictionary')
                result[b][n] = value[0, native]
    return dict(result, fixed_band=band), lookup


def as_chunk(payload):
    dataset, number, crops = payload
    model, rows, scales = (AS_STATE[key] for key in ('model', 'rows', 'scales'))
    target = AS_STATE['output']/f'{dataset}_{number:04d}.npz'
    if target.exists():
        with np.load(target, allow_pickle=False) as stored:
            if str(stored['signature']) != AS_STATE['signature']:
                raise ValueError('retained AS crop signature changed')
            np.testing.assert_array_equal(stored['crops'], crops)
        return dict(path=str(target), crops=len(crops), resumed=True)
    names = model.metadata['positions'][dataset]
    cache, codes, trace = OrderedDict(), {b: [] for b in 'SO'}, []
    began = time.monotonic()
    with threadpool_limits(AS_STATE['threads']):
        for i, start in crops:
            row = rows[i]
            positions = next(item['positions'] for item in row['inputs']['score'] if item['start_s'] == start)
            fields, observed, _ = base_tokenizer_joint_eeg(rows, dict(row=int(i), start_s=int(start), positions=positions),
                scales, names, cache, allowed_splits=('train',))
            assigned = model.encode(fields['x'].T.reshape(1, len(names), 15, 400), dataset, observed[None])
            trace.append(model.search_history[-1])
            for b in 'SO': codes[b].append(assigned[b][0])
    pending = target.with_suffix('.pending.npz')
    np.savez_compressed(pending, signature=np.asarray(AS_STATE['signature']), crops=np.asarray(crops),
        **{b: np.stack(value) for b, value in codes.items()})
    pending.replace(target)
    write_json(target.with_suffix('.json'), dict(status='completed', dataset=dataset, crops=len(crops),
        seconds=time.monotonic()-began, pid=os.getpid(), trace=trace))
    return dict(path=str(target), crops=len(crops), seconds=time.monotonic()-began)


def prepare_as_ids(args, rows, scales, model, panels, metadata):
    """Bounded complete-montage train search; frozen validation integers are reused."""
    output = args.output/'native_train_ids'
    output.mkdir(exist_ok=True)
    signature = json.dumps(dict(panel=metadata['panel_sha256'], encoder_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest()), sort_keys=True)
    AS_STATE.update(model=model, rows=rows, scales=scales, output=output, threads=args.threads, signature=signature)
    grouped = defaultdict(dict)
    for i, start, _ in panels['train']:
        grouped[rows[i]['dataset']][int(i), int(start)] = None
    jobs = [(dataset, n//16, np.asarray(crops[n:n+16]))
            for dataset, members in sorted(grouped.items())
            for crops in [list(members)] for n in range(0, len(crops), 16)]
    # Build the FFT dictionaries before fork; workers share these frozen pages.
    began = time.monotonic()
    with threadpool_limits(args.threads):
        for b in 'SO':
            for band in range(5): model._book(b, band)
    benchmark = []
    if not (args.output/'as_benchmark.json').exists():
        for dataset, _, crops in jobs:
            if any(item['dataset'] == dataset for item in benchmark): continue
            local = []
            cache = OrderedDict()
            for i, start in crops[:2]:
                row = rows[i]; names = model.metadata['positions'][dataset]
                positions = next(item['positions'] for item in row['inputs']['score'] if item['start_s'] == start)
                fields, observed, _ = base_tokenizer_joint_eeg(rows, dict(row=int(i), start_s=int(start), positions=positions),
                    scales, names, cache, allowed_splits=('train',))
                tick = time.monotonic()
                with threadpool_limits(args.threads):
                    model.encode(fields['x'].T.reshape(1, len(names), 15, 400), dataset, observed[None])
                local.append(time.monotonic()-tick)
            benchmark.append(dict(dataset=dataset, seconds_per_crop=float(np.mean(local)), crops=len(grouped[dataset])))
        expected = sum(v['seconds_per_crop']*v['crops'] for v in benchmark)/args.workers
        write_json(args.output/'as_benchmark.json', dict(status='completed', workers=args.workers,
            numerical_threads_per_worker=args.threads, frozen_fft_workers=model.settings['fft_workers'],
            estimated_parallel_wall_seconds=expected, measured=benchmark,
            fork_shared_fft_bytes=sum(v.nbytes for v in model.atom_spectra.values())))
    if args.benchmark_only: return None
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork')) as pool:
        iterator = iter(jobs)
        pending = {pool.submit(as_chunk, job) for job in [next(iterator, None) for _ in range(args.workers*2)] if job is not None}
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result(); completed += result['crops']
                write_json(args.output/'as_progress.json', dict(stage='train_integer_search', completed_crops=completed,
                    planned_crops=sum(len(v) for v in grouped.values()), elapsed_seconds=time.monotonic()-began))
                job = next(iterator, None)
                if job is not None: pending.add(pool.submit(as_chunk, job))
    train = {b: np.empty((len(panels['train']),15), np.int32) for b in 'SO'}
    band = np.empty(len(panels['train']), np.int64)
    local = {}
    for dataset, number, crops in jobs:
        with np.load(output/f'{dataset}_{number:04d}.npz', allow_pickle=False) as stored:
            for j, key in enumerate(crops):
                local[tuple(key)] = {b: stored[b][j] for b in 'SO'}
    for n, (i, start, p) in enumerate(panels['train']):
        names = model.metadata['positions'][rows[i]['dataset']]
        native = next(j for j, name in enumerate(names) if name.upper() == rows[i]['positions'][p].upper())
        band[n] = native % 5
        for b in 'SO': train[b][n] = local[int(i), int(start)][b][native]
    validation, lookup = native_screen_ids(args, rows, model, panels['validation'])
    probes = []
    with np.load(args.model.parent/'integer_ids.npz', allow_pickle=False) as saved:
        for dataset in sorted(grouped):
            (i, start), item = next((key, value) for key, value in lookup.items() if value['dataset'] == dataset)
            names = model.metadata['positions'][dataset]; row = rows[i]
            positions = next(v['positions'] for v in row['inputs']['score'] if v['start_s'] == start)
            fields, observed, _ = base_tokenizer_joint_eeg(rows, dict(row=i,start_s=start,positions=positions), scales,
                names, OrderedDict(), allowed_splits=('validation',))
            with threadpool_limits(args.threads):
                assigned = model.encode(fields['x'].T.reshape(1,len(names),15,400), dataset, observed[None])
            for b in 'SO': np.testing.assert_array_equal(assigned[b], saved[f'{b}_{item["crop"]}'])
            probes.append(dict(dataset=dataset,row=i,start_s=start,all_native_S_O_ids_exact=True))
    write_json(args.output/'as_integer_equivalence.json', dict(status='passed',
        screen_position_windows=len(panels['validation']), screen_crops=len(lookup),
        validation='every selected triple extracted exactly from original persisted native AS arrays; no new screen search except the four recorded probes',
        full_native_reencode_probes=probes, source_integer_archive=str(args.model.parent/'integer_ids.npz'),
        source_integer_sha256=hashlib.sha256((args.model.parent/'integer_ids.npz').read_bytes()).hexdigest(),
        train_position_windows=len(panels['train']), train_physical_crops=len(local),
        train_integer_source='complete observed X montage with original frozen AS settings, saved by physical crop before local selection',
        encoder_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest()))
    return dict(train=dict(train, fixed_band=band), validation=validation)


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    rows,scales,model,panels,panel_metadata = source(args)
    probes = equivalence(model,rows,scales,panels['train'])
    retained_codes = prepare_as_ids(args, rows, scales, model, panels, panel_metadata) if is_as(model) else None
    if args.benchmark_only: return
    size = model.codebook_size
    counts = {split:{branch:np.zeros((5,size),dtype=np.int64) for branch in 'SO'} for split in panels}
    encoded = {split:{branch:np.zeros((5,size),dtype=np.int64) for branch in 'SO'} for split in panels}
    per_subject = {branch:{} for branch in 'SO'}
    coverage = {}
    cache = OrderedDict()
    began = time.monotonic()
    for split,examples in panels.items():
        codes = {branch:[] for branch in 'SO'}
        band_ids = []
        complete = 0
        for start in range(0,len(examples),args.batch_size):
            take = examples[start:start+args.batch_size]
            batch = base_tokenizer_batch(rows,take,scales,array_cache=cache,cache_limit=64)
            if retained_codes is None:
                assigned,bands = local_ids(model,rows,take,batch,scales)
            else:
                assigned = {b: retained_codes[split][b][start:start+len(take)] for b in 'SO'}
                bands = retained_codes[split]['fixed_band'][start:start+len(take)]
            valid = batch['mask'].all(-1)
            complete += int(valid.sum())
            band_ids.append(bands)
            for branch in 'SO':
                codes[branch].append(assigned[branch])
                for band in range(5):
                    selected = bands == band
                    encoded[split][branch][band] += np.bincount(assigned[branch][selected].ravel(),minlength=size)
                    counts[split][branch][band] += np.bincount(assigned[branch][selected][valid[selected]],minlength=size)
                if split == 'train':
                    for n,(i,_,_) in enumerate(take):
                        subject = rows[i]['dataset'],rows[i]['subject']
                        value = per_subject[branch].setdefault(subject,np.zeros((5,size),dtype=np.int64))
                        value[bands[n]] += np.bincount(assigned[branch][n][valid[n]],minlength=size)
            write_json(args.output/'progress.json',dict(stage=split,inputs=start+len(take),planned=len(examples),
                                                       elapsed_seconds=time.monotonic()-began))
        packed = {branch:np.concatenate(value) for branch,value in codes.items()}
        np.savez_compressed(args.output/(split+'_IDs.npz'),examples=examples,fixed_band=np.concatenate(band_ids),**packed)
        coverage[split] = dict(contexts=len(examples),planned_patches=len(examples)*15,complete_patches=complete,
                               partial_or_missing_patches=len(examples)*15-complete)
    support = []
    for branch in 'SO':
        subject_count = np.sum([value>0 for value in per_subject[branch].values()],axis=0)
        qualified = (counts['train'][branch] >= 8) & (subject_count >= 2)
        numerator = int(counts['validation'][branch][qualified].sum())
        denominator = int(counts['validation'][branch].sum())
        for band in range(5):
            valid = counts['validation'][branch][band]
            train = counts['train'][branch][band]
            support.append(dict(branch=branch,fixed_band_index=band,fixed_band_hz=BANDS[band],
                actual_train_complete_patches=int(train.sum()),actual_train_occupied_codes=int(np.count_nonzero(train[1:])),
                actual_train_dead_nonmissing_codes=int(np.count_nonzero(train[1:]==0)),
                validation_complete_patches=int(valid.sum()),validation_occupied_codes=int(np.count_nonzero(valid[1:])),
                validation_mass_on_8patch_2subject_codes=float(valid[qualified[band]].sum()/max(valid.sum(),1)),
                branch_validation_mass_on_8patch_2subject_codes=numerator/max(denominator,1),
                validation_only_codes=int(np.sum((valid[1:]>0)&(train[1:]==0)))))
        with (args.output/(branch+'_code_support.csv')).open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['branch','fixed_band_index','ID','train_encoded_patches','train_complete_patches',
                             'train_subjects','validation_encoded_patches','validation_complete_patches','qualified_8patch_2subject'])
            for band in range(5):
                for code in range(size):
                    writer.writerow([branch,band,code,encoded['train'][branch][band,code],counts['train'][branch][band,code],
                        subject_count[band,code],encoded['validation'][branch][band,code],counts['validation'][branch][band,code],qualified[band,code]])
    write_json(args.output/'support_summary.json',dict(status='completed',stage='support_only',model=str(args.model.resolve()),
        model_metadata=model.metadata,panel_role='screen',panel_sha256=panel_metadata['panel_sha256'],
        namespace='(branch,fixed_band_index,integer_ID); ID0 is reserved missing and excluded from real-code occupancy',
        encoder_scope=('complete native X montage through frozen observation-only analysis-by-synthesis coordinate search before selecting local IDs'
                       if is_as(model) else 'complete native X montage through learned global analysis before selecting local IDs'
                       if model.metadata.get('global_analysis') else 'position-independent Wiener and band assignment; no spatial decoder called in this support screen'),
        full_native_encoder_equivalence=probes,coverage=coverage,support=support,
        elapsed_seconds=time.monotonic()-began,complete_spatial_reconstruction_audit='separate_required_evidence'))
    write_json(args.output/'progress.json',dict(stage='support_completed',elapsed_seconds=time.monotonic()-began))


def assess_attributes(args):
    """Score the preregistered single-ID primary attributes with band-local vocabularies."""
    from src.analysis.tokenizer_pattern_audit import SparsePatternAudit, attributes, feature_means
    from audit_joint_tokenizer_patterns import hierarchy
    rows, scales, model, panels, panel_metadata = source(args)
    retained = json.loads((args.output / 'support_summary.json').read_text())
    if retained['model_metadata'] != model.metadata or retained['panel_sha256'] != panel_metadata['panel_sha256']:
        raise ValueError('stored IDs use a different model or panel')
    sites = sorted({(row['dataset'], name) for row in rows if row['split'] == 'train' and row['modality'] == 'eeg'
                    for name in row['positions']})
    site_index = {key: n for n, key in enumerate(sites)}
    fields = {'power_4_8': (0, 1), 'power_8_13': (1, 2), 'rms_envelope': (2, 10)}
    width = 10
    size = model.codebook_size
    audits = {b: SparsePatternAudit(5*size, (len(sites)+1)*4, [], width, fields) for b in 'SO'}
    errors = defaultdict(list)
    readout_support = {b: dict(complete_validation_patches=0, ID_seen_in_train=0, background_ID_seen_in_train=0) for b in 'SO'}
    cache = OrderedDict()
    for split, examples in panels.items():
        with np.load(args.output / (split + '_IDs.npz')) as f:
            np.testing.assert_array_equal(f['examples'], examples)
            codes = {b: f[b].astype(np.int64) + size*f['fixed_band'][:, None] for b in 'SO'}
        for start in range(0, len(examples), args.batch_size):
            take = examples[start:start + args.batch_size]
            batch = base_tokenizer_batch(rows, take, scales, array_cache=cache, cache_limit=64)
            valid = batch['mask'].all(-1)
            loc = np.array([site_index.get((rows[i]['dataset'], rows[i]['positions'][p]),len(sites)) for i,_,p in take])
            bg = 4*loc[:,None] + np.searchsorted([.5,1.,2.], np.sqrt(np.mean(batch['x']**2, axis=-1)), side='right')
            for b, audit in audits.items():
                raw, slices = attributes(batch[b.lower()][valid], 'eeg')
                y = np.concatenate([raw[:,slice(*slices[k])] for k in fields], axis=1)
                ids = codes[b][start:start+len(take)][valid]
                if split == 'train':
                    audit.fit(ids,bg[valid],y)
                else:
                    background, _, joint = audit.read(ids,bg[valid])
                    keys = bg[valid]*audit.k+ids
                    readout_support[b]['complete_validation_patches'] += len(ids)
                    readout_support[b]['ID_seen_in_train'] += int((audit.local.count[ids] > 0).sum())
                    readout_support[b]['background_ID_seen_in_train'] += sum(audit.joint.count.get(int(k),0) > 0 for k in keys)
                    squared = np.stack([feature_means((v-y)**2, fields) for v in (background,joint)], axis=1)
                    expanded = np.zeros((*valid.shape, 2, len(fields)))
                    expanded[valid] = squared
                    means = expanded.sum(1)/np.maximum(valid.sum(1),1)[:,None,None]
                    means[~valid.any(1)] = np.nan
                    errors[b].append(means)
    result = {}
    with np.load(args.output/'validation_IDs.npz') as saved:
        validation_bands = saved['fixed_band'].copy()
    for b in 'SO':
        stats = hierarchy(np.concatenate(errors[b]),panels['validation'],rows)
        mse = np.asarray(stats['macro'])
        inc = 1 - mse[1]/mse[0]
        fractions = np.mean([np.asarray(v['value'])[1] < np.asarray(v['value'])[0] for v in stats['by_subject']],axis=0)
        band_stats = {}
        for band in range(5):
            take = validation_bands == band
            by_band = hierarchy(np.concatenate(errors[b])[take], panels['validation'][take], rows)
            value = np.asarray(by_band['macro'])
            band_stats[str(BANDS[band])] = dict(hierarchical_attribute_MSE=by_band, gain=dict(zip(fields,1-value[1]/value[0])))
        result[b] = dict(hierarchical_attribute_MSE=stats, gain=dict(zip(fields,inc)),
                         by_fixed_band=band_stats, validation_readout_support=readout_support[b],
                         subject_positive_fraction=dict(zip(fields,fractions)),
                         primary_numeric_pass=bool(min(inc[:2])>0 and max(inc)>=.05 and min(fractions[:2])>=.5))
    write_json(args.output/'attribute_summary.json',dict(status='completed',readouts=['background','background_plus_ID'],
        attributes=list(fields), namespace='branch/fixed_band/ID',branches=result,model=str(args.model),
        scope='single-ID attributes; full spatial sequence reconstruction and natural-context stability are separate checks'))
    np.savez_compressed(args.output/'attribute_errors.npz', examples=panels['validation'],
        fixed_band=validation_bands, **{b: np.concatenate(errors[b]) for b in 'SO'},
        coordinate=np.asarray('per position-window means over complete patches; [background, background+ID] x [theta,alpha,RMS]'))
    profiles={}
    for b,audit in audits.items():
        profiles[b+'_train_count']=audit.local.count.reshape(5,size)
        profiles[b+'_train_attribute_mean']=(audit.local.total/audit.local.count[:,None].clip(1)).reshape(5,size,width)
    np.savez_compressed(args.output/'training_attribute_profiles.npz',**profiles,
        attribute_order=np.asarray(['theta_power','alpha_power']+[f'quarter_second_RMS_{i}' for i in range(8)]),
        coordinate=np.asarray('G0_input_SD_power_squared_and_RMS; true_complete_train_target_patches_only'))


def natural_context_chunk(payload):
    dataset, number, jobs = payload
    state = AS_STATE
    target = state['output'] / f'{dataset}_{number:04d}.npz'
    if target.exists():
        with np.load(target) as stored:
            if str(stored['signature']) != state['signature']:
                raise ValueError('natural-context retained signature changed')
            np.testing.assert_array_equal(stored['jobs'], jobs)
        return str(target)
    names = state['model'].metadata['positions'][dataset]
    codes, records, flags = {b: [] for b in 'SO'}, [], defaultdict(list)
    with threadpool_limits(state['threads']):
        for i, start, direction in jobs:
            if time.process_time() > state['worker_CPU_seconds']:
                raise RuntimeError('natural-context worker CPU budget exhausted')
            row = state['rows'][i]
            supports = {v['start_s']: set(v['positions']) for v in row['inputs']['score']}
            positions = sorted(supports[int(start)] & supports[int(start + direction * 30)])
            crop = dict(row=int(i), start_s=int(start), positions=positions)
            original, old_observed, old_raw = base_tokenizer_joint_eeg_context(state['rows'], crop, state['scales'], names)
            shifted, new_observed, new_raw = base_tokenizer_joint_eeg_context(state['rows'], crop, state['scales'], names, int(direction))
            old_center = slice(2800, 3200)
            new_center = slice((7 - direction) * 400, (8 - direction) * 400)
            np.testing.assert_array_equal(old_raw['raw_normalized_x'][old_center], new_raw['raw_normalized_x'][new_center])
            np.testing.assert_array_equal(old_raw['observation_valid'][old_center], new_raw['observation_valid'][new_center])
            assigned = state['model'].encode(shifted.T.reshape(1, len(names), 15, 400), dataset, new_observed[None])
            for b in 'SO': codes[b].append(assigned[b][0])
            flags['original_observed'].append(old_observed)
            flags['shifted_observed'].append(new_observed)
            flags['center_observation_complete'].append(old_raw['observation_valid'][old_center].all(0))
            flags['model_input_center_equal'].append((original[old_center] == shifted[new_center]).all(0))
            records.append(state['model'].search_history[-1])
    pending = target.with_suffix('.pending.npz')
    np.savez_compressed(pending, signature=np.asarray(state['signature']), jobs=jobs,
        **{b: np.stack(values) for b, values in codes.items()},
        **{key: np.stack(values) for key, values in flags.items()})
    pending.replace(target)
    write_json(target.with_suffix('.json'), dict(status='completed', same_raw_X_and_observation_mask='exact every native center point',
        CPU_seconds=time.process_time(), search_trace=records))
    return str(target)


def natural_center_metrics(predictions, batch, factors):
    """Only the exact same physical 400-point piece; original G0 denominators."""
    mask = batch['mask'][:, 7].astype(bool)
    count = mask.sum(-1)
    def average(value):
        return np.divide(np.where(mask, value, 0.).sum(-1), count, out=np.full(len(mask), np.nan), where=count > 0)
    result = defaultdict(list)
    for branch, field in [('sum', 'x'), ('S', 's'), ('O', 'o')]:
        y, p = batch[field][:, 7], predictions[branch]
        error = p - y
        mse = average(error ** 2)
        variance = batch['v' + field][:, 0, 0]
        yc, pc = y - average(y)[:, None], p - average(p)[:, None]
        energy = average(yc ** 2) * average(pc ** 2)
        correlation = np.divide(average(yc * pc), np.sqrt(energy), out=np.full(len(mask), np.nan), where=energy > 1e-20)
        metrics = dict(nmse=mse / variance, nrmse=np.sqrt(mse / variance),
            rmse_native=np.sqrt(mse) * factors, bias_native=average(error) * factors,
            correlation=correlation, rms_gain=np.sqrt(average(p ** 2) / np.maximum(average(y ** 2), 1e-20)))
        for key, value in metrics.items(): result[key].append(value)
    return {key: np.stack(value, axis=1)[..., None] for key, value in result.items()}


def assess_natural_context(args):
    from src.analysis.dual_component_tokenizer import base_training_aggregate
    from src.analysis.tokenizer_pattern_audit import feature_means
    from audit_joint_tokenizer_patterns import hierarchy
    import fcntl
    cfg = yaml.safe_load(args.contract.read_text())
    if (cfg['schema'] != 'spatial_as_natural_context_v1' or cfg['data']['test'] != 'forbidden'
            or cfg['data']['confirmation'] != 'forbidden' or cfg['data']['reserved_four'] != 'forbidden'
            or args.model != (ROOT / cfg['source']['encoder']).resolve()
            or args.panels != (ROOT / cfg['source']['panels']).resolve()
            or args.workers != cfg['resources']['workers'] or args.threads != 1):
        raise ValueError('fixed original-AS natural-context contract required before arrays')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / 'summary.json').exists():
            raise FileExistsError('retained natural-context audit already completed')
        rows, scales, model, panels, panel_metadata = source(args)
        old_pattern = ROOT / cfg['source']['actual_pattern']
        if not is_as(model) or model.settings['fft_workers'] != 2:
            raise ValueError('original frozen two-FFT-worker AS required')
        old_summary = json.loads((old_pattern / 'support_summary.json').read_text())
        if old_summary['model_metadata'] != model.metadata or old_summary['panel_sha256'] != panel_metadata['panel_sha256']:
            raise ValueError('original AS pattern identity changed')
        examples = panels['validation']
        original_local, lookup = native_screen_ids(args, rows, model, examples)
        pairs, no_partner = [], []
        for n, (i, start, p) in enumerate(examples):
            support = {v['start_s'] for v in rows[i]['inputs']['score'] if p in v['positions']}
            direction = 1 if start + 30 in support else (-1 if start - 30 in support else 0)
            if direction: pairs.append((n, int(i), int(start), int(p), direction))
            else: no_partner.append(n)
        if not pairs:
            raise ValueError('no same-record natural adjacent contexts')
        grouped = defaultdict(dict)
        for _, i, start, _, direction in pairs: grouped[rows[i]['dataset']][i, start, direction] = None
        jobs = [(d, n // 8, np.asarray(list(members)[n:n + 8], np.int64))
            for d, members in sorted(grouped.items()) for n in range(0, len(members), 8)]
        source_paths = dict(encoder=args.model, decoder=ROOT / cfg['source']['decoder'],
            original_IDs=args.model.parent / 'integer_ids.npz', templates=old_pattern / 'training_attribute_profiles.npz',
            training_bank=ROOT / cfg['source']['training_bank'], panels=args.panels)
        hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in source_paths.items()}
        signature = json.dumps(dict(config=cfg, sources=hashes), sort_keys=True)
        write_json(args.output / 'resolved_config.json', dict(config=cfg, source_hashes=hashes,
            pairs=pairs, examples_without_adjacent_partner=no_partner, shifted_native_jobs=sum(map(len, grouped.values())),
            original_screen_windows=len(examples), original_native_crops=len(lookup), training_head_fitted=False))
        # Freeze exact original true-target templates and original complete-patch variance.
        fields = {'power_4_8': (0, 1), 'power_8_13': (1, 2), 'rms_envelope': (2, 10)}
        with np.load(source_paths['templates']) as f:
            templates = {b: f[b + '_train_attribute_mean'] for b in 'SO'}
            counts = {b: f[b + '_train_count'] for b in 'SO'}
        with np.load(source_paths['training_bank']) as bank, np.load(old_pattern / 'train_IDs.npz') as ids:
            np.testing.assert_array_equal(bank['examples'], panels['train'])
            np.testing.assert_array_equal(bank['examples'], ids['examples'])
            if len(bank['examples']) != 15360:
                raise ValueError('original 15360 variance panel required')
            variances = {}
            for b in 'SO':
                np.testing.assert_array_equal(bank[b], ids[b])
                values = bank[b + '_attributes'][bank['valid']]
                variance = np.maximum(values.var(axis=0), 0.)
                variances[b] = np.array([variance[a:z].mean() for a, z in fields.values()])
        write_json(args.output / 'training_variance.json', dict(source=str(source_paths['training_bank']),
            source_SHA256=hashes['training_bank'], complete_patches=len(values), original_windows=15360,
            fields=list(fields), variance=variances, fitted_new_template=False))
        out = args.output / 'native_shifted_ids'
        out.mkdir(exist_ok=True)
        AS_STATE.update(model=model, rows=rows, scales=scales, output=out, threads=args.threads,
            worker_CPU_seconds=cfg['resources']['worker_CPU_seconds'], signature=signature)
        probes = []
        with threadpool_limits(1), np.load(source_paths['original_IDs']) as original:
            for b in 'SO':
                for band in range(5): model._book(b, band)
            for dataset in sorted(grouped):
                (i, start), item = next((key, value) for key, value in lookup.items() if value['dataset'] == dataset)
                row, names = rows[i], model.metadata['positions'][dataset]
                positions = next(v['positions'] for v in row['inputs']['score'] if v['start_s'] == start)
                crop = dict(row=i, start_s=start, positions=positions)
                fields_old, observed_old, _ = base_tokenizer_joint_eeg(rows, crop, scales, names, OrderedDict(), allowed_splits=('validation',))
                fields_new, observed, _ = base_tokenizer_joint_eeg_context(rows, crop, scales, names)
                np.testing.assert_array_equal(fields_old['x'], fields_new)
                np.testing.assert_array_equal(observed_old, observed)
                assigned = model.encode(fields_new.T.reshape(1, len(names), 15, 400), dataset, observed[None])
                for b in 'SO': np.testing.assert_array_equal(assigned[b], original[f'{b}_{item["crop"]}'])
                probes.append(dict(dataset=dataset, row=i, start_s=start, all_native_IDs_exact=True, observation_reader_exact=True))
        write_json(args.output / 'original_reencoding_probes.json', probes)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork')) as pool:
            files = list(pool.map(natural_context_chunk, jobs))
        shifted, raw_flags = {}, {}
        for file in files:
            with np.load(file) as f:
                for j, identity in enumerate(f['jobs']):
                    key = tuple(map(int, identity))
                    shifted[key] = {b: f[b][j][None] for b in 'SO'}
                    raw_flags[key] = {k: f[k][j] for k in ['original_observed', 'shifted_observed', 'center_observation_complete', 'model_input_center_equal']}
        decoder = load_model(source_paths['decoder'])
        replay_decoder = load_model(source_paths['decoder'])
        metrics, changes, selected, selected_pairs, support = defaultdict(list), {b: [] for b in 'SO'}, [], [], []
        full_predictions, saved_IDs = {}, {}
        wave = defaultdict(list)
        cache = OrderedDict()
        with np.load(source_paths['original_IDs']) as old_ids:
            for n, i, start, p, direction in pairs:
                key = i, start, direction
                row, dataset = rows[i], rows[i]['dataset']
                native = next(j for j, name in enumerate(model.metadata['positions'][dataset]) if name.upper() == row['positions'][p].upper())
                if key not in full_predictions:
                    item = lookup[i, start]
                    old = {b: old_ids[f'{b}_{item["crop"]}'] for b in 'SO'}
                    new = shifted[key]
                    prediction = {arm: decoder.decode_ids(value, dataset) for arm, value in [('original', old), ('shifted', new)]}
                    for b in 'SO':
                        np.testing.assert_array_equal(old[b][0, native], original_local[b][n])
                        saved_IDs[f'original_{b}_{len(full_predictions)}'] = old[b]
                        saved_IDs[f'shifted_{b}_{len(full_predictions)}'] = new[b]
                    repeated = replay_decoder.decode_ids(new, dataset)
                    for b in ['S', 'O', 'sum']: np.testing.assert_array_equal(prediction['shifted'][b], repeated[b])
                    full_predictions[key] = prediction
                prediction = full_predictions[key]
                batch = base_tokenizer_batch(rows, np.array([[i, start, p]]), scales, array_cache=cache, cache_limit=32)
                complete = bool(batch['mask'][0, 7].all())
                band = native % 5
                factor = scales['positions']['eeg'][dataset][row['positions'][p]]['scale']
                for arm, center in [('original', 7), ('shifted', 7 - direction)]:
                    center_prediction = {b: prediction[arm][b][:, native, center] for b in ['S', 'O', 'sum']}
                    scored = natural_center_metrics(center_prediction, batch, np.array([factor]))
                    for name, value in scored.items(): metrics[arm + '__' + name].append(value)
                    for b, value in center_prediction.items(): wave[arm + '_' + b].append(value)
                for b in 'SO':
                    old = int(original_local[b][n, 7]); new = int(shifted[key][b][0, native, 7 - direction])
                    seen = old > 0 and new > 0 and counts[b][band, old] > 0 and counts[b][band, new] > 0
                    change = feature_means((templates[b][band, old] - templates[b][band, new])[None] ** 2, fields)[0]
                    if not (seen and complete): change[:] = np.nan
                    changes[b].append(np.r_[old, new, old != new, seen, old == 0, new == 0,
                        counts[b][band, old], counts[b][band, new], change])
                for field in ['x', 's', 'o', 'mask']: wave['target_' + field].append(batch[field][:, 7])
                wave['native_scale'].append(np.array([factor]))
                selected.append([i, start, p]); selected_pairs.append([n, direction, band])
                support.append([complete, *[bool(raw_flags[key][name][native]) for name in raw_flags[key]]])
        selected = np.asarray(selected)
        arrays = {key: np.concatenate(value) for key, value in metrics.items()}
        np.savez_compressed(args.output / 'center_waveforms_and_errors.npz', examples=selected,
            pairing=np.asarray(selected_pairs), support=np.asarray(support), **arrays,
            **{key: np.concatenate(value) for key, value in wave.items()})
        np.savez_compressed(args.output / 'all_original_and_shifted_native_IDs.npz', **saved_IDs)
        branches = {}
        for b in 'SO':
            value = np.asarray(changes[b])
            normalized = value[:, 8:] / variances[b]
            complete = np.asarray(support)[:, 0].astype(bool)
            stats = hierarchy(np.column_stack([np.where(complete, value[:, 2], np.nan),
                np.where(complete, value[:, 3], np.nan), normalized]), selected, rows)
            branches[b] = dict(metrics=stats, column_order=['ID_changed', 'both_nonzero_train_supported', *fields],
                attribute_change_over_training_variance=dict(zip(fields, np.asarray(stats['macro'])[2:])),
                natural_context_subgate=bool(np.all(np.asarray(stats['macro'])[2:] < .25)),
                counts=dict(paired=len(value), complete_center=int(complete.sum()),
                    both_supported=int((value[:, 3].astype(bool) & complete).sum()), original_zero=int(value[:, 4].sum()),
                    shifted_zero=int(value[:, 5].sum()), original_unseen=int((value[:, 6] == 0).sum()), shifted_unseen=int((value[:, 7] == 0).sum())))
            branches[b]['by_fixed_frequency_role'] = {
                str(band): hierarchy(np.column_stack([np.where(complete, value[:, 2], np.nan),
                    np.where(complete, value[:, 3], np.nan), normalized])[np.asarray(selected_pairs)[:, 2] == band],
                    selected[np.asarray(selected_pairs)[:, 2] == band], rows)
                for band in range(5) if np.any(np.asarray(selected_pairs)[:, 2] == band)}
            branches[b]['availability_control'] = dict(
                same_original_shifted_30s_count=int((np.asarray(support)[:, 1] == np.asarray(support)[:, 2]).sum()),
                different_original_shifted_30s_count=int((np.asarray(support)[:, 1] != np.asarray(support)[:, 2]).sum()),
                physical_center_input_equal_count=int(np.asarray(support)[:, 4].sum()))
            np.savez_compressed(args.output / (b + '_natural_context.npz'), examples=selected, pairing=np.asarray(selected_pairs),
                original_ID=value[:, 0].astype(np.int32), shifted_ID=value[:, 1].astype(np.int32),
                supported=value[:, 3].astype(bool), original_train_count=value[:, 6], shifted_train_count=value[:, 7],
                attribute_squared_change=value[:, 8:], normalized_attribute_squared_change=normalized)
        write_json(args.output / 'summary.json', dict(status='numerical_completed_pending_actual_image_open',
            native_positions=model.metadata['positions'],
            support_columns=['complete_supervised_center', 'original_observed_30s', 'shifted_observed_30s', 'center_observation_complete', 'model_input_center_equal'],
            planned_screen_windows=len(examples), paired_windows=len(pairs), examples_without_adjacent_partner=no_partner,
            shifted_native_crops=len(shifted), branch=branches,
            center_waveform_metrics={arm: base_training_aggregate({key.split('__')[1]: value for key, value in arrays.items() if key.startswith(arm + '__')}, selected, rows) for arm in ['original', 'shifted']},
            source_hashes=hashes, same_physical_X_and_observation_masks='exact all native sites in every shifted crop',
            ID_replay='static ID-only decoder reloaded; every new waveform exact',
            original_pattern_main_gates='unchanged; this audit adds only missing natural-context evidence',
            fitted_model_parameters=0, confirmation_reads=0, reserved_four_reads=0, test_reads=0))
        write_json(args.output / 'integer_metadata.json', [dict(ordinal=j, row=i, original_start_s=start, shifted_start_s=start + 2 * direction,
            direction=direction, dataset=rows[i]['dataset'], subject=rows[i]['subject'], record=rows[i]['record'],
            native_positions=model.metadata['positions'][rows[i]['dataset']]) for j, (i, start, direction) in enumerate(full_predictions)])
        render_natural_context(args, rows, selected, selected_pairs, support, full_predictions, changes, scales)


def render_natural_context(args, rows, examples, pairing, support, predictions, changes, scales):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = json.loads((args.output / 'summary.json').read_text())
    variance = json.loads((args.output / 'training_variance.json').read_text())['variance']
    support = np.asarray(support)
    pairing = np.asarray(pairing)
    visual, cache = [], OrderedDict()
    for dataset in sorted({rows[i]['dataset'] for i, _, _ in examples}):
        candidates = [j for j, (i, _, _) in enumerate(examples) if rows[i]['dataset'] == dataset and support[j, 0]]
        if not candidates: continue
        fixed = min(candidates, key=lambda j: tuple(examples[j]))
        random = min(candidates, key=lambda j: hashlib.sha256(json.dumps([17, dataset, examples[j].tolist()]).encode()).hexdigest())
        score = np.asarray(changes['S'])[:, 8:10] / np.asarray(variance['S'])[:2]
        score = score.sum(1)
        measured = [j for j in candidates if np.isfinite(score[j])]
        hard = min(measured, key=lambda j: abs(score[j] - np.quantile(score[measured], .95))) if measured else fixed
        chosen = [('fixed', fixed), ('random_seed17', random), ('p95_S_template_change', hard)]
        unsupported = [j for j in candidates if not np.asarray(changes['S'])[j, 3] or not np.asarray(changes['O'])[j, 3]]
        if unsupported: chosen.append(('first_unsupported_endpoint', min(unsupported, key=lambda j: tuple(examples[j]))))
        for rule, j in chosen:
            i, start, p = map(int, examples[j]); direction = int(pairing[j, 1]); row = rows[i]
            native = next(n for n, name in enumerate(summary['native_positions'][dataset]) if name.upper() == row['positions'][p].upper())
            batch = base_tokenizer_batch(rows, np.array([[i, start, p]]), scales, array_cache=cache, cache_limit=16)
            factor = scales['positions']['eeg'][dataset][row['positions'][p]]['scale']
            prediction = predictions[i, start, direction]
            time_axis = start + np.arange(6000) / 200
            shifted_time = time_axis + 2 * direction
            mask = batch['mask'][0].ravel().astype(bool)
            left, right = max(start, start + 2 * direction), min(start + 30, start + 30 + 2 * direction)
            fig, axes = plt.subplots(3, 3, figsize=(18, 11), layout='constrained')
            for line, (field, branch) in enumerate([('x', 'sum'), ('s', 'S'), ('o', 'O')]):
                truth = batch[field][0].ravel() * factor
                old = prediction['original'][branch][0, native].ravel() * factor
                new = prediction['shifted'][branch][0, native].ravel() * factor
                within_old = (time_axis >= left) & (time_axis < right) & mask
                within_new = (shifted_time >= left) & (shifted_time < right)
                minimum = min(truth[within_old].min(), old[within_old].min(), new[within_new].min())
                maximum = max(truth[within_old].max(), old[within_old].max(), new[within_new].max())
                margin = max((maximum - minimum) * .04, 1e-12)
                for col, bounds in enumerate([(left, right), (start + 14, start + 16), (start + 13.7, start + 14.3)]):
                    axis = axes[line, col]
                    axis.plot(time_axis, np.where(mask, truth, np.nan), color='#1a3444', lw=.9, label='true target')
                    axis.plot(time_axis, np.where(mask, old, np.nan), color='#3579a8', lw=.8, label='original integers')
                    # The common physical scoring mask is aligned by actual sample index.
                    shift_mask = np.zeros(6000, bool)
                    if direction == 1: shift_mask[:5600] = mask[400:]
                    else: shift_mask[400:] = mask[:5600]
                    axis.plot(shifted_time, np.where(shift_mask, new, np.nan), color='#b75d28', lw=.8, label='shifted-context integers')
                    for edge in range(start, start + 31, 2): axis.axvline(edge, color='#bac7cf', lw=.4, zorder=0)
                    axis.set(xlim=bounds, ylim=(minimum - margin, maximum + margin), xlabel='Physical record time (s)',
                        ylabel=branch + ' / ' + ('native export unit' if dataset == 'refed' else row['source_unit']))
                    if line == 0: axis.set_title(['Common 28 s', 'Identical physical center 2 s', 'Physical token boundary'][col])
                    if line == col == 0: axis.legend(fontsize=8)
            ids = ' | '.join(f'{b} {int(changes[b][j][0])}->{int(changes[b][j][1])}' for b in 'SO')
            fig.suptitle(f'{dataset} / {row["subject"]} / {row["record"]} / {row["positions"][p]} / original +{start}s / shifted {2 * direction:+d}s / {rule}\n{ids}; fixed native frequency role {native % 5}; actual 200Hz voltage, same axes, original mask', fontsize=12)
            path = args.output / f'{dataset}_{rule}.png'; fig.savefig(path, dpi=120); plt.close(fig)
            visual.append(dict(path=path.name, selection=rule, example=[i, start, p], subject=row['subject'],
                record=row['record'], direction=direction, center_physical_start_s=start + 14,
                original_shifted_IDs={b: list(map(int, changes[b][j][:2])) for b in 'SO'}))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    for j, field in enumerate(['power_4_8', 'power_8_13', 'rms_envelope']):
        for branch, color in [('S', '#3579a8'), ('O', '#b75d28')]:
            values = summary['branch'][branch]['metrics']['by_dataset']
            datasets = sorted(values)
            axes[j].plot(range(4), [values[d][j + 2] for d in datasets], 'o-', color=color, label=branch)
        axes[j].axhline(.25, color='grey', ls='--', label='unchanged context subgate')
        axes[j].set(xticks=range(4), xticklabels=['single_trial', 'REFED', 'simultaneous', 'visual_cognitive'],
            ylabel='Template change MSE / original full-train variance', title=field)
        axes[j].tick_params(axis='x', rotation=22); axes[j].legend(fontsize=7)
    fig.savefig(args.output / 'context_attribute_stability.png', dpi=160); plt.close(fig)
    write_json(args.output / 'figure_manifest.json', dict(status='rendered_pending_actual_open', waveforms=visual,
        summary='context_attribute_stability.png', selection='frozen fixed/random/P95; extra first unsupported endpoint if present'))
    with (args.output / 'per_subject_context_metrics.csv').open('w') as file:
        writer = csv.writer(file); writer.writerow(['branch', 'dataset', 'subject', 'ID_changed', 'both_nonzero_train_supported', 'theta_ratio', 'alpha_ratio', 'RMS_ratio'])
        for branch, result in summary['branch'].items():
            for row in result['metrics']['by_subject']: writer.writerow([branch, row['dataset'], row['subject'], *row['value']])


def render_context_failure_tails(args):
    """Supplementary maximum-change panels from retained centers; no signal reads."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = args.output
    metadata = json.loads((root / 'integer_metadata.json').read_text())
    lookup = {(v['row'], v['original_start_s'], v['direction']): v for v in metadata}
    with np.load(root / 'center_waveforms_and_errors.npz') as f:
        wave = {k: f[k] for k in f}
    local = {}
    for branch in 'SO':
        with np.load(root / (branch + '_natural_context.npz')) as f:
            local[branch] = {k: f[k] for k in f}
    examples, pairing = wave['examples'], wave['pairing']
    identities = [lookup[int(i), int(start), int(pairing[j, 1])] for j, (i, start, p) in enumerate(examples)]
    figures = []
    for dataset in sorted({v['dataset'] for v in identities}):
        candidates = np.array([j for j, v in enumerate(identities) if v['dataset'] == dataset])
        chosen = []
        for branch, column, label in [('S', 0, 'maximum S theta'), ('S', 1, 'maximum S alpha'), ('S', 2, 'maximum S RMS'), ('O', None, 'maximum O combined')]:
            values = local[branch]['normalized_attribute_squared_change']
            score = values[:, column] if column is not None else values.sum(1)
            valid = candidates[np.isfinite(score[candidates])]
            if len(valid): chosen.append((label, int(valid[np.argmax(score[valid])])))
        fig, axes = plt.subplots(len(chosen), 3, figsize=(18, 3.2 * len(chosen)), layout='constrained', squeeze=False)
        records = []
        for row, (label, j) in enumerate(chosen):
            i, start, p = map(int, examples[j]); item = identities[j]
            time_axis = start + 14 + np.arange(400) / 200
            factor = wave['native_scale'][j]
            mask = wave['target_mask'][j].astype(bool)
            for col, (target, branch) in enumerate([('x', 'sum'), ('s', 'S'), ('o', 'O')]):
                axis = axes[row, col]
                for name, color, value in [('target', '#1a3444', wave['target_' + target][j]), ('original', '#3579a8', wave['original_' + branch][j]), ('shifted', '#b75d28', wave['shifted_' + branch][j])]:
                    axis.plot(time_axis, np.where(mask, value * factor, np.nan), c=color, lw=1, label=name)
                axis.set(xlim=(start + 14, start + 16), xlabel='Identical physical record time (s)', ylabel=branch + ' / native voltage unit')
                if row == 0: axis.legend(fontsize=7)
                if col == 0: axis.set_title(f'{label}: subject {item["subject"]}, {item["record"]}, row {i}, source position {p}', fontsize=9)
                else:
                    b = 'S' if col == 1 else 'O'
                    ids = [int(local[b][k][j]) for k in ['original_ID', 'shifted_ID']]
                    ratios = local[b]['normalized_attribute_squared_change'][j]
                    axis.set_title(f'{b} ID {ids[0]}->{ids[1]}; theta/alpha/RMS ratios {ratios[0]:.3f}/{ratios[1]:.3f}/{ratios[2]:.3f}', fontsize=9)
            records.append(dict(selection=label, example=examples[j].tolist(), direction=int(pairing[j, 1]),
                subject=item['subject'], record=item['record'], normalized_change={b: local[b]['normalized_attribute_squared_change'][j] for b in 'SO'}))
        fig.suptitle(f'{dataset}: supplementary failure tails, chosen after the frozen primary audit\nEvery paired window remains in primary scores; these diagnostic maxima change no gate or model', fontsize=12)
        path = root / (dataset + '_maximum_template_change_centers.png'); fig.savefig(path, dpi=120); plt.close(fig)
        figures.append(dict(path=path.name, records=records))
    write_json(root / 'supplementary_tail_figure_manifest.json', dict(status='rendered_pending_actual_open',
        selection='post hoc diagnostic maximum S theta, alpha, RMS and O combined change in each dataset; no numerical selection or threshold change',
        measured_signal_reads=0, data='retained center_waveforms_and_errors and natural_context arrays only', figures=figures))


def render_actual_atlas(args):
    """Fixed major/low-support codes, train means and true validation members."""
    from audit_joint_tokenizer_patterns import MemberReservoir
    rows, scales, model, panels, metadata = source(args)
    size = model.codebook_size
    chosen, count_rows = {}, {}
    for b in 'SO':
        with (args.output/(b+'_code_support.csv')).open() as stream:
            entries = list(csv.DictReader(stream))
        lookup = {(int(item['fixed_band_index']),int(item['ID'])): item for item in entries}
        selected = []
        for band in range(5):
            available = [v for (f, code), v in lookup.items() if f == band and code > 0]
            major = max(available,key=lambda v:(int(v['train_complete_patches']),-int(v['ID'])))
            selected.append((band,int(major['ID']),'largest_train_occupancy'))
            rare = [v for v in available if int(v['validation_complete_patches']) > 0
                    and (int(v['train_complete_patches']) < 8 or int(v['train_subjects']) < 2)]
            if not rare:
                rare = [v for v in available if int(v['ID']) != int(major['ID'])]
            other = max(rare,key=lambda v:(int(v['validation_complete_patches']),-int(v['ID'])))
            selected.append((band,int(other['ID']),'largest_validation_low_support_or_next_code'))
        chosen[b] = selected
        count_rows[b] = lookup
    selected_keys = {b: {band*size+code for band,code,_ in chosen[b]} for b in 'SO'}
    reservoirs = {b:MemberReservoir() for b in 'SO'}
    totals = {b:defaultdict(lambda:np.zeros(400)) for b in 'SO'}
    counts = {b:defaultdict(int) for b in 'SO'}
    cache = OrderedDict()
    for split, examples in panels.items():
        with np.load(args.output/(split+'_IDs.npz'), allow_pickle=False) as stored:
            np.testing.assert_array_equal(examples,stored['examples'])
            ids = {b:stored[b].astype(np.int64)+size*stored['fixed_band'][:,None] for b in 'SO'}
        for start in range(0,len(examples),args.batch_size):
            take=examples[start:start+args.batch_size]
            batch=base_tokenizer_batch(rows,take,scales,array_cache=cache,cache_limit=64)
            valid=batch['mask'].all(-1)
            patch_info=np.concatenate([np.repeat(take,15,axis=0),np.tile(np.arange(15),len(take))[:,None]],axis=1).reshape(len(take),15,4)
            for b in 'SO':
                key=ids[b][start:start+len(take)]
                included=valid & np.isin(key,list(selected_keys[b]))
                value=batch[b.lower()][included]
                selected=key[included]
                reservoirs[b].add(split,selected,value,patch_info[included])
                if split == 'train':
                    for code in np.unique(selected):
                        totals[b][int(code)]+=value[selected==code].sum(0)
                        counts[b][int(code)]+=int((selected==code).sum())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=[]
    for b in 'SO':
        selected=chosen[b]
        arrays={}
        for first in range(0,len(selected),4):
            page=selected[first:first+4]
            fig,axes=plt.subplots(len(page),2,figsize=(13,2.7*len(page)),squeeze=False,layout='constrained')
            for n,(band,code,rule) in enumerate(page):
                composite=band*size+code
                training=totals[b][composite]/counts[b][composite] if counts[b][composite] else np.full(400,np.nan)
                arrays[f'train_mean_{composite}']=training
                train=reservoirs[b].members.get(('train',composite))
                validation=reservoirs[b].members.get(('validation',composite))
                waves=np.empty((0,400)) if validation is None else validation[1]
                trainwaves=np.empty((0,400)) if train is None else train[1]
                for split, member in [('train',train),('validation',validation)]:
                    if member is not None:
                        arrays[f'{split}_{composite}_waves']=member[1]
                        arrays[f'{split}_{composite}_row_start_position_patch']=member[2]
                def power(values):
                    w=np.hanning(400)
                    return abs(np.fft.rfft((values-values.mean(-1,keepdims=True))*w,axis=-1))**2*2/(200*(w*w).sum())
                displays=[waves,power(waves)]
                means=[training,power(trainwaves).mean(0) if len(trainwaves) else np.full(201,np.nan)]
                for column,(values,mean) in enumerate(zip(displays,means)):
                    axis=axes[n,column]; x=np.arange(400)/200 if column==0 else np.fft.rfftfreq(400,1/200)
                    if len(values):
                        low,mid,high=np.quantile(values,[.1,.5,.9],axis=0)
                        axis.fill_between(x,low,high,color='#e3a24a',alpha=.25,label='Validation 10-90% (<=32 members)')
                        axis.plot(x,values[:3].T,color='#9b6c3a',lw=.6,alpha=.6)
                        axis.plot(x,mid,color='#bf6b20',lw=1,label='Validation median')
                    axis.plot(x,mean,color='#236c9e',lw=1.2,label='Train full mean' if column==0 else 'Train reservoir mean PSD')
                    axis.set(xlabel='Time (s)' if column==0 else 'Frequency (Hz)',ylabel='Voltage / frozen train X SD' if column==0 else 'PSD / train X SD squared / Hz')
                    if column:axis.set_xlim(1,45)
                    axis.legend(fontsize=6)
                row=count_rows[b][band,code]
                axes[n,0].set_title(f'{b}, fixed band {BANDS[band]} Hz, ID {code}; train n={row["train_complete_patches"]}, subjects={row["train_subjects"]}; val n={row["validation_complete_patches"]}',fontsize=9)
            fig.suptitle('True AS single-ID target members, not full-sequence decoder outputs. Per-code axes; fixed selection.',fontsize=10)
            path=args.output/f'{b}_actual_atlas_{first//4+1:02d}.png';fig.savefig(path,dpi=180);plt.close(fig)
            records.append(dict(branch=b,path=str(path),codes=page))
        np.savez_compressed(args.output/f'{b}_actual_atlas_members.npz',**arrays)
    write_json(args.output/'atlas_manifest.json',dict(status='completed',selection='Per branch/band largest train occupancy and largest validation low-support code (fallback next code); fixed before member inspection',
        scope='diagnostic ten codes per branch; all-code support/dead/rare distribution is retained in CSV; not a major90-percent atlas',
        member_sampling='fixed seed817 uniform-priority 32 members per split/code',figures=records,visual_audit='pending_actual_open'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--panels',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--attributes', action='store_true')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--benchmark-only', action='store_true')
    parser.add_argument('--atlas', action='store_true')
    parser.add_argument('--natural-context', action='store_true')
    parser.add_argument('--natural-context-tails', action='store_true')
    parser.add_argument('--contract', type=Path)
    args = parser.parse_args()
    args.model, args.panels, args.output = (v.resolve() for v in (args.model, args.panels, args.output))
    if args.workers < 1 or args.threads < 1 or args.workers*max(args.threads,2) > 12:
        parser.error('at most twelve numerical/FFT threads across workers')
    if args.natural_context_tails:
        render_context_failure_tails(args)
    elif args.natural_context:
        if args.contract is None: parser.error('natural context requires its frozen contract')
        assess_natural_context(args)
    elif args.atlas:
        render_actual_atlas(args)
    elif args.attributes:
        assess_attributes(args)
    else:
        run(args)
