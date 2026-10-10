"""TK v1 identity, native measurement and frozen training-coordinate adapter.

Only registry-listed ordinary public inputs are read. Splits precede all array
access. Native readers and observation operators retain their existing owners.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import hashlib
import json

import numpy as np

from .unified_physiology import (
    UnifiedPhysiologyWindowDataset, canonical_label, load_native_fnirs_record,
    load_native_eeg_record,
)
from .ssm_prepared import fit_feature_coordinate
from .homer2_preprocessing import modified_beer_lambert
from src.inference.observation_baselines import eeg_band_power, disjoint_native_bin_operator


DATASETS = ('eeg_fnirs_single_trial', 'simultaneous_eeg_nirs')


def subject_partitions(subjects, seed, train_fraction=.6, validation_fraction=.2):
    """Dataset-stratified, deterministic and label-free subject assignment."""
    result = {}
    for offset, ds in enumerate(sorted(subjects)):
        ids = np.asarray(sorted(set(subjects[ds])))
        if len(ids) < 5:
            raise ValueError('at least five subjects per dataset required')
        ids = np.random.default_rng(seed+offset).permutation(ids)
        a, b = int(len(ids)*train_fraction), int(len(ids)*(train_fraction+validation_fraction))
        for split, part in (('train', ids[:a]), ('validation', ids[a:b]), ('test', ids[b:])):
            for subject in part:
                result[(ds, str(subject))] = split
    return result


def regional_anchors(record, geometry, config):
    meta = record.manifest['measurement']
    eeg = geometry.for_channels(record=record, modality='eeg', channel_names=meta['eeg_channel_names'])
    hb = geometry.for_channels(record=record, modality='fnirs', channel_names=meta['fnirs_channel_names'][::2])
    if any(r['coordinate_units'] != 'normalized_head_unit' for r in eeg+hb):
        raise ValueError('registered common geometry required')
    xyz = lambda rows: np.array([[r.get(k, np.nan) for k in ('x', 'y', 'z')] for r in rows], float)
    e, h = xyz(eeg), xyz(hb)
    if not np.isfinite(e).all() or not np.isfinite(h).all():
        raise ValueError('incomplete geometry')
    unit = h/np.linalg.norm(h, axis=1)[:, None]
    anchors = []
    for region, direction in zip(config['regions'], config['region_vectors']):
        direction = np.asarray(direction)/np.linalg.norm(direction)
        i = int(np.argmax(unit@direction))
        if unit[i]@direction < np.cos(np.deg2rad(50)):
            raise ValueError('unsupported anatomical region')
        neighbors = np.argsort(np.linalg.norm(e-h[i], axis=1), kind='stable')[:6]
        anchors.append(dict(region=region, hb_pair=i, hb_channel=hb[i]['base_channel_name'],
            eeg_channels=[eeg[j]['channel_name'] for j in neighbors],
            coordinate_system=hb[i]['coordinate_system'], geometry_source=hb[i]['source_file']))
    if len({a['hb_pair'] for a in anchors}) != len(anchors):
        raise ValueError('regions must use distinct native channels')
    return anchors


def inventory(project_root, cfg):
    """Read metadata only, freeze complete cohort/splits before any signals."""
    if tuple(cfg['data']['datasets']) != DATASETS or cfg['data']['protected_data'] != 'forbidden':
        raise ValueError('unregistered dataset or protected scope')
    data = UnifiedPhysiologyWindowDataset(Path(project_root)/cfg['data']['cache_root'],
        dataset_ids=DATASETS, window_duration_s=90, window_offset_s=-30)
    refs = [r for r in data.windows if canonical_label(r.event, r.record.dataset_id)['task'] in cfg['data']['tasks']]
    subjects = {ds: sorted({r.record.canonical_subject_id for r in refs if r.record.dataset_id == ds}) for ds in DATASETS}
    partitions = subject_partitions(subjects, cfg['seed'], cfg['data']['train_fraction'], cfg['data']['validation_fraction'])
    rows, excluded, anchor_cache = [], [], {}
    for ref in refs:
        r, event = ref.record, ref.event
        label = canonical_label(event, r.dataset_id)
        identity = dict(dataset=r.dataset_id, subject=r.canonical_subject_id, record=r.base_record_id,
            join_key=r.join_key, event_id=int(event['event_index']), task=label['task'], condition=label['condition'],
            split=partitions[(r.dataset_id, r.canonical_subject_id)])
        es, hs = float(event['eeg_time_ms'])/1000-30, float(event['fnirs_time_ms'])/1000-30
        shape = r.manifest['array_shapes']
        reason = None
        if label['condition'] not in cfg['data']['task_labels'][label['task']]:
            reason = 'unknown_or_unsupported_label'
        if min(es, hs) < 0 or es+90 > shape['eeg'][0]/200 or hs+90 > shape['fnirs'][0]/10:
            reason = 'incomplete_native_90s_context'
        if reason:
            excluded.append(dict(identity, reason=reason)); continue
        if r.join_key not in anchor_cache:
            anchor_cache[r.join_key] = regional_anchors(r, data.geometry_index, cfg['data'])
        trial_id = f'{r.dataset_id}__{r.canonical_subject_id}__{r.base_record_id}__{event["event_index"]:04d}'
        support = event.get('metadata', {}).get('alignment_support_ms', {})
        session = event.get('metadata', {}).get('session_idx')
        if session is None:
            # Concatenated sessions are distinguished by native alignment support,
            # never by guessed event numbering or task labels.
            session = json.dumps(support, sort_keys=True)
        for anchor in anchor_cache[r.join_key]:
            rows.append(dict(identity, id=trial_id+'__'+anchor['region'], trial_id=trial_id,
                session=str(session), eeg_start_s=es, hb_start_s=hs, anchor=anchor,
                coordinate=r.dataset_id+'__'+anchor['region'],
                source_identity=r.manifest['source_files']))
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate canonical window identity')
    return dict(schema='functional_tokenizer_inventory_v1', rows=rows, excluded=excluded,
        subjects={ds: {s: sorted(k[1] for k,v in partitions.items() if k[0]==ds and v==s)
                       for s in ('train','validation','test')} for ds in DATASETS},
        metadata_windows=len(refs), source_cache=cfg['data']['cache_root'],
        alignment_excluded_records=data.excluded_alignment_records,
        context='native continuous [-30,+60)s; student [0,+20)s plus visible Hb reference [-30,-25)s; all dependencies grouped by subject',
        measured_arrays_read=0, confirmation='public_development_previously_examined_population')


def prepare_native_record(project_root, cfg, rows):
    """One record read serves all planned windows/regions; no ad-hoc parser."""
    from .clean_physiology_cache import CleanPhysiologyCacheIndex
    if not rows or any(r['join_key'] != rows[0]['join_key'] for r in rows):
        raise ValueError('one canonical record per producer')
    if any(r['dataset'] not in DATASETS or r['split'] not in ('train','validation','test') for r in rows):
        raise ValueError('public split identity required before array access')
    root = Path(project_root)
    index = CleanPhysiologyCacheIndex(root/cfg['data']['cache_root'])
    matches = [r for r in index.records if r.join_key == rows[0]['join_key']]
    if len(matches) != 1:
        raise ValueError('unique registered record required')
    record = matches[0]
    if record.manifest['source_files'] != rows[0]['source_identity']:
        raise ValueError('source identity changed after inventory freeze')
    # Verify the registry's existing source identity before the first native
    # array read. This is a provenance check, not an additional permission gate.
    from .registry import REGISTERED_DATASETS
    admitted_root=(root/REGISTERED_DATASETS[record.dataset_id].default_root).resolve()
    for source in rows[0]['source_identity']:
        path=Path(source['path']);path=path if path.is_absolute() else root/path
        if not path.resolve().is_relative_to(admitted_root):
            raise ValueError('native source is outside the registered public dataset root')
        digest=hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(8*1024*1024),b''):digest.update(block)
        if digest.hexdigest()!=source['sha256']:
            raise ValueError('registered native source changed: '+str(path))
    eeg, hb = load_native_eeg_record(root, record), load_native_fnirs_record(root, record)
    if eeg.sample_rate_hz != 200 or eeg.native_unit not in ('uV', 'µV', 'μV'):
        raise ValueError('v1 requires verified native 200Hz microvolt EEG')
    names = {n.upper(): i for i,n in enumerate(eeg.channel_names)}
    for row in rows:
        try:
            start = round(row['eeg_start_s']*200)
            e = eeg.values[start:start+18000, [names[n.upper()] for n in row['anchor']['eeg_channels']]]
            take = (hb['time_s'] >= row['hb_start_s']) & (hb['time_s'] < row['hb_start_s']+90)
            t = hb['time_s'][take]-row['hb_start_s']
            ops = disjoint_native_bin_operator(t)
            pair = row['anchor']['hb_pair']
            h = hb['values'][take, pair]
            if hb['optical_intensity'] is not None:
                intensity = hb['optical_intensity'][take, pair]
                if not np.isfinite(intensity).all() or np.any(intensity <= 0):
                    raise ValueError('invalid native intensity')
                reference = intensity[t < 5].mean(0)
                h, _ = modified_beer_lambert((-np.log(intensity/reference))[:,None,:], wavelengths_nm=(760.,850.))
                h = h[:,0]
            if e.shape != (18000,6) or not np.isfinite(e).all() or not np.isfinite(h).all():
                raise ValueError('incomplete finite native support')
            power = eeg_band_power(e, bands=cfg['tensor']['eeg_bands_hz'], sample_rate=200, target_rate=4)
            ef = ops['eeg']@np.log(np.maximum(power, np.finfo(float).tiny)).reshape(360,30)
            hf = ops['raw']@h
            # Hidden Hb target cannot change any visible bin or optical reference.
            poisoned = h.copy(); hidden = (ops['native_groups']>=152)&(ops['native_groups']<168)
            poisoned[hidden] += np.array([1e6,-2e6])
            visible = ~((np.arange(360)>=152)&(np.arange(360)<168))
            if not np.array_equal((ops['raw']@poisoned)[visible], hf[visible]):
                raise AssertionError('hidden Hb contaminates visible support')
            payload = dict(eeg=e[6000:10000].astype('float32'), hb=hf,
                features=ef, native_time=t,
                trial_id=np.array(row['trial_id']), identity=np.array(row['id']))
            yield row, payload, None
        except (ValueError, FloatingPointError, IndexError) as exc:
            yield row, None, str(exc)


def training_coordinates(rows, read_arrays):
    coordinates = {}
    groups = defaultdict(list)
    for row in rows:
        if row['split']=='train':
            groups[row['coordinate']].append(row)
    for key, group in groups.items():
        chunks, e_sum, e_square, count = [], 0., 0., 0
        for row in group:
            a = read_arrays(row)
            chunks.append((a['features'], a['hb']))
            e = np.asarray(a['eeg'], float)
            e_sum += e.sum(); e_square += np.square(e).sum(); count += e.size
        co = fit_feature_coordinate(chunks)
        co.update(eeg_mean=e_sum/count, eeg_scale=max(np.sqrt(e_square/count-(e_sum/count)**2),1e-8),
            training_ids=[r['id'] for r in group], training_subjects=sorted({r['subject'] for r in group}))
        coordinates[key] = co
    if {r['coordinate'] for r in rows} != set(coordinates):
        raise ValueError('each evaluation coordinate requires training support')
    return coordinates


def label_blind_donors(rows, seed):
    """No condition/task/label access, different-subject same-coordinate donor."""
    rng = np.random.default_rng(seed)
    donors = []
    groups=defaultdict(list)
    for i,row in enumerate(rows):
        if row['split']=='train':groups[row['coordinate']].append(i)
    for row in rows:
        candidates = [i for i in groups[row['coordinate']] if rows[i]['subject']!=row['subject']]
        if not candidates:
            raise ValueError('no independent compatible training donor')
        donors.append(int(rng.choice(candidates)))
    return np.array(donors)


def identity_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':')).encode()).hexdigest()
