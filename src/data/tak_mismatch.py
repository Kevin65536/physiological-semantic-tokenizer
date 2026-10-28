"""Record-shared public measurements for the Tak mismatch analysis.

The metadata inventory never opens signal arrays. Preparation revalidates that
inventory against the central cache before calling the existing native EEG and
cached Hb readers. It neither parses raw Hb nor rebuilds a missing cache.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
from pathlib import Path
import re
from typing import Mapping

import numpy as np

from .clean_physiology_cache import CleanPhysiologyCacheIndex, DEFAULT_CLEAN_CACHE_ROOT
from .event_alignment import ADMISSIBLE_ALIGNMENT_CASES
from .registry import REGISTERED_DATASETS
from .ssm_prepared import EEG_PROXY_INPUT_SCHEMA
from .unified_physiology import (
    RAW_DATASET_IDS, CACHED_FNIRS_INPUT_SCHEMA, ChannelGeometryIndex,
    canonical_channel_name, canonical_label, load_cached_fnirs_record, load_native_eeg_record,
)
from src.inference.observation_baselines import eeg_band_power


SCHEMA = 'tak_mismatch_data_v1'
CHANNEL_CANDIDATES = ('nearest6', 'sixth_to_seventh', 'next6', 'random6')
BANDS_HZ = ((1., 4.), (4., 8.), (8., 13.), (13., 30.), (30., 45.))


def _settings(cfg):
    data = cfg.get('data', cfg)
    datasets = tuple(data.get('datasets', RAW_DATASET_IDS))
    if (not datasets or len(set(datasets)) != len(datasets)
            or set(datasets) - set(RAW_DATASET_IDS)
            or data.get('protected_data', 'forbidden') != 'forbidden'):
        raise ValueError('only the four registered public datasets are admitted')
    if (float(data.get('window_seconds', 30.)) != 30.
            or float(data.get('reference_seconds', 5.)) != 5.
            or float(data.get('eeg_filter_seconds', 90.)) != 90.):
        raise ValueError('30 s targets, 5 s references and at most 90 s EEG context required')
    seed = int(cfg.get('seed', data.get('seed', 20261028)))
    folds = int(data.get('folds', data.get('outer_folds', cfg.get('folds', 5))))
    if folds < 2:
        raise ValueError('at least two subject folds required')
    return data, datasets, seed, folds


def _positions(rows):
    values = np.asarray([[r.get(k) for k in ('x', 'y', 'z')] for r in rows], float)
    if values.shape != (len(rows), 3) or not np.isfinite(values).all():
        raise ValueError('complete registered channel geometry required')
    return values


def select_tak_mismatch_channels(eeg_geometry, hb_geometry, *, seed=20261028):
    """Freeze four six-channel sets per Hb pair using geometry alone.

    REFED drawing positions retain their planar coordinate identity. Visual's
    registered Hb coordinates are expressed in its admitted EEG CED frame;
    the loader labels that relationship separately from the EEG rows.
    Optical hardware IDs are unnecessary for a single-pair neighborhood.
    """
    eeg = list(eeg_geometry)
    names = [canonical_channel_name(str(r.get('channel_name', ''))) for r in eeg]
    if len(names) < 12 or any(not n for n in names) or len({n.casefold() for n in names}) != len(names):
        raise ValueError('at least twelve uniquely named EEG channels required')
    ep = _positions(eeg)
    eeg_frames = {(r.get('coordinate_system'), r.get('coordinate_units')) for r in eeg}
    if len(eeg_frames) != 1:
        raise ValueError('EEG geometry must use one registered frame')
    frame, units = next(iter(eeg_frames))
    if (not frame or frame in ('unknown', 'unavailable')
            or units not in ('normalized_head_unit', 'normalized_figure_radius')):
        raise ValueError('registered common template or paper-layout geometry required')
    hb = list(hb_geometry)
    if not hb or len(hb) % 2:
        raise ValueError('ordered complete HbO/HbR geometry pairs required')
    output = []
    for i in range(0, len(hb), 2):
        o, r = hb[i:i+2]
        name = canonical_channel_name(str(o.get('channel_name', '')))
        pair_name = name[:-4] if name.endswith('_HbO') else ''
        pair = dict(id=pair_name, name=pair_name, native_index=i//2,
                    component_indices=[i, i+1], channel_sets={}, status='unavailable')
        try:
            if not pair_name or canonical_channel_name(str(r.get('channel_name', ''))) != pair_name+'_HbR':
                raise ValueError('registered HbO/HbR identity order mismatch')
            hp = _positions([o, r])
            if not np.array_equal(hp[0], hp[1]):
                raise ValueError('HbO/HbR positions disagree')
            for row in (o, r):
                candidate = row.get('coordinate_system')
                same_frame = candidate == frame or (frame == 'EEGLAB_CED_cartesian'
                    and candidate == 'referenced_visual_eeg_head_coordinates')
                if row.get('coordinate_units') != units or not same_frame:
                    raise ValueError('Hb and EEG geometry do not share the admitted frame')
            distances = np.linalg.norm(ep-hp[0], axis=1)
            ranked = sorted(range(len(eeg)), key=lambda j: (float(distances[j]), names[j]))
            # A digest ties the random control to channel identities, never to
            # Python hash randomization or current source-array ordering.
            digest = hashlib.sha256((str(seed)+'|'+pair_name+'|'+frame).encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
            choices = [names[j] for j in ranked[6:]]
            random_names = set(rng.choice(sorted(choices), 6, replace=False).tolist())
            random_ids = [j for j in ranked if names[j] in random_names]
            sets = dict(nearest6=ranked[:6], sixth_to_seventh=ranked[:5]+ranked[6:7],
                        next6=ranked[6:12], random6=random_ids)
            pair.update(status='available', channel_sets=sets,
                channel_set_names={key: [names[j] for j in value] for key, value in sets.items()},
                ranked_eeg_indices=ranked, eeg_distances=distances.tolist(),
                coordinate_system=frame, coordinate_units=units,
                spatial_dimension=2 if frame == 'refed_paper_layout_2d_v1' else 3,
                distance_interpretation='registered_template_not_individual_cortical_distance')
        except (ValueError, TypeError) as exc:
            pair['reason'] = str(exc)
        output.append(pair)
    return output


def _registered_sources(record, root):
    identity = record.manifest.get('source_files')
    if not isinstance(identity, list) or not identity:
        raise ValueError('registered source identity missing')
    admitted = (root / REGISTERED_DATASETS[record.dataset_id].default_root).resolve()
    paths = []
    for source in identity:
        if not isinstance(source, Mapping) or not source.get('path'):
            raise ValueError('complete registered source identity required')
        p = Path(source['path'])
        p = (p if p.is_absolute() else root / p).resolve()
        if not p.is_relative_to(admitted):
            raise ValueError('source outside the registered public dataset root')
        paths.append(str(p))
    if len(set(paths)) != len(paths):
        raise ValueError('duplicate registered source path')
    return paths


def _pair_inventory(geometry, names, *, seed, record=None):
    try:
        pairs = select_tak_mismatch_channels(geometry['eeg'],geometry['fnirs'],seed=seed)
    except (ValueError,TypeError) as exc:
        pairs = [dict(id=n[:-4],name=n[:-4],native_index=i//2,component_indices=[i,i+1],
                     channel_sets={},status='unavailable',reason=str(exc))
                for i,n in enumerate(names) if i%2==0]
    probe = None
    if record is not None and record.dataset_id=='visual_cognitive_motivation':
        match = re.search(r'Probe([12])',record.base_record_id,flags=re.IGNORECASE)
        if match is None:
            raise ValueError('explicit Visual probe identity required for a PCA site')
        probe = 'Probe'+match.group(1)
    for p in pairs:
        p['site_id'] = f'{probe}:{p["name"]}' if probe else p['name']
        p['id'] = p['site_id']
        p['probe_id'] = probe
        rows = geometry['eeg'] + [geometry['fnirs'][i] for i in p['component_indices']]
        p['geometry_schemas'] = sorted({str(r.get('schema','unavailable')) for r in rows})
        p['geometry_versions'] = sorted({str(r.get('metadata',{}).get('geometry_version'))
                                         for r in rows if r.get('metadata',{}).get('geometry_version') is not None})
    return pairs


def require_fixed_tak_coordinate_groups(records):
    """A PCA feature slot must name the same electrode across all training rows."""
    seen = {}
    for record in records:
        for pair in record['pairs']:
            if pair['status'] != 'available':
                continue
            key = (record['dataset'],pair['id'])
            signature = (pair['coordinate_system'],pair['coordinate_units'],pair['spatial_dimension'],
                tuple(pair['geometry_schemas']),tuple(pair['geometry_versions']),
                tuple((c,tuple(pair['channel_set_names'][c])) for c in CHANNEL_CANDIDATES))
            if key in seen and seen[key] != signature:
                raise ValueError('PCA coordinate group changes ordered EEG identities or geometry: '+str(key))
            seen[key] = signature
    return len(seen)


def continuous_hb_support_segments(record,events):
    """Conservative Hb support from Hb-clock declarations, independent of EEG.

    Different EEG segments sharing the same Hb support do not create an Hb
    break. These intervals preserve the central index's unverified append-gap
    exclusion; they do not reconstruct unpublished physical session boundaries.
    """
    duration = record.manifest['array_shapes']['fnirs'][0]/10.
    bounds = set()
    for event in events:
        support = event.get('metadata',{}).get('alignment_support_ms',{}).get('fnirs')
        if support is None:
            continue
        if len(support) != 2:
            raise ValueError('malformed central Hb clock support')
        low = 0. if support[0] is None else max(0.,float(support[0])/1000.)
        high = duration if support[1] is None else min(duration,float(support[1])/1000.)
        if not np.isfinite([low,high]).all() or high <= low:
            raise ValueError('invalid central Hb clock interval')
        bounds.add((low,high))
    source = 'central_Hb_clock_support_conservative_not_recovered_session_boundary'
    if not bounds:
        bounds.add((0.,duration))
        source = 'released_record_support_no_finer_Hb_session_boundaries_available'
    ordered = sorted(bounds)
    for a,b in zip(ordered,ordered[1:]):
        if a[1] > b[0]+1e-9:
            raise ValueError('conflicting overlapping central Hb support intervals')
    return [dict(id=f'{record.join_key}|Hb_support={i:03d}|start={low:.6f}|end={high:.6f}',
                 index=i,start_s=low,end_s=high,source=source)
            for i,(low,high) in enumerate(ordered)]


def continuous_tak_support(record, events, reports):
    """Attach admissible EEG support to the one native-Hb 30 s target grid.

    One constant offset per centrally declared segment is the median paired
    marker offset. The original offset range and anchor identities remain in
    metadata; this is a declared alignment approximation, not drift correction.
    Unverified concatenation gaps are never crossed by targets or EEG context.
    """
    shapes = record.manifest['array_shapes']
    eeg_duration = float(shapes['eeg'][0])/200.
    hb_duration = float(shapes['fnirs'][0])/10.
    windows = _hb_only_windows(record,events)
    common = dict(eeg_record_seconds=eeg_duration, hb_record_seconds=hb_duration,
                  covered_seconds=30.*len(windows), tail_seconds=hb_duration-30.*len(windows),
                  prefix_seconds=0.,
                  offset_policy='median_paired_marker_offset_within_central_alignment_support')
    if (not reports or any(r.get('alignment_case') not in ADMISSIBLE_ALIGNMENT_CASES
                          or r.get('label_sequence_match') is False for r in reports)):
        return dict(common, status='hb_only', reason='nonadmitted_or_missing_alignment_report',
                    segments=[], windows=windows, paired_support_seconds=0., paired_covered_seconds=0.,
                    unavailable_filter_context_windows=0)
    grouped = defaultdict(list)
    for event in events:
        et, ht = event.get('eeg_time_ms'), event.get('fnirs_time_ms')
        if et is None or ht is None or not np.isfinite([et, ht]).all():
            continue
        support = event.get('metadata', {}).get('alignment_support_ms', {})
        es = tuple(support.get('eeg', (None, None)))
        hs = tuple(support.get('fnirs', (None, None)))
        if len(es) != 2 or len(hs) != 2:
            raise ValueError('malformed central alignment support')
        grouped[(es, hs)].append(event)
    segments = []
    for (es, hs), anchors in grouped.items():
        offsets = np.asarray([(float(a['fnirs_time_ms'])-float(a['eeg_time_ms']))/1000 for a in anchors])
        offset = float(np.median(offsets))
        elo, ehi = [None if x is None else float(x)/1000 for x in es]
        hlo, hhi = [None if x is None else float(x)/1000 for x in hs]
        low = max(0., -offset, elo if elo is not None else 0.,
                  hlo-offset if hlo is not None else 0.)
        high = min(eeg_duration, hb_duration-offset,
                   ehi if ehi is not None else eeg_duration,
                   hhi-offset if hhi is not None else eeg_duration)
        if not np.isfinite([low, high]).all() or high <= low:
            continue
        context_low = max(0., elo if elo is not None else 0.)
        context_high = min(eeg_duration, ehi if ehi is not None else eeg_duration)
        context_length = min(90., context_high-context_low)
        segment = dict(eeg_start_s=float(low), eeg_end_s=float(high),
            hb_start_s=float(low+offset), hb_end_s=float(high+offset), offset_s=offset,
            offset_min_s=float(offsets.min()), offset_max_s=float(offsets.max()),
            anchor_event_ids=sorted(int(a['event_index']) for a in anchors),
            paired_anchor_count=len(anchors), alignment_support_ms=dict(eeg=list(es), fnirs=list(hs)),
            eeg_context_support_s=[float(context_low),float(context_high)],
            actual_context_s=float(context_length))
        segments.append(segment)
    segments.sort(key=lambda s: (s['eeg_start_s'], s['hb_start_s']))
    for a, b in zip(segments, segments[1:]):
        if a['eeg_end_s'] > b['eeg_start_s']+1e-9 or a['hb_end_s'] > b['hb_start_s']+1e-9:
            raise ValueError('central continuous-support segments overlap')
    for w in windows:
        if not w['hb_support_admitted']:
            continue
        hs = w['hb_start_s']
        for i,s in enumerate(segments):
            if hs < s['hb_start_s']-1e-9 or hs+30 > s['hb_end_s']+1e-9:
                continue
            t = hs-s['offset_s']
            low,high = s['eeg_context_support_s']
            length = s['actual_context_s']
            if length < 90.-1e-9 and not (record.dataset_id == 'refed' and length >= 30.-1e-9):
                w['reason'] = 'continuous_EEG_filter_support_shorter_than_90s'
                break
            # Keep the EEG power bins on the target's phase, so cropping uses
            # integer 100 ms bins even when the marker offset is fractional.
            first_k = int(np.ceil((low-t)*10-1e-9))
            last_k = int(np.floor((high-length-t)*10+1e-9))
            if first_k > last_k:
                w['reason'] = 'no_complete_target_phase_EEG_context'
                break
            k = min(max(-300, first_k),last_k)
            fs = t+k/10.
            if t < fs-1e-9 or t+30 > fs+length+1e-9:
                w['reason'] = 'target_outside_continuous_EEG_context'
                break
            w.update(start_s=float(t),eeg_start_s=float(t),eeg_filter_start_s=float(fs),
                eeg_filter_end_s=float(fs+length),actual_context_s=float(length),
                status='available',mode='joint',reason=None,segment_index=i)
            break
        _window_labels(w,events,record.dataset_id)
    paired_count = sum(w['status']=='available' for w in windows)
    return dict(common,status='available' if paired_count else 'hb_only',
        reason=None if paired_count else 'no_complete_30s_paired_support',segments=segments,windows=windows,
        paired_support_seconds=sum(s['eeg_end_s']-s['eeg_start_s'] for s in segments),
        paired_covered_seconds=30.*paired_count,
        unavailable_filter_context_windows=sum('context' in str(w['reason']) for w in windows))


def _hb_only_windows(record,events=()):
    duration = record.manifest['array_shapes']['fnirs'][0]/10.
    count = int(np.floor((duration+1e-9)/30.))
    windows = [dict(id=f'{record.join_key}|window={i:05d}|hb={30*i:.6f}',
                 hb_start_s=float(30*i), window_seconds=30., reference_seconds=5.,
                 start_s=None,eeg_start_s=None,eeg_filter_start_s=None,eeg_filter_end_s=None,
                 actual_context_s=0.,status='hb_only',mode='hb_only',segment_index=None,
                 reason='no_complete_admitted_paired_clock_support',task=None,condition=None,
                 wrong_pairing_eligible=False)
            for i in range(count)]
    segments = continuous_hb_support_segments(record,events)
    for w in windows:
        matches = [s for s in segments if w['hb_start_s']>=s['start_s']-1e-9
                   and w['hb_start_s']+30<=s['end_s']+1e-9]
        s = matches[0] if len(matches)==1 else None
        w.update(hb_support_admitted=s is not None,hb_segment_index=s['index'] if s else None,
                 native_support_id=s['id'] if s else None)
        if s is None:
            w.update(status='unavailable',mode='unavailable',reason='crosses_declared_Hb_continuity_boundary_or_gap')
    return windows


def _window_labels(window,events,dataset):
    """Conservative labels for wrong-EEG matching only; never model inputs."""
    start = window.get('eeg_start_s')
    if start is None:
        return
    candidates = sorted((e for e in events if e.get('eeg_time_ms') is not None),
                        key=lambda e:float(e['eeg_time_ms']))
    previous = [e for e in candidates if float(e['eeg_time_ms'])/1000 <= start+1e-9]
    relevant = ([previous[-1]] if previous else []) + [e for e in candidates
        if start < float(e['eeg_time_ms'])/1000 < start+30-1e-9]
    if not previous or not relevant:
        return
    labels = [canonical_label(e,dataset) for e in relevant]
    tasks = {l['task'] for l in labels}
    conditions = {l['condition'] for l in labels}
    window['task'] = next(iter(tasks)) if len(tasks)==1 and 'unknown' not in tasks else None
    if window['task'] is not None and len(conditions)==1 and not conditions.intersection({'unknown','','None'}):
        window['condition'] = next(iter(conditions))
        window['wrong_pairing_eligible'] = True


def prepared_tak_hb_window(prepared, window):
    """Select and reference a Hb-only window from the one retained record array."""
    values = np.asarray(prepared['continuous_hb'], float)
    times = np.asarray(prepared['continuous_hb_time_s'], float)
    valid = np.asarray(prepared['continuous_hb_valid_mask'], bool)
    start = float(window['hb_start_s'])
    take = (times >= start-1e-9) & (times < start+30.-1e-9)
    local = times[take]-start
    groups = np.floor(np.maximum(local,0)*10+1e-8).astype(int)
    if (len(groups) != 300 or not np.array_equal(groups,np.arange(300))
            or valid.shape != values.shape):
        raise ValueError('complete cached Hb-only 30 s support required')
    supported = (valid[take] & np.isfinite(values[take])).all(axis=0)
    if not window.get('hb_support_admitted',True):
        supported[:] = False
    selected = values[take]-values[take][:50].mean(axis=0)
    return dict(hb=np.where(supported[None],selected,np.nan),
                valid_mask=np.broadcast_to(supported,selected.shape),
                time_s=np.arange(300)/10., hb_bin_phase_s=float(local[0]))


def inventory_tak_mismatch(cfg, project_root):
    """Enumerate all registered records/pairs and subject folds without arrays."""
    data, datasets, seed, folds = _settings(cfg)
    root = Path(project_root)
    cache = root / data.get('cache_root', DEFAULT_CLEAN_CACHE_ROOT)
    index = CleanPhysiologyCacheIndex(cache)
    default_geometry = ChannelGeometryIndex(cache, geometry_root=(root / data['geometry_root']) if data.get('geometry_root') else None)
    geometries = {ds: default_geometry for ds in datasets}
    if 'refed' in datasets:
        refed_root = data.get('refed_geometry_root', data.get('geometry_roots', {}).get('refed'))
        if not refed_root:
            raise ValueError('explicit REFED paper-layout geometry root required')
        geometries['refed'] = ChannelGeometryIndex(cache, geometry_root=root / refed_root)
    records = [r for r in index.records if r.dataset_id in datasets
               and (r.dataset_id != 'refed' or r.signal_branch == 'hbo_hbr')]
    subject_folds = {}
    for ds in datasets:
        subjects = sorted({r.canonical_subject_id for r in records if r.dataset_id == ds})
        shuffled = np.random.default_rng(seed).permutation(subjects).tolist()
        subject_folds[ds] = {s: i % folds for i, s in enumerate(shuffled)}
    planned = []
    for record in sorted(records, key=lambda r: (r.dataset_id, r.canonical_subject_id, r.base_record_id)):
        meta = record.manifest['measurement']
        row = dict(id=record.join_key, join_key=record.join_key, dataset=record.dataset_id,
            subject=record.canonical_subject_id, record=record.base_record_id, cache_record_id=record.record_id,
            fold=subject_folds[record.dataset_id][record.canonical_subject_id],
            source_identity=deepcopy(record.manifest['source_files']),
            eeg_channel_names=list(meta['eeg_channel_names']), hb_channel_names=list(meta['fnirs_channel_names']),
            array_shapes=deepcopy(record.manifest['array_shapes']), status='unavailable', windows=[], pairs=[],
            hb_only_admitted=True)
        try:
            _registered_sources(record, root)
            row['continuous_hb_segments'] = continuous_hb_support_segments(record,index.events_by_join_key.get(record.join_key,[]))
            if record.dataset_id == 'visual_cognitive_motivation' and record.canonical_subject_id == 'S06' and 'part1' in record.base_record_id.lower():
                row['hb_only_admitted'] = False
                raise ValueError('excluded_Visual_S06_Part1')
            gi = geometries[record.dataset_id]
            geometry = dict(eeg=gi.for_channels(record=record, modality='eeg', channel_names=row['eeg_channel_names']),
                            fnirs=gi.for_channels(record=record, modality='fnirs', channel_names=row['hb_channel_names']))
            row['geometry_identity'] = geometry
            row['pairs'] = _pair_inventory(geometry,row['hb_channel_names'],seed=seed,record=record)
            if record.dataset_id == 'refed' and any(p.get('coordinate_system') != 'refed_paper_layout_2d_v1' for p in row['pairs'] if p['status'] == 'available'):
                raise ValueError('REFED requires its admitted paper-layout frame')
            coverage = continuous_tak_support(record, index.events_by_join_key.get(record.join_key, []),
                                             index.reports_by_join_key.get(record.join_key, []))
            row['windows'] = coverage.pop('windows')
            row['coverage'] = coverage
            row['status'] = coverage['status']
            row['excluded_reason'] = coverage['reason']
            if not any(p['status']=='available' for p in row['pairs']):
                row['status'] = 'hb_only'
                row['excluded_reason'] = 'unavailable_EEG_Hb_pair_geometry'
        except (ValueError, KeyError, TypeError) as exc:
            row['excluded_reason'] = str(exc)
        # The complete native pair denominator survives unavailable geometry or
        # timing. A numerical failure cannot silently reduce the population.
        if not row['pairs']:
            row['pairs'] = [dict(id=n[:-4], name=n[:-4], native_index=i//2,
                component_indices=[i,i+1], channel_sets={}, status='unavailable',
                reason=row.get('excluded_reason', 'unavailable_geometry')) for i,n in enumerate(row['hb_channel_names']) if i%2 == 0]
        if row['hb_only_admitted'] and row['status'] != 'available':
            row['status'] = 'hb_only'
            if not row['windows']:
                row['windows'] = _hb_only_windows(record,index.events_by_join_key.get(record.join_key,[]))
        row['hb_only_tail_seconds'] = (record.manifest['array_shapes']['fnirs'][0]/10.
                                      - 30*len(row['windows']))
        planned.append(row)
    coordinate_groups = require_fixed_tak_coordinate_groups(planned)
    summary = {}
    for ds in datasets:
        rr = [r for r in planned if r['dataset'] == ds]
        summary[ds] = dict(subjects=len(subject_folds[ds]), registered_records=len(rr),
            available_records=sum(r['status'] == 'available' for r in rr),
            Hb_admitted_records=sum(r['hb_only_admitted'] for r in rr),
            planned_pair_records=sum(len(r['pairs']) for r in rr),
            planned_windows=sum(len(r['windows']) for r in rr),
            planned_pair_windows=sum(len(r['windows'])*len(r['pairs']) for r in rr),
            Hb_admissible_pair_windows=sum(sum(w['hb_support_admitted'] for w in r['windows'])*len(r['pairs']) for r in rr),
            available_pair_windows=sum(sum(w['status']=='available' for w in r['windows'])*sum(p['status']=='available' for p in r['pairs']) for r in rr),
            paired_support_seconds=sum(r.get('coverage',{}).get('paired_support_seconds',0) for r in rr),
            tail_uncovered_seconds=sum(r.get('coverage',{}).get('tail_seconds',0) for r in rr))
    return dict(schema=SCHEMA, status='available', cache_root=str(cache), records=planned,
        datasets=list(datasets), subject_folds=subject_folds, folds=folds, seed=seed,
        summary=summary, measured_arrays_read=0,
        coordinate_groups=coordinate_groups,
        window_selection='unique_native_Hb_30s_grid_with_optional_admitted_EEG_clock_support',
        interpretation='public_development_cross_subject_descriptive_reanalysis',
        fnirs_input=CACHED_FNIRS_INPUT_SCHEMA, eeg_proxy_input=EEG_PROXY_INPUT_SCHEMA)


def prepare_tak_mismatch_record(project_root, record_plan, *, cache_root=DEFAULT_CLEAN_CACHE_ROOT,
                                geometry_root=None, refed_geometry_root=None, seed=20261028,
                                record=None, index=None):
    """Read one central record and share arrays across all Hb pairs/candidates.

    Every frozen source, geometry, clock segment and window is revalidated
    before either signal reader is called. Optional index/record arguments let
    record-group workers reuse metadata; they do not bypass these checks.
    """
    root = Path(project_root)
    if index is None:
        index = CleanPhysiologyCacheIndex(root / cache_root)
    matches = [r for r in index.records_by_join_key.get(record_plan['join_key'], [])
               if r.record_id == record_plan['cache_record_id']]
    if len(matches) != 1 or (record is not None and record != matches[0]):
        raise ValueError('frozen canonical cache record unavailable or changed')
    record = matches[0]
    if (record.dataset_id not in RAW_DATASET_IDS or record_plan['dataset'] != record.dataset_id
            or record_plan['subject'] != record.canonical_subject_id
            or record_plan['record'] != record.base_record_id
            or record_plan['source_identity'] != record.manifest['source_files']
            or record_plan['array_shapes'] != record.manifest['array_shapes']):
        raise ValueError('frozen source or canonical identity changed before signal access')
    _registered_sources(record, root)
    geo_root = refed_geometry_root if record.dataset_id == 'refed' else geometry_root
    if record.dataset_id == 'refed' and not geo_root:
        raise ValueError('explicit admitted REFED geometry required before signal access')
    gi = ChannelGeometryIndex(root / cache_root, geometry_root=root / geo_root if geo_root else None)
    meta = record.manifest['measurement']
    geometry = dict(eeg=gi.for_channels(record=record, modality='eeg', channel_names=meta['eeg_channel_names']),
                    fnirs=gi.for_channels(record=record, modality='fnirs', channel_names=meta['fnirs_channel_names']))
    support = continuous_tak_support(record, index.events_by_join_key.get(record.join_key, []),
                                     index.reports_by_join_key.get(record.join_key, []))
    windows = support.pop('windows')
    if (record_plan.get('status') not in ('available', 'hb_only') or not record_plan.get('hb_only_admitted')
            or geometry != record_plan.get('geometry_identity')
            or _pair_inventory(geometry,meta['fnirs_channel_names'],seed=seed,record=record) != record_plan['pairs']
            or continuous_hb_support_segments(record,index.events_by_join_key.get(record.join_key,[])) != record_plan.get('continuous_hb_segments')
            or windows != record_plan['windows'] or support != record_plan['coverage']):
        raise ValueError('frozen geometry or continuous support changed before signal access')
    hb = load_cached_fnirs_record(root, record)
    if list(hb['channel_names']) != list(meta['fnirs_channel_names']):
        raise ValueError('central cached Hb identities differ from the frozen record')
    hb_values = np.asarray(hb['values'], float)
    hb_times = np.asarray(hb['time_s'], float)
    if (hb_values.shape != (record.manifest['array_shapes']['fnirs'][0],len(record_plan['pairs']),2)
            or hb_times.shape != (len(hb_values),)
            or not np.allclose(hb_times,np.arange(len(hb_values))/10.,rtol=0,atol=1e-10)
            or float(hb['sample_rate_hz']) != 10.):
        raise ValueError('cached Hb must preserve the exact registered regular 10 Hz clock')
    observation_valid = np.asarray(hb['valid_mask'],bool) & np.isfinite(hb_values)
    gap_valid = np.zeros(len(hb_times),bool)
    for segment in record_plan['continuous_hb_segments']:
        gap_valid |= (hb_times>=segment['start_s']-1e-9)&(hb_times<segment['end_s']-1e-9)
    valid_hb = observation_valid & gap_valid[:,None,None]
    count, channels, pairs = len(windows), len(meta['eeg_channel_names']), len(record_plan['pairs'])
    eeg_features = np.full((count,300,channels,5), np.nan)
    target_hb = np.full((count,300,pairs,2), np.nan)
    valid_eeg = np.zeros((count,300,channels), bool)
    target_hb_valid = np.zeros_like(target_hb, bool)
    failures, contexts = [], {}
    eeg_context_sample_bounds = np.full((count,2),-1,dtype=np.int64)
    eeg_context_rounding = np.full((count,2),np.nan)
    eeg_target_rounding = np.full(count,np.nan)
    eeg = None
    if any(w['status'] == 'available' for w in windows) and any(p['status']=='available' for p in record_plan['pairs']):
        try:
            eeg = load_native_eeg_record(root, record)
            if (list(eeg.channel_names) != list(meta['eeg_channel_names'])
                    or np.shape(eeg.values)[1] != channels
                    or not np.isfinite(eeg.sample_rate_hz) or eeg.sample_rate_hz <= 90
                    or not np.isclose(eeg.sample_rate_hz/10., round(eeg.sample_rate_hz/10.))):
                raise ValueError('central native EEG clocks or identities differ from the frozen record')
            if str(Path(eeg.source_path).resolve()) not in _registered_sources(record, root):
                raise ValueError('native EEG reader source differs from the registered source identity')
        except (ValueError, KeyError, OSError) as exc:
            eeg = None
            failures.append(dict(window_id=None, modality='EEG', reason=str(exc)))
    hb_bin_phase = np.full(count, np.nan)
    for i, window in enumerate(windows):
        # Hb-only fits remain available when the independent EEG preparation
        # fails. The complete continuous Hb copy is shared by weak diagnostics.
        try:
            if not window['hb_support_admitted']:
                continue
            hs = window['hb_start_s']
            take = (hb_times >= hs-1e-9) & (hb_times < hs+30.-1e-9)
            times = hb_times[take]-hs
            groups = np.floor(np.maximum(times,0)*10+1e-8).astype(int)
            if len(groups) != 300 or not np.array_equal(groups,np.arange(300)):
                raise ValueError('every Hb bin requires one actual cached 10 Hz sample')
            values = hb_values[take]
            good_hb = valid_hb[take].all(axis=0)
            values = values-values[:50].mean(axis=0)
            target_hb[i] = np.where(good_hb[None,:,:], values, np.nan)
            target_hb_valid[i] = np.broadcast_to(good_hb,target_hb[i].shape)
            hb_bin_phase[i] = float(times[0])
        except (ValueError, IndexError, FloatingPointError) as exc:
            failures.append(dict(window_id=window['id'], modality='Hb', reason=str(exc)))
        if window['status'] != 'available':
            continue
        if eeg is None:
            continue
        try:
            a, b = (int(round(window[k]*eeg.sample_rate_hz)) for k in ('eeg_filter_start_s','eeg_filter_end_s'))
            eeg_context_sample_bounds[i] = (a,b)
            eeg_context_rounding[i] = (a/eeg.sample_rate_hz-window['eeg_filter_start_s'],
                                       b/eeg.sample_rate_hz-window['eeg_filter_end_s'])
            context = (a,b)
            if context not in contexts:
                voltage = np.asarray(eeg.values[a:b], float)
                if len(voltage) != b-a or not len(voltage) or (b-a) % round(eeg.sample_rate_hz/10.):
                    raise ValueError('incomplete native EEG filtering support')
                finite = np.isfinite(voltage).all(axis=0)
                power = np.full((round(window['actual_context_s']*10), channels,5), np.nan)
                if finite.any():
                    power[:,finite] = eeg_band_power(voltage[:,finite], bands=BANDS_HZ,
                        sample_rate=eeg.sample_rate_hz, target_rate=10.)
                # Neighboring windows often share a boundary-clamped context.
                # Bound the small cache; complete raw records are not duplicated.
                if len(contexts) >= 2:
                    contexts.pop(next(iter(contexts)))
                contexts[context] = power
            crop = round((window['eeg_start_s']-window['eeg_filter_start_s'])*10)
            eeg_target_rounding[i] = a/eeg.sample_rate_hz+crop/10.-window['eeg_start_s']
            power = contexts[context][crop:crop+300]
            if power.shape != (300,channels,5):
                raise ValueError('30 s target outside its declared EEG context')
            good = np.isfinite(power).all(axis=(0,2)) & (power>0).all(axis=(0,2))
            features = np.log(power[:,good])
            features -= features[:50].mean(axis=0)
            eeg_features[i][:,good,:] = features
            valid_eeg[i][:,good] = True
        except (ValueError, IndexError, FloatingPointError) as exc:
            failures.append(dict(window_id=window['id'], modality='EEG', reason=str(exc)))
    return dict(eeg_logpower=eeg_features, hb=target_hb,
        eeg_valid_mask=valid_eeg, hb_valid_mask=target_hb_valid,
        time_s=np.arange(300)/10., window_ids=np.asarray([w['id'] for w in windows]),
        continuous_hb=hb_values, continuous_hb_time_s=hb_times,
        continuous_hb_valid_mask=valid_hb,
        continuous_hb_observation_valid_mask=observation_valid,
        continuous_hb_gap_valid_mask=gap_valid,
        hb_bin_phase_s=hb_bin_phase,
        eeg_native_context_sample_bounds=eeg_context_sample_bounds,
        eeg_context_rounding_s=eeg_context_rounding,
        eeg_target_start_rounding_s=eeg_target_rounding,
        eeg_proxy_input=np.array(EEG_PROXY_INPUT_SCHEMA), eeg_proxy_rate_hz=np.array(10.),
        fnirs_input=np.array(CACHED_FNIRS_INPUT_SCHEMA),
        metadata=dict(schema=SCHEMA, record_id=record.join_key, source_identity=record_plan['source_identity'],
            hb_provenance=deepcopy(hb['provenance']), eeg_native_unit=getattr(eeg,'native_unit',None),
            eeg_unit_evidence=getattr(eeg,'unit_evidence',None),
            eeg_native_sample_rate_hz=float(eeg.sample_rate_hz) if eeg is not None else None,
            eeg_reference=getattr(eeg,'reference',None), prepared_windows=count, failures=failures,
            window_actual_context_s=[w['actual_context_s'] for w in windows],
            continuous_hb_segments=record_plan['continuous_hb_segments'],
            reference_seconds=5., bands_hz=[list(b) for b in BANDS_HZ],
            hb_bin_phase='one_cached_sample_per_100ms_bin; preserve_phase_in_model_observation_operator',
            input_support='native_EEG_per_declared_context; full_record_offline_bandpassed_cached_Hb',
            last_bin='excluded_from_fitting_and_scoring_by_consumer',
            scale_policy='unscaled_prepared_arrays; PCA_common_positive_Hb_scale_and_SD_fit_outer_training_subjects_only'))
