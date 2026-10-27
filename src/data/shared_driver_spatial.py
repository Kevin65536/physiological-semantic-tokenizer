"""Metadata-frozen public spatial patches and cached Hb measurements.

The v1 surface admits Single-Trial only. Subject assignment and all spatial
identities are resolved without loading signal arrays. Hb pair zero is hidden
throughout the window; its own reference defines the scoring coordinate only.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .unified_physiology import (
    UnifiedPhysiologyWindowDataset, canonical_channel_name, canonical_label,
    load_native_eeg_record, load_cached_fnirs_record, CACHED_FNIRS_INPUT_SCHEMA,
)
from .registry import REGISTERED_DATASETS
from .clean_physiology_cache import canonical_subject_id, join_key
from src.inference.observation_baselines import (
    disjoint_native_bin_operator, eeg_band_power,
)


DATASETS = ('eeg_fnirs_single_trial',)
SPLITS = ('train', 'selection', 'evaluation')
DEFAULT_BANDS = ((1., 4.), (4., 8.), (8., 13.), (13., 30.), (30., 45.))
OPTODE_POLICY = 'exclude_target_shared_optodes_record_donor_donor_sharing'


def _settings(cfg):
    data, tensor = cfg['data'], cfg.get('tensor', {})
    if tuple(data.get('datasets', ())) != DATASETS or data.get('protected_data') != 'forbidden':
        raise ValueError('spatial v1 admits ordinary public Single-Trial only')
    if data.get('region', 'prefrontal') != 'prefrontal':
        raise ValueError('spatial v1 uses one prefrontal patch')
    if data.get('optode_policy', OPTODE_POLICY) != OPTODE_POLICY:
        raise ValueError('spatial v1 excludes target-shared optodes and records donor sharing')
    pairs, channels = int(data.get('hb_pairs', 4)), int(data.get('eeg_channels', 12))
    if pairs not in (2, 4) or channels not in (6, 12):
        raise ValueError('spatial v1 requires 2/4 Hb pairs and 6/12 EEG channels')
    dt = float(tensor.get('dt_s', .25))
    duration = float(data.get('window_seconds', 30.))
    reference = float(data.get('reference_seconds', 5.))
    if not np.isfinite([dt, duration, reference]).all() or min(dt, duration, reference) <= 0:
        raise ValueError('finite positive time support required')
    steps, reference_steps = round(duration / dt), round(reference / dt)
    if (dt != .25 or not np.isfinite([duration, reference]).all()
            or not np.isclose(steps * dt, duration) or not np.isclose(reference_steps * dt, reference)
            or not 1 <= reference_steps < steps or int(tensor.get('steps', steps)) != steps
            or int(tensor.get('reference_steps', reference_steps)) != reference_steps):
        raise ValueError('complete 4 Hz window and nonempty visible reference required')
    bands = np.asarray(tensor.get('eeg_bands_hz', DEFAULT_BANDS), float)
    if bands.shape != (5, 2) or not np.isfinite(bands).all() or np.any(bands[:, 0] <= 0) or np.any(bands[:, 1] >= 100) or np.any(bands[:, 0] >= bands[:, 1]):
        raise ValueError('five valid native EEG bands required')
    fractions = [float(data.get('train_fraction', .6)), float(data.get('selection_fraction', .2))]
    if not np.isfinite(fractions).all() or min(fractions) <= 0 or sum(fractions) >= 1:
        raise ValueError('nonempty train/selection/evaluation fractions required')
    limits = [int(data.get('max_records_per_subject', 2)), int(data.get('max_windows_per_record', 3))]
    if min(limits) < 1:
        raise ValueError('positive deterministic metadata limits required')
    offset = float(data.get('window_offset_s', -5.))
    if not np.isfinite(offset):
        raise ValueError('finite event-relative window offset required')
    max_angle = float(data.get('max_patch_angle_deg', 45.))
    if not np.isfinite(max_angle) or not 0 < max_angle <= 90:
        raise ValueError('engineering patch angle must be in (0,90] degrees')
    return dict(pairs=pairs, channels=channels, dt=dt, steps=steps,
                duration=duration, reference=reference, reference_steps=reference_steps,
                offset=offset, bands=bands.tolist(),
                fractions=fractions, limits=limits,
                seed=int(data.get('split_seed', cfg.get('seed', 0))),
                max_subjects=int(data.get('max_subjects', 12)), max_angle=max_angle)


def _positions(rows):
    try:
        coordinates = np.asarray([[r.get(k) for k in ('x', 'y', 'z')] for r in rows], float)
    except (TypeError, ValueError) as exc:
        raise ValueError('complete known geometry required') from exc
    if coordinates.shape != (len(rows), 3) or not np.isfinite(coordinates).all():
        raise ValueError('complete known geometry required')
    return coordinates


def _hb_pairs(rows):
    grouped = {}
    for index, row in enumerate(rows):
        name = canonical_channel_name(row.get('channel_name', ''))
        role = name.rsplit('_', 1)[-1]
        base = name.rsplit('_', 1)[0]
        if role not in ('HbO', 'HbR') or not base or row.get('component', role) != role or row.get('base_channel_name', base) != base:
            raise ValueError('explicit matching HbO/HbR channel identities required')
        if role in grouped.setdefault(base, {}):
            raise ValueError('duplicate Hb component identity')
        grouped[base][role] = index
    output = []
    for base, components in grouped.items():
        if set(components) != {'HbO', 'HbR'}:
            raise ValueError('complete HbO/HbR pair required')
        oi, ri = components['HbO'], components['HbR']
        o, r = rows[oi], rows[ri]
        if (not np.array_equal(_positions([o]), _positions([r]))
                or any(o.get(k) != r.get(k) for k in ('coordinate_units', 'coordinate_system', 'source_index', 'detector_index'))):
            raise ValueError('HbO/HbR pair geometry or hardware identity mismatch')
        hardware = [o.get(k) for k in ('source_index', 'detector_index')]
        if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 0 for x in hardware):
            raise ValueError('verified source/detector identities required for every Hb pair')
        output.append(dict(name=base, components=[oi, ri], row=o,
                           source=int(hardware[0]), detector=int(hardware[1])))
    return output


def select_spatial_patch(eeg_rows: Sequence[Mapping[str, Any]],
                         hb_rows: Sequence[Mapping[str, Any]], target_index: int,
                         *, hb_pairs=4, eeg_channels=12, disjoint_optodes=True,
                         max_patch_angle_deg=45.) -> dict:
    """Select nested nearest-channel patches without signal- or label-based choice.

    ``target_index`` indexes Hb pairs in first-appearance order. Disjoint mode
    admits nearest donors sharing neither a source nor a detector with the
    target. Donors may share optodes with each other; their actual edges remain
    explicit in the returned support. Source and detector numbering are separate.
    The first 2 pairs and first 6 EEG channels are prefixes of the larger patch.
    """
    if hb_pairs not in (2, 4) or eeg_channels not in (6, 12):
        raise ValueError('requires 2/4 Hb pairs and 6/12 EEG channels')
    pairs = _hb_pairs(hb_rows)
    if not isinstance(target_index, (int, np.integer)) or not 0 <= target_index < len(pairs):
        raise ValueError('target must identify an existing Hb pair')
    names = [canonical_channel_name(r.get('channel_name', '')) for r in eeg_rows]
    if any(not n for n in names) or len({n.casefold() for n in names}) != len(names):
        raise ValueError('EEG channel identities must be nonempty and unique')
    rows = list(eeg_rows) + list(hb_rows)
    units = {r.get('coordinate_units') for r in rows}
    systems = {r.get('coordinate_system') for r in rows}
    if (len(units) != 1 or next(iter(units), None) != 'normalized_head_unit'
            or len(systems) != 1 or next(iter(systems), None) in (None, '', 'unknown', 'unavailable')):
        raise ValueError('one known coordinate system and normalized_head_unit geometry required')
    e, h = _positions(eeg_rows), _positions([p['row'] for p in pairs])
    anchor = h[target_index]
    norms = np.linalg.norm(h, axis=1)
    if (np.any(norms <= 0) or not np.isfinite(max_patch_angle_deg)
            or not 0 < max_patch_angle_deg <= 90):
        raise ValueError('nonzero directional geometry and valid engineering patch angle required')
    cosines = (h @ anchor) / (norms * norms[target_index])
    local = [i for i in range(len(pairs)) if cosines[i] >= np.cos(np.deg2rad(max_patch_angle_deg)) - 1e-12]
    rank = sorted(local, key=lambda i: (float(np.linalg.norm(h[i] - anchor)), pairs[i]['name']))
    selected = [int(target_index)]
    target_source, target_detector = pairs[target_index]['source'], pairs[target_index]['detector']
    for index in rank:
        if index == target_index:
            continue
        pair = pairs[index]
        if disjoint_optodes and (pair['source'] == target_source or pair['detector'] == target_detector):
            continue
        selected.append(index)
        if len(selected) == hb_pairs:
            break
    if len(selected) != hb_pairs:
        raise ValueError('insufficient Hb pairs with the declared optode support')
    eeg_indices = sorted(range(len(e)), key=lambda i: (float(np.linalg.norm(e[i] - anchor)), names[i]))[:eeg_channels]
    if len(eeg_indices) != eeg_channels:
        raise ValueError('insufficient unique EEG channel support')
    # Two fixed spatial anchors, supported by the nearest six selected EEG
    # channels each. They are geometry descriptors, not fitted source truth.
    anchors = [0, len(selected) - 1]
    support = np.zeros((2, eeg_channels), bool)
    for k, index in enumerate(anchors):
        nearest = np.argsort(np.linalg.norm(e[eeg_indices] - h[selected[index]], axis=1), kind='stable')[:6]
        support[k, nearest] = True
    weights = support.astype(float) / support.sum(axis=1, keepdims=True)
    sources = np.asarray([pairs[i]['source'] for i in selected])
    detectors = np.asarray([pairs[i]['detector'] for i in selected])
    shared_source = sources[:, None] == sources[None, :]
    shared_detector = detectors[:, None] == detectors[None, :]
    np.fill_diagonal(shared_source, False)
    np.fill_diagonal(shared_detector, False)
    adjacency = shared_source | shared_detector
    donor_edges = [dict(pair_indices=[i, j],
                        source_index=int(sources[i]) if shared_source[i, j] else None,
                        detector_index=int(detectors[i]) if shared_detector[i, j] else None)
                   for i in range(1, hb_pairs) for j in range(i + 1, hb_pairs) if adjacency[i, j]]
    patch = dict(schema='shared_driver_spatial_patch_v1', target_index=0,
                 native_target_index=int(target_index), hb_pair_indices=selected,
                 hb_component_indices=[pairs[i]['components'] for i in selected],
                 hb_pair_names=[pairs[i]['name'] for i in selected],
                 hb_channel_names=[[hb_rows[j]['channel_name'] for j in pairs[i]['components']] for i in selected],
                 source_indices=sources.tolist(), detector_indices=detectors.tolist(),
                 shared_source_adjacency=shared_source.tolist(),
                 shared_detector_adjacency=shared_detector.tolist(),
                 shared_optode_adjacency=adjacency.tolist(),
                 donor_shared_optode_adjacency=adjacency[1:, 1:].tolist(),
                 donor_shared_optode_support=donor_edges,
                 eeg_indices=eeg_indices, eeg_channel_names=[names[i] for i in eeg_indices],
                 eeg_coordinates=e[eeg_indices].tolist(), hb_coordinates=h[selected].tolist(),
                 anchor_pair_indices=anchors, anchor_coordinates=h[np.asarray(selected)[anchors]].tolist(),
                 anchor_eeg_support=support.tolist(), anchor_eeg_weights=weights.tolist(),
                 coordinate_units=next(iter(units)), coordinate_system=next(iter(systems)),
                 max_patch_angle_deg=float(max_patch_angle_deg),
                 angle_interpretation='engineering_template_neighbourhood_not_physiological_distance',
                 disjoint_optodes=bool(disjoint_optodes),
                 optode_policy=OPTODE_POLICY if disjoint_optodes else 'retain_all_optode_sharing',
                 selection='target_first_then_nearest_target_disjoint_donors; nearest_unique_EEG' if disjoint_optodes else 'target_first_then_nearest_donors; nearest_unique_EEG',
                 anchor_interpretation='target_and_farthest_selected_donor_template_sites')
    identity = {k: patch[k] for k in ('hb_pair_names', 'eeg_channel_names', 'source_indices', 'detector_indices', 'coordinate_system')}
    patch['patch_id'] = 'prefrontal__' + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return patch


def _partitions(subjects, settings):
    subjects = sorted(set(subjects))
    if len(subjects) < 5:
        raise ValueError('at least five public subjects required for grouped partitions')
    permutation = np.random.default_rng(settings['seed']).permutation(subjects)
    a = int(len(subjects) * settings['fractions'][0])
    b = int(len(subjects) * sum(settings['fractions']))
    if not 0 < a < b < len(subjects):
        raise ValueError('all subject partitions must be nonempty')
    return {str(s): split for split, ids in zip(SPLITS, (permutation[:a], permutation[a:b], permutation[b:])) for s in ids}


def _geometry(record, data):
    measurement = record.manifest['measurement']
    return (data.geometry_index.for_channels(record=record, modality='eeg', channel_names=measurement['eeg_channel_names']),
            data.geometry_index.for_channels(record=record, modality='fnirs', channel_names=measurement['fnirs_channel_names']))


def _sources(record, project_root):
    """Validate the complete registered source scope without opening raw files."""
    fields = (record.dataset_id, record.subject, record.canonical_subject_id,
              record.record_id, record.base_record_id, record.join_key)
    if any(not isinstance(v, str) or not v.strip() for v in fields):
        raise ValueError('complete canonical record identity required')
    if (canonical_subject_id(record.dataset_id, record.subject) != record.canonical_subject_id
            or join_key(record.dataset_id, record.subject, record.record_id) != record.join_key
            or record.record_id != record.base_record_id):
        raise ValueError('registered canonical record identity mismatch')
    identity = record.manifest.get('source_files')
    if not isinstance(identity, list) or not identity:
        raise ValueError('registered source identity is missing')
    admitted_root = (project_root / REGISTERED_DATASETS[record.dataset_id].default_root).resolve()
    paths = []
    for source in identity:
        if not isinstance(source, Mapping) or not source.get('path'):
            raise ValueError('complete registered source identity required')
        path = Path(source['path'])
        path = (path if path.is_absolute() else project_root / path).resolve()
        if not path.is_relative_to(admitted_root):
            raise ValueError('native source outside registered public dataset root')
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise ValueError('duplicate registered source path')
    subject = record.canonical_subject_id.replace('subject_', 'subject ')
    eeg_dir = admitted_root / 'EEG_01-29' / subject / 'with occular artifact'
    if not eeg_dir.exists():
        eeg_dir = eeg_dir.parent
    if (eeg_dir / 'cnt.mat').resolve() not in paths:
        raise ValueError('native EEG reader source absent from registered source identity')
    optical = [p for p in paths if p.name == 'cnt.mat' and 'NIRS_01-29' in p.parts]
    if len(optical) != 1 or subject not in optical[0].parts:
        raise ValueError('unique subject-matched native optical source required')
    return paths


def inventory_spatial(cfg, project_root) -> dict:
    """Freeze metadata identities, splits and geometry without signal-array access."""
    settings, root = _settings(cfg), Path(project_root)
    kwargs = dict(dataset_ids=DATASETS, window_duration_s=settings['duration'], window_offset_s=settings['offset'])
    if cfg['data'].get('geometry_root'):
        kwargs['geometry_root'] = root / cfg['data']['geometry_root']
    data = UnifiedPhysiologyWindowDataset(root / cfg['data']['cache_root'], **kwargs)
    refs = list(data.windows)
    if any(r.record.dataset_id not in DATASETS for r in refs):
        raise ValueError('metadata inventory contains a nonadmitted dataset')
    rows, excluded, patches, geometries = [], [], {}, {}
    record_count, window_count, intervals = {}, {}, {}
    for ref in sorted(refs, key=lambda r: (r.record.canonical_subject_id, r.record.base_record_id,
                      float(r.event['eeg_time_ms']), float(r.event['fnirs_time_ms']), int(r.event['event_index']))):
        record, event = ref.record, ref.event
        identity = dict(dataset=record.dataset_id, subject=record.canonical_subject_id,
                        record=record.base_record_id, join_key=record.join_key,
                        event_id=int(event['event_index']))
        try:
            _sources(record, root)
            if record.join_key not in patches:
                eeg_rows, hb_rows = _geometry(record, data)
                pairs = _hb_pairs(hb_rows)
                xyz = _positions([p['row'] for p in pairs])
                norms = np.linalg.norm(xyz, axis=1)
                if np.any(norms <= 0):
                    raise ValueError('nonzero prefrontal directional geometry required')
                direction = xyz[:, 1] / norms
                target = min(range(len(pairs)), key=lambda i: (-direction[i], pairs[i]['name']))
                if direction[target] < np.cos(np.deg2rad(50)):
                    raise ValueError('prefrontal anatomical support unavailable')
                patches[record.join_key] = select_spatial_patch(eeg_rows, hb_rows, target,
                    hb_pairs=settings['pairs'], eeg_channels=settings['channels'],
                    max_patch_angle_deg=settings['max_angle'])
                geometries[record.join_key] = dict(eeg=eeg_rows, fnirs=hb_rows)
            shape = record.manifest['array_shapes']
            es = float(event['eeg_time_ms']) / 1000 + settings['offset']
            hs = float(event['fnirs_time_ms']) / 1000 + settings['offset']
            if (not np.isfinite([es, hs]).all() or min(es, hs) < 0
                    or es + settings['duration'] > shape['eeg'][0] / 200
                    or hs + settings['duration'] > shape['fnirs'][0] / float(record.sample_rate_hz)):
                raise ValueError('incomplete native window support')
            admitted = record_count.setdefault(record.canonical_subject_id, set())
            if record.join_key not in admitted and len(admitted) >= settings['limits'][0]:
                raise ValueError('deterministic per_subject_record_limit')
            if window_count.get(record.join_key, 0) >= settings['limits'][1]:
                raise ValueError('deterministic per_record_window_limit')
            for previous_eeg, previous_hb in intervals.get(record.join_key, []):
                eeg_overlap = es < previous_eeg + settings['duration'] - 1e-9 and es + settings['duration'] > previous_eeg + 1e-9
                hb_overlap = hs < previous_hb + settings['duration'] - 1e-9 and hs + settings['duration'] > previous_hb + 1e-9
                if eeg_overlap or hb_overlap:
                    raise ValueError('overlapping_native_window_support')
            admitted.add(record.join_key)
            window_count[record.join_key] = window_count.get(record.join_key, 0) + 1
            intervals.setdefault(record.join_key, []).append((es, hs))
            label = canonical_label(event, record.dataset_id)
            patch = deepcopy(patches[record.join_key])
            id_ = f'{record.join_key}|event={identity["event_id"]}|{patch["patch_id"]}'
            rows.append(dict(identity, id=id_, trial_id=f'{record.join_key}|event={identity["event_id"]}',
                task=label['task'], condition=label['condition'], region='prefrontal',
                eeg_start_s=es, hb_start_s=hs, window_seconds=settings['duration'],
                reference_seconds=settings['reference'], patch=patch,
                source_identity=deepcopy(record.manifest['source_files']),
                geometry_identity=deepcopy(geometries[record.join_key]),
                alignment_support=deepcopy(event.get('metadata', {}).get('alignment_support_ms', {})),
                masking='heldout_pair_all_times_scoring_reference_hidden', target_index=0))
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            excluded.append(dict(identity, reason=str(exc)))
    common = dict(schema='shared_driver_spatial_inventory_v1', source_cache=cfg['data']['cache_root'],
        metadata_windows=len(refs), measured_arrays_read=0,
        alignment_excluded_records=deepcopy(data.excluded_alignment_records),
        alignment_excluded_windows=data.excluded_alignment_support_windows,
        scope='ordinary_public_development_subject_partitions_not_protected_test',
        window_selection='chronological_nonoverlap_both_native_clocks',
        masking='heldout_pair_all_times_scoring_reference_hidden')
    subjects = sorted({r['subject'] for r in rows})
    if len(subjects) < 5 or settings['max_subjects'] < 5:
        excluded.extend(dict(row, reason='insufficient_metadata_supported_subjects_for_partitions') for row in rows)
        return dict(common, rows=[], excluded=excluded, subjects={s: [] for s in SPLITS},
                    metadata_supported_subjects=subjects, status='unavailable',
                    reason='fewer_than_five_metadata_supported_subjects_or_subject_limit')
    selected = set(np.random.default_rng(settings['seed']).permutation(subjects)[:settings['max_subjects']])
    partitions = _partitions(selected, settings)
    admitted_rows = []
    for row in rows:
        if row['subject'] not in selected:
            excluded.append({k: row[k] for k in ('dataset', 'subject', 'record', 'join_key', 'event_id')} | dict(reason='deterministic_metadata_subject_limit'))
            continue
        row['split'] = partitions[row['subject']]
        admitted_rows.append(row)
    rows = admitted_rows
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate canonical spatial-window identity')
    return dict(common, rows=rows, excluded=excluded, status='available', reason=None,
        subjects={s: sorted(k for k, v in partitions.items() if v == s) for s in SPLITS},
        metadata_supported_subjects=subjects)


def prepare_spatial_record(cfg, project_root, row) -> dict:
    """Revalidate metadata and consume the already filtered Hb cache once."""
    if cfg['data'].get('fnirs_input') != CACHED_FNIRS_INPUT_SCHEMA:
        raise ValueError('Native-Hb spatial preparation is retired; use the cached v2 contract and a fresh run')
    settings, root = _settings(cfg), Path(project_root)
    if row.get('dataset') not in DATASETS or row.get('split') not in SPLITS:
        raise ValueError('admitted public dataset and subject split required before array access')
    plan = inventory_spatial(cfg, root)
    if plan['status'] != 'available':
        raise ValueError('spatial metadata plan unavailable before array access: ' + plan['reason'])
    matches = [r for r in plan['rows'] if r['id'] == row.get('id')]
    if len(matches) != 1 or matches[0] != row:
        raise ValueError('frozen row identity, split, source or geometry changed before array access')
    # The index and readers have their existing owners; no cache array is read.
    from .clean_physiology_cache import CleanPhysiologyCacheIndex
    index = CleanPhysiologyCacheIndex(root / cfg['data']['cache_root'])
    records = [r for r in index.records if r.join_key == row['join_key']]
    if len(records) != 1:
        raise ValueError('unique registered canonical record required')
    record = records[0]
    if (record.dataset_id != row['dataset'] or record.canonical_subject_id != row['subject']
            or record.base_record_id != row['record']
            or record.manifest.get('source_files') != row['source_identity']):
        raise ValueError('registered source identity changed before array access')
    _sources(record, root)
    eeg = load_native_eeg_record(root, record)
    hb = load_cached_fnirs_record(root, record)
    names = list(eeg.channel_names)
    if (eeg.sample_rate_hz != 200 or eeg.native_unit not in ('uV', 'µV', 'μV') or not eeg.unit_evidence
            or names != list(record.manifest['measurement']['eeg_channel_names'])):
        raise ValueError('verified native 200 Hz microvolt EEG and exact channel identities required')
    if list(hb['channel_names']) != list(record.manifest['measurement']['fnirs_channel_names']):
        raise ValueError('native Hb channel identities differ from frozen metadata')
    patch = row['patch']
    start = round(row['eeg_start_s'] * 200)
    count = round(settings['duration'] * 200)
    e = np.asarray(eeg.values[start:start + count, patch['eeg_indices']], float)
    if e.shape != (count, settings['channels']) or not np.isfinite(e).all():
        raise ValueError('complete finite native EEG support required')
    take = (hb['time_s'] >= row['hb_start_s']) & (hb['time_s'] < row['hb_start_s'] + settings['duration'])
    times = np.asarray(hb['time_s'][take], float) - row['hb_start_s']
    operators = disjoint_native_bin_operator(times, steps=settings['steps'], dt=settings['dt'], reference_steps=settings['reference_steps'])
    cached_hb = np.asarray(hb['values'][take][:, patch['hb_pair_indices']], float)
    if cached_hb.shape != (len(times), settings['pairs'], 2) or not np.isfinite(cached_hb).all():
        raise ValueError('complete finite unified-cache Hb support required')
    powers = eeg_band_power(e, bands=settings['bands'], sample_rate=200., target_rate=4.)
    features = np.einsum('ij,jcb->icb', operators['eeg'], np.log(np.maximum(powers, np.finfo(float).tiny)))
    values = (operators['raw'] @ cached_hb.reshape(len(times), -1)).reshape(settings['steps'], settings['pairs'], 2)
    target = np.zeros_like(values, dtype=bool)
    target[:, 0, :] = True
    observed = ~target
    scoring = target.copy()
    scoring[:settings['reference_steps']] = False
    return dict(eeg=features, features=features.reshape(settings['steps'], -1), hb=values,
        input_hb=np.where(observed, values, np.nan),
        hb_observed_mask=observed, target_mask=target, scoring_mask=scoring,
        operators=operators, native_time_s=times,
        metadata=dict(row=deepcopy(row), eeg_native_unit=eeg.native_unit,
            eeg_unit_evidence=eeg.unit_evidence, eeg_reference=eeg.reference,
            eeg_native_sample_rate_hz=float(eeg.sample_rate_hz),
            hb_sample_rate_hz=float(hb['sample_rate_hz']),
            eeg_signal_branch='raw_with_ocular_artifact',
            eeg_processing='trial_local_five_band_4Hz_logpower_first_reference_subtraction',
            hb_provenance=deepcopy(hb['provenance']), fnirs_input=CACHED_FNIRS_INPUT_SCHEMA,
            hb_processing='unified_cache_bandpass_0.01_0.2Hz_then_bin_means_and_reference_no_second_filter',
            hb_reference_seconds=settings['reference'],
            preprocessing_support='full_record_offline_before_masking',
            masking='heldout_pair_all_times_scoring_reference_hidden',
            target_reference='scoring_coordinate_only_never_visible_input',
            target_unit=hb['provenance']['output_unit'],
            model_coordinate='cached_measurement_coordinate_not_raw_ODE_filter_equivalence'))
