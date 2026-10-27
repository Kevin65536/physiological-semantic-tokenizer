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


def select_proxy_channel_sets(eeg_rows, hb_rows, target_index, *,
                              random_seeds=(20261027, 20261028, 20261029)):
    """Freeze single-target sensor sets using geometry and declared seeds only.

    Distances are template-coordinate descriptors, not cortical distances.
    Random donors exclude the original inner six and are sampled from the
    remaining measured electrodes. Every expanded control contains 12 unique
    electrodes and uses the same six inner electrodes.
    """
    pairs = _hb_pairs(hb_rows)
    if (isinstance(target_index, bool) or not isinstance(target_index, (int, np.integer))
            or not 0 <= target_index < len(pairs)):
        raise ValueError('target must identify an existing Hb pair')
    names = [canonical_channel_name(r.get('channel_name', '')) for r in eeg_rows]
    if (len(names) < 12 or any(not n for n in names)
            or len({n.casefold() for n in names}) != len(names)):
        raise ValueError('at least twelve uniquely named measured EEG channels required')
    rows = list(eeg_rows) + list(hb_rows)
    units, systems = ({r.get(k) for r in rows} for k in ('coordinate_units', 'coordinate_system'))
    if (units != {'normalized_head_unit'} or len(systems) != 1
            or next(iter(systems), None) in (None, '', 'unknown', 'unavailable')):
        raise ValueError('one known normalized-head geometry frame required')
    positions = _positions(eeg_rows)
    anchor = _positions([pairs[target_index]['row']])[0]
    distances = np.linalg.norm(positions - anchor, axis=1)
    order = sorted(range(len(names)), key=lambda i: (float(distances[i]), names[i]))
    inner, outer = order[:6], order[6:12]
    far = sorted(order[6:], key=lambda i: (-float(distances[i]), names[i]))[:6]
    seeds = tuple(random_seeds)
    if (len(seeds) != 3 or len(set(seeds)) != 3
            or any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s < 0 for s in seeds)):
        raise ValueError('three distinct nonnegative integer random-set seeds required')
    sets = dict(nearest3=order[:3], nearest6=inner, nearest12=order[:12],
                sixth_to_seventh=order[:5] + [order[6]], outer6=outer, far6=far,
                nearest6_far6=inner + far)
    for i, seed in enumerate(seeds):
        chosen = np.random.default_rng(int(seed)).choice(order[6:], size=6, replace=False)
        donors = sorted(map(int, chosen), key=lambda j: (float(distances[j]), names[j]))
        sets[f'random6_{i}'] = donors
        sets[f'nearest6_random6_{i}'] = inner + donors
    return dict(schema='shared_driver_sensor_proxy_support_v1',
        native_target_index=int(target_index), hb_pair_name=pairs[target_index]['name'],
        hb_component_indices=pairs[target_index]['components'],
        eeg_channel_names=names, ranked_eeg_indices=order,
        distances=distances.tolist(), sets=sets,
        set_channel_names={name: [names[i] for i in ids] for name, ids in sets.items()},
        random_seeds=list(map(int, seeds)), coordinate_units='normalized_head_unit',
        coordinate_system=next(iter(systems)),
        selection='distance_then_channel_identity; random_donors_exclude_inner6',
        distance_interpretation='registered_template_not_individual_cortical_distance')


def _sensor_proxy_record_identity(project_root, record, row, channel_sets):
    """Common identity checks for target and EEG-only donor preparation."""
    root = Path(project_root)
    if (record.dataset_id != 'eeg_fnirs_single_trial' or row.get('dataset') != record.dataset_id
            or row.get('join_key') != record.join_key
            or row.get('subject') != record.canonical_subject_id
            or row.get('record') != record.base_record_id
            or row.get('split') not in SPLITS
            or row.get('source_identity') != record.manifest.get('source_files')):
        raise ValueError('registered public record, frozen source identity and subject split required')
    if (channel_sets.get('schema') != 'shared_driver_sensor_proxy_support_v1'
            or row.get('window_seconds') != 30 or row.get('reference_seconds') != 5
            or row.get('eeg_filter_seconds') != 90):
        raise ValueError('explicit 90 s EEG filtering context, 30 s target and 5 s reference required')
    geometry = row['geometry_identity']
    expected_sets = select_proxy_channel_sets(geometry['eeg'], geometry['fnirs'],
        row['patch']['native_target_index'], random_seeds=channel_sets['random_seeds'])
    if expected_sets != channel_sets:
        raise ValueError('proxy channel support differs from the frozen target geometry')
    es, hs = (float(row[k]) for k in ('eeg_start_s', 'hb_start_s'))
    filter_start = float(row['eeg_filter_start_s'])
    crop_start = round((es-filter_start)*10)
    if (not np.isfinite([es, hs, filter_start]).all() or min(es, hs, filter_start) < 0
            or not np.isclose(crop_start/10, es-filter_start, rtol=0, atol=1e-9)
            or not 0 <= crop_start <= 600):
        raise ValueError('30 s target must lie on the 10 Hz grid within its declared 90 s EEG context')
    pair = channel_sets['native_target_index']
    meta = record.manifest['measurement']
    if (list(meta['eeg_channel_names']) != channel_sets['eeg_channel_names']
            or meta['fnirs_channel_names'][2*pair:2*pair+2] !=
               [channel_sets['hb_pair_name'] + '_HbO', channel_sets['hb_pair_name'] + '_HbR']):
        raise ValueError('frozen EEG or target Hb channel identity differs from registered record')
    _sources(record, root)
    return root, es, hs, filter_start, crop_start, pair, meta


def prepare_sensor_proxy_eeg_record(project_root, record, row, channel_sets, *,
                                   bands=DEFAULT_BANDS, eeg_record=None):
    """Prepare only EEG, optionally reusing a central-reader record in one group.

    Callers keep the preloaded object scoped to its canonical record. Identity
    checks still precede feature extraction. No Hb reader is used in this path.
    """
    from .ssm_prepared import EEG_PROXY_INPUT_SCHEMA

    root, es, hs, filter_start, crop_start, pair, meta = _sensor_proxy_record_identity(
        project_root, record, row, channel_sets)
    eeg = load_native_eeg_record(root, record) if eeg_record is None else eeg_record
    if (float(eeg.sample_rate_hz) != 200 or eeg.native_unit not in ('uV', 'µV', 'μV')
            or not eeg.unit_evidence
            or list(eeg.channel_names) != list(meta['eeg_channel_names'])):
        raise ValueError('verified native EEG units/rate and exact registered channel identities required')
    start, count = round(filter_start * eeg.sample_rate_hz), round(90 * eeg.sample_rate_hz)
    values = np.asarray(eeg.values[start:start+count], float)
    if values.shape != (count, len(eeg.channel_names)) or not np.isfinite(values).all():
        raise ValueError('complete finite 90 s native EEG support required')
    power = eeg_band_power(values, bands=bands, sample_rate=eeg.sample_rate_hz, target_rate=10.)
    if power.shape != (900, len(eeg.channel_names), 5) or not np.isfinite(power).all() or np.any(power <= 0):
        raise ValueError('finite positive native 10 Hz five-band power required')
    reference = np.eye(300)
    reference[:, :50] -= 1/50
    features = np.log(power[crop_start:crop_start+300])
    features -= features[:50].mean(axis=0)
    provenance = dict(row=deepcopy(row), channel_sets=deepcopy(channel_sets),
        eeg_proxy_input=EEG_PROXY_INPUT_SCHEMA,
        eeg_native_unit=eeg.native_unit, eeg_unit_evidence=eeg.unit_evidence,
        eeg_native_sample_rate_hz=float(eeg.sample_rate_hz),
        eeg_reference=eeg.reference, native_start_sample=start, native_samples=count,
        start_rounding_s=start/eeg.sample_rate_hz-filter_start,
        eeg_filtering_support_seconds=90.,
        final_crop_seconds=[crop_start/10., crop_start/10.+30.],
        target_eeg_start_s=es, target_hb_start_s=hs,
        reference_seconds=5., proxy_rate_hz=10., bands_hz=np.asarray(bands).tolist(),
        preprocessing_support='90s_native_EEG_filtering',
        masking='after_feature_preprocessing; final_bin_excluded_from_fitting_and_scoring')
    return dict(eeg_logpower=features, eeg_operator=reference, time_s=np.arange(300)/10.,
                eeg_proxy_input=EEG_PROXY_INPUT_SCHEMA,
                eeg_proxy_rate_hz=10., metadata=provenance)


def prepare_sensor_proxy_record(project_root, record, row, channel_sets, *,
                                bands=DEFAULT_BANDS, eeg_record=None, hb_record=None):
    """Prepare a 30 s target, reusing central-reader objects within one record.

    The runner owns exact metadata-plan verification and software/synthetic
    prerequisites. Preloaded objects avoid decoding a whole native MAT again
    for each window; the caller owns their canonical-record association.
    """
    data = prepare_sensor_proxy_eeg_record(project_root, record, row, channel_sets,
        bands=bands, eeg_record=eeg_record)
    hb = load_cached_fnirs_record(Path(project_root), record) if hb_record is None else hb_record
    if list(hb['channel_names']) != list(record.manifest['measurement']['fnirs_channel_names']):
        raise ValueError('exact registered Hb channel identities required')
    hs, pair = float(row['hb_start_s']), channel_sets['native_target_index']
    take = (hb['time_s'] >= hs) & (hb['time_s'] < hs + 30)
    times = np.asarray(hb['time_s'][take], float) - hs
    observed_hb = np.asarray(hb['values'][take, pair], float)
    if observed_hb.shape != (len(times), 2) or not np.isfinite(observed_hb).all():
        raise ValueError('complete finite cached target Hb support required')
    operators = disjoint_native_bin_operator(times, steps=300, dt=.1, reference_steps=50)
    data.update(hb=operators['raw'] @ observed_hb, hb_operator=operators['model'],
                fnirs_input=CACHED_FNIRS_INPUT_SCHEMA)
    data['metadata'].update(fnirs_input=CACHED_FNIRS_INPUT_SCHEMA,
        hb_provenance=deepcopy(hb['provenance']),
        preprocessing_support='90s_native_EEG_filtering; full_record_offline_cached_Hb')
    return data


def metadata_sensor_proxy_donors(target_rows, parent_rows, records, events_by_join_key):
    """Select wrong EEG donors from canonical events of parent records only.

    No new target or fitted row is introduced. Actual 90 s filtering supports
    must be disjoint within a native record; different records remain distinct
    time supports. The first eligible donor in record/time/identity order is
    selected, without consulting arrays or fitting results.
    """
    from .event_alignment import window_within_alignment_support

    templates = {}
    for row in parent_rows:
        if row['dataset'] != 'eeg_fnirs_single_trial' or row['split'] not in SPLITS:
            raise ValueError('donor scope requires the public parent subject partitions')
        previous = templates.setdefault(row['join_key'], row)
        if any(previous[k] != row[k] for k in ('subject', 'split', 'source_identity', 'geometry_identity', 'patch')):
            raise ValueError('parent canonical record has inconsistent metadata')
    if any(r['join_key'] not in templates for r in target_rows):
        raise ValueError('target record lies outside the parent donor scope')
    candidates, excluded = [], []
    for key, template in sorted(templates.items()):
        record = records[key]
        if (record.join_key != key or record.dataset_id != template['dataset']
                or record.canonical_subject_id != template['subject']
                or record.manifest['source_files'] != template['source_identity']):
            raise ValueError('canonical donor source identity differs from parent metadata')
        duration_e = record.manifest['array_shapes']['eeg'][0]/200.
        duration_h = record.manifest['array_shapes']['fnirs'][0]/float(record.sample_rate_hz)
        geometry = template['geometry_identity']
        target = template['patch']['native_target_index']
        seeds = next((r['channel_sets']['random_seeds'] for r in target_rows if r['join_key'] == key),
                     (20261027, 20261028, 20261029))
        sets = select_proxy_channel_sets(geometry['eeg'], geometry['fnirs'], target, random_seeds=seeds)
        for event in events_by_join_key.get(key, ()):
            identity = dict(join_key=key, event_id=event.get('event_index'))
            try:
                if event.get('join_key') != key:
                    raise ValueError('canonical event/record identity mismatch')
                es, hs = (float(event[k])/1000.-5. for k in ('eeg_time_ms', 'fnirs_time_ms'))
                label = canonical_label(event, record.dataset_id)
                if label['task'] in ('', 'unknown') or label['condition'] in ('', 'unknown'):
                    raise ValueError('unknown canonical task or condition')
                if (not np.isfinite([es, hs, duration_e, duration_h]).all() or duration_e < 90
                        or min(es, hs) < 0 or es+30 > duration_e or hs+30 > duration_h
                        or not window_within_alignment_support(event, -5., 30.)):
                    raise ValueError('incomplete canonical window or alignment support')
                filter_start = max(0., min(es-30., duration_e-90.))
                crop = (es-filter_start)*10
                if not np.isclose(crop, round(crop), rtol=0, atol=1e-8):
                    raise ValueError('donor crop does not align to the declared 10 Hz grid')
                # The filtering context is also part of the native session
                # support; it may not cross a verified timing discontinuity.
                support = event.get('metadata', {}).get('alignment_support_ms', {}).get('eeg', (None, None))
                if ((support[0] is not None and filter_start*1000 < support[0]-1e-8)
                        or (support[1] is not None and (filter_start+90)*1000 > support[1]+1e-8)):
                    raise ValueError('EEG filtering context crosses alignment support')
                row = deepcopy(template)
                event_id = int(event['event_index'])
                row.update(id=f"{key}|event={event_id}|{template['patch']['patch_id']}",
                    trial_id=f'{key}|event={event_id}', event_id=event_id,
                    task=label['task'], condition=label['condition'], eeg_start_s=es, hb_start_s=hs,
                    window_seconds=30., reference_seconds=5., channel_sets=sets,
                    eeg_filter_start_s=filter_start, eeg_filter_seconds=90., eeg_crop_offset_s=es-filter_start,
                    target_hb_pair=target, target_hb_name=sets['hb_pair_name'], donor_only=True,
                    alignment_support=deepcopy(event.get('metadata', {}).get('alignment_support_ms', {})))
                candidates.append(row)
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                excluded.append(dict(identity, reason=str(exc)))
    if len({r['id'] for r in candidates}) != len(candidates):
        raise ValueError('duplicate canonical donor event identity')
    candidates.sort(key=lambda r: (r['record'], r['eeg_start_s'], r['id']))
    partners, selected = {}, {}
    target_ids = {r['id'] for r in target_rows}
    for target in target_rows:
        choices = []
        for row in candidates:
            if row['id'] == target['id'] or any(row[k] != target[k] for k in ('subject', 'task', 'condition', 'split')):
                continue
            if row['channel_sets']['eeg_channel_names'] != target['channel_sets']['eeg_channel_names']:
                continue
            if row['join_key'] == target['join_key']:
                a, b = target['eeg_filter_start_s'], row['eeg_filter_start_s']
                if a < b+90.-1e-9 and b < a+90.-1e-9:
                    continue
            choices.append(row)
        donor = choices[0] if choices else None
        partners[target['id']] = None if donor is None else donor['id']
        if donor is not None and donor['id'] not in target_ids:
            selected[donor['id']] = donor
    return dict(donor_rows=[selected[k] for k in sorted(selected)], wrong_eeg_partners=partners,
        candidate_count=len(candidates), excluded=excluded,
        availability={split: sum(partners[r['id']] is not None for r in target_rows if r['split'] == split)
                      for split in SPLITS},
        source_scope='canonical_events_of_exact_parent_records_only', measured_arrays_read=0)


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
