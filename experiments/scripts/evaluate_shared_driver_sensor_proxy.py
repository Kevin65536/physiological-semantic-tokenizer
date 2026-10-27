#!/usr/bin/env python3
"""Versioned, train-frozen low-capacity sensor proxy experiment.

The default check never opens measured data. Plan consumes retained metadata;
prepare alone calls the unified native readers after checking that exact plan.
The runner owns task state, while the existing data and inference owners own
signal transforms and numerical computations. Every logical task retains its
terminal result, including unavailable controls and optimization failures.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import yaml

from src.data.ssm_prepared import fit_feature_coordinate
from src.inference.t3a_balloon_robust_ssm import BalloonFixedParameters, BalloonFreeParameters, BalloonParameters

DEFAULT_CONFIG = ROOT / 'experiments/configs/physiology_semantic_tokenizer/shared_driver_sensor_proxy_v1.yaml'
ARTIFACT_ROOT = 'experiments/runs/physiology_semantic_tokenizer/shared_driver_sensor_proxy_v1'
STAGES = ('check', 'synthetic', 'plan', 'prepare', 'screen', 'nonlinear', 'summarize', 'analyze', 'run')
CANDIDATES = ('pc1', 'pc1_band_gain', 'mean6', 'sixth_to_seventh', 'mean3', 'mean12', 'ring025',
              'nearest6_far6', 'nearest6_random6_0', 'nearest6_random6_1', 'nearest6_random6_2')
TERMINAL = ('completed', 'failed_exception', 'failed_preparation', 'failed_numerical',
            'failed_integration', 'unavailable_wrong_EEG', 'unavailable_preparation')


def serial(value):
    if isinstance(value, dict): return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)): return [serial(v) for v in value]
    if isinstance(value, np.generic): return serial(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def read_json(path): return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(serial(value), ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    tmp.replace(path)


def write_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp, **arrays); tmp.replace(path)


def token(value): return hashlib.sha256(str(value).encode()).hexdigest()[:24]


def read_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    expected = yaml.safe_load(DEFAULT_CONFIG.read_text())
    # Reviewed experiment constants cannot be silently changed by a launch.
    # Worker count and numerical budgets remain explicit bounded resources.
    for key in ('schema', 'experiment_id', 'artifact_root', 'scope', 'seed', 'data', 'tensor', 'model', 'fixed',
                'screen', 'promotion', 'nonlinear', 'endpoints', 'stopping'):
        if cfg.get(key) != expected[key]: raise ValueError(f'versioned sensor proxy contract drift: {key}')
    if cfg['schema'] != 'shared_driver_sensor_proxy_v1' or cfg['artifact_root'] != ARTIFACT_ROOT:
        raise ValueError('wrong experiment or artifact root')
    solver = cfg['solver']
    for key in ('penalty', 'initial_penalty', 'flow_prior_weight', 'flow_prior_log_sd', 'substeps',
                'validation_substeps', 'maximum_integration_difference_training_sd',
                'numerical_backend', 'data_weight', 'starts'):
        if solver.get(key) != expected['solver'][key]: raise ValueError(f'fixed solver contract drift: {key}')
    if min(solver['max_iterations'], solver['max_evaluations_per_trial'], solver['gradient_tolerance']) <= 0:
        raise ValueError('positive optimization budgets required')
    r = cfg['resources']
    if (not 1 <= r['max_workers'] <= 52 or r['numerical_threads'] != 1 or r['max_in_flight_per_worker'] != 2):
        raise ValueError('bounded workers and single numerical thread required')
    return cfg


def parameters(cfg):
    p = dict(cfg['fixed']); p.pop('viscoelastic_s')
    tau, kappa = p.pop('tau'), p.pop('kappa')
    return BalloonParameters(fixed=BalloonFixedParameters(**p), free=BalloonFreeParameters(tau=tau, kappa=kappa))


def check(cfg):
    return dict(schema=cfg['schema'], experiment_id=cfg['experiment_id'], status='software_contract_valid',
        stages=list(STAGES), candidates=list(CANDIDATES), planned_logical_cells=420,
        protected_data='forbidden', measured_arrays_read=0,
        main_cells=264, extra_control_cells=156,
        interpretation='public_development_reanalysis_not_unseen_test')


def _paths(out, identity):
    name = token(identity)
    return Path(out)/'prepared'/f'{name}.json', Path(out)/'prepared'/f'{name}.npz'


def _load_prepared(out, row):
    record, arrays = _paths(out, row['id'])
    metadata = read_json(record)
    if metadata['status'] != 'completed': raise ValueError('preparation unavailable: '+metadata['status'])
    with np.load(arrays, allow_pickle=False) as z: result = {k: z[k] for k in z.files}
    result['row'] = row
    return result


def _available_prepared(out, rows):
    available, unavailable = [], []
    for row in rows:
        try: available.append(_load_prepared(out, row))
        except (ValueError, FileNotFoundError) as exc:
            unavailable.append(dict(id=row['id'], record_id=row['id'], native_record=row['record'], subject=row['subject'], success=False,
                status='unavailable_preparation', nmse=None, error=str(exc)))
    return available, unavailable


def wrong_eeg_partners(rows):
    partners = {}
    for a in rows:
        choices = []
        for b in rows:
            if a['id'] == b['id'] or any(a[k] != b[k] for k in ('subject', 'task', 'condition', 'split')): continue
            if a['channel_sets']['eeg_channel_names'] != b['channel_sets']['eeg_channel_names']: continue
            if a['record'] == b['record']:
                sa, sb = a['eeg_filter_start_s'], b['eeg_filter_start_s']
                if sa < sb+90.-1e-9 and sb < sa+90.-1e-9: continue
            choices.append(b['id'])
        partners[a['id']] = sorted(choices)[0] if choices else None
    return partners


def plan_manifest(cfg, parent, records, events_by_join_key=None):
    """Metadata-only deterministic sample choice preserving parent subjects."""
    from src.data.shared_driver_spatial import select_proxy_channel_sets
    if parent.get('status') != 'available': raise ValueError('retained public parent metadata unavailable')
    rows = deepcopy(parent['rows'])
    groups = {}
    for row in rows:
        if row['dataset'] != 'eeg_fnirs_single_trial' or row['split'] not in cfg['data']['expected_subjects']:
            raise ValueError('nonadmitted public dataset or partition')
        groups.setdefault(row['subject'], set()).add(row['split'])
    if any(len(x) != 1 for x in groups.values()): raise ValueError('subject split leakage')
    subjects = {s: sorted(k for k, v in groups.items() if s in v) for s in cfg['data']['expected_subjects']}
    if {k: len(v) for k, v in subjects.items()} != cfg['data']['expected_subjects']:
        raise ValueError('parent subject partition is not the frozen 17/6/6')
    selected = []
    for split in subjects:
        for subject in subjects[split]:
            subset = sorted((r for r in rows if r['subject'] == subject),
                key=lambda r: (r['record'], r['eeg_start_s'], r['hb_start_s'], r['id']))
            # Dev uses one chronological window from each retained native record.
            if split == 'selection':
                by_record = {}
                for row in subset: by_record.setdefault(row['record'], []).append(row)
                subset = [by_record[k][0] for k in sorted(by_record)]
            required = 6 if split != 'selection' else 2
            if len(subset) != required: raise ValueError(f'fixed metadata window denominator unavailable: {subject}')
            for row in subset:
                record = records.get(row['join_key'])
                if record is None: raise ValueError('missing canonical record')
                if record.manifest['source_files'] != row['source_identity']: raise ValueError('parent source identity changed')
                duration = record.manifest['array_shapes']['eeg'][0]/200.
                if duration < 90: raise ValueError('native record shorter than fixed 90s filtering support')
                start = float(row['eeg_start_s'])
                support = max(0., min(start-30., duration-90.))
                if start < support-1e-9 or start+30. > support+90.+1e-9: raise ValueError('target outside 90s EEG support')
                patch = row['patch']
                if (patch['native_target_index'] != cfg['data']['target_hb_native_index']
                        or patch['hb_pair_names'][0] != cfg['data']['target_hb_pair']):
                    raise ValueError('parent center target Hb pair changed')
                geometry = row['geometry_identity']
                sets = select_proxy_channel_sets(geometry['eeg'], geometry['fnirs'], patch['native_target_index'],
                    random_seeds=tuple(cfg['screen']['random_seeds']))
                row.update(channel_sets=sets, eeg_filter_start_s=support, eeg_filter_seconds=90.,
                    eeg_crop_offset_s=start-support, target_hb_pair=patch['native_target_index'],
                    target_hb_name=patch['hb_pair_names'][0], window_seconds=30., reference_seconds=5.)
                selected.append(row)
    if {s: sum(r['split'] == s for r in selected) for s in subjects} != cfg['data']['expected_windows']:
        raise ValueError('fixed metadata window counts changed')
    donor_plan = None
    if events_by_join_key is not None:
        from src.data.shared_driver_spatial import metadata_sensor_proxy_donors
        donor_plan = metadata_sensor_proxy_donors([r for r in selected if r['split'] != 'train'], rows, records, events_by_join_key)
    partners = wrong_eeg_partners(selected) if donor_plan is None else {r['id']: donor_plan['wrong_eeg_partners'].get(r['id']) for r in selected}
    return dict(schema='shared_driver_sensor_proxy_plan_v1', status='available', rows=selected,
        subjects=subjects, donor_rows=[] if donor_plan is None else donor_plan['donor_rows'],
        donor_inventory=donor_plan, parent_plan=cfg['data']['parent_plan'], parent_only='metadata_no_old_features_or_fits',
        excluded=parent.get('excluded', []), wrong_eeg_partners=partners,
        planned_windows={s: sum(r['split'] == s for r in selected) for s in subjects},
        wrong_control_availability={s: sum(partners[r['id']] is not None for r in selected if r['split'] == s)
                                    for s in subjects},
        planned_logical_cells=420, measured_arrays_read=0, scientific_verdict='not_assigned')


def create_plan(cfg, out, root):
    from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
    parent = read_json(Path(root)/cfg['data']['parent_plan'])
    index = CleanPhysiologyCacheIndex(Path(root)/cfg['data']['cache_root'])
    records = {r.join_key: r for r in index.records if r.dataset_id == 'eeg_fnirs_single_trial'}
    plan = plan_manifest(cfg, parent, records, index.events_by_join_key)
    path = Path(out)/'plan.json'
    if path.exists() and read_json(path) != serial(plan): raise ValueError('retained run plan changed; use a fresh versioned run')
    write_json(path, plan); return plan


def bounded_work(worker, payloads, workers, in_flight, progress=None):
    """Bound submitted futures, retaining parent progress after every completion."""
    iterator = iter(payloads); results = []
    if workers == 1:
        for payload in iterator:
            results.append(worker(payload))
            if progress: progress(results)
        return results
    with ProcessPoolExecutor(max_workers=workers) as pool:
        active = {}
        def fill():
            while len(active) < in_flight:
                try: payload = next(iterator)
                except StopIteration: break
                active[pool.submit(worker, payload)] = payload
        fill()
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                payload = active.pop(future)
                try: result = future.result()
                except Exception as exc:
                    if worker is _nonlinear_job:
                        _, out, task, *_ = payload
                        result = dict(task, status='failed_exception', success=False, nmse=None, error=str(exc))
                        write_json(Path(out)/'tasks'/f'{task["id"]}.json', result)
                    else:
                        result = dict(status='failed_exception', error=str(exc), payload=payload)
                results.append(result)
                if progress: progress(results)
            fill()
    return results


def _prepare_job(payload, *, record=None, eeg_record=None, hb_record=None):
    cfg, out, root, row = payload
    path, arrays = _paths(out, row['id'])
    if path.exists(): return read_json(path)
    result = dict(id=row['id'], subject=row['subject'], split=row['split'])
    try:
        from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
        from src.data.shared_driver_spatial import prepare_sensor_proxy_record
        if record is None:
            index = CleanPhysiologyCacheIndex(Path(root)/cfg['data']['cache_root'])
            records = [r for r in index.records if r.join_key == row['join_key']]
            if len(records) != 1: raise ValueError('unique canonical record required')
            record = records[0]
        if row.get('donor_only', False):
            from src.data.shared_driver_spatial import prepare_sensor_proxy_eeg_record
            data = prepare_sensor_proxy_eeg_record(root, record, row, row['channel_sets'], bands=cfg['tensor']['eeg_bands_hz'], eeg_record=eeg_record)
            numeric = {k: data[k] for k in ('eeg_logpower', 'eeg_operator', 'time_s')}
        else:
            data = prepare_sensor_proxy_record(root, record, row, row['channel_sets'], bands=cfg['tensor']['eeg_bands_hz'], eeg_record=eeg_record, hb_record=hb_record)
            numeric = {k: data[k] for k in ('eeg_logpower', 'hb', 'eeg_operator', 'hb_operator', 'time_s')}
            if numeric['hb'].shape != (300, 2): raise ValueError('wrong 10Hz Hb tensor shape')
        if numeric['eeg_logpower'].shape[0] != 300: raise ValueError('wrong 10Hz EEG tensor shape')
        write_npz(arrays, **numeric)
        result.update(status='completed', row=row, metadata=data['metadata'], array_path=str(arrays))
    except Exception as exc:
        result.update(status='failed_preparation', error=str(exc), traceback=traceback.format_exc())
    write_json(path, result); return result


def _prepare_record_job(payload):
    cfg, out, root, rows = payload
    root = Path(root)
    pending = [r for r in rows if not _paths(out, r['id'])[0].exists()]
    if not pending: return [read_json(_paths(out, r['id'])[0]) for r in rows]
    try:
        from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
        from src.data.unified_physiology import load_native_eeg_record, load_cached_fnirs_record
        index = CleanPhysiologyCacheIndex(Path(root)/cfg['data']['cache_root'])
        records = [r for r in index.records if r.join_key == rows[0]['join_key']]
        if len(records) != 1: raise ValueError('unique canonical record required')
        record = records[0]
        for row in pending:
            if (row['dataset'] != record.dataset_id or row['subject'] != record.canonical_subject_id
                    or row['record'] != record.base_record_id or row['source_identity'] != record.manifest['source_files']):
                raise ValueError('source identity drift before grouped native read')
        from src.data.shared_driver_spatial import select_proxy_channel_sets
        for row in pending:
            geometry = row['geometry_identity']
            sets = select_proxy_channel_sets(geometry['eeg'], geometry['fnirs'], row['patch']['native_target_index'],
                random_seeds=tuple(cfg['screen']['random_seeds']))
            if sets != row['channel_sets']: raise ValueError('sensor sets changed before grouped native read')
        eeg = load_native_eeg_record(root, record)
        hb = load_cached_fnirs_record(root, record) if any(not r.get('donor_only', False) for r in pending) else None
        return [_prepare_job((cfg, out, root, row), record=record, eeg_record=eeg, hb_record=hb) for row in rows]
    except Exception as exc:
        results = []
        for row in rows:
            path = _paths(out, row['id'])[0]
            if path.exists(): result = read_json(path)
            else:
                result = dict(id=row['id'], subject=row['subject'], split=row['split'], status='failed_preparation',
                    error=str(exc), traceback=traceback.format_exc())
                write_json(path, result)
            results.append(result)
        return results


def prepare_stage(cfg, out, root, plan, workers):
    # Recompute metadata immediately before any native read, fail closed on drift.
    if create_plan(cfg, out, root) != plan: raise ValueError('metadata identity changed before preparation')
    groups = {}
    for row in plan['rows']+plan.get('donor_rows', []): groups.setdefault(row['join_key'], []).append(row)
    return bounded_work(_prepare_record_job, [(cfg, str(out), str(root), rows) for rows in groups.values()],
        workers, workers*2, lambda groups_done: write_json(Path(out)/'preparation_progress.json',
            dict(terminal_records=len(groups_done), planned_records=len(groups),
                 terminal_windows=sum(len(rows) for rows in groups_done))))


def coordinate(cfg, records):
    return fit_feature_coordinate([(r['eeg_logpower'][:, r['row']['channel_sets']['sets']['nearest6']], r['hb'])
        for r in records], eeg_training_sd_target=cfg['screen']['eeg_training_sd_target'],
        common_hb_training_sd_target=cfg['screen']['hb_training_sd_target'])


def proxy_features(record, coord, candidate):
    from src.data.ssm_prepared import pc1_band_contributions, spatial_band_features, ring_band_features
    x, sets = record['eeg_logpower'], record['row']['channel_sets']['sets']
    if candidate in ('pc1', 'pc1_band_gain'):
        return pc1_band_contributions(x[:, sets['nearest6']], coord)
    if candidate == 'ring025':
        return ring_band_features(x, sets['nearest6'], sets['outer6'], alpha=.25, eeg_factor=coord['eeg_factor'])
    key = {'mean3': 'nearest3', 'mean6': 'nearest6', 'mean12': 'nearest12'}.get(candidate, candidate)
    return spatial_band_features(x, sets[key], eeg_factor=coord['eeg_factor'])


def observation_operator(record):
    n = len(record['hb']); operator = np.zeros((n, 3, n, 3))
    for k, name in enumerate(('eeg_operator', 'hb_operator', 'hb_operator')): operator[:, k, :, k] = record[name]
    return operator.reshape(3*n, 3*n)


_MAP_CACHE = {}


def conditional_map(cfg, record, coord):
    from src.inference.low_capacity_eeg_proxy import build_tak_eeg_conditional_map
    s = cfg['solver']; visible = np.ones(len(record['hb']), bool); visible[-1] = False
    operator = observation_operator(record)
    key = hashlib.sha256(operator.tobytes()+np.asarray(coord['sd'], float).tobytes()).hexdigest()
    if key in _MAP_CACHE: return _MAP_CACHE[key]
    result = build_tak_eeg_conditional_map(parameters(cfg), len(visible), cfg['tensor']['dt_s'],
        mean_operator=operator, sd=np.asarray(coord['sd']), eeg_visible=visible,
        penalty=s['penalty'], initial_penalty=s['initial_penalty'], flow_prior_weight=s['flow_prior_weight'],
        flow_prior_log_sd=s['flow_prior_log_sd'], data_weight=s['data_weight'], substeps=s['substeps'],
        viscoelastic_s=cfg['fixed']['viscoelastic_s'], initial_state_mode='volume_tied')
    if len(_MAP_CACHE) >= 8: _MAP_CACHE.pop(next(iter(_MAP_CACHE)))
    _MAP_CACHE[key] = result
    return result


def score_nmse(target, prediction, sd, score_steps=(50, 299)):
    left, right = score_steps
    error = (np.asarray(target)[left:right]-np.asarray(prediction)[left:right])/np.asarray(sd)
    if not error.size or not np.isfinite(error).all(): return None
    return float(np.mean(error**2))


def waveform_metrics(target, prediction, sd):
    target, prediction = np.asarray(target)[50:299], np.asarray(prediction)[50:299]
    errors = (target-prediction)/np.asarray(sd)
    correlation, rms_ratio = [], []
    for c in range(2):
        a, b = target[:, c]-target[:, c].mean(), prediction[:, c]-prediction[:, c].mean()
        denom = np.linalg.norm(a)*np.linalg.norm(b)
        correlation.append(float(a@b/denom) if denom > 0 else None)
        amplitude = np.sqrt(np.mean(target[:, c]**2))
        rms_ratio.append(float(np.sqrt(np.mean(prediction[:, c]**2))/amplitude) if amplitude > 0 else None)
    return dict(component_names=['HbO', 'HbR'], component_nmse=np.mean(errors**2, axis=0),
        target_prediction_correlation=correlation, prediction_to_target_RMS_ratio=rms_ratio,
        Hb_data_standardized_SSE=float(np.sum(errors**2)),
        amplitude_definition='RMS_on_scored_baseline_relative_coordinate_no_demeaning',
        score_steps=[50, 299], correlation_definition='Pearson_within_scored_window_descriptive')


def _subject_score(rows, penalty=100.):
    subjects = {}
    for row in rows:
        value = row.get('nmse') if row.get('success', True) else None
        record = row.get('native_record', row.get('record', row.get('record_id', row.get('id', 'unspecified'))))
        subjects.setdefault(row['subject'], {}).setdefault(record, []).append(penalty if value is None else value)
    return {subject: float(np.mean([np.mean(values) for values in records.values()]))
            for subject, records in subjects.items()}


def _fit_candidate(cfg, candidate, records, coord, maps, ridge):
    from src.inference.low_capacity_eeg_proxy import fit_anchored_ridge
    if candidate == 'pc1': return dict(coefficients=np.ones(5), ridge=0.)
    designs = [m.map_features(proxy_features(r, coord, candidate)) for r, m in zip(records, maps)]
    masks = []
    for r in records:
        mask = np.zeros_like(r['hb'], bool); mask[50:299] = True; masks.append(mask)
    return fit_anchored_ridge(designs, [r['hb']*coord['hb_factor'] for r in records],
        sd=np.asarray(coord['sd'])[1:], anchor=np.ones(5) if candidate == 'pc1_band_gain' else np.zeros(5),
        ridge=ridge, offsets=[m.hb_offset for m in maps], masks=masks,
        record_weights=[1/sum(v['row']['subject'] == r['row']['subject'] for v in records) for r in records])


def _linear_scores(cfg, records, coord, candidate, coefficients, maps, donors=None):
    result = []
    for r, mapping in zip(records, maps):
        source = r if donors is None else donors.get(r['row']['id'])
        if source is None:
            result.append(dict(id=r['row']['id'], native_record=r['row']['record'], subject=r['row']['subject'], success=False, status='unavailable_wrong_EEG', nmse=None)); continue
        proxy = proxy_features(source, coord, candidate)@np.asarray(coefficients)
        prediction = mapping.predict(proxy)['prediction'][:, 1:]
        nmse = score_nmse(r['hb']*coord['hb_factor'], prediction, np.asarray(coord['sd'])[1:], cfg['tensor']['score_steps'])
        result.append(dict(id=r['row']['id'], native_record=r['row']['record'], subject=r['row']['subject'], success=nmse is not None, nmse=nmse))
    return result


def choose_shortlist(cfg, scores):
    """Train-CV chooses one same-six and one distinct near-spatial alternative."""
    same = min(('pc1_band_gain', 'mean6'), key=lambda c: (scores[c], c))
    spatial = [c for c in cfg['screen']['spatial_candidates'] if c != same]
    best = min(spatial, key=lambda c: (scores[c], c))
    baseline = scores['pc1']
    if baseline <= 0 or 1-scores[best]/baseline < cfg['promotion']['minimum_relative_nmse_improvement']:
        best = 'mean12'
    return ['pc1', same, best]


def screen_stage(cfg, out, plan):
    """All PCA/scales/ridge selection are fit inside each subject CV fold."""
    target = Path(out)/'screen.json'
    if target.exists(): return read_json(target)
    train = [_load_prepared(out, r) for r in plan['rows'] if r['split'] == 'train']
    dev, unavailable_dev = _available_prepared(out, [r for r in plan['rows'] if r['split'] == 'selection'])
    subjects = sorted({r['row']['subject'] for r in train})
    folds = [set(x.tolist()) for x in np.array_split(np.random.default_rng(cfg['seed']).permutation(subjects), 3)]
    cv = {c: {str(l): [] for l in cfg['screen']['ridges']} for c in CANDIDATES}
    fold_coordinates, fold_coefficients = [], {}
    for fold, hidden in enumerate(folds):
        fit_rows = [r for r in train if r['row']['subject'] not in hidden]
        val_rows = [r for r in train if r['row']['subject'] in hidden]
        coord = coordinate(cfg, fit_rows)
        fold_coordinates.append(coord)
        maps = [conditional_map(cfg, r, coord) for r in fit_rows]
        validation_maps = [conditional_map(cfg, r, coord) for r in val_rows]
        for candidate in CANDIDATES:
            for ridge in cfg['screen']['ridges']:
                fitted = _fit_candidate(cfg, candidate, fit_rows, coord, maps, ridge)
                fold_coefficients[(fold, candidate, str(ridge))] = fitted['coefficients']
                rows = _linear_scores(cfg, val_rows, coord, candidate, fitted['coefficients'], validation_maps)
                cv[candidate][str(ridge)].extend([dict(row, fold=fold) for row in rows])
        write_json(Path(out)/'screen_progress.json', dict(completed_folds=fold+1, planned_folds=3))
    selected_ridges, cv_scores = {}, {}
    for candidate in CANDIDATES:
        values = {ridge: float(np.mean(list(_subject_score(rows).values()))) for ridge, rows in cv[candidate].items()}
        winner = min(values, key=lambda x: (values[x], float(x)))
        selected_ridges[candidate] = float(winner); cv_scores[candidate] = values[winner]
    coord = coordinate(cfg, train)
    maps = [conditional_map(cfg, r, coord) for r in train]
    dev_maps = [conditional_map(cfg, r, coord) for r in dev]
    donor_records, unavailable_donors = _available_prepared(out, [r for r in plan.get('donor_rows', []) if r['split'] == 'selection'])
    all_records = {r['row']['id']: r for r in train+dev+donor_records}
    donors = {r['row']['id']: all_records.get(plan['wrong_eeg_partners'][r['row']['id']]) for r in dev}
    fitted, scores = {}, {}
    for candidate in CANDIDATES:
        model = _fit_candidate(cfg, candidate, train, coord, maps, selected_ridges[candidate])
        fitted[candidate] = dict(coefficients=model['coefficients'], ridge=selected_ridges[candidate])
        scores[candidate] = dict(true=_linear_scores(cfg, dev, coord, candidate, model['coefficients'], dev_maps)+unavailable_dev,
            wrong=_linear_scores(cfg, dev, coord, candidate, model['coefficients'], dev_maps, donors)+unavailable_dev)
    shortlist = choose_shortlist(cfg, cv_scores)
    template = np.mean([r['hb']*coord['hb_factor'] for r in train], axis=0)
    write_npz(Path(out)/'training_template.npz', prediction=template)
    fir = fir_screen(cfg, out, train, dev, folds, fold_coordinates, fold_coefficients,
        selected_ridges, coord, fitted, shortlist)
    for model in fir['models'].values():
        model['development'].extend(unavailable_dev)
        model['development_subject_nmse'] = _subject_score(model['development'])
        values = list(model['development_subject_nmse'].values())
        model['development_subject_equal_nmse'] = float(np.mean(values)) if values else None
    result = dict(unavailable_development=unavailable_dev, unavailable_donors=unavailable_donors, schema='shared_driver_sensor_proxy_screen_v1', status='completed', coordinate=coord,
        models=fitted, fir=fir, selected_ridges=selected_ridges, cv_subject_mean_nmse=cv_scores,
        cv=cv, folds=[sorted(x) for x in folds], development=scores, shortlist=shortlist,
        selection='training_grouped_CV_shortlist_then_development_nonlinear_selection',
        training_ids=[r['row']['id'] for r in train], scientific_verdict='not_assigned')
    write_json(target, result)
    return result


def fir_design(cfg, record, coord, candidate, coefficients):
    """Six fixed scalar-proxy lags, explicit zero history, then Hb reference."""
    from src.inference.low_capacity_eeg_proxy import fixed_lag_fir_design
    proxy = proxy_features(record, coord, candidate)@np.asarray(coefficients)
    lags = np.rint(np.asarray(cfg['screen']['fir_lags_s'])/cfg['tensor']['dt_s']).astype(int)
    pad = int(max(lags))
    design, valid = fixed_lag_fir_design(np.r_[np.zeros(pad), proxy][:, None], lags)
    design = np.asarray(record['hb_operator'])@design[pad:]
    width = design.shape[1]; block = np.zeros((len(proxy), 2, width*2))
    block[:, 0, :width], block[:, 1, width:] = design, design
    return block


def _fit_fir(cfg, records, coord, candidate, coefficients, ridge):
    from src.inference.low_capacity_eeg_proxy import fit_anchored_ridge
    designs = [fir_design(cfg, r, coord, candidate, coefficients) for r in records]
    masks = [np.broadcast_to((np.arange(len(r['hb'])) >= 50)[:, None], r['hb'].shape).copy() for r in records]
    for mask in masks: mask[-1] = False
    return fit_anchored_ridge(designs, [r['hb']*coord['hb_factor'] for r in records],
        sd=np.asarray(coord['sd'])[1:], anchor=np.zeros(designs[0].shape[-1]), ridge=ridge, masks=masks,
        record_weights=[1/sum(v['row']['subject'] == r['row']['subject'] for v in records) for r in records])


def _fir_scores(cfg, records, coord, candidate, proxy_coefficients, fir_coefficients, out=None):
    rows = []
    for record in records:
        design = fir_design(cfg, record, coord, candidate, proxy_coefficients)
        prediction = np.einsum('tck,k->tc', design, fir_coefficients)
        row = record['row']
        nmse = score_nmse(record['hb']*coord['hb_factor'], prediction, np.asarray(coord['sd'])[1:])
        rows.append(dict(id=row['id'], native_record=row['record'], subject=row['subject'], success=nmse is not None, nmse=nmse))
        if out is not None:
            write_npz(Path(out)/'fir'/f'{token(row["id"]+candidate)}.npz', prediction=prediction,
                residual=record['hb']*coord['hb_factor']-prediction, time_s=record['time_s'])
    return rows


def fir_screen(cfg, out, train, dev, folds, fold_coords, fold_coefficients, ridges, coord, models, shortlist):
    results = {}
    for candidate in shortlist:
        cv = {str(ridge): [] for ridge in cfg['screen']['ridges']}
        for fold, hidden in enumerate(folds):
            fit_rows = [r for r in train if r['row']['subject'] not in hidden]
            val_rows = [r for r in train if r['row']['subject'] in hidden]
            coeff = fold_coefficients[(fold, candidate, str(ridges[candidate]))]
            for ridge in cfg['screen']['ridges']:
                fitted = _fit_fir(cfg, fit_rows, fold_coords[fold], candidate, coeff, ridge)
                cv[str(ridge)].extend(_fir_scores(cfg, val_rows, fold_coords[fold], candidate, coeff, fitted['coefficients']))
        scores = {ridge: float(np.mean(list(_subject_score(rows).values()))) for ridge, rows in cv.items()}
        ridge = min(scores, key=lambda k: (scores[k], float(k)))
        coefficients = models[candidate]['coefficients']
        fit = _fit_fir(cfg, train, coord, candidate, coefficients, float(ridge))
        development = _fir_scores(cfg, dev, coord, candidate, coefficients, fit['coefficients'], out)
        subject_values = _subject_score(development)
        results[candidate] = dict(coefficients=fit['coefficients'], ridge=float(ridge), cv_subject_mean_nmse=scores,
            development=development, development_subject_nmse=subject_values,
            development_subject_equal_nmse=float(np.mean(list(subject_values.values()))) if subject_values else None)
    return dict(models=results, input='frozen_scalar_proxy', lags_s=cfg['screen']['fir_lags_s'],
        history='zero_before_window_in_baseline_relative_EEG_coordinate',
        fir_boundary='zero_history_in_baseline_relative_processed_EEG_coordinate_engineering_assumption',
        target_operator='same_complete_5s_reference_operator_before_scoring',
        interpretation='empirical_crossmodal_information_diagnostic_not_fixed_physical_model')


def template_scores(cfg, out, plan, coord, *, split='selection'):
    with np.load(Path(out)/'training_template.npz', allow_pickle=False) as arrays: prediction = arrays['prediction']
    path = Path(out)/'training_template_scores.json'
    result = read_json(path) if path.exists() else {'selection': [], 'evaluation': []}
    result[split] = []
    for row in plan['rows']:
        if row['split'] != split: continue
        try:
            record = _load_prepared(out, row)
            nmse = score_nmse(record['hb']*coord['hb_factor'], prediction, np.asarray(coord['sd'])[1:])
            result[row['split']].append(dict(id=row['id'], record_id=row['id'], native_record=row['record'], subject=row['subject'], success=True, nmse=nmse))
        except (ValueError, FileNotFoundError) as exc:
            result[row['split']].append(dict(id=row['id'], record_id=row['id'], native_record=row['record'], subject=row['subject'], success=False,
                status='unavailable_preparation', nmse=None, error=str(exc)))
    write_json(Path(out)/'training_template_scores.json', result)
    return result


def evaluate_fir(cfg, out, plan, screen):
    records, unavailable = _available_prepared(out, [r for r in plan['rows'] if r['split'] == 'evaluation'])
    result = {}
    for candidate, model in screen['fir']['models'].items():
        rows = _fir_scores(cfg, records, screen['coordinate'], candidate,
            screen['models'][candidate]['coefficients'], model['coefficients'], out)+unavailable
        result[candidate] = dict(rows=rows, planned=36, available=len(records),
            independent_subjects=len({r['subject'] for r in rows}),
            subject_equal_nmse_with_failure_penalty=float(np.mean(list(_subject_score(rows).values()))))
    write_json(Path(out)/'fir_evaluation_scores.json', result)
    return result


def nonlinear_tasks(plan, screen, *, phase, evaluation_candidates=None):
    rows = [r for r in plan['rows'] if r['split'] in ('selection', 'evaluation')]
    tasks = []
    if phase == 'development':
        for row in rows:
            for candidate, condition in (('Hb_only', 'joint'), ('zero', 'prediction')):
                tasks.append(dict(record_id=row['id'], native_record=row['record'], subject=row['subject'], split=row['split'], candidate=candidate,
                                  condition=condition, pairing='paired'))
        selected = [r for r in rows if r['split'] == 'selection']; candidates = screen['shortlist']
    elif phase == 'evaluation':
        selected = [r for r in rows if r['split'] == 'evaluation']; candidates = evaluation_candidates
    else: raise ValueError('unknown nonlinear phase')
    for row in selected:
        for candidate in candidates:
            for condition, pairing in (('prediction', 'paired'), ('joint', 'paired'), ('prediction', 'wrong_EEG')):
                tasks.append(dict(record_id=row['id'], native_record=row['record'], subject=row['subject'], split=row['split'],
                                  candidate=candidate, condition=condition, pairing=pairing))
    for task in tasks:
        task['wrong_eeg_id'] = plan['wrong_eeg_partners'][task['record_id']] if task['pairing'] == 'wrong_EEG' else None
        task['id'] = token('|'.join(str(task[k]) for k in ('record_id', 'candidate', 'condition', 'pairing')))
    return tasks


def assemble_fit_input(cfg, record, coord, candidate, coefficients=None, donor=None, *, condition='joint'):
    n = len(record['hb']); visible = np.ones((n, 3), bool); visible[-1] = False
    if candidate in ('Hb_only', 'zero'):
        proxy = np.full(n, np.nan); visible[:, 0] = False
    else:
        proxy = proxy_features(record if donor is None else donor, coord, candidate)@np.asarray(coefficients)
    hb = np.full((n, 2), np.nan) if condition == 'prediction' else record['hb']*coord['hb_factor']
    if condition == 'prediction': visible[:, 1:] = False
    target = np.column_stack((proxy, hb))
    return target, visible


def _nonlinear_job(payload):
    cfg, out, task, row, donor_row, screen = payload
    path = Path(out)/'tasks'/f'{task["id"]}.json'
    if path.exists(): return read_json(path)
    result = dict(task, success=False, nmse=None)
    started = time.monotonic()
    try:
        if task['pairing'] == 'wrong_EEG' and donor_row is None:
            result['status'] = 'unavailable_wrong_EEG'
        else:
            prepared_status = read_json(_paths(out, row['id'])[0])['status']
            if prepared_status != 'completed':
                result.update(status='unavailable_preparation', error=prepared_status)
                write_json(path, result); return result
            record = _load_prepared(out, row)
            if donor_row is not None:
                donor_status = read_json(_paths(out, donor_row['id'])[0])['status']
                if donor_status != 'completed':
                    result.update(status='unavailable_preparation', error='donor: '+donor_status)
                    write_json(path, result); return result
            donor = None if donor_row is None else _load_prepared(out, donor_row)
            coord = screen['coordinate']; candidate = task['candidate']
            coeff = None if candidate in ('Hb_only', 'zero') else screen['models'][candidate]['coefficients']
            y, visible = assemble_fit_input(cfg, record, coord, candidate, coeff, donor, condition=task['condition'])
            if task['condition'] == 'prediction': visible[:, 1:] = False
            # No hidden Hb is present in the fit input or initialization.
            y = np.where(visible, y, np.nan)
            if candidate == 'zero':
                # No observations: deterministic prior-centered resting solution.
                from src.inference.shared_driver_reconstruction import nonlinear_driver_forward
                forward = nonlinear_driver_forward(np.zeros(len(y)), np.array([0., 1., 1., 1., 1.]),
                    parameters(cfg), cfg['tensor']['dt_s'], substeps=cfg['solver']['substeps'],
                    numerical_backend='numba', hemodynamics='tak', viscoelastic_s=2.)
                fit = dict(status='completed', converged=True, prediction=(observation_operator(record)@forward['canonical_prediction'].ravel()).reshape(len(y), 3)[None],
                    driver=np.zeros((1, len(y))), initial_state=np.array([[0., 1., 1., 1., 1.]]),
                    states=forward['states'][None], objective_components=dict(data=0., driver=0., initial=0., flow=0.))
            else:
                from src.inference.shared_driver_reconstruction import fit_nonlinear_shared_parameters
                s = cfg['solver']
                fit = fit_nonlinear_shared_parameters(y[None], parameters(cfg), cfg['tensor']['dt_s'],
                    parameter_names=[], parameter_bounds=[], sd=np.asarray(coord['sd']), mean_operator=observation_operator(record),
                    visible=visible[None], penalty=s['penalty'], initial_penalty=s['initial_penalty'],
                    flow_prior_weight=s['flow_prior_weight'], flow_prior_log_sd=s['flow_prior_log_sd'],
                    substeps=s['substeps'], max_iterations=s['max_iterations'], max_evaluations_per_trial=s['max_evaluations_per_trial'],
                    gradient_tolerance=s['gradient_tolerance'], numerical_backend=s['numerical_backend'],
                    hemodynamics='tak', viscoelastic_s=2., initial_state_mode='volume_tied', data_weight=s['data_weight'])
            if 'prediction' not in fit:
                result.update(status='failed_numerical', numerical_status=fit.get('status'), fit=fit)
            else:
                from src.inference.shared_driver_reconstruction import nonlinear_driver_forward
                s = cfg['solver']; prediction = np.asarray(fit['prediction'])[0]
                finer = nonlinear_driver_forward(np.asarray(fit['driver'])[0], np.asarray(fit['initial_state'])[0],
                    parameters(cfg), cfg['tensor']['dt_s'], substeps=s['validation_substeps'], derivative=False,
                    numerical_backend='numba', hemodynamics='tak', viscoelastic_s=2.)
                fine_prediction = (observation_operator(record)@finer['canonical_prediction'].ravel()).reshape(len(y), 3)
                integration = float(np.max(abs(fine_prediction-prediction)/np.asarray(coord['sd'])))
                success = bool(fit.get('converged', False) and integration <= s['maximum_integration_difference_training_sd'])
                nmse = score_nmse(record['hb']*coord['hb_factor'], prediction[:, 1:], np.asarray(coord['sd'])[1:])
                arrays = Path(out)/'tasks'/f'{task["id"]}.npz'
                write_npz(arrays, prediction=prediction, target_hb=record['hb']*coord['hb_factor'],
                    residual=record['hb']*coord['hb_factor']-prediction[:, 1:], driver=fit['driver'],
                    initial_state=fit['initial_state'], states=fit['states'], time_s=record['time_s'])
                from src.inference.low_capacity_eeg_proxy import residual_diagnostics
                residual_mask = np.zeros((len(y), 2), bool); residual_mask[50:299] = True
                diagnostics = residual_diagnostics(record['hb']*coord['hb_factor']-prediction[:, 1:], cfg['tensor']['dt_s'],
                    scale=np.asarray(coord['sd'])[1:], mask=residual_mask,
                    regressors=y[:, 0] if candidate not in ('Hb_only', 'zero') else None)
                waveform = waveform_metrics(record['hb']*coord['hb_factor'], prediction[:, 1:], np.asarray(coord['sd'])[1:])
                result.update(residual_diagnostics=diagnostics, waveform_metrics=waveform, status='completed' if success else 'failed_integration' if integration > s['maximum_integration_difference_training_sd'] else 'failed_numerical',
                    success=success, nmse=nmse, integration_difference_training_sd=integration,
                    numerical_status=fit.get('status'), converged=fit.get('converged', False),
                    objective_components=fit.get('objective_components'), objective=fit.get('objective'),
                    starts=fit.get('starts'), evaluations=fit.get('evaluations'), array_path=str(arrays))
    except Exception as exc:
        result.update(status='failed_exception', error=str(exc), traceback=traceback.format_exc())
    import resource
    result['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['elapsed_seconds'] = time.monotonic()-started
    write_json(path, result); return result


def promotion_assessment(cfg, candidate, baseline, wrong, zero, template=None):
    penalty = cfg['nonlinear']['failure_penalty_nmse']; c, b = _subject_score(candidate, penalty), _subject_score(baseline, penalty)
    common = sorted(set(c)&set(b)); minimum = cfg['promotion']['minimum_paired_subjects']
    if not common: return dict(eligible=False, reason='no_paired_subjects', paired_subjects=0)
    cm, bm = np.mean([c[s] for s in common]), np.mean([b[s] for s in common])
    relative = 1-cm/bm if bm > 0 else None
    fraction = float(np.mean([c[s] < b[s] for s in common]))
    # True/wrong comparison uses the exact same available identity set.
    wrong_by_id = {r['record_id']: r for r in wrong if r.get('success', False)}
    true_paired = [r for r in candidate if r.get('success', False) and r['record_id'] in wrong_by_id]
    wrong_paired = [wrong_by_id[r['record_id']] for r in true_paired]
    wt, ww = _subject_score(true_paired, penalty), _subject_score(wrong_paired, penalty)
    wrong_ok = (len(wt) >= minimum and len(true_paired) >= .5*len(candidate)
        and np.mean(list(wt.values())) < np.mean(list(ww.values())))
    z = _subject_score(zero, penalty); zcommon = sorted(set(c)&set(z))
    zero_ok = bool(zcommon) and np.mean([c[s] for s in zcommon]) < np.mean([z[s] for s in zcommon])
    template_ok = True
    if template is not None:
        t = _subject_score(template, penalty); paired = sorted(set(c)&set(t))
        template_ok = bool(paired) and np.mean([c[s] for s in paired]) < np.mean([t[s] for s in paired])
    failure_c = np.mean([not r.get('success', False) for r in candidate]) if candidate else 1.
    failure_b = np.mean([not r.get('success', False) for r in baseline]) if baseline else 1.
    eligible = (len(common) >= minimum and relative is not None and relative >= cfg['promotion']['minimum_relative_nmse_improvement']
        and fraction >= cfg['promotion']['minimum_subject_improvement_fraction'] and wrong_ok and zero_ok and template_ok and failure_c <= failure_b)
    return dict(eligible=bool(eligible), paired_subjects=len(common), subject_mean_nmse=float(cm),
        baseline_subject_mean_nmse=float(bm), relative_nmse_improvement=relative, subject_improvement_fraction=fraction,
        wrong_paired_subjects=len(wt), wrong_paired_windows=len(true_paired),
        wrong_minimum_coverage_windows=.5*len(candidate), true_better_than_wrong=bool(wrong_ok), true_better_than_zero=bool(zero_ok),
        true_better_than_template=bool(template_ok), failure_rate=float(failure_c), baseline_failure_rate=float(failure_b))


def select_evaluation_candidates(cfg, out, screen, rows):
    def get(candidate, pairing='paired'):
        return [r for r in rows if r['split'] == 'selection' and r['candidate'] == candidate and
                r['condition'] == 'prediction' and r['pairing'] == pairing]
    baseline, zero = get('pc1'), get('zero')
    template = read_json(Path(out)/'training_template_scores.json')['selection']
    assessments = {}
    for c in screen['shortlist'][1:]:
        assessments[c] = promotion_assessment(cfg, get(c), baseline, get(c, 'wrong_EEG'), zero, template)
    eligible = [c for c, a in assessments.items() if a['eligible']]
    pool = eligible or screen['shortlist'][1:]
    winner = min(pool, key=lambda c: (assessments[c].get('subject_mean_nmse', float('inf')), c))
    result = dict(candidates=['pc1', winner], promoted=bool(eligible), assessments=assessments,
        interpretation='development_selected_winner' if eligible else 'no_promotion_best_distinct_candidate_is_diagnostic',
        evaluation_tuning=False)
    path = Path(out)/'evaluation_selection.json'
    if path.exists() and read_json(path) != serial(result): raise ValueError('frozen evaluation selection changed')
    write_json(path, result); return result


def nonlinear_stage(cfg, out, plan, screen, workers):
    by_id = {r['id']: r for r in plan['rows']+plan.get('donor_rows', [])}
    template_scores(cfg, out, plan, screen['coordinate'])
    def execute(tasks):
        payloads = [(cfg, str(out), task, by_id[task['record_id']], by_id.get(task['wrong_eeg_id']), screen) for task in tasks]
        return bounded_work(_nonlinear_job, payloads, workers, workers*2,
            lambda results: write_json(Path(out)/'nonlinear_progress.json', dict(terminal_this_phase=len(results),
                planned_this_phase=len(tasks), last_status=results[-1]['status'])))
    dev_tasks = nonlinear_tasks(plan, screen, phase='development')
    write_json(Path(out)/'development_tasks.json', dev_tasks)
    rows = execute(dev_tasks)
    selection = select_evaluation_candidates(cfg, out, screen, rows)
    template_scores(cfg, out, plan, screen['coordinate'], split='evaluation')
    evaluate_fir(cfg, out, plan, screen)
    eval_tasks = nonlinear_tasks(plan, screen, phase='evaluation', evaluation_candidates=selection['candidates'])
    tasks = dev_tasks+eval_tasks
    if len(tasks) != cfg['nonlinear']['planned_logical_cells']: raise ValueError('planned nonlinear denominator changed')
    write_json(Path(out)/'nonlinear_tasks.json', tasks)
    execute(eval_tasks)
    return tasks


def paired_comparisons(cfg, rows):
    """Descriptive paired subject bootstrap; failure rows stay in denominators."""
    rng = np.random.default_rng(cfg['seed']); result = {}
    for split in ('selection', 'evaluation'):
        targets = [r for r in rows if r['split'] == split and r['condition'] == 'prediction' and r['pairing'] == 'paired']
        baseline = {r['record_id']: r for r in targets if r['candidate'] == 'pc1'}
        for candidate in sorted({r['candidate'] for r in targets} - {'pc1'}):
            true = [r for r in targets if r['candidate'] == candidate and r['record_id'] in baseline]
            comparison = [baseline[r['record_id']] for r in true]
            c, b = _subject_score(true), _subject_score(comparison)
            subjects = sorted(set(c)&set(b))
            if not subjects: continue
            a, reference = np.array([c[s] for s in subjects]), np.array([b[s] for s in subjects])
            draws = rng.integers(0, len(subjects), size=(cfg['endpoints']['bootstrap_repeats'], len(subjects)))
            difference = a[draws].mean(axis=1)-reference[draws].mean(axis=1)
            relative = 1-a[draws].mean(axis=1)/reference[draws].mean(axis=1)
            successful = [r for r in true if r.get('success') and baseline[r['record_id']].get('success')]
            sc = _subject_score(successful)
            sb = _subject_score([baseline[r['record_id']] for r in successful])
            common_success = sorted(set(sc)&set(sb))
            subset = dict(paired_windows=len(successful), paired_subjects=len(common_success))
            if common_success:
                sa, sr = np.array([sc[s] for s in common_success]), np.array([sb[s] for s in common_success])
                sd = rng.integers(0, len(common_success), size=(cfg['endpoints']['bootstrap_repeats'], len(common_success)))
                subset.update(candidate_nmse=float(sa.mean()), baseline_nmse=float(sr.mean()),
                    delta_nmse_candidate_minus_baseline=float((sa-sr).mean()),
                    delta_nmse_descriptive_95pct_interval=np.quantile(sa[sd].mean(axis=1)-sr[sd].mean(axis=1), [.025, .975]),
                    relative_nmse_improvement=float(1-sa.mean()/sr.mean()) if sr.mean() > 0 else None)
            result[f'{split}|{candidate}_vs_pc1'] = dict(paired_windows=len(true), paired_subjects=len(subjects),
                common_success_only=subset,
                subject_ids=subjects, candidate_nmse=float(a.mean()), baseline_nmse=float(reference.mean()),
                delta_nmse_candidate_minus_baseline=float((a-reference).mean()),
                delta_nmse_descriptive_95pct_interval=np.quantile(difference, [.025, .975]),
                relative_nmse_improvement=float(1-a.mean()/reference.mean()) if reference.mean() > 0 else None,
                relative_improvement_descriptive_95pct_interval=np.quantile(relative[np.isfinite(relative)], [.025, .975]) if np.isfinite(relative).any() else None,
                bootstrap='resample_subjects_keep_all_windows_and_paired_methods_together',
                interval_is_not='confirmatory_significance_or_equivalence_test')
    return result


def summarize_stage(cfg, out):
    tasks = read_json(Path(out)/'nonlinear_tasks.json')
    rows, missing = [], []
    for task in tasks:
        path = Path(out)/'tasks'/f'{task["id"]}.json'
        if path.exists(): rows.append(read_json(path))
        else: missing.append(task['id'])
    groups = {}
    for task in tasks:
        key = '|'.join(task[k] for k in ('split', 'candidate', 'condition', 'pairing'))
        groups.setdefault(key, dict(planned=0, terminal=0, successful=0, unavailable=0, failed=0, subjects=set(), rows=[]))['planned'] += 1
    for row in rows:
        key = '|'.join(row[k] for k in ('split', 'candidate', 'condition', 'pairing'))
        g = groups[key]; g['terminal'] += 1; g['subjects'].add(row['subject']); g['rows'].append(row)
        if row['status'].startswith('unavailable'): g['unavailable'] += 1
        elif row.get('success'): g['successful'] += 1
        else: g['failed'] += 1
    for g in groups.values():
        group_rows = g.pop('rows')
        successful = [r for r in group_rows if r.get('success')]
        success_values = _subject_score(successful, cfg['nonlinear']['failure_penalty_nmse'])
        g['success_only_subject_equal_nmse'] = float(np.mean(list(success_values.values()))) if success_values else None
        g['secondary_success_only'] = {}
        for metric in ('component_nmse', 'target_prediction_correlation', 'prediction_to_target_RMS_ratio'):
            component_values = []
            for component in range(2):
                metric_rows = []
                for row in successful:
                    value = row.get('waveform_metrics', {}).get(metric, [None, None])[component]
                    if value is not None: metric_rows.append(dict(row, nmse=value))
                metric_subjects = _subject_score(metric_rows)
                component_values.append(float(np.mean(list(metric_subjects.values()))) if metric_subjects else None)
            g['secondary_success_only'][metric] = component_values
        values = _subject_score(group_rows, cfg['nonlinear']['failure_penalty_nmse'])
        g['independent_subjects'] = len(g.pop('subjects'))
        g['subject_equal_nmse_with_failure_penalty'] = float(np.mean(list(values.values()))) if values else None
    summary = dict(schema='shared_driver_sensor_proxy_summary_v1', status='completed' if not missing else 'incomplete',
        planned=len(tasks), terminal=len(rows), missing=missing, groups=groups,
        evaluation_selection=read_json(Path(out)/'evaluation_selection.json'),
        interpretation='exploratory_public_development_reanalysis_not_unseen_confirmation',
        physical_reference='best_available_regularized_local_solution_not_global_unreachability_proof',
        paired_comparisons=paired_comparisons(cfg, rows), FIR=read_json(Path(out)/'fir_evaluation_scores.json'),
        training_template=read_json(Path(out)/'training_template_scores.json'), cross_hb_phase='not_estimated',
        failure_penalty_nmse=cfg['nonlinear']['failure_penalty_nmse'], scientific_verdict='requires_scientific_review')
    write_json(Path(out)/'summary.json', summary); return summary


def synthetic_stage(cfg, out, workers):
    """Ten real numerical tasks at the formal 300-point grid, no native reads."""
    from src.inference.shared_driver_reconstruction import nonlinear_driver_forward
    n = cfg['tensor']['steps']; t = np.arange(n)*cfg['tensor']['dt_s']
    reference = np.eye(n); reference[:, :50] -= 1/50
    sets = dict(nearest6=list(range(6)), nearest3=list(range(3)), nearest12=list(range(12)),
        outer6=list(range(6, 12)), sixth_to_seventh=list(range(5))+[6],
        nearest6_far6=list(range(6))+list(range(12, 18)))
    for i in range(3): sets[f'nearest6_random6_{i}'] = list(range(6))+list(range(6+6*i, 12+6*i))
    coord = dict(pc=np.r_[1., np.zeros(29)], eeg_factor=1., hb_factor=1., sd=[.025, .01, .005])
    rows = []
    for i in range(2):
        driver = .025*np.sin(2*np.pi*t/(12.+6*i)) + .008*np.cos(2*np.pi*t/22.+i)
        forward = nonlinear_driver_forward(driver, np.array([0., 1., 1., 1., 1.]), parameters(cfg), .1,
            substeps=8, derivative=False, numerical_backend='numba', hemodynamics='tak', viscoelastic_s=2.)
        canonical = forward['canonical_prediction']; raw = np.zeros((n, 30, 5))
        raw[:, 0, 0] = reference@driver
        rng = np.random.default_rng(cfg['seed']+i)
        raw[:, 1:, :] = (reference@rng.normal(scale=.002, size=(n, 145))).reshape(n, 29, 5)
        row = dict(id=f'synthetic-record-{i}', subject='synthetic_subject', split='selection',
            dataset='synthetic', record=f'synthetic-record-{i}', channel_sets=dict(sets=sets))
        meta, arrays = _paths(out, row['id'])
        write_npz(arrays, eeg_logpower=raw, hb=reference@canonical[:, 1:], eeg_operator=reference,
            hb_operator=reference, time_s=t, known_driver=driver, known_states=forward['states'])
        write_json(meta, dict(status='completed', row=row, source='known_fixed_Tak_generator_no_native_reads'))
        rows.append(row)
    screen = dict(coordinate=coord, models=dict(pc1=dict(coefficients=np.ones(5))), shortlist=['pc1'])
    tasks = []
    for i, row in enumerate(rows):
        for candidate, condition, pairing in (('Hb_only', 'joint', 'paired'), ('pc1', 'prediction', 'paired'),
                ('pc1', 'joint', 'paired'), ('pc1', 'prediction', 'wrong_EEG'), ('zero', 'prediction', 'paired')):
            task = dict(record_id=row['id'], native_record=row['record'], subject=row['subject'], split='selection', candidate=candidate,
                condition=condition, pairing=pairing, wrong_eeg_id=rows[1-i]['id'] if pairing == 'wrong_EEG' else None)
            task['id'] = token('|'.join(str(task[k]) for k in ('record_id', 'candidate', 'condition', 'pairing')))
            tasks.append(task)
    write_json(Path(out)/'synthetic_tasks.json', tasks)
    by_id = {r['id']: r for r in rows}
    started = time.monotonic()
    results = bounded_work(_nonlinear_job, [(cfg, str(out), task, by_id[task['record_id']],
        by_id.get(task['wrong_eeg_id']), screen) for task in tasks], workers, workers*2)
    summary = dict(schema='shared_driver_sensor_proxy_synthetic_v1', status='completed', planned=len(tasks),
        terminal=len(results), successful=sum(r.get('success', False) for r in results),
        failed=sum(not r.get('success', False) for r in results), elapsed_seconds=time.monotonic()-started,
        measured_arrays_read=0, dt_s=.1, steps=300, physical_model='fixed_Tak_p_equals_v', results=results,
        purpose='software_numerical_and_supervisor_throughput_check_not_measured_science')
    write_json(Path(out)/'synthetic_summary.json', summary)
    return summary


def _analysis_metric_mean(rows, name):
    """Average scalar/vector window metrics through native records and subjects."""
    first = next((r[name] for r in rows if r.get(name) is not None), None)
    if first is None: return None
    shape = np.asarray(first).shape
    values = np.full(shape or (), np.nan)
    for index in np.ndindex(shape or ()):
        metric_rows = []
        for row in rows:
            if row.get(name) is None: continue
            value = np.asarray(row[name], float)[index]
            if np.isfinite(value): metric_rows.append(dict(row, nmse=float(value), success=True))
        subjects = _subject_score(metric_rows)
        if subjects: values[index] = np.mean(list(subjects.values()))
    return float(values) if not shape else values


def _analysis_prediction_pair(cfg, planned_ids, left, right):
    """Pair successful retained scores by window, then bootstrap whole subjects."""
    common = [key for key in planned_ids if key in left and key in right and
              left[key].get('success') and right[key].get('success')]
    a, b = _subject_score([left[key] for key in common]), _subject_score([right[key] for key in common])
    subjects = sorted(set(a)&set(b))
    result = dict(planned_windows=len(planned_ids), common_success_windows=len(common),
        common_success_subjects=len(subjects), common_success_window_ids=common,
        excluded_window_ids=[key for key in planned_ids if key not in common])
    if subjects:
        av, bv = np.array([a[s] for s in subjects]), np.array([b[s] for s in subjects])
        draw = np.random.default_rng(cfg['seed']).integers(0, len(subjects),
            size=(cfg['endpoints']['bootstrap_repeats'], len(subjects)))
        result.update(left_nmse=float(av.mean()), right_nmse=float(bv.mean()),
            delta_nmse_left_minus_right=float((av-bv).mean()),
            delta_nmse_descriptive_95pct_interval=np.quantile(av[draw].mean(axis=1)-bv[draw].mean(axis=1), [.025, .975]),
            relative_improvement_left_over_right=float(1-av.mean()/bv.mean()) if bv.mean() > 0 else None,
            subject_ids=subjects, left_subject_nmse=a, right_subject_nmse=b)
    return result


def analyze_stage(out, analysis_output):
    """Read retained scores/residuals only; write a fresh derivative directory."""
    out, destination = Path(out).resolve(), Path(analysis_output).resolve()
    if destination == out or out not in destination.parents:
        raise ValueError('analysis output must be a fresh child directory within the completed run')
    if destination.exists(): raise ValueError('analysis output already exists; use a fresh versioned derivative')
    cfg = read_json(out/'resolved_config.json')
    if cfg.get('schema') != 'shared_driver_sensor_proxy_v1': raise ValueError('wrong retained experiment schema')
    summary = read_json(out/'summary.json')
    if summary.get('status') != 'completed' or summary.get('planned') != summary.get('terminal'):
        raise ValueError('read-only analysis requires a completed retained run')
    plan, screen = read_json(out/'plan.json'), read_json(out/'screen.json')
    tasks = read_json(out/'nonlinear_tasks.json')
    selection = read_json(out/'evaluation_selection.json')
    candidates = selection['candidates']
    if 'pc1' not in candidates: raise ValueError('retained comparison lacks PC1 baseline')
    cohort = [r for r in plan['rows'] if r['split'] == 'evaluation']
    identities = [r['id'] for r in cohort]
    if len(set(identities)) != len(identities): raise ValueError('duplicate planned evaluation window')
    metadata = {r['id']: dict(record_id=r['id'], native_record=r['record'], subject=r['subject']) for r in cohort}
    records = {}
    missing_tasks = []
    for task in tasks:
        path = out/'tasks'/f'{task["id"]}.json'
        if not path.exists(): missing_tasks.append(task['id']); continue
        row = read_json(path)
        if any(row.get(k) != task[k] for k in ('id', 'record_id', 'candidate', 'condition', 'pairing', 'split')):
            raise ValueError('retained task/result identity mismatch')
        records[task['id']] = row
    methods = ['Hb_only', *candidates]
    groups = {}
    exclusions = {}
    for candidate in methods:
        planned = [t for t in tasks if t['split'] == 'evaluation' and t['candidate'] == candidate and
                   t['condition'] == 'joint' and t['pairing'] == 'paired']
        if {t['record_id'] for t in planned} != set(identities): raise ValueError('joint/physical task plan denominator mismatch')
        by_id, excluded = {}, []
        for task in planned:
            row = records.get(task['id']); arrays = out/'tasks'/f'{task["id"]}.npz'
            if row is None: reason = 'missing_task_result'
            elif not row.get('success'): reason = row['status']
            elif not arrays.exists(): reason = 'missing_retained_arrays'
            else: by_id[task['record_id']] = row; continue
            excluded.append(dict(window_id=task['record_id'], task_id=task['id'], reason=reason))
        groups[candidate], exclusions[candidate] = by_id, excluded
    common = [key for key in identities if all(key in groups[c] for c in methods)]
    if not common: raise ValueError('no common successful physical/joint residual support')
    sd = np.asarray(screen['coordinate']['sd'], float)[1:]
    left, right = cfg['tensor']['score_steps']
    metrics = {candidate: [] for candidate in methods}
    physical_diagnostics = []
    per_window = []
    for key in common:
        residuals, targets = {}, []
        for candidate in methods:
            row = groups[candidate][key]
            with np.load(out/'tasks'/f'{row["id"]}.npz', allow_pickle=False) as arrays:
                residuals[candidate] = np.asarray(arrays['residual'], float)[left:right]/sd
                targets.append(np.asarray(arrays['target_hb'], float)[left:right])
            if residuals[candidate].shape != (right-left, 2) or not np.isfinite(residuals[candidate]).all():
                raise ValueError('retained residual scored support is incomplete or nonfinite')
        if any(not np.array_equal(targets[0], target) for target in targets[1:]):
            raise ValueError('paired methods do not retain identical Hb targets')
        window = dict(window_id=key, **metadata[key], methods={})
        for candidate in methods:
            e, reference = residuals[candidate], residuals['Hb_only']
            cosine, correlation = [], []
            for component in range(2):
                a, b = e[:, component], reference[:, component]
                norm = np.linalg.norm(a)*np.linalg.norm(b)
                cosine.append(float(a@b/norm) if norm > 0 else None)
                a, b = a-a.mean(), b-b.mean(); norm = np.linalg.norm(a)*np.linalg.norm(b)
                correlation.append(float(a@b/norm) if norm > 0 else None)
            entry = dict(metadata[key], component_nmse=np.mean(e**2, axis=0), joint_nmse=float(np.mean(e**2)),
                standardized_residual_change_RMS=np.sqrt(np.mean((e-residuals['pc1'])**2, axis=0)),
                standardized_residual_change_joint_RMS=float(np.sqrt(np.mean((e-residuals['pc1'])**2))),
                cosine_with_physical_residual=cosine, correlation_with_physical_residual=correlation)
            metrics[candidate].append(entry); window['methods'][candidate] = entry
        diagnostics = groups['Hb_only'][key]['residual_diagnostics']
        lags = np.asarray(diagnostics['lag_s'], float)
        indices = [int(np.flatnonzero(np.isclose(lags, lag))[0]) for lag in (1., 5.)]
        physical_diagnostics.append(dict(metadata[key], autocorrelation_1s_5s=np.asarray(diagnostics['lag_correlation'])[indices],
            spectral_band_power=np.asarray(diagnostics['spectral_band_power']),
            standardized_spectral_band_power=np.asarray(diagnostics['spectral_band_power'])/(sd[:, None]**2)))
        per_window.append(window)
    aggregated = {candidate: {metric: _analysis_metric_mean(rows, metric) for metric in (
        'component_nmse', 'joint_nmse', 'standardized_residual_change_RMS',
        'standardized_residual_change_joint_RMS', 'cosine_with_physical_residual', 'correlation_with_physical_residual')}
        for candidate, rows in metrics.items()}
    native_counts = {}
    for row in cohort:
        group = native_counts.setdefault(row['subject'], {}).setdefault(row['record'],
            dict(planned_windows=0, common_success_windows=0, common_success_window_ids=[]))
        group['planned_windows'] += 1
        if row['id'] in common:
            group['common_success_windows'] += 1; group['common_success_window_ids'].append(row['id'])
    prediction = {}
    for candidate in candidates:
        def prediction_rows(pairing):
            return {r['record_id']: r for r in records.values() if r['split'] == 'evaluation' and
                    r['candidate'] == candidate and r['condition'] == 'prediction' and r['pairing'] == pairing}
        true, wrong = prediction_rows('paired'), prediction_rows('wrong_EEG')
        prediction[f'{candidate}_true_vs_wrong'] = _analysis_prediction_pair(cfg, identities, true, wrong)
        zero = {r['record_id']: r for r in records.values() if r['split'] == 'evaluation' and r['candidate'] == 'zero'}
        prediction[f'{candidate}_true_vs_zero'] = _analysis_prediction_pair(cfg, identities, true, zero)
        template_path = out/'training_template_scores.json'
        if template_path.exists():
            template = {r['record_id']: r for r in read_json(template_path)['evaluation']}
            prediction[f'{candidate}_true_vs_template'] = _analysis_prediction_pair(cfg, identities, true, template)
    physical = dict(lag_s=[1., 5.], autocorrelation_1s_5s=_analysis_metric_mean(physical_diagnostics, 'autocorrelation_1s_5s'),
        spectral_bands_hz=diagnostics['spectral_bands_hz'],
        spectral_band_power=_analysis_metric_mean(physical_diagnostics, 'spectral_band_power'),
        standardized_spectral_band_power=_analysis_metric_mean(physical_diagnostics, 'standardized_spectral_band_power'),
        spectrum_rule=diagnostics['spectrum_rule'], source='retained_Hb_only_task_diagnostics_on_identical_common_support')
    result = dict(schema='shared_driver_sensor_proxy_residual_analysis_v1', status='completed', source_run=str(out),
        sources=['resolved_config.json', 'plan.json', 'screen.json', 'nonlinear_tasks.json', 'evaluation_selection.json',
                 'tasks/*.json', 'tasks/*.npz', 'training_template_scores.json'],
        split='evaluation', component_names=['HbO', 'HbR'], planned_windows=len(identities),
        common_success_windows=len(common), common_success_subjects=len({metadata[key]['subject'] for key in common}),
        common_success_window_ids=common, excluded_window_ids=[key for key in identities if key not in common],
        exclusions_by_method=exclusions, missing_task_ids=missing_tasks, native_record_counts=native_counts,
        aggregation='scored_time_mean_then_window_mean_within_native_record_then_record_mean_within_subject_then_subject_mean',
        score_steps=[left, right], training_sd=sd, methods=aggregated, physical_residual_diagnostics=physical,
        prediction_comparisons=prediction, per_window=per_window,
        definitions=dict(residual='target_minus_prediction', nmse='mean_of_squared_residual_divided_by_frozen_training_SD',
            residual_change='e_candidate_minus_e_PC1; RMS_per_window_then_record_then_subject',
            cosine='uncentered_signed_cosine_per_Hb_component; not_explained_variance',
            correlation='demeaned_Pearson_per_Hb_component; not_explained_variance',
            bootstrap='whole_subject_resampling_of_paired_record_aggregates_descriptive_95pct_not_confirmatory'),
        claim_limits=['common_success_subset_does_not_replace_full_failure_denominators',
                      'Hb_only_is_best_available_regularized_local_fit_not_global_physical_unreachability',
                      'residual_alignment_does_not_identify_physiological_sources'], native_reads=0, new_fits=0)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination/'residual_summary.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--stage', choices=STAGES, default='check')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--analysis-output', type=Path)
    args = parser.parse_args(argv)
    if args.stage == 'analyze':
        if args.output is None or args.analysis_output is None:
            parser.error('analyze requires --output <completed_run> and --analysis-output <fresh_child_directory>')
        result = analyze_stage(args.output, args.analysis_output)
        print(json.dumps(serial(dict(status=result['status'], planned_windows=result['planned_windows'],
            common_success_windows=result['common_success_windows'], methods=result['methods'])), ensure_ascii=False, indent=2))
        return 0
    cfg = read_config(args.config)
    workers = args.workers or cfg['resources']['max_workers']
    if not 1 <= workers <= cfg['resources']['max_workers']: raise ValueError('workers exceed bounded config')
    if args.stage == 'check': print(json.dumps(check(cfg), ensure_ascii=False, indent=2)); return 0
    if args.output is None: parser.error('--output is required for artifact-producing stages')
    root = args.project_root.resolve(); out = args.output.resolve(); artifact = (root/ARTIFACT_ROOT).resolve()
    if out == artifact or artifact not in out.parents: raise ValueError('fresh named run required below versioned artifact root')
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        resolved = out/'resolved_config.json'
        if resolved.exists() and read_json(resolved) != serial(cfg): raise ValueError('resolved configuration changed; fresh run required')
        write_json(resolved, cfg)
        stages = ('plan', 'prepare', 'screen', 'nonlinear', 'summarize') if args.stage == 'run' else (args.stage,)
        for stage in stages:
            write_json(out/'progress.json', dict(stage=stage, status='running', pid=os.getpid(),
                updated_utc=datetime.now(timezone.utc).isoformat(), workers=workers, numerical_threads=1))
            if stage == 'synthetic': result = synthetic_stage(cfg, out, workers)
            elif stage == 'plan': result = create_plan(cfg, out, root)
            elif stage == 'prepare': result = prepare_stage(cfg, out, root, read_json(out/'plan.json'), workers)
            elif stage == 'screen': result = screen_stage(cfg, out, read_json(out/'plan.json'))
            elif stage == 'nonlinear': result = nonlinear_stage(cfg, out, read_json(out/'plan.json'), read_json(out/'screen.json'), workers)
            else: result = summarize_stage(cfg, out)
            write_json(out/'progress.json', dict(stage=stage, status='completed', updated_utc=datetime.now(timezone.utc).isoformat()))
        print(json.dumps(serial(result if isinstance(result, dict) else dict(stage=stages[-1], terminal=len(result))), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        analyzing = '--stage' in sys.argv and sys.argv[sys.argv.index('--stage')+1] == 'analyze'
        if '--output' in sys.argv and not analyzing:
            destination = Path(sys.argv[sys.argv.index('--output')+1])
            if destination.exists():
                write_json(destination/'progress.json', dict(status='failed', error=str(error), traceback=traceback.format_exc(),
                    updated_utc=datetime.now(timezone.utc).isoformat()))
        raise
