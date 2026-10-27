#!/usr/bin/env python3
"""Frozen public-model compatibility panels; no raw or protected-data reader.

This module owns admission, projection panels and resumable task artifacts.
The census command calls ``run_panels``; it never imports another runner's
private readers. Numerical projection code lives in inference/model_compatibility.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.model_compatibility import linear_panel_profiles
from src.inference.observation_baselines import transform_spatial_coordinates
from src.inference.shared_driver_reconstruction import fit_vascular_flow_relaxation
from src.inference.source_observability import hb_representation_lower_bound
from src.inference.t3a_balloon_robust_ssm import (
    BalloonFixedParameters, BalloonFreeParameters, BalloonParameters,
)

OUTPUT_SUPPORT_RUNS = (
    'experiments/runs/physiology_semantic_tokenizer/ssm_output_support/20261010_v1',
    'experiments/runs/physiology_semantic_tokenizer/ssm_output_support/20261013_refed_v2',
)
SPATIAL_RUN = 'experiments/runs/physiology_semantic_tokenizer/shared_driver_spatial/20261020_measured_v2'
ARTIFACT_ROOT = 'experiments/runs/physiology_semantic_tokenizer/model_compatibility'
DATASETS = ('eeg_fnirs_single_trial', 'simultaneous_eeg_nirs', 'visual_cognitive_motivation', 'refed')
SPATIAL_ARMS = ('K1_true_H4_E12', 'K2_true_H4_E12')


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [serial(v) for v in value]
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n')
    temporary.replace(path)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def token(value):
    return hashlib.sha256(json.dumps(serial(value), sort_keys=True).encode()).hexdigest()[:24]


def inside(parent, relative):
    """Require every metadata/array path to remain in its exact retained parent."""
    parent = Path(parent)
    path = parent/relative
    if not path.resolve().is_relative_to(parent.resolve()):
        raise ValueError('retained parent path escaped through a symlink')
    return path


def panel_config(cfg):
    if cfg.get('schema') == 'model_compatibility_cached_v2':
        raise ValueError('Cached census requires newly fitted cached parents; retained native panels are historical replay only')
    p = cfg['panels']
    if (tuple(p['output_support_runs']) != OUTPUT_SUPPORT_RUNS
            or p['spatial_run'] != SPATIAL_RUN or p['lengths_s'] != [30, 90]
            or p['linear']['tau_grid'] != list(np.arange(.5, 8.001, .5))
            or p['linear']['kappa_grid'] != [.2, .4, .64, 1., 1.5]
            or p['linear']['tolerances'] != [.25, .5, 1.]):
        raise ValueError('frozen panel identity, sample support or engineering grid changed')
    if cfg.get('data', {}).get('protected_data', 'forbidden') != 'forbidden':
        raise ValueError('protected data are forbidden for the compatibility panels')
    nonlinear = p['nonlinear']
    if (nonlinear['substeps'] != 4 or nonlinear['validation_substeps'] != 8
            or isinstance(nonlinear['max_nfev'], bool)
            or not isinstance(nonlinear['max_nfev'], int) or nonlinear['max_nfev'] < 1
            or not 0 < p['linear'].get('rcond', 1e-10) < 1):
        raise ValueError('positive nonlinear budget and fixed 4/8 integration required')
    return p


def parameters(cfg, tau=None, kappa=None):
    values = dict(cfg.get('parameters', {}))
    values.pop('policy', None)
    default_tau = cfg['panels']['linear'].get('nominal_tau', 2.)
    default_kappa = cfg['panels']['linear'].get('nominal_kappa', .64)
    default_tau = values.pop('tau', default_tau)
    default_kappa = values.pop('kappa', default_kappa)
    p = BalloonParameters(fixed=BalloonFixedParameters(**values),
        free=BalloonFreeParameters(tau=default_tau if tau is None else tau,
                                  kappa=default_kappa if kappa is None else kappa))
    p.validate()
    return p


def output_support_sources(cfg, root=ROOT):
    """Metadata-only admission of the exact two completed public parents."""
    p = panel_config(cfg)
    sources = {}
    for relative in p['output_support_runs']:
        parent = Path(root)/relative
        if parent.resolve() != Path(root).resolve()/relative:
            raise ValueError('explicit retained parent may not redirect')
        summary = read_json(inside(parent, 'summary.json'))
        plan = read_json(inside(parent, 'plan.json'))
        pcfg = yaml.safe_load(inside(parent, 'resolved_config.yaml').read_text())
        coordinate = read_json(inside(parent, 'coordinates.json'))
        expected_schema = 'ssm_output_support_v1' if relative == OUTPUT_SUPPORT_RUNS[0] else 'ssm_output_support_refed_v2'
        if (summary.get('schema') != 'ssm_output_support_summary_v1'
                or summary.get('status') != 'completed'
                or plan.get('schema') != 'ssm_output_support_plan_v1'
                or pcfg.get('schema') != expected_schema
                or pcfg.get('data', {}).get('protected_data') != 'forbidden'):
            raise ValueError('completed public output-support boundary violated')
        rows = []
        for row in plan['records']:
            if row['partition'] != 'evaluation':
                continue
            if (row['dataset'] not in DATASETS or row['duration_s'] != 90
                    or Path(row['id']).name != row['id'] or row['id'] in ('.', '..')
                    or row['coordinate'] not in coordinate
                    or row['subject'] in coordinate[row['coordinate']]['training_subjects']
                    or row['subject'] not in plan['evaluation_subjects'][row['dataset']]):
                raise ValueError('public evaluation identity or frozen-coordinate split violated')
            rows.append(dict(row, parent=relative))
        sources[relative] = dict(path=parent, plan=plan, coordinate=coordinate, rows=rows)
    identities = [(r['dataset'], r['id']) for s in sources.values() for r in s['rows']]
    if len(identities) != len(set(identities)):
        raise ValueError('duplicate output-support evaluation identity')
    return sources


def crop_output_panel(y, operators, length_s):
    """Use retained bins and cancel the original reference before cropping."""
    y = np.asarray(y, float)
    if y.shape != (360, 3) or length_s not in (30, 90) or not np.isfinite(y).all():
        raise ValueError('finite retained [360,3] target and 30/90 second support required')
    left, right = (120, 240) if length_s == 30 else (0, 360)
    n = right-left
    reference = np.eye(n)
    reference[:, :20] -= 1/20
    target = reference@y[left:right]
    operator = np.zeros((3*n, 3*n))
    outside = np.r_[np.arange(left), np.arange(right, 360)]
    for j, name in enumerate(('eeg', 'hb', 'hb')):
        original = np.asarray(operators[name], float)
        if original.shape != (360, 360) or not np.isfinite(original).all():
            raise ValueError('finite actual parent [360,360] operators required')
        complete = reference@original[left:right]
        if outside.size and np.max(abs(complete[:-1, outside])) > 1e-12:
            raise ValueError('cropped scored bins depend on outside-window observations')
        operator[j::3, j::3] = complete[:, left:right]
    score_mask = np.ones((n, 3), bool)
    score_mask[-1] = False
    return target, operator, score_mask


def load_output_panel(cfg, ref, length_s, root=ROOT):
    """Recheck metadata and exact identity before the only prepared array read."""
    sources = output_support_sources(cfg, root)
    if ref.get('parent') not in sources:
        raise ValueError('unregistered output-support parent')
    source = sources[ref['parent']]
    if ref not in source['rows']:
        raise ValueError('only an exact retained evaluation identity can select arrays')
    co = source['coordinate'][ref['coordinate']]
    scale = np.asarray(co['sd'], float)
    pc = np.asarray(co['pc'], float)
    factors = np.asarray([co['eeg_factor'], co['hb_factor']], float)
    if (scale.shape != (3,) or pc.shape != (30,) or not np.isfinite(pc).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)
            or not np.isfinite(factors).all() or np.any(factors <= 0)):
        raise ValueError('finite frozen training PC, positive gains and SD required')
    prepared = read_json(inside(source['path'], f"tasks/prepare/{ref['id']}.json"))
    if prepared.get('status') != 'completed':
        raise ValueError('retained preparation did not complete before array read')
    path = inside(source['path'], f"prepared/{ref['id']}.npz")
    with np.load(path, allow_pickle=False) as values:
        arrays = {k: values[k].copy() for k in ('eeg', 'hb', 'eeg_operator', 'hb_operator')}
    if (arrays['eeg'].shape != (360, 30) or arrays['hb'].shape != (360, 2)
            or any(not np.isfinite(a).all() for a in arrays.values())):
        raise ValueError('finite complete frozen parent arrays required')
    constant_residual = []
    for name in ('eeg_operator', 'hb_operator'):
        actual = arrays[name]
        if actual.shape != (360, 360):
            raise ValueError('finite actual parent [360,360] operators required')
        error = float(np.max(np.abs(actual@np.ones(360))))
        if error > 1e-10:
            raise ValueError('unsupported parent observation operator: constants must be annihilated for comparable linear/nonlinear panels')
        constant_residual.append(error)
    y = np.column_stack((arrays['eeg']@pc*factors[0], arrays['hb']*factors[1]))
    target, operator, mask = crop_output_panel(y,
        dict(eeg=arrays['eeg_operator'], hb=arrays['hb_operator']), length_s)
    # The final interpolation bin is outside both fits and scores. Check each
    # modality constant on exactly the common support used by both models.
    constants = np.tile(np.eye(3), (len(target), 1))
    scored_constant_error = float(np.max(np.abs((operator@constants)[mask.ravel()])))
    if scored_constant_error > 1e-10:
        raise ValueError('unsupported cropped observation operator: scored modality constants must be annihilated')
    return dict(target=target, operator=operator, sd=scale, score_mask=mask,
                pc=pc, eeg_factor=factors[0], hb_factor=factors[1],
                parent_constant_null_max_abs=np.asarray(constant_residual),
                scored_constant_null_max_abs=scored_constant_error)


def spatial_source(cfg, root=ROOT):
    panel_config(cfg)
    source = Path(root)/SPATIAL_RUN
    if source.resolve() != Path(root).resolve()/SPATIAL_RUN:
        raise ValueError('explicit spatial parent may not redirect')
    names = ('run_identity.json', 'resolved_config.json', 'plan.json', 'coordinate.json',
             'summary.json', 'verification.json')
    identity, parent, plan, coordinate, summary, verification = [read_json(inside(source, n)) for n in names]
    if (identity.get('schema') != 'shared_driver_spatial_v1' or identity.get('family') != 'public'
            or parent.get('schema') != 'shared_driver_spatial_v1'
            or parent.get('data', {}).get('datasets') != ['eeg_fnirs_single_trial']
            or parent['data'].get('protected_data') != 'forbidden'
            or parent['tensor'].get('steps') != 120 or parent['tensor'].get('dt_s') != .25
            or parent['tensor'].get('temporal_modes') != 16
            or parent['evaluation'].get('score_steps') != [20, 120]
            or summary.get('execution') != 'completed' or summary.get('planned') != summary.get('terminal')
            or verification.get('status') != 'passed' or plan.get('status') != 'available'):
        raise ValueError('completed independently verified public spatial parent required')
    groups, ids = {}, set()
    for row in plan['rows']:
        if (row.get('dataset') != 'eeg_fnirs_single_trial'
                or row.get('split') not in ('train', 'selection', 'evaluation')
                or row['id'] in ids or row['split'] != groups.setdefault(row['subject'], row['split'])):
            raise ValueError('spatial public identity or subject split changed')
        ids.add(row['id'])
    training = {r['id'] for r in plan['rows'] if r['split'] == 'train'}
    sd = np.asarray(coordinate['sd'], float)
    if (set(coordinate['training_ids']) != training or sd.shape != (68,)
            or not np.isfinite(sd).all() or np.any(sd <= 0)):
        raise ValueError('positive frozen spatial SD and exact training identities required')
    calibrations = {}
    for arm in SPATIAL_ARMS:
        calibration = read_json(inside(source, f'calibration/{arm}.json'))
        k = int(arm[1])
        loading = np.asarray(calibration['design']['hb_loadings'], float)
        if (calibration.get('status') != 'completed'
                or calibration.get('evaluation_data_used') is not False
                or not set(calibration['readout']['training_ids']) <= training
                or loading.shape != (4, k) or not np.isfinite(loading).all()
                or np.any(loading < 0) or not np.allclose(loading.sum(1), 1.)
                or np.linalg.matrix_rank(loading) != k
                or calibration.get('physical_hb_gain') != 1.
                or calibration.get('hb_coordinate_gain') != 1.):
            raise ValueError('exact frozen train-only Hb loading and gain required')
        calibrations[arm] = calibration
    return dict(path=source, plan=plan, coordinate=coordinate, calibrations=calibrations)


def load_spatial_panel(cfg, row, root=ROOT):
    source = spatial_source(cfg, root)
    if row not in source['plan']['rows'] or row.get('split') != 'evaluation':
        raise ValueError('only exact spatial evaluation identities may select arrays')
    key = hashlib.sha256(row['id'].encode()).hexdigest()[:24]
    metadata = read_json(inside(source['path'], f'prepared/{key}.json'))
    if metadata.get('status') != 'completed' or metadata.get('row') != row:
        raise ValueError('spatial preparation identity failed before array read')
    path = inside(source['path'], f'prepared/{key}.npz')
    with np.load(path, allow_pickle=False) as values:
        arrays = {k: values[k].copy() for k in ('eeg', 'hb', 'hb_operator')}
    if (arrays['eeg'].shape != (120, 12, 5) or arrays['hb'].shape != (120, 4, 2)
            or arrays['hb_operator'].shape != (120, 120)
            or any(not np.isfinite(v).all() for v in arrays.values())):
        raise ValueError('finite complete frozen 120-step spatial arrays required')
    raw_hb = arrays['hb'].copy()
    _, arrays['hb'] = transform_spatial_coordinates(arrays['eeg'], arrays['hb'], source['coordinate'])
    return source, dict(hb=arrays['hb'], raw_hb=raw_hb, hb_operator=arrays['hb_operator'])


def panel_plan(cfg, root=ROOT):
    sources = output_support_sources(cfg, root)
    refs = [row for source in sources.values() for row in source['rows']]
    counts = {ds: sum(r['dataset'] == ds for r in refs) for ds in DATASETS}
    subjects = {ds: len({r['subject'] for r in refs if r['dataset'] == ds}) for ds in DATASETS}
    spatial = spatial_source(cfg, root)
    rows = [r for r in spatial['plan']['rows'] if r['split'] == 'evaluation']
    if (counts != {ds: 12 for ds in DATASETS} or subjects != {ds: 6 for ds in DATASETS}
            or len(rows) != 36 or len({r['subject'] for r in rows}) != 6):
        raise ValueError('frozen full 48-record/24-subject and spatial 36-window/6-subject denominators changed')
    output = [dict(kind='linear', ref=ref, length_s=length)
              for ref in refs for length in cfg['panels']['lengths_s']]
    for task in output:
        task['id'] = token(dict(ref=task['ref'], length_s=task['length_s']))
    space = [dict(kind='space', row=row, arm=arm, id=token(dict(row=row, arm=arm)))
             for row in rows for arm in SPATIAL_ARMS]
    return dict(schema='model_compatibility_panels_plan_v1', output_tasks=output, space_tasks=space,
        output_records=48, output_subjects=24, output_record_lengths=96, spatial_windows=36,
        spatial_subjects=6, spatial_arm_windows=72, dataset_records=counts, dataset_subjects=subjects,
        measured_arrays_read=0, protected_data_read=False, raw_data_read=False,
        config_identity=token(cfg), selection='all_exact_public_parent_evaluation_records',
        threshold_interpretation='discrete_engineering_NRMSE_nodes_not_noise_calibrated_or_confidence_sets')


def compatibility_sets(profiles, tolerances, taus, kappas):
    """Require EEG, HbO and HbR tolerance at each SAME joint solution/node."""
    rows = []
    for tolerance in tolerances:
        hb_cross = [r for r in profiles if np.isfinite(r['hb_nmse'])
                    and np.isfinite(r['cross_increment_nmse'])
                    and r['hb_nmse'] <= tolerance**2
                    and r['cross_increment_nmse'] <= tolerance**2]
        selected = [r for r in profiles if all(np.isfinite(r[name]) and r[name] <= tolerance**2
                    for name in ('joint_eeg_nmse', 'joint_hbo_nmse', 'joint_hbr_nmse'))]
        row = dict(tolerance=tolerance, n_nodes=len(selected), n_joint_nodes=len(selected),
                   nodes=[],
                   discovered_common_witness_nodes=len(selected),
                   witness_status='found' if selected else 'no_witness_found',
                   empty_set_certifies_infeasibility=False,
                   n_hb_cross_nodes=len(hb_cross),
                   hb_cross_nodes=[[r['tau'], r['kappa']] for r in hb_cross],
                   tau_min=None, tau_max=None, kappa_min=None, kappa_max=None,
                   tau_width=None, kappa_width=None, hit_boundary=False,
                   max_normalized_distance_to_boundary=None, all_nodes_on_boundary=None,
                   selection='discovered_same_theta_same_joint_projection_each_EEG_HbO_HbR_NRMSE_le_tolerance')
        if selected:
            tt, kk = [r['tau'] for r in selected], [r['kappa'] for r in selected]
            row.update(tau_min=min(tt), tau_max=max(tt), kappa_min=min(kk), kappa_max=max(kk),
                       tau_width=max(tt)-min(tt), kappa_width=max(kk)-min(kk),
                       hit_boundary=any(r['tau'] in (min(taus), max(taus)) or
                                        r['kappa'] in (min(kappas), max(kappas)) for r in selected),
                       nodes=[[r['tau'], r['kappa']] for r in selected])
            interior = [min((r['tau']-min(taus))/(max(taus)-min(taus)),
                            (max(taus)-r['tau'])/(max(taus)-min(taus)),
                            (r['kappa']-min(kappas))/(max(kappas)-min(kappas)),
                            (max(kappas)-r['kappa'])/(max(kappas)-min(kappas))) for r in selected]
            row['max_normalized_distance_to_boundary'] = max(interior)
            row['all_nodes_on_boundary'] = bool(max(interior) <= 1e-12)
        rows.append(row)
    return rows


def _identity(task):
    ref = task['ref']
    return dict(id=task['id'], record_id=ref['id'], dataset=ref['dataset'], subject=ref['subject'],
                parent=ref['parent'], length_s=task['length_s'])


def _linear_task(cfg, task, folder, root):
    record = load_output_panel(cfg, task['ref'], task['length_s'], root)
    linear = cfg['panels']['linear']
    result = linear_panel_profiles(record['target'], record['operator'], record['sd'],
        linear['tau_grid'], linear['kappa_grid'], record['score_mask'], dt=.25,
        parameters=parameters(cfg), rcond=linear.get('rcond', 1e-10), return_predictions=True)
    nominal = next(row for row in result['profiles'] if row['tau'] == parameters(cfg).free.tau
                   and row['kappa'] == parameters(cfg).free.kappa)
    sets = compatibility_sets(result['profiles'], linear['tolerances'],
                              linear['tau_grid'], linear['kappa_grid'])
    for subset in sets:
        subset['dynamic_support'] = result['dynamic_support']
        subset['evidence_interpretation'] = result['interpretation']
    # Prediction arrays are removed from JSON; selected witnesses retain the
    # complete common projection, target and operators in one small audit NPZ.
    arrays = {k: record[k] for k in ('target', 'operator', 'sd', 'score_mask', 'pc', 'eeg_factor', 'hb_factor')}
    for key in ('parent_constant_null_max_abs', 'scored_constant_null_max_abs'):
        if key in record:
            arrays[key] = record[key]
    for label, node in result['predictions'].items():
        for name in ('joint_prediction', 'hb_prediction', 'joint_coefficients', 'hb_coefficients'):
            if name in node:
                arrays[f'{label}_{name}'] = node[name]
    cleaned = [{k: v for k, v in row.items() if not isinstance(v, np.ndarray)} for row in result['profiles']]
    for row in cleaned:
        row['joint_residual_dimension'] = result['score_count']-row['joint_rank']
        row['hb_residual_dimension'] = result['hb_score_count']-row['hb_rank']
        for name in ('hb', 'joint', 'joint_hb', 'eeg', 'joint_hbo', 'joint_hbr', 'hb_hbo', 'hb_hbr'):
            row[name+'_nrmse'] = float(np.sqrt(row[name+'_nmse']))
    lookup = {(r['tau'], r['kappa']): r for r in cleaned}
    best_hb = lookup[(result['best_hb']['tau'], result['best_hb']['kappa'])]
    best_joint = lookup[(result['best_joint']['tau'], result['best_joint']['kappa'])]
    nominal = lookup[(nominal['tau'], nominal['kappa'])]
    scaled = record['target']/record['sd']
    centered = scaled-scaled[record['score_mask'][:, 0]].mean(axis=0)
    centered_rms = np.sqrt(np.mean(centered[record['score_mask'][:, 0]]**2, axis=0))
    summary = dict(_identity(task), status='completed', nominal=nominal,
        best_hb=best_hb, best_joint=best_joint, profiles=cleaned, compatibility_sets=sets,
        score_count=result['score_count'], hb_score_count=result['hb_score_count'],
        eeg_score_count=result['eeg_score_count'],
        target_centered_training_sd_rms=centered_rms,
        zero_prediction_nmse=result['zero_prediction_nmse'],
        zero_prediction_energy=result['zero_prediction_energy'],
        dynamic_nmse=result['dynamic_nmse'], dynamic_energy=result['dynamic_energy'],
        dynamic_support=result['dynamic_support'], evidence_interpretation=result['interpretation'],
        low_signal=bool(np.any(centered_rms < .1)),
        low_signal_threshold_training_sd_rms=.1,
        low_signal_interpretation='descriptive_engineering_scale_flag_not_estimated_measurement_noise',
        normalization='frozen_parent_training_SD_SSE_divided_by_number_of_scored_scalar_observations',
        initial_state='five_unrestricted_linearized_initial_coordinates',
        parameter_selection='minimum_joint_NMSE_same_theta_for_every_component_and_nonlinear_followup')
    write_npz(folder/(task['id']+'.npz'), **arrays)
    return summary


def _nonlinear_task(cfg, task, folder, root):
    linear_path = folder.parent/'linear'/(task['id']+'.json')
    linear = read_json(linear_path)
    if linear.get('status') != 'completed' or any(linear.get(k) != v for k, v in _identity(task).items()):
        raise ValueError('matching completed linear panel required for same-theta nonlinear fit')
    chosen = linear['best_joint']
    record = load_output_panel(cfg, task['ref'], task['length_s'], root)
    n = len(record['target'])
    hb_indices = np.arange(3*n).reshape(n, 3)[:, 1:].ravel()
    hb_operator = record['operator'][np.ix_(hb_indices, hb_indices)]
    p = cfg['panels']['nonlinear']
    # Two deterministic, target-independent starts, with every time-point flow
    # coefficient free. No HbO/HbR coordinate inversion or physiological fit.
    starts = [dict(coefficients=np.zeros(n), initial_state=np.ones(3)),
              dict(coefficients=np.zeros(n), initial_state=np.array([1.02, 1.04, .98]))]
    result = fit_vascular_flow_relaxation(record['target'][:, 1:],
        parameters(cfg, chosen['tau'], chosen['kappa']), .25,
        hb_operator=hb_operator, sd=record['sd'][1:], score_mask=record['score_mask'][:, 1:],
        starts=starts, max_nfev=p['max_nfev'], substeps=p['substeps'],
        validation_substeps=p['validation_substeps'],
        integration_tolerance=p.get('integration_tolerance', 1e-4),
        optimizer_tolerance=p.get('optimizer_tolerance', 1e-10),
        log_flow_bounds=p.get('log_flow_bounds', [-np.log(4), np.log(4)]),
        initial_log_bounds=p.get('initial_log_bounds', [-np.log(2), np.log(2)]),
        joint_hb_tolerance=.5)
    arrays = dict(target=record['target'][:, 1:], score_mask=record['score_mask'][:, 1:],
                  sd=record['sd'][1:], hb_operator=hb_operator)
    for name in ('prediction', 'canonical_prediction', 'coefficients', 'initial_state', 'states', 'flow', 'log_flow'):
        if result.get(name) is not None:
            arrays[name] = result[name]
    write_npz(folder/(task['id']+'.npz'), **arrays)
    count = int(record['score_mask'][:, 1:].sum())
    legal_sse = result.get('weighted_data_sse') if result['legal_witness'] else None
    nmse = None if legal_sse is None else float(legal_sse/count)
    details = {key: value for key, value in result.items() if key in (
        'status', 'legal_witness', 'optimizer_converged', 'joint_hb_witness', 'starts', 'evaluations',
        'validation_evaluations', 'retained_candidates', 'selected_start', 'selected_evaluation',
        'physical_checks', 'integration', 'weighted_data_sse', 'by_chromophore', 'optimizer_status',
        'selection', 'minimum_legal_weighted_data_sse', 'incompatibility_certified', 'global_lower_bound_certified')}
    return dict(_identity(task), status='completed', solver_status=result['status'],
        selected_tau=chosen['tau'], selected_kappa=chosen['kappa'],
        theta_selection='linear_joint_NMSE_minimum_same_record_length_even_if_no_threshold_node',
        dynamic_support=linear.get('dynamic_support'), low_signal=linear.get('low_signal'),
        D_Hb_NL_recon_nmse=nmse, D_Hb_NL_recon_nrmse=None if nmse is None else float(np.sqrt(nmse)),
        score_count=count, legal_witness=result['legal_witness'],
        optimizer_converged=result['optimizer_converged'],
        interpretation='found_legal_conditional_forward_reconstruction_error_upper_bound_not_global_exclusion',
        shared_EEG_Hb_witness=False,
        kappa_role='retained_linear_theta_metadata_not_used_by_free_flow_vascular_subsystem',
        temporal_basis='identity_one_free_log_flow_value_per_retained_model_time_point', details=details)


def spatial_distances(hb, raw_hb, operator, loading, sd):
    """Keep native energy fractions and training-SD lower bounds separate."""
    raw_hb, operator, loading = (np.asarray(v, float) for v in (raw_hb, operator, loading))
    n = len(hb)
    if raw_hb.shape != np.shape(hb) or raw_hb.shape[1:] != (4, 2) or not np.isfinite(raw_hb).all():
        raise ValueError('finite native Hb targets with the same [T,4,2] support required')
    if operator.shape == (4, n, n):
        if not np.array_equal(operator, np.broadcast_to(operator[0], operator.shape)):
            raise ValueError('stable spatial relaxation requires a common temporal operator across channels')
    elif operator.shape != (n, n):
        raise ValueError('stable spatial relaxation requires a common temporal operator')
    if not np.isfinite(operator).all() or np.linalg.matrix_rank(loading) != loading.shape[1]:
        raise ValueError('finite common operator and full-rank frozen spatial loading required')
    mask = np.zeros(np.shape(hb), bool)
    mask[20:] = True
    weighted = hb_representation_lower_bound(hb, hb_loadings=loading, hb_gains=np.ones(4),
        sd=sd, score_mask=mask, hb_operator=operator)
    # A common temporal operator preserves the frozen spatial span. Relaxing
    # its temporal constraints therefore gives a spatial lower bound directly,
    # including when the time operator is singular. Avoid ill-conditioned
    # latent-coordinate inversion followed by numerical forward cancellation.
    spatial_basis = np.linalg.qr(loading, mode='reduced')[0]
    raw_prediction = np.stack([raw_hb[:, :, j]@spatial_basis@spatial_basis.T for j in range(2)], axis=2)
    residual = raw_prediction-raw_hb
    energy = float(np.sum(np.asarray(raw_hb)[mask]**2))
    residual_energy = float(np.sum(residual[mask]**2))
    values = dict(D_space_fraction_raw=None if energy <= np.finfo(float).tiny else residual_energy/energy,
        raw_energy_status='ok' if energy > np.finfo(float).tiny else 'insufficient_energy',
        raw_energy=energy, raw_residual_energy=residual_energy,
        weighted_nmse=weighted['nmse'], weighted_nrmse=weighted['nrmse'],
        weighted_lower_bound_nmse=weighted['certified_nmse'],
        weighted_lower_bound_nrmse=weighted['certified_nrmse'],
        raw_spatial_projection_interpretation='stable_fixed_spatial_projection_relax_temporal_constraints',
        projection_rank=weighted['rank'], spatial_rank=int(np.linalg.matrix_rank(loading)),
        certified_relaxation=weighted['certificate_relaxed'], score_count=int(mask.sum()))
    stacked = np.asarray(raw_hb)[20:].transpose(0, 2, 1).reshape(-1, 4)
    joint_singular = np.linalg.svd(stacked, compute_uv=False)
    joint_rank_energy = float(np.sum(joint_singular[loading.shape[1]:]**2))
    values['joint_free_rank_K_raw_residual_energy'] = joint_rank_energy
    values['D_rank_joint'] = None if energy <= np.finfo(float).tiny else joint_rank_energy/energy
    # A separately named best rank-K fit is a more generous representation;
    # it cannot replace the distance to the frozen loading.
    for j, name in enumerate(('HbO', 'HbR')):
        y = np.asarray(raw_hb)[20:, :, j]
        singular = np.linalg.svd(y, compute_uv=False)
        native_energy = float(np.sum(y**2))
        k = loading.shape[1]
        values[f'{name}_raw_energy'] = native_energy
        values[f'{name}_raw_residual_energy'] = float(np.sum(residual[20:, :, j]**2))
        values[f'{name}_D_space_fraction_raw'] = None if native_energy <= np.finfo(float).tiny else values[f'{name}_raw_residual_energy']/native_energy
        values[f'{name}_free_rank_K_fraction_raw'] = None if native_energy <= np.finfo(float).tiny else float(np.sum(singular[k:]**2)/native_energy)
        values[f'{name}_weighted_nmse'] = weighted['by_chromophore'][name]['nmse']
        values[f'{name}_weighted_lower_bound_nmse'] = weighted['certified_by_chromophore'][name]['nmse']
    return values, dict(target=hb, raw_target=raw_hb, prediction=weighted['prediction'],
        certified_prediction=weighted['certified_prediction'], raw_prediction=raw_prediction,
        score_mask=mask, hb_operator=operator, hb_loadings=loading, hb_gains=np.ones(4), sd=sd)


def _space_task(cfg, task, folder, root):
    source, record = load_spatial_panel(cfg, task['row'], root)
    loading = np.asarray(source['calibrations'][task['arm']]['design']['hb_loadings'], float)
    sd = np.asarray(source['coordinate']['sd'], float)[60:].reshape(4, 2)
    values, arrays = spatial_distances(record['hb'], record['raw_hb'], record['hb_operator'], loading, sd)
    write_npz(folder/(task['id']+'.npz'), **arrays)
    return dict(id=task['id'], record_id=task['row']['id'], dataset=task['row']['dataset'],
        subject=task['row']['subject'], arm=task['arm'], K=loading.shape[1], status='completed',
        parent=SPATIAL_RUN, score_steps=[20, 120], **values,
        interpretation='fixed_training_Hb_readout_free_chromophore_curves_relax_all_vascular_dynamics')


def _execute(payload):
    cfg, task, kind, out, root = payload
    folder = Path(out)/'tasks'/kind
    path = folder/(task['id']+'.json')
    if path.exists():
        return read_json(path)
    started = time.monotonic()
    try:
        function = dict(linear=_linear_task, nonlinear=_nonlinear_task, space=_space_task)[kind]
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            result = function(cfg, task, folder, root)
    except Exception as error:
        result = dict(id=task['id'], status='failed', task=task,
                      error=repr(error), traceback=traceback.format_exc())
    result['elapsed_s'] = time.monotonic()-started
    write_json(path, result)
    return result


def _csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(serial(v), ensure_ascii=False) if isinstance(v, (list, dict)) else serial(v)
                             for k, v in row.items()})
    temporary.replace(path)


def export_panel_tables(out, plan):
    out = Path(out)
    profile, summary, sets, nonlinear, space = [], [], [], [], []
    for task in plan['output_tasks']:
        identity = _identity(task)
        path = out/'tasks'/'linear'/(task['id']+'.json')
        if path.exists():
            result = read_json(path)
            row = dict(identity, status=result['status'])
            if result['status'] == 'completed':
                row.update({k: result[k] for k in ('low_signal', 'low_signal_threshold_training_sd_rms',
                    'target_centered_training_sd_rms', 'zero_prediction_nmse', 'zero_prediction_energy',
                    'dynamic_nmse', 'dynamic_energy', 'dynamic_support', 'evidence_interpretation')})
                for label in ('nominal', 'best_hb', 'best_joint'):
                    row.update({f'{label}_{k}': v for k, v in result[label].items()})
                row['nominal_minus_best_joint_nmse'] = result['nominal']['joint_nmse']-result['best_joint']['joint_nmse']
                row['nominal_minus_best_hb_nmse'] = result['nominal']['hb_nmse']-result['best_hb']['hb_nmse']
                profile.extend(dict(identity, **p) for p in result['profiles'])
                sets.extend(dict(identity, **s) for s in result['compatibility_sets'])
            else:
                row['error'] = result.get('error')
            summary.append(row)
        path = out/'tasks'/'nonlinear'/(task['id']+'.json')
        if path.exists():
            result = read_json(path)
            row = dict(identity, **{k: v for k, v in result.items() if k not in identity and k not in ('details', 'traceback', 'task')})
            if 'details' in result:
                details = result['details']
                row['integration_difference'] = details.get('integration', {}).get('max_abs_training_sd_difference')
                row['evaluations'] = details.get('evaluations')
                for name in ('HbO', 'HbR'):
                    row[f'{name}_nrmse'] = details.get('by_chromophore', {}).get(name, {}).get('nrmse')
            nonlinear.append(row)
    for task in plan['space_tasks']:
        path = out/'tasks'/'space'/(task['id']+'.json')
        if path.exists():
            result = read_json(path)
            space.append({k: v for k, v in result.items() if k not in ('traceback', 'task')})
    for name, rows in [('linear_profile', profile), ('linear_summary', summary),
                       ('compatibility_sets', sets), ('nonlinear', nonlinear), ('space', space)]:
        _csv(out/(name+'.csv'), rows)
    return dict(linear=len(summary), nonlinear=len(nonlinear), space=len(space),
        failures=sum(r['status'] != 'completed' for r in summary+nonlinear+space),
        nonlinear_without_legal_witness=sum(r.get('legal_witness') is False for r in nonlinear),
        nonlinear_optimizer_not_converged=sum(r.get('optimizer_converged') is False for r in nonlinear))


def stage_tasks(plan, kind, pilot=False):
    tasks = plan['space_tasks'] if kind == 'space' else plan['output_tasks']
    if not pilot:
        return tasks
    if kind == 'space':
        first = tasks[0]['row']['id']
        return [t for t in tasks if t['row']['id'] == first]
    first = {}
    for task in tasks:
        ref = task['ref']
        first.setdefault(ref['dataset'], ref['id'])
    return [t for t in tasks if t['ref']['id'] == first[t['ref']['dataset']]]


def run_panels(cfg, out, stage='all', workers=1):
    """Run exact frozen panels; caller owns durable measured-job supervision.

    ``plan``/``dry-run`` read metadata only. The measured stages are ``linear``,
    ``nonlinear`` and ``space``; ``all`` performs them in that order. Each task
    has one terminal JSON, retained failures and a compact selected-witness NPZ.
    """
    if stage not in ('plan', 'dry-run', 'linear', 'nonlinear', 'space', 'all', 'export',
                     'pilot-linear', 'pilot-nonlinear', 'pilot-space'):
        raise ValueError('unknown model compatibility panel stage')
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    if Path(out).resolve().parent != (ROOT/ARTIFACT_ROOT).resolve():
        raise ValueError('a versioned run directly below the existing model-compatibility root is required')
    plan = panel_plan(cfg)
    out = Path(out)/'panels'
    out.mkdir(parents=True, exist_ok=True)
    path = out/'plan.json'
    if path.exists() and read_json(path) != serial(plan):
        raise ValueError('frozen panel plan changed; use a fresh versioned output')
    if not path.exists():
        write_json(path, plan)
    if stage in ('plan', 'dry-run'):
        return plan
    with (out/'.runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        kinds = ['linear', 'nonlinear', 'space'] if stage == 'all' else ([] if stage == 'export' else [stage.removeprefix('pilot-')])
        for kind in kinds:
            tasks = stage_tasks(plan, kind, stage.startswith('pilot-'))
            pending_tasks = [t for t in tasks if not (out/'tasks'/kind/(t['id']+'.json')).exists()]
            payloads = iter((cfg, t, kind, str(out), str(ROOT)) for t in pending_tasks)
            finished = len(tasks)-len(pending_tasks)
            if workers == 1:
                for payload in payloads:
                    _execute(payload)
                    finished += 1
                    write_json(out/'progress.json', dict(stage=kind, terminal=finished, planned=len(tasks)))
            else:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    active = set()
                    while len(active) < 2*workers:
                        payload = next(payloads, None)
                        if payload is None:
                            break
                        active.add(pool.submit(_execute, payload))
                    while active:
                        done, active = wait(active, return_when=FIRST_COMPLETED)
                        for future in done:
                            future.result()
                            finished += 1
                            payload = next(payloads, None)
                            if payload is not None:
                                active.add(pool.submit(_execute, payload))
                        write_json(out/'progress.json', dict(stage=kind, terminal=finished, planned=len(tasks)))
            export_panel_tables(out, plan)
        counts = export_panel_tables(out, plan)
        all_terminal = counts['linear'] == 96 and counts['nonlinear'] == 96 and counts['space'] == 72
        write_json(out/'summary.json', dict(schema='model_compatibility_panels_summary_v1',
            **counts, planned_linear=96, planned_nonlinear=96, planned_space=72,
            execution=('completed_with_task_failures' if counts['failures'] else 'completed') if all_terminal else 'partial',
            failures_retained=True, protected_data_read=False, raw_data_read=False))
        return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--stage', choices=['plan', 'dry-run', 'linear', 'nonlinear', 'space', 'all', 'export',
                                          'pilot-linear', 'pilot-nonlinear', 'pilot-space'], default='plan')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    print(json.dumps(serial(run_panels(cfg, args.output, args.stage, args.workers)), ensure_ascii=False))


if __name__ == '__main__':
    main()
