#!/usr/bin/env python3
"""Independent paired nonlinear mechanism audit and retained EOG reanalysis.

Run --stage check before a supervisor-owned pilot/all launch. Retained empirical
analysis consumes only the explicit public endpoint tables in the contract.
No dataset loader or protected evaluation is called by this runner.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
from datetime import datetime, timezone
import json
from itertools import product
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import yaml

from src.inference.mechanism_attribution import (
    make_worlds, training_scales, fit_profile, select_profile, recovery_metrics,
    paired_counterfactual_metrics, attribution_basis,
)
from src.inference.shared_driver_reconstruction import fit_nonlinear_shared_driver
from src.inference.mechanism_attribution import parameters_at_tau

DEFAULT_CONFIG = ROOT/'experiments/configs/physiology_semantic_tokenizer/mechanism_attribution_v1.yaml'
ARMS = ['fixed_no_common', 'fixed_common', 'free_no_common', 'free_common',
        'oracle_mismatch', 'oracle_driver_no_common', 'oracle_driver_common', 'oracle_both']
METRICS = ['tau_absolute_log_error', 'tau_error_s', 'delta_tau_error_s', 'absolute_delta_tau_error_s',
    'physical_Hb_nrmse', 'driver_nrmse', 'state_nrmse', 'component_nrmse',
    'observed_nrmse', 'delta_physical_Hb_nrmse', 'tau_change_extra_projection',
    'nuisance_physical_compensation_projection', 'true_component_change_projection',
    'gain_physical_compensation_projection']


def serial(x):
    if isinstance(x, dict):
        return {str(k): serial(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [serial(v) for v in x]
    if isinstance(x, np.generic):
        return serial(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return str(x) if isinstance(x, Path) else x


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_task_artifacts(out, records, profiles, arrays):
    for name, rows in [('endpoints.csv', records), ('profiles.csv', profiles)]:
        temporary = out/(name+'.tmp')
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(out/name)
    temporary = out/'trajectories.tmp.npz'
    np.savez_compressed(temporary, **arrays)
    temporary.replace(out/'trajectories.npz')


def read_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get('schema') != 'mechanism_attribution_v1' or cfg.get('protected_data') != 'forbidden':
        raise ValueError('only the synthetic/retained-public v1 contract is supported')
    if (cfg.get('experiment_id') != 'MECHANISM-ATTRIBUTION-v1' or
            cfg.get('artifact_root') != 'experiments/runs/physiology_semantic_tokenizer/mechanism_attribution'):
        raise ValueError('experiment identity and artifact namespace are fixed')
    if cfg['data_boundary'] != 'synthetic_and_explicit_retained_public_endpoint_tables_only':
        raise ValueError('the read boundary cannot be enlarged through configuration')
    if cfg['tensor'] != dict(steps=120, dt_s=.25,
            observation_order=['EEG', 'HbO', 'HbR'], state_order=['r', 's', 'f', 'v', 'p', 'q']):
        raise ValueError('unexpected tensor contract')
    syn = cfg['synthetic']
    if (set(syn['training_seeds']) & set(syn['evaluation_seeds']) or
            not set(syn['noiseless_seeds']) <= set(syn['evaluation_seeds']) or
            len(set(syn['evaluation_seeds'])) != len(syn['evaluation_seeds'])):
        raise ValueError('training/evaluation seeds must be disjoint and unique')
    if syn['tau_s'] != [2., 3.] or syn['hb_gain'] != [1., 1.25] or syn['common_present'] != [False, True]:
        raise ValueError('unexpected factorial levels')
    if syn['common_loading'] != [.65, .35] or syn['common_modes'] != 4:
        raise ValueError('common component is the fixed four-mode HbT basis')
    grid = cfg['fit']['tau_grid_s']
    if grid != [1., 1.5, 2., 3., 4.] or cfg['fit']['fixed_tau_s'] != 2.:
        raise ValueError('finite profile grid must remain frozen')
    expected = {
        'attribution_run': 'experiments/runs/physiology_semantic_tokenizer/shared_driver_attribution/20261002_v1',
        'observation_run': 'experiments/runs/physiology_semantic_tokenizer/observation_semantics/20261009_v2',
        'spatial_summary': 'experiments/runs/physiology_semantic_tokenizer/shared_driver_spatial/20261020_measured_v2/summary.json',
        'source_audit': 'experiments/runs/physiology_semantic_tokenizer/data_quality_audit/20260924_hbo_hbr_calibration_v2/source_audit.json',
    }
    if any(cfg['retained'][k] != v for k, v in expected.items()):
        raise ValueError('retained reads are restricted to the named public evidence')
    return cfg


def make_plan(cfg, pilot=False):
    syn = cfg['synthetic']
    tasks = [dict(key=f'seed{s}_noise{level:g}', seed=s, noise_multiplier=level)
        for level, seeds in [(0., syn['noiseless_seeds']),
                             (syn['noisy_sd_training_scale'], syn['evaluation_seeds'])]
        for s in seeds]
    if pilot:
        tasks = [next(t for t in tasks if t['noise_multiplier'] == level)
                 for level in (0., syn['noisy_sd_training_scale'])]
    return dict(schema='mechanism_attribution_plan_v1', tasks=tasks,
        planned_seed_noise_tasks=len(tasks), planned_worlds=8*len(tasks),
        planned_endpoints=8*len(ARMS)*len(tasks), planned_profile_nodes=240*len(tasks),
        inference_unit='seed; noise levels are separate strata, not independent replications',
        scope='pilot' if pilot else 'full', arms=ARMS)


def calibrate(cfg, run):
    """Select a nuisance prior from separate known-truth training seeds only."""
    target = run/'calibration.json'
    if target.exists():
        return read_json(target)
    scales = training_scales(cfg)
    rows = []
    syn, fit_cfg = cfg['synthetic'], cfg['fit']
    for seed in syn['training_seeds']:
        worlds = make_worlds(cfg, seed, syn['noisy_sd_training_scale'], scale=scales['observation'])
        for multiplier in fit_cfg['component_sd_multipliers']:
            cs = syn['common_coefficient_rms']*multiplier
            for world in worlds:
                # The same known training tau isolates nuisance-prior calibration.
                fit = fit_nonlinear_shared_driver(world['observation'], parameters_at_tau(world['tau']),
                    cfg['tensor']['dt_s'], sd=scales['observation'],
                    observation_basis=attribution_basis(cfg['tensor']['steps'], syn['common_modes']),
                    observation_coefficient_sd=cs,
                    **{k: fit_cfg[k] for k in ('penalty', 'initial_penalty', 'max_evaluations',
                        'gradient_tolerance', 'substeps', 'numerical_backend')})
                fit['tau'] = world['tau']
                metrics = recovery_metrics(world, fit if fit.get('converged') else None, scales)
                score = metrics['physical_Hb_nrmse']**2+metrics['driver_nrmse']**2 if metrics['converged'] else 100.
                rows.append(dict(seed=seed, world=world['key'], multiplier=multiplier,
                                 score=score, status=fit['status']))
    df = pd.DataFrame(rows)
    df.to_csv(run/'calibration_metrics.csv', index=False)
    scores = df.groupby('multiplier').score.mean()
    if not (df.status == 'completed').any():
        raise RuntimeError('all synthetic training fits failed; no coefficient prior can be selected')
    selected = float(scores.idxmin())
    result = dict(schema='mechanism_attribution_calibration_v1', scales=scales,
        resolved_reference_parameters=asdict(parameters_at_tau(cfg['fit']['fixed_tau_s'])),
        observation_coordinate='direct_canonical_model_output; no_reference_subtraction_or_filter',
        scale_definition='training_clean_signal_RMS_about_zero; vascular_RMS_of_departure_from_rest; not_SD',
        training_seeds=syn['training_seeds'], evaluation_seeds_used=False,
        selection='mean_physical_Hb_NMSE_plus_driver_NMSE_failure_score100',
        multiplier_scores={str(k): float(v) for k, v in scores.items()},
        selected_multiplier=selected, coefficient_sd=syn['common_coefficient_rms']*selected,
        status_counts=df.status.value_counts().to_dict())
    write_json(target, result)
    return read_json(target)


def evaluate_task(task, cfg, calibration, run_string):
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    run = Path(run_string)
    out = run/'tasks'/task['key']
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    write_json(out/'status.json', dict(status='running', task=task, pid=os.getpid()))
    try:
        scales = calibration['scales']
        worlds = make_worlds(cfg, task['seed'], task['noise_multiplier'], scale=scales['observation'])
        records, profiles, arrays, results = [], [], {}, {}
        for wi, world in enumerate(worlds):
            gain = np.array([1., world['gain'], world['gain']])
            corrected = (world['observation']-world['component'])/gain
            families = {}
            for common in (False, True):
                families['common' if common else 'plain'] = fit_profile(world['observation'], cfg,
                    scales['observation'], calibration['coefficient_sd'], common=common)
                families['oracle_driver_common' if common else 'oracle_driver_plain'] = fit_profile(
                    world['observation'], cfg, scales['observation'], calibration['coefficient_sd'],
                    common=common, oracle_driver=world['driver'], oracle_initial=world['initial_state'])
            families['oracle_mismatch'] = fit_profile(corrected, cfg,
                np.asarray(scales['observation'])/gain, calibration['coefficient_sd'])
            families['oracle_both'] = fit_profile(corrected, cfg,
                np.asarray(scales['observation'])/gain, calibration['coefficient_sd'],
                oracle_driver=world['driver'], oracle_initial=world['initial_state'])
            for family, nodes in families.items():
                for node in nodes:
                    curvature_cost = cfg['fit']['penalty']*node.get('roughness', 0.)
                    initial_cost = (cfg['fit']['initial_penalty']*float(np.sum(
                        (node['initial_state']-np.r_[0., np.ones(4)])**2))
                        if 'initial_state' in node and not family.startswith('oracle_driver') and family != 'oracle_both' else 0.)
                    profiles.append(dict(**task, world=world['key'], tau_true_s=world['tau'],
                        hb_gain=world['gain'], common_present=world['common_present'], family=family,
                        tau_candidate_s=node['tau'], status=node['status'],
                        converged=node.get('converged', False), objective=node.get('objective'),
                        weighted_data_sse=node.get('weighted_data_sse'),
                        driver_curvature_cost=curvature_cost, initial_state_cost=initial_cost,
                        observation_component_cost=node.get('observation_component_cost', 0.),
                        evaluations=node.get('evaluations'), convergence_reason=node.get('convergence_reason')))
                    if node.get('error'):
                        profiles[-1]['error'] = node['error']
            choices = dict(fixed_no_common=select_profile(families['plain'], 2.),
                fixed_common=select_profile(families['common'], 2.),
                free_no_common=select_profile(families['plain']), free_common=select_profile(families['common']),
                oracle_mismatch=select_profile(families['oracle_mismatch']),
                oracle_driver_no_common=select_profile(families['oracle_driver_plain']),
                oracle_driver_common=select_profile(families['oracle_driver_common']),
                oracle_both=select_profile(families['oracle_both']))
            for arm, fit in choices.items():
                family_name = {'fixed_no_common': 'plain', 'free_no_common': 'plain',
                    'fixed_common': 'common', 'free_common': 'common',
                    'oracle_driver_no_common': 'oracle_driver_plain',
                    'oracle_driver_common': 'oracle_driver_common'}.get(arm, arm)
                eligible_nodes = families[family_name]
                if arm.startswith('fixed_'):
                    eligible_nodes = [node for node in eligible_nodes if node['tau'] == cfg['fit']['fixed_tau_s']]
                available_nodes = sum(bool(node.get('converged')) for node in eligible_nodes)
                observed = None
                if fit is not None and arm in ('oracle_mismatch', 'oracle_both'):
                    # Keep recovered physical output in canonical coordinates while
                    # reporting residuals in the original observation coordinates.
                    observed = fit['physical_prediction']*gain+world['component']
                    fit = dict(fit, observation_component=world['component'].copy())
                results[world['key'], arm] = fit
                row = dict(**task, world=world['key'], arm=arm, tau_true_s=world['tau'],
                    hb_gain=world['gain'], common_present=world['common_present'],
                    profile_planned_nodes=len(eligible_nodes), profile_available_nodes=available_nodes,
                    profile_complete=available_nodes == len(eligible_nodes),
                    selection_scope='minimum_among_available_converged_nodes; incomplete_profile_is_conditional',
                    **recovery_metrics(world, fit, scales, observation_prediction=observed))
                records.append(row)
                if fit is not None:
                    for field, key in [('prediction', 'prediction'), ('physical', 'physical_prediction'),
                                       ('component', 'observation_component'), ('driver', 'driver'), ('states', 'states')]:
                        arrays[f'w{wi}_{arm}_{field}'] = fit[key]
            for field in ('observation', 'physical', 'component', 'driver', 'states', 'noise', 'nuisance'):
                arrays[f'w{wi}_truth_{field}'] = world[field]
            # A later unexpected task error preserves all earlier completed
            # worlds; summarize expands only genuinely missing terminal rows.
            write_task_artifacts(out, records, profiles, arrays)
            write_json(out/'status.json', dict(status='running', task=task, worlds_completed=wi+1,
                planned_worlds=8, elapsed_s=time.time()-started))
        baseline = worlds[0]
        by_world = {w['key']: w for w in worlds}
        for row in records:
            row.update(paired_counterfactual_metrics(by_world[row['world']], baseline,
                results[row['world'], row['arm']], results[baseline['key'], row['arm']], scales))
        write_task_artifacts(out, records, profiles, arrays)
        status = dict(status='completed', task=task, planned_endpoints=64, terminal_endpoints=len(records),
            succeeded_endpoints=sum(r['converged'] for r in records), profile_nodes=len(profiles),
            converged_profile_nodes=sum(r['converged'] for r in profiles), elapsed_s=time.time()-started)
        write_json(out/'status.json', status)
        return status
    except Exception as exc:
        status = dict(status='failed_task', task=task, error=repr(exc), traceback=traceback.format_exc(),
                      elapsed_s=time.time()-started)
        write_json(out/'status.json', status)
        return status


def bootstrap(values, seed=73):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(mean=None, low=None, high=None, independent_units=0)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), (2000, len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()), low=float(np.quantile(samples, .025)),
                high=float(np.quantile(samples, .975)), independent_units=len(values))


def retained_analysis(cfg, run):
    """New frozen-prediction stratification; EOG remains an association anchor."""
    retained = cfg['retained']
    obs = ROOT/retained['observation_run']
    attr = ROOT/retained['attribution_run']
    summary = read_json(obs/'summary.json')
    attr_summary = read_json(attr/'summary.json')
    if (summary['execution'] != 'completed' or attr_summary['execution'] != 'completed' or
            summary.get('experiment_id') != 'OBSERVATION-SEMANTICS-v1' or
            attr_summary.get('schema') != 'shared_driver_attribution_summary_v1'):
        raise ValueError('named retained evidence must be completed')
    metrics = pd.read_csv(obs/'event_metrics.csv')
    events = pd.read_csv(obs/'event_inventory.csv')
    plan = read_json(obs/'measured_plan.json')
    if plan.get('split') != 'odd_native_subject_number_train_even_evaluate':
        raise ValueError('unexpected retained public split')
    # Include zero-event windows via the owning measured plan, rather than
    # silently restricting the denominator to windows with detected events.
    identities = []
    for record in plan['records']:
        if record['dataset'] not in ('eeg_fnirs_single_trial', 'simultaneous_eeg_nirs'):
            raise ValueError('unexpected dataset on retained EOG surface')
        number = re.search(r'(\d+)$', record['subject'])
        if number is None or record['split'] != ('train' if int(number.group(1)) % 2 else 'evaluate'):
            raise ValueError('retained EOG subject partition is invalid')
        for window in record['windows']:
            identities.append(dict(dataset=record['dataset'], subject=record['subject'],
                id=window['id'], split=record['split']))
    inventory = pd.DataFrame(identities).drop_duplicates()
    counts = events.groupby(['dataset', 'subject', 'id']).size().rename('event_count').reset_index()
    inventory = inventory.merge(counts, how='left', on=['dataset', 'subject', 'id'], validate='one_to_one')
    inventory['event_count'] = inventory.event_count.fillna(0).astype(int)
    thresholds = inventory[inventory.split == 'train'].groupby('dataset').event_count.median().to_dict()
    inventory['training_median_threshold'] = inventory.dataset.map(thresholds)
    inventory['burden'] = np.where(inventory.event_count > inventory.training_median_threshold, 'high', 'low')
    joined = metrics.merge(inventory, on=['dataset', 'subject', 'id'], how='left', validate='many_to_one')
    if joined.burden.isna().any() or (joined.split != 'evaluate').any():
        raise ValueError('EOG endpoint identity or training/evaluation join failed')
    paired = joined.pivot(index=['dataset', 'subject', 'id', 'endpoint', 'burden'], columns='arm', values='nrmse').reset_index()
    rows = []
    for control in ['training_mean', 'EOG_wrong_subject', 'EOG_time_shift']:
        if control not in paired:
            continue
        paired['gain'] = paired[control]-paired.EOG_real
        for (dataset, endpoint, burden), block in paired.groupby(['dataset', 'endpoint', 'burden']):
            unit = block.groupby('subject').gain.mean()
            rows.append(dict(dataset=dataset, endpoint=endpoint, burden=burden, control=control,
                planned_windows=len(block), paired_windows=int(block.gain.notna().sum()),
                **bootstrap(unit.values), difference='control_NRMSE_minus_real_EOG_NRMSE'))
    pd.DataFrame(rows).to_csv(run/'retained_EOG_burden.csv', index=False)
    inventory.to_csv(run/'retained_event_burden_inventory.csv', index=False)
    associations = []
    for (dataset, endpoint), block in joined[joined.arm == 'EOG_real'].groupby(['dataset', 'endpoint']):
        unit = block.groupby('subject')[['nrmse', 'event_count']].mean()
        rho = (spearmanr(unit.event_count, unit.nrmse)
               if unit.event_count.nunique() > 1 and unit.nrmse.nunique() > 1 else None)
        associations.append(dict(dataset=dataset, endpoint=endpoint, subjects=len(unit),
            spearman=float(rho.statistic) if rho is not None else np.nan,
            p_uncorrected=float(rho.pvalue) if rho is not None else np.nan,
            scope='descriptive_across_subjects_frozen_EOG_prediction_error_vs_event_burden'))
    pd.DataFrame(associations).to_csv(run/'retained_EOG_subject_associations.csv', index=False)
    pd.read_csv(attr/'anchor_paired.csv').to_csv(run/'retained_component_anchor_context.csv', index=False)
    audit = read_json(ROOT/retained['source_audit'])
    source = dict(source_audit=retained['source_audit'], audited_records=len(audit['records']),
        audited_subjects=sorted({r['subject'] for r in audit['records']}),
        inspected_files=len(audit['inspected_files']), archive_subject_inventories=len(audit['archive_subject_members']),
        systemic_auxiliary_candidates=[r.get('systemic_auxiliary_candidates', []) for r in audit['records']],
        independent_calibration_established=audit['independent_calibration_established'],
        missing_semantics=audit['missing_semantics'],
        current_reader=retained['auxiliary_reader'], original_documentation=retained['original_documentation'],
        conclusion='selected_local_release_does_not_establish_ECG_respiration_or_short_separation_arrays; EOG_available',
        boundary='not_a_claim_of_global_dataset_absence; no_new_raw_file_reads',
        retained_event_windows=int(inventory[inventory.split == 'evaluate'].shape[0]),
        training_thresholds=thresholds, native_anchor_semantics='EOG_association_not_systemic_or_tissue_truth',
        source_summary=retained['observation_run']+'/summary.json',
        latest_spatial_context=retained['spatial_summary'],
        new_empirical_work='training_threshold_event_burden_stratification_of_retained_subject_heldout_predictions')
    write_json(run/'source_audit.json', source)
    return source


def summarize(cfg, run, plan):
    endpoints, profiles, statuses = [], [], []
    for task in plan['tasks']:
        out = run/'tasks'/task['key']
        status = dict(status='pending', task=task)
        if (out/'status.json').exists():
            status = read_json(out/'status.json')
            statuses.append(status)
        fill_status = 'failed_task' if status['status'] == 'failed_task' else 'pending'
        expected_endpoints, expected_profiles = [], []
        for tau, gain, present in product(cfg['synthetic']['tau_s'], cfg['synthetic']['hb_gain'], cfg['synthetic']['common_present']):
            world = f'tau{tau:g}_gain{gain:g}_common{int(present)}'
            common_fields = dict(**task, world=world, tau_true_s=tau, hb_gain=gain, common_present=present,
                                 status=fill_status, converged=False)
            for arm in ARMS:
                expected_endpoints.append(dict(**common_fields, arm=arm,
                    profile_planned_nodes=1 if arm.startswith('fixed_') else len(cfg['fit']['tau_grid_s']),
                    profile_available_nodes=0, profile_complete=False, **{m: np.nan for m in METRICS}))
            for family in ['plain', 'common', 'oracle_driver_plain', 'oracle_driver_common', 'oracle_mismatch', 'oracle_both']:
                for candidate in cfg['fit']['tau_grid_s']:
                    expected_profiles.append(dict(**common_fields, family=family, tau_candidate_s=candidate,
                        objective=np.nan, weighted_data_sse=np.nan, driver_curvature_cost=np.nan,
                        initial_state_cost=np.nan, observation_component_cost=np.nan))
        for name, expected, keys, collection in [
                ('endpoints.csv', expected_endpoints, ['world', 'arm'], endpoints),
                ('profiles.csv', expected_profiles, ['world', 'family', 'tau_candidate_s'], profiles)]:
            frame = pd.DataFrame(expected).set_index(keys)
            if status['status'] in ('completed', 'failed_task') and (out/name).exists():
                actual = pd.read_csv(out/name).set_index(keys)
                if actual.index.has_duplicates or not actual.index.isin(frame.index).all():
                    raise ValueError('task output violates the fixed factorial plan')
                for column in actual.columns:
                    if column not in frame:
                        frame[column] = np.nan
                # Replace whole recorded rows, including their legitimate NaNs.
                for column in actual.columns:
                    frame.loc[actual.index, column] = actual[column]
            collection.append(frame.reset_index())
    result = dict(schema='mechanism_attribution_summary_v1', experiment_id=cfg['experiment_id'],
        execution='completed' if len(statuses) == len(plan['tasks']) and all(s['status'] in ('completed', 'failed_task') for s in statuses) else 'incomplete',
        planned_tasks=len(plan['tasks']),
        terminal_tasks=sum(s['status'] in ('completed', 'failed_task') for s in statuses),
        planned_endpoints=plan['planned_endpoints'],
        source_boundary=cfg['data_boundary'], scope=plan['scope'], task_statuses=statuses,
        resolved_reference_parameters=asdict(parameters_at_tau(cfg['fit']['fixed_tau_s'])),
        observation_coordinate='direct_canonical_model_output; no_reference_subtraction_or_filter',
        scale_definition='training_clean_signal_RMS_about_zero; vascular_RMS_of_departure_from_rest; not_SD',
        failed_tasks=sum(s['status'] == 'failed_task' for s in statuses),
        nuisance_claim_limit=cfg['synthetic']['claim_limit'],
        claim_boundary='finite_generator_grid_and_regularization_diagnostic; no_measured_physiological_truth_or_universal_impossibility_claim')
    if endpoints:
        data = pd.concat(endpoints, ignore_index=True)
        profile = pd.concat(profiles, ignore_index=True)
        data.to_csv(run/'endpoints.csv', index=False)
        profile.to_csv(run/'profiles.csv', index=False)
        groups = []
        for keys, block in data.groupby(['noise_multiplier', 'tau_true_s', 'hb_gain', 'common_present', 'arm']):
            identity = dict(zip(['noise_multiplier', 'tau_true_s', 'hb_gain', 'common_present', 'arm'], keys))
            for metric in METRICS:
                if metric not in block:
                    continue
                groups.append(dict(**identity, metric=metric, planned_seeds=len(block),
                    successful_fits=int(block.converged.sum()), finite_metric=int(block[metric].notna().sum()),
                    **bootstrap(block.loc[block.converged, metric].values)))
        pd.DataFrame(groups).to_csv(run/'endpoint_summary.csv', index=False)
        comparisons = []
        for keys, block in data.groupby(['noise_multiplier', 'tau_true_s', 'hb_gain', 'common_present']):
            identity = dict(zip(['noise_multiplier', 'tau_true_s', 'hb_gain', 'common_present'], keys))
            for arm, control in [('free_common', 'free_no_common'), ('free_no_common', 'fixed_no_common'),
                    ('free_common', 'fixed_common'), ('oracle_mismatch', 'free_no_common'),
                    ('oracle_driver_common', 'free_common'), ('oracle_both', 'oracle_mismatch')]:
                a = block[block.arm == arm].set_index('seed')
                b = block[block.arm == control].set_index('seed')
                paired = a.join(b, lsuffix='_a', rsuffix='_b')
                ok = paired.converged_a & paired.converged_b
                for metric in METRICS:
                    if metric+'_a' not in paired:
                        continue
                    difference = paired[metric+'_b']-paired[metric+'_a']
                    comparisons.append(dict(**identity, arm=arm, control=control, metric=metric,
                        planned_pairs=len(paired), common_success=int(ok.sum()),
                        **bootstrap(difference[ok].values), difference='control_minus_candidate; error_metrics_positive_favors_candidate; signed_projections_have_no_universal_favorable_direction'))
        pd.DataFrame(comparisons).to_csv(run/'paired_comparisons.csv', index=False)
        result.update(endpoint_rows=len(data), terminal_endpoints=int((data.status != 'pending').sum()),
            succeeded_endpoints=int(data.converged.sum()),
            incomplete_profile_endpoints=int((~data.profile_complete).sum()),
            status_counts=data.status.value_counts().to_dict(), profile_nodes=len(profile),
            profile_status_counts=profile.status.value_counts().to_dict(),
            independent_seed_units=int(data.seed.nunique()),
            interval='2000_seed_percentile_bootstrap_within_noise_and_intervention_stratum; no_multiple_comparison_confirmation',
            endpoint_files=['endpoints.csv', 'profiles.csv', 'endpoint_summary.csv', 'paired_comparisons.csv',
                'retained_EOG_burden.csv', 'retained_EOG_subject_associations.csv', 'source_audit.json'])
    write_json(run/'summary.json', result)
    return result


def verify(cfg, run):
    plan, calibration = read_json(run/'plan.json'), read_json(run/'calibration.json')
    rows, profiles = pd.read_csv(run/'endpoints.csv'), pd.read_csv(run/'profiles.csv')
    errors = []
    if len(rows) != plan['planned_endpoints'] or len(profiles) != plan['planned_profile_nodes']:
        errors.append('incomplete_declared_denominator')
    finite = profiles.objective.notna()
    decomposed = (profiles.weighted_data_sse+profiles.driver_curvature_cost+
                  profiles.initial_state_cost+profiles.observation_component_cost)
    objective_error = float(np.max(np.abs(profiles.loc[finite, 'objective']-decomposed[finite]))) if finite.any() else 0.
    if objective_error > 1e-8:
        errors.append('objective_components_do_not_sum')
    for row in rows.itertuples():
        family = {'fixed_no_common': 'plain', 'free_no_common': 'plain',
            'fixed_common': 'common', 'free_common': 'common',
            'oracle_driver_no_common': 'oracle_driver_plain',
            'oracle_driver_common': 'oracle_driver_common'}.get(row.arm, row.arm)
        nodes = profiles[(profiles.key == row.key) & (profiles.world == row.world) &
                         (profiles.family == family)]
        if row.arm.startswith('fixed_'):
            nodes = nodes[nodes.tau_candidate_s == cfg['fit']['fixed_tau_s']]
        valid = nodes[nodes.converged]
        if len(valid) != row.profile_available_nodes or (len(valid) == len(nodes)) != row.profile_complete:
            errors.append('incorrect_profile_completeness')
        if row.converged:
            selected = valid.loc[valid.objective.idxmin()]
            if selected.tau_candidate_s != row.tau_hat_s or not np.isclose(selected.objective, row.objective):
                errors.append('incorrect_profile_argmin')
    max_noise_error, max_pair_error = 0., 0.
    for task in plan['tasks']:
        out = run/'tasks'/task['key']
        task_rows = rows[rows.key == task['key']]
        if not (out/'trajectories.npz').exists():
            if task_rows.converged.any() or read_json(out/'status.json')['status'] != 'failed_task':
                errors.append('missing_trajectories_for_successful_or_nonterminal_task')
            continue
        worlds = make_worlds(cfg, task['seed'], task['noise_multiplier'], scale=calibration['scales']['observation'])
        with np.load(out/'trajectories.npz') as arrays:
            for wi, world in enumerate(worlds):
                if f'w{wi}_truth_noise' not in arrays:
                    if task_rows[task_rows.world == world['key']].converged.any():
                        errors.append('missing_truth_for_successful_world')
                    continue
                max_noise_error = max(max_noise_error, float(np.max(np.abs(arrays[f'w{wi}_truth_noise']-arrays['w0_truth_noise']))))
                max_pair_error = max(max_pair_error, float(np.max(np.abs(arrays[f'w{wi}_truth_driver']-arrays['w0_truth_driver']))))
                for field in ('observation', 'physical', 'component', 'driver', 'states'):
                    if not np.array_equal(arrays[f'w{wi}_truth_{field}'], world[field]):
                        errors.append(f'nonreproducible_truth:{task["key"]}:{wi}:{field}')
                for arm in ARMS:
                    row = rows[(rows.key == task['key']) & (rows.world == world['key']) & (rows.arm == arm)].iloc[0]
                    if row.converged:
                        physical = arrays[f'w{wi}_{arm}_physical']
                        computed = np.sqrt(np.mean(((physical[:, 1:]-world['physical'][:, 1:])/np.asarray(calibration['scales']['observation'])[1:])**2))
                        if not np.isclose(computed, row.physical_Hb_nrmse, rtol=1e-10, atol=1e-12):
                            errors.append('metric_replay_mismatch')
    if max_noise_error != 0 or max_pair_error != 0:
        errors.append('broken_counterfactual_pairing')
    result = dict(status='passed' if not errors else 'failed', errors=errors,
        endpoint_rows=len(rows), profile_rows=len(profiles), max_shared_noise_error=max_noise_error,
        max_objective_decomposition_error=objective_error,
        max_shared_driver_error=max_pair_error, training_evaluation_disjoint=not bool(
            set(calibration['training_seeds']) & set(cfg['synthetic']['evaluation_seeds'])),
        check='exact_regeneration_truth_and_fixed_denominators_and_recomputed_physical_Hb_metrics')
    write_json(run/'verification.json', result)
    return result


def snapshot(cfg_path, run):
    source = run/'source_snapshot'
    for rel in [cfg_path.relative_to(ROOT), Path('src/inference/mechanism_attribution.py'),
                Path('experiments/scripts/evaluate_mechanism_attribution.py'),
                Path('src/inference/shared_driver_reconstruction.py'), Path('src/inference/shared_driver_rk4.py'),
                Path('src/inference/shared_driver_attribution.py'), Path('src/inference/t3a_balloon_robust_ssm.py')]:
        target = source/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, target)
    write_json(run/'provenance.json', dict(git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_snapshot=str(source.relative_to(ROOT)), command=sys.argv,
        python=sys.executable, created_utc=datetime.now(timezone.utc).isoformat(),
        owner='experiments/scripts/evaluate_mechanism_attribution.py', no_protected_or_raw_reads=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--stage', choices=['check', 'pilot', 'all', 'retained', 'summarize', 'verify'], default='check')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = read_config(args.config.resolve())
    if args.stage == 'check':
        scales = training_scales(cfg)
        plan = make_plan(cfg)
        print(json.dumps(serial(dict(status='passed', plan=plan, scales=scales)), indent=2))
        return
    if args.run is None:
        parser.error('--run is required for artifact stages')
    run = args.run.resolve()
    root = (ROOT/cfg['artifact_root']).resolve()
    if not run.is_relative_to(root) or run == root:
        parser.error('--run must be a named child of the owning artifact root')
    if args.workers < 1 or args.workers > cfg['resources']['max_workers']:
        parser.error('--workers exceeds the contract')
    if args.stage in ('pilot', 'all'):
        if run.exists() and any(run.iterdir()) and not args.resume:
            parser.error('existing run requires --resume; frozen exports cannot be overwritten')
        run.mkdir(parents=True, exist_ok=True)
        plan = make_plan(cfg, pilot=args.stage == 'pilot')
        if (run/'plan.json').exists() and read_json(run/'plan.json') != plan:
            raise ValueError('resume plan differs from the retained task plan')
        if (run/'resolved_config.yaml').exists() and yaml.safe_load((run/'resolved_config.yaml').read_text()) != cfg:
            raise ValueError('resume configuration differs from retained contract')
        if not (run/'plan.json').exists():
            (run/'resolved_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
            write_json(run/'plan.json', plan)
            snapshot(args.config.resolve(), run)
            write_json(run/'manifest.json', dict(schema='mechanism_attribution_artifacts_v1',
                task_directory='tasks/<seed_noise_key>/', terminal_record='status.json',
                tables=['endpoints.csv', 'profiles.csv'], array_file='trajectories.npz',
                array_key='w<world_index>_<arm_or_truth>_<field>',
                world_order='product(tau_s,hb_gain,common_present) in resolved_config.yaml',
                physical_coordinate='canonical EEG,HbO,HbR before gain and additive common',
                saved_prediction='canonical model+component; oracle_mismatch/oracle_both save corrected-coordinate model prediction',
                endpoint_observed_nrmse='original observation coordinates including known gain/component for mismatch oracles',
                profile_families=dict(plain='refit driver+5 initial, no common',
                    common='refit driver+5 initial+four profiled common coefficients',
                    oracle_mismatch='known gain/common correction, refit driver+5 initial',
                    oracle_driver_plain='known true driver AND true five initial, no common',
                    oracle_driver_common='known true driver AND true five initial, refit common with same coefficient penalty',
                    oracle_both='known true driver AND initial AND gain/common correction'),
                profile_denominator='192 worlds x5 tau nodes x6 profile families =5760; pilot16x5x6=480',
                oracle_penalties='known driver/initial receive no optimization priors; ordinary fits retain all declared priors',
                incomplete_profile='minimum over available converged nodes only, explicitly flagged; no interval interpretation'))
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
        calibration = calibrate(cfg, run)
        retained_analysis(cfg, run)
        pending = [t for t in plan['tasks'] if not (run/'tasks'/t['key']/'status.json').exists() or
            read_json(run/'tasks'/t['key']/'status.json')['status'] not in ('completed', 'failed_task')]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            active = {}
            iterator = iter(pending)
            while True:
                while len(active) < args.workers:
                    task = next(iterator, None)
                    if task is None:
                        break
                    active[pool.submit(evaluate_task, task, cfg, calibration, str(run))] = task
                if not active:
                    break
                ready, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in ready:
                    print(json.dumps(serial(future.result())), flush=True)
                    del active[future]
                summarize(cfg, run, plan)
        result = summarize(cfg, run, plan)
        print(json.dumps(serial(result), indent=2))
        if result['execution'] != 'completed':
            raise SystemExit(1)
    elif args.stage == 'retained':
        run.mkdir(parents=True, exist_ok=True)
        print(json.dumps(serial(retained_analysis(cfg, run)), indent=2))
    elif args.stage == 'summarize':
        print(json.dumps(serial(summarize(cfg, run, read_json(run/'plan.json'))), indent=2))
    elif args.stage == 'verify':
        result = verify(cfg, run)
        print(json.dumps(result, indent=2))
        if result['status'] != 'passed':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
