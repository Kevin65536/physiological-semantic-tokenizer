"""Independent read-only audit of the retained Suite 1 formal evidence.

Recompute arithmetic from arrays and replay the owning nonlinear forward;
do not call the runner, optimizer, or information helper under audit.
Versioned figure exports read retained evidence and preserve original figures.
"""
from __future__ import annotations
import collections
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path('/SSD_2/pid-mcm-implementation')
RUN = ROOT/'experiments/runs/physiology_semantic_tokenizer/source_observability/20261021_v1'
sys.path.insert(0, str(ROOT))
from src.inference.shared_driver_spatial import forward_spatial_shared_driver
from src.inference.shared_driver_modes import temporal_mode_basis
from src.inference.t3a_balloon_robust_ssm import BalloonFreeParameters, BalloonParameters


def read(path):
    return json.loads(Path(path).read_text())


def array(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


failures, maxima = [], collections.defaultdict(float)


def check(name, actual, expected, atol=1e-10, rtol=1e-8):
    a, b = np.asarray(actual), np.asarray(expected)
    error = float(np.max(np.abs(a-b))) if a.size else 0.
    maxima[name] = max(maxima[name], error)
    if a.shape != b.shape or not np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
        failures.append(dict(check=name, max_absolute_error=error))


def model_case(cfg, task):
    s = cfg['synthetic']
    basis = temporal_mode_basis(s['steps'], s['temporal_modes'])
    coef = np.random.default_rng(task['seed']).normal(size=(s['temporal_modes'], 1))
    coef /= np.arange(1, s['temporal_modes']+1)[:, None]**s['coefficient_decay']
    coef *= s['driver_rms']/np.sqrt(np.mean((basis@coef)**2))
    initial = np.asarray(s['initial_coordinates'])[None]*(1 if task['seed'] % 2 else -1)
    parameters = BalloonParameters(free=BalloonFreeParameters(tau=task['tau_truth'], kappa=task['kappa_truth']))
    kwargs = dict(eeg_loadings=np.asarray(s['eeg_loadings']), hb_loadings=np.asarray(s['hb_loadings']),
                  temporal_basis=basis, numerical_backend='numba')
    truth = forward_spatial_shared_driver(coef, initial, parameters, s['dt_s'],
        derivative=False, substeps=8, **kwargs)
    refined = forward_spatial_shared_driver(coef, initial, parameters, s['dt_s'],
        derivative=False, substeps=16, **kwargs)
    maxima['generator_8_vs_16_max_standardized_prediction'] = max(
        maxima['generator_8_vs_16_max_standardized_prediction'],
        float(np.max(np.abs(truth['prediction']-refined['prediction'])/s['weighting_sd'])))
    return coef, initial, parameters, kwargs, truth


def independent_local(cfg, task, case, record, saved):
    coef, initial, parameters, kwargs, truth = case
    s, p = cfg['synthetic'], cfg['priors']
    for modes in cfg['information']['driver_mode_controls']:
        basis = temporal_mode_basis(s['steps'], modes)
        coefficient = np.linalg.lstsq(basis, truth['drivers'], rcond=1e-12)[0]
        options = dict(kwargs, temporal_basis=basis, substeps=4)
        base = forward_spatial_shared_driver(coefficient, initial, parameters, .25,
                                             derivative=True, **options)
        n = modes+5
        jac = np.zeros((120, 5, n+2))
        jac[:, :, :n] = base['jacobian'].reshape(120, 5, n)
        flow_jac = np.zeros((120, 1, n+2))
        flow_jac[:, :, :n] = base['flow_jacobian']
        for index, parameter in enumerate(('tau', 'kappa')):
            shifted = []
            for sign in (-1, 1):
                param = replace(parameters, free=replace(parameters.free,
                    **{parameter: getattr(parameters.free, parameter)*np.exp(sign*1e-4)}))
                shifted.append(forward_spatial_shared_driver(coefficient, initial, param,
                    .25, derivative=False, **options))
            jac[:, :, n+index] = (shifted[1]['prediction']-shifted[0]['prediction'])/2e-4
            flow_jac[:, :, n+index] = (shifted[1]['states'][:, :, 2]-shifted[0]['states'][:, :, 2])/2e-4
        curvature = np.zeros((118, n+2))
        curvature[:, :modes] = np.sqrt(p['driver_curvature_weight'])*np.diff(basis, n=2, axis=0)/.25**1.5
        initial_jac = np.zeros((5, n+2))
        initial_jac[:, modes:n] = np.diag(np.sqrt(p['initial_weight'])*np.r_[1., base['initial_state'][0, 1:]])
        flow = base['states'][:, :, 2]
        flow_prior = (np.sqrt(p['flow_prior_weight']*.25)/p['flow_log_sd']/flow[:, :, None]*
                      flow_jac.transpose(0, 2, 1)).transpose(0, 2, 1).reshape(120, n+2)
        # Explicit form avoids ambiguous broadcasting across the coordinate axis.
        flow_prior = ((np.sqrt(p['flow_prior_weight']*.25)/p['flow_log_sd']/flow)[:, :, None]*flow_jac).reshape(120, n+2)
        prior = np.vstack((curvature, initial_jac, flow_prior, np.zeros((2, n+2))))
        scales = np.r_[np.full(modes, .025), np.full(5, .02), [.5, .5]]
        for modality in ('Hb_only', 'EEG_Hb'):
            mask = np.ones((120, 5), bool)
            if modality == 'Hb_only':
                mask[:, :3] = False
            weighted = (jac*(np.asarray(s['feature_weights'])/s['weighting_sd'])[None, :, None])[mask]*scales
            singular = np.linalg.svd(weighted, compute_uv=False)
            rank = int(np.sum(singular > singular[0]*1e-8))
            nuisance, theta = weighted[:, :-2], weighted[:, -2:]
            compensation = np.linalg.lstsq(nuisance, -theta, rcond=1e-8)[0]
            projected = theta+nuisance@compensation
            smallest = float(np.linalg.eigvalsh(projected.T@projected)[0])
            matches = [r for r in record['information'] if r['driver_modes'] == modes and
                       r['modality'] == modality and r['prior_multiplier'] == 0.]
            check('local_data_rank', rank, matches[0]['data_rank'], atol=0, rtol=0)
            check('local_projected_theta_min_eigen', smallest, matches[0]['theta_profile_min_eigen'], atol=1e-10)
            for multiplier in (0., 1.):
                block = ('' if modes == 12 else 'M120__')+f'{modality}__prior{multiplier:g}__'
                total = np.linalg.svd(np.vstack((weighted, np.sqrt(multiplier)*prior*scales)), compute_uv=False)
                check('local_data_singular_values', singular, saved[block+'data_singular_values'])
                check('local_total_singular_values', total, saved[block+'total_singular_values'])


def figure_export(cfg, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10, 'xtick.labelsize': 10,
                         'ytick.labelsize': 10, 'legend.fontsize': 10})
    folder = RUN/'figures/readable_v2'
    if folder.exists():
        return [str(p.relative_to(RUN)) for p in sorted(folder.glob('*.png'))]
    folder.mkdir(parents=True)
    group = [r for r in records if r['kind'] == 'profile' and r['tau_truth'] == 1.3
             and r['kappa_truth'] == .45 and r['seed'] == cfg['synthetic']['seeds'][0]]
    fig, axes = plt.subplots(2, 2, figsize=(7, 6.8))
    for col, parameter in enumerate(('tau', 'kappa')):
        for modality, color in (('Hb_only', '#0072B2'), ('EEG_Hb', '#D55E00')):
            for prior, style in ((0., '--'), (1., '-')):
                rows = sorted([r for r in group if r['profile_parameter'] == parameter and
                    r['modality'] == modality and r['prior_multiplier'] == prior], key=lambda r: r['profile_value'])
                x = [r['profile_value'] for r in rows]
                axes[0, col].plot(x, [r['objective_components']['data'] for r in rows],
                    marker='o', color=color, ls=style, label=f'{modality}, penalty={prior:g}')
                axes[1, col].plot(x, [r['objective']-r['objective_components']['data'] for r in rows],
                    marker='o', color=color, ls=style)
                for ax, values in ((axes[0, col], [r['objective_components']['data'] for r in rows]),
                                   (axes[1, col], [r['objective']-r['objective_components']['data'] for r in rows])):
                    for r, y in zip(rows, values):
                        if not r['success']:
                            ax.scatter(r['profile_value'], y, marker='x', s=65, color='black', zorder=5)
        ticks = [.6, 1.3, 4.] if parameter == 'tau' else [.2, .45, 1.3]
        for row in (0, 1):
            ax = axes[row, col]
            ax.axvline(1.3 if parameter == 'tau' else .45, color='k', ls=':')
            ax.set_xscale('log')
            ax.set_xticks(ticks, [f'{v:g}' for v in ticks])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set(xlabel=parameter, ylabel='Data objective' if row == 0 else 'Separate prior penalty')
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.extend([Line2D([0], [0], ls=':', color='k'), Line2D([0], [0], marker='x', ls='', color='k')])
    labels.extend(['Generator truth', 'Nonconverged finite fit'])
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=10)
    fig.suptitle('Nonlinear profiles: free nuisance refit at each node\nSame truth, same noisy observations; no interval claim')
    fig.tight_layout(rect=(0, .19, 1, .91))
    fig.savefig(folder/'nonlinear_profiles.png', dpi=240, bbox_inches='tight')
    plt.close(fig)
    info = [r for record in records if record['kind'] == 'local_information' for r in record['information']
            if r['prior_multiplier'] == 0. and r['modality'] in ('Hb_only', 'EEG_Hb')]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    labels, values = [], []
    for modality in ('Hb_only', 'EEG_Hb'):
        for modes in (12, 120):
            labels.append(f'{modality}\nM={modes}')
            values.append([r['theta_profile_min_eigen'] for r in info if
                           r['modality'] == modality and r['driver_modes'] == modes])
    ax.boxplot(values, tick_labels=labels, showfliers=False)
    for index, ys in enumerate(values, 1):
        ax.scatter(np.full(len(ys), index), ys, s=18, alpha=.6)
    ax.set_yscale('log')
    ax.set_ylabel('Minimum projected log-physiology information eigenvalue')
    ax.grid(axis='y', alpha=.2)
    ax.set_title('Same generator driver; larger nuisance space\n16 truth × seed variants are paired, not 16 subjects')
    fig.tight_layout()
    fig.savefig(folder/'driver_capacity_information.png', dpi=240, bbox_inches='tight')
    plt.close(fig)
    return [str(p.relative_to(RUN)) for p in sorted(folder.glob('*.png'))]


def main():
    started = time.monotonic()
    manifest, cfg, summary = read(RUN/'manifest.json'), read(RUN/'resolved_config.json'), read(RUN/'summary.json')
    records = [read(RUN/'tasks'/f"{task['id']}.json") for task in manifest['tasks']]
    counts = dict(collections.Counter(r['status'] for r in records))
    check('terminal_denominator', len(records), manifest['planned'], atol=0, rtol=0)
    if counts != summary['status_counts']:
        failures.append(dict(check='status_counts', actual=counts, expected=summary['status_counts']))
    for name, digest in manifest['source_identity'].items():
        if hashlib.sha256((RUN/'source_snapshot'/name).read_bytes()).hexdigest() != digest:
            failures.append(dict(check='frozen_source_digest', path=name))
    cases, fit_count, start_count, start_success = {}, 0, 0, 0
    for task, record in zip(manifest['tasks'], records):
        for name, value in task.items():
            if record.get(name) != value:
                failures.append(dict(check='task_identity', id=task['id'], key=name))
        start_count += len(record.get('starts', []))
        start_success += sum(r['success'] for r in record.get('starts', []))
        if task['kind'] == 'retained_information':
            source = ROOT/cfg['data']['spatial_run']
            spec = task['source_cell']
            source_record = read(source/'cells'/f"{spec['id']}.json")
            if not source_record['success']:
                if record['status'] != 'unavailable_source_failure':
                    failures.append(dict(check='source_failure_preserved', id=record['id']))
                continue
            source_arrays = array(source/'cells'/f"{spec['id']}.npz")
            calibration = read(source/'calibration'/f"{spec['arm']}.json")
            source_cfg = read(source/'resolved_config.json')
            prepared = array(source/'prepared'/f"{hashlib.sha256(spec['record_id'].encode()).hexdigest()[:24]}.npz")
            p = source_cfg['fixed']
            parameters = BalloonParameters(free=BalloonFreeParameters(tau=p['tau'], kappa=p['kappa']))
            parameters = replace(parameters, fixed=replace(parameters.fixed,
                **{key: p[key] for key in ('alpha', 'E0', 'gamma', 'P0', 'Q0', 'eeg_loading', 'neurovascular_gain')}))
            forward = forward_spatial_shared_driver(source_arrays['coefficients'], source_arrays['initial_coordinates'],
                parameters, .25, temporal_basis=temporal_mode_basis(120, 16),
                eeg_loadings=calibration['readout']['eeg_loadings'], hb_loadings=calibration['design']['hb_loadings'],
                eeg_operator=prepared['eeg_operator'], hb_operator=prepared['hb_operator'],
                substeps=4, derivative=False, numerical_backend='numba')
            check('retained_forward_reproduction', forward['prediction'], source_arrays['prediction'])
            continue
        key = task['tau_truth'], task['kappa_truth'], task['seed']
        if key not in cases:
            cases[key] = model_case(cfg, task)
        case = cases[key]
        values = array(RUN/'tasks'/f"{record['id']}.npz")
        if task['kind'] == 'local_information':
            independent_local(cfg, task, case, record, values)
            continue
        fit_count += 1
        _, _, _, kwargs, truth = case
        check('truth_prediction_reproduction', values['truth_prediction'], truth['prediction'])
        check('truth_driver_reproduction', values['truth_drivers'], truth['drivers'])
        check('truth_states_reproduction', values['truth_states'], truth['states'])
        check('truth_initial_reproduction', values['truth_initial'], truth['initial_state'])
        metrics = record['metrics']
        parameters = BalloonParameters(free=BalloonFreeParameters(tau=metrics['tau'], kappa=metrics['kappa']))
        replay = forward_spatial_shared_driver(values['coefficients'], values['initial_coordinates'], parameters,
            .25, derivative=False, substeps=4, **kwargs)
        refined = forward_spatial_shared_driver(values['coefficients'], values['initial_coordinates'], parameters,
            .25, derivative=False, substeps=8, **kwargs)
        check('fitted_forward_reproduction', values['prediction'], replay['prediction'])
        check('fitted_state_reproduction', values['states'], replay['states'])
        maxima['fit_4_vs_8_max_standardized_prediction'] = max(maxima['fit_4_vs_8_max_standardized_prediction'],
            float(np.max(np.abs(replay['prediction']-refined['prediction'])/cfg['synthetic']['weighting_sd'])))
        initial = np.column_stack((values['initial_coordinates'][:, 0], np.exp(values['initial_coordinates'][:, 1:])))
        factor, scale = np.asarray(cfg['synthetic']['feature_weights']), np.asarray(cfg['synthetic']['weighting_sd'])
        residual = ((values['prediction']-values['target'])/scale*factor)[values['visible']]
        multiplier, priors = task['prior_multiplier'], cfg['priors']
        components = dict(data=float(residual@residual),
            driver_curvature=float(multiplier*priors['driver_curvature_weight']*.25*np.sum((np.diff(values['drivers'], n=2, axis=0)/.25**2)**2)),
            initial_state=float(multiplier*priors['initial_weight']*np.sum((initial-np.r_[0., np.ones(4)])**2)),
            flow_prior=float(multiplier*priors['flow_prior_weight']*.25*np.sum(np.log(values['states'][:, :, 2])**2)/priors['flow_log_sd']**2),
            physiology_prior=0.)
        if 'shifted_center' in task:
            components['physiology_prior'] = float(priors['shifted_prior_weight']*np.sum(
                (np.log(np.asarray([metrics['tau'], metrics['kappa']])/task['shifted_center'])/priors['physiology_prior_log_sd'])**2))
        for name, value in components.items():
            check('objective_'+name, value, record['objective_components'][name])
        check('objective_total', sum(components.values()), record['objective'])
        difference = values['drivers']-truth['drivers']
        theta_log = np.log([metrics['tau']/task['tau_truth'], metrics['kappa']/task['kappa_truth']])
        std_error = (values['prediction']-truth['prediction'])/scale
        recomputed = dict(tau_absolute_error=abs(metrics['tau']-task['tau_truth']),
            kappa_absolute_error=abs(metrics['kappa']-task['kappa_truth']),
            tau_log_error=theta_log[0], kappa_log_error=theta_log[1],
            theta_log_rmse=np.sqrt(np.mean(theta_log**2)),
            driver_nrmse=np.sqrt(np.mean(difference**2)/np.mean(truth['drivers']**2)),
            driver_dc_error=difference.mean(), initial_physical_rmse=np.sqrt(np.mean((initial-truth['initial_state'])**2)),
            prediction_clean_rmse=np.sqrt(np.mean((values['prediction']-truth['prediction'])**2)),
            hb_clean_rmse=np.sqrt(np.mean((values['prediction'][:, 3:]-truth['prediction'][:, 3:])**2)),
            visible_clean_standardized_nrmse=np.sqrt(np.mean(std_error[values['visible']]**2)),
            hb_clean_standardized_nrmse=np.sqrt(np.mean(std_error[:, 3:]**2)))
        for col, name in enumerate(('r', 's', 'f', 'v', 'p', 'q')):
            recomputed[f'state_{name}_rmse'] = np.sqrt(np.mean((values['states'][:, :, col]-truth['states'][:, :, col])**2))
        for name, value in recomputed.items():
            check('metric_'+name, value, metrics[name])
    check('optimizer_start_denominator', start_count, manifest['planned_optimizer_starts'], atol=0, rtol=0)
    check('optimizer_start_success_count', start_success, summary['successful_optimizer_starts'], atol=0, rtol=0)
    # Independently reconstruct each displayed ordinary recovery group.
    for modality in ('Hb_only', 'EEG_Hb'):
        for noise in (0., 1.):
            for prior in (0., .1, 1., 10.):
                selected = [r for r in records if r['kind'] == 'recovery' and r['modality'] == modality
                            and r['noise_multiplier'] == noise and r['prior_multiplier'] == prior]
                group = summary['groups'][f'recovery|{modality}|prior={prior:g}|noise={noise:g}']
                check('recovery_group_planned', len(selected), group['planned'], atol=0, rtol=0)
                success = [r for r in selected if r['success']]
                check('recovery_group_success', len(success), group['succeeded'], atol=0, rtol=0)
                check('recovery_group_failure', len(selected)-len(success), group['failures'], atol=0, rtol=0)
                for name, value in group['medians'].items():
                    if success:
                        check('recovery_group_median_'+name, np.median([r['metrics'][name] for r in success]), value)
                    elif value is not None:
                        failures.append(dict(check='failed_group_median_must_remain_undefined'))
    exports = figure_export(cfg, records)
    report = dict(schema='source_observability_independent_verification_v1',
        status='pass' if not failures else 'fail', failure_count=len(failures), failures=failures,
        planned=manifest['planned'], terminal=len(records), status_counts=counts,
        optimizer_starts=start_count, successful_optimizer_starts=start_success,
        recomputed_fit_objectives_and_truth_metrics=fit_count, independent_generator_cases=len(cases),
        independently_recomputed_local_cases=16, independently_replayed_retained_fits=71,
        maximum_absolute_differences=dict(maxima), elapsed_s=time.monotonic()-started,
        versioned_figure_exports=exports,
        scope='software_and_arithmetic_consistency_not_scientific_acceptance_of_failed_fits',
        integration='8vs16_generator_and4vs8_fit_reported_as_numeric_refinement_diagnostics_no_new_scientific_gate',
        evidence_mutation='none_original_JSON_NPZ_and_figures_preserved')
    (RUN/'verification.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
