"""Paired known-truth attribution tests using the existing nonlinear Balloon.

The finite tau profile is a model comparison diagnostic, not an interval or an
identifiability theorem. Ordinary arms refit the entire driver and all five
initial coordinates; oracle arms are labelled unavailable-information controls.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np

from .shared_driver_attribution import equal_capacity_hb_basis, project_component
from .shared_driver_reconstruction import nonlinear_driver_forward, fit_nonlinear_shared_driver
from .t3a_balloon_robust_ssm import BalloonParameters


def parameters_at_tau(tau):
    p = BalloonParameters()
    return replace(p, free=replace(p.free, tau=float(tau)))


def attribution_basis(steps, modes=4):
    return equal_capacity_hb_basis(np.eye(3*steps), modes=modes, rho=.35)


def make_worlds(cfg, seed, noise_multiplier, *, scale=None):
    """Generate all 2x2x2 worlds with identical driver, initial and noise draw.

    Hb gains multiply the clean dynamic observation, then the additive common
    HbT term and shared sensor noise are added. States are never rescaled.
    """
    syn, tensor = cfg['synthetic'], cfg['tensor']
    n, dt = tensor['steps'], tensor['dt_s']
    rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], int(seed)]))
    t = np.arange(n)*dt
    phases = rng.uniform(-np.pi, np.pi, 4)
    r = sum(np.sin(2*np.pi*t/period+phase)/np.sqrt(k+1)
            for k, (period, phase) in enumerate(zip([24., 15., 9., 6.], phases)))
    r *= syn['driver_rms']/np.sqrt(np.mean(r*r))
    initial = np.r_[rng.normal(0, .005), 1+rng.uniform(-.008, .008, 4)]
    # Keep the mathematical HbO domain explicit; no state clipping is needed.
    basis = attribution_basis(n, syn['common_modes'])
    coefficient = rng.normal(size=syn['common_modes'])
    coefficient *= syn['common_coefficient_rms']/np.sqrt(np.mean(coefficient**2))
    common = (basis@coefficient).reshape(n, 3)
    standard_noise = rng.normal(size=(n, 3))
    scale = np.ones(3) if scale is None else np.asarray(scale, float)
    noise = noise_multiplier*scale*standard_noise
    worlds = []
    for tau, gain, present in product(syn['tau_s'], syn['hb_gain'], syn['common_present']):
        forward = nonlinear_driver_forward(r, initial, parameters_at_tau(tau), dt,
            derivative=False, substeps=cfg['fit']['substeps'], numerical_backend='numba', hemodynamics='inlet_balance', viscoelastic_s=0.)
        clean = forward['canonical_prediction']
        component = common if present else np.zeros_like(common)
        gained = clean*np.array([1., gain, gain])
        worlds.append(dict(key=f'tau{tau:g}_gain{gain:g}_common{int(present)}',
            tau=float(tau), gain=float(gain), common_present=bool(present),
            driver=r.copy(), initial_state=initial.copy(), states=forward['states'],
            physical=clean, component=component, noise=noise.copy(),
            observation=gained+component+noise,
            nuisance=gained-clean+component, coefficient=coefficient.copy()))
    return worlds


def training_scales(cfg):
    worlds = [make_worlds(cfg, seed, 0)[0] for seed in cfg['synthetic']['training_seeds']]
    obs = np.concatenate([w['physical'] for w in worlds])
    vascular = np.concatenate([w['states'][:, 1:]-np.r_[0., np.ones(4)] for w in worlds])
    return dict(observation=np.sqrt(np.mean(obs*obs, axis=0)),
                driver=float(np.sqrt(np.mean(np.concatenate([w['driver'] for w in worlds])**2))),
                state=np.sqrt(np.mean(vascular*vascular, axis=0)))


def fit_profile(observation, cfg, scale, coefficient_sd, *, common=False,
                oracle_driver=None, oracle_initial=None):
    """Fit every node independently from zero driver/rest; preserve failures."""
    basis = attribution_basis(len(observation), cfg['synthetic']['common_modes'])
    nodes = []
    for tau in cfg['fit']['tau_grid_s']:
        p = parameters_at_tau(tau)
        try:
            if oracle_driver is None:
                options = {k: cfg['fit'][k] for k in ('penalty', 'initial_penalty',
                    'max_evaluations', 'gradient_tolerance', 'substeps', 'numerical_backend')}
                if common:
                    options.update(observation_basis=basis, observation_coefficient_sd=coefficient_sd)
                fit = fit_nonlinear_shared_driver(observation, p, cfg['tensor']['dt_s'],
                    sd=scale, **options)
            else:
                forward = nonlinear_driver_forward(oracle_driver, oracle_initial, p,
                    cfg['tensor']['dt_s'], derivative=False, substeps=cfg['fit']['substeps'],
                    numerical_backend='numba', hemodynamics='inlet_balance', viscoelastic_s=0.)
                physical = forward['canonical_prediction']
                coefficients = project_component(observation-physical, basis, scale,
                    coefficient_sd=coefficient_sd) if common else np.zeros(basis.shape[1])
                component = (basis@coefficients).reshape(observation.shape)
                cost = float(np.sum(((physical+component-observation)/scale)**2))
                component_cost = float(np.sum((coefficients/coefficient_sd)**2)) if common else 0.
                fit = dict(status='completed', converged=True, physical_prediction=physical,
                    prediction=physical+component, observation_component=component,
                    driver=np.asarray(oracle_driver), initial_state=np.asarray(oracle_initial),
                    states=forward['states'], weighted_data_sse=cost,
                    objective=cost+component_cost, observation_component_cost=component_cost,
                    evaluations=1, convergence_reason='finite_oracle_forward_no_optimization')
        except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
            fit = dict(status='failed_node_exception', converged=False, error=repr(exc))
        fit['tau'] = float(tau)
        nodes.append(fit)
    return nodes


def select_profile(nodes, fixed_tau=None):
    eligible = [v for v in nodes if v.get('converged') and
                (fixed_tau is None or v['tau'] == fixed_tau)]
    return min(eligible, key=lambda v: v['objective']) if eligible else None


def recovery_metrics(world, fit, scales, *, observation_prediction=None):
    if fit is None:
        return dict(status='failed_profile', converged=False)
    prediction = fit['prediction'] if observation_prediction is None else observation_prediction
    state_error = (fit['states'][:, 1:]-world['states'][:, 1:])/np.asarray(scales['state'])
    return dict(status='completed', converged=True, tau_hat_s=fit['tau'],
        tau_error_s=fit['tau']-world['tau'],
        tau_absolute_log_error=float(abs(np.log(fit['tau']/world['tau']))),
        observed_nrmse=float(np.sqrt(np.mean(((prediction-world['observation'])/scales['observation'])**2))),
        physical_Hb_nrmse=float(np.sqrt(np.mean(((fit['physical_prediction'][:, 1:]-world['physical'][:, 1:])/np.asarray(scales['observation'])[1:])**2))),
        driver_nrmse=float(np.sqrt(np.mean((fit['driver']-world['driver'])**2))/scales['driver']),
        state_nrmse=float(np.sqrt(np.mean(state_error**2))),
        component_nrmse=float(np.sqrt(np.mean(((fit['observation_component'][:, 1:]-world['component'][:, 1:])/np.asarray(scales['observation'])[1:])**2))),
        weighted_data_sse=float(fit['weighted_data_sse']), objective=float(fit['objective']))


def signed_projection(value, reference, scale):
    """Signed least-squares coefficient; not an energy fraction."""
    a, b = (np.asarray(value)/scale).ravel(), (np.asarray(reference)/scale).ravel()
    return float(a@b/(b@b)) if b@b > 1e-20 else np.nan


def paired_counterfactual_metrics(world, baseline, fit, baseline_fit, scales):
    if fit is None or baseline_fit is None:
        return {}
    delta_true = world['physical'][:, 1:]-baseline['physical'][:, 1:]
    delta_fit = fit['physical_prediction'][:, 1:]-baseline_fit['physical_prediction'][:, 1:]
    delta_component = fit['observation_component'][:, 1:]-baseline_fit['observation_component'][:, 1:]
    sd = np.asarray(scales['observation'])[1:]
    result = dict(delta_tau_error_s=float((fit['tau']-baseline_fit['tau'])-(world['tau']-baseline['tau'])),
        delta_physical_Hb_nrmse=float(np.sqrt(np.mean(((delta_fit-delta_true)/sd)**2))),
        tau_change_extra_projection=signed_projection(delta_component, delta_true, sd),
        nuisance_physical_compensation_projection=signed_projection(delta_fit-delta_true,
            world['nuisance'][:, 1:]-baseline['nuisance'][:, 1:], sd),
        true_component_change_projection=signed_projection(delta_component,
            world['component'][:, 1:]-baseline['component'][:, 1:], sd),
        gain_physical_compensation_projection=signed_projection(delta_fit-delta_true,
            (world['gain']-1)*world['physical'][:, 1:]-(baseline['gain']-1)*baseline['physical'][:, 1:], sd))
    result['absolute_delta_tau_error_s'] = abs(result['delta_tau_error_s'])
    for k, name in enumerate(['s', 'f', 'v', 'p', 'q']):
        result[f'state_{name}_rmse'] = float(np.sqrt(np.mean((fit['states'][:, k+1]-world['states'][:, k+1])**2)))
    return result
