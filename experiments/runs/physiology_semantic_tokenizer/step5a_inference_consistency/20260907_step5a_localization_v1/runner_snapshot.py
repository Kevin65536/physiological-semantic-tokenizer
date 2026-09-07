#!/usr/bin/env python3
"""Step5A0: bounded synthetic inference localization, never teacher admission.

Matched law: z0 ~ N(0, diag(initial_std**2)); z[t+1] = RK4(z[t])
+ N(0, dt*diag(process_std**2)); y[t,m] = h_m(z[t]) + scale_m*t_nu.
This is the discrete transition law used by the EKF, not an exact SDE solver.
Priors are densities in g, w, and physical zeta (no logistic Jacobian).
Only the oracle and particle estimator supply parameter_log_likelihood.
The historical EKF marginal score defines a diagnostic generalized posterior.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import numpy as np
import scipy
import yaml
from scipy.integrate import cumulative_trapezoid, solve_ivp, trapezoid
from scipy.special import gammaln, logsumexp
from scipy.stats import multivariate_normal, t as student_t, truncnorm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import src.inference.t3a_balloon_robust_ssm as ssm
from experiments.evaluate_t3c_composite_synthetic_t2 import _truth_trial as step4_truth

DEFAULT_CONFIG = REPO_ROOT / "experiments/configs/physiology_semantic_tokenizer/step5a_inference_consistency_v1.yaml"
OUTPUT_ROOT = "experiments/runs/physiology_semantic_tokenizer/step5a_inference_consistency"
TARGETS = ("r", "clean_EEG", "clean_HbO", "clean_HbR")


def load_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    if cfg["schema"] != "step5a_inference_consistency_v1" or cfg["scope"] != "synthetic_only_localization":
        raise ValueError("Step5A is a synthetic-only localization experiment")
    if cfg["output_root"] != OUTPUT_ROOT or set(cfg["axes"]) != {"G", "W", "Z"}:
        raise ValueError("invalid output root or GWZ axes")
    if int(cfg["replicates_per_axis"]) < 1:
        raise ValueError("replicates must be positive")
    for axis, spec in cfg["axes"].items():
        lo, hi = spec["bounds"]
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError("invalid prior support")
        if axis == "Z":
            if spec["prior"] != "uniform" or not 0 < lo < hi < 1:
                raise ValueError("Z prior must be uniform in physical underdamped zeta")
        elif spec["prior"] != "truncated_normal" or spec["sd"] <= 0:
            raise ValueError("G/W require proper truncated normal priors")
    params, numerics = model(cfg, "G", 0.0)
    params.validate()
    numerics.validate()
    for section in ("inference", "reference"):
        n = cfg[section]["grid_points"]
        if n < 5 or n % 2 != 1:
            raise ValueError("nested quadrature requires an odd grid >= 5")
    ref = cfg["reference"]
    if len(ref["particles"]) != 2 or not 1 < ref["particles"][0] < ref["particles"][1]:
        raise ValueError("reference requires two increasing particle budgets")
    if ref["independent_runs"] < 4 or ref["independent_runs"] % 2:
        raise ValueError("reference requires >= 4 even independent runs")
    if not 0 <= ref["replicate_index"] < cfg["replicates_per_axis"]:
        raise ValueError("reference replicate outside panel")
    if not 2 <= ref["steps"] <= cfg["model"]["steps"]:
        raise ValueError("reference must be a short matched prefix")
    if not 0 < cfg["inference"]["mask_center_steps"] < cfg["model"]["steps"]:
        raise ValueError("invalid center mask")
    if not 0 < cfg["inference"]["boundary_band_fraction"] < 0.5:
        raise ValueError("invalid boundary band")


def raw_parameters(cfg, axis, value):
    ref = cfg["model"]["reference"]
    g, w, zeta = 0.0, 0.0, ref["kappa"] / (2 * np.sqrt(ref["gamma"]))
    if axis == "G":
        g = value
    elif axis == "W":
        w = value
    elif axis == "Z":
        zeta = value
    else:
        raise ValueError("unknown axis")
    omega = np.sqrt(ref["gamma"]) * np.exp(w)
    return dict(ref, beta=(ref["beta"] / ref["gamma"]) * np.exp(g) * omega**2,
                gamma=omega**2, kappa=2 * zeta * omega)


def model(cfg, axis, value):
    raw = raw_parameters(cfg, axis, value)
    m = cfg["model"]
    fixed = ssm.BalloonFixedParameters(
        alpha=raw["alpha"], E0=raw["E0"], gamma=raw["gamma"],
        neurovascular_gain=raw["beta"], process_std=tuple(m["process_std"]),
        observation_scale=tuple(m["observation_scale"]), student_nu=m["student_nu"])
    return (ssm.BalloonParameters(fixed, ssm.BalloonFreeParameters(raw["kappa"], raw["tau"])),
            ssm.BalloonConfig(dt=m["dt"], rk4_substeps=m["rk4_substeps"],
                              initial_state_std=tuple(m["initial_state_std"])))


def prior(cfg, axis):
    s = cfg["axes"][axis]
    lo, hi = s["bounds"]
    if axis == "Z":
        from scipy.stats import uniform
        return uniform(lo, hi - lo)
    return truncnorm((lo-s["mean"])/s["sd"], (hi-s["mean"])/s["sd"], loc=s["mean"], scale=s["sd"])


def independent_rhs(z, raw):
    """Independent batch implementation, checked against the production core."""
    r, s, lf, lv, lp, lq = np.moveaxis(z, -1, 0)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        f, v, p, q = np.exp([lf, lv, lp, lq])
        out = v ** (1 / raw["alpha"])
        extraction = -np.expm1(np.log1p(-raw["E0"]) / f)
        result = np.stack((-0.45*r, raw["beta"]*r - raw["kappa"]*s - raw["gamma"]*(f-1),
                           s/f, (f-out)/(raw["tau"]*v),
                           (f-out*p/v)/(raw["tau"]*p),
                           (f*extraction/raw["E0"]-out*q/v)/(raw["tau"]*q)), axis=-1)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite reference dynamics; no clipping or redraw")
    return result


def independent_transition(z, raw, cfg):
    h = cfg["model"]["dt"] / cfg["model"]["rk4_substeps"]
    for _ in range(cfg["model"]["rk4_substeps"]):
        k1 = independent_rhs(z, raw)
        k2 = independent_rhs(z + h*k1/2, raw)
        k3 = independent_rhs(z + h*k2/2, raw)
        k4 = independent_rhs(z + h*k3, raw)
        z = z + h*(k1+2*k2+2*k3+k4)/6
    return z


def clean_from_z(z):
    with np.errstate(over="raise", invalid="raise"):
        p, q = np.exp(z[..., 4]), np.exp(z[..., 5])
    hbr = 0.35*(q-1)
    return np.stack((z[..., 0], p-1-hbr, hbr), axis=-1)


def generate_matched(cfg, axis, theta, seed):
    rng = np.random.default_rng(seed)
    m = cfg["model"]
    z = np.empty((m["steps"], 6))
    z[0] = rng.normal(size=6)*m["initial_state_std"]
    raw = raw_parameters(cfg, axis, theta)
    for t in range(1, len(z)):
        z[t] = independent_transition(z[t-1], raw, cfg) + rng.normal(size=6)*np.asarray(m["process_std"])*np.sqrt(m["dt"])
    clean = clean_from_z(z)
    y = clean + rng.standard_t(m["student_nu"], clean.shape)*m["observation_scale"]
    states = z.copy()
    states[:, 2:] = np.exp(states[:, 2:])
    return dict(observations=y, clean=clean, states=states, transformed_states=z)


def stress_contract(cfg):
    m = cfg["model"]
    return {"simulation": {
        "sampling_hz": 1/m["dt"], "duration_s_per_trial": m["steps"]*m["dt"],
        "driver": {"decay_per_s": 0.45, "diffusion_sd_per_sqrt_s": 0.08, "pulse_amplitude": 0.12},
        "noise": {"df": m["student_nu"], "scale": dict(zip(("EEG", "HbO", "HbR"), m["observation_scale"]))},
        "observation": {"P0": 1.0, "Q0": 0.35}}}


def deterministic_clean(cfg, axis, grid, driver, method="DOP853", tighter=False):
    """Physical ODE with known, linearly interpolated r; zero state diffusion."""
    raw = raw_parameters(cfg, axis, np.asarray(grid))
    n = len(grid)
    times = np.arange(len(driver))*cfg["model"]["dt"]
    def rhs(t, flat):
        s, f, v, p, q = flat.reshape(n, 5).T
        out = v**(1/raw["alpha"])
        ef = -np.expm1(np.log1p(-raw["E0"])/f)
        return np.stack((raw["beta"]*np.interp(t, times, driver)-raw["kappa"]*s-raw["gamma"]*(f-1),
                         s, (f-out)/raw["tau"], (f-out*p/v)/raw["tau"],
                         (f*ef/raw["E0"]-out*q/v)/raw["tau"]), axis=-1).ravel()
    factor = 0.1 if tighter else 1.0
    sol = solve_ivp(rhs, (times[0], times[-1]), np.tile([0., 1., 1., 1., 1.], n),
                    t_eval=times, method=method, max_step=cfg["model"]["dt"]/4,
                    rtol=cfg["oracle"]["integration_rtol"]*factor,
                    atol=cfg["oracle"]["integration_atol"]*factor)
    if not sol.success:
        raise RuntimeError(sol.message)
    states = sol.y.T.reshape(len(times), n, 5).transpose(1, 0, 2)
    hbr = 0.35*(states[..., 4]-1)
    return np.stack((np.broadcast_to(driver, hbr.shape), states[..., 3]-1-hbr, hbr), axis=-1)


def posterior_grid(cfg, axis, grid, log_likelihood):
    log_density = np.asarray(log_likelihood) + prior(cfg, axis).logpdf(grid)
    density = np.exp(log_density - np.max(log_density))
    density /= trapezoid(density, grid)
    cdf = cumulative_trapezoid(density, grid, initial=0)
    cdf[-1] = 1.0
    weights = np.empty(len(grid))
    weights[0], weights[-1] = (grid[1]-grid[0])/2, (grid[-1]-grid[-2])/2
    weights[1:-1] = (grid[2:]-grid[:-2])/2
    weights *= density
    return dict(density=density, cdf=cdf, weights=weights)


def posterior_summary(cfg, axis, grid, score, truth):
    p = posterior_grid(cfg, axis, grid, score)
    lo, hi = grid[0], grid[-1]
    band = cfg["inference"]["boundary_band_fraction"]*(hi-lo)
    mle, mode = grid[np.argmax(score)], grid[np.argmax(p["density"])]
    q = np.interp([0.025, 0.975], p["cdf"], grid)
    return dict(mean=float(np.dot(p["weights"], grid)), lower95=float(q[0]), upper95=float(q[1]),
                width_fraction=float((q[1]-q[0])/(hi-lo)), rank_u=float(np.interp(truth, grid, p["cdf"])),
                covered95=bool(q[0] <= truth <= q[1]), likelihood_only_mode=float(mle), posterior_mode=float(mode),
                mode_separation_fraction=float(abs(mle-mode)/(hi-lo)),
                mle_boundary_distance_fraction=float(min(mle-lo, hi-mle)/(hi-lo)),
                boundary_mass=float(np.interp(lo+band, grid, p["cdf"])+1-np.interp(hi-band, grid, p["cdf"])))


def parameterization_check(cfg):
    rng = np.random.default_rng(cfg["seed"]+1)
    transition_error, derivative_error, chain_error, observation_error = [], [], [], []
    for axis in cfg["axes"]:
        lo, hi = cfg["axes"][axis]["bounds"]
        for value in (lo, (lo+hi)/2, hi):
            p, c = model(cfg, axis, value)
            raw = raw_parameters(cfg, axis, value)
            z = rng.normal(size=6)*np.array([.06, .02, .015, .015, .015, .015])
            for _ in range(16):
                z_ref = independent_transition(z, raw, cfg)
                z_prod, jac = ssm.rk4_transition_with_jacobian(z, p, c)
                transition_error.append(np.max(abs(z_ref-z_prod)))
                eps = 1e-5
                finite = np.column_stack([(independent_transition(z+eps*np.eye(6)[j], raw, cfg)-independent_transition(z-eps*np.eye(6)[j], raw, cfg))/(2*eps) for j in range(6)])
                derivative_error.append(np.max(abs(jac-finite)))
                observation_error.append(np.max(abs(clean_from_z(z)-ssm.observation_map(ssm.transformed_to_physical(z), p))))
                z = z_prod
            d_axis = (ssm.rk4_transition(z, model(cfg, axis, value+eps)[0], c)-ssm.rk4_transition(z, model(cfg, axis, value-eps)[0], c))/(2*eps)
            coefficients = {"G": {"beta": raw["beta"]}, "W": {"beta": 2*raw["beta"], "gamma": 2*raw["gamma"], "kappa": raw["kappa"]}, "Z": {"kappa": 2*np.sqrt(raw["gamma"])}}[axis]
            d_chain = np.zeros(6)
            for name, coefficient in coefficients.items():
                plus, minus = dict(raw), dict(raw)
                plus[name] += eps
                minus[name] -= eps
                d_chain += coefficient*(independent_transition(z, plus, cfg)-independent_transition(z, minus, cfg))/(2*eps)
            chain_error.append(np.max(abs(d_axis-d_chain)))
    a, d, k, h = map(lambda x: float(max(x)), (transition_error, derivative_error, chain_error, observation_error))
    return dict(transition_max_abs=a, state_jacobian_max_abs=d, parameter_chain_rule_max_abs=k,
                observation_max_abs=h, passed=bool(max(a,h) <= cfg["checks"]["transition_absolute_tolerance"] and max(d,k) <= cfg["checks"]["derivative_absolute_tolerance"]))


def linear_gaussian_check(cfg):
    """Production filter/RTS specialized to a linear, Gaussian limiting case.

    Patch only nonlinear model operators/moments; use the actual IRLS, covariance
    propagation, and RTS code. nu=1e12 gives the Gaussian observation limit.
    """
    p, c = model(cfg, "G", 0.)
    p = replace(p, fixed=replace(p.fixed, student_nu=1e12))
    a = np.eye(6)*.85
    a[4, 0], a[5, 4] = .15, .1
    h = np.zeros((3, 6))
    h[0, 0], h[1, 4], h[1, 5], h[2, 5] = 1., 1., -.35, .35
    q = np.diag(np.square(p.fixed.process_std))*c.dt
    r = np.diag(np.square(p.fixed.observation_scale))
    rng = np.random.default_rng(cfg["seed"]+2)
    y = rng.normal(size=(20, 3))*p.fixed.observation_scale
    mask = np.ones_like(y, dtype=bool)
    mask[7:10, 1:] = False
    mu, cov = np.zeros(6), np.diag(np.square(c.initial_state_std))
    fm, fc, pm, pc = [], [], [], []
    exact_ll = 0.
    for t, row in enumerate(y):
        if t:
            mu, cov = a@mu, a@cov@a.T+q
        pm.append(mu.copy()); pc.append(cov.copy())
        ht, rt = h[mask[t]], r[np.ix_(mask[t], mask[t])]
        innovation = row[mask[t]]-ht@mu
        pred = ht@cov@ht.T+rt
        exact_ll += multivariate_normal.logpdf(innovation, cov=pred)
        gain = np.linalg.solve(pred, ht@cov).T
        mu, cov = mu+gain@innovation, cov-gain@ht@cov
        fm.append(mu.copy()); fc.append(cov.copy())
    sm, sc = np.array(fm), np.array(fc)
    for t in range(len(y)-2, -1, -1):
        gain = np.linalg.solve(pc[t+1], a@fc[t]).T
        sm[t] += gain@(sm[t+1]-pm[t+1])
        sc[t] += gain@(sc[t+1]-pc[t+1])@gain.T
    moment_covariances = []
    def linear_moments(mu, cov):
        moment_covariances.append(cov.copy())
        return mu, cov
    with patch.multiple(ssm,
            rk4_transition_with_jacobian=lambda z, *args: (a@z, a),
            _transformed_to_physical_unchecked=lambda z: z,
            transformed_gaussian_moments=linear_moments,
            _observation_map_unchecked=lambda z, *args: h@z,
            _observation_jacobian_unchecked=lambda *args: h,
            _observation_physical_matrix=lambda *args: h,
            _physical_checks=lambda *args: {}):
        result = ssm.smooth_balloon(y, p, config=c, observation_mask=mask)
    mean_error = float(np.max(abs(result.state_mean-sm)))
    cov_error = float(np.max(abs(np.array(moment_covariances[-len(y):])-sc)))
    return dict(mean_max_abs=mean_error, covariance_max_abs=cov_error,
                exact_joint_log_likelihood=float(exact_ll), predictive_score=result.predictive_log_likelihood,
                score_minus_joint=float(result.predictive_log_likelihood-exact_ll),
                passed=bool(max(mean_error, cov_error) <= cfg["checks"]["linear_mean_covariance_tolerance"]))


def particle_filter(y, cfg, axis, theta, particles, seed, paths=False):
    """Bootstrap PF with joint conditional t density and systematic resampling.

    Returned exp(log_likelihood) is an unbiased likelihood estimator for the
    declared discrete model. No probability clipping or particle rejection.
    """
    rng = np.random.default_rng(seed)
    m = cfg["model"]
    raw = raw_parameters(cfg, axis, theta)
    z = rng.normal(size=(particles, 6))*m["initial_state_std"]
    scales = np.asarray(m["observation_scale"])
    nu = m["student_nu"]
    constant = gammaln((nu+1)/2)-gammaln(nu/2)-.5*np.log(nu*np.pi)-np.log(scales)
    ll, min_ess = 0., 1.
    labels = np.arange(particles)
    history, parents = [], []
    weights = np.ones(particles)/particles
    for t, row in enumerate(y):
        if t:
            cdf = np.cumsum(weights)
            cdf[-1] = 1.
            ix = np.searchsorted(cdf, (rng.random()+np.arange(particles))/particles)
            labels = labels[ix]
            z = independent_transition(z[ix], raw, cfg)+rng.normal(size=z.shape)*np.asarray(m["process_std"])*np.sqrt(m["dt"])
            if paths:
                parents.append(ix)
        visible = np.isfinite(row)
        residual = (row[visible]-clean_from_z(z)[:, visible])/scales[visible]
        lw = np.sum(constant[visible]-(nu+1)/2*np.log1p(residual**2/nu), axis=1)
        norm = logsumexp(lw)
        ll += norm-np.log(particles)
        weights = np.exp(lw-norm)
        min_ess = min(min_ess, 1/(particles*np.sum(weights**2)))
        if paths:
            history.append(z.copy())
    result = dict(parameter_log_likelihood=float(ll), minimum_ess_fraction=float(min_ess),
                  unique_ancestor_fraction=float(len(np.unique(labels))/particles))
    if paths:
        ix = np.arange(particles)
        trajectories = np.empty((len(y), particles, 4))
        for t in range(len(y)-1, -1, -1):
            states = history[t][ix]
            trajectories[t] = np.column_stack((states[:, 0], clean_from_z(states)))
            if t:
                ix = parents[t-1][ix]
        mean = np.einsum("tnm,n->tm", trajectories, weights)
        variance = np.einsum("tnm,n->tm", (trajectories-mean[:, None])**2, weights)
        result.update(mean=mean, variance=variance)
    return result


def oracle_case(cfg, axis, truth, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(cfg["oracle"]["steps"])*cfg["model"]["dt"]
    driver = .08*np.sin(.35*t)+.05*np.sin(.9*t)+.12*np.exp(-.5*((t-.52*t[-1])/max(.8,.06*t[-1]))**2)
    clean = deterministic_clean(cfg, axis, [truth], driver, method="RK45", tighter=True)[0]
    y = clean+rng.standard_t(cfg["model"]["student_nu"], clean.shape)*cfg["model"]["observation_scale"]
    grid = np.linspace(*cfg["axes"][axis]["bounds"], 2*cfg["inference"]["grid_points"]-1)
    fitted = deterministic_clean(cfg, axis, grid, driver)
    score = np.sum(student_t.logpdf((y[None]-fitted)/cfg["model"]["observation_scale"], df=cfg["model"]["student_nu"])-np.log(cfg["model"]["observation_scale"]), axis=(1,2))
    fine, coarse = posterior_grid(cfg, axis, grid, score), posterior_grid(cfg, axis, grid[::2], score[::2])
    difference = float(max(abs(fine["cdf"]-np.interp(grid, grid[::2], coarse["cdf"]))))
    truth_check = deterministic_clean(cfg, axis, [truth], driver)[0]
    return dict(**posterior_summary(cfg, axis, grid, score, truth), grid=grid, parameter_log_likelihood=score,
                grid_cdf_difference=difference, integration_max_abs=float(np.max(abs(truth_check-clean))),
                quadrature_pass=bool(difference <= cfg["checks"]["oracle_grid_cdf_tolerance"]))


def fit_score_grid(y, cfg, axis, grid):
    """No truth/driver argument: diagnostic score-based parameter distribution."""
    score = []
    for value in grid:
        p, c = model(cfg, axis, value)
        score.append(ssm.smooth_balloon(y, p, config=c).predictive_log_likelihood)
    return np.asarray(score)


def state_metrics(generated, result, selector=None):
    truth = np.column_stack((generated["states"][:, 0], generated["clean"]))
    mean = np.column_stack((result.shared_driver_mean, result.trajectory_mean))
    variance = np.column_stack((result.shared_driver_variance, result.epistemic_variance))
    sel = np.ones(len(truth), dtype=bool) if selector is None else selector
    err = (mean-truth)[sel]
    sd = np.sqrt(variance[sel])
    return {name: dict(rmse=float(np.sqrt(np.mean(err[:, j]**2))),
                       nrmse=float(np.sqrt(np.mean(err[:, j]**2))/max(np.std(truth[sel, j]), 1e-12)),
                       coverage95=float(np.mean(abs(err[:, j]) <= 1.959963984540054*sd[:, j])),
                       mean_state_posterior_variance=float(np.mean(variance[sel, j]))) for j, name in enumerate(TARGETS)}


def reference_case(cfg, axis, truth, y, seed):
    ref = cfg["reference"]
    y = y[:ref["steps"]]
    grid = np.linspace(*cfg["axes"][axis]["bounds"], ref["grid_points"])
    logs, ess, ancestry = [], [], []
    for level, n in enumerate(ref["particles"]):
        level_logs = []
        for run in range(ref["independent_runs"]):
            row = []
            for j, theta in enumerate(grid):
                out = particle_filter(y, cfg, axis, theta, n, seed+level*100000+run*1000+j)
                row.append(out["parameter_log_likelihood"])
                ess.append(out["minimum_ess_fraction"])
                ancestry.append(out["unique_ancestor_fraction"])
            level_logs.append(row)
        logs.append(np.array(level_logs))
    logs = np.array(logs)
    combined = logsumexp(logs, axis=1)-np.log(ref["independent_runs"])
    low, high = [posterior_grid(cfg, axis, grid, row) for row in combined]
    split = ref["independent_runs"]//2
    halves = [posterior_grid(cfg, axis, grid, logsumexp(part, axis=0)-np.log(len(part))) for part in (logs[1,:split], logs[1,split:])]
    normalized_l = np.exp(logs[1]-np.max(logs[1], axis=0))
    se = normalized_l.std(axis=0, ddof=1)/np.sqrt(ref["independent_runs"])/normalized_l.mean(axis=0)
    cdf_diff = float(max(abs(low["cdf"]-high["cdf"])))
    split_diff = float(max(abs(halves[0]["cdf"]-halves[1]["cdf"])))
    coarse = posterior_grid(cfg, axis, grid[::2], combined[1,::2])
    grid_diff = float(max(abs(high["cdf"]-np.interp(grid, grid[::2], coarse["cdf"]))))
    resolved = bool(max(se) <= ref["max_log_likelihood_se"] and max(cdf_diff, split_diff, grid_diff) <= ref["max_cdf_difference"] and min(ess) >= ref["minimum_ess_fraction"] and min(ancestry) >= ref["minimum_unique_ancestor_fraction"])
    score = fit_score_grid(y, cfg, axis, grid)
    approximate = posterior_grid(cfg, axis, grid, score)
    # Conditional true-parameter smoothing is a separate state diagnostic.
    paths = [particle_filter(y, cfg, axis, truth, ref["particles"][1], seed+300000+k, paths=True) for k in range(ref["independent_runs"])]
    means = np.array([p["mean"] for p in paths])
    mean = means.mean(axis=0)
    variance = np.mean([p["variance"] for p in paths], axis=0)+means.var(axis=0)
    p, c = model(cfg, axis, truth)
    ekf = ssm.smooth_balloon(y, p, config=c)
    emean = np.column_stack((ekf.shared_driver_mean, ekf.trajectory_mean))
    state_se = means.std(axis=0, ddof=1)/np.sqrt(len(paths))
    return dict(grid=grid, independent_log_likelihoods=logs, parameter_log_likelihood=combined[1],
                predictive_score=score, maximum_log_likelihood_se=float(max(se)),
                particle_budget_cdf_difference=cdf_diff, split_run_cdf_difference=split_diff, grid_cdf_difference=grid_diff,
                minimum_ess_fraction=float(min(ess)), minimum_unique_ancestor_fraction=float(min(ancestry)),
                reference_precision_pass=resolved,
                comparison_status="RESOLVED_SHORT_CASE" if resolved else "INCONCLUSIVE_REFERENCE_PRECISION",
                posterior_cdf_difference=float(max(abs(high["cdf"]-approximate["cdf"]))),
                centered_score_shape_max_abs=float(max(abs((score-score.max())-(combined[1]-combined[1].max())))),
                posterior=posterior_summary(cfg, axis, grid, combined[1], truth),
                reference_state_mean=mean, reference_state_posterior_variance=variance,
                reference_state_mean_max_mc_se=np.max(state_se, axis=0),
                ekf_reference_state_rmse=np.sqrt(np.mean((emean-mean)**2, axis=0)),
                state_comparison_status="DESCRIPTIVE_MONTE_CARLO_SE_REPORTED_NO_STATE_PRECISION_GATE")


def run_case(cfg, axis, replicate):
    base = cfg["seed"]+1000000*(list(cfg["axes"]).index(axis)+1)+replicate*10000
    truth = float(prior(cfg, axis).rvs(random_state=np.random.default_rng(base)))
    row = dict(axis=axis, replicate=replicate, truth=truth, seed=base)
    row["oracle_r_known"] = oracle_case(cfg, axis, truth, base+1)
    grid = np.linspace(*cfg["axes"][axis]["bounds"], cfg["inference"]["grid_points"])
    matched = generate_matched(cfg, axis, truth, base+2)
    stress = step4_truth(raw_parameters(cfg, axis, truth), base+3, stress_contract(cfg))
    payload = {}
    for name, generated in (("matched_model_calibration", matched), ("misspecification_stress_test", stress)):
        y = generated["observations"]
        score = fit_score_grid(y, cfg, axis, grid)
        post = posterior_summary(cfg, axis, grid, score, truth)
        full = posterior_grid(cfg, axis, grid, score)
        coarse = posterior_grid(cfg, axis, grid[::2], score[::2])
        post["grid_cdf_difference"] = float(max(abs(full["cdf"]-np.interp(grid, grid[::2], coarse["cdf"]))))
        post["objective_kind"] = "generalized_posterior_from_marginal_predictive_score"
        post["parameter_log_likelihood"] = None
        post["predictive_score"] = score
        post["grid"] = grid
        post["state_recovery"] = {}
        center = np.zeros(len(y), dtype=bool)
        start = (len(y)-cfg["inference"]["mask_center_steps"])//2
        center[start:start+cfg["inference"]["mask_center_steps"]] = True
        baseline = 0. if axis != "Z" else cfg["model"]["reference"]["kappa"]/(2*np.sqrt(cfg["model"]["reference"]["gamma"]))
        for label, value in (("U0_FIXED", baseline), ("true_parameters", truth), ("score_mode", post["likelihood_only_mode"])):
            p, c = model(cfg, axis, value)
            result = ssm.smooth_balloon(y, p, config=c)
            metrics = dict(full=state_metrics(generated, result))
            metrics["full_predictive_score"] = result.predictive_log_likelihood
            # Frozen scales, no preprocessing. Mask before all estimation.
            hidden = y.copy()
            hidden[center, 1:] = np.nan
            masked = ssm.smooth_balloon(hidden, p, config=c)
            metrics["masked_center"] = state_metrics(generated, masked, center)
            # Fitted masked parameters must also use visible observations only.
            metrics["masked_parameter_conditioning"] = "fixed_a_priori_or_true_parameter_diagnostic"
            if label == "score_mode":
                hidden_score = fit_score_grid(hidden, cfg, axis, grid)
                masked_theta = float(grid[np.argmax(hidden_score)])
                mp, mc = model(cfg, axis, masked_theta)
                masked = ssm.smooth_balloon(hidden, mp, config=mc)
                metrics["masked_center"] = state_metrics(generated, masked, center)
                metrics["masked_parameter_conditioning"] = "refitted_on_visible_observations_only"
                metrics["masked_parameter"] = masked_theta
            metrics["paired_mask_variance_change"] = {target: metrics["masked_center"][target]["mean_state_posterior_variance"]-state_metrics(generated, result, center)[target]["mean_state_posterior_variance"] for target in TARGETS}
            # Approximate future noisy-observation intervals reported separately.
            noise_truth = y[center, 1:]
            width = 1.959963984540054*np.sqrt(masked.total_variance[center, 1:])
            metrics["masked_noisy_observation_coverage95_gaussian_moment_interval"] = np.mean(abs(noise_truth-masked.trajectory_mean[center, 1:]) <= width, axis=0)
            post["state_recovery"][label] = metrics
            payload[f"{name}_{label}_state_mean"] = result.state_mean
            payload[f"{name}_{label}_state_posterior_variance"] = result.state_variance
            payload[f"{name}_{label}_clean_mean"] = result.trajectory_mean
            payload[f"{name}_{label}_clean_state_posterior_variance"] = result.epistemic_variance
        row[name] = post
        for key in ("observations", "states", "clean"):
            payload[f"{name}_{key}"] = generated[key]
    if replicate == cfg["reference"]["replicate_index"]:
        row["nonlinear_reference"] = reference_case(cfg, axis, truth, matched["observations"], base+1000)
    return row, payload


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def summarize(cfg, rows, checks):
    summary = dict(schema=cfg["schema"], execution="completed", scientific_verdict="localization_only_no_admission",
                   checks=checks, independent_cases=len(rows), axes={})
    rng = np.random.default_rng(cfg["seed"]+3)
    for axis in cfg["axes"]:
        cases = [r for r in rows if r["axis"] == axis]
        result = {}
        for kind in ("oracle_r_known", "matched_model_calibration", "misspecification_stress_test"):
            values = [r[kind] for r in cases]
            bias = np.array([v["mean"]-r["truth"] for v,r in zip(values,cases)])
            result[kind] = dict(parameter_bias=float(bias.mean()), parameter_rmse=float(np.sqrt(np.mean(bias**2))),
                                parameter_coverage95=float(np.mean([v["covered95"] for v in values])),
                                ranks=[v["rank_u"] for v in values],
                                max_grid_cdf_difference=max(v["grid_cdf_difference"] for v in values),
                                mean_boundary_mass=float(np.mean([v["boundary_mass"] for v in values])))
            if kind != "oracle_r_known":
                result[kind]["state_recovery"] = {}
                for label in ("U0_FIXED", "true_parameters", "score_mode"):
                    state = {}
                    for target in TARGETS:
                        coverage = np.array([v["state_recovery"][label]["full"][target]["coverage95"] for v in values])
                        boot = coverage[rng.integers(0,len(cases),(cfg["inference"]["bootstrap_repetitions"],len(cases)))].mean(axis=1)
                        state[target] = dict(mean_coverage95=float(coverage.mean()),
                            replicate_bootstrap_ci95=np.quantile(boot,[.025,.975]),
                            mean_rmse=float(np.mean([v["state_recovery"][label]["full"][target]["rmse"] for v in values])),
                            mean_masked_center_coverage95=float(np.mean([v["state_recovery"][label]["masked_center"][target]["coverage95"] for v in values])),
                            mean_paired_mask_variance_change=float(np.mean([v["state_recovery"][label]["paired_mask_variance_change"][target] for v in values])))
                    result[kind]["state_recovery"][label] = state
        reference = next((r["nonlinear_reference"] for r in cases if "nonlinear_reference" in r), None)
        if reference:
            result["nonlinear_reference"] = {k:v for k,v in reference.items() if k not in {"grid", "independent_log_likelihoods", "parameter_log_likelihood", "predictive_score", "reference_state_mean", "reference_state_posterior_variance"}}
        summary["axes"][axis] = result
    return summary


def report_markdown(summary):
    lines = ["# Step5A_inference_consistency — synthetic localization", "",
             "这是 Step5A0 小样本定位实验，不是 60 次 SBC，也不授予参数解释或 teacher 资格。",
             "matched 是显式离散 RK4 + Gaussian innovation 模型；stress 原样调用 Step 4 生成函数，缩短记录并保留脉冲/确定性血流失配。",
             "oracle 先验定义在 g/w/物理 zeta 上；网格使用梯形积分。EKF 曲线仅称 predictive_score，参数分布为 generalized posterior。",
             "状态区间为条件于参数的 Gaussian moment 近似；clean 区间不加入观测噪声。coverage 先按独立 replicate 汇总，bootstrap 仅作描述。", "",
             "| Check | Result |", "|---|---|",
             *[f"| {k} | {json.dumps(v, ensure_ascii=False)} |" for k,v in summary["checks"].items()], "",
             "| Axis / law | Bias | Parameter 95% coverage | Max grid CDF Δ |", "|---|---:|---:|---:|"]
    for axis, groups in summary["axes"].items():
        for law in ("oracle_r_known", "matched_model_calibration", "misspecification_stress_test"):
            v = groups[law]
            lines.append(f"| {axis} / {law} | {v['parameter_bias']:.5f} | {v['parameter_coverage95']:.3f} | {v['max_grid_cdf_difference']:.5f} |")
    lines += ["", "| Axis | Reference precision | Max logL SE | Budget / split / grid CDF Δ | EKF–PF CDF Δ |", "|---|---|---:|---|---:|"]
    for axis,g in summary["axes"].items():
        v = g.get("nonlinear_reference")
        if v:
            lines.append(f"| {axis} | {v['comparison_status']} | {v['maximum_log_likelihood_se']:.4f} | {v['particle_budget_cdf_difference']:.4f} / {v['split_run_cdf_difference']:.4f} / {v['grid_cdf_difference']:.4f} | {v['posterior_cdf_difference']:.4f} |")
    lines += ["", "| Axis / law / estimator | r coverage | EEG coverage | HbO coverage | HbR coverage |", "|---|---:|---:|---:|---:|"]
    for axis,g in summary["axes"].items():
        for law in ("matched_model_calibration", "misspecification_stress_test"):
            for label, v in g[law]["state_recovery"].items():
                lines.append(f"| {axis} / {law} / {label} | "+" | ".join(f"{v[t]['mean_coverage95']:.3f}" for t in TARGETS)+" |")
    lines += ["", "完整误差、遮挡覆盖、边界质量、区间宽度、rank 和参考曲线见 summary.json 与 case_*.json；原始合成数组见 case_*.npz。",
              "reference 未通过精度检查的案例只能记为未确定；通过也只支持相应短案例，不外推到长序列或完整 SBC。",
              "未运行 measured/protected、tokenizer、60-repeat calibration、Step5A1/5B 或 U3 联合参数拟合。", ""]
    return "\n".join(lines)


def run(cfg, run_dir, workers=3):
    validate_config(cfg)
    run_dir = Path(run_dir).resolve()
    root = (REPO_ROOT/cfg["output_root"]).resolve()
    if run_dir.parent != root:
        raise ValueError("run must be a fresh direct child of the configured artifact root")
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    (run_dir/"resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    sources = [Path(__file__), REPO_ROOT/"src/inference/t3a_balloon_robust_ssm.py", REPO_ROOT/"experiments/evaluate_t3c_composite_synthetic_t2.py"]
    manifest = dict(schema=cfg["schema"], execution="running", scope=cfg["scope"],
                    started_at=datetime.now(timezone.utc).isoformat(),
                    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,text=True).strip(),
                    source_sha256={str(p.relative_to(REPO_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    config_sha256=hashlib.sha256((run_dir/"resolved_config.yaml").read_bytes()).hexdigest(),
                    resolved_reference_model=asdict(model(cfg, "G", 0.)[0]),
                    resolved_filter_numerics=asdict(model(cfg, "G", 0.)[1]),
                    python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                    workers=workers, expected_cases=len(cfg["axes"])*cfg["replicates_per_axis"])
    write_json(run_dir/"manifest.json", manifest)
    rows = []
    try:
        checks = dict(parameterization=parameterization_check(cfg), linear_gaussian=linear_gaussian_check(cfg))
        write_json(run_dir/"implementation_checks.json", checks)
        if not all(v["passed"] for v in checks.values()):
            raise RuntimeError("implementation checks failed; do not continue localization")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = {pool.submit(run_case,cfg,a,r):(a,r) for a in cfg["axes"] for r in range(cfg["replicates_per_axis"])}
            for job in as_completed(jobs):
                row, arrays = job.result()
                stem = f"case_{row['axis']}_{row['replicate']:02d}"
                write_json(run_dir/f"{stem}.json", row)
                np.savez_compressed(run_dir/f"{stem}.npz", **arrays)
                rows.append(row)
                print(f"completed {stem} ({len(rows)}/{len(jobs)})", flush=True)
        rows.sort(key=lambda r:(r["axis"],r["replicate"]))
        summary = summarize(cfg, rows, checks)
        write_json(run_dir/"summary.json", summary)
        (run_dir/"summary.md").write_text(report_markdown(summary))
        manifest.update(execution="completed", completed_cases=len(rows), elapsed_seconds=time.time()-started)
    except BaseException as exc:
        manifest.update(execution="failed", completed_cases=len(rows), error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.time()-started)
        raise
    finally:
        write_json(run_dir/"manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.check_only:
        checks = dict(parameterization=parameterization_check(cfg), linear_gaussian=linear_gaussian_check(cfg))
        print(json.dumps(checks, indent=2))
        return 0 if all(v["passed"] for v in checks.values()) else 1
    if args.run_dir is None or args.workers < 1:
        parser.error("--run-dir and positive --workers required")
    run(cfg, args.run_dir, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
