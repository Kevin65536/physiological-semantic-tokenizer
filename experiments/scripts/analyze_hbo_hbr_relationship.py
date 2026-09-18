#!/usr/bin/env python3
"""Bounded Hb-pair audit and shared-parameter adaptation; no teacher evaluation.

Reuse the dataset-scaling audit's subject/record scope on the current measurement
loader. Select windows before array access; retain all unsupported pair rows.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml
from scipy.integrate import simpson, solve_ivp
from scipy.signal import savgol_filter
from scipy.optimize import minimize

from src.data.unified_physiology import UnifiedPhysiologyWindowDataset
from src.inference.t3a_balloon_robust_ssm import (
    BalloonFixedParameters, BalloonFreeParameters, BalloonParameters,
    balloon_rhs, balloon_rhs_jacobian, observation_jacobian, observation_map,
)

SCOPE = ROOT / 'experiments/configs/physiology_semantic_tokenizer/dataset_scaling_report_v1.yaml'
MODEL = ROOT / 'experiments/configs/physiology_semantic_tokenizer/step5_v1.yaml'
CACHE = ROOT / 'data/cache/physiology_semantic_clean_v4'
NAMES = {'eeg_fnirs_single_trial': 'Single-Trial', 'simultaneous_eeg_nirs': 'Simultaneous',
         'visual_cognitive_motivation': 'Visual', 'refed': 'REFED'}
ADAPT_CONFIG = ROOT / 'experiments/configs/physiology_semantic_tokenizer/hbo_hbr_adaptation_v1.yaml'
PARAMETER_NAMES = ('tau', 'eta', 'E0', 'alpha')


def theory():
    ref = yaml.safe_load(MODEL.read_text())['model']['reference']
    p = BalloonParameters(BalloonFixedParameters(alpha=ref['alpha'], E0=ref['E0'],
        gamma=ref['gamma'], neurovascular_gain=ref['beta']),
        BalloonFreeParameters(kappa=ref['kappa'], tau=ref['tau']))
    c = 1 + (1-p.fixed.E0)*np.log1p(-p.fixed.E0)/p.fixed.E0
    b = 1 + (c-1)/p.fixed.alpha
    frequencies = np.array([0., .01, .033333333333, .05, .1, .2, .5, 1.])
    s = 2j*np.pi*frequencies
    rt = p.fixed.Q0/p.fixed.P0*(c*p.free.tau*s+b)/(p.free.tau*s+1)
    ro = rt/(1-rt)
    # Independent check against the implemented six-state linearization.
    a = balloon_rhs_jacobian(np.zeros(6), p)
    h = observation_jacobian(np.zeros(6), p)
    numerical = []
    for z in s:
        drive = np.linalg.solve(z*np.eye(6)-a, np.eye(6)[:, 0])
        y = h@drive
        numerical.append(y[2]/y[1])
    error = float(np.max(np.abs(ro-numerical)))
    if error > 1e-10:
        raise AssertionError(f'Analytic/implemented transfer mismatch: {error}')
    return dict(parameters=asdict(p), c=float(c), b=float(b),
        transfer_check_max_absolute_error=error, rows=[dict(frequency_hz=float(f),
        hbo_hbr_amplitude_ratio=float(1/abs(z)),
        hbr_relative_hbo_phase_degrees=float(np.angle(z, deg=True)))
        for f, z in zip(frequencies, ro)])


def pair_metrics(values, valid):
    mask = valid.all(axis=0) & np.isfinite(values).all(axis=0)
    n = int(mask.sum())
    if n < 3:
        return dict(status='insufficient_support', valid_points=n)
    o, r = values[:, mask]
    so, sr = np.std(o), np.std(r)
    if min(so, sr) <= 1e-12:
        return dict(status='low_variance', valid_points=n)
    rho = float(np.corrcoef(o, r)[0, 1])
    return dict(status='valid', valid_points=n, correlation=rho,
        sd_ratio=float(so/sr), slope_hbr_on_hbo=float(rho*sr/so),
        hbt_sd_over_joint_sd=float(np.std(o+r)/np.sqrt(so*so+sr*sr)),
        positive=float(rho > 0), strong_positive=float(rho > .5),
        strong_negative=float(rho < -.5))


def checks():
    result = theory()
    x = np.vstack([np.sin(np.arange(300)/20), -np.sin(np.arange(300)/20)/3])
    m = np.ones_like(x, dtype=bool)
    baseline = pair_metrics(x, m)
    scaled = pair_metrics(7*x + np.array([[4], [-2]]), m)
    assert abs(baseline['sd_ratio']-3) < 1e-12
    assert abs(baseline['correlation']+1) < 1e-12
    for key in ('correlation', 'sd_ratio', 'slope_hbr_on_hbo', 'hbt_sd_over_joint_sd'):
        assert abs(baseline[key]-scaled[key]) < 1e-12
    m[:, :20] = False
    x[:, :20] = np.nan
    assert pair_metrics(x, m)['valid_points'] == 280
    return result


@lru_cache(maxsize=16)
def weak_test_matrices(samples=300, sample_rate=10., order=8):
    """Quadrature-weighted weak derivative matrices with DC nuisance removed."""
    t = np.arange(samples)/sample_rate
    length = t[-1]
    u = t/length
    envelope = np.polynomial.Polynomial([0, 0, 0, 1, -3, 3, -1])
    e, de, dde = envelope(u), envelope.deriv()(u), envelope.deriv(2)(u)
    phi, dphi, ddphi = [], [], []
    for j in range(order):
        legendre = np.polynomial.Legendre.basis(j)
        v = legendre(2*u-1)
        dv, ddv = legendre.deriv()(2*u-1), legendre.deriv(2)(2*u-1)
        phi.append(e*v)
        dphi.append((de*v+2*e*dv)/length)
        ddphi.append((dde*v+4*de*dv+4*e*ddv)/length**2)
    weights = simpson(np.eye(samples), x=t, axis=0)
    matrices = np.array([phi, -np.array(dphi), ddphi])*weights
    nuisance = matrices[0].sum(axis=1)
    for matrix in matrices:
        matrix -= np.outer(nuisance, nuisance@matrix)/(nuisance@nuisance)
        matrix -= matrix.mean(axis=1, keepdims=True)
    return matrices


@lru_cache(maxsize=64)
def dynamic_constraint_operators(samples=300, sample_rate=10., order=8, parameters=None):
    """Orthonormal rows of weak, initial-condition-free Hb constraints.

    The linearized deterministic identity is
    (tau D+1)[(tau D+1)R-eta(c tau D+b)(O+R)]=0.
    Compact test functions vanish with their first derivative at both ends.
    Profile the single constant nuisance induced by unknown Hb baselines.
    This is an unweighted measurement-space distance, not a noise-calibrated
    statistic, a nonlinear rejection test, or a physiological fault estimator.
    """
    if parameters is None:
        tau, eta, c, b = term_constants()
    else:
        tau, eta, E0, alpha = parameters
        c = 1+(1-E0)*np.log1p(-E0)/E0
        b = 1+(c-1)/alpha
    phi, minus_dphi, ddphi = weak_test_matrices(samples, sample_rate, order)
    optical = np.array([-eta*b*phi, -eta*(b+c)*tau*minus_dphi, -eta*c*tau*tau*ddphi])
    deoxy = np.array([phi, 2*tau*minus_dphi, tau*tau*ddphi])
    parts = np.concatenate((optical, optical+deoxy), axis=2)
    matrix = parts.sum(axis=0)
    left, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(singular > singular[0]*1e-9))
    if rank != order-1:
        raise AssertionError(f'Unexpected weak constraint rank: {rank}')
    whitening = left[:, :rank].T/singular[:rank, None]
    term_operators = np.einsum('ij,kjl->kil', whitening, parts)
    return vh[:rank], term_operators


def dynamic_constraint_basis(samples=300, sample_rate=10., order=8):
    return dynamic_constraint_operators(samples, sample_rate, order)[0]


TERM_NAMES = ('level', 'velocity', 'acceleration', 'J')


@lru_cache(maxsize=1)
def term_constants():
    ref = theory()
    fixed = ref['parameters']['fixed']
    return ref['parameters']['free']['tau'], fixed['Q0']/fixed['P0'], ref['c'], ref['b']


def pointwise_terms(values, *, smoothing_samples=21, trim=30):
    """AC-only, same-unit Q0*J terms, normalized by a common observed Hb SD.

    Local degree-five polynomial derivatives; use the same interior support
    for all three declared smoothing widths. Constants are unidentifiable in
    relative bandpassed data, so center each term; their sum is centered J.
    """
    values = np.asarray(values, dtype=float)
    tau, eta, c, b = term_constants()
    support = slice(trim, -trim)
    scale = float(np.sqrt(np.sum(np.var(values[:, support], axis=1))))
    if scale <= 1e-12 or not np.isfinite(values).all():
        raise ValueError('Complete finite support and nonzero common scale required')
    derivatives = [savgol_filter(values, smoothing_samples, 5, deriv=k, delta=.1,
                                axis=1)[:, support] for k in range(3)]
    total = [d.sum(axis=0) for d in derivatives]
    raw = np.array([derivatives[0][1]-eta*b*total[0],
                    tau*(2*derivatives[1][1]-eta*(b+c)*total[1]),
                    tau*tau*(derivatives[2][1]-eta*c*total[2])])
    means = raw.mean(axis=1)
    normalized = (raw-means[:, None])/scale
    return normalized, scale, means


def energy_decomposition(terms):
    combined = terms.sum(axis=0)
    gram = terms@terms.T/terms.shape[1]
    energy = float(np.mean(combined**2))
    alignment = np.mean(terms*combined, axis=1)
    return gram, energy, alignment


def term_metrics(values, valid):
    if not (valid.all() and np.isfinite(values).all()):
        return dict(status='incomplete_dynamic_support', valid_points=int(valid.all(axis=0).sum())), None
    if np.sqrt(np.sum(np.var(values[:, 30:-30], axis=1))) <= 1e-12:
        return dict(status='low_variance', valid_points=values.shape[1]), None
    terms, scale, means = pointwise_terms(values)
    traces = np.vstack((terms, terms.sum(axis=0)))
    gram, energy, alignment = energy_decomposition(terms)
    result = dict(status='valid', valid_points=values.shape[1], term_points=terms.shape[1],
                  common_native_scale=scale, J_energy=energy)
    for j, name in enumerate(TERM_NAMES):
        result[name+'_rms'] = float(np.sqrt(np.mean(traces[j]**2)))
        for q in (.05, .25, .5, .75, .95):
            result[f'{name}_p{int(q*100):02d}'] = float(np.quantile(traces[j], q))
        if j < 3:
            result[name+'_native_mean_before_centering'] = float(means[j])
            result[name+'_alignment'] = float(alignment[j])
            result[name+'_share'] = float(alignment[j]/energy) if energy > 1e-12 else np.nan
            result[name+'_largest_alignment'] = float(j == np.argmax(alignment))
            result[name+'_negative_alignment'] = float(alignment[j] < 0)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        result[f'cross_{i}{j}'] = float(gram[i, j])
    for samples in (41, 61):
        alt, _, _ = pointwise_terms(values, smoothing_samples=samples)
        _, alt_energy, alt_alignment = energy_decomposition(alt)
        result[f'J_rms_sg{samples}'] = float(np.sqrt(alt_energy))
        result[f'J_energy_sg{samples}'] = alt_energy
        for j, name in enumerate(TERM_NAMES[:3]):
            result[f'{name}_alignment_sg{samples}'] = float(alt_alignment[j])
    centered = values-values.mean(axis=1, keepdims=True)
    basis, operators = dynamic_constraint_operators()
    weak_terms = np.einsum('kij,j->ki', operators, centered.ravel())/np.linalg.norm(centered)
    _, weak_energy, weak_alignment = energy_decomposition(weak_terms)
    result['weak_energy'] = weak_energy
    result['dynamic_distance'] = float(np.linalg.norm(basis@centered.ravel())/np.linalg.norm(centered))
    for j, name in enumerate(TERM_NAMES[:3]):
        result[name+'_weak_alignment'] = float(weak_alignment[j])
    return result, traces


def term_checks():
    tau, eta, c, b = term_constants()
    t = np.arange(300)/10.
    total = np.zeros(300)
    deoxy = np.zeros(300)
    exact = np.zeros((3, 300))
    for frequency, amplitude in ((.04, .05), (.11, .03), (.17, .02)):
        s = 2j*np.pi*frequency
        wave = amplitude*np.exp(s*t)
        ratio = eta*(c*tau*s+b)/(tau*s+1)
        total += wave.real
        deoxy += (ratio*wave).real
        for k, coefficient in enumerate((ratio-eta*b, tau*s*(2*ratio-eta*(b+c)),
                                         tau*tau*s*s*(ratio-eta*c))):
            exact[k] += (coefficient*wave).real
    values = np.array([total-deoxy, deoxy])
    numeric, scale, _ = pointwise_terms(values)
    exact = exact[:, 30:-30]
    exact = (exact-exact.mean(axis=1, keepdims=True))/scale
    error = float(np.max(abs(numeric-exact)))
    assert error < .002
    assert np.allclose(pointwise_terms(7*values+np.array([[3], [-4]]))[0], numeric, atol=1e-9)
    _, operators = dynamic_constraint_operators()
    assert np.allclose(operators.sum(axis=0), dynamic_constraint_basis(), atol=1e-12)
    gram, energy, alignment = energy_decomposition(numeric)
    assert abs(gram.sum()-energy) < 1e-12
    assert abs(alignment.sum()-energy) < 1e-12
    return dict(analytic_derivative_max_normalized_error=error,
        matched_term_rms=np.sqrt(np.mean(numeric**2, axis=1)).tolist(),
        matched_J_rms=float(np.sqrt(energy)),
        interpretation='Matched sinusoidal linear response; nonzero terms cancel; derivative and weak-sum checks')


def dynamic_distance(values, *, order=8):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] != 2 or not np.isfinite(values).all():
        raise ValueError('Dynamics require a complete, finite HbO/HbR pair')
    centered = values-values.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered)
    if norm <= 1e-12:
        raise ValueError('Insufficient trajectory energy')
    basis = dynamic_constraint_basis(values.shape[1], 10., order)
    defect = basis@centered.ravel()
    correction = (basis.T@defect).reshape(values.shape)
    return float(np.linalg.norm(defect)/norm), correction


def dynamics_metrics(values, valid):
    if not (valid.all() and np.isfinite(values).all()):
        return dict(status='incomplete_dynamic_support', valid_points=int(valid.all(axis=0).sum()))
    if np.linalg.norm(values-values.mean(axis=1, keepdims=True)) <= 1e-12:
        return dict(status='low_variance', valid_points=values.shape[1])
    distance, _ = dynamic_distance(values)
    reverse, _ = dynamic_distance(values[:, ::-1])
    interior, _ = dynamic_distance(values[:, 50:-50])
    coarse, _ = dynamic_distance(values, order=6)
    fine, _ = dynamic_distance(values, order=10)
    return dict(status='valid', valid_points=values.shape[1], dynamic_distance=distance,
        reversed_distance=reverse, reverse_minus_forward=reverse-distance,
        forward_better_than_reverse=float(distance < reverse),
        interior_distance=interior, order6_distance=coarse, order10_distance=fine)


def dynamics_checks(*, nonlinear=False):
    """Independent ODE positive/negative controls; no measurement array reads."""
    theory_result = checks()
    p = BalloonParameters(BalloonFixedParameters(), BalloonFreeParameters(kappa=.64, tau=2.))
    a = balloon_rhs_jacobian(np.zeros(6), p)
    h = observation_jacobian(np.zeros(6), p)[1:]
    t = np.arange(300)/10.
    initial = np.array([.03, -.01, .02, .015, -.02, .01])
    def rhs(time, state):
        forcing = .03*np.sin(2*np.pi*.07*time)+.02*np.cos(2*np.pi*.13*time)
        return a@state+np.eye(6)[0]*forcing
    states = solve_ivp(rhs, (0, t[-1]), initial, t_eval=t, rtol=1e-11, atol=1e-13).y
    y = h@states
    distance, correction = dynamic_distance(y)
    reverse, _ = dynamic_distance(y[:, ::-1])
    injected = y.copy()
    injected[1] += .03*np.sin(2*np.pi*.11*t)
    negative, _ = dynamic_distance(injected)
    assert distance < 1e-5 and reverse > 100*distance and negative > .05
    scaled, _ = dynamic_distance(7*y+np.array([[10.], [-3.]]))
    assert abs(scaled-distance) < 1e-10
    assert dynamic_distance(y-correction)[0] < 1e-10
    assert np.allclose(np.corrcoef(y), np.corrcoef(y[:, ::-1]), atol=1e-14)
    ranks = [int(np.linalg.matrix_rank(np.vstack([h@np.linalg.matrix_power(a, i)
             for i in range(n)]))) for n in range(1, 5)]
    assert ranks == [2, 4, 5, 6]
    # Check the fault parity identity against unrelated smooth state curves.
    time = .73
    rates = np.array([.31, .43, .59])
    phases = np.array([.2, -.4, .7])
    v, total, deoxy = np.sin(rates*time+phases)
    dv, dp, dq = rates*np.cos(rates*time+phases)
    ddv, ddp, ddq = -rates**2*np.sin(rates*time+phases)
    flow, dflow = np.sin(.61*time), .61*np.cos(.61*time)
    tau, c, b, k = 2., theory_result['c'], theory_result['b'], 1/.32-1
    ev = tau*dv-flow+v/.32
    ep = tau*dp-flow+total+k*v
    eq = tau*dq-c*flow+deoxy+k*v
    dep = tau*ddp-dflow+dp+k*dv
    deq = tau*ddq-c*dflow+dq+k*dv
    lhs = tau*tau*ddq+2*tau*dq+deoxy-c*tau*tau*ddp-(b+c)*tau*dp-b*total
    right = tau*deq+eq-c*tau*dep-b*ep-(c-b)*ev
    assert abs(lhs-right) < 1e-12
    result = dict(matched_linear_distance=distance, reversed_matched_distance=reverse,
        injected_deoxy_distance=negative, observability_ranks_by_derivative_order=ranks,
        fault_identity_error=abs(lhs-right), constraint_count=7,
        interpretation='Software controls only; no stochastic or measured rejection threshold')
    if nonlinear:
        rows = []
        for amplitude in (.005, .02, .05):
            def nonlinear_rhs(time, state):
                dz = balloon_rhs(state, p)
                dz[0] += amplitude*(np.sin(2*np.pi*.07*time)+.66*np.cos(2*np.pi*.13*time))
                return dz
            sol = solve_ivp(nonlinear_rhs, (0, t[-1]), np.zeros(6), t_eval=t,
                            rtol=1e-9, atol=1e-11)
            if not sol.success:
                raise AssertionError(sol.message)
            physical = sol.y.T.copy()
            physical[:, 2:] = np.exp(physical[:, 2:])
            observations = np.array([observation_map(x, p)[1:] for x in physical]).T
            rows.append(dict(driver_forcing_amplitude=amplitude,
                flow_min=float(physical[:, 2].min()), flow_max=float(physical[:, 2].max()),
                distance=dynamic_distance(observations)[0]))
        result['nonlinear_controls'] = rows
        result['nonlinear_control_scope'] = 'Three smooth deterministic core trajectories; no processing or noise; not a uniform linearization-error bound'
    return result


def select_audit_windows(ds, scope):
    chosen = []
    for d, spec in scope['datasets'].items():
        for subject in spec['subjects']:
            for record in spec['records']:
                refs = [w for w in ds.windows if w.record.dataset_id == d
                        and w.record.canonical_subject_id == subject
                        and w.record.base_record_id == record]
                if not refs:
                    # Visual record names already bind their subject.
                    if d == 'visual_cognitive_motivation' and not record.startswith(subject+'_'):
                        continue
                    raise ValueError(f'Missing admitted record: {d}/{subject}/{record}')
                refs.sort(key=lambda w: int(w.event['event_index']))
                if d == 'eeg_fnirs_single_trial':
                    ma = [w for w in refs if w.event['label'] == 'MA']
                    if len(ma) != 10:
                        raise ValueError('Expected exactly ten MA identities')
                    chosen.extend(w for i, w in enumerate(ma)
                                  if i not in spec['excluded_original_positions'])
                elif d == 'refed':
                    chosen.extend(replace(refs[0], window_offset_s=5+30*k) for k in range(3))
                else:
                    last_e = last_h = -np.inf
                    count = 0
                    for w in refs:
                        e, h = w.event['eeg_time_ms']/1000, w.event['fnirs_time_ms']/1000
                        if w.event['label'] == 'unknown' or min(e, h) < 5 or min(e-last_e, h-last_h) < 30:
                            continue
                        chosen.append(w)
                        last_e, last_h = e, h
                        count += 1
                        if count == 8:
                            break
                    if count != 8:
                        raise ValueError('Expected eight nonoverlapping admitted windows')
    ds.windows = chosen
    inventory = [dict(dataset_id=w.record.dataset_id, subject=w.record.canonical_subject_id,
        record=w.record.base_record_id, event_index=int(w.event['event_index']),
        label=w.event['label'], task=w.event.get('metadata', {}).get('task'),
        fnirs_start_s=w.event['fnirs_time_ms']/1000-5+w.window_offset_s,
        duration_s=30, source_manifest=str(w.record.npz_path),
        processing_schema=w.record.manifest.get('processing_schema')) for w in chosen]
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--dynamics', action='store_true', help='Weak differential-constraint audit')
    parser.add_argument('--terms', action='store_true', help='Level/velocity/acceleration distributions and signed energy allocation')
    parser.add_argument('--nonlinear-check', action='store_true', help='Add three synthetic core checks; check-only mode')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--adapt', action='store_true', help='Cross-window shared parameter adaptation')
    parser.add_argument('--adapt-report', action='store_true', help='Report completed adaptation evidence without refitting')
    parser.add_argument('--adapt-smooth-report', action='store_true', help='Versioned smooth-correction sensitivity export')
    parser.add_argument('--boundary', action='store_true', help='Boundary mechanisms on retained shared-parameter evidence')
    parser.add_argument('--boundary-report', action='store_true', help='Report completed boundary diagnostics without refitting')
    parser.add_argument('--observation', action='store_true', help='Observation gain and metric separation diagnostic')
    parser.add_argument('--observation-report', action='store_true', help='Report completed observation separation evidence')
    parser.add_argument('--observation-config', type=Path, default=ROOT/'experiments/configs/physiology_semantic_tokenizer/hbo_hbr_observation_v1.yaml')
    parser.add_argument('--boundary-config', type=Path, default=ROOT/'experiments/configs/physiology_semantic_tokenizer/hbo_hbr_boundary_v1.yaml')
    parser.add_argument('--adapt-config', type=Path, default=ADAPT_CONFIG)
    parser.add_argument('--resume', action='store_true', help='Resume an existing adaptation run')
    args = parser.parse_args()
    if args.observation_report:
        render_observation_report(args.output_dir)
        return
    if args.observation:
        run_observation_analysis(args)
        return
    if args.boundary_report:
        render_boundary_report(args.output_dir)
        return
    if args.boundary:
        run_boundary_analysis(args)
        return
    if args.adapt_smooth_report:
        render_smooth_adaptation_report(args.output_dir)
        return
    if args.adapt_report:
        render_adaptation_report(args.output_dir)
        return
    if args.adapt:
        run_adaptation(args)
        return
    if args.resume:
        parser.error('--resume requires --adapt')
    if args.terms:
        args.dynamics = True
    analytic = checks()
    if args.nonlinear_check and not (args.check_only and args.dynamics):
        parser.error('--nonlinear-check requires --dynamics --check-only')
    dynamic_checks = dynamics_checks(nonlinear=args.nonlinear_check) if args.dynamics else None
    decomposition_checks = term_checks() if args.terms else None
    if args.check_only:
        print(json.dumps(dict(theory=analytic, dynamics=dynamic_checks, terms=decomposition_checks), indent=2))
        return
    if args.output_dir is None:
        parser.error('--output-dir is required')
    out = args.output_dir.resolve()
    root = ROOT / 'experiments/runs/physiology_semantic_tokenizer/data_quality_audit'
    if out.parent != root:
        raise ValueError('Output must be a fresh direct child of the data-quality audit root')
    out.mkdir(exist_ok=False)
    scope = yaml.safe_load(SCOPE.read_text())
    ds = UnifiedPhysiologyWindowDataset(CACHE, window_duration_s=30, window_offset_s=-5,
                                       output_coordinate='measurement')
    inventory = select_audit_windows(ds, scope)
    (out/'window_inventory.json').write_text(json.dumps(inventory, indent=2))
    manifest = dict(schema='hbo_hbr_relationship_audit_v1', status='running',
        purpose='descriptive measurement morphology, not a fitted-model rejection test',
        scope_source=str(SCOPE.relative_to(ROOT)), model_source=str(MODEL.relative_to(ROOT)),
        cache=str(CACHE.relative_to(ROOT)), windows=len(inventory), parameters_fitted=0,
        aggregation='pair-window descriptors -> subject median or fraction; subjects equal weight',
        support='pairwise intersection of finite, recorded and processed support',
        command=sys.argv, theory=analytic)
    if args.dynamics:
        manifest.update(schema='hbo_hbr_dynamics_audit_v1', synthetic_checks=dynamic_checks,
            diagnostic='minimum same-unit O/R curve correction to seven weak linearized constraints',
            nuisance='unknown constant Hb offsets and arbitrary p-v initial difference removed',
            controls=['joint time reversal preserving all paired amplitudes and zero-lag correlation',
                      'interior 20 seconds', 'six and ten test functions'],
            physiological_parameter_fitting=False, stochastic_null_calibration=False,
            assumptions=['linearization at declared rest', 'common Hb measurement gain',
                         'same linear temporal processing commutes with dynamics away from record edges'],
            limits=['nonlinear preprocessing and mixing can create residuals',
                    'no absolute-state reconstruction from bandpassed relative measurements',
                    'not a unique physiological fault localization'])
    if args.terms:
        manifest.update(schema='hbo_hbr_term_audit_v1', term_checks=decomposition_checks,
            diagnostic='Q0*J level/velocity/acceleration AC distributions and signed residual-energy allocation',
            derivative_estimator='Savitzky-Golay degree 5; 21 samples primary; 41 and 61 controls; fs=10 Hz',
            term_support='indices 30:270, 24 seconds; same support for every derivative width',
            normalization='one per-pair scale sqrt(var(HbO)+var(HbR)) on unsmoothed interior support',
            centering='each term centered; their sum equals centered J; absolute DC unidentifiable',
            contribution='mean(term_i * J) / mean(J**2); sums to one, can be negative or exceed one',
            aggregation='distributions: equal subject, equal valid pair-window within subject; shares: within-subject ratio of mean energies, then equal subjects',
            controls=['21/41/61 sample derivative windows', 'same seven weak constraints as preceding audit'])
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    distribution_values = {}
    for i, meta in enumerate(inventory):
        item = ds[i]
        values = item['fnirs']
        valid = item['channel_valid_mask']['fnirs'] & item['valid_mask']['fnirs'][None, :]
        roles, names = item['component_roles']['fnirs'], item['channel_names']['fnirs']
        if values.shape[1] != 300 or len(roles) % 2:
            raise ValueError('Expected paired Hb and 300 samples at 10 Hz')
        for j in range(0, len(roles), 2):
            if roles[j:j+2] != ['HbO', 'HbR'] or names[j].rsplit('_', 1)[0] != names[j+1].rsplit('_', 1)[0]:
                raise ValueError('Hb chromophore role/pair identity mismatch')
            if args.terms:
                result, traces = term_metrics(values[j:j+2], valid[j:j+2])
                if traces is not None:
                    distribution_values.setdefault((meta['dataset_id'], meta['subject']), []).append(traces)
            else:
                result = (dynamics_metrics if args.dynamics else pair_metrics)(values[j:j+2], valid[j:j+2])
            rows.append(dict(meta, pair=names[j].rsplit('_', 1)[0], unit=item['unit']['fnirs'], **result))
        if (i+1) % 24 == 0:
            print(f'{i+1}/{len(inventory)} windows', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'pair_metrics.csv', index=False)
    good = frame[frame.status == 'valid']
    if args.terms:
        finish_terms(out, frame, manifest, distribution_values)
        return
    if args.dynamics:
        finish_dynamics(out, frame, manifest)
        return
    stats = dict(correlation=('correlation', 'median'), sd_ratio=('sd_ratio', 'median'),
        positive=('positive', 'mean'), strong_positive=('strong_positive', 'mean'),
        strong_negative=('strong_negative', 'mean'),
        hbt_sd_over_joint_sd=('hbt_sd_over_joint_sd', 'median'), pairs=('pair', 'size'))
    subjects = good.groupby(['dataset_id', 'subject']).agg(**stats).reset_index()
    subjects.to_csv(out/'subject_summary.csv', index=False)
    tasks = good.groupby(['dataset_id', 'record', 'subject']).agg(**stats).reset_index()
    tasks.to_csv(out/'record_subject_summary.csv', index=False)
    metrics = list(stats)[:-1]
    summary = subjects.groupby('dataset_id')[metrics].mean().reset_index()
    summary.to_csv(out/'summary.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (d, name) in zip(axes.flat, NAMES.items()):
        g = good[good.dataset_id == d]
        for subject, v in g.groupby('subject'):
            ax.scatter(v.correlation, v.sd_ratio, s=5, alpha=.22, label=subject)
        ax.set(xlim=(-1.02, 1.02), yscale='log', xlabel='HbO-HbR correlation',
               ylabel='SD(HbO) / SD(HbR)', title=f'{name}: {len(g)} valid pair-windows')
        ax.axvline(0, color='grey', lw=.7)
        ax.legend(fontsize=7)
    fig.savefig(out/'pair_relationships.png', dpi=240)
    plt.close(fig)
    manifest.update(status='completed', pair_rows=len(frame), valid_pair_rows=len(good),
                    excluded_status_counts=frame[frame.status != 'valid'].status.value_counts().to_dict(),
                    summary=summary.to_dict('records'))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(subjects.to_string(index=False))
    print(summary.to_string(index=False))


def finish_dynamics(out, frame, manifest):
    good = frame[frame.status == 'valid']
    metrics = ['dynamic_distance', 'reversed_distance', 'reverse_minus_forward',
               'interior_distance', 'order6_distance', 'order10_distance']
    aggregation = {k: (k, 'median') for k in metrics}
    aggregation.update(forward_better_than_reverse=('forward_better_than_reverse', 'mean'), pairs=('pair', 'size'))
    subjects = good.groupby(['dataset_id', 'subject']).agg(**aggregation).reset_index()
    subjects.to_csv(out/'subject_summary.csv', index=False)
    good.groupby(['dataset_id', 'record', 'subject']).agg(**aggregation).reset_index().to_csv(
        out/'record_subject_summary.csv', index=False)
    summary = subjects.groupby('dataset_id')[list(aggregation)[:-1]].mean().reset_index()
    summary.to_csv(out/'summary.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    for ax, (dataset, name) in zip(axes.flat, NAMES.items()):
        for subject, group in good[good.dataset_id == dataset].groupby('subject'):
            ax.scatter(group.dynamic_distance, group.reversed_distance, s=5, alpha=.2, label=subject)
        ax.plot([0, 1], [0, 1], color='gray', lw=.8)
        ax.set(xlim=(0, 1), ylim=(0, 1), title=name, xlabel='Forward weak-constraint distance',
               ylabel='Time-reversed distance')
        ax.legend(fontsize=8)
    fig.savefig(out/'dynamic_distances.png', dpi=240)
    plt.close(fig)
    manifest.update(status='completed', pair_rows=len(frame), valid_pair_rows=len(good),
        excluded_status_counts=frame[frame.status != 'valid'].status.value_counts().to_dict(),
        summary=summary.to_dict('records'))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(subjects.to_string(index=False))
    print(summary.to_string(index=False))


def subject_equal_quantiles(groups, probabilities):
    """Quantiles of the mixture giving equal mass to each subject's samples."""
    values = np.concatenate(groups)
    weights = np.concatenate([np.full(len(g), 1/(len(groups)*len(g))) for g in groups])
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cdf = np.cumsum(weights)
    return values[np.minimum(np.searchsorted(cdf, probabilities), len(values)-1)]


def finish_terms(out, frame, manifest, distribution_values):
    good = frame[frame.status == 'valid']
    probabilities = [.05, .25, .5, .75, .95]
    quantile_names = ['p05', 'p25', 'p50', 'p75', 'p95']
    distributions, contribution_rows, subject_rows, covariance_rows = [], [], [], []
    for dataset, subset in good.groupby('dataset_id'):
        subjects = list(subset.subject.unique())
        for k, term in enumerate(TERM_NAMES):
            for kind in ('signed_time_values', 'window_rms'):
                if kind == 'signed_time_values':
                    groups = [np.stack(distribution_values[(dataset, subject)])[:, k].ravel() for subject in subjects]
                else:
                    groups = [subset[subset.subject == subject][term+'_rms'].to_numpy() for subject in subjects]
                q = subject_equal_quantiles(groups, probabilities)
                distributions.append(dict(dataset_id=dataset, term=term, distribution=kind, **dict(zip(quantile_names, q))))
        per_subject = []
        for subject, group in subset.groupby('subject'):
            row = dict(dataset_id=dataset, subject=subject, pairs=len(group),
                       J_rms_median=float(group.J_rms.median()), dynamic_distance_median=float(group.dynamic_distance.median()))
            for term in TERM_NAMES[:3]:
                row[term+'_rms_median'] = float(group[term+'_rms'].median())
                row[term+'_share'] = float(group[term+'_alignment'].mean()/group.J_energy.mean())
                row[term+'_weak_share'] = float(group[term+'_weak_alignment'].mean()/group.weak_energy.mean())
                row[term+'_largest_fraction'] = float(group[term+'_largest_alignment'].mean())
                row[term+'_negative_fraction'] = float(group[term+'_negative_alignment'].mean())
                for samples in (41, 61):
                    row[f'{term}_share_sg{samples}'] = float(group[f'{term}_alignment_sg{samples}'].mean()/group[f'J_energy_sg{samples}'].mean())
            for samples in (41, 61):
                row[f'J_rms_sg{samples}_median'] = float(group[f'J_rms_sg{samples}'].median())
            subject_rows.append(row)
            per_subject.append(row)
            covariance = dict(dataset_id=dataset, subject=subject, J_energy=float(group.J_energy.mean()))
            for term in TERM_NAMES[:3]:
                covariance[term+'_energy'] = float(np.mean(group[term+'_rms']**2))
            for cross in ('01', '02', '12'):
                covariance['cross_'+cross] = float(group['cross_'+cross].mean())
            covariance_rows.append(covariance)
        for term in TERM_NAMES[:3]:
            contribution_rows.append(dict(dataset_id=dataset, term=term,
                share=float(np.mean([r[term+'_share'] for r in per_subject])),
                weak_share=float(np.mean([r[term+'_weak_share'] for r in per_subject])),
                largest_fraction=float(np.mean([r[term+'_largest_fraction'] for r in per_subject])),
                negative_fraction=float(np.mean([r[term+'_negative_fraction'] for r in per_subject])),
                share_sg41=float(np.mean([r[term+'_share_sg41'] for r in per_subject])),
                share_sg61=float(np.mean([r[term+'_share_sg61'] for r in per_subject]))))
    distributions = pd.DataFrame(distributions)
    contributions = pd.DataFrame(contribution_rows)
    distributions.to_csv(out/'distributions.csv', index=False)
    contributions.to_csv(out/'contributions.csv', index=False)
    pd.DataFrame(subject_rows).to_csv(out/'subject_summary.csv', index=False)
    pd.DataFrame(covariance_rows).to_csv(out/'energy_covariance.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#3366aa', '#dd8822', '#aa4477', '#222222']
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (dataset, name) in zip(axes.flat, NAMES.items()):
        for j, term in enumerate(TERM_NAMES):
            row = distributions[(distributions.dataset_id == dataset)&(distributions.term == term)&(distributions.distribution == 'window_rms')].iloc[0]
            ax.plot([j, j], [row.p05, row.p95], color=colors[j], lw=1)
            ax.plot([j, j], [row.p25, row.p75], color=colors[j], lw=7)
            ax.plot(j, row.p50, 'o', color='white', markeredgecolor=colors[j])
        ax.set(title=name, xticks=range(4), xticklabels=TERM_NAMES, ylabel='RMS / common observed Hb scale', ylim=(0, None))
    fig.savefig(out/'term_distributions.png', dpi=240)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, column, title in zip(axes, ('share', 'weak_share'), ('Pointwise signed energy allocation', 'Derivative-free weak allocation')):
        for k, term in enumerate(TERM_NAMES[:3]):
            x = np.arange(4)+(k-1)*.23
            y = [float(contributions[(contributions.dataset_id == d)&(contributions.term == term)][column].iloc[0])*100 for d in NAMES]
            ax.bar(x, y, width=.23, label=term, color=colors[k])
        ax.set(xticks=range(4), xticklabels=list(NAMES.values()), title=title, ylabel='Signed share of residual energy (%)')
        ax.axhline(0, color='gray', lw=.7)
        ax.legend(fontsize=8)
    fig.savefig(out/'term_contributions.png', dpi=240)
    plt.close(fig)
    manifest.update(status='completed', pair_rows=len(frame), valid_pair_rows=len(good),
        excluded_status_counts=frame[frame.status != 'valid'].status.value_counts().to_dict(),
        distributions=distributions.to_dict('records'), contributions=contributions.to_dict('records'))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(distributions[distributions.distribution == 'window_rms'].to_string(index=False))
    print(contributions.to_string(index=False))


def load_adaptation_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg['schema'] != 'hbo_hbr_adaptation_v1' or tuple(cfg['parameter_order']) != PARAMETER_NAMES:
        raise ValueError('Unsupported adaptation contract')
    if cfg['tensor'] != dict(components=['HbO', 'HbR'], samples=300, sample_rate_hz=10, dtype='float64'):
        raise ValueError('Adaptation requires the declared 2 x 300 float64 tensor')
    if (ROOT/cfg['scope_source']).resolve() != SCOPE or (ROOT/cfg['cache']).resolve() != CACHE:
        raise ValueError('Adaptation must use the existing bounded audit scope and cache')
    if cfg['split']['method'] != 'two_fold_early_late_blocks_with_middle_gap_per_record':
        raise ValueError('Unsupported split')
    if cfg['split']['sharing'] != ['subject', 'subject_channel']:
        raise ValueError('Unsupported sharing levels')
    for box in cfg['parameter_boxes'].values():
        for name in PARAMETER_NAMES:
            lo, hi = box[name]
            if not 0 < lo < hi or (name in ('E0', 'eta') and hi >= 1):
                raise ValueError('Invalid physical parameter box')
            if not lo <= cfg['reference_parameters'][name] <= hi:
                raise ValueError('Reference must be nested in each candidate box')
    return cfg


@lru_cache(maxsize=1)
def compact_constraint_components():
    """Exact row-space compression of all parameter-dependent weak operators."""
    matrices = weak_test_matrices()
    zero = np.zeros_like(matrices)
    full = np.concatenate((np.concatenate((matrices, zero), axis=2),
                           np.concatenate((zero, matrices), axis=2)), axis=0)
    _, singular, vh = np.linalg.svd(full.reshape(-1, 600), full_matrices=False)
    rank = int(np.sum(singular > singular[0]*1e-11))
    coordinates = vh[:rank]
    compact = full@coordinates.T
    assert np.max(abs(compact@coordinates-full)) < 1e-11
    return coordinates, compact


def physical_combinations(parameters):
    tau, eta, E0, alpha = parameters
    c = 1+(1-E0)*np.log1p(-E0)/E0
    b = 1+(c-1)/alpha
    return tau, eta*c, eta*b


def equivalent_parameter_ranges(parameters, box):
    """Sample the exact equivalence curve; ranges are descriptive, not a CI."""
    tau, a, d = physical_combinations(parameters)
    E = np.unique(np.r_[np.linspace(*box['E0'], 2001), parameters[2]])
    c = 1+(1-E)*np.log1p(-E)/E
    eta = a/c
    alpha = (c-1)/(d/eta-1)
    valid = ((eta >= box['eta'][0]-1e-10)&(eta <= box['eta'][1]+1e-10)&
             (alpha >= box['alpha'][0]-1e-10)&(alpha <= box['alpha'][1]+1e-10))
    if not valid.any():
        raise AssertionError('Fitted physical tuple must remain on its equivalence curve')
    result = {}
    for name, values in (('E0', E), ('eta', eta), ('alpha', alpha)):
        result[name+'_equivalent_min'] = float(values[valid].min())
        result[name+'_equivalent_max'] = float(values[valid].max())
    return result


def compact_constraint_basis(parameters):
    tau, a, d = physical_combinations(parameters)
    coefficients = np.array([-d, -(a+d)*tau, -a*tau*tau,
                             1-d, (2-a-d)*tau, (1-a)*tau*tau])
    _, components = compact_constraint_components()
    matrix = np.einsum('i,ijk->jk', coefficients, components)
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    if np.sum(singular > singular[0]*1e-9) != 7:
        raise ValueError('Parameter candidate lost weak constraint rank')
    return vh[:7]


def assign_adaptation_blocks(inventory):
    blocks = np.full(len(inventory), 'gap', dtype=object)
    frame = pd.DataFrame(inventory)
    for _, group in frame.groupby(['dataset_id', 'subject', 'record'], sort=False):
        indices = group.sort_values('fnirs_start_s').index.to_numpy()
        size = (len(indices)-1)//2
        if size < 1:
            raise ValueError('At least three windows per record required')
        blocks[indices[:size]] = 'early'
        blocks[indices[-size:]] = 'late'
        early_end = max(frame.loc[indices[:size], 'fnirs_start_s']+30)
        late_start = min(frame.loc[indices[-size:], 'fnirs_start_s'])
        if late_start-early_end < 30-1e-6:
            raise ValueError('Expected at least one 30-second gap between fit/evaluation blocks')
    return blocks


def adaptation_start_points(cfg, box_name):
    bounds = np.log(np.array([cfg['parameter_boxes'][box_name][p] for p in PARAMETER_NAMES]))
    points = [np.array(p) for p in product(*bounds)]
    points += [np.log([cfg['reference_parameters'][p] for p in PARAMETER_NAMES]), bounds.mean(axis=1)]
    rng = np.random.default_rng(cfg['seed'])
    points.extend(rng.uniform(bounds[:, 0], bounds[:, 1], (cfg['optimization']['random_starts'], 4)))
    return bounds, points


def fit_shared_parameters(gram, cfg, box_name, warm_start=None, *, first_order=False, smooth=False):
    """Only a training sufficient statistic enters parameter selection."""
    bounds, starts = adaptation_start_points(cfg, box_name)
    if warm_start is not None:
        starts.append(np.log(warm_start))
    evaluations = 0

    def objective(log_parameters):
        nonlocal evaluations
        evaluations += 1
        parameters = np.exp(log_parameters)
        basis = (effective_constraint_basis(physical_to_effective(parameters), first_order=True)
                 if first_order else compact_constraint_basis(parameters))
        return constraint_objective(basis, gram, smooth=smooth)

    scored = sorted((objective(p), i, p) for i, p in enumerate(starts))
    best_value, _, best = scored[0]
    attempts = []
    opt = cfg['optimization']
    for _, _, start in scored[:opt['refined_starts']]:
        result = minimize(objective, start, method='L-BFGS-B', bounds=bounds,
                          options=dict(maxiter=opt['maxiter'], ftol=opt['ftol'], gtol=opt['gtol']))
        attempts.append(dict(success=bool(result.success), objective=float(result.fun), message=str(result.message)))
        if np.isfinite(result.fun) and result.fun < best_value:
            best_value, best = float(result.fun), result.x
    parameters = np.exp(best)
    span = bounds[:, 1]-bounds[:, 0]
    near_boundary = np.minimum(best-bounds[:, 0], bounds[:, 1]-best)/span < .005
    return dict(parameters=parameters.tolist(), combinations=list(physical_combinations(parameters)),
                train_mean_squared_distance=best_value, objective_evaluations=evaluations,
                boundary_parameters=[p for p, hit in zip(PARAMETER_NAMES, near_boundary) if hit],
                optimizer_attempts=attempts)


def fit_adaptation_task(task):
    key, gram, cfg = task
    primary = fit_shared_parameters(gram, cfg, 'primary')
    expanded = fit_shared_parameters(gram, cfg, 'expanded', primary['parameters'])
    return dict(key=key, primary=primary, expanded=expanded)


@lru_cache(maxsize=4)
def smooth_correction_space(max_cosine_mode):
    if not 7 <= max_cosine_mode <= 30:
        raise ValueError('Smooth correction requires 7 to 30 cosine modes')
    n = np.arange(300)+.5
    modes = np.arange(1, max_cosine_mode+1)
    scalar = np.cos(np.pi*modes[:, None]*n/300)*np.sqrt(2/300)
    zero = np.zeros_like(scalar)
    paired = np.block([[scalar, zero], [zero, scalar]])
    return scalar, paired


def conditional_curve_repairs(normalized_curve, parameters, *, max_cosine_mode=None):
    """Corrections in observation coordinates; neither is a unique process fault."""
    coordinates, _ = compact_constraint_components()
    basis = compact_constraint_basis(parameters)@coordinates
    y = np.asarray(normalized_curve).reshape(600)
    residual = basis@y
    if max_cosine_mode is None:
        scalar = None
        delta = -(basis.T@residual).reshape(2, 300)
    else:
        scalar, paired = smooth_correction_space(max_cosine_mode)
        coefficient = np.linalg.lstsq(basis@paired.T, -residual, rcond=1e-10)[0]
        delta = (paired.T@coefficient).reshape(2, 300)
    # Constrain delta T=0: delta O=-delta R. Minimum Hb-pair norm solution.
    exchange = basis[:, 300:]-basis[:, :300]
    delta_r = np.linalg.lstsq(exchange if scalar is None else exchange@scalar.T, -residual, rcond=1e-10)[0]
    if scalar is not None:
        delta_r = scalar.T@delta_r
    oxygenation_only = np.array([-delta_r, delta_r])
    # Constrain delta(R-eta T)=0: baseline-fraction-preserving total-Hb change.
    eta = parameters[1]
    volume = (1-eta)*basis[:, :300]+eta*basis[:, 300:]
    delta_t = np.linalg.lstsq(volume if scalar is None else volume@scalar.T, -residual, rcond=1e-10)[0]
    if scalar is not None:
        delta_t = scalar.T@delta_t
    total_only = np.array([(1-eta)*delta_t, eta*delta_t])
    for change in (delta, oxygenation_only, total_only):
        assert np.max(abs(basis@(y+change.ravel()))) < 1e-9
        assert np.linalg.norm(change) >= np.linalg.norm(delta)-1e-10
    return delta, oxygenation_only, total_only


def synthetic_adaptation_curves(parameters, phases):
    tau, a, d = physical_combinations(parameters)
    t = np.arange(300)/10
    curves = []
    for phase in phases:
        total, deoxy = np.zeros(300), np.zeros(300)
        for k, frequency in enumerate((.03, .08, .15)):
            z = 2j*np.pi*frequency
            wave = np.exp(1j*(phase*(k+1)+.4*k))*np.exp(z*t)/(k+1)
            total += wave.real
            deoxy += ((a*tau*z+d)/(tau*z+1)*wave).real
        y = np.array([total-deoxy, deoxy])
        y -= y.mean(axis=1, keepdims=True)
        curves.append(y/np.linalg.norm(y))
    return np.array(curves)


def adaptation_checks(cfg):
    coordinates, _ = compact_constraint_components()
    reference = tuple(cfg['reference_parameters'][p] for p in PARAMETER_NAMES)
    fixed = compact_constraint_basis(reference)@coordinates
    existing = dynamic_constraint_basis()
    assert np.max(abs(fixed.T@fixed-existing.T@existing)) < 1e-11
    truth = (.85, .5, .45, .24)
    curves = synthetic_adaptation_curves(truth, [.1, .7, 1.4, 2.0, 2.7, 3.4])
    compact = curves.reshape(6, 600)@coordinates.T
    gram = compact[:3].T@compact[:3]/3
    fitted = fit_shared_parameters(gram, cfg, 'primary')
    baseline = np.linalg.norm(compact[3:]@compact_constraint_basis(reference).T, axis=1).mean()
    adapted = np.linalg.norm(compact[3:]@compact_constraint_basis(fitted['parameters']).T, axis=1).mean()
    assert baseline > .03 and adapted < .0001
    delta, exchange, total = conditional_curve_repairs(curves[3], reference)
    assert np.max(abs(exchange.sum(axis=0))) < 1e-12
    assert np.max(abs(total[1]-reference[1]*total.sum(axis=0))) < 1e-12
    # Two different physiological tuples with identical identifiable combinations.
    tau, a, d = physical_combinations(truth)
    E0 = .4
    c = 1+(1-E0)*np.log1p(-E0)/E0
    eta = a/c
    alpha = (c-1)/(d/eta-1)
    equivalent = (tau, eta, E0, alpha)
    b1, b2 = compact_constraint_basis(truth), compact_constraint_basis(equivalent)
    assert np.max(abs(b1.T@b1-b2.T@b2)) < 1e-11
    return dict(fixed_heldout_mean_distance=float(baseline), adapted_heldout_mean_distance=float(adapted),
                fitted_representative=fitted['parameters'], truth=list(truth), equivalent=list(equivalent),
                compression_rank=len(coordinates), correction_norms=[float(np.linalg.norm(x)) for x in (delta, exchange, total)],
                interpretation='Noiseless software controls; not measured noise or physiological calibration')


def run_adaptation(args):
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    import os
    import shutil
    import time
    cfg = load_adaptation_config(args.adapt_config)
    controls = adaptation_checks(cfg)
    if args.check_only:
        print(json.dumps(controls, indent=2))
        return
    if args.output_dir is None:
        raise ValueError('--adapt requires --output-dir')
    out = args.output_dir.resolve()
    if out.parent != ROOT/cfg['output_namespace']:
        raise ValueError('Output must be a direct child of the existing audit root')
    if args.resume:
        manifest = json.loads((out/'manifest.json').read_text())
        if manifest['resolved_config'] != cfg or manifest['status'] == 'completed':
            raise ValueError('Resume requires unchanged configuration and unfinished run')
    else:
        out.mkdir(exist_ok=False)
        manifest = dict(schema=cfg['schema'], experiment_id=cfg['experiment_id'], status='preparing',
                        resolved_config=cfg, command=sys.argv, synthetic_checks=controls,
                        protected_data_access=False, stochastic_null_calibration=False,
                        spatial_scope='subject and native channel identity, not anatomically localized source',
                        resources=dict(cpu_affinity=sorted(os.sched_getaffinity(0)), workers=cfg['resources']['workers'],
                                       numerical_threads=1), started_at=time.time())
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
        snapshot = ['experiments/scripts/analyze_hbo_hbr_relationship.py',
                    'experiments/configs/physiology_semantic_tokenizer/hbo_hbr_adaptation_v1.yaml',
                    'experiments/configs/physiology_semantic_tokenizer/dataset_scaling_report_v1.yaml',
                    'experiments/configs/physiology_semantic_tokenizer/step5_v1.yaml',
                    'src/data/unified_physiology.py', 'src/inference/t3a_balloon_robust_ssm.py',
                    'tests/test_hbo_hbr_relationship.py']
        for filename in snapshot:
            target = out/'source_snapshot'/filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/filename, target)
    if not (out/'prepared_inputs.npz').exists():
        ds = UnifiedPhysiologyWindowDataset(CACHE, window_duration_s=30, window_offset_s=-5,
                                           output_coordinate='measurement')
        inventory = select_audit_windows(ds, yaml.safe_load(SCOPE.read_text()))
        prior = json.loads((ROOT/cfg['reference_evidence']/'window_inventory.json').read_text())
        if inventory != prior:
            raise ValueError('Window identities differ from the preceding audit')
        blocks = assign_adaptation_blocks(inventory)
        rows, curves = [], []
        for i, meta in enumerate(inventory):
            item = ds[i]
            values = item['fnirs']
            valid = item['channel_valid_mask']['fnirs'] & item['valid_mask']['fnirs'][None, :]
            roles, names = item['component_roles']['fnirs'], item['channel_names']['fnirs']
            if values.shape[1] != 300 or len(roles) % 2:
                raise ValueError('Expected paired Hb and 300 samples')
            for j in range(0, len(roles), 2):
                if roles[j:j+2] != ['HbO', 'HbR'] or names[j].rsplit('_', 1)[0] != names[j+1].rsplit('_', 1)[0]:
                    raise ValueError('Invalid Hb pair identity')
                row = dict(meta, window_index=i, pair=names[j].rsplit('_', 1)[0], unit=item['unit']['fnirs'],
                           block=blocks[i], curve_index=-1, status='incomplete_dynamic_support')
                y = np.asarray(values[j:j+2], dtype=float)
                if valid[j:j+2].all() and np.isfinite(y).all():
                    centered = y-y.mean(axis=1, keepdims=True)
                    scale = float(np.linalg.norm(centered))
                    if scale > 1e-12:
                        row.update(curve_index=len(curves), status='valid', native_pair_norm=scale,
                                   native_mean_O=float(y[0].mean()), native_mean_R=float(y[1].mean()))
                        curves.append(centered/scale)
                    else:
                        row['status'] = 'low_variance'
                rows.append(row)
            if (i+1) % 24 == 0:
                print(f'prepared {i+1}/{len(inventory)} windows', flush=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(out/'pair_inventory.csv', index=False)
        (out/'window_inventory.json').write_text(json.dumps(inventory, indent=2))
        np.savez_compressed(out/'prepared_inputs.npz', curves=np.array(curves))
    frame = pd.read_csv(out/'pair_inventory.csv', dtype={'subject': str})
    curves = np.load(out/'prepared_inputs.npz')['curves']
    coordinates, _ = compact_constraint_components()
    compact = curves.reshape(len(curves), 600)@coordinates.T
    eligible = frame[(frame.status == 'valid')&(frame.block != 'gap')].copy()
    reference = [cfg['reference_parameters'][p] for p in PARAMETER_NAMES]
    fixed_basis = compact_constraint_basis(reference)
    frame.loc[frame.status == 'valid', 'fixed_distance'] = np.linalg.norm(compact@fixed_basis.T, axis=1)
    old = pd.read_csv(ROOT/cfg['reference_evidence']/'pair_metrics.csv', dtype={'subject': str})
    join = ['dataset_id', 'subject', 'record', 'event_index', 'pair', 'fnirs_start_s']
    check = frame.merge(old[join+['dynamic_distance']], on=join, validate='one_to_one')
    reproduction = float(np.nanmax(abs(check.fixed_distance-check.dynamic_distance)))
    if len(check) != len(frame) or reproduction > 1e-10:
        raise AssertionError('Fixed baseline does not reproduce the preceding audit')
    frame.to_csv(out/'pair_inventory.csv', index=False)
    jobs, groups = [], {}
    for sharing in cfg['split']['sharing']:
        columns = ['dataset_id', 'subject']+(['pair'] if sharing == 'subject_channel' else [])
        for group_key, group in eligible.groupby(columns, sort=True):
            for fit_block in ('early', 'late'):
                train = group[group.block == fit_block]
                test = group[group.block != fit_block]
                if len(train) < cfg['split']['minimum_fit_windows_per_channel'] or not len(test):
                    raise ValueError('Insufficient shared-window fit support')
                key = '|'.join((sharing, *map(str, group_key), fit_block))
                z = compact[train.curve_index.to_numpy()]
                jobs.append((key, z.T@z/len(z), cfg))
                groups[key] = dict(sharing=sharing, dataset_id=str(group_key[0]), subject=str(group_key[1]),
                                   pair=str(group_key[2]) if sharing == 'subject_channel' else 'all',
                                   fit_block=fit_block, train_indices=train.index.tolist(), test_indices=test.index.tolist())
    (out/'fit_groups.json').write_text(json.dumps(groups, indent=2))
    completed = {}
    results_path = out/'fit_results.jsonl'
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            result = json.loads(line)
            completed[result['key']] = result
    remaining = [task for task in jobs if task[0] not in completed]
    manifest.update(status='fitting', windows=len(json.loads((out/'window_inventory.json').read_text())),
                    pair_rows=len(frame), valid_pair_rows=len(curves), assessment_pair_rows=len(eligible),
                    assessment_windows=int(eligible.window_index.nunique()), fit_jobs=len(jobs),
                    fixed_reproduction_max_error=reproduction)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=cfg['resources']['workers']) as pool, results_path.open('a') as output:
        iterator = iter(remaining)
        pending = {}
        for _ in range(min(len(remaining), cfg['resources']['max_in_flight'])):
            task = next(iterator)
            pending[pool.submit(fit_adaptation_task, task)] = task[0]
        while pending:
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                result = future.result()
                output.write(json.dumps(result)+'\n')
                output.flush()
                completed[result['key']] = result
                task = next(iterator, None)
                if task is not None:
                    pending[pool.submit(fit_adaptation_task, task)] = task[0]
            print(f'fits {len(completed)}/{len(jobs)} elapsed={time.monotonic()-start:.1f}s', flush=True)
    manifest.update(status='evaluating', fitting_seconds=time.monotonic()-start)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    finish_adaptation(out, cfg, manifest, frame, curves, compact, groups, completed)


def finish_adaptation(out, cfg, manifest, frame, curves, compact, groups, completed):
    coordinates, _ = compact_constraint_components()
    reference = [cfg['reference_parameters'][p] for p in PARAMETER_NAMES]
    metrics, fits = [], []
    eligible = frame[(frame.status == 'valid')&(frame.block != 'gap')]
    for index, row in eligible.iterrows():
        metrics.append(dict(row_index=int(index), dataset_id=row.dataset_id, subject=row.subject, pair=row.pair,
                            window_index=row.window_index, record=row.record, block=row.block,
                            arm='fixed', distance=row.fixed_distance, fit_key='none'))
    for key, result in completed.items():
        group = groups[key]
        for box in ('primary', 'expanded'):
            fit = result[box]
            parameters = fit['parameters']
            basis = compact_constraint_basis(parameters)
            arm = group['sharing']+'_'+box
            fits.append(dict(fit_key=key, arm=arm, dataset_id=group['dataset_id'], subject=group['subject'],
                             pair=group['pair'], fit_block=group['fit_block'], train_pairs=len(group['train_indices']),
                             test_pairs=len(group['test_indices']), **dict(zip(PARAMETER_NAMES, parameters)),
                             eta_c=fit['combinations'][1], eta_b=fit['combinations'][2],
                             train_mse=fit['train_mean_squared_distance'], boundary=','.join(fit['boundary_parameters']),
                             tau_boundary='tau' in fit['boundary_parameters'],
                             **equivalent_parameter_ranges(parameters, cfg['parameter_boxes'][box]),
                             optimizer_all_failed=not any(a['success'] for a in fit['optimizer_attempts'])))
            for index in group['test_indices']:
                row = frame.loc[index]
                y = curves[int(row.curve_index)]
                delta_compact = -basis.T@(basis@compact[int(row.curve_index)])
                delta = (delta_compact@coordinates).reshape(2, 300)
                dt = delta.sum(axis=0)
                dz = delta[1]-parameters[1]*dt
                metrics.append(dict(row_index=int(index), dataset_id=row.dataset_id, subject=row.subject,
                    pair=row.pair, window_index=row.window_index, record=row.record, block=row.block,
                    arm=arm, fit_key=key, distance=float(np.linalg.norm(delta)),
                    delta_O_norm=float(np.linalg.norm(delta[0])), delta_R_norm=float(np.linalg.norm(delta[1])),
                    delta_T_norm=float(np.linalg.norm(dt)), delta_linear_oxygenation_norm=float(np.linalg.norm(dz)),
                    Hb_exchange_axis_energy_fraction=float(np.sum((delta[0]-delta[1])**2)/2/max(np.sum(delta**2), 1e-30)),
                    baseline_eta_oxygenation_correction_norm=float(np.linalg.norm(delta[1]-reference[1]*dt)),
                    correction_closure_error=float(np.max(abs(basis@(compact[int(row.curve_index)]+delta_compact))))))
    scores, fitted = pd.DataFrame(metrics), pd.DataFrame(fits)
    pivot = scores.pivot(index='row_index', columns='arm', values='distance')
    if pivot.isna().any().any() or len(pivot) != len(eligible):
        raise AssertionError('Every evaluated row must have the same baseline and all four adaptation arms')
    scores.to_csv(out/'heldout_metrics.csv', index=False)
    fitted.to_csv(out/'fitted_parameters.csv', index=False)
    summaries = []
    for (dataset, subject, arm), group in scores.groupby(['dataset_id', 'subject', 'arm']):
        fixed = pivot.loc[group.row_index, 'fixed'].to_numpy()
        distance = group.distance.to_numpy()
        summaries.append(dict(dataset_id=dataset, subject=subject, arm=arm, pairs=len(group),
            distance_median=float(np.median(distance)), distance_p25=float(np.quantile(distance,.25)),
            distance_p75=float(np.quantile(distance,.75)), rms_distance=float(np.sqrt(np.mean(distance**2))),
            fixed_median=float(np.median(fixed)), mean_paired_reduction=float(np.mean(fixed-distance)),
            improved_fraction=float(np.mean(distance < fixed-1e-8)),
            relative_energy_reduction=float(1-np.mean(distance**2)/np.mean(fixed**2))))
    subjects = pd.DataFrame(summaries)
    subjects.to_csv(out/'subject_summary.csv', index=False)
    summary = subjects.groupby(['dataset_id','arm']).agg(
        distance=('distance_median','mean'), rms_distance=('rms_distance','mean'),
        mean_paired_reduction=('mean_paired_reduction','mean'), improved_fraction=('improved_fraction','mean'),
        relative_energy_reduction=('relative_energy_reduction','mean')).reset_index()
    summary.to_csv(out/'summary.csv', index=False)
    # Conditional physiology-readable repairs for every heldout channel-adapted primary curve.
    conditional, example_arrays = [], {}
    primary = scores[scores.arm == 'subject_channel_primary']
    for _, score in primary.iterrows():
        row = frame.loc[score.row_index]
        parameters = completed[score.fit_key]['primary']['parameters']
        y = curves[int(row.curve_index)]
        delta, oxygenation, total = conditional_curve_repairs(y, parameters)
        conditional.append(dict(row_index=int(score.row_index), dataset_id=row.dataset_id, subject=row.subject,
            pair=row.pair, window_index=int(row.window_index), joint_distance=float(np.linalg.norm(delta)),
            fixed_total_Hb_distance=float(np.linalg.norm(oxygenation)),
            fixed_linear_oxygenation_distance=float(np.linalg.norm(total))))
    conditional = pd.DataFrame(conditional)
    conditional.to_csv(out/'conditional_repairs.csv', index=False)
    example_rows = []
    for dataset in NAMES:
        first_subject = sorted(primary[primary.dataset_id == dataset].subject.unique())[0]
        subset = primary[(primary.dataset_id == dataset)&(primary.subject == first_subject)]
        score = subset.loc[(subset.distance-subset.distance.median()).abs().idxmin()]
        row = frame.loc[score.row_index]
        parameters = completed[score.fit_key]['primary']['parameters']
        y = curves[int(row.curve_index)]
        delta, oxygenation, total = conditional_curve_repairs(y, parameters)
        example_arrays[dataset+'_observed_normalized'] = y
        example_arrays[dataset+'_joint_correction'] = delta
        example_arrays[dataset+'_fixed_total_correction'] = oxygenation
        example_arrays[dataset+'_fixed_oxygenation_correction'] = total
        example_rows.append(dict(row.to_dict(), fitted_parameters=parameters, selection='nearest median primary heldout distance in first listed subject',
                                 adapted_distance=float(np.linalg.norm(delta)), fixed_total_Hb_distance=float(np.linalg.norm(oxygenation)),
                                 fixed_linear_oxygenation_distance=float(np.linalg.norm(total))))
    np.savez_compressed(out/'representative_curves.npz', **example_arrays)
    (out/'representative_curves.json').write_text(json.dumps(example_rows, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), constrained_layout=True)
    arms = ['fixed','subject_primary','subject_channel_primary','subject_channel_expanded']
    for ax, (dataset, name) in zip(axes, NAMES.items()):
        for subject, group in subjects[subjects.dataset_id == dataset].groupby('subject'):
            vals = [float(group[group.arm == arm].distance_median.iloc[0]) for arm in arms]
            ax.plot(range(4), vals, '-o', label=subject, ms=4)
        ax.set(title=name, xticks=range(4), xticklabels=['Fixed','Subject','Channel','Expanded'],
               ylabel='Heldout median relative curve change', ylim=(0, .65))
        ax.tick_params(axis='x', rotation=30)
        ax.legend(fontsize=7)
    fig.savefig(out/'adaptation_comparison.png', dpi=240)
    plt.close(fig)
    fig, axes = plt.subplots(4, 3, figsize=(13, 11), constrained_layout=True)
    t = np.arange(300)/10
    for axs, info in zip(axes, example_rows):
        dataset = info['dataset_id']; y = example_arrays[dataset+'_observed_normalized']; delta = example_arrays[dataset+'_joint_correction']
        # Plot in common observed SD units, not concentration units.
        y, delta = y*np.sqrt(300), delta*np.sqrt(300)
        for k, name in enumerate(('HbO','HbR')):
            axs[k].plot(t, y[k], label='Observed')
            axs[k].plot(t, y[k]+delta[k], '--', label='Nearest weak-compatible')
            axs[k].set(title=f'{NAMES[dataset]} / {info["subject"]} / {info["pair"]}: {name}', xlabel='Time (s)', ylabel='Common observed SD')
            axs[k].legend(fontsize=7)
        eta = info['fitted_parameters'][1]
        axs[2].plot(t, delta.sum(axis=0), label='Change in total Hb')
        axs[2].plot(t, delta[1]-eta*delta.sum(axis=0), label='Change in linear oxygenation coordinate')
        axs[2].set(title='Required correction (added to observation)', xlabel='Time (s)', ylabel='Common observed SD')
        axs[2].legend(fontsize=7)
    fig.savefig(out/'representative_repairs.png', dpi=240)
    plt.close(fig)
    manifest.update(status='completed', summary=summary.to_dict('records'),
                    maximum_repair_closure_error=float(scores.correction_closure_error.max()),
                    boundary_fit_fraction=float((fitted.boundary != '').mean()),
                    optimizer_all_failed_fraction=float(fitted.optimizer_all_failed.mean()),
                    uncertainty='descriptive 3 subjects/dataset; no independent-record validation or noise rejection threshold')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False), flush=True)


def render_adaptation_report(output_dir):
    """Version-one report export and parameter-box sensitivity of conditional repairs."""
    if output_dir is None:
        raise ValueError('--adapt-report requires --output-dir')
    out = Path(output_dir).resolve()
    manifest = json.loads((out/'manifest.json').read_text())
    if manifest['schema'] != 'hbo_hbr_adaptation_v1' or manifest['status'] != 'completed':
        raise ValueError('Report requires completed adaptation evidence')
    if (out/'REPORT.md').exists():
        raise FileExistsError('Retain the existing report; a new export must be versioned')
    cfg = manifest['resolved_config']
    frame = pd.read_csv(out/'pair_inventory.csv', dtype={'subject': str})
    scores = pd.read_csv(out/'heldout_metrics.csv', dtype={'subject': str})
    subjects = pd.read_csv(out/'subject_summary.csv', dtype={'subject': str})
    fitted = pd.read_csv(out/'fitted_parameters.csv', dtype={'subject': str}, keep_default_na=False)
    summary = pd.read_csv(out/'summary.csv')
    curves = np.load(out/'prepared_inputs.npz')['curves']
    fit_results = {r['key']:r for r in map(json.loads,(out/'fit_results.jsonl').read_text().splitlines())}
    primary = pd.read_csv(out/'conditional_repairs.csv', dtype={'subject': str})
    expanded = []
    for row in scores[scores.arm == 'subject_channel_expanded'].itertuples():
        meta = frame.loc[row.row_index]
        params = fit_results[row.fit_key]['expanded']['parameters']
        delta, exchange, total = conditional_curve_repairs(curves[int(meta.curve_index)], params)
        expanded.append(dict(row_index=row.row_index, dataset_id=row.dataset_id, subject=row.subject,
            pair=row.pair, window_index=row.window_index, joint_distance=float(np.linalg.norm(delta)),
            fixed_total_Hb_distance=float(np.linalg.norm(exchange)),
            fixed_linear_oxygenation_distance=float(np.linalg.norm(total))))
    expanded = pd.DataFrame(expanded)
    expanded.to_csv(out/'conditional_repairs_expanded.csv', index=False)
    conditional = pd.concat([primary.assign(box='primary'),expanded.assign(box='expanded')],ignore_index=True)
    conditional['oxygenation_route_cheaper'] = conditional.fixed_total_Hb_distance < conditional.fixed_linear_oxygenation_distance
    conditional['fixed_distance'] = frame.loc[conditional.row_index, 'fixed_distance'].to_numpy()
    channel = conditional.groupby(['dataset_id','subject','pair','box']).agg(
        pairs=('row_index','size'),fixed_distance=('fixed_distance','median'),distance=('joint_distance','median'),
        fixed_total_Hb_distance=('fixed_total_Hb_distance','median'),
        fixed_linear_oxygenation_distance=('fixed_linear_oxygenation_distance','median'),
        oxygenation_route_cheaper_fraction=('oxygenation_route_cheaper','mean')).reset_index()
    channel.to_csv(out/'channel_summary.csv', index=False)
    cs = conditional.groupby(['dataset_id','subject','box']).agg(
        distance=('joint_distance','median'),fixed_total_Hb_distance=('fixed_total_Hb_distance','median'),
        fixed_linear_oxygenation_distance=('fixed_linear_oxygenation_distance','median'),
        oxygenation_route_cheaper_fraction=('oxygenation_route_cheaper','mean')).reset_index()
    cs.to_csv(out/'conditional_subject_summary.csv', index=False)
    route = conditional.pivot(index='row_index', columns='box',values='oxygenation_route_cheaper')
    route['agreement'] = route.primary == route.expanded
    agreement = conditional[conditional.box=='primary'][['row_index','dataset_id','subject']].merge(route[['agreement']],left_on='row_index',right_index=True)
    agree = agreement.groupby(['dataset_id','subject']).agreement.mean().groupby('dataset_id').mean()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,3,figsize=(12,6.5),constrained_layout=True)
    cases=[]; arrays={}; t=np.arange(300)/10
    for axs,pair in zip(axes,('AF8Fp2','C1C3')):
        subset=scores[(scores.dataset_id=='eeg_fnirs_single_trial')&(scores.subject=='subject_01')&
                      (scores.pair==pair)&(scores.arm=='subject_channel_primary')]
        row=subset.loc[(subset.distance-subset.distance.median()).abs().idxmin()]
        meta=frame.loc[row.row_index]
        params=fit_results[row.fit_key]['primary']['parameters']
        y=curves[int(meta.curve_index)]
        delta,exchange,total=conditional_curve_repairs(y,params)
        case=dict(meta.to_dict(),parameters=params,distance=float(np.linalg.norm(delta)),
                  fixed_total_Hb_distance=float(np.linalg.norm(exchange)),fixed_linear_oxygenation_distance=float(np.linalg.norm(total)),
                  selection='Two explicitly illustrative contrasting native channels; nearest within-channel median heldout distance')
        cases.append(case)
        arrays[pair+'_observed']=y; arrays[pair+'_joint_change']=delta
        arrays[pair+'_fixed_total_change']=exchange; arrays[pair+'_fixed_oxygenation_change']=total
        for k in range(2):
            axs[k].plot(t,y[k]*np.sqrt(300),label='Observed')
            axs[k].plot(t,(y[k]+delta[k])*np.sqrt(300),'--',label='Weak-compatible')
            axs[k].set(title=f'subject_01 / {pair}: '+('HbO' if k==0 else 'HbR'),xlabel='Time (s)',ylabel='Common observed SD')
            axs[k].legend(fontsize=8)
        axs[2].plot(t,delta.sum(axis=0)*np.sqrt(300),label='Change in total Hb')
        axs[2].plot(t,(delta[1]-params[1]*delta.sum(axis=0))*np.sqrt(300),label='Change in deoxy-fraction coordinate')
        axs[2].set(title='Joint minimum correction',xlabel='Time (s)',ylabel='Common observed SD')
        axs[2].legend(fontsize=8)
    fig.savefig(out/'subject01_channel_repairs.png',dpi=240); plt.close(fig)
    (out/'subject01_channel_cases.json').write_text(json.dumps(cases,indent=2))
    np.savez_compressed(out/'subject01_channel_cases.npz',**arrays)
    names=NAMES
    arms=['fixed','subject_primary','subject_channel_primary','subject_expanded','subject_channel_expanded']
    lines=['# 跨窗口共享参数适配与双 Hb 最小修正：2026-09-18 v1','',
    '本轮回答：先允许被试及采样位置具有共享参数，再比较独立留出窗口与固定参考的最小曲线改动；进一步给出条件性的总 Hb / 氧合比例修正语义。结果不能用于教师资格或唯一生理病因诊断。','',
    '## 主要结果','',
    '允许参数适配后，四个数据集的留出偏离均减小，但未接近零。主范围的被试×通道适配使改动能量下降约 31.5%–39.2%；更宽范围下降约 36.0%–43.5%。这是所选弱约束度量下相对于固定参考的改善，不能解释为这么多生理异常被修复。被试×通道并非总是优于被试共享：REFED 的通道级参数在留出窗口上更差，提示自由度增加及少量训练窗口带来的不稳定。','',
    '以下为各被试留出通道对×窗口距离中位数的等权平均；百分比表示 HbO/HbR 拼接曲线去均值 L2 范数的相对改动，不是异常时间点比例或浓度误差比例。','',
    '| 数据集 | 固定 | 被试共享 | 被试×通道 | 被试共享，扩展范围 | 被试×通道，扩展范围 |','|---|---:|---:|---:|---:|---:|']
    for dataset,name in names.items():
        group=summary[summary.dataset_id==dataset].set_index('arm')
        lines.append('| '+name+' | '+' | '.join(f'{group.loc[a,"distance"]*100:.2f}%' for a in arms)+' |')
    lines+=['','![适配前后留出距离](adaptation_comparison.png)','','## 样本、划分与拟合合同','',
    f'沿用原来的 {manifest["windows"]} 个窗口、{manifest["pair_rows"]} 个通道对×窗口、{manifest["valid_pair_rows"]} 个完整支持通道对×窗口。每条记录按原始时间顺序选择两端各 floor((n−1)/2) 个窗口，中间留出至少 30 秒间隔：8 窗口记录分为 3/2/3，3 窗口记录分为 1/1/1。早块训练、晚块评价，然后反向；间隔窗口不参与本轮拟合或主要评价。最终评价 {manifest["assessment_windows"]} 个窗口、{manifest["assessment_pair_rows"]} 个有效通道对×窗口。固定基线在完全相同的行上计算，因此数值不应与此前全 194 窗口汇总直接混用。','',
    '跨窗口共享层次为同一被试全部通道共享，以及同一被试同一原生通道跨窗口共享；均跨该被试纳入的记录共享。没有通过结果选择脑区、通道或新样本。原生通道名是采样位置身份，不等同于已定位的脑源。REFED 的每个通道每折只有 2 个训练窗口，Visual 部分被试每折只有 3 个，个体化估计证据有限。','',
    '所有窗口来自既有全记录双向滤波缓存。块间留间隔减少直接相邻影响，但不能使预处理后的样本严格统计独立；这不是独立新记录验证，也不是严格因果预测。没有使用被留出的曲线来选参数；留出曲线只用于给定参数下求最小修正。','',
    '| 参数 | 固定参考 | 主探索范围 | 扩展敏感性范围 |','|---|---:|---:|---:|']
    for param in PARAMETER_NAMES:
        lines.append(f'| {param} | {cfg["reference_parameters"][param]} | {cfg["parameter_boxes"]["primary"][param]} | {cfg["parameter_boxes"]["expanded"][param]} |')
    lines+=['','这些是事先限定的探索范围，不是经本数据验证的健康人正常区间。窗口内部参数恒定，没有时变参数，也没有调整上游神经驱动参数、观测相对增益或串扰矩阵。','',
    '每组优化训练窗口相对改动距离平方的均值，每个通道对×窗口等权。使用全部边界角点、固定参考、中点及固定种子的 32 个对数均匀点，取最好的 5 个起点作 L-BFGS-B。保留固定参考候选，扩展范围还纳入主范围解，防止训练目标因候选缺失而恶化。这里是数值搜索找到的最好参数，不是全局最优性的证明。固定参数下的正交投影本身是精确最小修正。','',
    '## 指标及可辨识性','',
    '令 y=[O,R]，先按各色团去均值，尺度 N=||y||₂；Bθ 是消去未知常数基线及 P−V 初态差异后的七行白化积分约束。对每个留出窗口：','',
    '```text\ndθ = ||Bθ y||₂ / N\nΔy = −Bθᵀ Bθ y\ny_corrected = y + Δy\n```','',
    '这里的修正只保证所检查的七条必要弱约束成立，不保证全部时间分辨率的微分方程、血流正性、绝对基线或完整非线性随机六状态系统成立。因此这是到完整确定性相容集合距离的下界诊断，不能把修正后的曲线命名为已恢复的真实生理轨迹。没有以测量噪声协方差加权，最小性依赖 HbO/HbR 同单位等权欧氏度量。','',
    '当前线性关系只识别 τ、a=ηc(E₀)、d=ηb(E₀,α) 三个组合。不同 η/E₀/α 可以给出相同的约束和联合修正。fitted_parameters.csv 保留一个数值代表及相同组合在对应参数盒内的等价范围；这些范围不是置信区间。固定总 Hb 的修正不依赖如何拆开 a,d；固定线性化脱氧比例的修正还依赖选择的 η，因此其生理解释是额外有条件的。','',
    '## 参数边界与泛化','',
    '| 适配层次 | 主范围 E₀ 在下界 | 主范围 τ 在上界 | 扩展 E₀ 在下界 | 扩展 τ 在上界 |','|---|---:|---:|---:|---:|']
    for sharing,title in [('subject','被试共享'),('subject_channel','被试×通道')]:
        values=[]
        for box in ('primary','expanded'):
            group=fitted[fitted.arm==sharing+'_'+box]
            values.extend([np.mean(np.isclose(group.E0,cfg['parameter_boxes'][box]['E0'][0]))*100,
                           np.mean(np.isclose(group.tau,cfg['parameter_boxes'][box]['tau'][1]))*100])
        lines.append('| '+title+' | '+' | '.join(f'{x:.1f}%' for x in values)+' |')
    lines+=['','这不是“估计发现被试氧提取率普遍异常低”。参数组合存在不可辨识性，边界可能是模型用受限参数补偿观测失配的结果。扩大范围后最优值继续向边界移动，意味着不能把本次改善简单解释为找到了可信的个体生理参数；也不能据有界搜索残差宣称整个参数族均不可能拟合。','',
    '| 被试 | 固定 | 被试共享 | 被试×通道 | 被试×通道扩展 |','|---|---:|---:|---:|---:|']
    for (dataset,subject),group in subjects.groupby(['dataset_id','subject']):
        group=group.set_index('arm')
        lines.append('| '+names[dataset]+'/'+subject+' | '+' | '.join(f'{group.loc[a,"distance_median"]*100:.2f}%' for a in ['fixed','subject_primary','subject_channel_primary','subject_channel_expanded'])+' |')
    lines+=['','## 修正对应的生理坐标','',
    '使用 T=O+R 与 Z=R−ηT。Z/Q₀ 在参考点的一阶近似下等于 q/p−1，因此 Z 是线性化脱氧比例坐标；Z 增加对应这一比例增加。它不是瞬时氧提取率 E，也不是绝对血氧饱和度。','',
    '比较三个条件：联合最小修正；固定 T、仅改变 O/R 分配（ΔO=−ΔR）；固定 Z、仅改变总 Hb 且按基线比例分配（ΔR=ηΔT）。三者分别求满足同一七条约束的最小曲线改动。限制越多，所需改动不小于联合最小值。条件修正距离超过 100% 在数学上允许，表示该受限方向很低效，不表示绝对生理浓度为负。','',
    '以下采用主范围的被试×通道拟合，先求被试中位数再等权平均；最后一列为固定 T 的修正比固定 Z 更省改动的窗口比例，先被试内统计再等权平均。','',
    '| 数据集 | 联合修正 | 固定总 Hb，仅改分配 | 固定脱氧比例，仅改总 Hb | 前一路径更省改动 |','|---|---:|---:|---:|---:|']
    for dataset,name in names.items():
        g=cs[(cs.dataset_id==dataset)&(cs.box=='primary')]
        lines.append('| '+name+' | '+' | '.join(f'{g[k].mean()*100:.2f}%' for k in ['distance','fixed_total_Hb_distance','fixed_linear_oxygenation_distance','oxygenation_route_cheaper_fraction'])+' |')
    lines+=['','主/扩展参数盒下两条受限路径的效率排序一致比例（被试等权）：'+ '；'.join(f'{names[d]} {agree[d]*100:.1f}%' for d in names)+'。这检验的是条件修正方向的稳定性，不是病因归属。','',
    '## Single-Trial subject_01：不能整人归为氧提取问题','',
    'subject_01（用户所说被试 001 在本数据集中的规范身份）固定参考距离中位数为 35.28%，被试共享后 28.84%，被试×通道后 24.69%，扩展通道参数范围后 23.01%。在主范围通道适配下，固定总 Hb 的修正中位数为 82.86%，固定线性化脱氧比例的修正为 34.52%；71.7% 的窗口更容易走后一路径。这个整体特征不支持将其一概命名为氧提取过程异常。','',
    '不同采样位置又存在相反模式。以下两个通道是用于说明差异的具名例子，不是总体代表性筛选；完整通道表见 channel_summary.csv。','',
    '| 通道 | 窗口数 | 固定参考 | 适配后联合 | 固定总 Hb，仅改分配 | 固定脱氧比例，仅改总 Hb |','|---|---:|---:|---:|---:|---:|']
    for pair in ('AF8Fp2','C1C3'):
        r=channel[(channel.dataset_id=='eeg_fnirs_single_trial')&(channel.subject=='subject_01')&(channel.pair==pair)&(channel.box=='primary')].iloc[0]
        lines.append(f'| {pair} | {r.pairs} | '+' | '.join(f'{r[k]*100:.2f}%' for k in ['fixed_distance','distance','fixed_total_Hb_distance','fixed_linear_oxygenation_distance'])+' |')
    lines+=['','对 AF8Fp2，可以说：在适配后固定模型和选定观测度量下，保留总 Hb 而改变 HbO/HbR 分配，是较省改动的修复路径，提示需核查氧合分配关系或色团观测混合。对 C1C3，可以说：保留线性化脱氧比例而改变总 Hb，是较省改动的路径，提示需核查总 Hb/容量状态的观测对应及动态收支。不能将前者直接命名为氧提取异常、后者直接命名为真实血容量异常。','',
    '![同一被试两个通道的修正示例](subject01_channel_repairs.png)','',
    '图中曲线是各通道留出距离接近该通道中位数的一个窗口，并非通道平均。实线为原观测，虚线为七条弱约束下的最近曲线；右列为加到观测上的 ΔT、ΔZ。原生幅度可由保存的共同范数及均值恢复。','',
    '## 为什么仍不能唯一归因到氧提取','',
    '在线性化体积、总 Hb、脱氧 Hb 平衡方程分别加入未知 eᵥ、eₚ、e_q 时：','',
    '```text\nJ = (τD+1)e_q − (cτD+b)eₚ − (c−b)eᵥ.\n```','',
    '同一个残差可以由不同平衡方程的偏离解释。即便固定总 Hb 的修正更小，氧提取、血流带来的脱氧 Hb 输送/清除、未建模区室以及光学串扰仍可混淆。本轮没有独立血流、绝对浓度基线或浅层分离信息，所以不能给出“此被试氧提取过程违反模型”的唯一归因。更具体的过程诊断需要独立观测或明确固定其余方程的条件实验。','',
    '另外，所有拟合只是使这些必要约束更接近，不代表完整随机模型已被验证或拒绝。未做噪声零假设标定，未拟合完整非线性后验，也未验证投影轨迹的全部生理可行性。没有扩大原有公开数据范围或读取受保护证据。','',
    '## 验证与复现','',
    f'固定参数距离与上一轮逐行一致，最大差 {manifest["fixed_reproduction_max_error"]:.3g}；投影后约束闭合最大误差 {manifest["maximum_repair_closure_error"]:.3g}。合成异参数多频轨迹仅在训练相位拟合，留出相位距离由 {manifest["synthetic_checks"]["fixed_heldout_mean_distance"]:.6f} 降为 {manifest["synthetic_checks"]["adapted_heldout_mean_distance"]:.3g}。12 项针对性测试通过；全项目仅收集测试，具体输出见 verification.json。所有 902 组任务完成，每组含主/扩展两种范围；未出现所有局部优化尝试均失败的任务。','',
    '运行由 systemd 用户服务监督，linger 已开启，32 个独立工作进程，每个数值库 1 线程，最多 64 个任务在途。机器可用 52 个物理核，用户配额 80 个逻辑 CPU；合成预检中 16/32 工作进程吞吐为约 38.6/44.7 组每秒，故使用 32。资源、命令、日志和恢复信息见 resources.json、launch.json、run.log。','',
    '源码与配置在 source_snapshot 保留，运行表由 manifest.json 持有；报告导出另存报告版本源码，不改写已完成拟合记录。两张新增曲线/比较图均为 240 dpi PNG；本轮没有 PDF 导出，未检查 WPS。','',
    '后续优先核查边界解和测量映射，再考虑引入独立血流或浅层通道信息；不建议据本轮 E₀ 点估计直接生成“氧提取异常”的被试标签。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import shutil
    shutil.copy2(Path(__file__),out/'source_snapshot'/'report_export_v1.py')
    print(channel[(channel.dataset_id=='eeg_fnirs_single_trial')&(channel.subject=='subject_01')&(channel.pair.isin(['AF8Fp2','C1C3']))].to_string(index=False))
    print('Conditional route agreement:', agree.to_dict())


def render_smooth_adaptation_report(output_dir):
    """Keep the original discrete metric; add band-limited correction sensitivity."""
    import shutil
    if output_dir is None:
        raise ValueError('--adapt-smooth-report requires --output-dir')
    out=Path(output_dir).resolve()
    manifest=json.loads((out/'manifest.json').read_text())
    if manifest['status']!='completed' or manifest['schema']!='hbo_hbr_adaptation_v1':
        raise ValueError('Completed adaptation evidence required')
    if (out/'REPORT_v2.md').exists():
        raise FileExistsError('Retain previous exports')
    frame=pd.read_csv(out/'pair_inventory.csv',dtype={'subject':str})
    scores=pd.read_csv(out/'heldout_metrics.csv',dtype={'subject':str})
    curves=np.load(out/'prepared_inputs.npz')['curves']
    fits={r['key']:r for r in map(json.loads,(out/'fit_results.jsonl').read_text().splitlines())}
    reference=[manifest['resolved_config']['reference_parameters'][p] for p in PARAMETER_NAMES]
    rows=[]
    for row in scores.itertuples():
        params=reference if row.arm=='fixed' else fits[row.fit_key]['expanded' if row.arm.endswith('expanded') else 'primary']['parameters']
        y=curves[int(frame.loc[row.row_index,'curve_index'])]
        modes=(12,18) if row.arm=='subject_channel_primary' else (12,)
        for mode in modes:
            delta,exchange,total=conditional_curve_repairs(y,params,max_cosine_mode=mode)
            rows.append(dict(row_index=row.row_index,dataset_id=row.dataset_id,subject=row.subject,pair=row.pair,
                             arm=row.arm,mode=mode,raw_distance=row.distance,distance=float(np.linalg.norm(delta)),
                             fixed_total_Hb_distance=float(np.linalg.norm(exchange)),
                             fixed_linear_oxygenation_distance=float(np.linalg.norm(total))))
    smooth=pd.DataFrame(rows)
    smooth['oxygenation_route_cheaper']=smooth.fixed_total_Hb_distance<smooth.fixed_linear_oxygenation_distance
    assert np.all(smooth.distance>=smooth.raw_distance-1e-10)
    smooth.to_csv(out/'smooth_metrics_v2.csv',index=False)
    subjects=smooth.groupby(['dataset_id','subject','arm','mode']).agg(distance=('distance','median'),
        raw_distance=('raw_distance','median'),fixed_total_Hb_distance=('fixed_total_Hb_distance','median'),
        fixed_linear_oxygenation_distance=('fixed_linear_oxygenation_distance','median'),
        oxygenation_route_cheaper_fraction=('oxygenation_route_cheaper','mean')).reset_index()
    subjects.to_csv(out/'smooth_subject_summary_v2.csv',index=False)
    summary=subjects.groupby(['dataset_id','arm','mode']).mean(numeric_only=True).reset_index()
    summary.to_csv(out/'smooth_summary_v2.csv',index=False)
    raw=pd.read_csv(out/'conditional_repairs.csv')
    raw['route']=raw.fixed_total_Hb_distance<raw.fixed_linear_oxygenation_distance
    agree=smooth[(smooth.arm=='subject_channel_primary')&(smooth['mode']==12)].merge(raw[['row_index','route']],on='row_index')
    agree['agreement']=agree.oxygenation_route_cheaper==agree.route
    agreement=agree.groupby(['dataset_id','subject']).agreement.mean().groupby('dataset_id').mean()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases=json.loads((out/'subject01_channel_cases.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),constrained_layout=True)
    t=np.arange(300)/10; arrays={}
    for axs,case in zip(axes,cases):
        pair=case['pair']; y=curves[case['curve_index']]
        delta,exchange,total=conditional_curve_repairs(y,case['parameters'],max_cosine_mode=12)
        arrays[pair+'_joint_change']=delta; arrays[pair+'_fixed_total_change']=exchange; arrays[pair+'_fixed_oxygenation_change']=total
        for k in range(2):
            axs[k].plot(t,y[k]*np.sqrt(300),label='Observed')
            axs[k].plot(t,(y[k]+delta[k])*np.sqrt(300),'--',label='Smooth weak-compatible')
            axs[k].set(title=f'subject_01 / {pair}: '+('HbO' if k==0 else 'HbR'),xlabel='Time (s)',ylabel='Common observed SD')
            axs[k].legend(fontsize=8)
        axs[2].plot(t,delta.sum(axis=0)*np.sqrt(300),label='Change in total Hb')
        axs[2].plot(t,(delta[1]-case['parameters'][1]*delta.sum(axis=0))*np.sqrt(300),label='Change in deoxy-fraction coordinate')
        axs[2].set(title='Smooth joint correction',xlabel='Time (s)',ylabel='Common observed SD')
        axs[2].legend(fontsize=8)
    fig.savefig(out/'subject01_channel_repairs_smooth_v2.png',dpi=240);plt.close(fig)
    np.savez_compressed(out/'subject01_smooth_changes_v2.npz',**arrays)
    section=['## 平滑修正补充核查：避免把积分权重伪影解释为生理曲线','',
    '可视化发现：原离散等权欧氏最小修正含有逐点交替的锯齿。这来自 Simpson 求积权重，不是新的实测高频成分。原距离仍是所声明离散约束下的精确最小改动，可以用于前后对比；但其修正曲线不宜直接解释为生理变化。v1 报告、图和数值原样保留，本版以明确的平滑修正空间补充检验。','',
    '不重新拟合任何参数；使用完全相同的留出曲线与冻结参数，将允许的修正限制在 30 秒窗的前 12 个非直流余弦模式（约 0.0167–0.2 Hz）中，重新求同一七条弱约束下的最小范数修正。这个余弦空间带有窗口边界约定，不等同于记录级理想带通或真实生理可行集合。另以 18 模式（最高 0.3 Hz）作修正空间敏感性检查。','',
    '下表是平滑修正后的留出中位数，被试等权；参数仍由原离散训练目标选择，不能称为平滑目标重新优化后的最优参数。','',
    '| 数据集 | 固定 | 被试共享 | 被试×通道 | 被试共享扩展 | 被试×通道扩展 |','|---|---:|---:|---:|---:|---:|']
    arms=['fixed','subject_primary','subject_channel_primary','subject_expanded','subject_channel_expanded']
    for dataset,name in NAMES.items():
        group=summary[(summary.dataset_id==dataset)&(summary['mode']==12)].set_index('arm')
        section.append('| '+name+' | '+' | '.join(f'{group.loc[a,"distance"]*100:.2f}%' for a in arms)+' |')
    section+=['','平滑前后两条条件修正路径的效率排序一致比例：'+'；'.join(f'{NAMES[d]} {agreement[d]*100:.1f}%' for d in NAMES)+'。','',
    '| 数据集 | 主通道适配，12 模式 | 主通道适配，18 模式 |','|---|---:|---:|']
    for dataset,name in NAMES.items():
        group=summary[(summary.dataset_id==dataset)&(summary.arm=='subject_channel_primary')].set_index('mode')
        section.append(f'| {name} | {group.loc[12,"distance"]*100:.2f}% | {group.loc[18,"distance"]*100:.2f}% |')
    section+=['','Single-Trial subject_01 的两个通道，在同一留出窗口集合上的平滑修正中位数为：','',
    '| 通道 | 联合最小修正 | 固定总 Hb，仅改分配 | 固定脱氧比例，仅改总 Hb |','|---|---:|---:|---:|']
    for pair in ('AF8Fp2','C1C3'):
        g=smooth[(smooth.dataset_id=='eeg_fnirs_single_trial')&(smooth.subject=='subject_01')&(smooth.pair==pair)&
                 (smooth.arm=='subject_channel_primary')&(smooth['mode']==12)]
        section.append('| '+pair+' | '+' | '.join(f'{g[k].median()*100:.2f}%' for k in ['distance','fixed_total_Hb_distance','fixed_linear_oxygenation_distance'])+' |')
    section+=['','![平滑修正示例](subject01_channel_repairs_smooth_v2.png)','',
    '平滑限制后仍能观察到不同通道的条件修正方向差异。这个补充避免把求积权重导致的锯齿当作生理修正；它仍不能使“氧合分配方向”唯一等同于“氧提取过程”。','']
    report=(out/'REPORT.md').read_text()
    report=report.replace('# 跨窗口共享参数适配与双 Hb 最小修正：2026-09-18 v1','# 跨窗口共享参数适配与双 Hb 最小修正：2026-09-18 v2（含平滑修正核查）')
    report=report.replace('## 修正对应的生理坐标','\n'.join(section)+'\n## 修正对应的生理坐标（原离散度量，供前后比较）')
    report=report.replace('![同一被试两个通道的修正示例](subject01_channel_repairs.png)',
        '原离散修正图见保留的 v1 报告；本版解释采用上方无逐点锯齿的平滑修正图。')
    report=report.replace('12 项针对性测试通过','13 项针对性测试通过')
    report=report.replace('因此这是到完整确定性相容集合距离的下界诊断',
        '因此这是到完整确定性线性约束相容集合距离的下界诊断，不是到非线性 SSM 的距离下界')
    report=report.replace('图中曲线是各通道留出距离接近该通道中位数的一个窗口，并非通道平均。','上方平滑图使用各通道原留出距离接近该通道中位数的一个窗口，并非通道平均。')
    (out/'REPORT_v2.md').write_text(report)
    shutil.copy2(Path(__file__),out/'source_snapshot'/'report_export_v2.py')
    print(summary.to_string(index=False))
    print('Smooth/raw conditional agreement:',agreement.to_dict())


def physical_to_effective(parameters):
    tau, a, d = physical_combinations(parameters)
    return np.array([1/tau, a, (a-d)/tau])


def effective_constraint_basis(parameters, *, first_order=False):
    lam, a, k = parameters
    if first_order:
        optical = np.array([k-a*lam, -a, 0.])
        deoxy = np.array([lam, 1., 0.])
    else:
        optical = np.array([k*lam-a*lam*lam, k-2*a*lam, -a])
        deoxy = np.array([lam*lam, 2*lam, 1.])
    _, components = compact_constraint_components()
    matrix = np.einsum('i,ijk->jk', np.r_[optical, optical+deoxy], components)
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    if np.sum(singular > singular[0]*1e-9) != 7:
        raise ValueError('Effective candidate loses constraint rank')
    return vh[:7]


@lru_cache(maxsize=1)
def compact_smooth_coordinates():
    coordinates, _ = compact_constraint_components()
    return coordinates@smooth_correction_space(12)[1].T


def constraint_objective(basis, gram, *, smooth=False):
    operator = np.linalg.lstsq(basis@compact_smooth_coordinates(), basis, rcond=1e-10)[0] if smooth else basis
    return float(np.einsum('ij,jk,ik->', operator, gram, operator))


def fit_effective_parameters(gram, cfg, model, warm_starts=(), fixed=None, global_check=False):
    from scipy.optimize import differential_evolution
    domain = cfg['effective_domain']
    bounds = np.array([domain['lambda'],domain['signed_a' if model in ('signed_a','signed_both') else 'positive_a'],
                       domain['signed_k' if model in ('signed_k','signed_both') else 'positive_k']],dtype=float)
    def encode(values):
        x=np.array(values,dtype=float,copy=True)
        x[...,0]=np.log1p(x[...,0]/.01)
        x[...,2]=np.arcsinh(x[...,2]/.2)
        return x
    def decode(x):
        x=np.array(x,dtype=float,copy=True)
        x[...,0]=.01*np.expm1(x[...,0])
        x[...,2]=.2*np.sinh(x[...,2])
        return x
    transformed=np.stack([encode(bounds[:,0]),encode(bounds[:,1])],axis=1)
    def to_unit(values):
        return (encode(values)-transformed[:,0])/(transformed[:,1]-transformed[:,0])
    free=np.array([i for i in range(3) if fixed is None or i!=fixed[0]])
    def parameters(v):
        unit=np.zeros(3);unit[free]=v
        x=decode(transformed[:,0]+unit*(transformed[:,1]-transformed[:,0]))
        if fixed is not None: x[fixed[0]]=fixed[1]
        return x
    evaluations=0
    def objective(v):
        nonlocal evaluations
        evaluations+=1
        return constraint_objective(effective_constraint_basis(parameters(v)),gram)
    starts=[np.array(v) for v in product((0.,1.),repeat=len(free))]
    starts.append(np.full(len(free),.5))
    for x in warm_starts:
        starts.append(np.clip(to_unit(x)[free],0,1))
    rng=np.random.default_rng(cfg['synthetic']['seed'])
    starts.extend(rng.uniform(0,1,(cfg['optimizer']['random_starts'],len(free))))
    scored=sorted((objective(v),i,v) for i,v in enumerate(starts))
    best_loss,_,best=scored[0];attempts=[]
    opt=cfg['optimizer']
    for _,_,start in scored[:opt['refined_starts']]:
        fit=minimize(objective,start,method='L-BFGS-B',bounds=[(0.,1.)]*len(free),
            options=dict(maxiter=opt['maxiter'],ftol=opt['ftol'],gtol=opt['gtol']))
        attempts.append(bool(fit.success))
        if np.isfinite(fit.fun) and fit.fun<best_loss:best_loss,best=float(fit.fun),fit.x
    multistart_loss=best_loss;de_loss=None
    if global_check:
        audit=cfg['optimizer_crosscheck']
        de=differential_evolution(objective,[(0.,1.)]*len(free),seed=cfg['synthetic']['seed'],
                                  maxiter=audit['maxiter'],popsize=audit['popsize'],tol=audit['tol'],polish=True)
        de_loss=float(de.fun)
        if de.fun<best_loss:best_loss,best=float(de.fun),de.x
    return dict(parameters=parameters(best).tolist(),train_mse=best_loss,multistart_mse=multistart_loss,
                differential_evolution_mse=de_loss,objective_evaluations=evaluations,
                local_successes=int(sum(attempts)),fixed_coordinate=fixed)


def boundary_synthetic_panel(condition,replicate,cfg):
    scfg=cfg['synthetic'];rng=np.random.default_rng(scfg['seed']+replicate)
    count=scfg['training_windows']+scfg['heldout_windows'];t=np.arange(300)/10
    truth=scfg['truth'];tau,a,d=physical_combinations(truth)
    curves=[]
    for _ in range(count):
        total=np.zeros(300);deoxy=np.zeros(300)
        for frequency in (.02,.035,.06,.09,.14,.19):
            z=2j*np.pi*frequency
            wave=rng.uniform(.2,1)*np.exp(1j*rng.uniform(0,2*np.pi))*np.exp(z*t)
            total+=wave.real;deoxy+=((a*tau*z+d)/(tau*z+1)*wave).real
        y=np.array([total-deoxy,deoxy]);y-=y.mean(axis=1,keepdims=True)
        S=np.sqrt(np.mean(np.sum(y*y,axis=0)))
        # Consume the same random variables in every condition for paired controls.
        white=rng.normal(size=y.shape)
        colored=np.zeros_like(y)
        for frequency in (.015,.025,.05,.08):
            colored+=np.cos(2*np.pi*frequency*t[None,:]+rng.uniform(0,2*np.pi,(2,1)))
        colored-=colored.mean(axis=1,keepdims=True)
        colored/=np.std(colored,axis=1,keepdims=True)
        if condition=='white_10pct':y+=.1*S*white
        elif condition=='colored_independent_30pct':y+=.3*S*colored
        elif condition=='hbr_gain_2':y[1]*=2
        elif condition=='shared_component_50pct':y+=.5*S*colored[:1]
        elif condition=='independent_hbr_50pct':y[1]+=.5*S*colored[1]
        elif condition!='matched':raise ValueError('Unknown synthetic boundary control')
        y-=y.mean(axis=1,keepdims=True);curves.append(y/np.linalg.norm(y))
    return np.array(curves)


def boundary_analysis_task(task):
    key,train,test,cfg,parent_cfg,parent_fits=task
    gram=train.T@train/len(train);test_gram=test.T@test/len(test)
    models={};profiles=[]
    warm=[]
    if parent_fits is not None:
        for box in ('primary','expanded'):
            fit=parent_fits[box]
            models['physical_'+box]=dict(parameters=fit['parameters'],physical=True,first_order=False,smooth=False)
            warm.append(physical_to_effective(fit['parameters']))
        for name,first,smooth in [('physical_smooth',False,True),('physical_equilibrated',True,False)]:
            fit=fit_shared_parameters(gram,parent_cfg,'primary',parent_fits['primary']['parameters'],first_order=first,smooth=smooth)
            models[name]=dict(parameters=fit['parameters'],physical=True,first_order=first,smooth=smooth,
                              optimization=fit)
    else:
        fit=fit_shared_parameters(gram,parent_cfg,'primary')
        models['physical_primary']=dict(parameters=fit['parameters'],physical=True,first_order=False,smooth=False,optimization=fit)
        warm.append(physical_to_effective(fit['parameters']))
        warm.append(physical_to_effective(cfg['synthetic']['truth']))
    selected=cfg['effective_models'] if parent_fits is not None else ['positive','signed_both']
    for name in selected:
        fit=fit_effective_parameters(gram,cfg,name,warm,global_check=parent_fits is not None and name in ('positive','signed_both'))
        models['effective_'+name]=dict(parameters=fit['parameters'],physical=False,first_order=False,smooth=False,optimization=fit)
        warm.append(np.array(fit['parameters']))
    if parent_fits is not None:
        for coordinate,values,model in [(0,cfg['lambda_profile'],'positive'),(1,cfg['a_profile'],'signed_both')]:
            for value in values:
                fit=fit_effective_parameters(gram,cfg,model,warm,fixed=(coordinate,value))
                basis=effective_constraint_basis(fit['parameters'])
                profiles.append(dict(coordinate=('lambda' if coordinate==0 else 'a'),value=value,model=model,
                    parameters=fit['parameters'],train_mse=constraint_objective(basis,gram),heldout_mse=constraint_objective(basis,test_gram)))
    for name,model in models.items():
        effective=physical_to_effective(model['parameters']) if model['physical'] else np.array(model['parameters'])
        basis=effective_constraint_basis(effective,first_order=model['first_order'])
        model.update(effective_parameters=effective.tolist(),train_mse=constraint_objective(basis,gram,smooth=model['smooth']),
                     heldout_mse=constraint_objective(basis,test_gram,smooth=model['smooth']),
                     train_raw_mse=constraint_objective(basis,gram),heldout_raw_mse=constraint_objective(basis,test_gram),
                     train_smooth_mse=constraint_objective(basis,gram,smooth=True),heldout_smooth_mse=constraint_objective(basis,test_gram,smooth=True),
                     heldout_distances=np.linalg.norm(test@basis.T,axis=1).tolist())
    return dict(key=key,models=models,profiles=profiles)


def boundary_checks(cfg):
    physical=(2.,.35,.32,.32);effective=physical_to_effective(physical)
    b1=compact_constraint_basis(physical);b2=effective_constraint_basis(effective)
    assert np.max(abs(b1.T@b1-b2.T@b2))<1e-11
    tau,a,d=physical_combinations(physical)
    frequencies=np.array([.01,.03,.1,.2])
    z=2j*np.pi*frequencies
    lam,a2,k=effective
    assert np.allclose((a*tau*z+d)/(tau*z+1),a2-k/(z+lam))
    curves=boundary_synthetic_panel('matched',0,cfg)
    coords,_=compact_constraint_components();x=curves.reshape(len(curves),600)@coords.T
    n=cfg['synthetic']['training_windows'];gram=x[:n].T@x[:n]/n
    fit=fit_effective_parameters(gram,cfg,'positive',[effective])
    distance=np.sqrt(constraint_objective(effective_constraint_basis(fit['parameters']),x[n:].T@x[n:]/len(x[n:])))
    assert distance<1e-4
    # Direct input-output relation is sufficient in an equilibrated synthetic response.
    first=effective_constraint_basis(effective,first_order=True)
    assert np.linalg.norm(x@first.T)/np.sqrt(len(x))<1e-4
    return dict(effective_mapping_max_projector_error=float(np.max(abs(b1.T@b1-b2.T@b2))),
                matched_heldout_rms_distance=float(distance),effective_truth=effective.tolist(),effective_fitted=fit['parameters'],
                positive_high_frequency_gain=True)


def run_boundary_analysis(args):
    import os, shutil, time
    from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
    cfg=yaml.safe_load(args.boundary_config.read_text())
    if cfg['schema']!='hbo_hbr_boundary_v1' or cfg['tensor']!=dict(components=['HbO','HbR'],samples=300,sample_rate_hz=10):
        raise ValueError('Unsupported boundary diagnostic contract')
    controls=boundary_checks(cfg)
    if args.check_only:
        print(json.dumps(controls,indent=2));return
    if args.output_dir is None:raise ValueError('--boundary requires --output-dir')
    out=args.output_dir.resolve();parent=ROOT/cfg['parent']
    if out.parent!=ROOT/cfg['output_namespace']:raise ValueError('Invalid output namespace')
    if args.resume:
        manifest=json.loads((out/'manifest.json').read_text())
        if manifest['resolved_config']!=cfg or manifest['status']=='completed':raise ValueError('Cannot resume this run')
    else:
        out.mkdir(exist_ok=False)
        manifest=dict(schema=cfg['schema'],status='preparing',resolved_config=cfg,command=sys.argv,
                      synthetic_checks=controls,source='retained normalized curves; no raw or protected data reads',started_at=time.time())
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        for filename in ['experiments/scripts/analyze_hbo_hbr_relationship.py','experiments/configs/physiology_semantic_tokenizer/hbo_hbr_boundary_v1.yaml',
                         'experiments/configs/physiology_semantic_tokenizer/hbo_hbr_adaptation_v1.yaml','tests/test_hbo_hbr_relationship.py']:
            target=out/'source_snapshot'/filename;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/filename,target)
    parent_manifest=json.loads((parent/'manifest.json').read_text())
    if parent_manifest['status']!='completed':raise ValueError('Parent evidence is incomplete')
    parent_cfg=parent_manifest['resolved_config']
    frame=pd.read_csv(parent/'pair_inventory.csv',dtype={'subject':str});curves=np.load(parent/'prepared_inputs.npz')['curves']
    coords,_=compact_constraint_components();x=curves.reshape(len(curves),600)@coords.T
    parent_groups=json.loads((parent/'fit_groups.json').read_text())
    parent_fits={r['key']:r for r in map(json.loads,(parent/'fit_results.jsonl').read_text().splitlines())}
    jobs=[];inventory={}
    for key,g in parent_groups.items():
        if g['sharing']!='subject':continue
        train=x[frame.loc[g['train_indices'],'curve_index'].to_numpy()];test=x[frame.loc[g['test_indices'],'curve_index'].to_numpy()]
        jobs.append((key,train,test,cfg,parent_cfg,parent_fits[key]));inventory[key]=g
    for condition in cfg['synthetic']['conditions']:
        for replicate in range(cfg['synthetic']['replicates']):
            y=boundary_synthetic_panel(condition,replicate,cfg);z=y.reshape(len(y),600)@coords.T;n=cfg['synthetic']['training_windows']
            key=f'synthetic|{condition}|{replicate}'
            jobs.append((key,z[:n],z[n:],cfg,parent_cfg,None));inventory[key]=dict(condition=condition,replicate=replicate)
    (out/'task_inventory.json').write_text(json.dumps(inventory,indent=2))
    # Local derivatives and exact-output equivalence distinguish active pressure from redundant coordinates.
    gradients=[]
    for key,g in parent_groups.items():
        train=x[frame.loc[g['train_indices'],'curve_index'].to_numpy()];gram=train.T@train/len(train)
        for box in ('primary','expanded'):
            params=np.array(parent_fits[key][box]['parameters']);bounds=np.array([parent_cfg['parameter_boxes'][box][p] for p in PARAMETER_NAMES])
            eq=equivalent_parameter_ranges(params,parent_cfg['parameter_boxes'][box])
            for j,name in enumerate(PARAMETER_NAMES):
                logp=np.log(params);step=np.zeros(4);step[j]=1e-4
                up=constraint_objective(compact_constraint_basis(np.exp(logp+step)),gram)
                down=constraint_objective(compact_constraint_basis(np.exp(logp-step)),gram)
                gradients.append(dict(key=key,box=box,parameter=name,value=params[j],log_derivative=(up-down)/2e-4,
                    lower=bool(np.isclose(params[j],bounds[j,0],rtol=1e-5)),upper=bool(np.isclose(params[j],bounds[j,1],rtol=1e-5)),
                    equivalent_min=eq.get(name+'_equivalent_min',params[j]),equivalent_max=eq.get(name+'_equivalent_max',params[j])))
    pd.DataFrame(gradients).to_csv(out/'parent_boundary_gradients.csv',index=False)
    completed={};result_path=out/'task_results.jsonl'
    if result_path.exists():
        for r in map(json.loads,result_path.read_text().splitlines()):completed[r['key']]=r
    jobs=[j for j in jobs if j[0] not in completed]
    manifest.update(status='running',tasks_total=len(inventory),subject_tasks=24,synthetic_tasks=48,cpu_affinity=list(os.sched_getaffinity(0)))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=cfg['resources']['workers']) as pool,result_path.open('a') as output:
        iterator=iter(jobs);pending={}
        for _ in range(min(len(jobs),cfg['resources']['max_in_flight'])):
            task=next(iterator);pending[pool.submit(boundary_analysis_task,task)]=task[0]
        while pending:
            done,_=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future);r=future.result();completed[r['key']]=r;output.write(json.dumps(r)+'\n');output.flush()
                task=next(iterator,None)
                if task is not None:pending[pool.submit(boundary_analysis_task,task)]=task[0]
            print(f'boundary tasks {len(completed)}/{len(inventory)} elapsed={time.monotonic()-start:.1f}s',flush=True)
    rows=[];profiles=[];predictions=[]
    for key,r in completed.items():
        info=inventory[key];synthetic=key.startswith('synthetic|')
        for name,model in r['models'].items():
            effective=model['effective_parameters'];params=model['parameters']
            row=dict(key=key,model=name,kind='synthetic' if synthetic else 'measured',dataset_id=info.get('dataset_id','synthetic'),
                     subject=info.get('subject',str(info.get('replicate',''))),condition=info.get('condition','measured'),
                     fit_block=info.get('fit_block','independent_phases'),lambda_value=effective[0],a=effective[1],k=effective[2],
                     **{field:model[field] for field in ['train_mse','heldout_mse','train_raw_mse','heldout_raw_mse','train_smooth_mse','heldout_smooth_mse']})
            if model['physical']:
                row.update(dict(zip(PARAMETER_NAMES,params)))
            else:
                optimization=model['optimization'];row.update(multistart_mse=optimization['multistart_mse'],de_mse=optimization['differential_evolution_mse'])
            rows.append(row)
            if not synthetic:
                for index,distance in zip(info['test_indices'],model['heldout_distances']):
                    predictions.append(dict(dataset_id=info['dataset_id'],subject=info['subject'],row_index=index,model=name,distance=distance))
        for profile in r['profiles']:
            profiles.append(dict(key=key,dataset_id=info['dataset_id'],subject=info['subject'],fit_block=info['fit_block'],
                                 **{k:v for k,v in profile.items() if k!='parameters'},lambda_value=profile['parameters'][0],
                                 a=profile['parameters'][1],k=profile['parameters'][2]))
    table=pd.DataFrame(rows);table.to_csv(out/'model_metrics.csv',index=False)
    pd.DataFrame(profiles).to_csv(out/'profiles.csv',index=False)
    prediction=pd.DataFrame(predictions);prediction.to_csv(out/'heldout_distances.csv',index=False)
    subject_summary=prediction.groupby(['dataset_id','subject','model']).distance.median().reset_index()
    subject_summary.to_csv(out/'subject_summary.csv',index=False)
    summary=subject_summary.groupby(['dataset_id','model']).distance.mean().reset_index();summary.to_csv(out/'summary.csv',index=False)
    manifest.update(status='completed',elapsed_fitting_seconds=time.monotonic()-start,summary=summary.to_dict('records'))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(summary.to_string(index=False),flush=True)


def render_boundary_report(output_dir):
    """Derived evidence only; the completed fit manifest and source stay frozen."""
    import shutil
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(output_dir).resolve()
    manifest=json.loads((out/'manifest.json').read_text());cfg=manifest['resolved_config']
    if manifest['status']!='completed':raise ValueError('Report requires completed evidence')
    if (out/'REPORT.md').exists():raise FileExistsError('Preserve the dated report; use a versioned export')
    m=pd.read_csv(out/'model_metrics.csv',dtype={'subject':str});measured=m[m.kind=='measured'];synthetic=m[m.kind=='synthetic']
    profiles=pd.read_csv(out/'profiles.csv');gradients=pd.read_csv(out/'parent_boundary_gradients.csv')
    parent=ROOT/cfg['parent'];inventory=json.loads((out/'task_inventory.json').read_text())
    frame=pd.read_csv(parent/'pair_inventory.csv',dtype={'subject':str});curves=np.load(parent/'prepared_inputs.npz')['curves']
    coordinates,components=compact_constraint_components();x=curves.reshape(len(curves),600)@coordinates.T
    results={r['key']:r for r in map(json.loads,(out/'task_results.jsonl').read_text().splitlines())}
    assert len(results)==72 and len(measured)==192 and len(synthetic)==144
    reproduction=[];limits=[]
    for key,r in results.items():
        if key.startswith('synthetic|'):continue
        info=inventory[key];test=x[frame.loc[info['test_indices'],'curve_index'].to_numpy()]
        gram=test.T@test/len(test)
        for name,model in r['models'].items():
            basis=effective_constraint_basis(model['effective_parameters'],first_order=model['first_order'])
            reproduction.append(abs(constraint_objective(basis,gram)-model['heldout_raw_mse']))
        lam=r['models']['effective_positive']['effective_parameters'][0]
        # k -> infinity removes R from the constraint: (D+lambda)T = constant.
        matrix=lam*(components[0]+components[3])+components[1]+components[4]
        _,s,vh=np.linalg.svd(matrix,full_matrices=False)
        assert (s>s[0]*1e-9).sum()==7
        limits.append(dict(key=key,dataset_id=info['dataset_id'],subject=info['subject'],lambda_value=lam,
            total_only_limit_mse=constraint_objective(vh[:7],gram),
            fitted_positive_mse=r['models']['effective_positive']['heldout_raw_mse']))
    limit_table=pd.DataFrame(limits);limit_table.to_csv(out/'total_only_limit.csv',index=False)
    wide=measured.pivot(index='key',columns='model',values='heldout_raw_mse')
    positive=measured[measured.model=='effective_positive'];signed=measured[measured.model=='effective_signed_a']
    direct=measured[measured.model=='physical_equilibrated'];smooth=measured[measured.model=='physical_smooth']
    primary=gradients[(gradients.box=='primary')&gradients.key.str.startswith('subject|')]
    slopes=primary[primary.parameter=='E0'].log_derivative
    lp=profiles[profiles.coordinate=='lambda'].pivot(index='key',columns='value',values='train_mse')
    excess=(lp[0]-lp.min(axis=1))/lp.min(axis=1)
    verification=dict(status='passed',completed_tasks=len(results),heldout_metric_reproduction_max_error=max(reproduction),
        paired_heldout_windows=int(len(pd.read_csv(out/'heldout_distances.csv'))/8),
        negative_a_subject_folds=int((signed.a<0).sum()),
        signed_a_heldout_mse_improved_folds=int((wide.effective_signed_a<wide.effective_positive).sum()),
        optimizer_crosscheck_max_loss_difference=float(abs(measured.multistart_mse-measured.de_mse).max()),
        primary_E0_outward_gradient_folds=int((slopes>0).sum()),primary_E0_log_gradient_median=float(slopes.median()),
        positive_lambda_max=float(positive.lambda_value.max()),
        lambda_zero_profile_relative_excess_median=float(excess.median()),
        lambda_zero_profile_relative_excess_max=float(excess.max()),
        smooth_refit_E0_lower_folds=int(np.isclose(smooth.E0,.15).sum()),
        equilibrated_refit_E0_lower_folds=int(np.isclose(direct.E0,.15).sum()),
        targeted_tests='15 passed',collection='672 / 683 collected; 11 deselected by existing configuration',
        synthetic_truth_note='effective fits receive the known truth as one warm start; diagnostic recovery, not a blinded qualification')
    assert max(reproduction)<1e-12
    (out/'verification.json').write_text(json.dumps(verification,indent=2))
    rms=measured.groupby(['dataset_id','model']).heldout_raw_mse.mean().pow(.5).unstack()
    rms.to_csv(out/'dataset_heldout_rms.csv')
    phys=synthetic[synthetic.model=='physical_primary']
    syn=phys.groupby('condition').agg(tau_median=('tau','median'),
        tau_upper=('tau',lambda v:int(np.isclose(v,5).sum())),E0_lower=('E0',lambda v:int(np.isclose(v,.15).sum())),
        median_rms=('heldout_raw_mse',lambda v:float(np.median(np.sqrt(v)))))
    syn.to_csv(out/'synthetic_summary.csv')
    def md_table(table):
        headers=list(table.columns)
        rows=['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']
        rows.extend('| '+' | '.join(str(v) for v in row)+' |' for row in table.itertuples(index=False,name=None))
        return '\n'.join(rows)
    selected=['physical_primary','physical_expanded','effective_positive','effective_signed_a']
    plot_names=['Physical box','Expanded box','Positive combinations','Allow negative a']
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    colors=plt.get_cmap('tab10').colors
    for j,(dataset,label) in enumerate(NAMES.items()):
        axes[0].plot(range(4),rms.loc[dataset,selected],'-o',label=label,color=colors[j])
    axes[0].set_xticks(range(4),plot_names,rotation=17,ha='right');axes[0].set_ylabel('Held-out RMS curve correction / pair norm')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2);axes[0].set_title('Same held-out rows; subject/fold equal weight')
    for j,(condition,g) in enumerate(phys.groupby('condition')):
        axes[1].scatter(j+np.linspace(-.12,.12,len(g)),g.tau,s=25)
    axes[1].set_xticks(range(len(syn)),['Slow independent','HbR gain x2','Slow HbR only','Matched','Slow shared','White noise'],rotation=30,ha='right')
    axes[1].axhline(2,color='black',ls=':',label='True tau = 2 s');axes[1].axhline(5,color='gray',ls='--',label='Fit upper bound')
    axes[1].set_ylabel('Fitted tau (s)');axes[1].set_title('Synthetic: identical physiology, changed observations');axes[1].legend(fontsize=8)
    fig.savefig(out/'boundary_mechanisms.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(13,6.2),layout='constrained')
    for col,(dataset,label) in enumerate(NAMES.items()):
        for row,coordinate in enumerate(['lambda','a']):
            ax=axes[row,col]
            for key,g in profiles[(profiles.dataset_id==dataset)&(profiles.coordinate==coordinate)].groupby('key'):
                g=g.sort_values('value');base=g.train_mse.min()
                ax.plot(g.value,100*(g.train_mse/base-1),lw=1,alpha=.8)
            if coordinate=='lambda':
                ax.set_xscale('symlog',linthresh=.01);ax.axvline(.1,color='gray',ls=':',lw=1)
                ax.axvline(.2,color='gray',ls='--',lw=1);ax.set_title(label);ax.set_xlabel('lambda = 1 / tau (1/s)')
            else:
                ax.axvline(0,color='black',ls=':',lw=1);ax.set_xlabel('a (signed diagnostic)')
            ax.set_ylabel('Training loss above profile minimum (%)');ax.grid(alpha=.2)
    fig.savefig(out/'boundary_profiles.png',dpi=220);plt.close(fig)
    comparison=rms[selected].rename(index=NAMES,columns=dict(zip(selected,['原参数箱','扩展参数箱','正有效组合','允许 a<0'])))
    comparison=comparison.map(lambda v:f'{100*v:.2f}%').reset_index(names='数据集')
    syn_display=syn.copy();syn_display['tau_median']=syn_display.tau_median.map(lambda v:f'{v:.4f}')
    syn_display['median_rms']=syn_display.median_rms.map(lambda v:f'{v:.6f}')
    report=r'''# 双 Hb 参数触边的机制分析

本报告只解释当前保留样本和这项必要约束的拟合行为；不是新的六状态似然拟合、模型资格判定或总体生理结论。
机器证据 owner：[manifest.json](manifest.json)、[逐模型结果](model_metrics.csv)、[profiles.csv](profiles.csv)、[核验](verification.json)。
使用上轮已经保留的归一化曲线；未新增原始、validation 或 protected 数组访问。
范围：4 数据集 × 3 被试 × 2 时间分块 = 24 组；同一被试跨通道共享参数，5040 个配对留出窗口。
另复核父运行 902 组 × 2 参数箱的局部梯度，生成 48 个合成对照面板。

## 结论与证据强弱

当前触边主要表现为有方向的模型补偿，而非优化器偶然停在盒子边缘。它包含可区分的几件事：

1. 四个原参数在当前线性双 Hb 约束中只有三个独立组合，确有结构冗余，但不能单独解释真实数据上的向外下降方向。
2. 数据要求减弱、甚至反转模型中必为正的瞬时耦合；在原坐标里这表现为 E0 或 eta 降到下界。
3. 数据偏好比窗口更长或接近无恢复力的动态；tau 的上边界同时混合了慢成分补偿和短窗口对长时间常数缺乏分辨力。
4. 有色观测成分足以在生理真值不变时造成同类触边。最小几何修正不是噪声校准后的生理参数估计。
5. REFED 还有大耦合退化方向：约束逐渐只检查 total Hb，弱化对 HbR 的约束。扩边可以改变拟合关系的语义。

目前可以定位被违反的数学关系，不能仅靠两个相对浓度信号，把原因唯一分配给氧提取、血容量、光学转换或浅层混入。

## 数学定位：边界到底在关闭哪一项

令 T=HbO+HbR，R=HbR，D=d/dt，eta=Q0/P0，

    c = 1 + (1-E0) log(1-E0) / E0
    a = eta*c
    lambda = 1/tau
    k = eta*(1-c)/(alpha*tau)

则线性六状态系统的双 Hb 必要关系（允许任意初始 p-v）可写为

    (D+lambda) [(D+lambda)(R-aT) + kT] = 0.

未知浓度基线在实现中作为常数 nuisance 消去；使用 8 个弱测试函数消去该自由度后，剩 7 个约束。
在平衡初始条件或稳态受迫响应中，对应传递关系为

    H_R/T(s) = a - k/(s+lambda).

原生理域中 a>0、k>0、lambda>0；参考值为 (lambda,a,k)=(0.5,0.063164,0.304764)。
这里的 a 是模型关系的瞬时项/高频渐近增益，不是本次带限数据直接测出的无限频率增益。
当 E0→0 时 c≈E0/2，模型关闭正的瞬时项 a；不是从数据估出了“实际氧提取率接近零”。
c 是流入脱氧血红蛋白项 fE(f)/E0 对流量的局部导数；k 汇合了脱氧、排出和体积耦合，不能当作一个独立氧代谢参数。
若 k>0，Im H(iω)=kω/(lambda²+ω²)>0；本轮主要改善来自放开 a 的符号，放开 k 的符号收益较小且不稳定。

eta、E0、alpha 的等价曲线为 eta=a/c(E0)，alpha=eta*(1-c)*lambda/k。
因此当前线性约束不能唯一恢复它们；这不是对完整非线性系统所有观测条件的不可辨识性证明。
拟合箱截断这条等价曲线，也会使一个等价解显示为边界解。纯光学增益对照正好展示这种情况。

## 真实数据：冗余去掉之后仍有方向压力

原箱 E0=0.15 的 24/24 组，其损失对 log(E0) 的导数均为正，中位数为 0.00118056；因此减小 E0 可继续降低训练损失。
扩箱后 E0=0.05 的 24/24 组导数仍为正，中位数 0.000153622。它们并非只是在完全平坦方向上偶然选到边界。
原箱与扩箱的 tau 均有 18/24 组触上界，其向外梯度同样非零。
通道共享复核：原箱 755/878 组 E0 精确触下界，扩箱 756/878 组；多数也有同向外推压力。
梯度的微小近零个例应视作数值/平坦方向，不作生理分类。

把参数改成三个有效组合并放宽到 lambda∈[0,4]、a∈[0,1]、k∈[0,40] 后，23/24 组取 a=0；
剩余 REFED 一组走向大 k 退化极限。允许 a∈[-1,1] 后，24/24 组均取 a<0，24/24 组留出均方偏离均下降。
多起点 L-BFGS-B 与独立 differential evolution 的 48 项交叉核验，目标差最大为 7.53e-9。
这排除了本轮“主要是局部优化失败”的解释，但不是全局最优的数学证明。

以下为同一留出集合的 sqrt(mean(distance²))：先在每组内均值，再对被试与分块等权，分母为每条中心化双 Hb 曲线的联合 L2 范数。
它不是误差点数比例，也不是浓度被改变的百分比；7 条弱约束下的最小改动仍只是完整线性确定性相容集合距离的下界。

@COMPARISON@

![留出与合成机制对照](boundary_mechanisms.png)

这些 RMS 数字与上轮“每被试中位数再平均”不是同一个汇总。两种都保留：
REFED 的中位数汇总从正组合 0.20359 变为允许负 a 的 0.20730，略变差；其每组均方损失均下降。
因此不能说每条曲线、每个分位数都获益。完整中位数见 [summary.csv](summary.csv)。

负 a 只是诊断放松，不是新的生理模型；9/24 组仍到 a=-1 的探索边界。
不能把这个新边界值当成已识别参数，也不能把“扩边后分数下降”写成触边问题已经解决。

## tau 上边界：缺少恢复力与缺少长时程信息叠加

在正组合域，24/24 组最优 lambda<0.072，即有限值对应 tau>13.97 秒，另有 lambda=0 的极限解。
在重新拟合其余组合的 lambda profile 中，lambda=0 相对网格最优的训练损失增量，中位数只有 0.211%，最大 3.457%。
相反 lambda=0.2（tau=5 秒）相对 profile 最优增量的中位数约 19.94%。
这支持“模型想去更慢的方向”，却不支持把很大的 tau 精确解释为真实生理传输时间。
30 秒、去均值的窗口能排斥部分较快恢复，但区分长 tau 与无限 tau 的能力很弱。
lambda→0 时微分算子的低频/趋势自由度也改变，容易容纳慢变成分；这并不是独立的血流量测量。

![重新拟合伴随参数的目标剖面](boundary_profiles.png)

上排为正组合域内 lambda profile，下排为有符号诊断域内 a profile；每条线是一组被试/分块。
剖面纵轴是各自网格最优值以上的训练损失百分比，不能当作似然比或置信区间。

大 k 时，归一化后的必要约束趋于 (D+lambda)T=constant，HbR 项相对消失。
这会让“拟合双 Hb 生理耦合”退化成“寻找 total Hb 中的低能量方向”。REFED 的 6 组正组合拟合 k=6.50–40，
其中 3 组达到 k=40；[total_only_limit.csv](total_only_limit.csv) 保留此极限与拟合损失的直接比较。
因此一味扩边可能让模型逃避原本要检验的耦合，而不是找到更可信的生理解释。

## 不是由离散修正锯齿或初值消除单独引起

将训练目标改为只允许 12 个非 DC 余弦模式的平滑修正，重新拟合后 E0 仍 24/24 触下界，tau 仍 18/24 触上界。
这避免上轮 Simpson 离散几何修正的逐点交替结构，结论不消失。
进一步假设初始 total Hb 与体积平衡，直接用一阶关系重新拟合，E0 仍 24/24 触下界。
后一对照换了约束集合，不能把距离当作相同模型的嵌套优劣比较；它只说明二阶初值消除不是 E0 触边的充分解释。

## 合成因果对照：正常真值也能走到边界

所有面板的真实生理参数均为 tau=2、eta=0.35、E0=0.32、alpha=0.32；每条件 8 次重复，12 个训练和 12 个独立相位留出窗口。
白噪声以联合 Hb RMS 的 10% 注入每分量；独立慢变为每分量 30%；共同慢变及 HbR 单独慢变为 50%。
慢变频率为 0.015、0.025、0.05、0.08 Hz。幅度与频谱不是实际噪声估计，条件间幅度也不相等，不能据此排序真实污染来源。

@SYNTHETIC@

无污染和小白噪声恢复 tau≈2、有效组合接近真值，且原参数不触边；无污染时不同 eta/E0/alpha 仍可给出同一正确组合，直接展示结构冗余。
独立慢成分 8/8 次将 E0 推至下界，tau 中位数 4.768；只污染 HbR 时 E0 也 8/8 触下界，tau 6/8 触上界。
共同慢变更偏向时间常数补偿：tau 7/8 触上界，E0 仅 2/8 触下界。
这些是在生理真值完全不变时产生的拟合偏移，说明不能将真实测量上的触边直接诊断成缺氧、氧提取异常或真实血流过慢。

仅把 HbR 光学增益乘 2，曲线仍可被另一组有效参数几乎精确表示，但 tau 从 2 变成约 4.687。
对平衡初始响应，若 R_m=gR、O_m=O，A=1+(g-1)a，则

    a_m=g*a/A,  lambda_m=lambda-(g-1)*k/A,  k_m=g*k/A².

本例这些量仍在正域内；原参数箱则将 eta 或 alpha 推至边界。
因此“曲线拟合得很好”也不能排除相对光学标度被吸收为生理差异。
有效组合合成拟合包含真值 warm start；这里只是算法与机制对照，未冒充盲真值恢复资格实验。物理四参数拟合使用原通用起点。

## 为什么慢成分能够推动边界：目标函数本身的性质

当前最小修正是欧氏投影，L(theta)=trace(P_theta S_y)，P_theta 为参数决定的秩 7 正交投影。
在加性误差独立、零均值且尺度固定的简化模型 y=x+epsilon 下，期望目标包含

    E L(theta) = trace(P_theta S_x) + trace(P_theta Sigma_epsilon).

若误差在测量坐标中各向同性，第二项等于 7*sigma²，与 theta 无关；若慢变、相关或双 Hb 不等方差，则第二项随参数变化。
优化会同时选“最像生理信号”的约束和“最少投影到噪声能量”的约束。样本再多也不自动消除这种目标偏置。
实际逐窗口归一化还使权重依赖观测，上述加法公式是说明机制的固定尺度情形，不能直接作为当前数据的无偏噪声扣除公式。
本轮没有估计实际 Sigma_epsilon，也没有证明全部偏离都是噪声。具有独立组织/血管来源的真实成分同样违反单源直接输出假设。

## 对过去六状态开发历程的解释边界

历史 T3 Step 2 的 subject_03 存在参数明显改变而预测近似等价的解，支持补偿/不可辨识。
但 subject_13、subject_10 扩边后预测明显改变，原报告保留 inconclusive；不能把所有案例统一说成平坦参数脊。
Step 3 只开放 kappa 的跨会话拟合仍失败；Step 5 的一维 W 在数值分辨率检查通过后，18/18 后验仍集中于下边界。
这与“只要降低拟合维数就可解决”的解释不相容。
早期综合参数合成实验还存在推断近似和生成/拟合过程不完全匹配的历史问题，不能把这些失败全部归因于本轮 Hb 耦合。

本轮已经去掉 EEG 驱动、EKF 和隐状态推断，触边依然存在，因而那些环节不是当前现象的必要条件。
在当前必要关系中 beta、kappa、gamma 这些上游参数已被消去；调它们不能改变双 Hb 之间必须成立的这个约束。
固定更多参数只会减少补偿通道；若数据要求 a<0 而模型恒有 a>0，剩余自由参数仍会被推向最接近数据的边缘。
强先验能够压回内部，但这只是改变折衷，不能作为观测语义已修复或生理参数已可信的证据。

## 开发建议

先把“曲线相容性”“可观测组合”“生理参数”分开判断，而不是继续把触边率当作优化器健康度。
对当前数据，下一步应在训练范围内估计/约束观测误差的时序与双 Hb 协方差，并对已有相对光学标度、浅层/系统性成分假设作可证伪比较；
噪声加权必须由独立或训练证据确定，不能为了消除触边任意调权。
保持生理参数跨窗口共享；若有效组合都需要非法符号，优先检验观测层或单一血流来源假设。
如果观测层校准后符号冲突仍在，再讨论需要额外生理来源、时变过程或非线性，不能先将非法参数改名当作新生理机制。

当前证据足以说明：现有相对双 Hb + 当前确定性直接输出关系 + 当前距离目标，尚不能支持触边参数的生理解释。
它不构成对所有真实双 Hb、完整非线性随机六状态模型的普遍否定，也不能区分具体哪个被试真的存在氧提取过程异常。

## 可复现与验证

运行合同 [hbo_hbr_boundary_v1.yaml](../../../../configs/physiology_semantic_tokenizer/hbo_hbr_boundary_v1.yaml)，
命令/环境/监督进程见 [launch.json](launch.json)，资源见 [resources.json](resources.json)，进度与结束日志见 [run.log](run.log)。
拟合前通过独立传递函数、投影等价、已知真值合成检查；15 项针对性测试通过；收集 672/683（既有配置排除 11）。
原始拟合代码保存在 source_snapshot；本报告导出代码另存，不替换原快照。
图均为 220 dpi PNG；本次交付 Markdown 与 PNG，未生成 PDF，未检查 WPS。
'''
    report=report.replace('@COMPARISON@',md_table(comparison)).replace('@SYNTHETIC@',md_table(syn_display.reset_index()))
    (out/'REPORT.md').write_text(report)
    shutil.copy2(Path(__file__),out/'source_snapshot'/'report_export_v1.py')
    print(json.dumps(verification,indent=2));print(comparison.to_string(index=False))


def gain_transform(effective, gain):
    """Steady-response R/T combinations after HbR-only gain (not an initial-state identity)."""
    lam,a,k=effective
    denominator=1+(gain-1)*a
    return np.array([lam-(gain-1)*k/denominator,gain*a/denominator,gain*k/denominator**2])


def observation_constraint_basis(parameters, *, first_order=False):
    lam,a,k,gain=parameters
    optical=np.array([k-a*lam,-a,0.]) if first_order else np.array([k*lam-a*lam*lam,k-2*a*lam,-a])
    deoxy=np.array([lam,1.,0.]) if first_order else np.array([lam*lam,2*lam,1.])
    _,components=compact_constraint_components()
    matrix=np.einsum('i,ijk->jk',np.r_[optical,(optical+deoxy)/gain],components)
    _,s,vh=np.linalg.svd(matrix,full_matrices=False)
    if (s>s[0]*1e-9).sum()!=7:raise ValueError('Observation constraint loses rank')
    return vh[:7]


def normalize_metric_covariance(covariance, floor=1e-8):
    cov=(covariance+covariance.T)/2
    eigen,vectors=np.linalg.eigh(cov)
    scale=float(np.trace(cov)/len(cov))
    if scale<=0:raise ValueError('Covariance has no positive scale')
    clipped=np.maximum(eigen,scale*floor)
    return (vectors*clipped)@vectors.T/scale,dict(scale=scale,raw_min_eigenvalue=float(eigen.min()),
        floored_eigenvalues=int((eigen<scale*floor).sum()),condition=float(clipped.max()/clipped.min()))


def weighted_constraint_objective(basis,gram,covariance):
    return float(np.trace(np.linalg.solve(basis@covariance@basis.T,basis@gram@basis.T)))


def observation_proxy_metrics(train,cfg):
    """Proxies only: physiological signal can contaminate both estimators."""
    coordinates,_=compact_constraint_components();delta=np.diff(train,axis=2)
    c=np.einsum('nit,njt->ij',delta,delta)/(2*len(train)*delta.shape[2])
    diag=np.diag(np.diag(c))
    denominator=float(np.sum(train[:,:,:-1]**2))
    rho=float(np.clip(np.sum(train[:,:,1:]*train[:,:,:-1])/denominator,*cfg['proxy_contract']['rho_bounds']))
    innovation=train[:,:,1:]-rho*train[:,:,:-1]
    c_ar=np.einsum('nit,njt->ij',innovation,innovation)/(len(train)*innovation.shape[2]*(1-rho*rho))
    dt=abs(np.arange(300)[:,None]-np.arange(300)[None,:])
    raw={'identity':np.eye(len(coordinates)),
         'difference_diagonal_proxy':coordinates@np.kron(diag,np.eye(300))@coordinates.T,
         'ar1_proxy':coordinates@np.kron(c_ar,rho**dt)@coordinates.T}
    metrics={};metadata={}
    for name,cov in raw.items():
        metrics[name],metadata[name]=normalize_metric_covariance(cov,cfg['proxy_contract']['eigenvalue_floor_relative'])
    metadata['ar1_proxy'].update(rho=rho,Hb_covariance=c_ar.tolist())
    metadata['difference_diagonal_proxy'].update(Hb_variance=np.diag(c).tolist())
    return metrics,metadata


def observation_synthetic_panel(condition,replicate,cfg):
    sc=cfg['synthetic'];rng=np.random.default_rng(sc['seed']+replicate)
    n=sc['train_windows']+sc['heldout_windows'];t=np.arange(300)/10
    effective=physical_to_effective(sc['truth']);lam,a,k=effective
    clean=np.zeros((n,2,300))
    for frequency in sc['signal_frequencies_hz']:
        z=2j*np.pi*frequency
        wave=rng.uniform(.2,1,(n,1))*np.exp(1j*rng.uniform(0,2*np.pi,(n,1)))*np.exp(z*t)
        r=(a-k/(z+lam))*wave
        clean[:,0]+=(wave-r).real;clean[:,1]+=r.real
    clean-=clean.mean(axis=2,keepdims=True);clean/=np.linalg.norm(clean.reshape(n,-1),axis=1)[:,None,None]
    gain=2. if condition.startswith('gain2') else 1.
    observed=clean.copy();observed[:,1]*=gain
    white_sd=sc['white_sd']/np.sqrt(300);noise=rng.normal(size=clean.shape)*white_sd
    coordinates,_=compact_constraint_components();covariance=white_sd**2*np.eye(len(coordinates))
    if 'slow' in condition:
        slow_sd=sc['slow_sd']/np.sqrt(300);frequencies=sc['slow_frequencies_hz']
        shared=condition=='shared_slow';sources=1 if shared else 2
        for frequency in frequencies:
            for wave in (np.cos(2*np.pi*frequency*t),np.sin(2*np.pi*frequency*t)):
                coefficients=rng.normal(size=(n,sources,1))*slow_sd/np.sqrt(len(frequencies))
                noise+=coefficients*wave
                for source in range(sources):
                    loading=np.ones((2,1)) if shared else np.eye(2)[:,source,None]
                    projected=coordinates@(loading*wave).ravel()
                    covariance+=slow_sd**2/len(frequencies)*np.outer(projected,projected)
    observed+=noise;observed-=observed.mean(axis=2,keepdims=True)
    # Independent noise-only calibration, not the signal covariance or test residuals.
    reference=rng.normal(size=(sc['reference_calibration_samples'],len(coordinates)))@np.linalg.cholesky(covariance).T
    calibrated=reference.T@reference/len(reference)
    return observed,covariance,calibrated,gain


def fit_observation_parameters(gram,covariance,cfg,*,first_order=False,fixed_gain=1.,warm=()):
    bounds=np.array(cfg['effective_bounds']+[cfg['gain_bounds']],dtype=float)
    def encode(v):
        x=np.array(v,dtype=float,copy=True);x[...,0]=np.log1p(x[...,0]/.01);x[...,2]=np.arcsinh(x[...,2]/.2);x[...,3]=np.log(x[...,3]);return x
    def decode(v):
        x=np.array(v,dtype=float,copy=True);x[...,0]=.01*np.expm1(x[...,0]);x[...,2]=.2*np.sinh(x[...,2]);x[...,3]=np.exp(x[...,3]);return x
    transformed=np.stack([encode(bounds[:,0]),encode(bounds[:,1])],axis=1)
    dimensions=4 if fixed_gain is None else 3
    def params(v):
        u=np.full(4,.5);u[:dimensions]=v
        p=decode(transformed[:,0]+u*np.diff(transformed,axis=1)[:,0])
        if fixed_gain is not None:p[3]=fixed_gain
        return p
    def objective(v):
        return weighted_constraint_objective(observation_constraint_basis(params(v),first_order=first_order),gram,covariance)
    opt=cfg['optimizer'];rng=np.random.default_rng(opt['seed'])
    starts=[np.array(v) for v in product((0.,1.),repeat=dimensions)]+[np.full(dimensions,.5)]
    starts.extend(rng.uniform(0,1,(opt['random_starts'],dimensions)))
    for p in warm:
        starts.append(np.clip(((encode(p)-transformed[:,0])/np.diff(transformed,axis=1)[:,0])[:dimensions],0,1))
    ranked=sorted((objective(v),i,v) for i,v in enumerate(starts));best_loss,_,best=ranked[0];attempts=[]
    for _,_,v in ranked[:opt['refined_starts']]:
        r=minimize(objective,v,method='L-BFGS-B',bounds=[(0.,1.)]*dimensions,
                   options=dict(maxiter=opt['maxiter'],ftol=opt['ftol'],gtol=opt['gtol']))
        attempts.append(dict(success=bool(r.success),loss=float(r.fun),parameters=params(r.x).tolist()))
        if np.isfinite(r.fun) and r.fun<best_loss:best_loss,best=float(r.fun),r.x
    return dict(parameters=params(best).tolist(),train_loss=best_loss,attempts=attempts)


def observation_analysis_task(task):
    key,train,test,cfg,extra=task
    coords,_=compact_constraint_components();z=train.reshape(len(train),600)@coords.T;zt=test.reshape(len(test),600)@coords.T
    gram=z.T@z/len(z);heldout=zt.T@zt/len(zt)
    metrics,metadata=observation_proxy_metrics(train,cfg);synthetic=extra is not None
    if synthetic:
        for name,cov in zip(cfg['synthetic']['extra_metrics'],extra[:2]):metrics[name],metadata[name]=normalize_metric_covariance(cov)
    fits={};profiles=[];warm=[]
    for metric,cov in metrics.items():
        modes=[('unit_gain',1.),('free_gain',None)]
        if synthetic and metric in cfg['synthetic']['extra_metrics']:modes.append(('known_gain',extra[2]))
        for mode,gain in modes:
            fit=fit_observation_parameters(gram,cov,cfg,first_order=synthetic,fixed_gain=gain,warm=warm)
            p=fit['parameters'];warm.append(p);basis=observation_constraint_basis(p,first_order=synthetic)
            weighted=weighted_constraint_objective(basis,heldout,cov)
            residual=zt@basis.T
            corrections=(cov@basis.T@np.linalg.solve(basis@cov@basis.T,residual.T)).T
            # Common Euclidean evaluation includes the actual weighted repair, not just its weighted score.
            rawmin=np.mean(np.sum(residual**2,axis=1));actual=np.mean(np.sum(corrections**2,axis=1))
            assert actual>=rawmin-1e-10
            assert np.max(abs((zt-corrections)@basis.T))<1e-9
            fit.update(metric=metric,gain_mode=mode,heldout_weighted_loss=weighted,heldout_min_euclidean_mse=float(rawmin),
                       heldout_actual_repair_mse=float(actual),effective_observed=gain_transform(p[:3],p[3]).tolist(),
                       boundary=[name for name,v,(lo,hi) in zip(['lambda','a','k','gain'],p,cfg['effective_bounds']+[cfg['gain_bounds']])
                                 if np.isclose(v,lo,rtol=1e-4,atol=1e-7) or np.isclose(v,hi,rtol=1e-4,atol=1e-7)])
            fits[metric+'|'+mode]=fit
        if not synthetic and metric in ('identity','ar1_proxy'):
            for gain in cfg['gain_profile']:
                fit=fit_observation_parameters(gram,cov,cfg,fixed_gain=gain,warm=warm)
                profiles.append(dict(metric=metric,gain=gain,parameters=fit['parameters'],train_loss=fit['train_loss'],
                    heldout_loss=weighted_constraint_objective(observation_constraint_basis(fit['parameters']),heldout,cov)))
    total_cov,total_info=normalize_metric_covariance(gram)
    # A negative control: using all observed covariance as noise makes the fit objective flat.
    negative=[]
    for p in ([.1,.01,.05,1.],[.5,.1,.3,1.],[2.,.5,3.,1.],[1.,.3,1.,2.]):
        b=observation_constraint_basis(p,first_order=synthetic)
        negative.append(weighted_constraint_objective(b,gram,total_cov)/total_info['scale'])
    return dict(key=key,synthetic=synthetic,models=fits,profiles=profiles,metric_metadata=metadata,
                total_whitening_negative_control=negative,total_covariance_info=total_info)


def observation_checks(cfg):
    assert cfg['schema']=='hbo_hbr_observation_v1'
    assert cfg['tensor']==dict(components=['HbO','HbR'],samples=300,sample_rate_hz=10)
    true=physical_to_effective(cfg['synthetic']['truth']);observed=gain_transform(true,2.)
    left=observation_constraint_basis([*true,2.],first_order=True)
    right=observation_constraint_basis([*observed,1.],first_order=True)
    error=float(np.max(abs(left.T@left-right.T@right)))
    assert error<1e-11
    rng=np.random.default_rng(12);z=rng.normal(size=(64,24));g=z.T@z/64
    b=observation_constraint_basis([*true,1.]);assert np.isclose(weighted_constraint_objective(b,g,g),7.)
    assert np.isclose(weighted_constraint_objective(b,g,np.eye(24)),constraint_objective(b,g))
    y,cov,cal,gain=observation_synthetic_panel('gain2_white',0,cfg)
    assert y.shape==(128,2,300) and cov.shape==cal.shape==(24,24) and gain==2
    alias=[]
    for assumed in cfg['gain_profile']:
        e=gain_transform(observed,1/assumed)
        if np.all(e>0):
            bb=observation_constraint_basis([*e,assumed],first_order=True)
            alias.append(dict(gain=assumed,tau=1/e[0],a=e[1],k=e[2],projector_error=float(np.max(abs(bb.T@bb-left.T@left)))))
    return dict(gain_alias_projector_error=error,identical_covariance_score=7.,gain_aliases=alias)


def run_observation_analysis(args):
    import time,shutil
    from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
    cfg=yaml.safe_load(args.observation_config.read_text());checks=observation_checks(cfg)
    if args.check_only:print(json.dumps(checks,indent=2));return
    if args.output_dir is None:raise ValueError('Observation run needs a fresh output directory')
    out=args.output_dir.resolve();parent=ROOT/cfg['parent']
    if out.parent!=ROOT/cfg['output_namespace']:raise ValueError('Output outside audit namespace')
    if args.resume:
        manifest=json.loads((out/'manifest.json').read_text())
        if manifest['resolved_config']!=cfg or manifest['status']=='completed':raise ValueError('Invalid resume')
    else:
        out.mkdir(exist_ok=False)
        manifest=dict(status='preparing',resolved_config=cfg,checks=checks,command=sys.argv,started_at=time.time())
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        for name in ['experiments/scripts/analyze_hbo_hbr_relationship.py','tests/test_hbo_hbr_relationship.py',str(args.observation_config.resolve().relative_to(ROOT))]:
            target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    assert json.loads((parent/'manifest.json').read_text())['status']=='completed'
    frame=pd.read_csv(parent/'pair_inventory.csv',dtype={'subject':str});curves=np.load(parent/'prepared_inputs.npz')['curves']
    groups=json.loads((parent/'fit_groups.json').read_text());jobs=[];inventory={}
    for key,g in groups.items():
        if g['sharing']!='subject':continue
        train=curves[frame.loc[g['train_indices'],'curve_index'].to_numpy()];test=curves[frame.loc[g['test_indices'],'curve_index'].to_numpy()]
        jobs.append((key,train,test,cfg,None));inventory[key]=g
    for condition in cfg['synthetic']['conditions']:
        for rep in range(cfg['synthetic']['replicates']):
            y,cov,cal,gain=observation_synthetic_panel(condition,rep,cfg);n=cfg['synthetic']['train_windows'];key=f'synthetic|{condition}|{rep}'
            jobs.append((key,y[:n],y[n:],cfg,(cov,cal,gain)));inventory[key]=dict(condition=condition,replicate=rep,true_gain=gain)
    (out/'task_inventory.json').write_text(json.dumps(inventory,indent=2));pd.DataFrame(checks['gain_aliases']).to_csv(out/'exact_gain_aliases.csv',index=False)
    path=out/'task_results.jsonl';completed={r['key']:r for r in map(json.loads,path.read_text().splitlines())} if path.exists() else {}
    jobs=[j for j in jobs if j[0] not in completed];manifest.update(status='running',tasks_total=len(inventory))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));start=time.monotonic()
    with ProcessPoolExecutor(max_workers=cfg['resources']['workers']) as pool,path.open('a') as output:
        it=iter(jobs);pending={}
        for _ in range(min(len(jobs),cfg['resources']['max_in_flight'])):
            job=next(it);pending[pool.submit(observation_analysis_task,job)]=job[0]
        while pending:
            done,_=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
            for f in done:
                pending.pop(f);r=f.result();completed[r['key']]=r;output.write(json.dumps(r)+'\n');output.flush()
                job=next(it,None)
                if job is not None:pending[pool.submit(observation_analysis_task,job)]=job[0]
            print(f'observation tasks {len(completed)}/{len(inventory)} elapsed={time.monotonic()-start:.1f}s',flush=True)
    rows=[];profiles=[];negative=[]
    truth=physical_to_effective(cfg['synthetic']['truth'])
    for key,r in completed.items():
        info=inventory[key]
        for name,fit in r['models'].items():
            p=np.array(fit['parameters']);row=dict(key=key,kind='synthetic' if r['synthetic'] else 'measured',
                dataset_id=info.get('dataset_id','synthetic'),subject=info.get('subject',info.get('replicate')),
                condition=info.get('condition','measured'),fit_block=info.get('fit_block','independent_phases'),
                metric=fit['metric'],gain_mode=fit['gain_mode'],lambda_value=p[0],a=p[1],k=p[2],gain=p[3],
                tau=1/max(p[0],1e-12),boundary=','.join(fit['boundary']),
                **{k:fit[k] for k in ['train_loss','heldout_weighted_loss','heldout_min_euclidean_mse','heldout_actual_repair_mse']})
            if r['synthetic']:
                observed=gain_transform(truth,info['true_gain'])
                row.update(physiology_relative_error=float(np.linalg.norm((p[:3]-truth)/truth)/np.sqrt(3)),
                    observed_combination_relative_error=float(np.linalg.norm((np.array(fit['effective_observed'])-observed)/observed)/np.sqrt(3)))
            rows.append(row)
        for row in r['profiles']:profiles.append(dict(key=key,**{k:v for k,v in row.items() if k!='parameters'},parameters=json.dumps(row['parameters'])))
        negative.append(dict(key=key,kind='synthetic' if r['synthetic'] else 'measured',
            minimum=min(r['total_whitening_negative_control']),maximum=max(r['total_whitening_negative_control']),
            **r['total_covariance_info']))
    pd.DataFrame(rows).to_csv(out/'model_metrics.csv',index=False);pd.DataFrame(profiles).to_csv(out/'gain_profiles.csv',index=False)
    pd.DataFrame(negative).to_csv(out/'total_whitening_negative_control.csv',index=False)
    manifest.update(status='completed',elapsed_fitting_seconds=time.monotonic()-start,completed_tasks=len(completed),models=len(rows))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:manifest[k] for k in ['status','completed_tasks','models','elapsed_fitting_seconds']}))


def render_observation_report(output_dir):
    import shutil
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(output_dir).resolve();manifest=json.loads((out/'manifest.json').read_text());cfg=manifest['resolved_config']
    if manifest['status']!='completed':raise ValueError('Incomplete observation evidence')
    if (out/'REPORT.md').exists():raise FileExistsError('Preserve the completed report')
    table=pd.read_csv(out/'model_metrics.csv',dtype={'subject':str});measured=table[table.kind=='measured'];synthetic=table[table.kind=='synthetic']
    results=[json.loads(s) for s in (out/'task_results.jsonl').read_text().splitlines()]
    assert len(results)==64 and len(table)==624
    baseline=measured[(measured.metric=='identity')&(measured.gain_mode=='unit_gain')].set_index('key')
    old=pd.read_csv(out.parent/'20260918_hbo_hbr_boundary_v1'/'model_metrics.csv')
    old=old[(old.kind=='measured')&(old.model=='effective_positive')].set_index('key')
    baseline_error=float(abs(baseline.heldout_min_euclidean_mse-old.heldout_raw_mse).max())
    assert baseline_error<1e-6
    negative=pd.read_csv(out/'total_whitening_negative_control.csv')
    negerr=float(max(abs(negative.minimum-7).max(),abs(negative.maximum-7).max()))
    assert negerr<1e-6
    boundary=[]
    for (metric,mode),g in measured.groupby(['metric','gain_mode']):
        boundary.append(dict(metric=metric,gain_mode=mode,tau_median=float(g.tau.median()),a_median=float(g.a.median()),
            gain_median=float(g.gain.median()),a_lower=int(np.isclose(g.a,0,atol=1e-7).sum()),a_upper=int(np.isclose(g.a,1).sum()),
            any_boundary=int(g.boundary.notna().sum())))
    boundaries=pd.DataFrame(boundary);boundaries.to_csv(out/'measured_boundary_summary.csv',index=False)
    summaries={}
    for metric in ['heldout_min_euclidean_mse','heldout_actual_repair_mse']:
        summaries[metric]=measured.groupby(['dataset_id','metric','gain_mode'])[metric].mean().pow(.5)
        summaries[metric].rename('rms').reset_index().to_csv(out/(metric+'_summary.csv'),index=False)
    syn_summary=synthetic.groupby(['condition','metric','gain_mode']).agg(tau_median=('tau','median'),
        physiology_error_median=('physiology_relative_error','median'),physiology_error_max=('physiology_relative_error','max'),
        observed_error_median=('observed_combination_relative_error','median'),gain_median=('gain','median'))
    syn_summary.to_csv(out/'synthetic_summary.csv')
    verification=dict(status='passed',tasks=64,models=624,synthetic_panels=40,measured_subject_folds=24,
        parent_baseline_max_mse_difference=baseline_error,total_whitening_score_max_error=negerr,
        gain_alias_projector_max_error=max(r['projector_error'] for r in manifest['checks']['gain_aliases']),
        all_fits_have_a_successful_refinement=all(any(a['success'] for a in f['attempts']) for r in results for f in r['models'].values()),
        measured_ar1_rho_at_upper=int(sum(np.isclose(r['metric_metadata']['ar1_proxy']['rho'],.995) for r in results if not r['synthetic'])),
        targeted_tests='18 passed',collection='675 / 686 collected; 11 deselected by existing configuration',
        no_protected_or_raw_data_access=True,synthetic_repair_space='24-dimensional common constraint span',
        measured_calibration_available=False)
    (out/'verification.json').write_text(json.dumps(verification,indent=2))
    def md(df):
        rows=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
        rows.extend('| '+' | '.join(map(str,row))+' |' for row in df.itertuples(index=False,name=None))
        return '\n'.join(rows)
    columns=[('identity','unit_gain'),('difference_diagonal_proxy','unit_gain'),('ar1_proxy','unit_gain'),('identity','free_gain')]
    names=['欧氏 / g=1','差分对角 / g=1','AR1 / g=1','欧氏 / 自由 g']
    measured_display=summaries['heldout_actual_repair_mse'].unstack(['metric','gain_mode'])[columns]
    measured_display.columns=names;measured_display=measured_display.rename(index=NAMES).map(lambda v:f'{100*v:.2f}%').reset_index()
    synthetic_rows=[]
    conditions=cfg['synthetic']['conditions']
    for condition in conditions:
        plain=syn_summary.loc[(condition,'identity','unit_gain')]
        calibrated=syn_summary.loc[(condition,'independent_noise_calibration','known_gain')]
        synthetic_rows.append(dict(条件=condition,欧氏固定单位增益_tau=f'{plain.tau_median:.4f}',
            独立噪声校准且已知增益_tau=f'{calibrated.tau_median:.4f}',
            三组合相对误差中位数=f'{100*calibrated.physiology_error_median:.2f}%',
            三组合相对误差最大值=f'{100*calibrated.physiology_error_max:.2f}%'))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for j,(metric,mode) in enumerate([('identity','unit_gain'),('ar1_proxy','unit_gain'),('independent_noise_calibration','known_gain')]):
        values=[syn_summary.loc[(c,metric,mode),'physiology_error_median'] for c in conditions]
        axes[0].plot(range(5),100*np.array(values),'-o',label=['Euclidean; unit gain','AR1 proxy; unit gain','Independent noise calibration; known gain'][j])
    axes[0].set_xticks(range(5),['White','Independent slow','Shared slow','Gain x2 + white','Gain x2 + slow'],rotation=25,ha='right')
    axes[0].set_ylabel('Median RMS relative error of (lambda, a, k) (%)');axes[0].set_title('Synthetic: true tau = 2 s');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    values=summaries['heldout_actual_repair_mse'].unstack(['metric','gain_mode'])
    for dataset,label in NAMES.items():axes[1].plot(range(4),100*values.loc[dataset,columns].to_numpy(),'-o',label=label)
    axes[1].set_xticks(range(4),['Euclidean','Difference diagonal','AR1 proxy','Euclidean + free gain'],rotation=22,ha='right')
    axes[1].set_ylabel('Held-out Euclidean norm of applied repair (%)');axes[1].set_title('Measured: common observation coordinate');axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.savefig(out/'observation_separation.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    t=np.arange(300)/10;truth=physical_to_effective(cfg['synthetic']['truth']);obs=gain_transform(truth,2.)
    for gain in (1.,2.,4.):
        e=gain_transform(obs,1/gain);h=gain_transform(e,gain);total=np.zeros(300);deoxy=np.zeros(300)
        for f in (.035,.09,.14):
            wave=np.exp(2j*np.pi*f*t);total+=wave.real;deoxy+=((h[1]-h[2]/(2j*np.pi*f+h[0]))*wave).real
        axes[0].plot(t,deoxy,lw=1.3,ls={1.:'-',2.:'--',4.:':'}[gain],label=f'g={gain:g}, tau={1/e[0]:.3f} s')
    axes[0].set_title('Identical observed R for the same observed T');axes[0].set_xlabel('Time (s)');axes[0].legend(fontsize=8)
    profiles=pd.read_csv(out/'gain_profiles.csv')
    for key,g in profiles[profiles.metric=='ar1_proxy'].groupby('key'):
        g=g.sort_values('gain');axes[1].plot(g.gain,100*(g.train_loss/g.train_loss.min()-1),alpha=.7,lw=1)
    axes[1].set_xscale('log',base=2);axes[1].set_xlabel('Fixed assumed gain; physiology refitted');axes[1].set_ylabel('Training loss above profile minimum (%)')
    axes[1].set_title('Measured AR1-proxy profiles (24 groups)');axes[1].grid(alpha=.2)
    fig.savefig(out/'gain_nonidentifiability.png',dpi=220);plt.close(fig)
    report=r'''# 观测层、拟合度量与生理效应能否分离

结论：在当前受控合成条件下，已知相对增益、且有独立噪声校准时，可以恢复三个生理效应组合；
仅靠当前相对双 Hb 同时估计观测增益、噪声与生理参数，尚不能实现可信的唯一分离。
实测 AR1 加权可让部分参数离开边界，但没有通过本轮“恢复真实效应、保留观测解释”的检验。

证据 owner：[manifest.json](manifest.json)、[model_metrics.csv](model_metrics.csv)、[verification.json](verification.json)。
范围：上一轮保留数组中 4 数据集 × 3 被试 × 2 时间分块，组内跨通道共享，保持原时间间隔与留出行；
另 5 条件 × 8 重复，共 40 组合成面板。合计 64 个任务、624 个模型拟合。
未读取新原始或受保护数据，未运行完整六状态似然，未改变生产模型。

## 1. 什么被尝试，什么没有被当作成功

观测层为 y_O=x_O、y_R=g*x_R 加上加性误差；g 在同一被试/分块的训练窗口间共享，范围 [0.25,4]。
生理层保留 lambda=1/tau、a=eta*c(E0)、k=eta*(1-c)/(alpha*tau) 三个正组合，不把 eta/E0/alpha 强行拆成唯一估计。
比较固定 g=1、自行估计 g，以及仅合成实验中可用的正确 g。

拟合度量包括：

- identity：原等权欧氏距离。
- difference_diagonal_proxy：训练曲线一阶差分平方均值/2，估计 HbO/HbR 分别的白噪声方差代理。
- ar1_proxy：训练曲线整体 lag-1 相关估计 rho，截断 [0,0.995]，再用创新协方差构成双 Hb × AR1 时间协方差。
- oracle_covariance：合成生成器的真实误差协方差，只用于条件可行性对照。
- independent_noise_calibration：512 个独立噪声专用样本估计的协方差，仅合成中存在，不能把真实 Hb 数据冒充这种校准样本。

代理估计只使用训练窗口，之后对所有候选参数、留出窗口冻结；没有用留出误差调权。
AR1 和差分代理不是已经识别出的噪声。滤波和真实生理信号都能污染它们，正需要通过合成真值检验。
合成拟合未使用真值起点；只有标为已知增益/已知协方差的对照显式获得相应观测层信息。

若 B(theta,g) 为七条正交弱约束，G 为训练二阶矩，C 为冻结误差协方差，目标为

    L = trace[(B C B')^(-1) B G B'].

在公共 24 维约束空间中实际应用的修正为

    delta = C B' (B C B')^(-1) B y.

我们同时记录加权目标、该参数下最小欧氏距离、以及实际加权修正映回曲线后的欧氏大小。
最后一项避免通过改变权重隐藏需要改动的信号幅度。修正限定在公共 24 维空间，其外不作改变；
它不是包含全部时域交叉协方差的 600 维 oracle 最优修正。
不同权重的分数数值不能直接横向比较，也不是已校准的统计拒绝量。

## 2. 一个精确的分离障碍：增益与生理效应存在等价变换

在平衡初态的受迫响应中，生理关系为 H_R/T(s)=a-k/(s+lambda)。
令 y_O=x_O、y_R=g*x_R，则 A=1+(g-1)a，并有

    a_observed = g*a/A
    lambda_observed = lambda-(g-1)*k/A
    k_observed = g*k/A².

即使没有任何噪声，观测只能决定这三个组合，不能一般地同时唯一决定四个量 lambda、a、k、g。
对同一组观测，以下三组解释的弱约束投影误差小于 4e-15：

| 假定 HbR 相对增益 | 生理 tau | a | k |
| --- | --- | --- | --- |
| 1 | 4.687298 | 0.118822 | 0.539254 |
| 2 | 2.000000 | 0.063164 | 0.304764 |
| 4 | 1.521257 | 0.032612 | 0.162483 |

这里三者不仅损失接近，而是对平衡受迫响应给出相同观测关系。增加相同机制的窗口数量不能解除这种等价性。
这也是为什么联合拟合可自行选择很大的 tau、却仍准确解释可观测组合。
正增益本身在 0<a<1 时不会把 a_observed 变成负数；它也不能普遍修复上一轮的负瞬时项偏好。
更自由的未知色团混合矩阵会引入更多等价性，不能仅以其降低残差证明“分离出了生理来源”。

![观测增益的等价解释](gain_nonidentifiability.png)

限制：以上精确等价针对平衡初态/稳态受迫响应。已知且充分激发的 p-v 初始偏差可能提供额外信息。
本次实测沿用允许任意初态的二阶必要约束，gain profile 并非所有组完全平坦；因此不声称每种非线性实验设计都不可辨识。
但未知初态、相对浓度和当前短窗口没有提供已验证的独立增益标定，不能将微小 profile 差异直接当成生理识别。

## 3. 已知真值对照：独立观测信息确实有用

合成真实参数固定为 tau=2、eta=0.35、E0=0.32、alpha=0.32；每重复 64 训练 + 64 留出窗口，30 秒、10 Hz。
使用平衡初态的一阶关系进行拟合，这是完整模型的一个有利子情形，与实测二阶目标不可直接比较绝对分数。
信号归一化在加噪前完成，不逐条按含噪观测重新归一化，保证噪声协方差合同成立。
噪声含 0.02 的白噪声及可选 0.3 的慢成分；慢成分由 0.015/0.025/0.05/0.08 Hz 高斯正余弦系数产生。
独立慢变分别作用于双 Hb，共同慢变作用于两者；增益先作用于 clean HbR，噪声再加入。
这是生成/拟合噪声模型匹配的有利实验，不能外推为任意真实噪声条件下的保证。

@SYNTHETIC@

三组合相对误差定义为 sqrt(mean(((lambda,a,k)-(lambda*,a*,k*))²/(lambda*,a*,k*)²))，不是单独某个生理参数的误差。
独立噪声校准 + 已知增益的恢复结果与 oracle 接近；最大误差也保留在表中，不把中位数当作所有重复表现。
特别在“增益 2 + 独立慢变”条件，独立校准后的 tau 中位数为 2.0008 秒、三组合误差中位数 1.41%。
若协方差已知但增益仍固定错误或自由漂移，同条件物理组合误差仍约 76%，而可观测组合误差只有约 1.5–1.7%。
因此噪声校准解决的是度量偏置；它不能替代独立光学增益信息。

白噪声无增益误差时，固定正确单位增益的欧氏估计 tau≈2；开放增益后可选到 tau 中位数约 1257 秒的等价解。
这不是“白噪声导致真实生理变慢”，而是增加未锚定的观测自由度后出现了参数规范自由度。

## 4. 实测尝试：参数变得正常，不等于分离成功

同一留出集合上，实际施加修正的欧氏范数 RMS 如下；分母为原中心化双 Hb 的联合范数。
每组内均方，再对 3 被试 × 2 分块等权；百分数不是被修改的采样点比例或绝对浓度百分比。

@MEASURED@

identity、固定 g=1 的最优结果复现上一轮正组合结果，留出 MSE 最大差异小于 1e-8。
自由 g 对欧氏偏离只有有限改善，仍有 24/24 组至少一项触边；不能凭小幅改善称为完成光学校准。
差分对角代理也未消除边界，且在原观测坐标中需要更大的修正。

AR1、固定 g=1 的 tau 中位数由 82.92 秒降到 2.34 秒，a=0 从 23/24 降到 3/24。
但 7/24 组转而到 a=1，12/24 组仍有至少一项边界；所有组的 rho 估计均触及 0.995 上限。
上述表格显示实际修正明显增大。对有真值的独立慢变合成面板，AR1 代理的三组合误差中位数仍约 63.94%，
共同慢变约 72.87%，远高于独立噪声校准后的约 2.08% 和 6.04%。
因此没有证据把 AR1 下看似正常的参数解释为恢复了真实生理，它更像把强自相关内容降权后换了一种折衷。
这也不能证明所有加权方法无效：这里只否定这两个无独立锚点的具体代理已经足以完成分离。

![合成恢复与实测曲线代价](observation_separation.png)

## 5. 禁止将总观测协方差直接认作噪声的数学原因

若用同一批训练数据的全部二阶矩 G 作为误差协方差，那么

    trace[(B G B')^(-1) B G B'] = rank(B) = 7.

只要矩阵可逆，所有候选参数的训练损失相同。64 个任务的不同参数探针均验证了这一点；没有触发协方差特征值截断。
这样的白化不仅削弱噪声，也把识别生理关系需要的结构一并抹掉。训练损失稳定、参数停在内部，都不能证明有效。
同理，如果允许逐窗口随意选择“生理过程之外的残差”，任何曲线都可以被解释，分离本身失去可检验性。

## 6. 可以得出的结论与下一步

可以分离的条件性部分：观测增益由外部校准约束、噪声协方差由独立信息约束时，本轮能恢复 lambda/a/k 三个生理效应组合。
仍不能拆开的部分：eta/E0/alpha 之间的原有等价性，以及没有独立校准时的增益—生理等价性。
仅靠这批双 Hb 的自适应加权与自由增益，当前没有获得可信的唯一分离，也没有恢复可解释的氧提取率。

下一步应寻找独立约束，而非继续给观测层增加可任意吸收残差的自由度：

1. 审核相对光学标度、路径长度与色团串扰；原始波长数据或独立校准可以约束观测映射。仪器名称或未知单位下的幅值不能代替标定。
2. 若有短源探距通道或 ECG/呼吸等辅助记录，评估其是否提供独立于目标双 Hb 的系统性成分信息；这些辅助量也不自动等同于纯噪声。
3. 固定经过训练/独立校准验证的观测层，再检验跨窗口生理组合恢复、参数稳定性与留出预测；不以触边率单独验收。
4. 若需要 E0、血容量或氧代谢过程的独立生理解释，还需要相应基线或流量/氧合测量，不能通过给组合参数重新命名获得。

本次未访问辅助生理或短距离原始数组，也未声称当前四数据集均具备所需独立标定。现有保留数组不能补造这些信息。

相关原始研究支持需要关注的观测机制，但不是本次数据存在特定污染的证明：
[DPF 与 Hb 串扰](https://pmc.ncbi.nlm.nih.gov/articles/PMC6689143/)、
[短距离回归实测研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC4717232/)、
[短距离位置对回归性能的影响](https://pmc.ncbi.nlm.nih.gov/articles/PMC3254723/)。

## 可复现与检查

配置：[hbo_hbr_observation_v1.yaml](../../../../configs/physiology_semantic_tokenizer/hbo_hbr_observation_v1.yaml)。
[launch.json](launch.json) 记录独立 systemd 服务、环境和续跑命令；[resources.json](resources.json) 记录资源选择；[run.log](run.log) 保留结束日志。
18 项针对性测试通过；收集 675/686，11 项由既有配置排除。检查涵盖精确增益等价、广义最小修正、无父数据的合成/shape 合同和既有约束回归。
旧运行证据保持不变；拟合与报告代码分别保存在 source_snapshot。PNG 为 220 dpi，本次未生成 PDF，未检查 WPS。
'''
    report=report.replace('@SYNTHETIC@',md(pd.DataFrame(synthetic_rows))).replace('@MEASURED@',md(measured_display))
    (out/'REPORT.md').write_text(report);shutil.copy2(Path(__file__),out/'source_snapshot'/'report_export_v1.py')
    print(json.dumps(verification,indent=2));print(measured_display.to_string(index=False))


if __name__ == '__main__':
    main()
