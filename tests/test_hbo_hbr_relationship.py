"""Array-only checks for the descriptive Hb dynamic-constraint audit."""
import numpy as np
import pytest

from experiments.scripts.analyze_hbo_hbr_relationship import (
    dynamic_constraint_basis, dynamic_distance, dynamics_checks, dynamics_metrics,
    energy_decomposition, pointwise_terms, term_checks, term_metrics,
    ADAPT_CONFIG, adaptation_checks, assign_adaptation_blocks,
    compact_constraint_basis, compact_constraint_components, conditional_curve_repairs,
    dynamic_constraint_operators, load_adaptation_config,
    equivalent_parameter_ranges,
    smooth_correction_space,
    physical_to_effective, effective_constraint_basis, constraint_objective,
    compact_smooth_coordinates,
    gain_transform, observation_constraint_basis, weighted_constraint_objective,
    observation_checks,
)


def test_independent_ode_with_unequal_initial_total_hb_and_volume():
    result = dynamics_checks()
    assert result['matched_linear_distance'] < 1e-5
    assert result['reversed_matched_distance'] > .1
    assert result['injected_deoxy_distance'] > .1
    assert result['observability_ranks_by_derivative_order'] == [2, 4, 5, 6]
    assert result['fault_identity_error'] < 1e-12


def test_projection_is_minimum_curve_change_and_offset_invariant():
    rng = np.random.default_rng(318)
    y = rng.normal(size=(2, 300))
    distance, correction = dynamic_distance(y)
    z = y-y.mean(axis=1, keepdims=True)
    assert np.isclose(np.linalg.norm(correction)/np.linalg.norm(z), distance)
    basis = dynamic_constraint_basis()
    assert np.max(abs(basis@(y-correction).ravel())) < 1e-12
    # Every other feasible correction has at least the orthogonal projection norm.
    tangent = rng.normal(size=600)
    tangent -= basis.T@(basis@tangent)
    assert np.linalg.norm(correction.ravel()+tangent) >= np.linalg.norm(correction)
    assert np.isclose(dynamic_distance(3*y+np.array([[8.], [-4.]]))[0], distance)


def test_missing_interior_support_is_not_interpolated_into_dynamics():
    t = np.arange(300)/10
    y = np.array([np.sin(t), np.cos(t)])
    valid = np.ones_like(y, dtype=bool)
    valid[1, 170] = False
    result = dynamics_metrics(y, valid)
    assert result['status'] == 'incomplete_dynamic_support'
    assert 'dynamic_distance' not in result


def test_more_nested_weak_constraints_cannot_reduce_distance():
    rng = np.random.default_rng(55)
    y = rng.normal(size=(2, 300))
    distances = [dynamic_distance(y, order=k)[0] for k in (6, 8, 10)]
    assert np.all(np.diff(distances) >= -1e-12)


def test_term_derivatives_against_matched_analytic_multifrequency_response():
    result = term_checks()
    assert result['analytic_derivative_max_normalized_error'] < .002
    assert result['matched_J_rms'] < .001
    assert min(result['matched_term_rms']) > .2


def test_signed_energy_allocation_preserves_cancellation():
    x = np.sin(np.arange(100)/10)
    terms = np.array([x, -.8*x, .1*x])
    gram, energy, alignment = energy_decomposition(terms)
    assert np.allclose(alignment/energy, [10/3, -8/3, 1/3])
    assert np.isclose(gram.sum(), energy)
    assert np.isclose(alignment.sum(), energy)


def test_missing_values_do_not_enter_term_distributions():
    values = np.array([np.sin(np.arange(300)/20), np.cos(np.arange(300)/20)])
    valid = np.ones_like(values, dtype=bool)
    valid[:, 100] = False
    result, traces = term_metrics(values, valid)
    assert result['status'] == 'incomplete_dynamic_support'
    assert traces is None


def test_shared_parameter_adaptation_generalizes_to_unseen_synthetic_phases():
    result = adaptation_checks(load_adaptation_config(ADAPT_CONFIG))
    assert result['fixed_heldout_mean_distance'] > .1
    assert result['adapted_heldout_mean_distance'] < 1e-4


def test_compressed_projection_matches_full_operator_across_parameter_box():
    coordinates, _ = compact_constraint_components()
    for params in ((2., .35, .32, .32), (.25, .05, .8, .1), (10., .8, .05, .8)):
        basis = compact_constraint_basis(params)@coordinates
        full = dynamic_constraint_operators(parameters=params)[0]
        assert np.allclose(basis.T@basis, full.T@full, atol=1e-10)


def test_chronological_split_has_gap_and_is_signal_independent():
    inventory = [dict(dataset_id='fixture', subject='s1', record='r1', fnirs_start_s=30*i)
                 for i in range(8)]
    blocks = assign_adaptation_blocks(inventory)
    assert list(blocks) == ['early']*3+['gap']*2+['late']*3
    assert list(assign_adaptation_blocks(inventory[:3])) == ['early','gap','late']
    with pytest.raises(ValueError, match='three windows'):
        assign_adaptation_blocks(inventory[:2])


def test_conditional_repairs_preserve_declared_physiological_coordinates():
    rng = np.random.default_rng(998)
    y = rng.normal(size=(2, 300))
    y -= y.mean(axis=1, keepdims=True)
    y /= np.linalg.norm(y)
    params = (1.2, .4, .25, .3)
    joint, exchange, total = conditional_curve_repairs(y, params)
    assert np.allclose(exchange.sum(axis=0), 0)
    assert np.allclose(total[1]-.4*total.sum(axis=0), 0)
    assert np.linalg.norm(exchange) >= np.linalg.norm(joint)
    assert np.linalg.norm(total) >= np.linalg.norm(joint)


def test_equivalent_physiology_is_reported_as_a_range_not_a_unique_estimate():
    cfg = load_adaptation_config(ADAPT_CONFIG)
    ranges = equivalent_parameter_ranges((.85,.5,.45,.24), cfg['parameter_boxes']['primary'])
    assert ranges['E0_equivalent_min'] < .45 < ranges['E0_equivalent_max']


def test_smooth_repairs_satisfy_constraints_without_quadrature_checkerboard():
    rng = np.random.default_rng(903)
    y = rng.normal(size=(2, 300))
    y -= y.mean(axis=1, keepdims=True)
    y /= np.linalg.norm(y)
    params = (2., .35, .32, .32)
    raw = conditional_curve_repairs(y, params)
    smooth = conditional_curve_repairs(y, params, max_cosine_mode=12)
    _, paired = smooth_correction_space(12)
    for old, new in zip(raw, smooth):
        assert np.linalg.norm(new) >= np.linalg.norm(old)-1e-10
        assert np.allclose(paired.T@paired@new.ravel(),new.ravel(),atol=1e-12)


def test_effective_coordinates_preserve_constraint_and_transfer_function():
    from experiments.scripts.analyze_hbo_hbr_relationship import physical_combinations
    for physical in ((2.,.35,.32,.32),(.25,.05,.8,.1),(10.,.8,.05,.8)):
        effective=physical_to_effective(physical)
        old=compact_constraint_basis(physical);new=effective_constraint_basis(effective)
        assert np.allclose(old.T@old,new.T@new,atol=1e-11)
        tau,a,d=physical_combinations(physical);lam,a2,k=effective
        z=2j*np.pi*np.array([.01,.1,.2])
        assert np.allclose((a*tau*z+d)/(tau*z+1),a2-k/(z+lam))


def test_effective_limits_and_smooth_minimum_are_well_defined():
    rng=np.random.default_rng(559)
    curves=rng.normal(size=(8,24));gram=curves.T@curves/len(curves)
    for effective in ((0.,0.,0.),(0.,-.4,.2),(4.,1.,40.),(.5,.1,-.2)):
        basis=effective_constraint_basis(effective)
        assert np.allclose(basis@basis.T,np.eye(7),atol=1e-11)
        delta=np.linalg.lstsq(basis@compact_smooth_coordinates(),basis@curves.T,rcond=1e-10)[0]
        assert np.allclose(basis@compact_smooth_coordinates()@delta,basis@curves.T)
        assert np.isclose(constraint_objective(basis,gram,smooth=True),np.mean(np.sum(delta**2,axis=0)))
        assert constraint_objective(basis,gram,smooth=True)>=constraint_objective(basis,gram)-1e-10


def test_unknown_observation_gain_can_exactly_alias_a_different_physiology():
    true=physical_to_effective((2.,.35,.32,.32));alias=gain_transform(true,2.)
    left=observation_constraint_basis([*true,2.],first_order=True)
    right=observation_constraint_basis([*alias,1.],first_order=True)
    assert np.allclose(left.T@left,right.T@right,atol=1e-11)
    assert 1/alias[0]>4.6
    assert np.allclose(gain_transform(alias,.5),true)


def test_weighted_repair_closes_constraint_and_minimizes_declared_metric():
    rng=np.random.default_rng(882);z=rng.normal(size=24);a=rng.normal(size=(24,24));cov=a@a.T+np.eye(24)
    b=observation_constraint_basis([.5,.1,.3,1.4])
    correction=cov@b.T@np.linalg.solve(b@cov@b.T,b@z)
    assert np.allclose(b@(z-correction),0,atol=1e-11)
    loss=correction@np.linalg.solve(cov,correction)
    assert np.isclose(loss,weighted_constraint_objective(b,np.outer(z,z),cov))
    tangent=rng.normal(size=24);tangent-=b.T@(b@tangent)
    assert (correction+tangent)@np.linalg.solve(cov,correction+tangent)>=loss


def test_observation_contract_and_synthetic_checks_do_not_read_parent_data():
    import yaml
    cfg=yaml.safe_load(ADAPT_CONFIG.with_name('hbo_hbr_observation_v1.yaml').read_text())
    cfg['parent']='missing_nonexistent_parent'
    result=observation_checks(cfg)
    assert result['gain_alias_projector_error']<1e-11
    assert result['identical_covariance_score']==7
    cfg['tensor']['samples']=299
    with pytest.raises(AssertionError):observation_checks(cfg)
