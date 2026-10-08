"""Independent observations, exact profiles and finite-candidate contracts."""
from copy import deepcopy

import numpy as np
import pytest

from src.inference.shared_driver_attribution import (
    fit_conditioned_shared_driver, equal_capacity_hb_basis,
    fit_reduced_rank_readout, predict_reduced_rank_readout, grouped_component_overlap,
)
from src.inference.shared_driver_reconstruction import nonlinear_driver_forward
from src.inference.t3a_balloon_robust_ssm import BalloonParameters


def test_reduced_rank_ridge_solves_penalized_objective_and_freezes_scale():
    rng = np.random.default_rng(58)
    x = rng.normal(size=(80,6))*np.arange(1,7)
    y = x@rng.normal(size=(6,2))+.1*rng.normal(size=(80,2))
    model = fit_reduced_rank_readout(x,y,rank=2,weight=.1)
    z = x/model['scale']
    expected = np.linalg.solve(z.T@z+8*np.eye(6),z.T@y)
    np.testing.assert_allclose(model['coefficient'],expected,atol=1e-12)
    small = fit_reduced_rank_readout(x,y,rank=1,weight=.1)
    assert np.linalg.matrix_rank(small['coefficient']) == 1
    def objective(b):
        return np.sum((y-z@b)**2)+8*np.sum(b*b)
    u,s,vt = np.linalg.svd(expected,full_matrices=False)
    naive = (u[:,:1]*s[:1])@vt[:1]
    assert objective(small['coefficient']) <= objective(naive)+1e-10
    saved = deepcopy(small)
    prediction = predict_reduced_rank_readout(small,x+3)
    assert prediction.shape == (80,2)
    for key in small:
        np.testing.assert_array_equal(small[key],saved[key])
    with pytest.raises(ValueError):
        predict_reduced_rank_readout(small,x[:,:4])


def test_conditioned_profile_hidden_values_and_known_units_are_invariant():
    n = 24
    p = BalloonParameters()
    driver = .01*np.sin(np.arange(n)*.2)
    truth = nonlinear_driver_forward(driver,[0.,1.,1.,1.,1.],p,.25,substeps=4,derivative=False)
    basis = equal_capacity_hb_basis(np.eye(3*n),modes=2)
    mean = (basis@np.array([.003,-.001])).reshape(n,3)
    y = truth['canonical_prediction']+mean
    mask = np.ones_like(y,bool); mask[8:16,1:] = False
    options = dict(mean_operator=np.eye(3*n),sd=np.array([.01,.003,.001]),visible=mask,
        component_mean=mean,observation_basis=basis,observation_coefficient_sd=.02,
        penalty=.01,initial_penalty=100.,max_evaluations=120,gradient_tolerance=1e-5,
        numerical_backend='numba',return_jacobian=True)
    fit = fit_conditioned_shared_driver(y,p,.25,**options)
    assert fit['converged']
    hidden = y.copy(); hidden[~mask] = np.nan
    other = fit_conditioned_shared_driver(hidden,p,.25,**options)
    np.testing.assert_array_equal(fit['prediction'],other['prediction'])
    np.testing.assert_array_equal(fit['residual_jacobian'],other['residual_jacobian'])
    np.testing.assert_allclose(fit['prediction'],fit['physical_prediction']+fit['observation_component'])
    unit = np.array([1000.,.001,.001])
    converted = dict(options,sd=options['sd']*unit,mean_operator=np.diag(np.tile(unit,n)),
                     component_mean=mean*unit,observation_basis=np.tile(unit,n)[:,None]*basis)
    transformed = fit_conditioned_shared_driver(y*unit,p,.25,**converted)
    assert transformed['converged']
    np.testing.assert_allclose(transformed['prediction']/unit,fit['prediction'],atol=1e-10,rtol=1e-8)
    np.testing.assert_allclose(transformed['driver'],fit['driver'],atol=1e-10,rtol=1e-8)
    # For an independent constant mean the profiled objective and derivatives
    # must equal the old zero-mean solver applied to visible y-mean.
    shifted = dict(options,component_mean=None)
    zero = fit_conditioned_shared_driver(y-mean,p,.25,**shifted)
    np.testing.assert_array_equal(zero['residual_jacobian'],fit['residual_jacobian'])
    assert zero['objective'] == fit['objective']


def test_conditional_mean_does_not_modify_EEG_or_hide_missing_donors():
    y = np.zeros((24,3)); invalid = y.copy(); invalid[0,0]=1.
    with pytest.raises(ValueError,match='zero EEG'):
        fit_conditioned_shared_driver(y,BalloonParameters(),.25,component_mean=invalid)
    invalid[:] = np.nan
    with pytest.raises(ValueError,match='finite'):
        fit_conditioned_shared_driver(y,BalloonParameters(),.25,component_mean=invalid)


def test_data_overlap_is_not_confused_with_prior_stability():
    j = np.eye(3)[:,:2]
    b = j[:,:1]
    result = grouped_component_overlap(j,b,groups={'driver':[0],'initial':[1]},
        physical_penalty=10*np.eye(2),component_penalty=np.eye(1))
    assert result['driver']['data']['cosines'][0] == pytest.approx(1.)
    assert result['driver']['objective']['cosines'][0] < .1
    assert result['initial']['data']['cosines'][0] == pytest.approx(0.)
    assert result['driver']['data']['rank'] == 1
