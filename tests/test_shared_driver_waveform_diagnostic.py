from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from src.inference.t3a_balloon_robust_ssm import BalloonParameters
from src.inference.shared_driver_reconstruction import (
    build_shared_driver_design, fit_waveform_subspace, waveform_component_basis,
)
from experiments.scripts import evaluate_shared_driver_reconstruction as run


def test_viscoelastic_rest_design_against_independent_mass_balance_ode():
    p=BalloonParameters();n=24;dt=.1;viscosity=2.3
    d=build_shared_driver_design(p,n,dt,venous_viscoelastic_s=viscosity)
    initial=np.array([.2,-.1,.3,-.15,.2])*1e-4;r=np.full(n,3e-5)
    tau=p.free.tau;a=p.fixed.alpha;e=p.fixed.E0
    def rhs(t,x):
        s,f,v,h,q=x;out=(tau*v**(1/a)+viscosity*f)/(tau+viscosity)
        return [r[0]-p.free.kappa*s-p.fixed.gamma*(f-1),s,(f-out)/tau,
                (f-out*h/v)/tau,(f*(1-(1-e)**(1/f))/e-out*q/v)/tau]
    physical=initial+np.array([0,1,1,1,1])
    solution=solve_ivp(rhs,[0,(n-1)*dt],physical,t_eval=np.arange(n)*dt,rtol=1e-11,atol=1e-13).y.T
    linear=np.einsum('tij,j->ti',d.state_design,np.r_[r,initial])+[0,1,1,1,1]
    np.testing.assert_allclose(linear,solution,atol=2e-8,rtol=0)
    np.testing.assert_array_equal(build_shared_driver_design(p,n,dt).observation_design,
                                  build_shared_driver_design(p,n,dt,venous_viscoelastic_s=0).observation_design)
    with pytest.raises(ValueError):build_shared_driver_design(p,n,dt,venous_viscoelastic_s=-1)


def test_slow_mechanism_mass_balance_and_equal_dimensions():
    p=BalloonParameters();n=40;op=np.eye(3*n)
    volume=waveform_component_basis(p,n,.25,op,'volume').reshape(n,3,-1)
    exchange=waveform_component_basis(p,n,.25,op,'exchange').reshape(n,3,-1)
    assert volume.shape==exchange.shape==(n,3,6)
    np.testing.assert_array_equal(volume[:,0],0)
    np.testing.assert_allclose(volume[:,1]*p.fixed.Q0,volume[:,2]*(p.fixed.P0-p.fixed.Q0))
    np.testing.assert_allclose(exchange[:,1]+exchange[:,2],0)


def test_batch_projection_and_hidden_values_cannot_change_fit():
    p=BalloonParameters();n=40;d=build_shared_driver_design(p,n,.25)
    rng=np.random.default_rng(19);y=rng.normal(size=(3,n,3));sd=np.array([1.,2.,3.])
    mask=np.ones((n,3),bool);mask[10:20,1:]=False
    a=fit_waveform_subspace(y,d,sd,visible=mask)
    changed=y.copy();changed[:,~mask]=1e6
    b=fit_waveform_subspace(changed,d,sd,visible=mask)
    np.testing.assert_array_equal(a['prediction'],b['prediction'])
    for i in range(3):
        single=fit_waveform_subspace(y[i:i+1],d,sd,visible=mask)
        np.testing.assert_allclose(a['prediction'][i],single['prediction'][0],atol=1e-7)
    for bad in [np.ones((n,3)),np.zeros((n,3),bool)]:
        with pytest.raises(ValueError):fit_waveform_subspace(y,d,sd,visible=bad)


def test_known_tau_is_selected_from_training_while_validation_can_change():
    p=BalloonParameters();n=80;rng=np.random.default_rng(2)
    truth=build_shared_driver_design(replace(p,free=replace(p.free,tau=4.)),n,.25)
    x=rng.normal(size=(n+5,3))*.02
    y=(truth.observation_design@x).T.reshape(3,n,3)
    selected=[]
    for altered in [y,y.copy()]:
        if altered is not y:altered[-1]=rng.normal(size=(n,3))*100
        losses=[]
        for tau in [1.,2.,4.,8.]:
            d=build_shared_driver_design(replace(p,free=replace(p.free,tau=tau)),n,.25)
            fit=fit_waveform_subspace(altered,d,np.ones(3))
            losses.append(np.sum(fit['visible_sse'][:2]))
        selected.append([1.,2.,4.,8.][np.argmin(losses)])
    assert selected==[4.,4.]


def test_contract_and_check_only_do_not_open_arrays(monkeypatch,capsys,tmp_path):
    path=run.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_waveform_diagnostic_v1.yaml'
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('array access'))
    run.waveform_diagnostic_main(SimpleNamespace(config=path,check_only=True))
    assert '"measured_arrays_read": 0' in capsys.readouterr().out
    cfg=run.read_waveform_config(path)
    with pytest.raises((ValueError,FileNotFoundError)):
        run.waveform_parent(cfg,tmp_path,'mne_tddr__subject_29_o3')


def test_volume_fraction_preserves_total_and_does_not_change_balloon():
    p=BalloonParameters();n=40;op=np.eye(3*n)
    a=waveform_component_basis(p,n,.25,op,'volume',volume_deoxy_fraction=.1).reshape(n,3,6)
    b=waveform_component_basis(p,n,.25,op,'volume',volume_deoxy_fraction=.35).reshape(n,3,6)
    np.testing.assert_allclose(a[:,1]+a[:,2],b[:,1]+b[:,2],atol=1e-15)
    np.testing.assert_allclose(a[:,2]/.1,b[:,2]/.35,atol=1e-15)
    for bad in [0,1,np.nan]:
        with pytest.raises(ValueError):waveform_component_basis(p,n,.25,op,'volume',volume_deoxy_fraction=bad)


def test_default_slow_flow_hbr_direction_matches_equilibrium_derivative():
    p=BalloonParameters();e=p.fixed.E0;alpha=p.fixed.alpha
    c=1+(1-e)*np.log(1-e)/e
    predicted=1-(1-c)/alpha
    eps=1e-6
    def q_of_f(f):return f**alpha*(1-(1-e)**(1/f))/e
    measured=(q_of_f(1+eps)-q_of_f(1-eps))/((1+eps)**alpha-(1-eps)**alpha)
    assert predicted<0
    np.testing.assert_allclose(predicted,measured,rtol=1e-8)


def test_fine_domain_failure_keeps_optimizer_and_coarse_evidence(tmp_path,monkeypatch):
    from src.inference import shared_driver_reconstruction as lib
    cfg=run.read_waveform_config(run.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_waveform_diagnostic_v1.yaml')
    group='mne_tddr__subject_09_o3';parent=tmp_path/'parent';cell=parent/'cells'/f'{group}__conditional_trained_gain__full';cell.mkdir(parents=True)
    states=np.tile([0.,0.,1.,1.,1.,1.],(120,1));target=np.zeros((24,120,3))
    np.savez(cell/'trajectories.npz',trial_indices=[3],states=states[None],prediction=target[3:4],target=target[3:4])
    monkeypatch.setattr(run,'waveform_parent',lambda *a:(parent,{},target,np.ones(3),BalloonParameters(),np.eye(360)))
    monkeypatch.setattr(lib,'fit_nonlinear_shared_driver',lambda *a,**k:dict(status='failed_numerical',converged=False,prediction=target[3],driver=np.zeros(120),initial_state=states[0,1:],states=states))
    def fail(*a,**k):raise FloatingPointError('fine domain')
    monkeypatch.setattr(lib,'nonlinear_driver_forward',fail)
    out=tmp_path/'new'
    r=run.waveform_nonlinear_case((cfg,out,tmp_path,group,'Hb_only_no_penalties'))
    assert r['status']=='failed_integration_check' and r['optimization_status']=='failed_numerical'
    assert r['integration_max_difference_training_sd'] is None
    assert (out/'nonlinear'/f'{group}__Hb_only_no_penalties'/'trajectory.npz').exists()


def test_extra_volume_loading_selection_ignores_validation_targets():
    cfg=run.read_waveform_config(run.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_waveform_volume_fraction_v1.yaml')
    p=BalloonParameters();n=120;op=np.eye(360);rng=np.random.default_rng(43)
    d=build_shared_driver_design(p,n,.25,processed_mean_operator=op)
    basis=waveform_component_basis(p,n,.25,op,'volume',volume_deoxy_fraction=.2)
    y=(d.observation_design@(rng.normal(size=(125,24))*.004)+basis@(rng.normal(size=(6,24))*.01)).T.reshape(24,120,3)
    train=list(range(18));valid=list(range(18,24));sd=np.std(y[:18],axis=(0,1))
    a,_=run.waveform_volume_fit(cfg,y,sd,p,op,train,valid)
    y[18:]=rng.normal(size=(6,120,3))*1000
    b,_=run.waveform_volume_fit(cfg,y,sd,p,op,train,valid)
    assert a['selected_fraction']==b['selected_fraction']==.2
    np.testing.assert_array_equal([v['training_sse'] for v in a['profiles']],[v['training_sse'] for v in b['profiles']])
