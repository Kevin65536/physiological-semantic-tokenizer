from copy import deepcopy
import numpy as np
import pytest
from scipy.integrate import solve_ivp, cumulative_trapezoid

from experiments.scripts import evaluate_shared_driver_reconstruction as run
from src.inference.observation_baselines import disjoint_native_bin_operator
from src.inference.shared_driver_reconstruction import build_shared_driver_design
from src.inference.t3a_balloon_robust_ssm import balloon_rhs, physical_to_transformed


@pytest.fixture
def cfg():
    return run.volume_drive_config(run.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_volume_drive_v1.yaml')


def test_contract_rejects_raw_and_changed_scope_before_arrays(cfg,tmp_path,monkeypatch):
    import yaml
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('unexpected array read'))
    for key in ('raw_reads','labels_as_inputs'):
        changed=deepcopy(cfg);changed['data'][key]=True
        path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(changed))
        with pytest.raises(ValueError,match='boundary'):run.volume_drive_config(path)
    changed=deepcopy(cfg);changed['parents'][0]='data/protected'
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError,match='boundary'):run.volume_drive_config(path)


@pytest.mark.parametrize('offset',[-.08,.08])
def test_delta_is_an_unforced_decaying_mode_even_with_oscillating_neural_input(cfg,offset):
    p=run.volume_drive_parameters(cfg)
    def rhs(t,z):
        z=z.copy();z[0]=.1*np.sin(.8*t)
        d=balloon_rhs(z,p);d[0]=0
        return d
    initial=np.array([0.,.02,1.04,.97,.97+offset,1.01])
    time=np.linspace(0,25,10001)
    sol=solve_ivp(rhs,[0,25],physical_to_transformed(initial),t_eval=time,rtol=1e-11,atol=1e-13)
    volume=np.exp(sol.y[3]);total=np.exp(sol.y[4]);delta=total-volume
    expected=(initial[4]-initial[3])*np.exp(-cumulative_trapezoid(volume**(1/p.fixed.alpha-1)/p.free.tau,time,initial=0))
    np.testing.assert_allclose(delta,expected,atol=1e-9,rtol=2e-6)
    assert np.max(np.diff(np.abs(delta)))<1e-10
    # Direct mass-balance identity, away from the linearization point.
    d=balloon_rhs(physical_to_transformed(initial),p)
    assert initial[4]*d[4]-initial[3]*d[3]==pytest.approx(-initial[3]**(1/p.fixed.alpha-1)*(initial[4]-initial[3])/p.free.tau)


@pytest.mark.parametrize('length',[30,90])
def test_crop_preserves_native_bins_and_removes_only_reference(length):
    ops=disjoint_native_bin_operator(np.arange(900)/10.)
    rng=np.random.default_rng(5);canonical=rng.normal(size=(360,3))
    original=np.column_stack((ops['eeg']@canonical[:,0],ops['model']@canonical[:,1:]))
    target,operator=run.volume_drive_crop(original,dict(eeg=ops['eeg'],hb=ops['model']),length)
    sl=slice(120,240) if length==30 else slice(0,360)
    prediction=(operator@canonical[sl].ravel()).reshape(-1,3)
    np.testing.assert_allclose(prediction[:-1],target[:-1],atol=1e-14)
    changed=original+np.array([22.,-3.,90.])
    np.testing.assert_allclose(run.volume_drive_crop(changed,dict(eeg=ops['eeg'],hb=ops['model']),length)[0],target,atol=1e-13)
    if length==30:
        contaminated=dict(eeg=ops['eeg'].copy(),hb=ops['model'].copy());contaminated['hb'][150,310]=1
        with pytest.raises(ValueError,match='outside'):run.volume_drive_crop(original,contaminated,length)


@pytest.mark.parametrize('length',[30,90])
def test_component_capacity_norm_mass_balance_and_mask(cfg,length):
    n=int(length/.25);reference=np.eye(n);reference[:,:20]-=1/20
    operator=np.kron(reference,np.eye(3));sd=np.array([.025,.02,.01])
    vol=run.volume_drive_basis(cfg,operator,sd,'volume').reshape(n,3,-1)
    exc=run.volume_drive_basis(cfg,operator,sd,'exchange').reshape(n,3,-1)
    assert vol.shape==exc.shape==(n,3,int(length/5))
    np.testing.assert_allclose(vol[:,1]*.35,vol[:,2]*.65,atol=1e-15)
    np.testing.assert_allclose(exc[:,1]+exc[:,2],0,atol=1e-15)
    np.testing.assert_allclose(np.mean((vol[:-1,1:]/sd[None,1:,None])**2,axis=(0,1)),1)
    np.testing.assert_allclose(np.mean((exc[:-1,1:]/sd[None,1:,None])**2,axis=(0,1)),1)
    mask=run.volume_drive_mask(n,'joint_blocked')
    assert mask[:20].all() and not mask[-1].any()
    assert (~mask[:-1,1]).sum()==40*(length//30)


def test_exact_linear_matched_model_and_extra_direction(cfg):
    from scipy.linalg import lstsq
    n=120;p=run.volume_drive_parameters(cfg);sd=np.array([.025,.025,.015])
    op=np.eye(3*n);design=build_shared_driver_design(p,n,.25)
    rng=np.random.default_rng(4);coef=rng.normal(size=n+5)*.001
    matched=(design.observation_design@coef).reshape(n,3)
    mask=run.volume_drive_mask(n,'Hb_only_full').ravel()
    weights=np.tile(sd,n)[mask]
    x=design.observation_design[mask]/weights[:,None]
    for mechanism in ('volume','exchange'):
        extra=run.volume_drive_basis(cfg,op,sd,mechanism)
        y=matched.ravel()+extra[:,2]
        base_error=np.linalg.norm(x@lstsq(x,y[mask]/weights,cond=1e-10)[0]-y[mask]/weights)
        extended=np.column_stack((x,extra[mask]/weights[:,None]))
        ext_error=np.linalg.norm(extended@lstsq(extended,y[mask]/weights,cond=1e-10)[0]-y[mask]/weights)
        assert base_error>1e-2 and ext_error<1e-7


def test_hidden_scores_and_training_scale_do_not_fit_local_sd():
    rng=np.random.default_rng(19);y=rng.normal(size=(120,3));pred=y.copy()
    mask=run.volume_drive_mask(120,'joint_blocked');pred[~mask]=pred[~mask]+3
    a=run.volume_drive_metrics(y,pred,np.array([1.,2.,2.]),'joint_blocked')
    b=run.volume_drive_metrics(y,pred,np.array([1.,4.,4.]),'joint_blocked')
    assert a['score_hb_nmse']==pytest.approx(2.25)
    assert a['score_hb_nmse']==pytest.approx(b['score_hb_nmse']*4)
    assert a['full_hb_nmse']<a['score_hb_nmse']
