"""Float64 algebra equivalence without measured data or a required GPU."""
import numpy as np
import pytest
from scipy.linalg import cho_factor,cho_solve

from src.inference.shared_driver_reconstruction import (
    _nonlinear_trial_objective, _volume_tied_initial_embedding,
    fit_nonlinear_shared_parameters, nonlinear_driver_forward, torch_normal_steps,
)
from src.inference.t3a_balloon_robust_ssm import BalloonParameters


@pytest.mark.parametrize('steps',[60,130])
def test_structured_reference_and_initial_embedding_preserve_full_objective_and_derivatives(steps):
    rng=np.random.default_rng(41);p=BalloonParameters();dt=.1
    driver=.02*np.sin(np.arange(steps)/17)
    initial=np.array([.01,1.02,1.01,1.01,1.])
    x=np.delete(np.r_[driver,initial[0],np.log(initial[1:])],steps+3)
    y=rng.normal(0,.01,(steps,3));mask=np.ones_like(y,bool);mask[-1]=False;mask[52:55,1]=False
    reference=np.eye(steps);reference[:,:50]-=.02;op=np.kron(reference,np.eye(3))
    regularizer=np.zeros((steps-2,steps+5));regularizer[:,:steps]=np.diff(np.eye(steps),n=2,axis=0)
    kwargs=dict(initial_embedding=_volume_tied_initial_embedding(steps),numerical_backend='numba',
        hemodynamics='tak',viscoelastic_s=2.,initial_coordinate_weights=np.array([1,1,1,0,1]),data_weight=.4)
    dense=_nonlinear_trial_objective(x,p,dt,op,y,mask,np.full_like(y,.025),regularizer,100.,4,True,
        ['tau','neurovascular_gain'],1.,np.log(2),**kwargs)
    fast=_nonlinear_trial_objective(x,p,dt,None,y,mask,np.full_like(y,.025),regularizer,100.,4,True,
        ['tau','neurovascular_gain'],1.,np.log(2),**kwargs,
        observation_reference_steps=50,structured_initial_mode='volume_tied')
    for key in ('prediction','residual','jacobian','parameter_jacobian','initial'):
        np.testing.assert_allclose(fast[key],dense[key],rtol=1e-12,atol=1e-12)
    assert fast['objective']==pytest.approx(dense['objective'],rel=1e-13)


@pytest.mark.parametrize('parameter_count',[0,2])
def test_batched_torch_normal_blocks_match_scipy_with_different_damping(parameter_count):
    rng=np.random.default_rng(5);js=[rng.normal(size=(80,23)) for _ in range(3)]
    rs=[rng.normal(size=80) for _ in js];ps=[rng.normal(size=(80,parameter_count)) for _ in js]
    damping=[1e-3,1.,100.]
    blocks=torch_normal_steps(js,rs,ps,damping,device='cpu')
    for j,r,p,d,(diagonal,g,c,step) in zip(js,rs,ps,damping,blocks):
        h=j.T@j;right=np.column_stack((j.T@r,j.T@p))
        expected=cho_solve(cho_factor(h+d*np.diag(np.maximum(np.diag(h),1.)),lower=True),right)
        np.testing.assert_allclose(diagonal,np.diag(h),rtol=1e-13)
        np.testing.assert_allclose(g,right[:,0],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(c,right[:,1:],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(step,expected,rtol=1e-12,atol=1e-12)


def test_reference_shortcut_rejects_a_different_operator_before_fitting():
    n=60;wrong=np.eye(3*n)
    with pytest.raises(ValueError,match='does not match'):
        fit_nonlinear_shared_parameters(np.zeros((1,n,3)),BalloonParameters(),.1,
            parameter_names=[],parameter_bounds=[],sd=np.ones(3),mean_operator=wrong,
            observation_reference_steps=50)


def test_structured_external_solver_preserves_small_full_fit_and_budget_failure():
    n=65;dt=.1;p=BalloonParameters();t=np.arange(n)*dt
    driver=.025*np.sin(t/3);initial=np.array([.01,1.02,1.01,1.01,1.])
    true=nonlinear_driver_forward(driver,initial,p,dt,substeps=8,derivative=False)
    reference=np.eye(n);reference[:,:50]-=.02;op=np.kron(reference,np.eye(3))
    y=(op@true['canonical_prediction'].ravel()).reshape(1,n,3)
    mask=np.ones_like(y,bool);mask[:,-1]=False
    kwargs=dict(parameter_names=['tau','neurovascular_gain'],parameter_bounds=[[.25,12.],[.1,10.]],
        sd=np.array([.025,.025,.015]),visible=mask,
        starts=[{'parameter_values':[2.,1.]}],max_iterations=80,max_evaluations_per_trial=250,
        parameter_prior_log_mean=np.log([2.,1.]),parameter_prior_precision=np.full(2,64.),
        data_weight=.4)
    def callback(j,r,jp,d):
        return torch_normal_steps([j],[r],[jp],[d],device='cpu')[0]
    dense=fit_nonlinear_shared_parameters(y,p,dt,mean_operator=op,**kwargs)
    fast=fit_nonlinear_shared_parameters(y,p,dt,observation_reference_steps=50,linear_solver=callback,**kwargs)
    assert dense['converged'] and fast['converged']
    np.testing.assert_allclose(fast['prediction'],dense['prediction'],rtol=1e-5,atol=1e-7)
    np.testing.assert_allclose(fast['parameter_values'],dense['parameter_values'],rtol=1e-5,atol=1e-7)
    kwargs['max_iterations']=0
    a=fit_nonlinear_shared_parameters(y,p,dt,mean_operator=op,**kwargs)
    b=fit_nonlinear_shared_parameters(y,p,dt,observation_reference_steps=50,linear_solver=callback,**kwargs)
    assert not a['converged'] and not b['converged']
    assert a['starts'][0]['convergence_reason']==b['starts'][0]['convergence_reason']=='iteration_budget'


def test_gpu_transport_sends_only_small_headers_and_isolates_worker_buffers():
    import multiprocessing as mp
    from experiments.scripts import evaluate_shared_driver_reconstruction as runner
    storage=mp.RawArray('d',2*runner._TAK_CACHE_GPU_SLOT)
    view=np.frombuffer(storage,dtype=np.float64);view[:100]=77.
    class Connection:
        def send(self,header):
            assert len(header)==4 and all(np.isscalar(v) for v in header)
            self.header=header
        def poll(self,timeout):return True
        def recv(self):
            m,n,k,d=self.header;slot=view[runner._TAK_CACHE_GPU_SLOT:]
            j=slot[:m*n].reshape(m,n);rhs=slot[m*n:m*n+m*(k+1)].reshape(m,k+1)
            h=j.T@j;b=j.T@rhs
            step=cho_solve(cho_factor(h+d*np.diag(np.maximum(np.diag(h),1.)),lower=True),b)
            return np.diag(h),b[:,0],b[:,1:],step
    client=runner._TakCacheGPUClient(Connection(),0,storage,1)
    rng=np.random.default_rng(28);j=rng.normal(size=(30,7));r=rng.normal(size=30);p=rng.normal(size=(30,2))
    diagonal,g,c,step=client(j,r,p,.01)
    expected=torch_normal_steps([j],[r],[p],[.01],device='cpu')[0]
    for a,b in zip((diagonal,g,c,step),expected):np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(view[:100],np.full(100,77.))
    assert client.requests==1


def test_mixed_pool_assigns_only_declared_workers_to_gpu(monkeypatch):
    import multiprocessing as mp
    from experiments.scripts import evaluate_shared_driver_reconstruction as runner
    monkeypatch.setattr(runner,'_TAK_CACHE_STRUCTURED',False)
    monkeypatch.setattr(runner,'_TAK_CACHE_GPU_CLIENT',None)
    storage=mp.RawArray('d',2*runner._TAK_CACHE_GPU_SLOT)
    counter=mp.Value('i',0)
    class Connection:
        def __init__(self):self.closed=False
        def close(self):self.closed=True
    for index in range(4):
        clients=[Connection(),Connection()]
        runner._tak_cache_worker_initialize(True,clients,counter,[0,1],storage)
        assert runner._TAK_CACHE_STRUCTURED
        if index<2:
            assert runner._TAK_CACHE_GPU_CLIENT.connection is clients[index]
            assert runner._TAK_CACHE_GPU_CLIENT.backend.endswith(str(index))
        else:
            assert runner._TAK_CACHE_GPU_CLIENT is None
        assert [c.closed for c in clients]==[i!=index for i in range(2)]
    assert counter.value==4
