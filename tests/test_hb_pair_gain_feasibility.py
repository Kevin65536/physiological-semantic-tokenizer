import importlib.util
from pathlib import Path
import numpy as np
import pytest


path = Path(__file__).resolve().parents[1]/'experiments/scripts/audit_hb_pair_gain_feasibility.py'
spec = importlib.util.spec_from_file_location('hb_pair_gain_geometry', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_exact_loss_curve_with_native_scale_and_mask():
    rng = np.random.default_rng(776)
    p, d, t = rng.normal(size=(3,7,2)), rng.normal(size=(3,7,2)), rng.normal(size=(3,7,2))
    mask = rng.random(p.shape) > .2
    variance = rng.uniform(.2,2,(3,2))
    q = module.squared_curve(p,d,t,mask,variance)
    for gain in [-1.,-.3,0.,.6,1.]:
        truth = np.where(mask,(p+gain*d-t)**2,0.).sum(1)/mask.sum(1)/variance
        np.testing.assert_allclose(module.evaluate(q,gain),truth,atol=2e-15)
    scale = np.array([.1,2.,5.])[:,None,None]
    np.testing.assert_allclose(module.squared_curve(p*scale,d*scale,t*scale,mask,variance*scale[:,0]**2),q,atol=2e-15)


def test_exact_signed_improvement_interval():
    assert module.quadratic_improvement_interval([2,2,1]) == [-1.,0.]
    assert module.quadratic_improvement_interval([2,-.5,1]) == [0.,.5]
    assert module.quadratic_improvement_interval([2,0,1]) == []


def test_amplitude_intervals_and_tail_relaxation_are_honest():
    q=np.array([[1.,-.4,.1],[.5,.2,.1]])
    allowed,curve=module.rms_intervals(q,np.array([.4,.6]),.8,1.1)
    for gain in np.linspace(-1,1,101):
        inside=any(a-1e-10<=gain<=b+1e-10 for a,b in allowed)
        assert inside == (.8-1e-10<=curve(gain)<=1.1+1e-10)
    rng=np.random.default_rng(12)
    rows=np.stack([rng.uniform(.1,2,20),rng.uniform(-.1,.1,20),rng.uniform(.2,.5,20)],-1)
    necessary=module.tail_necessary_intervals(rows,1.7,.9)
    for gain in np.linspace(-1,1,501):
        if np.quantile(module.evaluate(rows,gain),.9)<=1.7:
            assert any(a-1e-12<=gain<=b+1e-12 for a,b in necessary)


def test_masked_correlation_moments_match_direct_waveforms():
    rng=np.random.default_rng(33);p,d,y=rng.normal(size=(3,5,17));mask=rng.random(y.shape)>.2
    moments=module.correlation_moments(p,d,y,mask)
    for gain in [-1.,0.,.3,1.]:
        exact=[np.corrcoef((a+gain*b)[v],c[v])[0,1] for a,b,c,v in zip(p,d,y,mask)]
        np.testing.assert_allclose(module.correlations(moments,[gain])[0],exact,atol=1e-15)


@pytest.mark.parametrize('flag,phrase',[('--gain-replay-only','requires the signed-gain contract'),('--spectral-review-all','requires its spectral contract')])
def test_replay_flag_rejects_an_unrelated_contract_before_data_access(tmp_path,flag,phrase):
    import os,subprocess,sys
    config=tmp_path/'fixture.yaml';config.write_text('schema: hb_native_id_readout_v1\ndata:\n  test_access: forbidden\n')
    output=tmp_path/'output'
    process=subprocess.run([sys.executable,str(path.parent/'run_hb_native_id_readout.py'),'--config',str(config),'--output',str(output),flag],capture_output=True,text=True,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    assert process.returncode!=0 and phrase in process.stderr
    assert not output.exists()
