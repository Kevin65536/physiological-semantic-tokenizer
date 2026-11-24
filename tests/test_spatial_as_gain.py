"""Band-gain calibration matches the full waveform/energy audit exactly."""
import numpy as np
import yaml

from experiments.scripts.calibrate_spatial_analysis_synthesis import sufficient_statistics,item_metrics,fit,ROOT
from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics
from src.tokenizers.spatial_waveform_dictionary import band_filter


def example():
    rng=np.random.default_rng(7)
    features=np.stack([band_filter(rng.normal(size=(4,6000)),band,axis=1) for branch in range(2) for band in range(5)],axis=-1)
    gain=np.r_[np.full(5,1.3),np.full(5,.8)]
    targets=np.stack([features@gain,features[:,:,:5]@gain[:5],features[:,:,5:]@gain[5:]],axis=-1)
    mask=np.ones((4,6000),bool);mask[:,-100:]=False
    targets[~mask]=np.nan
    return features,targets,mask,gain


def test_sufficient_statistics_reproduce_masked_metrics_and_patch_band_energy():
    features,targets,mask,_=example()
    variance=np.ones((4,3));data=sufficient_statistics(features,targets,mask,variance)
    gain=np.linspace(.8,1.2,10)
    values=item_metrics(data,gain)
    prediction=dict(sum=(features@gain).reshape(4,15,400),S=(features[:,:,:5]@gain[:5]).reshape(4,15,400),O=(features[:,:,5:]@gain[5:]).reshape(4,15,400))
    batch={key:targets[:,:,c].reshape(4,15,400) for c,key in enumerate(('x','s','o'))}
    batch['mask']=mask.reshape(4,15,400)
    for key in ('x','s','o'):
        batch['v'+key]=np.ones((4,15,400));batch['dv'+key]=np.ones((4,15,400))
    actual=joint_reconstruction_metrics(prediction,batch,1,np.ones(4))['metrics']
    for key in ('nmse','rms_gain'):
        np.testing.assert_allclose(values[key],actual[key][:,:,0],atol=1e-12,rtol=1e-12)
    for j,band in enumerate(('4_8','8_13')):
        np.testing.assert_allclose(values['band_db'][:,:,j],actual['band_'+band+'_db'][:,:,0],atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(values['band_abs_db'][:,:,j],actual['band_'+band+'_abs_db'][:,:,0],atol=1e-12,rtol=1e-12)


def test_train_only_quadratic_fit_recovers_a_fixed_known_gain():
    features,targets,mask,truth=example()
    data=sufficient_statistics(features,targets,mask,np.ones((4,3)))
    cfg=yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_spatial_as_gain_v1.yaml').read_text())
    for constrained in (False,True):
        result=fit(data,np.full(4,.25),cfg,constrained)
        np.testing.assert_allclose(result['gain'],truth,atol=2e-4)
        assert result['success']
        if constrained:assert result['feasible']
