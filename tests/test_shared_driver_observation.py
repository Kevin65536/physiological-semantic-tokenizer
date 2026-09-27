"""Array-only regressions for the shared native feature observation contract."""

import numpy as np
import pytest
from scipy.signal import resample_poly

from src.inference.observation_baselines import (
    native_feature_operators,
    visible_feature_interpolation,
)


def test_retained_runner_uses_shared_operator_aliases():
    from experiments import evaluate_ssm_overnight_diagnostics as suite

    assert suite.v3_native_operators is native_feature_operators
    assert suite.v3_visible_interpolation is visible_feature_interpolation


def test_full_mean_matches_native_processing_and_clocks():
    from src.data.homer2_preprocessing import bandpass_fnirs

    op = native_feature_operators(120)
    t = np.arange(120) / 4.
    native_t = np.arange(300) / 10.
    clean = np.column_stack((np.cos(.4*t), np.sin(.7*t), np.cos(.3*t)))
    native_hb = np.column_stack([
        np.interp(native_t, t, clean[:, column]) for column in (1, 2)
    ])
    filtered, quality = bandpass_fnirs(native_hb, sample_rate_hz=10.)
    assert quality['status'] == 'applied'
    direct = np.column_stack((clean[:, 0], resample_poly(filtered, 2, 5, axis=0)))
    direct -= direct[:20].mean(axis=0)
    mean = np.stack((op['eeg'], op['fnirs'] @ op['native_interpolation'],
                     op['fnirs'] @ op['native_interpolation']))
    rebuilt = np.einsum('ctu,uc->tc', mean, clean)
    np.testing.assert_allclose(rebuilt, direct, atol=1e-10, rtol=1e-10)
    np.testing.assert_array_equal(op['model_time'], t)
    np.testing.assert_array_equal(op['fnirs_time'], native_t)
    assert op['baseline_evidence']['sample_count'] == 20
    assert op['baseline_evidence']['resting_state_inferred'] is False
    np.testing.assert_allclose(op['native_interpolation'][-1], np.eye(120)[-1])


@pytest.mark.parametrize('modality,length,left,right', [
    ('eeg', 120, 52, 68), ('fnirs', 300, 130, 170),
])
def test_hidden_feature_perturbations_cannot_enter_visible_output(modality, length, left, right):
    op = native_feature_operators(120)
    interpolation = visible_feature_interpolation(length, left, right)
    values = np.random.default_rng(14).normal(size=(length, 2))
    changed = values.copy()
    changed[left:right] += 1e6
    processor = op[modality] @ interpolation
    np.testing.assert_array_equal(processor @ values, processor @ changed)
    visible = np.r_[0:left, right:length]
    np.testing.assert_array_equal((interpolation @ values)[visible], values[visible])
    np.testing.assert_array_equal(visible_feature_interpolation(length, 0, length),
                                  np.zeros((length, length)))


def test_observation_support_rejects_insufficient_baseline():
    with pytest.raises(ValueError, match='five-second baseline'):
        native_feature_operators(20)


@pytest.fixture
def fixed_roi_source(tmp_path):
    """Retained-shape fixture; unrelated to repository measured artifacts."""
    from experiments.scripts import evaluate_shared_driver_reconstruction as runner
    from src.inference.observation_baselines import robust_mad
    cfg=runner.read_gain_prior_config(runner.CODE_ROOT/'experiments/configs/physiology_semantic_tokenizer/shared_driver_gain_prior_v1.yaml')
    cfg['fixed_roi']=dict(pair_name='AF7Fp1',pair_index=0,
        selection='predeclared_first_registry_pair_not_validation_score',eeg='exact_parent_projection_unchanged',
        fnirs_scale='training_common_MAD_with_frozen_parent_observation_loading')
    source=tmp_path/cfg['source_run']/'prepared';source.mkdir(parents=True)
    rng=np.random.default_rng(79);hb=rng.normal(size=(24,300,36,2))*.03
    eeg=rng.normal(size=(24,120))*.1;op=native_feature_operators(120)
    validation=[0,4,8,12,16,20];train=[i for i in range(24) if i not in validation]
    unscaled=np.array([op['fnirs']@x[:,0] for x in hb]);common=float(robust_mad(unscaled[train].reshape(-1)))
    factor=.04/common;feature_hb=hb[:,:,0]*factor
    target=np.array([np.column_stack((op['eeg']@e,op['fnirs']@h)) for e,h in zip(eeg,feature_hb)])
    sd=np.std(np.concatenate(target[train]),axis=0)
    for subject in cfg['subjects']:
        identities=[dict(subject=subject,session=s,original_ma_trial_position=p)
                    for s in cfg['sessions'] for p in cfg['original_positions']]
        runner.write_json(source/f'{subject}.json',dict(subject=subject,trials=identities,synthetic=False,
            original_heldout_trials_processed=0,fnirs_pairs=['AF7Fp1','AF3Fp1']+[f'pair{i}' for i in range(34)]))
        for outer in cfg['outer_folds']:
            val=[i for i in range(24) if i%4==outer];tr=[i for i in range(24) if i not in val]
            runner.write_json(source/f'{subject}_o{outer}_E0.json',dict(subject=subject,outer=outer,inner=None,
                train=tr,validation=val,normalization_sd=sd,feature_noise=dict(old_channel_only=True),
                array_sha256='parent-projection-fixture',source_preparation_sha256='fullsubject-fixture',
                projection=dict(fnirs_pair=0,fnirs_factor=factor,inverse_fnirs_factor=1/factor,
                    measurement_scale=dict(fnirs_common=common),computational_scale=dict(fnirs_common=1/common),
                    observation_loading=dict(fnirs_common=.04),pair_eligible=[True]*36,pair_scores=[1.]*36,
                    loading=[.1,.2],eeg_factor=3.)))
    np.savez(source/'subject_01.npz',feature_fnirs=hb,eligible=np.ones((24,36),bool))
    np.savez(source/'subject_01_o0_E0.npz',feature_eeg=eeg,feature_fnirs=feature_hb,target=target,normalizer=sd)
    return runner,cfg,tmp_path,source,train,validation


def test_fixed_roi_recomposes_original_pair_and_freezes_eeg(fixed_roi_source):
    runner,cfg,root,source,train,val=fixed_roi_source
    original,oldmeta=runner.load_measured(cfg,root,'subject_01',0)
    arrays,meta=runner.load_fixed_roi_measured(cfg,root,'subject_01',0)
    np.testing.assert_allclose(arrays['target'],original['target'],rtol=1e-12,atol=1e-14)
    np.testing.assert_array_equal(arrays['feature_eeg'],original['feature_eeg'])
    np.testing.assert_array_equal(arrays['target'][:,:,0],original['target'][:,:,0])
    assert arrays['normalizer'][0]==original['normalizer'][0]
    assert meta['projection']['loading']==oldmeta['projection']['loading']
    assert meta['projection']['eeg_factor']==oldmeta['projection']['eeg_factor']
    assert 'feature_noise' not in meta and 'pair_scores' not in meta['projection']
    assert 'array_sha256' not in meta
    assert meta['fixed_roi']['parent_projection_array_sha256']=='parent-projection-fixture'
    assert meta['source_preparation_sha256']=='fullsubject-fixture'
    assert 'not_reestimated_or_reused' in meta['fixed_roi']['noise']
    assert meta['train']==train and meta['validation']==val


def test_fixed_roi_validation_perturbation_cannot_change_training_scales(fixed_roi_source):
    runner,cfg,root,source,train,val=fixed_roi_source
    before,meta=runner.load_fixed_roi_measured(cfg,root,'subject_01',0)
    with np.load(source/'subject_01.npz') as z:a={k:z[k] for k in z.files}
    a['feature_fnirs'][val,:,0,:]=a['feature_fnirs'][val[::-1],:,0,:]*50.+20.
    np.savez(source/'subject_01.npz',**a)
    after,newmeta=runner.load_fixed_roi_measured(cfg,root,'subject_01',0)
    np.testing.assert_array_equal(after['normalizer'],before['normalizer'])
    np.testing.assert_array_equal(after['target'][train],before['target'][train])
    np.testing.assert_array_equal(after['feature_eeg'],before['feature_eeg'])
    assert newmeta['fixed_roi']['fnirs_factor']==meta['fixed_roi']['fnirs_factor']
    assert not np.allclose(after['target'][val,:,1:],before['target'][val,:,1:])


@pytest.mark.parametrize('case',['scope','pair','remote_identity','remote_eligibility'])
def test_fixed_roi_rejects_metadata_before_any_array_read(fixed_roi_source,monkeypatch,case):
    import json
    runner,cfg,root,source,train,val=fixed_roi_source
    if case=='scope':cfg['subjects'].append('subject_02')
    elif case=='pair':cfg['fixed_roi']['pair_name']='AF3Fp1'
    elif case=='remote_identity':
        p=source/'subject_18.json';d=json.loads(p.read_text());d['trials'][0]['original_ma_trial_position']=99;runner.write_json(p,d)
    else:
        p=source/'subject_18_o3_E0.json';d=json.loads(p.read_text());d['projection']['pair_eligible'][0]=False;runner.write_json(p,d)
    def forbidden(*a,**k):raise AssertionError('read array before full metadata preflight')
    monkeypatch.setattr(np,'load',forbidden)
    with pytest.raises(ValueError):runner.load_fixed_roi_measured(cfg,root,'subject_01',0)


@pytest.mark.parametrize('case',['shape','finite','eligible'])
def test_fixed_roi_rejects_invalid_selected_feature_boundary(fixed_roi_source,case):
    runner,cfg,root,source,train,val=fixed_roi_source
    p=source/'subject_01.npz'
    with np.load(p) as z:a={k:z[k] for k in z.files}
    if case=='shape':a['feature_fnirs']=a['feature_fnirs'][:,:,:35]
    elif case=='finite':a['feature_fnirs'][0,0,0,0]=np.nan
    else:a['eligible'][0,0]=False
    np.savez(p,**a)
    with pytest.raises(ValueError,match='shape/finite/eligibility'):
        runner.load_fixed_roi_measured(cfg,root,'subject_01',0)
