"""Contract, pairing, bin denoms, selection and end-to-end software test; synthetic X only."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from threadpoolctl import threadpool_limits

from experiments.scripts import run_observation_gain_shape as g


def fixture(n=4):
    rng=np.random.default_rng(7283); t=np.arange(400)/200
    frame=pd.DataFrame([dict(dataset=d,subject=f'{i:02d}',record='a',parent='p',start_s=0,position='C3',patch=j)
                        for d in g.DATASETS for i in range(2) for j in range(n)])
    amp=np.linspace(.01,2,len(frame)); x=amp[:,None,None]*(np.sin(2*np.pi*35*t)[None,None]+.4*rng.normal(size=(len(frame),1,400)))+.17
    return frame,x,np.ones((len(frame),1))


def test_contract_synthetic_phase_and_immutable_failure(tmp_path,monkeypatch):
    cfg=g.configuration(g.CONFIG); assert json.loads(json.dumps(cfg))==cfg
    other=tmp_path/'bad.yaml';bad=dict(cfg,rates=[512]);other.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError): g.configuration(other)
    monkeypatch.setattr(g,'SUITE',tmp_path);out=tmp_path/'20261120_gain_v1'
    monkeypatch.setattr(g,'replay_panel',lambda *_args,**_kw:pytest.fail('no measured panel in software tests'))
    def cli(stage): return g.main(['--output',str(out),'--stage',stage])
    with pytest.raises(ValueError,match='passed synthetic'): cli('screen')
    cli('synthetic'); assert g.read(out/'controller.json')['status']=='synthetic_passed'
    with pytest.raises(ValueError,match='immutable'): cli('synthetic')
    def fake_screen(*_a): raise RuntimeError('retained synthetic exception')
    monkeypatch.setattr(g,'screen',fake_screen)
    with pytest.raises(RuntimeError,match='retained'): cli('screen')
    state=g.read(out/'controller.json'); assert state['status']=='failed' and (out/state['error']).exists()
    with pytest.raises(ValueError,match='previous attempt'): cli('screen')


def test_rms_bins_count_absent_strata_and_band_power_definition():
    f,x,v=fixture(); bins=np.full(len(f),1); bins[:8]=0
    y=2*x+4; t=g.rms_diagnostic(f,x,y,v,bins); k=t[(t.level=='macro')&(t.rms_bin=='low')].iloc[0]
    assert (k.n_patches,k.n_datasets,k.all_datasets)==(8,1,4)
    assert k[g.HF]==pytest.approx(20*np.log10(2)) and k[g.HC]==pytest.approx(1)
    assert k.predicted_30_45_power_db_train_X2-k.observed_30_45_power_db_train_X2==pytest.approx(20*np.log10(2))
    missing=t[(t.rms_bin=='high')&(t.level=='macro')].iloc[0]; assert missing.n_patches==0 and np.isnan(missing[g.HF])
    absent=t[(t.rms_bin=='low')&(t.level=='dataset')&(t.dataset=='refed')].iloc[0]
    assert absent.n_subjects==0 and absent.all_subjects==2 and np.isnan(absent[g.HC])
    sine=np.sin(2*np.pi*35*np.arange(400)/200)[None,None,:]
    np.testing.assert_allclose(g.band_power_db(sine),10*np.log10(.5),atol=2e-5)
    np.testing.assert_allclose(g.ac_rms(7+3*sine),3/np.sqrt(2))


def test_mixed_modality_historical_csv_does_not_convert_subject_ids_or_spectra_to_strings(tmp_path):
    f,x,v=fixture(); tab=g.reconstruction(f,x,x,v,'eeg'); eeg=tab.assign(method='flat',bits_per_patch=256,modality='eeg',split='validation')
    hb=eeg.copy(); hb['modality']='hb'; hb[g.HF]=''; hb[g.HC]=''
    fn=tmp_path/'synthetic.csv'; pd.concat([eeg,hb],ignore_index=True).to_csv(fn,index=False)
    loaded=g.scores(fn); assert set(loaded.loc[loaded.level=='subject','subject'])=={'00','01'}
    assert g.check_replay(loaded,tab,g.configuration(g.CONFIG)) < 1e-12


def test_hierarchy_replay_rejects_missing_subjects_and_spectral_drift():
    cfg=g.configuration(g.CONFIG); f,x,v=fixture(); sc=g.reconstruction(f,x,x,v,'eeg')
    assert g.check_replay(sc, sc.sample(frac=1,random_state=3),cfg)==0
    with pytest.raises(ValueError,match='replay'): g.check_replay(sc,sc.iloc[1:],cfg)
    bad=sc.copy(); bad.loc[bad.index[-1],g.HF]=3
    with pytest.raises(ValueError,match='replay'):g.check_replay(sc,bad,cfg)
    with pytest.raises(ValueError,match='duplicate'):g.check_replay(sc,pd.concat([sc,sc]),cfg)


def test_macro_selection_does_not_claim_single_trial_repair_or_hide_sd():
    cfg=g.configuration(g.CONFIG); f,x,v=fixture(); models=[]
    for b in (256,512,1024):
        for name in ['gain_conditional','adaptive_pattern','gain_independent','independent_tail','flat']:
            t=g.reconstruction(f,x,x,v,'eeg').assign(method=name,bits_per_patch=b)
            t.loc[t.level=='macro',g.HF]=4 if b==256 else 0
            t.loc[t.level=='dataset',g.HF]=[4,-4,5,-5]  # signed macro alone can hide four failures
            t.loc[t.level=='subject',g.HF]=[4,4,-4,-4,5,5,-5,-5]
            models.append(t)
    tab=pd.concat(models,ignore_index=True); selected,candidates=g.decisions(tab,cfg)
    assert selected['bits']==512 and selected['macro_pass'] and not selected['single_trial']['meets_local_band_checks_and_improves_abs_bias']
    valid=(tab.method=='gain_conditional') & (tab.level=='macro'); tab.loc[valid,g.HC]=.69
    missed,_=g.decisions(tab,cfg); assert missed['bits']==1024 and not missed['macro_pass']
    a=g.band_audit(tab); key=a[(a.level=='four_datasets')&(a.band_hz=='30_45')].iloc[0]
    assert key.max_abs_dataset_db==5 and key.signed_db==0 and key.sd_four_dataset_signed_db==pytest.approx(np.std([4,-4,5,-5],ddof=1))
    delta=g.contrasts(tab,cfg); assert len(delta)==3*4*4*5 and set(delta[delta.level=='macro'].n_subjects)=={8}


def test_end_to_end_synthetic_cell_and_matched_parent_replay(tmp_path):
    c=g.configuration(g.CONFIG); f,z,v=fixture(n=2); rng=np.random.default_rng(613)
    train=rng.normal(size=(96,1,400)); bits=256
    with threadpool_limits(limits=1):
        router=g.FactorizedObservationTokenizer('eeg',seed=17).fit(train,partition='train')
        old=g.AdaptiveObservationDetail(router,bits).fit(train,partition='train')
        ref=[]
        for mode,method in [('conditional','adaptive_pattern'),('independent','independent_tail')]:
            y=old.decode(old.encode(z,mode=mode),mode=mode)
            ref.append(g.reconstruction(f,z,y,v,'eeg').assign(method=method,bits_per_patch=bits))
        hist=pd.concat(ref,ignore_index=True); pid=lambda wave: router.encode(wave,mode='prototype')['pattern']
        ids=pid(z); bins=np.searchsorted(np.quantile(g.ac_rms(train),[.25,.75]),g.ac_rms(z),side='right'); sample=np.array([0,4,8,12]);
        r=g.cell(bits,router,train,train,(f,z,v),pid(train),ids,bins,hist,c,sample)
        assert max(r['replay'].values())==0 and set(r['scores'].method)=={'adaptive_pattern','independent_tail','gain_independent','gain_conditional'}
        fn=tmp_path/'synthetic_model.joblib';joblib.dump(r['model'],fn); reload=joblib.load(fn)
        np.testing.assert_allclose(g.packed_replay(reload,z[sample], 'conditional',bits,ids[sample]),r['examples']['gain_conditional'],atol=0,rtol=0)
        with pytest.raises(ValueError,match='IDs'):g.packed_replay(reload,z,'conditional',bits,(ids+1)%16)
        rows=r['bins']; counts=rows[(rows.method=='gain_conditional')&(rows.level=='macro')].n_patches
        assert counts.sum()==len(f)
