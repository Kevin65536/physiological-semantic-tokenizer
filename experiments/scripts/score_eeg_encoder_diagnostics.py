"""Matched development scoring, endpoint verification and fixed-window comparisons."""
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import run_eeg_encoder_diagnostics as suite
from run_dual_component_tokenizer import _base_training_model, identity, now, read_json, write_json
from score_single_vq_spectral_location import extract, patch_means
from src.analysis.dual_component_tokenizer import base_training_aggregate

CAP = suite.aligned.capacity
CELLS = [(arch,fit) for arch, spec in suite.ARCHS.items() for fit in spec['fits']]
CONTRASTS = [(arch,fit) for arch,fit in CELLS if arch != 'observed']
METRICS = ('nmse', 'correlation', 'decoded')


def verify_training(arch, fit):
    cfg = suite.settings(arch)
    folder = suite.job(arch, fit)
    state = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=False)
    record = read_json(folder/'summary.json')
    sig = record['signature']
    probe = read_json(suite.OUT/arch/'resolved_config.json')['development_probe']
    if (record['status'] != 'completed' or record['completed_step'] != state['step'] or state['step'] != 10000
            or state['signature'] != sig or sig['config'] != identity(cfg) or sig['task'] != suite.task_for(arch,fit)
            or sig['development_probe'] != identity(probe) or state['history'] != record['history']
            or sig.get('spectral_objective') != (identity(suite.aligned.objective()) if fit=='hard' else None)):
        raise ValueError(f'unmatched terminal checkpoint for {arch}/{fit}')
    return cfg, state


@torch.inference_mode()
def score_continuous(arch):
    cfg, state = verify_training(arch,'continuous')
    net = _base_training_model(cfg, suite.task_for(arch,'continuous')).to('cuda:0').eval()
    net.load_state_dict(state['model_state'])
    dest = suite.job(arch,'continuous') / 'evaluation'
    dest.mkdir(exist_ok=True)
    CAP.prepare_panel(recorded_root=suite.aligned.spectral.PLAN['recorded_root'])
    rows, scales = CAP.PANEL['rows'], CAP.PANEL['scales']
    sites = {key:i for i,key in enumerate(sorted({(r['dataset'],p) for r in rows
                                                   if r['modality']=='eeg' for p in r['positions']}))}
    cases = read_json(CAP.OUT/'review_20261124_v1/tables/wave_eeg.json')
    fixed = {(c['row'],c['start_s'],c['position']):name
             for c,name in zip(cases, ('median','difficult'), strict=True)}
    cal = suite.aligned.objective()['calibration']
    parts = defaultdict(list)
    examples = []
    for ds in CAP.DATASETS:
        ex = CAP.PANEL['examples']['eeg','validation',ds]
        file = dest/f'{ds}_validation.npz'
        if not file.exists():
            extract(net, ex, ds, 'validation', rows, scales, cal, sites, fixed, dest)
        with np.load(file) as z:
            np.testing.assert_array_equal(ex, z['examples'])
            for key in ('nmse','correlation'):
                parts[key].append(z[key])
            parts['decoded'].append(patch_means((z['target']-z['decoded'])**2, z['valid']))
        examples.append(ex)
    result = base_training_aggregate({k:np.concatenate(v) for k,v in parts.items()}, np.concatenate(examples), rows)
    if (result['subjects'] != 16 or result['inputs'] != 53476
            or any(result['metrics'][k]['nonfinite_entries'] for k in METRICS)):
        raise ValueError('incomplete continuous development endpoint')
    write_json(dest/'summary.json', dict(result, status='completed', input=suite.ARCHS[arch]['source']))


def contrast(candidate, baseline):
    def by_subject(score):
        return {(r['dataset'],r['subject']):np.mean(r['value']) for r in score['metrics']['nmse']['by_subject']}
    left, right = by_subject(candidate), by_subject(baseline)
    if left.keys() != right.keys() or len(left) != 16:
        raise ValueError('16 matching subjects required')
    paired = [dict(dataset=d,subject=s,delta_nmse=left[d,s]-right[d,s]) for d,s in sorted(left)]
    # Resample whole people within their original datasets; keep equal dataset weights.
    strata = {d:np.array([v['delta_nmse'] for v in paired if v['dataset']==d]) for d in CAP.DATASETS}
    delta = np.mean([a.mean() for a in strata.values()])
    np.testing.assert_allclose(delta, candidate['selection_score']-baseline['selection_score'], atol=1e-12)
    rng = np.random.default_rng(17)
    boot = np.mean([rng.choice(v,size=(10000,len(v)),replace=True).mean(1) for v in strata.values()],axis=0)
    return dict(delta_nmse=delta, relative_percent=100*delta/baseline['selection_score'],
                lower_error_people=sum(v['delta_nmse']<0 for v in paired),
                bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                by_dataset={d:v.mean() for d,v in strata.items()}, people=paired)


def draw_waveforms(case, fit, fixed_scores):
    cells = [arch for arch,candidate in CELLS if candidate==fit]
    fig, axes = plt.subplots(3,len(cells),sharex=True,sharey='row',figsize=(4.5*len(cells),7.5))
    titles = dict(observed='Observed X', privileged='Privileged S/O', wide='Wider input', deep='Deeper encoder')
    previous = None
    for col,arch in enumerate(cells):
        folder = suite.job(arch,fit)/'evaluation'
        meta = read_json(folder/'simultaneous_eeg_nirs_fixed_windows.json')[case]
        with np.load(folder/(case+'_waveforms.npz')) as z:
            truth = {k:z[k] for k in ('example','mask','x','s','o')}
            if previous is not None:
                for k,v in truth.items():
                    np.testing.assert_array_equal(v, previous[k])
            previous = truth
            for row,(k,key) in enumerate([('x','sum'),('s','S'),('o','O')]):
                ax = axes[row,col]
                t = np.arange(z[k].size)/200
                mask = z['mask'].ravel()
                ax.plot(t,np.where(mask,z[k].ravel(),np.nan),color='.45',lw=.65,
                        label='Observation' if row==0 else 'Teacher component')
                ax.plot(t,np.where(mask,z['reconstructed_'+key].ravel(),np.nan),color='#007d8c',lw=.7,
                        label='Reconstruction')
                ax.set_title(f"{titles[arch]}   NMSE {meta['nmse'][row]:.3f}",fontsize=10)
                if col==0:
                    ax.set_ylabel(k.upper()+' / training input SD')
                    ax.legend(fontsize=7,loc='upper right')
                if row==2:
                    ax.set_xlabel('Time within window (s)')
        fixed_scores.append(dict(case=case,arch=arch,fit=fit,**meta))
    fig.suptitle(f"Fixed {case} EEG window  |  {fit}  |  {meta['subject']}  {meta['record']}  {meta['position']}")
    fig.tight_layout()
    fig.savefig(suite.OUT/f'{case}_{fit}.png',dpi=220)
    plt.close(fig)


def collect_curves():
    curves=[]
    for arch,fit in CELLS:
        for stage in ('warmup',fit):
            record=read_json(suite.job(arch,stage)/'summary.json')
            for row in record['history']:
                tr=row['monitor']
                curves.append(dict(architecture=arch,fit=fit,stage=stage,step=row['step'],
                                   train_nmse=np.mean([tr[k] for k in 'xso']),
                                   development_nmse=np.mean(row['development_nmse_xso'])))
    frame=pd.DataFrame(curves)
    frame.to_csv(suite.OUT/'monitor_curves.csv',index=False)
    fig,axes=plt.subplots(2,2,sharex=True,sharey=True,figsize=(11,7))
    colors=dict(observed='#303030',privileged='#9b4060',wide='#008678',deep='#4074aa')
    for col,fit in enumerate(('continuous','hard')):
        for row,key in enumerate(('train_nmse','development_nmse')):
            ax=axes[row,col]
            for arch,points in frame[frame.fit==fit].groupby('architecture',sort=False):
                ax.plot(points.step,points[key],label=arch,color=colors[arch],lw=1.3)
            ax.axvline(1000,ls=':',lw=1,color='.5')
            ax.set_title(f"{fit}  |  {'Training' if row==0 else 'Development'} monitor")
            ax.set_ylabel('Mean example X/S/O NMSE')
            ax.set_xlabel('Training step')
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(suite.OUT/'monitor_curves.png',dpi=220)
    plt.close(fig)


def summarize():
    scores={}
    table=[]
    pairing=None
    initialization=None
    fits_done=[]
    CAP.prepare_panel(recorded_root=suite.aligned.spectral.PLAN['recorded_root'])
    panel=CAP.PANEL
    original=suite.ROOT/read_json(suite.aligned.OUT/'eeg/result.json')['checkpoint']
    parent=original.parent/'evaluation'
    for arch,fit in CELLS:
        cfg, state=verify_training(arch,fit)
        job=suite.job(arch,fit)
        dest=job/'evaluation'
        result=read_json(dest/'summary.json')
        sig=state['signature']
        shared={k:sig[k] for k in ('input','scales','development_probe')}
        shared['terminal_training_sampler'] = identity(state['numpy_rng'])
        if pairing is None:
            pairing=shared
        if (shared!=pairing or result['status']!='completed' or result['subjects']!=16
                or result['inputs']!=53476 or any(result['metrics'][k]['nonfinite_entries'] for k in METRICS)):
            raise ValueError(f'incomplete or unpaired {arch}/{fit}')
        if fit == 'hard':
            init = read_json(job/'initialization.json')
            if init['split'] != 'train' or (initialization is not None and init != initialization):
                raise ValueError('hard models used different initialization examples')
            initialization = init
        values=defaultdict(list)
        examples=[]
        for ds in CAP.DATASETS:
            name=f'{ds}_validation.npz'
            with np.load(dest/name) as data,np.load(parent/name) as ref:
                for key in ('examples','target','valid','bg'):
                    np.testing.assert_array_equal(data[key],ref[key])
                np.testing.assert_array_equal(data['examples'],panel['examples']['eeg','validation',ds])
                examples.append(data['examples'])
                for k in ('nmse','correlation'):
                    values[k].append(data[k])
                values['decoded'].append(patch_means((data['target']-data['decoded'])**2,data['valid']))
        actual=base_training_aggregate({k:np.concatenate(v) for k,v in values.items()},np.concatenate(examples),panel['rows'])
        for k in METRICS:
            assert actual['metrics'][k]==result['metrics'][k]
        assert actual['selection_score']==result['selection_score']
        scores[arch,fit]=result
        nmse=np.ravel(result['metrics']['nmse']['macro'])
        corr=np.ravel(result['metrics']['correlation']['macro'])
        decoded=np.ravel(result['metrics']['decoded']['macro'])
        table.append(dict(architecture=arch,fit=fit,macro_nmse=result['selection_score'],
                          parameters=read_json(job/'summary.json')['parameters'],
                          **dict(zip(('X_nmse','S_nmse','O_nmse'),nmse,strict=True)),
                          **dict(zip(('X_corr','S_corr','O_corr'),corr,strict=True)),
                          **dict(zip(('S_logtheta_mse','S_logalpha_mse','S_theta_fraction_mse','S_alpha_fraction_mse'),decoded,strict=True)),
                          S_single_id_mean_gain=result.get('mean_id_gain')))
        fits_done.append(dict(arch=arch,fit=fit,step=state['step'],inputs=actual['inputs'],subjects=actual['subjects']))
        del state
    contrasts={a+'_'+f:contrast(scores[a,f],scores['observed',f]) for a,f in CONTRASTS}
    pd.DataFrame(table).to_csv(suite.OUT/'endpoints.csv',index=False)
    pd.DataFrame([dict(contrast=k,delta_nmse=v['delta_nmse'],relative_percent=v['relative_percent'],
                       lower_error_people=v['lower_error_people'],low95=v['bootstrap95'][0],
                       high95=v['bootstrap95'][1]) for k,v in contrasts.items()]).to_csv(suite.OUT/'contrasts.csv',index=False)
    pd.DataFrame([dict(contrast=k,**person) for k,v in contrasts.items() for person in v['people']]).to_csv(suite.OUT/'subject_contrasts.csv',index=False)
    collect_curves()
    fixed=[]
    for fit in ('continuous','hard'):
        for case in ('median','difficult'):
            draw_waveforms(case,fit,fixed)
    write_json(suite.OUT/'fixed_windows.json',fixed)
    reference_cont=read_json(CAP.OUT/'evaluation/eeg_d128_k0_s17/eeg_summary.json')
    reference_hard=read_json(parent/'summary.json')
    baseline_replay={fit:scores['observed',fit]['selection_score']-ref['selection_score']
                     for fit,ref in [('continuous',reference_cont),('hard',reference_hard)]}
    write_json(suite.OUT/'summary.json',dict(status='completed',finished=now(),contract=suite.PLAN,planned=7,
               completed=fits_done,paired_signatures=pairing,hard_initialization_examples=len(initialization['examples']),
               metrics=table,contrasts=contrasts,
               baseline_replay_delta=baseline_replay,
               interval='Descriptive paired 95% bootstrap percentile interval, 10000 draws of subjects within each dataset, frozen fits and seed; no multiplicity adjustment.'))
