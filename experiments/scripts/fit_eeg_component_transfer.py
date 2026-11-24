"""Test whether dataset-specific local or cross-electrode observed EEG improves S/O supervision."""

import argparse
import fcntl
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import resource
import time
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluate_joint_tokenizer import input_scales, write
from src.analysis.dual_component_tokenizer import (
    base_training_aggregate,
    joint_reconstruction_metrics,
)
from src.data.dual_component_tokenizer import (
    base_tokenizer_batch,
    base_tokenizer_joint_eeg,
    base_tokenizer_training_data,
)
from src.tokenizers.eeg_component_transfer import (
    component_residual,
    frequency_groups,
    transfer_coefficients,
    transfer_waveform,
)
from src.tokenizers.spatial_waveform_dictionary import load_model
from src.tokenizers.eeg_transient_gain import (
    prepare_x,fit_background,fit_gain,apply_gain,band_indices,FEATURES,
    BoundedTransientSource,load_model as load_gain_model,
)

ARMS = ("legacy", "local", "montage")
STATE = {}
METRICS = (
    "nmse",
    "nrmse",
    "rmse_native",
    "bias_native",
    "rms_gain",
    "correlation",
    "local_rms_correlation",
    "band_1_4_abs_db",
    "band_4_8_abs_db",
    "band_8_13_abs_db",
    "band_13_30_abs_db",
    "band_30_45_abs_db",
)


def peak_metrics(prediction,batch,scale):
    mask=batch['mask'].reshape(len(scale),6000,1);result={}
    x=batch['x'].reshape(mask.shape);vx=batch['vx'][:,0,:1]
    observed_peak=mask&(abs(x)>=4*np.sqrt(vx)[:,None])
    for j,(branch,target) in enumerate([('sum','x'),('S','s'),('O','o')]):
        truth=np.asarray(batch[target],float).reshape(mask.shape)
        pred=np.asarray(prediction[branch],float).reshape(mask.shape)
        variance=batch['v'+target][:,0,:1]
        for name,region in [('true_high',mask&(abs(truth)>=2*np.sqrt(variance)[:,None])),
                            ('true_extreme',mask&(abs(truth)>=4*np.sqrt(variance)[:,None])),
                            ('observed_extreme',observed_peak)]:
            count=region.sum(1);energy=np.where(region,truth*truth,0).sum(1)
            values={name+'_nmse':np.divide(np.where(region,(pred-truth)**2,0).sum(1),count*variance,out=np.full(count.shape,np.nan),where=count>0),
                    name+'_signed_gain':np.divide(np.where(region,pred*truth,0).sum(1),energy,out=np.full(energy.shape,np.nan),where=energy>0),
                    name+'_points':count.astype(float)}
            for k,v in values.items():
                if k not in result:result[k]=np.full((len(scale),3,1),np.nan)
                result[k][:,j]=v
    return result


def bounded_training_items(dataset):
    rows,scales,source=STATE['rows'],STATE['scales'],STATE['source']
    out=STATE['output'];names=source.metadata['positions'][dataset]
    plan=STATE['plan'][dataset]
    references={n.upper():v for n,v in scales['positions']['eeg'][dataset].items()}
    vs,vo=np.array([[references[n.upper()]['target_variance'][b][0] for n in names] for b in ['s','o']])
    cache=OrderedDict();items=[]
    for subject,crops in plan.items():
        records=Counter(rows[c['row']]['record'] for c in crops)
        for crop in crops:
            fields,seen,valid,_,prior=load_crop(crop,names,True,cache)
            p=prepare_x(fields['x'],seen,source.transfer)
            for b in ['S','O','sum']:np.testing.assert_array_equal(p['prior'][b].T,prior[b])
            wanted=(fields['s'].T*vo[:,None]+(p['prior']['sum']-fields['o'].T)*vs[:,None])/(vs+vo)[:,None]
            spectrum=np.fft.rfft(wanted,axis=-1);energy=[];cross=[]
            for b,indices in enumerate(band_indices()):
                f=np.zeros_like(spectrum);f[:,indices]=spectrum[:,indices]
                target=np.fft.irfft(f,n=6000,axis=-1).reshape(len(names),300,20)
                u=p['carriers'][b].reshape(len(names),300,20)
                norm=(1/vs+1/vo)*valid
                energy.append(np.mean(u*u,axis=-1)*norm[:,None]);cross.append(np.mean(u*target,axis=-1)*norm[:,None])
            prepared={k:p[k] for k in ['amplitude','coherence','window_rms','observed']}
            items.append(dict(subject=subject,crop=crop,prepared=prepared,frame_energy=np.asarray(energy),frame_cross=np.asarray(cross),
                              weight=1/(len(records)*records[rows[crop['row']]['record']]),complete_targets=valid))
        write(out/f'{dataset}_progress.json',dict(stage='X_only_frame_moments',subjects_prepared=len({i['subject'] for i in items}),CPU_seconds=time.process_time()))
    return items


def bounded_full_fit(dataset):
    resource.setrlimit(resource.RLIMIT_CPU,(STATE['config']['resources']['per_worker_CPU_seconds'],STATE['config']['resources']['per_worker_CPU_seconds']+10))
    cfg=STATE['config'];items=bounded_training_items(dataset);n=len(STATE['plan'][dataset])
    fitting=[dict(item,weight=item['weight']/n) for item in items]
    background=fit_background(fitting);beta=fit_gain(fitting,background,True,cfg['model']['ridge_fraction'])
    np.savez_compressed(STATE['output']/f'{dataset}_all_train_gain.npz',background=background,beta=beta)
    return dataset,dict(crops=len(items),subjects=n,parameters=int(beta.size),CPU_seconds=time.process_time())


def bounded_dataset_fit(dataset):
    resource.setrlimit(resource.RLIMIT_CPU,(STATE['config']['resources']['per_worker_CPU_seconds'],STATE['config']['resources']['per_worker_CPU_seconds']+10))
    rows,scales,source=STATE['rows'],STATE['scales'],STATE['source']
    cfg,out=STATE['config'],STATE['output'];names=source.metadata['positions'][dataset]
    plan,folds=STATE['plan'][dataset],STATE['folds'][dataset]
    lookup={v.upper():j for j,v in enumerate(names)}
    cache=OrderedDict();items=bounded_training_items(dataset)
    metrics=defaultdict(list);identities=[];fold_ids=[];captures={};maximum_closure=0.;replay_difference=0.
    oldrun=ROOT/cfg['source']['linear_run'];frozen_examples={}
    for rule in ['fixed_hash','highest_montage_S_NMSE']:
        with np.load(oldrun/f'{dataset}_{rule}.npz') as example:frozen_examples[tuple(example['triple'])]=rule
    for fold in range(4):
        people=[s for s in plan if folds[s]!=fold]
        fitting=[dict(item,weight=item['weight']/len(people)) for item in items if item['subject'] in people]
        background=fit_background(fitting)
        fitted={arm:fit_gain(fitting,background,arm=='transient',cfg['model']['ridge_fraction']) for arm in ['background','transient']}
        paths={}
        for arm,beta in fitted.items():
            metadata=dict(schema='eeg_bounded_transient_source_v1',positions={dataset:names},transient=arm=='transient',gain_bounds=cfg['model']['gain_bounds'],
                          stage='continuous_only_before_integer_assignment',context_seconds=30,sample_rate_hz=200,frame_seconds=.1,
                          input_identity=source.metadata['input_identity'],scales_identity=source.metadata['scales_identity'],
                          input='X_and_observation_availability_only',fit_subjects=people,held_fold=fold,features=FEATURES,
                          objective='regularized_per_band_frame_energy_surrogate_followed_by_bounded_gain_and_band_projection',
                          historical_exposure='frozen_Wiener_and_G0_scales_saw_all60_train_subjects_only_new_gains_and_background_exclude_held')
            model=BoundedTransientSource(source.transfer,{dataset:dict(background=background,beta=beta)},metadata)
            path=out/f'{dataset}_fold{fold}_{arm}.npz';model.save(path);paths[arm]=path
        with np.load(oldrun/f'{dataset}_fold{fold}.npz') as old:
            np.testing.assert_array_equal(old['names'],names)
            if set(old['fit_subjects'])!=set(people):raise ValueError('original linear fold mismatch')
            montage=old['montage'].copy()
        for item in items:
            if folds[item['subject']]!=fold:continue
            crop=item['crop'];fields,seen,_,spectrum,prior=load_crop(crop,names,False,cache)
            prepared=prepare_x(fields['x'],seen,source.transfer)
            ex=np.array([(crop['row'],crop['start_s'],p) for p in crop['positions']])
            site=np.array([lookup[rows[i]['positions'][p].upper()] for i,t,p in ex]);batch=base_tokenizer_batch(rows,ex,scales,array_cache=cache,cache_limit=12)
            scale=input_scales(rows,ex,scales)
            delta=transfer_waveform(spectrum,montage)
            outputs={'legacy':{b:prior[b].T for b in ['S','O','sum']},
                     'montage':dict(S=prior['S'].T+delta.T,O=prior['O'].T-delta.T,sum=prior['sum'].T)}
            gains={}
            for arm in ['background','transient']:outputs[arm],gains[arm]=apply_gain(prepared,background,fitted[arm],arm=='transient',cfg['model']['gain_bounds'])
            flat={}
            for arm,wave in outputs.items():
                np.testing.assert_array_equal(wave['sum'],prior['sum'].T)
                closure=float(abs(wave['S']+wave['O']-wave['sum']).max())
                if not np.allclose(wave['S']+wave['O'],wave['sum'],rtol=1e-12,atol=1e-11):raise ValueError('source sum not conserved')
                maximum_closure=max(maximum_closure,closure)
                pred={b:w[site].reshape(len(ex),15,400) for b,w in wave.items()};flat[arm]=pred
                audit=joint_reconstruction_metrics(pred,batch,1,scale)
                if audit['support']['nonfinite_prediction_points'].any():raise ValueError('nonfinite source output')
                audit['metrics'].update(peak_metrics(pred,batch,scale))
                for k,v in audit['metrics'].items():metrics[arm+'__'+k].append(v)
            errors=metrics['transient__nmse'][-1][:,1,0]
            true_peak=np.max(np.where(batch['mask'],abs(batch['s']),0).reshape(len(ex),6000)*scale[:,None],axis=1)
            candidates=[(int(np.argmax(errors)),'new_maximum_S_NMSE',float(np.max(errors))),
                        (int(np.argmax(true_peak)),'maximum_true_S_peak',float(np.max(true_peak)))]
            for j,e in enumerate(ex):
                if tuple(e) in frozen_examples:candidates.append((j,'original_'+frozen_examples[tuple(e)],0.))
            for j,rule,rank in candidates:
                if rule in captures and rank<=captures[rule]['rank']:continue
                target={b:batch[k][j].reshape(6000)*scale[j] for b,k in [('sum','x'),('S','s'),('O','o')]}
                waves={a:{b:v[j].reshape(6000)*scale[j] for b,v in pred.items()} for a,pred in flat.items()}
                mask=batch['mask'][j].reshape(6000)
                peak=int(np.argmax(np.where(mask,abs(waves['legacy']['S']-target['S']),-1)))
                captures[rule]=dict(rank=rank,triple=ex[j],dataset=dataset,subject=item['subject'],record=rows[crop['row']]['record'],
                    site=names[site[j]],native_index=int(site[j]),fold=fold,scale=scale[j],unit=rows[crop['row']]['source_unit'],
                    target=target,outputs=waves,mask=mask,peak_sample=peak,x_array=fields['x'].T,observed=seen,
                    gain=gains['transient'][:,site[j]],amplitude=prepared['amplitude'][:,site[j]],background=background[:,site[j]],
                    checkpoint=str(paths['transient']))
            identities.extend(ex);fold_ids.extend([fold]*len(ex))
        write(out/f'{dataset}_progress.json',dict(stage='held_scoring',held_folds_finished=fold+1,CPU_seconds=time.process_time()))
    for rule,capture in captures.items():
        scalar={k:v for k,v in capture.items() if k not in ['target','outputs']}
        arrays={**scalar,**{'target_'+b:v for b,v in capture['target'].items()},
                **{a+'_'+b:v for a,w in capture['outputs'].items() for b,v in w.items()}}
        np.savez_compressed(out/f'{dataset}_{rule}_wave.npz',**arrays)
        model=load_gain_model(capture['checkpoint'])
        p=model.predict_continuous(capture['x_array'][None].reshape(1,len(names),15,400),dataset,capture['observed'][None])
        for b in ['S','O','sum']:
            expected=capture['outputs']['transient'][b]
            actual=p[b][0,capture['native_index']].reshape(6000)*capture['scale']
            np.testing.assert_array_equal(actual,expected)
    arrays={k:np.concatenate(v) for k,v in metrics.items()}
    with np.load(oldrun/f'{dataset}_metrics.npz') as old:
        previous={tuple(e):j for j,e in enumerate(old['examples'])}
        take=np.array([previous[tuple(e)] for e in identities])
        for arm in ['legacy','montage']:
            for k in METRICS:np.testing.assert_array_equal(arrays[arm+'__'+k],old[arm+'__'+k][take])
    np.savez_compressed(out/f'{dataset}_metrics.npz',examples=identities,fold=fold_ids,**arrays)
    return dataset,dict(scored=len(identities),crops=len(items),subjects=len(plan),max_abs_closure=maximum_closure,
                        original_legacy_and_montage_metrics_bitwise=True,frozen_reload_max_abs=replay_difference,
                        parameters_per_arm=int(sum(v.size for v in fitted.values())/2),CPU_seconds=time.process_time())


def load_crop(crop, names, targets, cache):
    rows, scales, source = STATE["rows"], STATE["scales"], STATE["source"]
    fields, seen, complete = base_tokenizer_joint_eeg(
        rows,
        crop if targets else dict(crop, positions=[]),
        scales,
        names,
        cache,
        allowed_splits=("train",),
    )
    x = fields["x"]
    original = source.predict_continuous(x.T.reshape(1, len(names), 15, 400))
    prior = {b: v.reshape(len(names), 6000).T for b, v in original.items()}
    return fields, seen, complete, np.fft.rfft(x, axis=0), prior


def subject_moments(crops, names, variance, cache):
    channels = len(names)
    cov = np.zeros((5, channels, channels, channels), complex)
    cross = np.zeros((5, channels, channels), complex)
    rows = STATE["rows"]
    counts = Counter(rows[c["row"]]["record"] for c in crops)
    usable = 0
    for crop in crops:
        fields, _, valid, spectrum, prior = load_crop(crop, names, True, cache)
        weight = 1 / (len(counts) * counts[rows[crop["row"]]["record"]])
        residual = component_residual(
            fields["s"], fields["o"], prior["S"], prior["O"], variance
        )
        labels = np.fft.rfft(residual[:, valid], axis=0)
        usable += int(valid.sum())
        for k, idx in enumerate(frequency_groups()):
            x = spectrum[idx]
            cov[k, valid] += weight * (x.conj().T @ x) / len(idx)
            cross[k, valid] += weight * (x.conj().T @ labels[idx]).T / len(idx)
    return cov, cross, usable


def dataset_fit(dataset):
    rows, scales, source = STATE["rows"], STATE["scales"], STATE["source"]
    cfg, out = STATE["config"], STATE["output"]
    names = source.metadata["positions"][dataset]
    plan, folds = STATE["plan"][dataset], STATE["folds"][dataset]
    lookup = {v.upper(): i for i, v in enumerate(names)}
    references = {n.upper(): v for n, v in scales["positions"]["eeg"][dataset].items()}
    variance = np.array(
        [
            [references[v.upper()]["target_variance"][b][0] for v in names]
            for b in ("s", "o")
        ]
    )
    cache = OrderedDict()
    moments = {
        s: subject_moments(crops, names, variance, cache) for s, crops in plan.items()
    }
    metrics = defaultdict(list)
    identities = []
    fold_ids = []
    fixed_examples = []
    hardest = None
    max_closure = 0.0
    for fold in range(4):
        fit = [s for s in plan if folds[s] != fold]
        gram = sum((moments[s][0] for s in fit)) / len(fit)
        cross = sum((moments[s][1] for s in fit)) / len(fit)
        local, montage = transfer_coefficients(
            gram, cross, cfg["model"]["ridge_fraction_of_input_diagonal"]
        )
        np.savez_compressed(
            out / f"{dataset}_fold{fold}.npz",
            local=local,
            montage=montage,
            names=names,
            fit_subjects=fit,
        )
        for subject in plan:
            if folds[subject] != fold:
                continue
            for crop in plan[subject]:
                _, _, _, spectrum, original = load_crop(crop, names, False, cache)
                ex = np.array(
                    [(crop["row"], crop["start_s"], p) for p in crop["positions"]]
                )
                site = np.array(
                    [lookup[rows[i]["positions"][p].upper()] for i, t, p in ex]
                )
                batch = base_tokenizer_batch(
                    rows, ex, scales, array_cache=cache, cache_limit=12
                )
                scale = input_scales(rows, ex, scales)
                rank = hashlib.sha256(ex[0].tobytes()).hexdigest()
                outputs = {}
                for arm, coef in zip(ARMS, (None, local, montage), strict=True):
                    delta = 0 if coef is None else transfer_waveform(spectrum, coef)
                    wave = {
                        "S": original["S"] + delta,
                        "O": original["O"] - delta,
                        "sum": original["sum"],
                    }
                    closure = np.max(np.abs(wave["S"] + wave["O"] - wave["sum"]))
                    if not np.allclose(
                        wave["S"] + wave["O"], wave["sum"], atol=1e-11, rtol=1e-12
                    ):
                        raise ValueError("sum not conserved")
                    max_closure = max(max_closure, float(closure))
                    outputs[arm] = {
                        b: w[:, site].T.reshape(len(ex), 15, 400)
                        for b, w in wave.items()
                    }
                    audit = joint_reconstruction_metrics(outputs[arm], batch, 1, scale)
                    if audit["support"]["nonfinite_prediction_points"].any():
                        raise ValueError("nonfinite prediction")
                    for k in METRICS:
                        metrics[f"{arm}__{k}"].append(audit["metrics"][k])
                error = metrics["montage__nmse"][-1][:, 1, 0]
                h = int(np.argmax(error))

                def capture(j):
                    return dict(
                        triple=ex[j],
                        dataset=dataset,
                        subject=subject,
                        record=rows[ex[j, 0]]["record"],
                        position=names[site[j]],
                        scale=scale[j],
                        target=batch["s"][j].reshape(-1) * scale[j],
                        mask=batch["mask"][j].reshape(-1),
                        **{
                            arm: outputs[arm]["S"][j].reshape(-1) * scale[j]
                            for arm in ARMS
                        },
                    )

                if hardest is None or error[h] > hardest[0]:
                    hardest = (float(error[h]), capture(h))
                if not fixed_examples or rank < fixed_examples[0]:
                    fixed_examples = (rank, capture(0))
                identities.extend(ex)
                fold_ids.extend([fold] * len(ex))
        write(out / f"{dataset}_progress.json", dict(held_folds_finished=fold + 1))
    for typ, sample in [
        ("fixed_hash", fixed_examples[1]),
        ("highest_montage_S_NMSE", hardest[1]),
    ]:
        np.savez_compressed(out / f"{dataset}_{typ}.npz", **sample)
    np.savez_compressed(
        out / f"{dataset}_metrics.npz",
        examples=identities,
        fold=fold_ids,
        **{k: np.concatenate(v) for k, v in metrics.items()},
    )
    return dataset, dict(
        scored=len(identities),
        complete_target_site_crops_for_moments=sum(v[2] for v in moments.values()),
        max_abs_reconstruction_closure=max_closure,
    )


def synthetic_bounded(cfg,out):
    """No measured-data read. The fixture separates a shared artifact and local S burst."""
    from src.tokenizers.eeg_transient_gain import band_indices
    def item(seed):
        rng=np.random.default_rng(seed);t=np.arange(6000)/200
        truth=np.sin(2*np.pi*10*t[:,None]+rng.uniform(-np.pi,np.pi,(1,4)))
        truth[:,seed%4]+=5*np.exp(-((t-18)/.22)**2)*np.sin(2*np.pi*10*t)
        x=truth+9*np.exp(-((t-(5+seed%5))/.23)**2)[:,None]*np.sin(2*np.pi*10*t)[:,None]
        p=prepare_x(x,np.ones(4,bool),{b:np.full(3001,.5) for b in 'SO'})
        spectrum=np.fft.rfft(truth.T,axis=-1);energy=[];cross=[]
        for b,idx in enumerate(band_indices()):
            f=np.zeros_like(spectrum);f[:,idx]=spectrum[:,idx]
            y=np.fft.irfft(f,n=6000,axis=-1).reshape(4,300,20)
            u=p['carriers'][b].reshape(4,300,20)
            energy.append((u*u).mean(-1));cross.append((u*y).mean(-1))
        return dict(prepared=p,frame_energy=np.array(energy),frame_cross=np.array(cross),weight=1/12),truth
    train=[item(i)[0] for i in range(12)];bg=fit_background(train)
    beta={a:fit_gain(train,bg,a=='transient',cfg['model']['ridge_fraction']) for a in ['background','transient']}
    errors=[];closure=0
    for seed in range(12,16):
        test,truth=item(seed);p=test['prepared'];row=[np.mean((p['prior']['S'].T-truth)**2)]
        for a in ['background','transient']:
            pred,g=apply_gain(p,bg,beta[a],a=='transient',cfg['model']['gain_bounds'])
            closure=max(closure,float(abs(pred['S']+pred['O']-p['prior']['sum']).max()))
            if g.min()<0 or g.max()>3:raise ValueError('gain bound fixture')
            row.append(np.mean((pred['S'].T-truth)**2))
        errors.append(row)
    mean=np.mean(errors,axis=0)
    if not (mean[2]<.8*mean[0] and mean[2]<.95*mean[1] and closure<1e-12):raise ValueError('synthetic separation failed')
    out.mkdir(parents=True,exist_ok=True)
    write(out/'synthetic.json',dict(status='pass',measured_reads=0,held_MSE=dict(zip(['legacy','background','transient'],mean.tolist())),closure=closure))


def summarize_bounded(cfg,out,data,rows,support):
    arms={};ex=data['examples'];datasets=sorted(support);rules=cfg['selection']
    for arm in ['legacy','montage','background','transient']:
        metrics={k.split('__',1)[1]:v for k,v in data.items() if k.startswith(arm+'__')}
        arms[arm]=base_training_aggregate(metrics,ex,rows)
        arms[arm]['fold_nmse']=[base_training_aggregate({'nmse':metrics['nmse'][data['fold']==f]},ex[data['fold']==f],rows)['metrics']['nmse']['macro'] for f in range(4)]
        arms[arm]['position_window_tails']={d:{str(q):np.nanquantile(metrics['nmse'][[rows[e[0]]['dataset']==d for e in ex]],q,axis=0).tolist() for q in [.5,.9,.95,.99,1.]} for d in datasets}
        arms[arm]['native_position_nmse']={d:{p:np.nanmean(metrics['nmse'][[rows[e[0]]['dataset']==d and rows[e[0]]['positions'][e[2]]==p for e in ex]],axis=0).tolist() for p in sorted({rows[e[0]]['positions'][e[2]] for e in ex if rows[e[0]]['dataset']==d})} for d in datasets}
    old=arms['legacy'];eligible=[]
    for arm in ['montage','background','transient']:
        current=arms[arm];m=current['metrics'];reference=old['metrics']
        ratio=np.ravel(m['nmse']['macro'])/np.ravel(reference['nmse']['macro'])
        d_ratios=np.array([np.ravel(m['nmse']['by_dataset'][d])/np.ravel(reference['nmse']['by_dataset'][d]) for d in datasets])
        f_ratio=np.array(current['fold_nmse'])[:,:,0]/np.array(old['fold_nmse'])[:,:,0]
        tail_ratios={d:{q:(np.ravel(current['position_window_tails'][d][q])/np.ravel(old['position_window_tails'][d][q])).tolist() for q in ['0.95','0.99']} for d in datasets}
        high_ratios={d:(np.ravel(m['true_high_nmse']['by_dataset'][d])/np.ravel(reference['true_high_nmse']['by_dataset'][d])).tolist() for d in datasets}
        gates=dict(components=bool(np.all(ratio[1:]<=1-rules['S_and_O_relative_improvement_min'])),
                   datasets=int(np.all(d_ratios[:,1:]<1,axis=1).sum())>=rules['min_improving_datasets_both_components'],
                   folds=int(np.all(f_ratio[:,1:]<1,axis=1).sum())>=rules['min_improving_folds_both_components'],
                   dataset_component_protection=bool(np.all(d_ratios[:,1:]<=rules['no_dataset_S_or_O_NMSE_ratio_above'])),
                   tail=all(all(max(r[1:])<=rules['P95_P99_component_ratio_max'] for r in qq.values()) for qq in tail_ratios.values()),
                   true_S_high=all(r[1]<=rules['true_high_S_ratio_max'] for r in high_ratios.values()),
                   theta_alpha=all(np.ravel(m[k]['macro'])[1]<=np.ravel(reference[k]['macro'])[1]+rules['S_band_macro_extra_abs_db_max'] and all(np.ravel(m[k]['by_dataset'][d])[1]<=np.ravel(reference[k]['by_dataset'][d])[1]+rules['S_band_dataset_extra_abs_db_max'] for d in datasets) for k in ['band_4_8_abs_db','band_8_13_abs_db']),
                   S_envelope=np.ravel(m['local_rms_correlation']['macro'])[1]>=np.ravel(reference['local_rms_correlation']['macro'])[1]-rules['S_envelope_correlation_drop_max'])
        current.update(gates=gates,NMSE_ratios_macro=ratio.tolist(),NMSE_ratios_by_dataset=dict(zip(datasets,d_ratios.tolist())),tail_ratios=tail_ratios,true_high_ratios=high_ratios)
        if arm=='transient' and all(gates.values()):eligible.append(arm)
    selected=min(eligible,key=lambda a:arms[a]['selection_score']) if eligible else None
    write(out/'summary.json',dict(status='complete',selected=selected,arms=arms,support=support,version=cfg['experiment_id'],validation_reads=0,
        interface='continuous_X_only_source_estimator_not_a_discrete_candidate',integer_count=None,bitrate=None,
        held_exposure='new_gain_and_background_fit_exclude_held_people; frozen_Wiener_and_G0_scales_historically_saw_all60_train_people',
        scoring='all_original400_train_arrays17040_scored_positions_in_fixed4_subject_folds',
        loss='bandwise_frame_energy_quadratic_surrogate; actual emitted fullwave scores determine gates; no held choice of gain or parameters'))
    print(json.dumps(dict(selected=selected,macro={a:v['metrics']['nmse']['macro'] for a,v in arms.items()}),ensure_ascii=False),flush=True)


def render_bounded(out,review=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors={'legacy':'#df8f00','montage':'#9d67b0','background':'#47a889','transient':'#0072b2'}
    figures=out/('waveforms_review_v2' if review else 'waveforms')
    if figures.exists() and any(figures.glob('*.png')):raise FileExistsError('use a fresh figure export version')
    figures.mkdir(exist_ok=True);manifest=[]
    for path in sorted(out.glob('*_wave.npz')):
        with np.load(path) as f:case={k:f[k] for k in f}
        true_zoom=review and 'maximum_true_S_peak' in path.stem
        peak=(int(np.argmax(np.where(case['mask'],abs(case['target_S']),-1))) if true_zoom else int(case['peak_sample']))/200
        near=float(np.clip(round(peak/2)*2,2,28))
        display_unit='native REFED export' if str(case['unit'])=='unknown_REFED_EEG_export' else str(case['unit'])
        spans=[(0,30),(max(0,min(28,peak-1)),min(30,max(2,peak+1))),(near-.25,near+.25)]
        t=np.arange(6000)/200;fig,axes=plt.subplots(3,3,figsize=(17,10),layout='constrained')
        for row,b in enumerate(['sum','S','O']):
            for col,(lo,hi) in enumerate(spans):
                ax=axes[row,col];target=case['target_'+b];mask=case['mask'].astype(bool)
                ax.plot(t,np.where(mask,target,np.nan),c='black',lw=.85,label='target')
                for arm,color in colors.items():ax.plot(t,case[arm+'_'+b],color=color,lw=.6,alpha=.8,label=arm)
                for boundary in np.arange(0,31,2):
                    if lo<=boundary<=hi:ax.axvline(boundary,c='.7',lw=.5,ls=':')
                if (~mask).any():ax.fill_between(t,0,1,where=~mask,transform=ax.get_xaxis_transform(),color='.8',alpha=.4)
                ax.set_xlim(lo,hi);ax.set_xlabel('time (s)');ax.set_ylabel(f'{b}\n({display_unit})');ax.grid(alpha=.2)
                if row==0:ax.set_title(['30s complete window','2s around '+('maximum true S peak' if true_zoom else 'maximum legacy S error'),'nearest integer boundary'][col])
        handles=[Line2D([],[],c='black',label='target')]+[Line2D([],[],c=v,label=k) for k,v in colors.items()]
        fig.legend(handles=handles,loc='outside lower center',ncol=5)
        fig.suptitle(f'{case["dataset"]} / {case["subject"]} / {case["record"]} / {case["site"]} / start {case["triple"][1]}s\n{path.stem}; zoom center {peak:.3f}s; same physical y coordinates; dotted boundaries every2s; gray=unscored')
        target=figures/(path.stem+'.png');fig.savefig(target,dpi=180);plt.close(fig)
        fig,axes=plt.subplots(5,1,figsize=(13,10),sharex=True,layout='constrained');tf=(np.arange(300)+.5)/10
        for b,(ax,band) in enumerate(zip(axes,[(1,4),(4,8),(8,13),(13,30),(30,45)])):
            ax.plot(tf,case['gain'][b],c=colors['transient'],lw=1);ax.axhline(1,c='black',ls=':',lw=.6);ax.set_ylim(-.1,3.1);ax.set_ylabel(f'{band[0]}–{band[1]}Hz\nS multiplier');ax.grid(alpha=.2)
            twin=ax.twinx();twin.plot(tf,case['amplitude'][b]*float(case['scale']),c='#df8f00',alpha=.5,lw=.7);twin.set_ylabel(f'X band RMS\n({display_unit})',color='#b27000')
        axes[-1].set_xlabel('time (s)');axes[-1].set_xlim(0,30)
        fig.suptitle(f'{case["dataset"]}/{case["subject"]}/{case["site"]}: emitted gain [0,3] and analytic band amplitude; no teacher inputs')
        fig.savefig(figures/(path.stem+'_gain.png'),dpi=180);plt.close(fig)
        manifest.append(dict(path=str(target),dataset=str(case['dataset']),subject=str(case['subject']),record=str(case['record']),position=str(case['site']),start_seconds=float(case['triple'][1]),zoom_seconds=peak,zoom_rule='maximum_true_S' if true_zoom else 'maximum_legacy_S_error',native_unit=str(case['unit'])))
    write(out/('figure_manifest_review_v2.json' if review else 'figure_manifest.json'),manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--fit-full", action="store_true")
    parser.add_argument("--render-review", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = args.output
    if args.render_review:
        render_bounded(out,review=True)
        return
    if args.render:
        render_bounded(out)
        return
    if args.synthetic:
        synthetic_bounded(cfg,out)
        return
    if cfg["data"]["allowed_splits"] != ["train"]:
        raise ValueError("training-only source separation")
    base = yaml.safe_load((ROOT / cfg["source"]["base"]).read_text())["source"]
    g0 = yaml.safe_load((ROOT / base["g0_config"]).read_text())
    manifest, scales, _, _ = base_tokenizer_training_data(g0, ROOT)
    rows = manifest["rows"]
    source = load_model(ROOT / base["waveform_dictionary"])
    if (
        source.metadata["input_identity"] != manifest["cfg_identity"]
        or source.metadata["scales_identity"] != manifest["scales_identity"]
    ):
        raise ValueError("Wiener/G0 identity mismatch")
    plan = json.loads((ROOT / base["native_train_plan"]).read_text())["plan"]
    folds = json.loads((ROOT / cfg["source"]["folds"]).read_text())
    expected = []
    for d, people in plan.items():
        if set(people) != set(folds[d]):
            raise ValueError("held panel mismatch")
        for s, crops in people.items():
            for c in crops:
                row = rows[c["row"]]
                if (row["dataset"], row["subject"], row["split"], row["modality"]) != (
                    d,
                    s,
                    "train",
                    "eeg",
                ):
                    raise ValueError("outside frozen training EEG support")
                expected.extend((c["row"], c["start_s"], p) for p in c["positions"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists() and not args.fit_full:
        raise FileExistsError("preserve completed run")
    if args.fit_full:
        prior_summary=json.loads((out/'summary.json').read_text())
        if prior_summary['selected']!='transient':raise ValueError('full fit requires all frozen CV gates to pass')
        if yaml.safe_load(args.config.read_text())!=json.loads((out/'resolved_config.json').read_text()):raise ValueError('full-fit configuration changed')
        if (out/'full_train_source.npz').exists():raise FileExistsError('preserve frozen full source')
    bounded=cfg.get('model',{}).get('estimator')=='bounded_transient_gain_v1'
    lock=open(out/'controller.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if bounded:
        if len(expected)!=17040 or sum(len(v) for v in plan.values())!=60 or sum(len(c) for v in plan.values() for c in v.values())!=400:
            raise ValueError('frozen training support changed')
        if not (out/'synthetic.json').exists():raise ValueError('run the synthetic contract first')
        write(out/'resolved_config.json',cfg)
    STATE.update(
        config=cfg,
        output=out,
        rows=rows,
        scales=scales,
        source=source,
        plan=plan,
        folds=folds,
    )
    if args.fit_full:
        with ProcessPoolExecutor(max_workers=cfg['resources']['dataset_workers'],mp_context=multiprocessing.get_context('fork')) as pool:
            full_support=dict(pool.map(bounded_full_fit,sorted(plan)))
        models={}
        for d in plan:
            with np.load(out/f'{d}_all_train_gain.npz') as f:models[d]={k:f[k].copy() for k in ['background','beta']}
        metadata=dict(schema='eeg_bounded_transient_source_v1',positions=source.metadata['positions'],transient=True,gain_bounds=cfg['model']['gain_bounds'],
            stage='continuous_only_before_integer_assignment',input_identity=source.metadata['input_identity'],scales_identity=source.metadata['scales_identity'],
            context_seconds=30,sample_rate_hz=200,frame_seconds=.1,features=FEATURES,fit_subjects={d:sorted(v) for d,v in plan.items()},
            input='X_and_observation_availability_only',objective='fixed_ridge_per_band_frame_energy_surrogate',source_exposure='all_original_train_people_only',experiment_id=cfg['experiment_id'])
        model=BoundedTransientSource(source.transfer,models,metadata);model.save(out/'full_train_source.npz')
        replay=load_gain_model(out/'full_train_source.npz');difference=0.
        for case_path in sorted(out.glob('*_wave.npz')):
            with np.load(case_path) as case:
                d=str(case['dataset']);x=case['x_array'][None].reshape(1,len(metadata['positions'][d]),15,400);seen=case['observed'][None]
                before=model.predict_continuous(x,d,seen);after=replay.predict_continuous(x,d,seen)
                for b in before:np.testing.assert_array_equal(before[b],after[b])
        write(out/'full_fit_summary.json',dict(status='complete',source=str(out/'full_train_source.npz'),support=full_support,frozen_source_replay_max_abs=difference,
            qualification='CV_continuous_source_gates_passed; not_a_discrete_tokenizer; full_train_outputs_not_rescored_as_held',validation_reads=0))
        return
    with ProcessPoolExecutor(
        max_workers=cfg["resources"]["dataset_workers"],
        mp_context=multiprocessing.get_context("fork"),
    ) as pool:
        support = dict(pool.map(bounded_dataset_fit if bounded else dataset_fit, sorted(plan)))
    buf = defaultdict(list)
    for d in sorted(plan):
        with np.load(out / f"{d}_metrics.npz") as f:
            for k in f:
                buf[k].append(f[k])
    data = {k: np.concatenate(v) for k, v in buf.items()}
    ex = data["examples"]
    if sorted(map(tuple, ex.tolist())) != sorted(expected):
        raise ValueError("missing held results")
    if bounded:
        summarize_bounded(cfg,out,data,rows,support)
        return
    arms = {}
    for a in ARMS:
        arrays = {k: data[a + "__" + k] for k in METRICS}
        arms[a] = base_training_aggregate(arrays, ex, rows)
        arms[a]["fold_nmse"] = [
            base_training_aggregate(
                {"nmse": arrays["nmse"][data["fold"] == f]}, ex[data["fold"] == f], rows
            )["metrics"]["nmse"]["macro"]
            for f in range(4)
        ]
    old = arms["legacy"]
    rules = cfg["selection"]
    eligible = []
    for a in ARMS[1:]:
        current = arms[a]
        nm = current["metrics"]["nmse"]
        ref = old["metrics"]["nmse"]
        per_dataset = np.array(
            [
                np.ravel(nm["by_dataset"][d]) / np.ravel(ref["by_dataset"][d])
                for d in sorted(plan)
            ]
        )
        ratios = np.ravel(nm["macro"]) / np.ravel(ref["macro"])
        folds_ratio = (
            np.array(current["fold_nmse"])[:, :, 0]
            / np.array(old["fold_nmse"])[:, :, 0]
        )
        gates = dict(
            components=bool(
                np.all(ratios[1:] <= 1 - rules["S_and_O_relative_improvement_min"])
            ),
            datasets=int(np.all(per_dataset[:, 1:] < 1, axis=1).sum())
            >= rules["min_improving_datasets_both_components"],
            folds=int(np.all(folds_ratio[:, 1:] < 1, axis=1).sum())
            >= rules["min_improving_folds_both_components"],
            tail=bool(
                np.all(
                    per_dataset[:, 1:] <= rules["no_dataset_S_or_O_NMSE_ratio_above"]
                )
            ),
        )
        current["gates"] = gates
        if all(gates.values()):
            eligible.append(a)
    chosen = (
        min(eligible, key=lambda a: arms[a]["selection_score"]) if eligible else None
    )
    write(
        out / "summary.json",
        dict(
            status="complete",
            selected=chosen,
            arms=arms,
            support=support,
            version=cfg["experiment_id"],
            validation_reads=0,
        ),
    )
    print(chosen, {k: v["metrics"]["nmse"]["macro"] for k, v in arms.items()})


if __name__ == "__main__":
    main()
