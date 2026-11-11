#!/usr/bin/env python3
"""Read-only cross-position and synthesis audit of the completed public S/O cache."""
import os
for _v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_v]='1'
import argparse
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import lru_cache
import json
from pathlib import Path
import resource
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import yaml
from src.data.so_supervision_cache import (parent_plan, window_projection, bind_reference,
    write_json, save_arrays, select_proxy_rows)
from src.data.dual_component_tokenizer import (voltage_power_cap_reference,
    voltage_power_cap_bank, voltage_power_cap_frames, voltage_power_cap_target,
    project_voltage_second_moment)

CONFIG=ROOT/'experiments/configs/physiology_semantic_tokenizer/so_supervision_synthesis_audit_v1.yaml'

def quant(x):
    x=np.asarray(x).ravel();x=x[np.isfinite(x)]
    return dict(n=len(x),**{k:float(v) for k,v in zip(('p10','p50','p90','p99','max'),np.quantile(x,[.1,.5,.9,.99,1]))}) if len(x) else dict(n=0)

def rms(x,axis=None):return np.sqrt(np.mean(np.asarray(x)**2,axis=axis))

def reference_disagreement(cov,a,p,mu,neighbors):
    """RMS difference in implied log power, for each site pair/shared electrode.

    Each individual conditional completion is mu + (p_i-a_i mu) cov a_i / var_i.
    Use temporal Gram moments to avoid materializing [site,time,electrode,band].
    """
    variance=np.einsum('ij,ij->i',a@cov,a)
    v=(a@cov)/variance[:,None];d=p-a@mu;gram=d.T@d/len(d)
    values=[];identical=[];pairs=[]
    for i in range(len(a)):
        for j in range(i):
            shared=sorted(set(neighbors[i])&set(neighbors[j]))
            if not shared:continue
            for c in shared:
                vi=v[i,c*5:c*5+5];vj=v[j,c*5:c*5+5]
                ms=(gram[i,i]*vi@vi+gram[j,j]*vj@vj-2*gram[i,j]*vi@vj)/5
                values.append(np.sqrt(max(0.,ms)))
                pairs.append((i,j,c))
            if np.array_equal(a[i],a[j]):identical.append(float(rms(p[:,i]-p[:,j])/np.sqrt(variance[i])))
    return np.array(values),np.array(identical),np.array(pairs,dtype=int).reshape(-1,3)

@lru_cache(maxsize=1)
def context(root):
    root=Path(root);cfg=json.loads((root/'resolved_config.json').read_text())['config']
    source=ROOT/cfg['source_ssm'];_,records,coords=parent_plan(str(source.resolve()))
    stats={}
    for ds in json.loads((root/'plan.json').read_text())['counts']:
        with np.load(root/'references'/(ds+'.npz')) as z:stats[ds]=dict(z)
    return source,records,coords,stats

def load_sites(root,g,meta):
    source,records,coords,stats=context(root)
    pc=coords['individual'][g['dataset']+'__'+g['subject']]['pc']
    rows=[];neighbors=[];all_positions=[]
    for m in g['members']:
        r=records[m['record_key']]
        for pair in r['pairs']:
            rows.append(window_projection(r,pc,[pair])[0]);neighbors.append(pair['channel_sets']['nearest6']);all_positions.append(pair['id'])
    valid=[];s=[];x=[]
    for i,sp in enumerate(meta['source_task_paths']):
        task=json.loads(Path(sp).read_text())
        assert task['identity']['pair']['id']==all_positions[i]
        if task['qualified']:
            with np.load(Path(sp).with_suffix('.npz')) as z:s.append(z['prediction'][:,0]);x.append(z['target'][:,0])
            valid.append(i)
    return stats[g['dataset']],np.array(rows)[valid],np.stack(s,1),np.stack(x,1),[neighbors[i] for i in valid],valid

def population_job(payload):
    root,out,g=payload;out=Path(out);dest=out/'population'/g['key'];start=time.monotonic()
    if dest.with_suffix('.json').exists():return json.loads(dest.with_suffix('.json').read_text())
    meta=json.loads((Path(root)/'windows'/g['key']).with_suffix('.json').read_text())
    result=dict(key=g['key'],dataset=g['dataset'],subject=g['subject'],EEG_available=meta['EEG_available'],status='failed')
    try:
        stats,a,p,x,neighbors,valid=load_sites(root,g,meta)
        vals,identical,pairs=reference_disagreement(stats['covariance'],a,p,stats['mean'],neighbors)
        selected=select_proxy_rows(stats,a,3);model=bind_reference(stats,a[selected],len(p))
        ell=stats['mean']+(p[:,selected]-a[selected]@stats['mean'])@model['lift'].T
        sd=np.sqrt(np.einsum('ij,ij->i',a@stats['covariance'],a))
        mismatch=rms(ell@a.T-p,axis=0)/sd;keep=np.ones(len(a),bool);keep[selected]=False
        arrays=dict(overlap_logpower_rms=vals,identical_projection_proxy_rms=identical,
                    unselected_proxy_mismatch_train_sd=mismatch[keep])
        result.update(status='completed',qualified_sites=len(a),selected_sites=len(selected),
            overlap_logpower_rms=quant(vals),unselected_proxy_mismatch_train_sd=quant(mismatch[keep]),
            selected_proxy_replay_max=float(np.max(abs(ell@a[selected].T-p[:,selected]))),
            reference_condition=float(model['completion_condition']),shared_electrodes=len(set(pairs[:,2])) if len(pairs) else 0)
        if meta['EEG_available']:
            with np.load(meta['array_file']) as z:
                cap=z['final_cap_ratio'];local=z['local_cap_ratio'];np.testing.assert_allclose(z['eeg_reference_projection'],a[selected],atol=1e-12,rtol=1e-12)
            interior=(np.arange(len(cap))*.1>=15)&(np.arange(len(cap))*.1<g['duration_s']-15)
            result.update(final_cap=quant(cap),local_cap_max=float(local.max()),
                final_cap_by_carrier=[quant(cap[:,b]) for b in range(6)],
                exceedance_by_carrier=(cap>1+1e-8).mean(0).tolist(),
                interior_exceedance=float((cap[interior]>1+1e-8).mean()),edge_exceedance=float((cap[~interior]>1+1e-8).mean()))
        save_arrays(dest.with_suffix('.npz'),**arrays)
    except Exception:result['error']=traceback.format_exc()
    result.update(seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(dest.with_suffix('.json'),result);return result

def audit_ratios(frames,power,correlation,batch=8):
    result=[];diagonal=[]
    for first in range(0,len(frames),batch):
        f=frames[first:first+batch];d=np.sqrt(power[first:first+batch]);cov=d[..., :,None]*correlation[None]*d[...,None,:]
        _,diag=project_voltage_second_moment(f,cov,1.)
        result.append(diag['max_input_eigenvalue'])
        diagonal.append(np.mean(f*f,axis=-1)/power[first:first+batch])
    return np.concatenate(result),np.concatenate(diagonal)

def staged_synthesis(x,model,proxy):
    """Same production projection, retaining carrier OLA to isolate refiltering."""
    rate=int(model['sample_rate_hz']);hop=rate//10
    frames,t=voltage_power_cap_frames(voltage_power_cap_bank(x,sample_rate_hz=rate),window=rate,hop=hop,sample_rate_hz=rate)
    power=voltage_power_cap_reference(model,proxy,t);acc=np.zeros((len(x)+rate,6,x.shape[1]));weight=np.zeros(len(x)+rate)
    win=np.hanning(rate+1)[:-1];local=[]
    for first in range(0,len(frames),8):
        d=np.sqrt(power[first:first+8]);cov=d[..., :,None]*model['correlation'][None]*d[...,None,:]
        estimated,diag=project_voltage_second_moment(frames[first:first+8],cov,1.);local.append(diag['max_output_eigenvalue'])
        for k,block in enumerate(estimated,first):
            start=k*hop;acc[start:start+rate]+=block.transpose(2,0,1)*win[:,None,None];weight[start:start+rate]+=win
    carriers=acc[rate//2:rate//2+len(x)]/weight[rate//2:rate//2+len(x),None,None]
    s=carriers.sum(1)
    same,_=voltage_power_cap_frames(carriers,window=rate,hop=hop,sample_rate_hz=rate)
    ola,ola_diagonal=audit_ratios(same,power,model['correlation'])
    after_bank=voltage_power_cap_bank(s,sample_rate_hz=rate)
    after,_=voltage_power_cap_frames(after_bank,window=rate,hop=hop,sample_rate_hz=rate)
    final,final_diagonal=audit_ratios(after,power,model['correlation'])
    return dict(s=s,local=np.concatenate(local),ola=ola,final=final,
                ola_diagonal=ola_diagonal,final_diagonal=final_diagonal,
                reference_power=power,refilter_change_rms=rms(after_bank-carriers,axis=0),carrier_rms=rms(carriers,axis=0))

def mechanism_job(payload):
    root,out,g=payload;out=Path(out);dest=out/'mechanism'/g['key'];start=time.monotonic()
    if dest.with_suffix('.json').exists():return json.loads(dest.with_suffix('.json').read_text())
    meta=json.loads((Path(root)/'windows'/g['key']).with_suffix('.json').read_text())
    result=dict(key=g['key'],dataset=g['dataset'],subject=g['subject'],status='failed')
    try:
        stats,a,p,observed,neighbors,valid=load_sites(root,g,meta)
        with np.load(meta['array_file']) as z:x=z['eeg_x'];cached_s=z['eeg_s'];cached_final=z['final_cap_ratio']
        selected=select_proxy_rows(stats,a,3);model=bind_reference(stats,a[selected],len(p))
        stages=staged_synthesis(x,model,p[:,selected]);np.testing.assert_allclose(stages['s'],cached_s,rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(stages['final'],cached_final,rtol=1e-7,atol=1e-7)
        const=staged_synthesis(x,model,None)
        arrays={}
        for name,item in [('dynamic',stages),('constant',const)]:
            result[name]={}
            for stage in ('local','ola','final','final_diagonal'):
                v=item[stage];result[name][stage]=dict(quant(v),exceedance=float((v>1+1e-8).mean()))
                arrays[name+'_'+stage]=v
            result[name]['by_carrier']={stage:[dict(quant(item[stage][:,b]),exceedance=float((item[stage][:,b]>1+1e-8).mean())) for b in range(6)] for stage in ('ola','final')}
            result[name]['s_over_x_rms']=float(rms(item['s'])/rms(x))
            result[name]['refilter_change_relative_rms']=quant(item['refilter_change_rms']/np.maximum(item['carrier_rms'],1e-300))
        # Geometry-only pair: first maximum overlap, independent of solved values.
        candidates=[(len(set(neighbors[i])&set(neighbors[j])),i,j) for i in range(len(a)) for j in range(i)]
        overlap=max(v[0] for v in candidates);_,i,j=next(v for v in candidates if v[0]==overlap)
        shared=sorted(set(neighbors[i])&set(neighbors[j]));single=[]
        for k in (i,j):
            mm=bind_reference(stats,a[k:k+1],len(p))
            ss=voltage_power_cap_target(x,mm,p[:,k:k+1],audit_final=False,frame_batch_size=8)['s'];single.append(ss)
        delta=single[0]-single[1];den=rms(x,axis=0)
        correlations=[]
        for c in shared:correlations.append(float(np.corrcoef(single[0][:,c],single[1][:,c])[0,1]))
        result['counterfactual']=dict(site_indices=[valid[i],valid[j]],sites=[meta['fNIRS_positions'][valid[k]] for k in (i,j)],
            shared_channels=[g['EEG_channels'][c] for c in shared],S_difference_over_X_rms=(rms(delta,axis=0)/den)[shared].tolist(),
            S_correlation=correlations,S_over_X_rms=[(rms(v,axis=0)/den)[shared].tolist() for v in single],
            O_difference_is_negative_S_difference_max=float(abs(((x-single[0])-(x-single[1]))+delta).max()))
        arrays.update(time_s=np.arange(len(x))/g['rate'],shared_channel_indices=np.array(shared),
            x_shared=x[:,shared],joint_s_shared=cached_s[:,shared],single_i_s_shared=single[0][:,shared],single_j_s_shared=single[1][:,shared])
        result['status']='completed';save_arrays(dest.with_suffix('.npz'),**arrays)
    except Exception:result['error']=traceback.format_exc()
    result.update(seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(dest.with_suffix('.json'),result);return result

def validate_source(root):
    root=Path(root)
    manifest=json.loads((root/'cache_manifest.json').read_text());verify=json.loads((root/'verification.json').read_text())
    assert manifest['schema']=='physiology_so_supervision_v2' and manifest['status']=='completed' and verify['status']=='passed' and not verify['issues']
    plan=json.loads((root/'plan.json').read_text())
    assert all(g['split'] in ('train','validation','test') for g in plan['groups'])
    assert len(manifest['windows'])==len(plan['groups'])
    return plan

def position_metrics(x,si,sj,joint):
    delta=si-sj;den=rms(x,axis=0);sp=np.sqrt((rms(si,axis=0)**2+rms(sj,axis=0)**2)/2)
    oi=x-si;oj=x-sj;op=np.sqrt((rms(oi,axis=0)**2+rms(oj,axis=0)**2)/2)
    corr=lambda u,v:[float(np.corrcoef(u[:,c],v[:,c])[0,1]) for c in range(u.shape[1])]
    return dict(S_difference_over_X_rms=(rms(delta,axis=0)/den).tolist(),
        S_difference_over_pooled_S_rms=(rms(delta,axis=0)/sp).tolist(),
        O_difference_over_pooled_O_rms=(rms(delta,axis=0)/op).tolist(),
        S_correlation=corr(si,sj),O_correlation=corr(oi,oj),
        S_amplitude_ratio_i_over_j=(rms(si,axis=0)/rms(sj,axis=0)).tolist(),
        S_over_X_rms=[(rms(v,axis=0)/den).tolist() for v in (si,sj)],
        joint_difference_over_X_rms=[(rms(v-joint,axis=0)/den).tolist() for v in (si,sj)],
        O_difference_negative_S_error=float(np.max(abs((oi-oj)+delta))))

def position_plan(root,out):
    source=validate_source(root);chosen={}
    for g in sorted(source['groups'],key=lambda g:g['key']):
        key=(g['dataset'],g['subject'])
        if key in chosen or g['duration_s']!=90:continue
        d=json.loads((Path(root)/'windows'/g['key']).with_suffix('.json').read_text())
        if d['EEG_available']:chosen[key]=g
    assert len(chosen)==len({(g['dataset'],g['subject']) for g in source['groups']})
    plan=dict(source=str(root),groups=list(chosen.values()),selection='one first lexicographic successful 90s window per dataset/subject; geometry-only maximum-overlap position pair')
    write_json(Path(out)/'position_plan.json',plan)
    cfg=yaml.safe_load((CONFIG.parent/'so_supervision_position_audit_v1.yaml').read_text())
    write_json(Path(out)/'position_resolved_config.json',cfg);return plan

def position_job(payload):
    root,out,g=payload;dest=Path(out)/'position'/g['key'];start=time.monotonic()
    if dest.with_suffix('.json').exists():return json.loads(dest.with_suffix('.json').read_text())
    result=dict(key=g['key'],dataset=g['dataset'],subject=g['subject'],status='failed')
    try:
        meta=json.loads((Path(root)/'windows'/g['key']).with_suffix('.json').read_text())
        stats,a,p,observed,neighbors,valid=load_sites(root,g,meta)
        with np.load(meta['array_file']) as z:x=z['eeg_x'];joint=z['eeg_s']
        pairs=[(len(set(neighbors[i])&set(neighbors[j])),i,j) for i in range(len(a)) for j in range(i)]
        n=max(v[0] for v in pairs);_,i,j=next(v for v in pairs if v[0]==n);shared=sorted(set(neighbors[i])&set(neighbors[j]));single=[]
        for k in (i,j):
            model=bind_reference(stats,a[k:k+1],len(p));v=voltage_power_cap_target(x,model,p[:,k:k+1],audit_final=False,frame_batch_size=8)
            single.append(v['s'][:,shared]);del v
        result.update(status='completed',sites=[meta['fNIRS_positions'][valid[k]] for k in (i,j)],
            shared_channels=[g['EEG_channels'][c] for c in shared],metrics=position_metrics(x[:,shared],*single,joint[:,shared]))
        save_arrays(dest.with_suffix('.npz'),time_s=np.arange(len(x))/g['rate'],x=x[:,shared],single_i_s=single[0],single_j_s=single[1],joint_s=joint[:,shared])
    except Exception:result['error']=traceback.format_exc()
    result.update(seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(dest.with_suffix('.json'),result);return result

def temporal_job(payload):
    root,out,g=payload;dest=Path(out)/'temporal'/g['key'];start=time.monotonic()
    if dest.with_suffix('.json').exists():return json.loads(dest.with_suffix('.json').read_text())
    result=dict(key=g['key'],dataset=g['dataset'],subject=g['subject'],status='failed')
    try:
        meta=json.loads((Path(root)/'windows'/g['key']).with_suffix('.json').read_text())
        stats,a,p,observed,neighbors,valid=load_sites(root,g,meta);selected=select_proxy_rows(stats,a,3)
        model=bind_reference(stats,a[selected],len(p));dynamic=p[:,selected]
        flat=np.broadcast_to((dynamic.sum(0)+dynamic[-1])/(len(p)+1),dynamic.shape).copy()
        times=np.arange(len(p)+1)*.1
        original=voltage_power_cap_reference(model,dynamic,times);fixed=voltage_power_cap_reference(model,flat,times)
        np.testing.assert_allclose(np.log(original).mean(0),np.log(fixed).mean(0),rtol=1e-12,atol=1e-12)
        with np.load(meta['array_file']) as z:x=z['eeg_x']
        stages=staged_synthesis(x,model,flat)
        for name in ('local','ola','final','final_diagonal'):
            result[name]=dict(quant(stages[name]),exceedance=float((stages[name]>1+1e-8).mean()))
        result.update(status='completed',s_over_x_rms=float(rms(stages['s'])/rms(x)),
            preserved_mean_log_power_error=float(abs(np.log(original).mean(0)-np.log(fixed).mean(0)).max()))
        save_arrays(dest.with_suffix('.npz'),**{k:stages[k] for k in ('local','ola','final','final_diagonal')})
    except Exception:result['error']=traceback.format_exc()
    result.update(seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(dest.with_suffix('.json'),result);return result

def summarize_positions(out):
    out=Path(out);plan=json.loads((out/'position_plan.json').read_text());by=defaultdict(list);failed=[]
    for g in plan['groups']:
        r=json.loads((out/'position'/g['key']).with_suffix('.json').read_text())
        if r['status']=='completed':by[g['dataset']].append(r)
        else:failed.append(r)
    summary=dict(status='completed' if not failed else 'completed_with_failures',planned=len(plan['groups']),failed=failed,datasets={})
    metrics=('S_difference_over_X_rms','S_difference_over_pooled_S_rms','O_difference_over_pooled_O_rms','S_correlation','O_correlation','S_amplitude_ratio_i_over_j')
    for ds,rr in by.items():
        summary['datasets'][ds]=dict(subjects=len(rr),electrode_comparisons=sum(len(r['shared_channels']) for r in rr),
            pooled_electrodes={k:quant(np.concatenate([r['metrics'][k] for r in rr])) for k in metrics},
            subject_medians={k:quant([np.median(r['metrics'][k]) for r in rr]) for k in metrics},
            O_difference_negative_S_error=max(r['metrics']['O_difference_negative_S_error'] for r in rr))
    write_json(out/'position_summary.json',summary);return summary

def summarize_temporal(out):
    out=Path(out);plan=json.loads((out/'plan.json').read_text());rows=[];comparisons=defaultdict(list)
    for g in plan['mechanism']:
        t=json.loads((out/'temporal'/g['key']).with_suffix('.json').read_text());rows.append(t)
        m=json.loads((out/'mechanism'/g['key']).with_suffix('.json').read_text())
        if t['status']!='completed' or m['status']!='completed':continue
        comparisons['dynamic'].append(m['dynamic']);comparisons['training_constant'].append(m['constant']);comparisons['matched_mean_log_power_constant'].append(t)
    result=dict(status='completed' if len(comparisons['dynamic'])==len(rows) else 'incomplete',windows=len(rows),rows=rows,
        preserved_mean_log_power_error=max(r.get('preserved_mean_log_power_error',0) for r in rows),
        aggregation='Across-window distribution of each window statistic; same12 windows, no new SSM fits; mean log power, not arithmetic mean power, is matched.',controls={})
    for condition,rr in comparisons.items():
        result['controls'][condition]={stage:{key:quant([r[stage][key] for r in rr]) for key in ('p50','p99','max','exceedance')} for stage in ('local','ola','final','final_diagonal')}
    write_json(out/'temporal_summary.json',result);return result

def render(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);summary=json.loads((out/'summary.json').read_text());position=json.loads((out/'position_summary.json').read_text())
    datasets=list(summary['datasets']);names=['Single','REFED','Sim','Visual'];colors=['#2374ab','#b85b24','#39844d','#8757a6']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    ax=axs[0,0]
    for i,ds in enumerate(datasets):
        q=summary['datasets'][ds]['final_cap_window_medians'];ax.errorbar(i,q['p50'],yerr=[[q['p50']-q['p10']],[q['p90']-q['p50']]],fmt='o',color=colors[i],capsize=4)
    ax.axhline(1,color='black',ls='--');ax.set(xticks=range(4),xticklabels=names,ylabel='Largest whitened power ratio',title='A. Final cap: window median, P10-P90 across windows')
    ax=axs[0,1]
    for i,ds in enumerate(datasets):ax.plot(range(6),[v['p50'] for v in summary['datasets'][ds]['by_carrier_ratio_window_medians']],'-o',label=names[i],color=colors[i])
    ax.axhline(1,color='black',ls='--');ax.set(xticks=range(6),xticklabels=['1-4','4-8','8-13','13-30','30-45','Rest'],xlabel='Carrier (Hz; Rest = exact voltage complement)',ylabel='Median of window-median ratios',title='B. Low-frequency carriers have larger excess');ax.legend()
    for ax,key,title,factor,ylabel in [(axs[1,0],'overlap_logpower_rms','C. All shared-electrode reference comparisons',10/np.log(10),'RMS log-power difference (dB)'),(axs[1,1],'unselected_proxy_mismatch_train_sd','D. Unselected SSM proxy mismatch',1,'RMS mismatch / training proxy SD')]:
        for i,ds in enumerate(datasets):
            q=summary['datasets'][ds]['pooled'][key];ax.errorbar(i,q['p50']*factor,yerr=[[(q['p50']-q['p10'])*factor],[(q['p90']-q['p50'])*factor]],fmt='o',color=colors[i],capsize=4)
        ax.set(xticks=range(4),xticklabels=names,ylabel=ylabel,title=title+'\nP50 and P10-P90; dependent comparisons')
    fig.savefig(out/'population.png',dpi=220);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained');good=[r for r in summary['mechanism'] if r['status']=='completed']
    for r in good:
        color=colors[datasets.index(r['dataset'])]
        for ax,condition in [(axs[0,0],'dynamic'),(axs[0,1],'constant')]:
            ax.plot(range(3),[r[condition][s]['p50'] for s in ('local','ola','final')],'-o',color=color,alpha=.65)
    for ax,title in [(axs[0,0],'A. Current dynamic reference'),(axs[0,1],'B. Constant training-reference control')]:
        ax.axhline(1,color='black',ls='--');ax.set(xticks=range(3),xticklabels=['Local projection','Carrier overlap-add','Sum + refilter'],ylabel='Median largest whitened power ratio',ylim=(.98,1.75),title=title+'\nEach line: one of 12 diagnostic windows')
    for name,color in zip(names,colors):axs[0,0].plot([],[],color=color,label=name)
    axs[0,0].legend(ncol=2,fontsize=8,loc='upper left')
    for i,ds in enumerate(datasets):
        a=position['datasets'][ds]['subject_medians']['S_difference_over_X_rms'];b=position['datasets'][ds]['subject_medians']['S_correlation']
        for ax,q,factor in [(axs[1,0],a,100),(axs[1,1],b,1)]:ax.errorbar(i,q['p50']*factor,yerr=[[(q['p50']-q['p10'])*factor],[(q['p90']-q['p50'])*factor]],fmt='o',color=colors[i],capsize=4)
    axs[1,0].set(xticks=range(4),xticklabels=names,ylabel='RMS(S_i - S_j) / RMS(X), %',title='C. Position-dependent voltage difference\nOne window per subject; subject median and P10-P90')
    axs[1,1].set(xticks=range(4),xticklabels=names,ylabel='Correlation(S_i, S_j)',title='D. Same-electrode morphology agreement\nOne geometry-selected pair per subject')
    fig.savefig(out/'mechanism_and_positions.png',dpi=220);plt.close(fig)
    plan=json.loads((out/'position_plan.json').read_text());fig,axs=plt.subplots(4,2,figsize=(14,11),layout='constrained');examples=[]
    for row,ds in enumerate(datasets):
        rr=[json.loads((out/'position'/g['key']).with_suffix('.json').read_text()) for g in plan['groups'] if g['dataset']==ds]
        rr=[r for r in rr if r['status']=='completed'];rr.sort(key=lambda r:np.median(r['metrics']['S_difference_over_X_rms']));r=rr[len(rr)//2]
        c=int(np.argsort(r['metrics']['S_difference_over_X_rms'])[len(r['shared_channels'])//2]);examples.append(dict(key=r['key'],electrode=r['shared_channels'][c],sites=r['sites'],selection='median subject and median shared electrode by normalized S difference; display30-32s, metrics use full90s'))
        with np.load((out/'position'/r['key']).with_suffix('.npz')) as z:
            t=z['time_s'];mask=(t>=30)&(t<32);x=z['x'][:,c];scale=rms(x)
            si=z['single_i_s'][:,c];sj=z['single_j_s'][:,c];joint=z['joint_s'][:,c]
            for ax in axs[row]:ax.plot(t[mask],x[mask]/scale,color='#a5adb5',alpha=.55,lw=.8,label='X')
            for v,label,color in [(si,'S: site i','#287bb5'),(sj,'S: site j','#e17c26'),(joint,'S: joint 3','#3a8b54')]:axs[row,0].plot(t[mask],v[mask]/scale,label=label,color=color,lw=1)
            for v,label,color in [(x-si,'O: site i','#287bb5'),(x-sj,'O: site j','#e17c26')]:axs[row,1].plot(t[mask],v[mask]/scale,label=label,color=color,lw=1)
        for col,ax in enumerate(axs[row]):
            ax.set(xlabel='Time within full window (s)',ylabel='Voltage / full-window X RMS',title=f"{names[row]} / {r['subject']} / {r['shared_channels'][c]} | {'S' if col==0 else 'O'}\ni={r['sites'][0]}, j={r['sites'][1]}")
            if row==0:ax.legend(ncol=2,fontsize=8)
    fig.savefig(out/'same_electrode_waveforms.png',dpi=220);plt.close(fig)
    write_json(out/'figure_examples.json',examples)

def make_plan(root,out):
    plan=validate_source(root);by=defaultdict(list)
    for g in plan['groups']:
        d=json.loads((Path(root)/'windows'/g['key']).with_suffix('.json').read_text())
        if d['EEG_available'] and g['duration_s']==90:by[g['dataset']].append((g,d['final_cap_max']))
    chosen=[]
    for ds,items in sorted(by.items()):
        used=set();logs=np.log([v for g,v in items])
        for q in (.1,.5,.9):
            target=np.quantile(logs,q)
            order=sorted(range(len(items)),key=lambda i:(abs(logs[i]-target),items[i][0]['key']))
            g=next(items[i][0] for i in order if items[i][0]['subject'] not in used);used.add(g['subject']);chosen.append(g)
    result=dict(source=str(root),groups=plan['groups'],mechanism=chosen,selection='metadata final_cap_max log quantiles .1/.5/.9; distinct subjects per dataset; diagnostic examples, not population random sample')
    write_json(Path(out)/'plan.json',result);return result

def summarize(out,plan):
    out=Path(out);rows=[];by=defaultdict(list)
    for g in plan['groups']:
        p=out/'population'/g['key'];r=json.loads(p.with_suffix('.json').read_text());rows.append(r);by[g['dataset']].append(r)
    result=dict(status='completed',planned=len(rows),completed=sum(r['status']=='completed' for r in rows),datasets={},mechanism=[])
    for ds,rr in by.items():
        good=[r for r in rr if r['status']=='completed'];metrics=defaultdict(list)
        for r in good:
            with np.load((out/'population'/r['key']).with_suffix('.npz')) as z:
                for k in z.files:metrics[k].append(z[k])
        result['datasets'][ds]=dict(windows=len(rr),completed=len(good),subjects=len({r['subject'] for r in good}),
            pooled={k:quant(np.concatenate(v)) for k,v in metrics.items()},
            window_medians={k:quant([r[k]['p50'] for r in good if r[k]['n']]) for k in ('overlap_logpower_rms','unselected_proxy_mismatch_train_sd')},
            subject_medians={k:quant([np.median([r[k]['p50'] for r in good if r['subject']==s and r[k]['n']]) for s in sorted({r['subject'] for r in good})]) for k in ('overlap_logpower_rms','unselected_proxy_mismatch_train_sd')},
            final_cap_window_medians=quant([r['final_cap']['p50'] for r in good if r['EEG_available']]),
            final_cap_window_p99=quant([r['final_cap']['p99'] for r in good if r['EEG_available']]),
            interior_exceedance_window_medians=quant([r['interior_exceedance'] for r in good if r['EEG_available']]),
            edge_exceedance_window_medians=quant([r['edge_exceedance'] for r in good if r['EEG_available']]),
            by_carrier_exceedance_window_medians=[quant([r['exceedance_by_carrier'][b] for r in good if r['EEG_available']]) for b in range(6)],
            by_carrier_ratio_window_medians=[quant([r['final_cap_by_carrier'][b]['p50'] for r in good if r['EEG_available']]) for b in range(6)])
    for g in plan['mechanism']:
        p=out/'mechanism'/g['key']
        if p.with_suffix('.json').exists():result['mechanism'].append(json.loads(p.with_suffix('.json').read_text()))
    result['population_failures']=[r for r in rows if r['status']!='completed']
    result['mechanism_completed']=sum(r['status']=='completed' for r in result['mechanism'])
    result['status']='completed' if result['completed']==result['planned'] and result['mechanism_completed']==len(plan['mechanism']) else 'incomplete'
    write_json(out/'summary.json',result);return result

SPECTRAL_BANDS=((1,4),(4,8),(8,13),(13,30),(30,45))
SPECTRAL_SIGNALS=('X','S_i','S_j','S_joint','O_i','O_j','O_joint')

def band_integrals(f,psd):
    """Disjoint left-closed FFT bins; frequency is the leading dimension."""
    df=f[1]-f[0]
    return np.stack([psd[(f>=lo)&(f<hi)].sum(0)*df for lo,hi in SPECTRAL_BANDS])

def spectral_metrics(signals,rate):
    from scipy.signal import welch,csd
    values=np.stack([signals[k] for k in SPECTRAL_SIGNALS],axis=-1)
    options=dict(fs=rate,window='hann',nperseg=rate*8,noverlap=rate*4,detrend='constant',axis=0,scaling='density')
    f,p=welch(values,**options);p=p.mean(1)
    _,cross=csd(signals['S_joint'],signals['O_joint'],**options);cross=cross.real.mean(1)
    scale=max(float(p[:,0].max()),1e-300)
    error=float(abs(p[:,0]-p[:,3]-p[:,6]-2*cross).max()/scale)
    if error>1e-10:raise ValueError('Spectral S/O closure failed')
    bands=band_integrals(f,p);cross_band=band_integrals(f,cross)
    shape=bands/bands.sum(0,keepdims=True);retention=bands/bands[:,0,None]
    outside=np.stack([p[(f>0)&(f<1)].sum(0),p[f>=45].sum(0)])*(f[1]-f[0])
    return dict(frequency=f,psd=p,band_power=bands,shape=shape,retention=retention,cross_ratio=2*cross_band/bands[:,0],
        outside_over_X_same_range=outside/outside[:,0,None],closure_error=np.array(error))

def spectrum_job(payload):
    root,out,g=payload;out=Path(out);dest=out/'spectra'/g['key'];start=time.monotonic()
    if dest.with_suffix('.json').exists():return json.loads(dest.with_suffix('.json').read_text())
    result=dict(key=g['key'],dataset=g['dataset'],subject=g['subject'],status='failed')
    try:
        source=(out.parent/'position'/g['key']).with_suffix('.json');meta=json.loads(source.read_text())
        assert meta['status']=='completed' and meta['key']==g['key']
        with np.load(source.with_suffix('.npz')) as z:
            x=z['x'];si=z['single_i_s'];sj=z['single_j_s'];s=z['joint_s']
            np.testing.assert_allclose(z['time_s'],np.arange(len(x))/g['rate'],rtol=0,atol=1e-12)
        signals=dict(X=x,S_i=si,S_j=sj,S_joint=s,O_i=x-si,O_j=x-sj,O_joint=x-s)
        arrays=spectral_metrics(signals,g['rate']);save_arrays(dest.with_suffix('.npz'),**arrays)
        result.update(status='completed',sites=meta['sites'],channels=meta['shared_channels'],
            shape=arrays['shape'].tolist(),retention=arrays['retention'].tolist(),
            S_i_vs_j_band_power_dB=(10*np.log10(arrays['band_power'][:,1]/arrays['band_power'][:,2])).tolist(),
            cross_ratio=arrays['cross_ratio'].tolist(),closure_error=float(arrays['closure_error']))
    except Exception:result['error']=traceback.format_exc()
    result.update(seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(dest.with_suffix('.json'),result);return result

def whiten_moments(moment,power,correlation):
    scale=np.sqrt(power);r=np.linalg.cholesky(correlation)
    v=moment/(scale[..., :,None]*scale[...,None,:]);v=np.linalg.solve(r,v)
    v=np.linalg.solve(r,v.swapaxes(-1,-2)).swapaxes(-1,-2)
    return (v+v.swapaxes(-1,-2))/2

def normalized_overlap(blocks,hop,taper):
    """Small analytic-test utility; blocks [frame,time,...]."""
    n=(len(blocks)-1)*hop+len(taper);out=np.zeros((n,*blocks.shape[2:]));weight=np.zeros(n)
    for k,b in enumerate(blocks):
        sl=slice(k*hop,k*hop+len(taper));out[sl]+=b*taper.reshape((-1,)+(1,)*(b.ndim-1));weight[sl]+=taper
    np.divide(out,weight.reshape((-1,)+(1,)*(out.ndim-1)),out=out,where=weight.reshape((-1,)+(1,)*(out.ndim-1))>0)
    return out,weight

def overlap_analysis(x,model,proxy):
    """Exact Jensen decomposition for the production Hann carrier synthesis.

    Accumulate weighted outer products in 100ms bins, then sum ten bins per
    analysis frame. Memory is O(time_bins*carriers*channels^2), not native time.
    Only interior frames enter the diagnostics, so endpoint reflection does not
    alter the mixture identity.
    """
    from scipy.signal import periodogram
    rate=int(model['sample_rate_hz']);hop=rate//10;length=len(x);c=x.shape[1]
    frames,t=voltage_power_cap_frames(voltage_power_cap_bank(x,sample_rate_hz=rate),window=rate,hop=hop,sample_rate_hz=rate)
    power=voltage_power_cap_reference(model,proxy,t);win=np.hanning(rate+1)[:-1]
    hann=np.zeros((length+rate,6,c));box=np.zeros_like(hann);wh=np.zeros(length+rate);wb=np.zeros_like(wh)
    for k in range(len(frames)):wh[k*hop:k*hop+rate]+=win;wb[k*hop:k*hop+rate]+=1
    mix_bins=np.zeros(((length+rate)//hop,6,c,c));weighted_ratios=[];local_ratios=[];adjacent=[];cosines=[];previous=None;local_psd=[]
    interior=(t>=15)&(t<75)
    for first in range(0,len(frames),8):
        d=np.sqrt(power[first:first+8]);cov=d[..., :,None]*model['correlation'][None]*d[...,None,:]
        estimated,diagnostic=project_voltage_second_moment(frames[first:first+8],cov,1.)
        local_ratios.append(diagnostic['max_output_eigenvalue'])
        weighted=(estimated*win)@estimated.swapaxes(-1,-2)/win.sum()
        weighted_ratios.append(np.linalg.eigvalsh(whiten_moments(weighted,power[first:first+8],model['correlation']))[...,-1])
        for k,block in enumerate(estimated,first):
            start=k*hop;sl=slice(start,start+rate)
            hann[sl]+=block.transpose(2,0,1)*win[:,None,None];box[sl]+=block.transpose(2,0,1)
            alpha=np.divide(win,wh[sl],out=np.zeros_like(win),where=wh[sl]>0)
            bins=(block*np.sqrt(alpha)).reshape(6,c,10,hop).transpose(2,0,1,3)
            mix_bins[k:k+10]+=bins@bins.swapaxes(-1,-2)
            if interior[k]:
                f,p=periodogram(block.sum(0),fs=rate,window='hann',detrend='constant',axis=-1,scaling='density')
                local_psd.append(p.mean(0))
                if previous is not None:
                    a=previous[:,:,hop:];b=block[:,:,:-hop];aa=np.sum(a*a,axis=(1,2));bb=np.sum(b*b,axis=(1,2))
                    adjacent.append(np.sqrt(np.sum((a-b)**2,axis=(1,2))/((aa+bb)/2)))
                    cosines.append(np.sum(a*b,axis=(1,2))/np.sqrt(aa*bb))
            previous=block.copy()
    section=slice(rate//2,rate//2+length)
    hann=hann[section]/wh[section,None,None];box=box[section]/wb[section,None,None]
    cumulative=np.concatenate([np.zeros_like(mix_bins[:1]),np.cumsum(mix_bins,axis=0)])
    mixture=(cumulative[10:]-cumulative[:-10])/rate
    del cumulative,mix_bins
    mix_white=whiten_moments(mixture[interior],power[interior],model['correlation']);del mixture
    arrays=dict(local_ratio=np.concatenate(local_ratios)[interior],hann_weighted_local_ratio=np.concatenate(weighted_ratios)[interior],
        adjacent_overlap_relative_rms=np.stack(adjacent),adjacent_overlap_cosine=np.stack(cosines),
        mixture_ratio=np.linalg.eigvalsh(mix_white)[...,-1])
    for name,carriers in [('hann',hann),('boxcar',box)]:
        ff,_=voltage_power_cap_frames(carriers,window=rate,hop=hop,sample_rate_hz=rate)
        ff=ff[interior];moment=ff@ff.swapaxes(-1,-2)/rate
        white=whiten_moments(moment,power[interior],model['correlation']);del moment,ff
        arrays[name+'_ratio']=np.linalg.eigvalsh(white)[...,-1]
        if name=='hann':
            difference=mix_white-white;min_eig=np.linalg.eigvalsh(difference)[...,0]
            if min_eig.min() < -1e-7*max(1.,float(arrays['mixture_ratio'].max())):raise ValueError('Jensen covariance identity failed')
            arrays['jensen_min_eigenvalue']=min_eig
            arrays['disagreement_energy_fraction']=(np.trace(difference,axis1=-2,axis2=-1)/np.trace(mix_white,axis1=-2,axis2=-1))
    # Spectra before/after synthesis use identical one-second analysis windows.
    stage_psd=[np.mean(local_psd,axis=0)]
    for signal in (x,hann.sum(1),x-hann.sum(1),box.sum(1)):
        padded=np.pad(signal,((rate//2,rate//2),(0,0)),mode='reflect')
        ff=np.lib.stride_tricks.sliding_window_view(padded,rate,axis=0)[::hop][interior]
        f,p=periodogram(ff,fs=rate,window='hann',detrend='constant',axis=-1,scaling='density');stage_psd.append(p.mean((0,1)))
    arrays.update(local_spectrum_frequency=f,local_stage_psd=np.stack(stage_psd,axis=1),
        weight_sum_min=np.array(wh[rate: -rate].min()),weight_sum_max=np.array(wh[rate:-rate].max()))
    return hann.sum(1),arrays

def overlap_job(payload):
    root,out,g=payload;out=Path(out);dest=out/'overlap'/g['key'];start=time.monotonic()
    if dest.with_suffix('.json').exists():return json.loads(dest.with_suffix('.json').read_text())
    result=dict(key=g['key'],dataset=g['dataset'],subject=g['subject'],status='failed')
    try:
        meta=json.loads((Path(root)/'windows'/g['key']).with_suffix('.json').read_text())
        stats,a,p,observed,neighbors,valid=load_sites(root,g,meta);selected=select_proxy_rows(stats,a,3);model=bind_reference(stats,a[selected],len(p))
        with np.load(meta['array_file']) as z:x=z['eeg_x'];expected=z['eeg_s']
        s,arrays=overlap_analysis(x,model,p[:,selected]);np.testing.assert_allclose(s,expected,rtol=1e-10,atol=1e-10)
        save_arrays(dest.with_suffix('.npz'),**arrays)
        result.update(status='completed',replay_max_abs=float(abs(s-expected).max()),
            weight_sum=[float(arrays['weight_sum_min']),float(arrays['weight_sum_max'])])
        for k in ('local_ratio','hann_weighted_local_ratio','mixture_ratio','hann_ratio','boxcar_ratio','adjacent_overlap_relative_rms','adjacent_overlap_cosine','disagreement_energy_fraction','jensen_min_eigenvalue'):
            result[k]=dict(quant(arrays[k]),by_carrier=[quant(arrays[k][:,b]) for b in range(6)])
            if k.endswith('ratio'):result[k]['exceedance']=float((arrays[k]>1+1e-8).mean())
        bp=band_integrals(arrays['local_spectrum_frequency'],arrays['local_stage_psd'])
        result['local_stage_band_retention']=(bp/bp[:,1,None]).tolist()
    except Exception:result['error']=traceback.format_exc()
    result.update(seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(dest.with_suffix('.json'),result);return result

def summarize_spectra(out):
    out=Path(out);plan=json.loads((out/'plan.json').read_text());by=defaultdict(list);fail=[];overlap=[]
    for g in plan['groups']:
        r=json.loads((out/'spectra'/g['key']).with_suffix('.json').read_text())
        if r['status']=='completed':by[g['dataset']].append(r)
        else:fail.append(r)
    result=dict(status='completed',spectral_planned=len(plan['groups']),failures=fail,datasets={},overlap=[])
    for ds,rr in by.items():
        shapes=np.array([r['shape'] for r in rr]);retention=np.array([r['retention'] for r in rr]);delta=np.array([r['S_i_vs_j_band_power_dB'] for r in rr])
        result['datasets'][ds]=dict(subjects=len(rr),channels=sum(len(r['channels']) for r in rr),
            shape_subject_mean=shapes.mean(0).tolist(),retention_subject_median=np.median(retention,axis=0).tolist(),
            retention_subject_p10=np.quantile(retention,.1,axis=0).tolist(),retention_subject_p90=np.quantile(retention,.9,axis=0).tolist(),
            absolute_S_i_j_dB=[quant(abs(delta[:,b])) for b in range(5)],
            joint_vs_mean_single_dB=[quant(10*np.log10(retention[:,b,3]/retention[:,b,1:3].mean(1))) for b in range(5)],
            cross_ratio_subject_median=np.median([r['cross_ratio'] for r in rr],axis=0).tolist(),
            spectral_closure_max=max(r['closure_error'] for r in rr))
    for g in plan['mechanism']:
        r=json.loads((out/'overlap'/g['key']).with_suffix('.json').read_text());overlap.append(r)
    result['overlap']=overlap;good=[r for r in overlap if r['status']=='completed']
    result['overlap_summary']={k:{stat:quant([r[k][stat] for r in good]) for stat in ('p50','p90','max')} for k in ('local_ratio','hann_weighted_local_ratio','mixture_ratio','hann_ratio','boxcar_ratio','adjacent_overlap_relative_rms','adjacent_overlap_cosine','disagreement_energy_fraction')}
    result['overlap_summary']['by_carrier_cosine_median']=[quant([r['adjacent_overlap_cosine']['by_carrier'][b]['p50'] for r in good]) for b in range(6)]
    result['overlap_summary']['by_carrier_disagreement_median']=[quant([r['adjacent_overlap_relative_rms']['by_carrier'][b]['p50'] for r in good]) for b in range(6)]
    result['overlap_summary']['local_stage_band_retention_median']=np.median([r['local_stage_band_retention'] for r in good],axis=0).tolist()
    result['overlap_summary']['local_stage_order']=['local_projected_carrier_sum','X','Hann_S','Hann_O','boxcar_S']
    result['status']='completed' if not fail and len(good)==len(overlap) else 'incomplete'
    write_json(out/'summary.json',result);return result

def render_spectra(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);summary=json.loads((out/'summary.json').read_text());plan=json.loads((out/'plan.json').read_text())
    datasets=list(summary['datasets']);names=['Single','REFED','Sim','Visual'];bands=['1-4','4-8','8-13','13-30','30-45']
    colors=['#777777','#287bb5','#e18527','#27874b','#845b9f','#b88b47','#c2393b'];styles=['-','--',':','-','--',':','-']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,ds,name in zip(axs.flat,datasets,names):
        v=summary['datasets'][ds];median=np.array(v['retention_subject_median']);lo=np.array(v['retention_subject_p10']);hi=np.array(v['retention_subject_p90'])
        for s in (1,2,3,6):
            ax.plot(range(5),100*median[:,s],styles[s],marker='o',color=colors[s],label=SPECTRAL_SIGNALS[s])
            if s in (3,6):ax.fill_between(range(5),100*lo[:,s],100*hi[:,s],color=colors[s],alpha=.1)
        ax.axhline(100,color='#777777',ls='--',label='X = 100%');ax.set(xticks=range(5),xticklabels=bands,xlabel='Frequency band (Hz)',ylabel='Power / X power in the same band (%)',title=f"{name}: {v['subjects']} subjects\nP50 lines, P10-P90 shading; selected shared electrodes",ylim=(0,110))
    axs[0,0].legend(fontsize=8,ncol=3);fig.savefig(out/'band_retention.png',dpi=220);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained');bandcolors=['#405e88','#4f9eb5','#72b79a','#e6ba5d','#c67a58']
    for ax,ds,name in zip(axs.flat,datasets,names):
        shape=np.array(summary['datasets'][ds]['shape_subject_mean'])*100;left=np.zeros(7)
        for b in range(5):ax.barh(range(7),shape[b],left=left,label=bands[b]+' Hz',color=bandcolors[b]);left+=shape[b]
        ax.set(yticks=range(7),yticklabels=SPECTRAL_SIGNALS,xlim=(0,100),xlabel='Fraction of each signal\'s own 1-45 Hz power (%)',title=name+' | equal-subject mean of normalized spectra');ax.invert_yaxis()
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5)
    fig.savefig(out/'band_shapes.png',dpi=220);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,ds,name in zip(axs.flat,datasets,names):
        psds=[]
        for g in plan['groups']:
            if g['dataset']!=ds:continue
            with np.load((out/'spectra'/g['key']).with_suffix('.npz')) as z:
                f=z['frequency'];psds.append(z['psd']/z['band_power'][:,0].sum())
        median=np.median(psds,axis=0);mask=(f>=1)&(f<45)
        for s in (0,1,2,3,6):ax.plot(f[mask],median[mask,s],styles[s],color=colors[s],label=SPECTRAL_SIGNALS[s],lw=1.4)
        ax.set(xscale='log',yscale='log',xlabel='Frequency (Hz)',ylabel='PSD / same-window X 1-45 Hz power (1/Hz)',title=name+' | median across subjects',xticks=[1,4,8,13,30,45],xticklabels=['1','4','8','13','30','45'])
    axs[0,0].legend(fontsize=8,ncol=3);fig.savefig(out/'spectra.png',dpi=220);plt.close(fig)
    good=[r for r in summary['overlap'] if r['status']=='completed'];fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for r in good:
        axs[0,0].plot(range(4),[r[k]['p50'] for k in ('local_ratio','mixture_ratio','hann_ratio','boxcar_ratio')],'-o',alpha=.55,lw=1)
    axs[0,0].axhline(1,ls='--',color='black');axs[0,0].set(xticks=range(4),xticklabels=['Local bound','Weighted mixture','Hann mean','Boxcar control'],ylabel='Median largest whitened eigenvalue',title='Same 12 windows; mixture includes disagreement energy')
    carriers=bands+['Complement']
    for ax,key,factor,title,ylabel in [(axs[0,1],'adjacent_overlap_relative_rms',100,'Adjacent blocks disagree on the same 0.9s overlap','Difference / pooled block RMS (%)'),(axs[1,0],'disagreement_energy_fraction',100,'Averaging removes energy from the weighted mixture','Whitened trace-energy reduction (%)')]:
        vv=np.array([[r[key]['by_carrier'][b]['p50'] for b in range(6)] for r in good])*factor
        mid=np.median(vv,axis=0);lo,hi=np.quantile(vv,[.1,.9],axis=0)
        ax.errorbar(range(6),mid,yerr=[mid-lo,hi-mid],fmt='o-',capsize=4);ax.set(xticks=range(6),xticklabels=carriers,ylabel=ylabel,title=title+'\nP50 and P10-P90 across 12 window medians')
    v=np.array(summary['overlap_summary']['local_stage_band_retention_median'])*100
    for j,label,color in [(0,'Local projected sum','#287bb5'),(2,'Hann S','#27874b'),(3,'Hann O','#c2393b'),(4,'Boxcar S','#845b9f')]:axs[1,1].plot(range(5),v[:,j],'-o',label=label,color=color)
    axs[1,1].set(xticks=range(5),xticklabels=bands,ylabel='Power / X power in same band (%)',xlabel='Frequency band (Hz)',title='Matched 1s Hann spectra, interior 15-75s\nMedian over 12 diagnostic windows');axs[1,1].legend(fontsize=8)
    fig.savefig(out/'overlap_mechanism.png',dpi=220);plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['plan','pilot','population','mechanism','summarize','position-plan','position','position-summary','temporal','temporal-summary','render','spectrum-plan','spectrum','overlap','overlap-pilot','spectrum-summary','spectrum-render'],required=True);parser.add_argument('--workers',type=int,default=16)
    args=parser.parse_args();cfg=yaml.safe_load(CONFIG.read_text());root=ROOT/cfg['source'];out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=True)
    if args.stage in ('spectrum-plan','spectrum','overlap','overlap-pilot','spectrum-summary','spectrum-render'):
        parent=out;out=parent/'spectral_overlap_v1';out.mkdir(exist_ok=True)
        if args.stage=='spectrum-plan':
            validate_source(root)
            write_json(out/'plan.json',dict(groups=json.loads((parent/'position_plan.json').read_text())['groups'],mechanism=json.loads((parent/'plan.json').read_text())['mechanism']))
            write_json(out/'resolved_config.json',yaml.safe_load((CONFIG.parent/'so_supervision_spectral_overlap_v1.yaml').read_text()));return
        if args.stage=='spectrum-summary':summarize_spectra(out);return
        if args.stage=='spectrum-render':render_spectra(out);return
    if args.stage=='render':render(out);return
    if args.stage=='temporal-summary':summarize_temporal(out);return
    if args.stage=='plan':
        make_plan(root,out);write_json(out/'resolved_config.json',cfg);return
    if args.stage=='position-plan':position_plan(root,out);return
    if args.stage=='position-summary':
        print(json.dumps(summarize_positions(out),ensure_ascii=False));return
    plan=json.loads((out/'plan.json').read_text());validate_source(root)
    if args.stage=='summarize':
        print(json.dumps({k:v for k,v in summarize(out,plan).items() if k not in ('datasets','mechanism')},ensure_ascii=False));return
    groups=plan['mechanism'] if args.stage in ('pilot','mechanism','temporal') else plan['groups']
    if args.stage=='position':groups=json.loads((out/'position_plan.json').read_text())['groups']
    if args.stage in ('overlap','overlap-pilot'):groups=plan['mechanism']
    if args.stage=='overlap-pilot':groups=[next(g for g in groups if g['dataset']==ds) for ds in sorted({g['dataset'] for g in groups})]
    if args.stage=='pilot':groups=[next(g for g in groups if g['dataset']==ds) for ds in sorted({g['dataset'] for g in groups})]
    fun=mechanism_job if args.stage=='mechanism' else population_job
    if args.stage=='position':fun=position_job
    if args.stage=='spectrum':fun=spectrum_job
    if args.stage in ('overlap','overlap-pilot'):fun=overlap_job
    if args.stage=='temporal':
        fun=temporal_job
        write_json(out/'temporal_resolved_config.json',yaml.safe_load((CONFIG.parent/'so_supervision_temporal_audit_v1.yaml').read_text()))
    start=time.monotonic();completed=0;failed=0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=iter(groups);pending=set()
        for _ in range(args.workers):
            g=next(jobs,None)
            if g is not None:pending.add(pool.submit(fun,(str(root),str(out),g)))
        while pending:
            done,pending=wait(pending,return_when=FIRST_COMPLETED)
            for f in done:
                r=f.result();completed+=1;failed+=r['status']!='completed'
                progress=dict(stage=args.stage,completed=completed,total=len(groups),failed=failed,seconds=time.monotonic()-start,workers=args.workers)
                write_json(out/(args.stage+'_progress.json'),progress)
                if completed%50==0 or completed==len(groups):print(json.dumps(progress),flush=True)
                g=next(jobs,None)
                if g is not None:pending.add(pool.submit(fun,(str(root),str(out),g)))
    write_json(out/(args.stage+'_execution.json'),dict(progress,status='completed' if failed==0 else 'completed_with_failures'))

if __name__=='__main__':main()
