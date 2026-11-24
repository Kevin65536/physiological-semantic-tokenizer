"""Train-only independent 13-bit replay and fixed-panel physical EEG review."""
from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.tokenizers.band_exchange_words import load_fold_books
from src.tokenizers.cross_branch_sampling import decode_words

STATE={}
ARMS=('staggered_joint_calibrated','aligned_joint_calibrated','aligned_separate_hard')
BANDS=((1,4),(4,8),(8,13),(13,30),(30,45))


def write(path,value):
    def scalar(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    path=Path(path);tmp=path.with_name(path.name+'.pending')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=scalar)+'\n');tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def pack_words(values):
    """Independent accumulator implementation, not NumPy packbits or producer code."""
    a=np.asarray(values)
    if a.dtype.kind not in 'iu' or np.any(a<0) or np.any(a>5115):raise ValueError('integer alphabet0..5115 required')
    result=bytearray();buffer=0;bits=0
    for item in a.reshape(-1):
        buffer |= int(item)<<bits;bits+=13
        while bits>=8:result.append(buffer&255);buffer>>=8;bits-=8
    if bits:result.append(buffer)
    return bytes(result)


def unpack_words(payload,shape):
    count=int(np.prod(shape));result=np.empty(count,np.uint16)
    if len(payload)!=(count*13+7)//8:raise ValueError('incorrect payload length')
    buffer=0;bits=0;cursor=0
    for i in range(count):
        while bits<13:buffer|=payload[cursor]<<bits;bits+=8;cursor+=1
        value=buffer&8191;buffer>>=13;bits-=13
        if value>5115:raise ValueError('unassigned 13-bit integer')
        result[i]=value
    if buffer:raise ValueError('nonzero padding')
    return result.reshape(shape)


def physical_metrics(prediction,batch,scales):
    """Recompute elementary metrics from raw masked samples and frozen denominators."""
    n=len(batch['x']);mask=np.asarray(batch['mask'],bool).reshape(n,6000)
    count=mask.sum(1).clip(1);factors=np.asarray(scales,float)
    def mean(a):return np.where(mask,a,0).sum(1)/count
    result=defaultdict(list)
    for output,target in [('sum','x'),('S','s'),('O','o')]:
        p=np.asarray(prediction[output],float).reshape(n,6000)
        y=np.asarray(batch[target],float).reshape(n,6000)
        if np.any(mask&(~np.isfinite(p)|~np.isfinite(y))):raise ValueError('nonfinite scored voltage')
        error=p-y;mse=mean(error*error);variance=np.asarray(batch['v'+target])[:,0,0]
        ym,pm=mean(y),mean(p);yc=y-ym[:,None];pc=p-pm[:,None]
        cov=mean(yc*pc);denominator=mean(yc*yc)*mean(pc*pc)
        corr=np.divide(cov,np.sqrt(denominator),out=np.full(n,np.nan),where=denominator>1e-20)
        energy=mean(y*y);gain=np.sqrt(np.divide(mean(p*p),energy,out=np.full(n,np.nan),where=energy>0))
        for name,value in dict(nmse=mse/variance,nrmse=np.sqrt(mse/variance),rmse_native=np.sqrt(mse)*factors,
                               bias_native=mean(error)*factors,correlation=corr,rms_gain=gain).items():result[name].append(value[:,None])
    return {k:np.stack(v,axis=1) for k,v in result.items()}


def budget():
    unit=os.environ.get('CROSS_REPLAY_UNIT')
    if unit:
        used=int(subprocess.check_output(['systemctl','--user','show',unit,'-p','CPUUsageNSec','--value'],text=True))/1e9
        used+=float(os.environ.get('CROSS_REPLAY_PRIOR_CPU','0'))
        if used>=880:raise RuntimeError('15CPU-minute review budget reached')


def books(fold):return {b:load_fold_books(STATE['books'],fold,b) for b in 'SO'}


def model(dataset,fold):
    path=STATE['parent']/dataset/f'fold_{fold}_model.npz';cases=STATE['cases'][dataset]
    with np.load(path,allow_pickle=False) as z:state={k:z[k].copy() for k in z}
    expected={c['subject'] for c in cases if c['fold']!=fold}
    if set(state['fit_people'])!=expected:raise ValueError('held person entered fitted state')
    n=len(STATE['population']['native_positions'][dataset]);a,j=state['independent'],state['joint']
    if a.shape!=j.shape or a.shape!=(5,2*n,2*n):raise ValueError('invalid paired covariance shape')
    np.testing.assert_array_equal(a[:,:n,n:],0);np.testing.assert_array_equal(a[:,n:,:n],0)
    np.testing.assert_array_equal(a[:,:n,:n],j[:,:n,:n]);np.testing.assert_array_equal(a[:,n:,n:],j[:,n:,n:])
    for v in (a,j):
        np.testing.assert_allclose(v,v.swapaxes(-2,-1),rtol=1e-12,atol=1e-12)
        if np.linalg.eigvalsh(v).min()<=0:raise ValueError('nonpositive frozen covariance')
    for layout in ('aligned','staggered'):
        gain,noise=state[layout+'_gain'],state[layout+'_noise']
        if gain.shape!=(5,2*n) or noise.shape!=gain.shape or not np.isfinite(gain).all() or not np.isfinite(noise).all() or (noise<0).any():
            raise ValueError('invalid frozen gain/noise')
    return state


def decode(sent,book,state,arm):
    layout,spatial,kind=arm.split('_');n=sent['S'].shape[0]
    cov=state['joint' if spatial=='joint' else 'independent']
    gain=state[layout+'_gain'] if kind=='calibrated' else np.ones((5,2*n))
    noise=state[layout+'_noise'] if kind=='calibrated' else np.zeros((5,2*n))
    return decode_words(sent,book,cov,gain,noise)


def stream_data(dataset,fold):
    path=STATE['parent']/dataset/f'fold_{fold}_ids.npz'
    with np.load(path,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['examples'],STATE['examples'][dataset])
        np.testing.assert_array_equal(z['fold'],STATE['folds'][dataset])
        fields={l+'_'+b:z[l+'_'+b].copy() for l in ('aligned','staggered') for b in 'SO'}
    return fields


def worker_probe(dataset):
    start=time.process_time();book=books(0);state=model(dataset,0);fields=stream_data(dataset,0)
    cases=STATE['cases'][dataset];offset=0;done=0;positions=0
    for c in cases:
        take=np.arange(offset,offset+c['count']);offset+=c['count']
        if c['fold']!=0:continue
        inv=np.argsort(c['native'])
        for arm in ARMS:
            layout=arm.split('_')[0]
            sent={b:fields[layout+'_'+b][take[inv]] for b in 'SO'}
            sent={b:unpack_words(pack_words(v),v.shape) for b,v in sent.items()}
            result=decode(sent,book,state,arm)
            if any(not np.isfinite(v).all() for v in result.values()):raise ValueError('nonfinite probe decode')
        positions+=c['count'];done+=1
        if done==2:break
    record=dict(dataset=dataset,CPU_seconds=time.process_time()-start,crops=done,positions=positions,
                all_three_paid_integer_decoders_finite=True,estimated_full_dataset_CPU_seconds=(time.process_time()-start)*len(cases)/max(done,1))
    write(STATE['out']/dataset/'probe.json',record);return record


def worker_decode(dataset):
    here=STATE['out']/dataset;cases=STATE['cases'][dataset];records=[]
    for fold in range(4):
        budget();book=books(fold);state=model(dataset,fold);fields=stream_data(dataset,fold)
        all_take=[];offset=0
        for c in cases:
            if c['fold']==fold:all_take.extend(range(offset,offset+c['count']))
            offset+=c['count']
        fields={k:v[all_take] for k,v in fields.items()};transport=[]
        for name,v in fields.items():
            path=here/f'fold_{fold}_{name}.13bit';path.write_bytes(pack_words(v))
            restored=unpack_words(path.read_bytes(),v.shape);np.testing.assert_array_equal(restored,v);fields[name]=restored
            transport.append(dict(file=path.name,shape=v.shape,IDs=v.size,paid_bits=v.size*13,bytes=path.stat().st_size,sha256=sha(path)))
        predictions={arm:{b:[] for b in ('S','O','sum')} for arm in ARMS};offset=0
        for c in cases:
            if c['fold']!=fold:continue
            budget();site=np.asarray(c['native']);np.testing.assert_array_equal(np.sort(site),np.arange(len(site)))
            indices=np.arange(offset,offset+c['count'])[site.argsort()];offset+=c['count']
            for arm in ARMS:
                layout=arm.split('_')[0];sent={b:fields[layout+'_'+b][indices] for b in 'SO'}
                output=decode(sent,book,state,arm)
                for b in ('S','O','sum'):predictions[arm][b].append(output[b][site])
        for arm in ARMS:np.savez(here/f'fold_{fold}_{arm}_prediction.npz',**{b:np.concatenate(v) for b,v in predictions[arm].items()})
        record=dict(fold=fold,positions=offset,payloads=transport,model_sha256=sha(STATE['parent']/dataset/f'fold_{fold}_model.npz'),
            original_IDs_sha256=sha(STATE['parent']/dataset/f'fold_{fold}_ids.npz'),
            raw_X_reads=0,target_reads=0,encoder_calls=0,decode_inputs='13bit restored S/O words, frozen books, joint/independent covariance, gain/noise, fixed native order')
        records.append(record);write(here/'decode_progress.json',record)
    write(here/'decode.json',records);return dataset


def training_data():
    from src.data.dual_component_tokenizer import base_tokenizer_training_data
    cfg=yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text())
    manifest,scales,_,_=base_tokenizer_training_data(cfg,ROOT)
    if (manifest['cfg_identity'],manifest['scales_identity'])!=(STATE['population']['cfg_identity'],STATE['population']['scales_identity']):raise ValueError('G0 identity changed')
    for cases in STATE['cases'].values():
        for c in cases:
            row=manifest['rows'][c['crop']['row']]
            if (row['split'],row['modality'],row['subject'])!=('train','eeg',c['subject']):raise ValueError('train boundary before signal read')
    return manifest['rows'],scales


def worker_score(dataset):
    from src.data.dual_component_tokenizer import base_tokenizer_batch
    rows,scales=training_data();here=STATE['out']/dataset;cache=OrderedDict();checks=[];all_ex=[];values=defaultdict(list)
    for fold in range(4):
        budget();predictions={}
        for arm in ARMS:
            with np.load(here/f'fold_{fold}_{arm}_prediction.npz',allow_pickle=False) as z:predictions[arm]={b:z[b] for b in ('S','O','sum')}
        with np.load(STATE['parent']/dataset/f'fold_{fold}_metrics.npz',allow_pickle=False) as source:
            offset=0;ex=[];parts=defaultdict(list)
            for c in STATE['cases'][dataset]:
                if c['fold']!=fold:continue
                budget();examples=np.array([(c['crop']['row'],c['crop']['start_s'],p) for p in c['crop']['positions']])
                batch=base_tokenizer_batch(rows,examples,scales,array_cache=cache,cache_limit=8)
                factors=np.array([scales['positions']['eeg'][dataset][rows[i]['positions'][p]]['scale'] for i,t,p in examples])
                for arm in ARMS:
                    predicted={b:v[offset:offset+c['count']] for b,v in predictions[arm].items()}
                    score=physical_metrics(predicted,batch,factors)
                    for key,v in score.items():parts[arm+'__'+key].append(v)
                ex.extend(examples);offset+=c['count']
            ex=np.array(ex);np.testing.assert_array_equal(source['examples'],ex)
            for key,part in parts.items():
                result=np.concatenate(part);np.testing.assert_array_equal(result,source[key],err_msg=dataset+'/'+str(fold)+'/'+key)
                values[key].append(result);checks.append(dict(fold=fold,metric=key,positions=len(ex),exact=True,max_error=0))
            all_ex.append(ex)
    np.savez_compressed(here/'independent_metrics.npz',examples=np.concatenate(all_ex),**{k:np.concatenate(v) for k,v in values.items()})
    write(here/'metric_comparison.json',checks);return dataset


def select():
    from src.analysis.dual_component_tokenizer import base_training_aggregate
    rows,scales=training_data();chosen={};parts=defaultdict(list);examples=[]
    def add(c,rule):
        key=c['row'],c['start_s'],c['position']
        if key not in chosen:chosen[key]=dict({k:v for k,v in c.items() if k!='nmse'},audit_index=len(chosen),audit_selection=[rule])
        else:chosen[key]['audit_selection'].append(rule)
    old=json.loads((STATE['run']/'component_source_transfer/integer_sum_source_v1/training_examples/waveform_examples.json').read_text())['examples']
    for c in old:add(c,'original_'+','.join(c['selection']))
    for dataset in STATE['cases']:
        with np.load(STATE['out']/dataset/'independent_metrics.npz') as z:
            ex=z['examples'];nmse=z[ARMS[0]+'__nmse'];examples.append(ex)
            for key in z:
                if key!='examples':parts[key].append(z[key])
        for label,rank in [('new_primary_joint_worst',nmse.mean((1,2))),('new_primary_S_worst',nmse[:,1,0]),('new_primary_O_worst',nmse[:,2,0])]:
            i,t,p=map(int,ex[np.argmax(rank)]);row=rows[i]
            c=dict(dataset=dataset,subject=row['subject'],record=row['record'],parent=row['parent'],row=i,start_s=t,position=p,
                position_name=row['positions'][p],scale=scales['positions']['eeg'][dataset][row['positions'][p]]['scale'],source_unit=row['source_unit'])
            add(c,label)
    summaries={}
    for arm in ARMS:
        summaries[arm]=base_training_aggregate({k.split('__',1)[1]:np.concatenate(v) for k,v in parts.items() if k.startswith(arm+'__')},np.concatenate(examples),rows)
        if (STATE['parent']/'summary.json').exists():
            original=json.loads((STATE['parent']/'summary.json').read_text())['arms'][arm]['summary']['metrics']['nmse']
            np.testing.assert_array_equal(summaries[arm]['metrics']['nmse']['macro'],original['macro'])
    write(STATE['out']/'selected_examples.json',dict(examples=list(chosen.values()),rule='all original17 fixed/random/K128-P90/oldhard, plus each dataset primary joint/S/O argmax, deduplicate only physical triples'))
    write(STATE['out']/'summary.json',dict(status='numerical_replay_passed_awaiting_actual_images',arms=summaries,positions=17040,crops=400,people=60,
        selected_examples=list(chosen.values()),independent13bit_decoder=True,maximum_pointwise_NMSE_error=0,validation_reads=0,confirmation_reads=0,test_reads=0))


def worker_render(dataset):
    from src.data.dual_component_tokenizer import base_tokenizer_batch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows,scales=training_data();out=STATE['out'];folder=out/STATE.get('figure_directory','figures');folder.mkdir(exist_ok=True)
    selected=[c for c in json.loads((out/'selected_examples.json').read_text())['examples'] if c['dataset']==dataset]
    if STATE.get('original_hard_only',False):selected=[c for c in selected if any('candidate_max' in x for x in c['audit_selection'])]
    cases=STATE['cases'][dataset];lookup={(c['crop']['row'],c['crop']['start_s']):c for c in cases};records=[];cache={}
    palette={'target':'#222222','aligned_separate_hard':'#c88832','aligned_joint_calibrated':'#547fb6','staggered_joint_calibrated':'#178067'}
    styles={'target':'-','aligned_separate_hard':'--','aligned_joint_calibrated':'-.','staggered_joint_calibrated':':'}
    labels={'target':'target','aligned_separate_hard':'aligned separate hard','aligned_joint_calibrated':'aligned joint calibrated','staggered_joint_calibrated':'staggered joint calibrated'}
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
    for c in selected:
        budget();physical=lookup[c['row'],c['start_s']];fold=physical['fold'];key=fold
        if key not in cache:
            cache.clear();cache[key]={}
            for arm in ARMS:
                with np.load(out/dataset/f'fold_{fold}_{arm}_prediction.npz') as z:cache[key][arm]={b:z[b] for b in ('S','O','sum')}
        offset=0
        for p in cases:
            if p is physical:break
            if p['fold']==fold:offset+=p['count']
        position=physical['crop']['positions'].index(c['position']);j=offset+position
        ex=np.array([[c['row'],c['start_s'],c['position']]]);batch=base_tokenizer_batch(rows,ex,scales)
        scale=float(c['scale']);mask=batch['mask'][0].reshape(6000).astype(bool);t=np.arange(6000)/200
        waves={'target':{b:batch[k][0].reshape(6000).astype(float)*scale for b,k in [('X','x'),('S','s'),('O','o')]}}
        for arm in ARMS:waves[arm]={b:cache[key][arm][k][j].reshape(6000)*scale for b,k in [('X','sum'),('S','S'),('O','O')]}
        event_component=STATE.get('event_component','S')
        point=int(np.argmax(np.where(mask,abs(waves['target'][event_component]),-np.inf)));center=np.clip(point/200,1,29);boundary=np.clip(2*round(center/2),2,28)
        unit='REFED export units (physical scale unknown)' if dataset=='refed' else str(c['source_unit'])
        axis_unit='REFED export unit' if dataset=='refed' else unit
        unit_caption='; REFED physical scale unknown' if dataset=='refed' else ''
        identity=f"{dataset} | {c['subject']} | {c['record']} | {c['position_name']} | +{c['start_s']}s | fold{fold}"
        fig,axes=plt.subplots(3,4,figsize=(17.5,8.8),layout='constrained')
        windows=[(0,30),(14,16),(center-1,center+1),(boundary-.5,boundary+.5)]
        for row,b in enumerate(('X','S','O')):
            extrema=np.array([v[b] for v in waves.values()]);low,high=float(np.nanmin(extrema)),float(np.nanmax(extrema));margin=.06*(high-low or 1)
            for col,(left,right) in enumerate(windows):
                ax=axes[row,col]
                for name,v in waves.items():ax.plot(t,v[b],color=palette[name],ls=styles[name],lw=.85,label=labels[name])
                for edge in range(0,31,2):ax.axvline(edge,color='#b7bec3',lw=.45,alpha=.6)
                ax.fill_between(t,low-margin,high+margin,where=~mask,color='#bfc3c7',alpha=.45)
                ax.set(xlim=(left,right),ylim=(low-margin,high+margin),xlabel='Time within30s (s)')
                if col==0:ax.set_ylabel(b+' / '+axis_unit)
                if row==0:ax.set_title(['Complete30s','Fixed14–16s','2s at true '+event_component+' peak','Nearest2s boundary'][col])
        axes[0,0].legend(fontsize=6.5);fig.suptitle(identity+'\nReal paid S/O IDs; shared axes; gray=original mask; vertical lines=2s boundaries'+unit_caption,fontsize=10)
        stem=f"{c['audit_index']:03d}_{dataset}_{c['subject']}_{c['position_name']}";voltage=folder/(stem+'_voltage.png');fig.savefig(voltage,dpi=220);plt.close(fig)
        if event_component!='S':
            records.append(dict(c,voltage=str(voltage.relative_to(out)),fold=fold,source_unit_display=unit,
                supplemental_component=event_component,target_peak_s=point/200,local_window=[float(center-1),float(center+1)],
                boundary_s=float(boundary),selection_preserved='original preselected candidate_max only'))
            continue
        full=mask.reshape(15,400).all(1);valid_block=mask.reshape(120,50).all(1);hz=np.fft.rfftfreq(400,1/200);win=np.hanning(400)
        fig,axes=plt.subplots(3,4,figsize=(17.5,8.8),layout='constrained');dynamics={};peaks={}
        for row,b in enumerate(('X','S','O')):
            dynamics[b]={};at=int(np.argmax(np.where(mask,abs(waves['target'][b]),-np.inf)))
            peaks[b]=dict(time_s=at/200,target=float(waves['target'][b][at]))
            for mi,(name,v) in enumerate(waves.items()):
                y=v[b];env=np.sqrt((y.reshape(120,50)**2).mean(1));env[~valid_block]=np.nan
                patch=y.reshape(15,400);power=abs(np.fft.rfft((patch-patch.mean(1,keepdims=True))*win,axis=-1))**2*2/(400*(win*win).sum())
                energy=np.stack([power[:,(hz>=lo)&((hz<=hi) if hi==45 else (hz<hi))].sum(1) for lo,hi in BANDS],axis=1);energy[~full]=np.nan
                axes[row,0].plot(np.arange(120)*.25+.125,env,color=palette[name],ls=styles[name],label=labels[name])
                for col,band in [(1,1),(2,2)]:axes[row,col].plot(np.arange(15)*2+1,energy[:,band],color=palette[name],ls=styles[name],marker='.',ms=3,label=labels[name])
                avg=np.nanmean(energy,axis=0);axes[row,3].bar(np.arange(5)+(mi-1.5)*.19,np.maximum(avg,1e-25),.18,color=palette[name],label=labels[name])
                dynamics[b][name]=dict(mean_band_energy=avg,complete_patches=int(full.sum()));peaks[b][name]=dict(value_at_target_peak=float(y[at]),RMSE=float(np.sqrt(np.mean((y[mask]-waves['target'][b][mask])**2))))
            for col in range(3):axes[row,col].set(xlim=(0,30),xlabel='Time within30s (s)',ylabel=b+(' / '+axis_unit if col==0 else ' / source unit²'))
            axes[row,3].set(xticks=range(5),xticklabels=['1–4','4–8','8–13','13–30','30–45'],yscale='log',xlabel='Frequency (Hz)',ylabel=b+' / source unit²')
            if row==0:
                for col,title in enumerate(('0.25s local RMS','2s theta4–8Hz power','2s alpha8–13Hz power','Mean complete-patch energy')):axes[row,col].set_title(title)
        axes[0,0].legend(fontsize=6.5);fig.suptitle(identity+'\nOriginal complete-block mask; no curve smoothing; all bands within1–45Hz'+unit_caption,fontsize=10)
        dynamic=folder/(stem+'_dynamics.png');fig.savefig(dynamic,dpi=220);plt.close(fig)
        record=dict(c,voltage=str(voltage.relative_to(out)),dynamics=str(dynamic.relative_to(out)),fold=fold,source_unit_display=unit,
            true_S_peak_s=point/200,local_window=[float(center-1),float(center+1)],boundary_s=float(boundary),peak_and_RMSE=peaks,band_energies=dynamics)
        records.append(record);write(out/dataset/'render_progress.json',dict(completed=len(records),planned=len(selected)))
    write(out/dataset/'figures.json',records);return records


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('probe','decode','score','select','render'),required=True);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--dataset',help='one dataset, for a local versioned figure re-export only')
    parser.add_argument('--figure-directory',default='figures')
    parser.add_argument('--event-component',choices=('S','O'),default='S')
    parser.add_argument('--original-hard-only',action='store_true')
    args=parser.parse_args();run=args.run.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_cross_branch_sampling_v1.yaml').read_text())
    if cfg['data']['development']!='forbidden' or cfg['data']['confirmation']!='forbidden' or cfg['data']['test']!='forbidden':raise ValueError('train-only review contract required')
    pop=json.loads((run/cfg['population']).read_text());cases={d:[c for c in pop['cases'] if c['dataset']==d] for d in sorted(pop['native_positions'])}
    if len(pop['cases'])!=400 or sum(c['count'] for c in pop['cases'])!=17040:raise ValueError('original400/17040 population required')
    for d in cases:(out/d).mkdir(exist_ok=True)
    examples={d:np.array([(c['crop']['row'],c['crop']['start_s'],p) for c in cc for p in c['crop']['positions']]) for d,cc in cases.items()}
    if args.dataset and (args.stage!='render' or args.dataset not in cases):raise ValueError('explicit render-only dataset required')
    if Path(args.figure_directory).name!=args.figure_directory:raise ValueError('figure directory must be a local artifact name')
    STATE.update(run=run,out=out,parent=out.parent,books=run/cfg['dictionaries']/'books',population=pop,cases=cases,examples=examples,figure_directory=args.figure_directory,
        event_component=args.event_component,original_hard_only=args.original_hard_only,
        folds={d:np.array([c['fold'] for c in cc for _ in c['native']]) for d,cc in cases.items()})
    if args.stage in ('score','select','render'):training_data()
    begin=time.monotonic()
    if args.stage=='select':select()
    else:
        worker={'probe':worker_probe,'decode':worker_decode,'score':worker_score,'render':worker_render}[args.stage]
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('fork')) as pool:results=list(pool.map(worker,[args.dataset] if args.dataset else cases))
        if args.stage=='probe':write(out/'probe.json',results)
        if args.stage=='render':
            records=sorted([v for group in results for v in group],key=lambda c:c['audit_index'])
            name='figure_manifest.json' if args.figure_directory=='figures' else 'figure_manifest_'+args.figure_directory+'.json'
            write(out/name,dict(status='rendered_pending_actual_image_inspection',examples=records,figure_count=sum(('voltage' in x)+('dynamics' in x) for x in records)))
    stage_name=args.stage if args.stage!='render' or args.figure_directory=='figures' else 'render_'+args.figure_directory
    write(out/(stage_name+'_complete.json'),dict(stage=args.stage,wall_seconds=time.monotonic()-begin))


if __name__=='__main__':main()
