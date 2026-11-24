"""Actual frequency-tagged EEG word members on the original training-person folds."""
from __future__ import annotations
import argparse
import base64
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import fcntl
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
from numpy.lib.format import open_memmap
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.data.dual_component_tokenizer import base_tokenizer_training_data,base_tokenizer_batch
from src.tokenizers.band_exchange_words import ATOMS

STATE={}
VOCAB=5116


def write(path,value):
    def default(a):
        if isinstance(a,np.ndarray):return a.tolist()
        if isinstance(a,np.generic):return a.item()
        raise TypeError(type(a).__name__)
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=default)+'\n')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def budget():
    unit=os.environ.get('BAND_WORD_ATLAS_UNIT');prior=float(os.environ.get('BAND_WORD_ATLAS_PRIOR_CPU','0'))
    used=time.process_time() if not unit else int(subprocess.check_output(['systemctl','--user','show',unit,'-p','CPUUsageNSec','--value'],text=True))/1e9
    if prior+used>=STATE['cfg']['resources']['CPU_seconds']-15:raise RuntimeError('bounded atlas CPU budget reached; completed chunks retained')
    return used+prior


def occupancy(codes,valid,folds,people,datasets,fold):
    if codes.shape!=valid.shape or codes.dtype.kind not in 'iu' or codes.min()<0 or codes.max()>=VOCAB:raise ValueError('actual K5116 paid integer support required')
    result={}
    for name,selected in [('train',folds!=fold),('held',folds==fold)]:
        support=valid&selected[:,None];ids=codes[support]
        count=np.bincount(ids,minlength=VOCAB)
        unique_people=np.unique(np.broadcast_to(np.asarray(people,np.int64)[:,None],codes.shape)[support]*VOCAB+ids)
        unique_sets=np.unique(np.broadcast_to(datasets[:,None],codes.shape)[support]*VOCAB+ids)
        result[name]=count
        result[name+'_subjects']=np.bincount(unique_people%VOCAB,minlength=VOCAB)
        result[name+'_datasets']=np.bincount(unique_sets%VOCAB,minlength=VOCAB)
    order=np.argsort(-result['train'],kind='stable');rank=np.empty(VOCAB,int);rank[order]=np.arange(1,VOCAB+1)
    mass=result['train']/max(result['train'].sum(),1);cumulative=np.cumsum(mass[order]);cum=np.empty(VOCAB);cum[order]=cumulative
    before=np.r_[0,cumulative[:-1]];major=np.zeros(VOCAB,bool);major[order]=before<.9
    major[0]=False
    result.update(rank=rank,mass=mass,cumulative=cum,major=major,low_support=(result['train']<8)|(result['train_subjects']<2))
    return result


def representative_indices(indices,people,seed,code,maximum=3):
    ix=np.asarray(indices,np.int64)
    if not len(ix):return ix
    x=ix.astype(np.uint64)+np.uint64(seed+code*104729)
    with np.errstate(over='ignore'):
        x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9);x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb);x=x^(x>>np.uint64(31))
    order=ix[np.argsort(x,kind='stable')];out=[];seen=set()
    for n in order:
        if int(people[n]) not in seen:out.append(int(n));seen.add(int(people[n]))
        if len(out)==maximum:return np.asarray(out)
    for n in order:
        if int(n) not in out:out.append(int(n))
        if len(out)==maximum:break
    return np.asarray(out)


def encoded(a):return base64.b64encode(np.ascontiguousarray(a,dtype='<f4').tobytes()).decode('ascii')


def waveform_record(values,indices,chosen):
    if not len(indices):return None
    v=np.asarray(values[indices],np.float32)
    take={int(x):j for j,x in enumerate(indices)}
    rows=np.vstack((v.mean(0,dtype=np.float64),np.quantile(v,[.1,.9],axis=0),np.asarray([v[take[int(i)]] for i in chosen])))
    return dict(length=v.shape[1],rows=encoded(rows))


def initialize(cfg):
    global VOCAB
    VOCAB=cfg['vocabulary']
    if VOCAB not in (5116,15351):raise ValueError('unsupported paid alphabet')
    run=ROOT/cfg['run'];model=run/cfg['model_run'];out=model/cfg['output'];out.mkdir(parents=True,exist_ok=True)

    if not (model/'fit_complete.json').is_file() or not (model/'encode_complete.json').is_file():raise ValueError('completed immutable books and paid IDs required')
    if cfg['primary']=='innovation':
        owner=json.loads((model/'resolved_config.json').read_text())
        pop_path=run/owner['config']['source']['population']
    else:
        owner=json.loads((model/'word_contract.json').read_text());pop_path=run/owner['config']['source']['same_source_bank']/'population.json'
    pop=json.loads(pop_path.read_text());info=pop_path.parent
    with np.load(info/'training_attributes.npz',allow_pickle=False) as z:bank={k:z[k].copy() for k in z.files}
    manifest,scales,_,_=base_tokenizer_training_data(yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text()),ROOT)
    if (cfg['allowed_splits']!=['train'] or len(pop['cases'])!=400 or len(pop['subjects'])!=60 or len(bank['examples'])!=17040
            or owner['input_identity']!=manifest['cfg_identity'] or owner['scales_identity']!=manifest['scales_identity'] or owner['population_sha256']!=sha(pop_path)):
        raise ValueError('original train population and G0 identities required before arrays')
    for c in pop['cases']:
        r=manifest['rows'][c['crop']['row']]
        if (r['split'],r['modality'],r['dataset'],r['subject'])!=('train','eeg',c['dataset'],c['subject']):raise ValueError('original train boundary before arrays')
    datasets=sorted(pop['native_positions']);di={d:j for j,d in enumerate(datasets)}
    STATE.update(cfg=cfg,run=run,model=model,out=out,owner=owner,pop=pop,bank=bank,rows=manifest['rows'],scales=scales,datasets=datasets,
        dataset_index=np.asarray([di[manifest['rows'][i]['dataset']] for i,_,p in bank['examples']]))


def paid_ids(fold):
    parts=defaultdict(list);identity=[]
    for d in STATE['datasets']:
        if STATE['cfg']['primary']=='innovation':
            path=STATE['model']/d/f'fold_{fold}_ids.npz'
            subset=STATE['dataset_index']==STATE['datasets'].index(d)
            with np.load(path) as z:
                np.testing.assert_array_equal(z['examples'],STATE['bank']['examples'][subset])
                np.testing.assert_array_equal(z['fold'],STATE['bank']['fold'][subset])
                with np.load(STATE['model']/d/f'fold_{fold}_state.npz') as state:native=state['native']
                for b in 'SO':
                    expected=np.load(STATE['model']/d/f'fold_{fold}_prepared'/(b+'_role.npy'),mmap_mode='r')
                    for arm in (STATE['cfg']['primary'],STATE['cfg']['reference']):
                        ids=z[arm+'_'+b];stage=ids>=5116
                        roles=np.where(ids==0,-1,(ids.astype(np.int32)-1)//1023)
                        roles[stage]=(ids[stage].astype(np.int32)-5116)//2047
                        np.testing.assert_array_equal(roles,np.broadcast_to(expected[:,None],ids.shape))
                        second=((native%2)==(1 if b=='S' else 0))
                        np.testing.assert_array_equal(stage,(ids>0)&second[:,None])
                        parts[arm,b].append(ids)
        else:
            path=STATE['model']/d/f'fold_{fold}_ids.npz';native=np.load(STATE['model']/'prepared'/d/f'fold_{fold}'/'native.npy',mmap_mode='r')
            seen=np.load(STATE['model']/'prepared'/d/f'fold_{fold}'/'seen.npy',mmap_mode='r')
            expected=STATE['bank']['examples'][STATE['dataset_index']==STATE['datasets'].index(d)]
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['examples'],expected);np.testing.assert_array_equal(z['fold'],STATE['bank']['fold'][STATE['dataset_index']==STATE['datasets'].index(d)])
                for b in 'SO':
                    roles=np.load(STATE['model']/'prepared'/d/f'fold_{fold}'/(b+'_role.npy'),mmap_mode='r')
                    for arm in (STATE['cfg']['primary'],STATE['cfg']['reference']):
                        ids=z[arm+'_'+b].copy();tags=np.where(ids>0,(ids.astype(np.int32)-1)//ATOMS,-1)
                        expect=roles if arm.startswith('exchange') else np.broadcast_to(np.where(seen,native%5,-1)[:,None],ids.shape)
                        np.testing.assert_array_equal(tags,expect)
                        parts[arm,b].append(ids)
        identity.append(dict(dataset=d,fold=fold,path=str(path.relative_to(ROOT)),sha256=sha(path),actual_paid_role_verified=True))
    return {k:np.concatenate(v) for k,v in parts.items()},identity


def usage():
    out=STATE['out'];bank=STATE['bank'];summaries={};catalog={};all_rows=[];identities=[]
    for fold in range(4):
        budget();codes,identity=paid_ids(fold);identities+=identity;summaries[fold]={}
        for (arm,b),ids in codes.items():
            stats=occupancy(ids,bank['valid'],bank['fold'],bank['subject'],STATE['dataset_index'],fold)
            rows=[]
            for code in range(VOCAB):
                item={k:(int(v[code]) if np.asarray(v).dtype.kind in 'ibu' else float(v[code])) for k,v in stats.items()}
                item.update(fold=fold,arm=arm,branch=b,ID=code,band=-1 if code==0 else (code-1)//ATOMS,atom=-1 if code==0 else (code-1)%ATOMS,
                    reserved_missing=code==0,dead=int(stats['train'][code])==0,held_only=int(stats['train'][code])==0 and int(stats['held'][code])>0)
                if STATE['cfg']['primary']=='innovation' and code>=5116:
                    item['band'],item['atom']=divmod(code-5116,2047)
                rows.append(item)
            all_rows+=rows
            if fold==0:catalog[arm,b]=sorted(rows,key=lambda x:x['rank'])
            supported=(stats['train']>=8)&(stats['train_subjects']>=2);validheld=int(stats['held'].sum())
            summaries[fold][arm+'_'+b]=dict(fit_patches=int(stats['train'].sum()),held_patches=validheld,fit_used_positive=int((stats['train'][1:]>0).sum()),dead_positive=int((stats['train'][1:]==0).sum()),held_only_positive=int(((stats['train'][1:]==0)&(stats['held'][1:]>0)).sum()),
                major90_codes=int(stats['major'].sum()),major90_fit_mass=float(stats['mass'][stats['major']].sum()),held_supported_mass=float(stats['held'][supported].sum()/max(validheld,1)),fit_people=int(len(np.unique(bank['subject'][bank['fold']!=fold]))),held_people=int(len(np.unique(bank['subject'][bank['fold']==fold]))))
    with (out/'all_fold_code_usage.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    with (out/'rare_dead_heldonly_codes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(x for x in all_rows if x['low_support'] or x['dead'] or x['held_only'])
    write(out/'usage_summary.json',dict(folds=summaries,word_identity='each_fold_separate_no_numeric_ID_comparison_across_folds',actual_ID_sources=identities))
    chosen={}
    for b in 'SO':
        entries=catalog[STATE['cfg']['primary'],b];chosen[b]=[]
        for band in range(5):
            available=[x for x in entries if x['band']==band];major=max(available,key=lambda x:(x['train'],-x['ID']))
            weak=[x for x in available if x['held']>0 and x['low_support']]
            if not weak:weak=[x for x in available if x['held']>0 and x['ID']!=major['ID']]
            rare=max(weak,key=lambda x:(x['held'],-x['ID']))
            chosen[b]+=[dict(ID=major['ID'],band=band,rule='highest_actual_fit_occupancy'),dict(ID=rare['ID'],band=band,rule='highest_actual_held_low_support_or_next')]
    write(out/'selection.json',dict(config=STATE['cfg'],detailed=chosen,major90={b:[r['ID'] for r in catalog[STATE['cfg']['primary'],b] if r['major']] for b in 'SO'},selection_before_waveform_inspection=True))
    return catalog,chosen


def prepare_cache():
    out=STATE['out'];directory=out/'cache';directory.mkdir(exist_ok=True);bank=STATE['bank'];n=int(bank['valid'].sum())
    if (directory/'ready.json').exists():
        old=json.loads((directory/'ready.json').read_text())
        if old['complete_members']!=n or old['G0_input_identity']!=STATE['owner']['input_identity'] or old['valid_mask_sha256']!=hashlib.sha256(bank['valid'].tobytes()).hexdigest():raise ValueError('cached member contract changed')
        return
    shape=(n,3,400);waves=open_memmap(directory/'waves.pending.npy',mode='w+',dtype='f4',shape=shape)
    psd={b:open_memmap(directory/f'{b}_PSD.pending.npy',mode='w+',dtype='f4',shape=(n,89)) for b in 'SO'}
    rms={b:open_memmap(directory/f'{b}_RMS.pending.npy',mode='w+',dtype='f4',shape=(n,8)) for b in 'SO'}
    keys=np.empty((n,2),np.int32);cache=OrderedDict();offset=0;hashes={b:hashlib.sha256() for b in ['S','O','X']};window=np.hanning(400)
    for number,c in enumerate(STATE['pop']['cases']):
        budget();sl=slice(c['first'],c['first']+c['count']);ex=bank['examples'][sl]
        batch=base_tokenizer_batch(STATE['rows'],ex,STATE['scales'],array_cache=cache,cache_limit=16);valid=batch['mask'].all(-1)
        np.testing.assert_array_equal(valid,bank['valid'][sl]);v=np.stack([batch[k][valid] for k in ('s','o','x')],axis=1).astype('f4')
        stop=offset+len(v);waves[offset:stop]=v
        p,t=np.nonzero(valid);keys[offset:stop]=np.column_stack((p+c['first'],t))
        for j,b in enumerate('SO'):
            value=v[:,j].astype(float);spectrum=2*abs(np.fft.rfft((value-value.mean(1,keepdims=True))*window,axis=1))**2/(200*np.square(window).sum());psd[b][offset:stop]=spectrum[:,2:91]
            rms[b][offset:stop]=np.sqrt(np.mean(value.reshape(-1,8,50)**2,axis=2));hashes[b].update(v[:,j].copy().tobytes())
        hashes['X'].update(v[:,2].copy().tobytes());offset=stop
        if number%25==0:write(out/'progress.json',dict(stage='original_true_member_cache',crops=number+1,planned=400,CPU_seconds=budget()))
    if offset!=n:raise ValueError('complete scoring denominator changed')
    del waves;psd.clear();rms.clear()
    np.save(directory/'keys.npy',keys)
    for p in directory.glob('*.pending.npy'):p.replace(p.with_name(p.name.replace('.pending','')))
    write(directory/'ready.json',dict(schema='band_exchange_true_member_cache_v1',valid_mask_sha256=hashlib.sha256(bank['valid'].tobytes()).hexdigest(),examples_sha256=hashlib.sha256(bank['examples'].tobytes()).hexdigest(),complete_members=n,shape=shape,G0_input_identity=STATE['owner']['input_identity'],G0_scales_identity=STATE['owner']['scales_identity'],content_sha256={b:h.hexdigest() for b,h in hashes.items()},dtype='float32_canonical_original_scoring_coordinate',validation_reads=0,test_reads=0))


def identity(n,keys):
    window,patch=map(int,keys[n]);row,start,p=map(int,STATE['bank']['examples'][window]);r=STATE['rows'][row];site=r['positions'][p]
    return dict(dataset=r['dataset'],subject=r['subject'],record=r['record'],position=site,parent=r['parent'],crop_s=start,patch=patch,g0_row=row,
        input_scale=STATE['scales']['positions']['eeg'][r['dataset']][site]['scale'],source_unit=r.get('source_unit','unknown'),coordinate='voltage / this fixed training input SD')


def export_branch(branch):
    budget();out=STATE['out'];cache=out/'cache';keys=np.load(cache/'keys.npy',mmap_mode='r');waves=np.load(cache/'waves.npy',mmap_mode='r')
    psd=np.load(cache/f'{branch}_PSD.npy',mmap_mode='r');rms=np.load(cache/f'{branch}_RMS.npy',mmap_mode='r')
    codes,_=paid_ids(0);ids=codes[STATE['cfg']['primary'],branch][STATE['bank']['valid']];people=STATE['bank']['subject'][keys[:,0]];fit=STATE['bank']['fold'][keys[:,0]]!=0
    order=np.argsort(ids,kind='stable');ordered=ids[order];unique,starts=np.unique(ordered,return_index=True);stops=np.r_[starts[1:],len(order)];groups={int(k):(int(a),int(b)) for k,a,b in zip(unique,starts,stops)}
    entries=STATE['catalog'][STATE['cfg']['primary'],branch];folder=out/('EEG-'+branch);folder.mkdir(exist_ok=True);chunk=[];profiles=[];member_count=0
    books=[]
    if STATE['cfg']['primary']!='innovation':
        for band in range(5):
            with np.load(STATE['model']/'books'/f'fold_0_{branch}_{band}.npz',allow_pickle=False) as z:
                meta=json.loads(str(z['metadata']));assert(meta['fold'],meta['branch'],meta['role'])==(0,branch,band)
                books.append(dict(atoms=z['balanced_atoms'].copy(),gain=z['balanced_gain'].copy(),fit_sample_count=z['balanced_count'].copy()))
    for rank,entry in enumerate(entries):
        if rank%128==0:budget()
        code=entry['ID'];left,right=groups.get(code,(0,0));allix=order[left:right];datum={};profile={}
        for split,ix in [('train',allix[fit[allix]]),('held',allix[~fit[allix]])]:
            chosen=representative_indices(ix,people,STATE['cfg']['seed'],code)
            datum[split]=dict(count=len(ix),subjects=int(len(np.unique(people[ix]))),members=[identity(int(n),keys) for n in chosen],panels=[])
            for signal in (waves[:,0],waves[:,1],waves[:,2],psd,rms):datum[split]['panels'].append(waveform_record(signal,ix,chosen))
            if len(ix):
                values=STATE['bank'][branch+'_attributes'][keys[ix,0],keys[ix,1]]
                profile[split]=dict(mean=values.mean(0),P10=np.quantile(values,.1,axis=0),P90=np.quantile(values,.9,axis=0))
            else:profile[split]=dict(mean=np.full(10,np.nan),P10=np.full(10,np.nan),P90=np.full(10,np.nan))
        if code and books:
            book=books[(code-1)//ATOMS];atom=(code-1)%ATOMS;datum['atom']=dict(length=400,rows=encoded(np.stack((book['atoms'][atom],book['atoms'][atom]*book['gain'][atom]))),fit_kmeans_sample_count=int(book['fit_sample_count'][atom]),energy_gain=float(book['gain'][atom]))
        else:datum['atom']=None
        profiles.append(dict(ID=code,**profile));chunk.append(datum);member_count+=datum['train']['count']+datum['held']['count']
        if len(chunk)==32 or rank+1==len(entries):
            (folder/f'{rank//32}.js').write_text('window.atlasData='+json.dumps(chunk,separators=(',',':'),ensure_ascii=False)+';');chunk=[]
        if rank%256==0:write(out/f'progress_{branch}.json',dict(stage='all_word_true_member_statistics',codes=rank+1,planned=VOCAB,CPU_seconds=budget()))
    if member_count!=len(ids):raise ValueError('word partition coverage changed')
    write(out/f'{branch}_true_properties.json',profiles)
    return dict(branch=branch,codes=VOCAB,all_complete_members=member_count,major_codes=sum(x['major'] for x in entries),all_code_JS_chunks=(VOCAB+31)//32,CPU_seconds=budget())


def html_index():
    out=STATE['out'];primary=STATE['cfg']['primary'];catalog={}
    for b in 'SO':
        catalog['EEG-'+b]=[dict(r,chunk=(r['rank']-1)//32) for r in STATE['catalog'][primary,b]]
    (out/'atlas_index.js').write_text('window.atlasIndex='+json.dumps(catalog,separators=(',',':'))+';')
    with (out/'all_codes.csv').open('w') as f:
        data=[dict(r,branch=b) for b in 'SO' for r in catalog['EEG-'+b]];w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    template=(ROOT/STATE['cfg']['template']).read_text()
    template=template.replace('四分支单码图册','EEG 实发单码图册 · fold 0').replace('validation','held').replace('验证集','折外训练人群').replace('验证','折外训练').replace('开发观察','折外训练观察')
    template=template.replace('EEG 码义按分支、角色频带与 ID 解释。','仅解释同一 fold 0 字典内的实发全 ID；其他折的数字 ID 不作词义对齐。')
    template=template.replace('显示的是给定单码的目标分布；','全部真实S/O/X与分位按冻结位置训练X标准差表示；具体原生尺度与单位在成员身份中。黑色原子仅是模型内部局部原子。显示的是给定单码的目标分布；')
    template=template.replace('{{profile_sources}}','仅原400裁片/60训练人，fold 0拟合模板 vs 同折外训练人；不读取validation、confirmation或test。<a href="all_fold_code_usage.csv">全部四折两布局完整码使用</a> · <a href="rare_dead_heldonly_codes.csv">稀有/死码/折外专有码</a>')
    start=template.index('        let panels = eeg ? [');stop=template.index('        panels.forEach',start)
    template=template[:start]+'''        let panels = [
            ['真实 S 电压 / 固定训练 X SD','秒',[0,1.995]],
            ['真实 O 电压 / 固定训练 X SD','秒',[0,1.995]],
            ['观测 X 电压 / 固定训练 X SD','秒',[0,1.995]],
            ['本支真实成员 Hann PSD (G0²/Hz)','Hz',[1,45]],
            ['本支真实成员 0.25秒 RMS (G0)','块中心/秒',[.125,1.875]]
        ];
''' +template[stop:]
    marker='        for (let sp of [\'train\', \'held\']) {'
    atom='''        if(datum.atom){let can=document.createElement('canvas');can.width=590;can.height=273;el('charts').append(can);let a=decode(datum.atom);plot(can,'内部单原子：均值与训练能量修正；不是30秒解码','秒',[0,1.995],[a[0],a[0],a[0]],[a[1],a[1],a[1]]);}
'''
    assert marker in template;template=template.replace(marker,atom+marker)
    template=template.replace("+ '片').join", "+ '片 / 固定训练尺度 ' + m.input_scale.toPrecision(5) + ' ' + m.source_unit).join")
    template=template.replace('<section id="charts">','<p>最后的内部原子图：蓝色为训练码本均值原子，橙色为训练能量修正；两条均为固定模型参数，均不是折外统计或完整30秒解码。</p><section id="charts">')
    if primary=='innovation':
        template=template.replace('黑色原子仅是模型内部局部原子。','')
        template=template.replace('最后的内部原子图：蓝色为训练码本均值原子，橙色为训练能量修正；两条均为固定模型参数，均不是折外统计或完整30秒解码。','这些曲线来自完整实发 ID 对应的真实监督成员。后发残差字经冻结空间解码器及先发字合成；完整30秒重建另见波形图。').replace('全部四折两布局完整码使用','全部四折两种编码的完整码使用')
    (out/'index.html').write_text(template)


def branch_static(branch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=STATE['out'];catalog=STATE['catalog'][STATE['cfg']['primary'],branch];profiles={r['ID']:r for r in json.loads((out/f'{branch}_true_properties.json').read_text())};folder=out/'figures';folder.mkdir(exist_ok=True)
    chosen=STATE['chosen'][branch];records=[]
    for first in range(0,len(chosen),2):
        page=chosen[first:first+2];fig,axes=plt.subplots(2,4,figsize=(18,8),layout='constrained')
        for row,selected in enumerate(page):
            entry=next(x for x in catalog if x['ID']==selected['ID']);chunk=(entry['rank']-1)//32;payload=(out/('EEG-'+branch)/f'{chunk}.js').read_text();datum=json.loads(payload[len('window.atlasData='):-1])[(entry['rank']-1)%32]
            def decode(panel):
                if panel is None:return np.empty((0,400))
                return np.frombuffer(base64.b64decode(panel['rows']),dtype='<f4').reshape(-1,panel['length'])
            for col,panel_index in enumerate((0,1,2,3)):
                ax=axes[row,col];x=np.arange(400)/200 if col<3 else np.arange(2,91)/2
                for split,color in [('train','#1e688f'),('held','#b0642d')]:
                    v=decode(datum[split]['panels'][panel_index])
                    if len(v):
                        ax.fill_between(x,v[1],v[2],color=color,alpha=.17);ax.plot(x,v[0],color=color,lw=1.2,label=split+' mean / P10–P90')
                        for member in v[3:]:ax.plot(x,member,color=color,alpha=.45,lw=.55)
                if col==(0 if branch=='S' else 1) and datum['atom']:
                    v=decode(datum['atom']);ax.plot(x,v[0],color='black',ls=':',lw=1,label='Model role atom (not sequence)')
                ax.set(xlabel='Time within physical 2-s member (s)' if col<3 else 'Frequency (Hz)',ylabel='Voltage / fixed train X SD' if col<3 else 'PSD / train X SD² / Hz',title=['Paired true S members','Paired true O members','Paired observed X members','True '+branch+' member spectrum'][col])
                if col==3:ax.set_xlim(1,45)
                ax.legend(fontsize=6.5)
            axes[row,0].set_title(f'Fold0 {branch} ID {entry["ID"]}, paid band {entry["band"]}; {selected["rule"]}\nfit {entry["train"]} / {entry["train_subjects"]} people, held {entry["held"]} / {entry["held_subjects"]}; mass {100*entry["mass"]:.3f}%, cumulative {100*entry["cumulative"]:.2f}%',fontsize=9)
        fig.suptitle(f'Actual paid ID, same fold0 book | original training people only | {STATE["cfg"]["primary"]}\nAll-member conditional means and P10–P90; three deterministic true members prefer different people; per-code shared target axes.',fontsize=11)
        fig.supxlabel('G0 coordinate uses fixed native-site training X SD, no per-member normalization. Exact physical identities and native multipliers: index.html.\nMean cancellation of high-frequency phase does not imply absent semantics. Full 30-s decoded sequence is a separate result; no development/confirmation/test.',fontsize=8)
        path=folder/f'{branch}_detail_band{first//2}.png';fig.savefig(path,dpi=165,bbox_inches='tight');plt.close(fig);records.append(dict(branch=branch,figure=str(path.relative_to(ROOT)),codes=page))
    major=[e for e in catalog if e['major']];fig,ax=plt.subplots(1,3,figsize=(16,9),layout='constrained');rank=np.arange(len(major));a=np.asarray([profiles[e['ID']]['train']['mean'] for e in major],float);h=np.asarray([profiles[e['ID']]['held']['mean'] for e in major],float)
    transform=np.log1p(np.column_stack((a[:,:2],a[:,2:].mean(1))))
    im=ax[0].imshow(transform,aspect='auto',cmap='viridis');ax[0].set(xticks=range(3),xticklabels=['theta','alpha','mean RMS'],ylabel='Actual fit-frequency rank (fold0 only)',title=f'All {len(major)} major90 codes | log1p true train properties');fig.colorbar(im,ax=ax[0],label='log1p(G0 property)')
    ax[1].plot(np.cumsum([e['mass'] for e in major]),rank,color='#225e83');ax[1].axvline(.9,color='gray',ls=':');ax[1].set(xlabel='Cumulative complete fit-member mass',ylabel='Frequency rank',title='Every major code is in searchable index');ax[1].invert_yaxis()
    sc=ax[2].scatter(a[:,0],h[:,0],s=4,c=[e['band'] for e in major],cmap='viridis',vmin=0,vmax=4,alpha=.65);fig.colorbar(sc,ax=ax[2],ticks=range(5),label='Paid role band (Hz)').ax.set_yticklabels(['1–4','4–8','8–13','13–30','30–45']);ax[2].set(xlabel='Train true theta mean (G0²)',ylabel='Held true theta mean (G0²)',title='One point = one paid code in same book')
    fig.suptitle(f'{branch}: complete major90 coverage, not a selection of attractive prototypes\nFour-fold complete usage/dead/rare/cross-person coverage remains in CSV; no cross-fold ID correspondence assumed.',fontsize=11)
    path=folder/f'{branch}_all_major90_overview.png';fig.savefig(path,dpi=175,bbox_inches='tight');plt.close(fig);records.append(dict(branch=branch,figure=str(path.relative_to(ROOT)),major_code_count=len(major),major_fit_mass=sum(e['mass'] for e in major)))
    write(out/f'{branch}_figures.json',records);return records


def revise_display():
    out=STATE['out'];template=(out/'index.html').read_text()
    template=template.replace("(k.startsWith('EEG') ? ' · ' + bands[r.band] + ' Hz' : '')", "(r.ID===0 ? ' · reserved missing' : (k.startsWith('EEG') ? ' · ' + bands[r.band] + ' Hz' : ''))")
    template=template.replace("f === 'dead' && r.train === 0", "f === 'dead' && r.train === 0 && r.ID !== 0")
    template=template.replace('每个成员是一个完整评分的两秒片段，来自真实训练或折外训练观察。','每个成员是一个完整评分的两秒片段。这里只读原60名训练被试；蓝色来自fold 0拟合人，橙色来自同一字典的折外训练人。ID 0仅为缺测保留，不算正码死码。')
    template=template.replace('细线为按固定种子均匀抽取的至多三个真实成员','细线为固定身份哈希、优先不同被试选出的至多三个真实成员')
    (out/'index_v3.html').write_text(template)
    write(out/'display_revision_v3.json',dict(reason='explicit_missing_zero_training_only_and_exact_representative_sampling_language',numeric_data_changed=False,retained_v1='index.html',preferred='index_v3.html'))


def numerical_reference():
    out=STATE['out'];bank=STATE['bank'];codes,_=paid_ids(0);rows=[]
    for (arm,b),word in codes.items():
        ids=word[bank['valid']];values=bank[b+'_attributes'][bank['valid']];fit=np.broadcast_to((bank['fold']!=0)[:,None],bank['valid'].shape)[bank['valid']]
        fields={}
        for part,selection in [('train',fit),('held',~fit)]:
            total=np.zeros((VOCAB,10));np.add.at(total,ids[selection],values[selection]);count=np.bincount(ids[selection],minlength=VOCAB);mean=np.divide(total,count[:,None],out=np.full_like(total,np.nan),where=count[:,None]>0)
            fields[part]=(count,mean)
        for code in range(VOCAB):
            item=dict(fold=0,arm=arm,branch=b,ID=code)
            for part,(count,mean) in fields.items():item.update({part+'_count':int(count[code]),part+'_theta_mean':float(mean[code,0]),part+'_alpha_mean':float(mean[code,1]),part+'_local_RMS_mean':float(mean[code,2:].mean())})
            rows.append(item)
    with (out/'same_fold_layout_true_property_reference.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    p=STATE['model']/'single_word_information.json'
    if p.exists():
        raw=json.loads(p.read_text());write(out/'owning_numeric_reference.json',dict(path=str(p.relative_to(ROOT)),sha256=sha(p),scope='owner_original_raw_readout_no_refit_by_atlas',atlas_remains_diagnostic=True,**{a:raw[a] for a in (STATE['cfg']['primary'],STATE['cfg']['reference'])}))


def review():
    out=STATE['out'];bank=STATE['bank'];rows=STATE['rows'];cache=out/'cache';keys=np.load(cache/'keys.npy',mmap_mode='r');waves=np.load(cache/'waves.npy',mmap_mode='r')
    table={}
    with (out/'all_fold_code_usage.csv').open() as f:
        for row in csv.DictReader(f):
            k=int(row['fold']),row['arm'],row['branch'],int(row['ID'])
            if k in table:raise ValueError('duplicate usage row')
            table[k]=row
    if len(table)!=4*2*2*VOCAB:raise ValueError('full vocabulary/fold table incomplete')
    lookup={tuple(map(int,e)):j for j,e in enumerate(bank['examples'])}
    if len(lookup)!=len(bank['examples']):raise ValueError('duplicate physical position-window')
    source_checks=[];fold0={}
    for fold in range(4):
        budget();actual={(a,b):np.full(bank['valid'].shape,-1,np.int32) for a in (STATE['cfg']['primary'],STATE['cfg']['reference']) for b in 'SO'}
        for d in STATE['datasets']:
            path=STATE['model']/d/f'fold_{fold}_ids.npz'
            with np.load(path,allow_pickle=False) as z:
                ix=np.asarray([lookup[tuple(map(int,e))] for e in z['examples']]);np.testing.assert_array_equal(z['fold'],bank['fold'][ix])
                for (a,b),v in actual.items():v[ix]=z[a+'_'+b]
        for (a,b),word in actual.items():
            if np.any(word<0):raise ValueError('missing actual physical ID')
            for part,selected in [('train',bank['fold']!=fold),('held',bank['fold']==fold)]:
                chosen=bank['valid']&selected[:,None];u,c=np.unique(word[chosen],return_counts=True);expected=np.zeros(VOCAB,np.int64);expected[u]=c
                subject_count=np.zeros(VOCAB,np.int32);dataset_count=np.zeros(VOCAB,np.int32)
                for person in np.unique(bank['subject'][selected]):subject_count[np.unique(word[chosen&(bank['subject'][:,None]==person)])]+=1
                for ds in np.unique(STATE['dataset_index'][selected]):dataset_count[np.unique(word[chosen&(STATE['dataset_index'][:,None]==ds)])]+=1
                for code in range(VOCAB):
                    row=table[fold,a,b,code]
                    if (int(row[part]),int(row[part+'_subjects']),int(row[part+'_datasets']))!=(int(expected[code]),int(subject_count[code]),int(dataset_count[code])):raise ValueError('independent physical-key usage mismatch')
            source_checks.append(dict(fold=fold,arm=a,branch=b,declared_vocabulary=VOCAB,all_counts_people_datasets_exact=True))
        if fold==0:fold0=actual
    # Check every offline record against the independent all-vocabulary count table.
    checked_records=0;details={}
    for b in 'SO':
        index=json.loads((out/'atlas_index.js').read_text()[len('window.atlasIndex='):-1])['EEG-'+b]
        for chunk in range((VOCAB+31)//32):
            datum=json.loads((out/('EEG-'+b)/f'{chunk}.js').read_text()[len('window.atlasData='):-1])
            for j,d in enumerate(datum):
                row=index[chunk*32+j];expected=table[0,STATE['cfg']['primary'],b,int(row['ID'])]
                for part in ('train','held'):
                    if d[part]['count']!=int(expected[part]) or d[part]['subjects']!=int(expected[part+'_subjects']):raise ValueError('offline record support mismatch')
                    for panel in d[part]['panels']:
                        if panel is None:
                            if d[part]['count']!=0:raise ValueError('missing nonempty record')
                        else:
                            value=np.frombuffer(base64.b64decode(panel['rows']),dtype='<f4')
                            if len(value)!=(3+len(d[part]['members']))*panel['length'] or not np.isfinite(value).all():raise ValueError('offline data finite/shape check failed')
                checked_records+=1
                if any(x['ID']==row['ID'] for x in STATE['chosen'][b]):details[b,int(row['ID'])]=d
    # All twenty preregistered code cards: templates from every true member, fresh real-source probes.
    array_cache=OrderedDict();probes=[];template_checks=[];member_lookup={(int(n),int(p)):j for j,(n,p) in enumerate(keys)}
    member_folds=bank['fold'][keys[:,0]];member_people=bank['subject'][keys[:,0]]
    for b in 'SO':
        ids=fold0[STATE['cfg']['primary'],b][bank['valid']]
        for selected in STATE['chosen'][b]:
            code=selected['ID'];datum=details[b,code]
            for part,mask in [('train',member_folds!=0),('held',member_folds==0)]:
                ix=np.flatnonzero((ids==code)&mask)
                for col in range(3):
                    panel=datum[part]['panels'][col]
                    if not len(ix):continue
                    actual=np.frombuffer(base64.b64decode(panel['rows']),dtype='<f4').reshape(-1,400)
                    target=np.asarray(waves[ix,col]);expected=np.vstack((target.mean(0,dtype=np.float64),np.quantile(target,[.1,.9],axis=0))).astype('f4');np.testing.assert_array_equal(actual[:3],expected)
                if datum[part]['members']:
                    member=datum[part]['members'][0];example=np.asarray([[member['g0_row'],member['crop_s'],rows[member['g0_row']]['positions'].index(member['position'])]])
                    window=lookup[tuple(example[0])];patch=member['patch'];n=member_lookup[window,patch]
                    if int(ids[n])!=code or (part=='held')!=(bank['fold'][window]==0):raise ValueError('representative code/fold mismatch')
                    raw=base_tokenizer_batch(rows,example,STATE['scales'],array_cache=array_cache,cache_limit=16)
                    if not raw['mask'][0,patch].all():raise ValueError('incomplete raw member')
                    for col,key in enumerate(('s','o','x')):
                        np.testing.assert_array_equal(raw[key][0,patch],waves[n,col]);panel=np.frombuffer(base64.b64decode(datum[part]['panels'][col]['rows']),dtype='<f4').reshape(-1,400);np.testing.assert_array_equal(panel[3],raw[key][0,patch])
                    probes.append(dict(branch=b,ID=code,partition=part,identity=member,raw_S_O_X_exact=True))
                template_checks.append(dict(branch=b,ID=code,partition=part,complete_members=len(ix),full_true_template_and_quantiles_exact=True))
    write(out/'independent_review.json',dict(passed=True,independent_alignment='physical_G0_row_crop_position_keys_no_dataset_iteration_assumption',all_fold_full_vocabulary_checks=source_checks,offline_all_records_checked=checked_records,
        all_twenty_code_template_checks=template_checks,fresh_original_data_probes=probes,validation_reads=0,confirmation_reads=0,test_reads=0,CPU_seconds=budget()))
    return True


def browser_render():
    from playwright.sync_api import sync_playwright
    out=STATE['out'];folder=out/'browser_v3';folder.mkdir(exist_ok=True);catalog=json.loads((out/'atlas_index.js').read_text()[len('window.atlasIndex='):-1]);records=[];errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox','--allow-file-access-from-files','--renderer-process-limit=2','--num-raster-threads=1','--disable-gpu'])
        page=browser.new_page(viewport={'width':1440,'height':1500},device_scale_factor=1)
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto((out/'index_v3.html').resolve().as_uri(),wait_until='load')
        for b in 'SO':
            entries=catalog['EEG-'+b];major=entries[0];weak=max((r for r in entries if r['held']>0 and r['low_support']),key=lambda r:(r['held'],-r['ID']))
            for rule,row in [('highest_fit_occupancy',major),('highest_held_low_support',weak)]:
                budget();page.select_option('#branch','EEG-'+b);page.select_option('#filter','all');page.select_option('#code',str(row['rank']));page.wait_for_function("count => document.querySelectorAll('#charts canvas').length === count",arg=(5 if STATE['cfg']['primary']=='innovation' else 6))
                file=folder/f'{b}_{rule}_ID{row["ID"]}.png';page.screenshot(path=str(file),full_page=True)
                members=page.locator('#members').inner_text();status=page.locator('#status').inner_text()
                if '固定训练尺度' not in members:raise ValueError('native scale/identity missing in rendered page')
                records.append(dict(branch=b,ID=row['ID'],rule=rule,status=status,members=members,screenshot=str(file.relative_to(ROOT)),rendered_panel_count=(5 if STATE['cfg']['primary']=='innovation' else 6),all_required_canvases_drawn=True))
        browser.close()
    if errors:raise ValueError('offline browser script errors: '+str(errors))
    write(out/'browser_review_v3.json',dict(passed=True,page_errors=errors,records=records,CPU_seconds=budget(),URL='file:// local offline only'))


def coverage():
    """Compact exact all-fold code-by-person occupancy, without treating windows as people."""
    bank=STATE['bank'];arms=(STATE['cfg']['primary'],STATE['cfg']['reference']);counts=np.zeros((4,2,2,len(STATE['pop']['subjects']),VOCAB),np.int32)
    person_fold=np.asarray([np.unique(bank['fold'][bank['subject']==p]).item() for p in range(len(STATE['pop']['subjects']))],np.int16)
    for fold in range(4):
        budget();codes,_=paid_ids(fold)
        for a,arm in enumerate(arms):
            for j,b in enumerate('SO'):
                word=codes[arm,b]
                for person in range(len(person_fold)):
                    selected=bank['valid']&(bank['subject'][:,None]==person)
                    counts[fold,a,j,person]=np.bincount(word[selected],minlength=VOCAB)
    path=STATE['out']/'code_by_subject_counts.npz'
    metadata=dict(schema='band_exchange_all_fold_code_by_subject_v1',axes=['source_fold','layout','branch','original_training_person','paid_ID'],layouts=list(arms),branches=['S','O'],people=STATE['pop']['subjects'],person_held_fold=person_fold.tolist(),complete_patch_only=True,missing_reserved_ID=0,compare_numeric_ID_across_folds=False)
    np.savez_compressed(path,counts=counts,metadata=json.dumps(metadata,ensure_ascii=False))
    with (STATE['out']/'all_fold_code_usage.csv').open() as f:
        for row in csv.DictReader(f):
            fold=int(row['fold']);a=arms.index(row['arm']);b='SO'.index(row['branch']);code=int(row['ID']);v=counts[fold,a,b,:,code]
            for name,selected in [('train',person_fold!=fold),('held',person_fold==fold)]:
                if int(v[selected].sum())!=int(row[name]) or int(np.count_nonzero(v[selected]))!=int(row[name+'_subjects']):raise ValueError('per-person occupancy sum or support mismatch')
    write(STATE['out']/'cross_subject_coverage_review.json',dict(passed=True,archive=path.name,shape=counts.shape,axes=metadata['axes'],all_vocab_all_fold_all_layout_person_counts_exact=True,complete_patch_totals=counts.sum(axis=(3,4)),CPU_seconds=budget()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--review',action='store_true');p.add_argument('--browser',action='store_true');p.add_argument('--coverage',action='store_true');args=p.parse_args();cfg=yaml.safe_load((ROOT/args.config).read_text());initialize(cfg);out=STATE['out'];lock=(out/'producer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);threadpool_limits(1)
    if args.coverage:coverage();return
    if args.review or args.browser:
        selection=json.loads((out/'selection.json').read_text());STATE['chosen']=selection['detailed']
        if args.review:review();numerical_reference();revise_display()
        if args.browser:revise_display();browser_render()
        return
    if not (out/'resolved_config.json').exists():write(out/'resolved_config.json',dict(config=cfg,source_contract_sha256=sha(ROOT/cfg['source_contract']),word_contract_sha256=sha(STATE['model']/('resolved_config.json' if cfg['primary']=='innovation' else 'word_contract.json')),code_sha256=sha(Path(__file__)),HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),unit=os.environ.get('BAND_WORD_ATLAS_UNIT')))
    catalog,chosen=usage();STATE.update(catalog=catalog,chosen=chosen);prepare_cache();ctx=multiprocessing.get_context('fork')
    with ProcessPoolExecutor(max_workers=2,mp_context=ctx) as pool:exports=list(pool.map(export_branch,'SO'))
    html_index()
    with ProcessPoolExecutor(max_workers=2,mp_context=ctx) as pool:figures=list(pool.map(branch_static,'SO'))
    write(out/'summary.json',dict(status='research_diagnostic_atlas_complete_pending_actual_open',primary=cfg['primary'],reference=cfg['reference'],fold0_atlas=exports,figures=figures,all_four_fold_usage='all_fold_code_usage.csv',rare_and_failure_list='rare_dead_heldonly_codes.csv',indexed_true_members='index.html',CPU_seconds=budget(),validation_reads=0,confirmation_reads=0,test_reads=0,parent_joint_verdict='separate_numeric_owner_required_never_atlas_success'))

if __name__=='__main__':main()
