"""Plot paired development examples and actual token members of the full QC refit."""
import argparse
from collections import Counter, OrderedDict, defaultdict
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

from apply_joint_tokenizer import checked_path, load_branch_pair
from evaluate_joint_tokenizer import input_scales
from export_joint_member_atlas import derived
from retrain_joint_qc_tokenizer import ROOT, OUT, configuration
from src.analysis.dual_component_tokenizer import joint_reconstruction_metrics
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_joint_eeg, base_tokenizer_training_data

EVAL = OUT/'evaluation/retrained'
DATASETS = dict(eeg_fnirs_single_trial='Single-Trial', refed='REFED',
                simultaneous_eeg_nirs='Simultaneous', visual_cognitive_motivation='Visual')
INK, TEAL, RUST, BLUE = '#193545', '#007a73', '#b94c29', '#315e94'
BRANCH = ['X', 'S', 'O']
plt.rcParams.update({'font.family':['Noto Sans CJK JP','DejaVu Sans'],'font.size':11,
                     'axes.spines.right':False,'axes.spines.top':False,'text.color':INK,
                     'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'axes.unicode_minus':False})


def json_value(v):
    if isinstance(v,np.ndarray):
        return v.tolist()
    if isinstance(v,np.generic):
        return v.item()
    raise TypeError(type(v))


def select_cases(rows, admitted):
    """Equal rank-distance targets; within a library pair both modalities and both tails."""
    scores={}
    populations={}
    for m in ['eeg','hb']:
        parts=[]
        for ds in DATASETS:
            with np.load(EVAL/f'{m}_{ds}_validation.npz') as z:
                ex=z['examples']
                assert all(rows[i]['split']=='validation' and (ds,rows[i]['subject']) in admitted
                           and rows[i]['dataset']==ds and rows[i]['modality']==m for i in np.unique(ex[:,0]))
                parts.append(pd.DataFrame(dict(ds=ds,row=ex[:,0],start=ex[:,1],pos=ex[:,2],
                    parent=[rows[i]['parent'] for i in ex[:,0]],subject=[rows[i]['subject'] for i in ex[:,0]],
                    err=z['nmse'].mean((1,2)),corr=z['correlation'].mean((1,2)))))
        full=pd.concat(parts,ignore_index=True)
        pop=full[np.isfinite(full.err)&np.isfinite(full['corr'])].copy()
        populations[m]={'total':len(full),'finite_both':len(pop)}
        pop['err_rank']=pop.err.rank(pct=True)
        pop['corr_rank']=pop['corr'].rank(pct=True)
        scores[m]=pop
    keys=['ds','subject','parent','start']
    pools={}
    for label,target in [('good',.10),('poor',.90)]:
        pairs=[]
        for m,frame in scores.items():
            f=frame.assign(cost=(frame.err_rank-target).abs()+(frame.corr_rank-(1-target)).abs())
            f=f.sort_values(['cost','row','pos']).drop_duplicates(keys)
            pairs.append(f.rename(columns={k:m+'_'+k for k in f.columns if k not in keys}))
        both=pairs[0].merge(pairs[1],on=keys,validate='one_to_one')
        both['loss']=both.eeg_cost+both.hb_cost
        pools[label]=both.sort_values(['loss',*keys])
    choices=[]
    for ds in DATASETS:
        # For each subject retain their nearest paired window; ties follow the G0 keys.
        g=pools['good'].query('ds == @ds').drop_duplicates('subject').to_dict('records')
        p=pools['poor'].query('ds == @ds').drop_duplicates('subject').to_dict('records')
        matches=[(a['loss']+b['loss'],a,b) for a in g for b in p if a['subject']!=b['subject']]
        if matches:
            choices.append(min(matches,key=lambda v:v[0]))
    selected=sorted(choices,key=lambda v:v[0])[:2]
    assert len(selected)==2
    return populations,len(pools['good']),[(label,j,choice[i]) for label,i in [('good',1),('poor',2)]
                                         for j,choice in enumerate(selected,1)]


def decode(mod,ds,take,model,rows,scales):
    i,start,position=take
    batch=base_tokenizer_batch(rows,[take],scales)
    if mod=='eeg':
        names=model.metadata['positions'][ds]
        crop=dict(row=i,start_s=start,positions=[position])
        fields,seen,_=base_tokenizer_joint_eeg(rows,crop,scales,names,allowed_splits=('validation',))
        ids=model.encode(fields['x'].T.reshape(1,len(names),15,400),ds,seen[None])
        full=model.decode_ids(ids,ds)
        idx=[s.upper() for s in names].index(rows[i]['positions'][position].upper())
        ids={b:v[:,idx] for b,v in ids.items()}
        pred={b:v[:,idx] for b,v in full.items()}
    else:
        ids=model.encode(batch['x'])
        pred=model.decode_ids(ids)
    colors=1 if mod=='eeg' else 2
    metric=joint_reconstruction_metrics(pred,batch,colors,input_scales(rows,[take],scales))['metrics']
    with np.load(EVAL/f'{mod}_{ds}_validation.npz') as z:
        where=np.flatnonzero((z['examples']==take).all(1))
        assert len(where)==1
        for b in 'SO':
            np.testing.assert_array_equal(ids[b],z[b][where])
        for k in ['nmse','correlation','rms_gain']:
            np.testing.assert_allclose(metric[k],z[k][where],atol=2e-6,rtol=2e-5)
    mask=batch['mask'].reshape(-1,colors)
    target=[np.where(mask,batch[b].reshape(-1,colors),np.nan) for b in ('x','s','o')]
    predicted=[np.where(mask,pred[b].reshape(-1,colors),np.nan) for b in ('sum','S','O')]
    return target,predicted,{k:metric[k][0] for k in ['nmse','correlation','rms_gain']},mask


def export_case(out,label,j,record,models,rows,scales):
    ds=record['ds']; data={}; detail=dict(selection=record)
    for mod in ['eeg','hb']:
        i,t,k=(int(record[mod+'_row']),int(record['start']),int(record[mod+'_pos']))
        data[mod]=decode(mod,ds,(i,t,k),models[mod],rows,scales)
        r=rows[i]
        detail[mod]=dict(position=r['positions'][k],record=r['record'],parent=r['parent'],row=i,
                         start_s=t,position_index=k,valid_points=data[mod][3].sum(0),
                         metrics=data[mod][2],sync_start_200=round(r['native_start_sample']*200/r['native_rate_hz'])+200*t)
    assert detail['eeg']['sync_start_200']==detail['hb']['sync_start_200']
    wave=data['eeg'][0][0][:,0].reshape(15,400)
    complete=np.isfinite(wave).all(1)
    assert complete.any()
    zoom=int(np.argmax(np.where(complete,np.nanstd(wave,axis=1),-np.inf)))*2
    detail['zoom_start_s']=zoom
    fig,axes=plt.subplots(3,4,figsize=(17.2,8.4),gridspec_kw={'width_ratios':[1.25,1,1,1]})
    fig.subplots_adjust(left=.061,right=.985,bottom=.15,top=.76,wspace=.28,hspace=.55)
    name='良好' if label=='good' else '不良'
    fig.text(.04,.95,f'{name}重建  {j}   ·   {DATASETS[ds]}  /  {record["subject"]}',fontsize=22,weight='bold')
    fig.text(.04,.895,f'同一 30 秒窗   ·   记录 {detail["eeg"]["record"]}   ·   同步时钟起点 {detail["eeg"]["sync_start_200"]/200:.3f} s'
             f'     |     EEG {detail["eeg"]["position"]}      fNIRS {detail["hb"]["position"]}',fontsize=13)
    for col,title in enumerate(['EEG  ·  全窗',f'EEG  ·  {zoom}–{zoom+2} s 放大','fNIRS  ·  HbO 氧合','fNIRS  ·  HbR 脱氧']):
        box=axes[0,col].get_position()
        fig.text((box.x0+box.x1)/2,.815,title,ha='center',fontsize=14,weight='bold',color=TEAL)
    for col in range(4):
        m='eeg' if col<2 else 'hb';ch=1 if col==3 else 0
        original,recon,metrics,_=data[m];rate=200 if col<2 else 10
        time=np.arange(30*rate)/rate
        for b,ax in enumerate(axes[:,col]):
            y=original[b][:,ch];pr=recon[b][:,ch]
            if col==0:ax.axvspan(zoom,zoom+2,color='#c8a863',alpha=.23)
            ax.plot(time,y,color=INK,lw=.85 if col==0 else 1.5,label='目标')
            ax.plot(time,pr,color=RUST,lw=.85 if col==0 else 1.45,ls='--',label='整数解码')
            ax.grid(axis='y',alpha=.16)
            ax.set_xlim((zoom,zoom+2) if col==1 else (0,30))
            if col in (0,1):
                lo=np.nanmin([y,pr]);hi=np.nanmax([y,pr]);pad=max((hi-lo)*.06,.00001)
                ax.set_ylim(lo-pad,hi+pad)
            if col!=1:
                ax.set_title(f'NMSE {metrics["nmse"][b,ch]:.3f}   ·   r {metrics["correlation"][b,ch]:.2f}',fontsize=10,loc='left',pad=4)
            if col==0:ax.set_ylabel(['X  总观测','S  分量','O  分量'][b]+'（SD）',fontsize=12)
            if b==2:ax.set_xlabel('窗内时间（s）')
            ax.tick_params(labelsize=9,length=2)
    fig.legend([Line2D([],[],c=INK),Line2D([],[],c=RUST,ls='--')],['目标：观测 X 或模型监督 S、O','重建：仅由整数解码'],
               loc='lower center',bbox_to_anchor=(.5,.061),frameon=False,ncols=2)
    fig.text(.04,.024,'幅度单位为该位置的训练输入标准差（SD）。各面板按本波形取纵轴，EEG 放大图沿用左图纵轴。阴影选取窗内观测电压起伏最大的完整 2 秒。指标均按全窗有效点计算。',fontsize=10)
    filename=f'{label}_{j}.png'
    fig.savefig(out/filename,dpi=205,facecolor='white')
    plt.close(fig)
    detail['file']=filename
    return detail


def word_batches(rows,scales,admitted,mod,split,chosen=None):
    cache=OrderedDict()
    for ds in DATASETS:
        with np.load(EVAL/f'{mod}_{ds}_{split}.npz') as z:
            ex=z['examples'];ids={b:z[b] for b in 'SO'}
        for row_id in np.unique(ex[:,0]):
            r=rows[row_id]
            assert r['split']==split and r['dataset']==ds and r['modality']==mod
            assert split=='train' or (ds,r['subject']) in admitted
        indices=np.arange(len(ex)) if chosen is None else np.flatnonzero(np.logical_or.reduce(
            [np.isin(ids[b],list(chosen[b])).any(1) for b in 'SO']))
        for start in range(0,len(indices),96):
            idx=indices[start:start+96];take=ex[idx]
            batch=base_tokenizer_batch(rows,take,scales,array_cache=cache,cache_limit=64)
            yield ds,take,batch,{b:ids[b][idx] for b in 'SO'}


def select_words(rows,scales,admitted,mod):
    counts={b:Counter() for b in 'SO'}
    slopes={b:Counter() for b in 'SO'}
    for _,_,batch,ids in word_batches(rows,scales,admitted,mod,'train'):
        valid=batch['mask'].all(2)
        for b in 'SO':
            v=ids[b][valid].astype(int)
            cnt=np.bincount(v)
            counts[b].update({int(k):int(cnt[k]) for k in np.flatnonzero(cnt)})
            if mod=='hb':
                y=batch[b.lower()][valid].reshape(-1,20,2)[:,:,0]
                t=(np.arange(20)-9.5)/10
                sums=np.bincount(v,weights=y@(t/(t@t)))
                slopes[b].update({int(k):float(sums[k]) for k in np.flatnonzero(cnt)})
    result={}
    for b in 'SO':
        ordered=sorted(counts[b],key=lambda k:(-counts[b][k],k))
        if mod=='eeg':
            result[b]={}
            for band,role_name in [(1,'θ'),(2,'α')]:
                for offset,width,kind in [(1,1023,'导引词'),(5116,2047,'残差词')]:
                    low=offset+band*width
                    code=next(k for k in ordered if low<=k<low+width)
                    result[b][code]=role_name+' · '+kind
        else:
            top=[v for v in ordered if v!={'S':128,'O':256}[b]][:10]
            top.sort(key=lambda k:(slopes[b][k]/counts[b][k],k))
            result[b]={top[0]:'平均斜率最低',top[-1]:'平均斜率最高'}
    return result,counts


def members(rows,scales,admitted,mod,chosen):
    values=defaultdict(list);people=defaultdict(set);examples={}
    for split in ('train','validation'):
        for ds,take,batch,ids in word_batches(rows,scales,admitted,mod,split,chosen):
            valid=batch['mask'].all(2)
            for b in 'SO':
                for code in chosen[b]:
                    key=(b,code,split)
                    ix,slot=np.where((ids[b]==code)&valid)
                    if not len(ix):continue
                    v=batch[b.lower()][ix,slot].astype('f4')
                    values[key].append(v)
                    people[key].update((ds,rows[take[i,0]]['subject']) for i in np.unique(ix))
                    if key not in examples:
                        examples[key]=(take[ix[0]].tolist()+[int(slot[0])],v[0].copy())
    return {k:np.concatenate(v) for k,v in values.items()},people,examples


def plot_words(out,mod,branches,chosen,counts,values,people,examples):
    rows=[(b,code,desc) for b in branches for code,desc in chosen[b].items()]
    eeg=mod=='eeg'
    titles=['单片分量电压','单片 Hann 功率谱','0.25 秒局部 RMS'] if eeg else ['HbO 原幅度','HbR 原幅度','HbO 去均值形状','HbR 去均值形状']
    fig,axs=plt.subplots(len(rows),len(titles),figsize=(15.2,9.6))
    fig.subplots_adjust(left=.148,right=.985,top=.795,bottom=.12,wspace=.23,hspace=.37)
    label='EEG-'+branches[0] if eeg else 'fNIRS-S / O'
    fig.text(.035,.95,f'{label}   ·   同一个 token 的实际成员',fontsize=22,weight='bold')
    desc=('各分支在 θ、α 频带的导引词与残差词中，分别取训练完整片段数最多的 ID。频带标签描述编码角色。'
          if eeg else '各分支从训练占用最高的十个普通词中，按训练成员 HbO 平均斜率选最低和最高的 ID。')
    fig.text(.035,.897,desc,fontsize=12)
    for col,t in enumerate(titles):
        pos=axs[0,col].get_position()
        fig.text((pos.x0+pos.x1)/2,.833,t,ha='center',fontsize=14,weight='bold',color=TEAL)
    stats=[]
    for i,(b,code,role) in enumerate(rows):
        rowdata={}
        ax=axs[i,0]; pos=ax.get_position()
        fig.text(.021,pos.y0+pos.height*.77,f'{b}  ·  ID {code}',weight='bold',fontsize=13)
        fig.text(.021,pos.y0+pos.height*.53,role,fontsize=11)
        lines=[]
        item=dict(branch=b,ID=code,rule=role)
        for split,cn,col,ls in [('train','训',BLUE,'-'),('validation','开',RUST,'--')]:
            key=(b,code,split)
            mat=values.get(key)
            if mat is None:
                lines.append(cn+' 0 片');item[split]=dict(patches=0,subjects=0);continue
            if split=='train':assert len(mat)==counts[b][code]
            lines.append(f'{cn} {len(mat):,} 片 / {len(people[key])} 人')
            data=derived(mat,mod);rowdata[split]=data
            item[split]=dict(patches=len(mat),subjects=len(people[key]),example=examples[key][0])
            if not eeg:
                item[split]['mean_HbO_end_minus_start']=float((data[0][:,-1]-data[0][:,0]).mean())
            for j,y in enumerate(data):
                t=(np.arange(y.shape[1])/200 if j==0 else np.arange(2,91)*.5 if j==1 else .125+np.arange(8)*.25) if eeg else np.arange(20)/10
                mean=y.mean(0,dtype=np.float64);lo,hi=np.percentile(y,[10,90],axis=0)
                a=axs[i,j]
                if eeg and j==0 and split=='validation':
                    a.plot(t,examples[key][1],color='#89968a',lw=.8,alpha=.85,zorder=1)
                a.fill_between(t,lo,hi,color=col,alpha=.14,linewidth=0)
                a.plot(t,mean,color=col,ls=ls,lw=1.45)
                if eeg and j==1:
                    hz=(4,8) if 'θ' in role else (8,13)
                    if split=='train':a.axvspan(*hz,color='#cfac60',alpha=.17)
                    a.set_yscale('log')
                    a.set_xticks([1,10,20,30,45]);a.set_xlim(1,45)
                else:a.set_xlim(t[0],t[-1])
                a.grid(axis='y',alpha=.16);a.tick_params(labelsize=8,length=2)
                if i==len(rows)-1:a.set_xlabel('频率（Hz）' if eeg and j==1 else '片内时间（s）',fontsize=10)
        fig.text(.021,pos.y0+pos.height*.11,'\n'.join(lines),fontsize=9,linespacing=1.45)
        stats.append(item)
    for j in range(len(titles)):
        limits=[a.get_ylim() for a in axs[:,j]]
        for a in axs[:,j]:
            a.set_ylim(min(x[0] for x in limits),max(x[1] for x in limits))
    handles=[Line2D([],[],color=BLUE,lw=2),Line2D([],[],color=RUST,ls='--',lw=2)]
    labels=['训练成员均值','开发成员均值']
    if eeg:
        handles.append(Line2D([],[],color='#89968a',lw=1));labels.append('首个完整开发成员（电压）')
    fig.legend(handles,labels,ncols=len(labels),loc='lower center',bbox_to_anchor=(.5,.035),frameon=False)
    foot='功率谱为 SD²/Hz（对数纵轴），金色阴影标示该词频带。' if eeg else '去均值仅减去各成员本色团 2 秒平均值，保留片内起伏。'
    fig.text(.035,.012,'色带为全部成员逐点 P10–P90；幅度为该位置训练 X 的 SD。同列各行共用纵轴。'+foot,fontsize=10)
    file=f'pattern_{mod}'+('_'+branches[0] if eeg else '')+'.png'
    fig.savefig(out/file,dpi=205,facecolor='white');plt.close(fig)
    return dict(file=file,words=stats)


def main(out):
    torch.set_num_threads(1)
    cfg=configuration();admitted={tuple(v) for v in cfg['data']['validation_subjects']}
    assert len(admitted)==16
    manifest,scales,_,_=base_tokenizer_training_data(cfg,ROOT)
    rows=manifest['rows']
    spec=json.loads((OUT/'bundle.json').read_text())
    assert json.loads((EVAL/'completion.json').read_text())['status']=='completed'
    models={m:load_branch_pair(spec[m]['kind'],checked_path(OUT/'bundle.json',spec[m])) for m in ['hb','eeg']}
    populations,paired,chosen=select_cases(rows,admitted)
    out.mkdir(parents=True,exist_ok=False)
    results=dict(bundle=str(OUT/'bundle.json'),model_hashes={m:spec[m]['sha256'] for m in models},
                 position_population=populations,paired_parent_windows=paired,cases=[],patterns=[])
    for label,j,rec in chosen:
        results['cases'].append(export_case(out,label,j,rec,models,rows,scales))
        print(label,j,rec['ds'],rec['eeg_err'],rec['hb_err'],flush=True)
    for mod in ['eeg','hb']:
        chosen,counts=select_words(rows,scales,admitted,mod)
        val,people,examples=members(rows,scales,admitted,mod,chosen)
        for branches in (['S','O'] if mod=='eeg' else ['SO']):
            results['patterns'].append(plot_words(out,mod,branches,chosen,counts,val,people,examples))
        print(mod,'patterns rendered',flush=True)
    (out/'examples.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,default=json_value)+'\n')
    entries=[(v['file'],('良好' if v['file'].startswith('good') else '不良')+'重建 · '+DATASETS[v['selection']['ds']]) for v in results['cases']]
    entries+= [(v['file'],v['file'].replace('pattern_eeg_','EEG-').replace('pattern_hb','fNIRS').replace('.png','')+' 实际 token 成员') for v in results['patterns']]
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>重训 tokenizer 的时序重建与单词成员</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1350px;margin:24px auto;padding:0 18px;color:#193545}figure{margin:24px 0 42px}img{width:100%;height:auto}a{color:#007a73}figcaption{font-weight:650}p{max-width:1100px}</style><h1>四分支 tokenizer 的时序重建与单词成员</h1>'''
    html+='<p>使用 2026-11-24 质控后完整重训模型。每组为同一被试、同一父窗、同一 30 秒的 EEG 与 fNIRS，位置各自选取。编码只见观测 X；S、O 是分量监督。NMSE 用该位置的冻结训练目标方差归一化，r 为有效点上的皮尔逊相关。</p>'
    html+=f'<p>选例来自 16 名普通开发被试。按模态分别在四库有限指标的位置窗中排名，每个位置先对 X/S/O 和色团等权平均 NMSE 与 r。良好组以误差 P10、相关 P90 为目标，不良组反向。在 {paired:,} 个可配对的父窗及起点组合中，最小化两模态到目标百分位的绝对距离之和；每库两组用不同被试，再取两组总距离最小的两个库。这是展示选例，组内 EEG 与 fNIRS 位置不作解剖配对。</p>'
    html+='<p>单词面板仅使用该 ID 的完整 2 秒片段。训练侧为冻结的词义审计样本，开发侧为完整普通开发面板。训练侧选词，训练和开发分别汇总。均值与分位带描述给定词号时的监督分量分布。</p>'
    html+=''.join(f'<figure><figcaption>{escape(label)}</figcaption><a href="{file}"><img src="{file}" alt="{escape(label)}"></a></figure>' for file,label in entries)
    (out/'index.html').write_text(html+'<p><a href="examples.json">样例身份、评分及单词支持数</a></p></html>')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
