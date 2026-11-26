"""Score matched inner checkpoints and render real waveform and word-member comparisons."""
import argparse
from collections import OrderedDict, defaultdict
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import periodogram

from run_aligned_tokenizer_optimization import OUT, load, settings, _base_training_model, write_json, read_json, owner_lock
from src.analysis.dual_component_tokenizer import base_training_metrics, base_training_aggregate
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_examples

plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
STAGES=('parent_10000','full_11000','full_12000','decoder_11000','decoder_12000')


def anchors(rows,examples):
    groups=defaultdict(list)
    for j,(idx,_,_) in enumerate(examples):
        r=rows[idx]
        groups[r['dataset'],r['subject']].append(j)
    rng=np.random.default_rng(1126)
    return sorted(int(j) for k,ids in sorted(groups.items()) for j in rng.choice(ids,min(len(ids),64),replace=False))


def checkpoint(mod,stage,task):
    arm,step=stage.split('_')
    if arm=='parent':
        return OUT/mod/'models'/'__'.join(task.values())/'checkpoint.pt'
    return OUT/mod/arm/f'step_{step}.pt'


def infer(model,raw,device):
    x=torch.as_tensor(raw['x'],device=device)
    res=model(x)
    dec=model.decode_ids(res['ids'])
    for k in ('sum','S','O'):
        torch.testing.assert_close(res[k],dec[k],atol=0,rtol=0)
    pred={k:dec[k].cpu().numpy() for k in ('sum','S','O')}
    ids=np.stack([res['ids'][b].cpu().numpy() for b in 'SO'],-1)
    return pred,ids


def wavefig(mod,stage,selected,model,rows,scales,device,dest):
    ledger=[]
    for label,triple in selected.items():
        i,t,p=triple
        r=rows[i]
        raw=base_tokenizer_batch(rows,[triple],scales)
        pred,ids=infer(model,raw,device)
        ch=r['channels']
        tim=np.arange(30*r['rate'])/r['rate']
        fig,axs=plt.subplots(3,ch,figsize=(12,7),squeeze=False,layout='constrained')
        for j,(k,b) in enumerate(zip('xso',('sum','S','O'))):
            tgt=raw[k][0].reshape(-1,ch)
            rec=pred[b][0].reshape(-1,ch)
            mask=raw['mask'][0].reshape(-1,ch)
            for c in range(ch):
                ax=axs[j,c];target=np.where(mask[:,c],tgt[:,c],np.nan)
                rmse=float(np.sqrt(np.nanmean((target-rec[:,c])**2)))
                color='EEG' if ch==1 else ('HbO','HbR')[c]
                ax.plot(tim,target,c='#59656b',lw=.8,label='实测 X' if k=='x' else 'v3 目标 '+k.upper())
                ax.plot(tim,rec[:,c],c='#18728d',lw=1,label='整数解码')
                ax.set(title=f'{color} · {k.upper()} · RMSE {rmse:.3f} SD',ylabel='训练输入 SD')
                if j==2:ax.set_xlabel('窗内时间（秒）')
                ax.legend(fontsize=8,ncol=2)
                ledger.append(dict(case=label,dataset=r['dataset'],subject=r['subject'],record=r['record'],parent=r['parent'],
                                   start=t,position=r['positions'][p],component=k,color=color,rmse_sd=rmse,valid=int(mask[:,c].sum())))
        fig.suptitle(f'{stage} · {label} · {r["dataset"]} / {r["subject"]} / {r["record"]} / {r["positions"][p]} / {t} s')
        fig.savefig(dest/(label+'.png'),dpi=210);plt.close(fig)
    pd.DataFrame(ledger).to_csv(dest/'wave_examples.csv',index=False)


def partner_figures(mod,stage,parts,dest):
    """Fix one actual ID, vary the other. All pair selection uses training frequencies."""
    tr,_,tc=parts['train']
    dev,w,dc=parts['validation']
    outputs=[]
    for j,branch in enumerate('SO'):
        ids=pd.DataFrame({'fixed':tc[:,j],'partner':tc[:,1-j],'person':tr.dataset+' / '+tr.subject.astype(str)})
        pairs=ids.groupby(['fixed','partner']).agg(n=('person','size'),people=('person','nunique')).reset_index()
        pairs=pairs[pairs.people>=3].sort_values(['n','fixed','partner'],ascending=[False,True,True])
        sizes=pairs.groupby('fixed').size()
        candidates=pairs[pairs.fixed.isin(sizes[sizes>=2].index)]
        if candidates.empty:
            outputs.append(dict(fixed_branch=branch,status='no_train_pair_with_three_people_and_two_partners'))
            continue
        fixed=int(candidates.iloc[0].fixed)
        chosen=candidates[candidates.fixed==fixed].head(3)
        fig,axes=plt.subplots(2,len(chosen),figsize=(4.3*len(chosen),6),squeeze=False,layout='constrained')
        for col,(_,r) in enumerate(chosen.iterrows()):
            partner=int(r.partner);take=np.flatnonzero((dc[:,j]==fixed)&(dc[:,1-j]==partner))
            n=len(take);den=dev.iloc[take].drop_duplicates(['dataset','subject'])
            axes[0,col].set_title(f'另一支 ID {partner} · {n} 个片段 / {len(den)} 人')
            info=dict(fixed_branch=branch,fixed_word=fixed,partner_word=partner,training_pairs=int(r.n),
                      training_people=int(r.people),inner_pairs=n,inner_people=len(den))
            if n==0:
                for a in axes[:,col]:
                    a.set_axis_off();a.text(.5,.5,'内部留出无该组合',ha='center',transform=a.transAxes)
                outputs.append(info);continue
            x=w[take,2].reshape(n,-1,1 if mod=='eeg' else 2)
            t=np.arange(x.shape[1])/(200 if mod=='eeg' else 10)
            for c in range(x.shape[-1]):
                color=['#18728d'] if mod=='eeg' else ['#b44e20','#18728d']
                name='EEG' if mod=='eeg' else ('HbO','HbR')[c]
                axes[0,col].plot(t,x[0,:,c],color=color[c],label=name)
            first=dev.iloc[take[0]]
            info.update(example=first.to_dict())
            axes[0,col].set(xlabel='秒',ylabel='实测 X / 训练输入 SD')
            axes[0,col].legend(fontsize=8)
            if mod=='eeg':
                hz,ps=periodogram(x[:,:,0],fs=200,axis=1,window='hann')
                sel=(hz>=1)&(hz<=45)
                axes[1,col].semilogy(hz[sel],np.maximum(np.median(ps[:,sel],axis=0),1e-12))
                axes[1,col].set(xlabel='频率（Hz）',ylabel='成员 PSD 中位数（SD²/Hz）')
            else:
                clock=t-t.mean()
                for c,color in enumerate(['#b44e20','#18728d']):
                    slopes=x[:,:,c]@clock/(clock@clock)
                    axes[1,col].plot(np.sort(slopes),np.arange(1,n+1)/n,color=color,marker='.' if n<4 else None)
                axes[1,col].set(xlabel='双色实测斜率（SD/s）',ylabel='累计片段比例')
            outputs.append(info)
        fig.suptitle(f'{stage} · {mod.upper()} 固定 {branch}-ID {fixed} 的真实共现组合\n上排为各组合首个留出成员的实测 X，下排为全部成员；按训练支持选词，每列独立纵轴')
        fig.savefig(dest/f'fixed_{branch}_partners.png',dpi=210);plt.close(fig)
    write_json(dest/'partner_support.json',outputs)


def patternfig(mod,stage,rows,scales,model,groups,device,dest):
    """Fixed panels; training counts choose the word. Holdout only supplies members and controls."""
    parts={}
    channels=1 if mod=='eeg' else 2
    cache=OrderedDict()
    for split,triples in groups.items():
        meta=[]; waves=[]; codes=[]
        for j in range(0,len(triples),64):
            items=triples[j:j+64]
            raw=base_tokenizer_batch(rows,items,scales,array_cache=cache)
            _,ids=infer(model,raw,device)
            for u,(i,t,p) in enumerate(items):
                row=rows[i]
                for patch in np.flatnonzero(raw['mask'][u].all(-1)):
                    x=raw['x'][u,patch]
                    meta.append((row['dataset'],row['subject'],row['record'],row['parent'],t,row['positions'][p],patch,float(np.sqrt(np.mean(x*x)))))
                    waves.append(np.stack([raw[k][u,patch] for k in ('s','o','x')]))
                    codes.append(ids[u,patch])
        parts[split]=(pd.DataFrame(meta,columns=['dataset','subject','record','parent','start','position','patch','rms']),np.array(waves),np.array(codes))
    partner_figures(mod,stage,parts,dest)
    tr,_,tc=parts['train']
    dev,w,dc=parts['validation']
    led=[];summary=[]
    for j,b in enumerate('SO'):
        counts=tr.assign(word=tc[:,j]).groupby('word').agg(n=('word','size'),people=('subject','nunique'))
        eligible=counts[counts.people>=3].sort_values('n',kind='stable')
        for fraction,tag in [(.5,'median'),(.9,'common')]:
            if eligible.empty:
                summary.append(dict(branch=b,group=tag,status='no_eligible_train_words'));continue
            word=int(eligible.index[round((len(eligible)-1)*fraction)])
            pairs=[]
            for u in np.flatnonzero(dc[:,j]==word):
                row=dev.iloc[u]
                keep=(dev.dataset==row.dataset)&(dev.subject==row.subject)&(dev.position==row.position)&(dc[:,j]!=word)
                keep &= (dev.parent!=row.parent) | ((dev.start-row.start).abs()>=30)
                keep &= (dev.rms>=row.rms*.8)&(dev.rms<=row.rms*1.25)
                if keep.any():
                    ctrl=(dev.loc[keep,'rms']-row.rms).abs().idxmin()
                    pairs.append((u,ctrl))
                    led.append(dict(branch=b,group=tag,word=word,**row.to_dict(),control=dev.iloc[ctrl].to_dict()))
            if not pairs:
                summary.append(dict(branch=b,group=tag,word=word,status='no_matched_inner_members'));continue
            members,controls=np.array(pairs).T
            people=dev.iloc[members].drop_duplicates(['dataset','subject'])
            summary.append(dict(branch=b,group=tag,word=word,status='scored',pairs=len(pairs),people=len(people)))
            fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
            rate=200 if mod=='eeg' else 10
            t=np.arange(2*rate)/rate
            for a,(u,row) in zip(ax.flat[:3],people.iloc[:3].iterrows()):
                v=controls[np.flatnonzero(members==u)[0]]
                for idx,style,condition in [(u,'-','同词'),(v,':','对照')]:
                    for c,color in enumerate(['#18728d'] if channels==1 else ['#b44e20','#18728d']):
                        name='EEG' if channels==1 else ('HbO','HbR')[c]
                        a.plot(t,w[idx,j].reshape(-1,channels)[:,c],style,color=color,lw=1.2,label=name+condition)
                a.set(title=f'{row.dataset} / {row.subject} / {row.position}',xlabel='秒',ylabel='训练输入 SD')
                a.legend(fontsize=8,ncol=channels)
            for a in ax.flat[len(people.iloc[:3]):3]:a.set_axis_off()
            a=ax.flat[3]
            for idx,style,lab in [(members,'-','同词成员'),(controls,'--','匹配其他词')]:
                y=w[idx,j].reshape(len(idx),-1,channels)[:,:,0]
                if mod=='eeg':
                    hz,ps=periodogram(y,fs=rate,axis=1,window='hann')
                    sel=(hz>=1)&(hz<=45)
                    a.semilogy(hz[sel],np.maximum(np.median(ps[:,sel],axis=0),1e-12),style,label=lab)
                    a.set(xlabel='频率（Hz）',ylabel='PSD 中位数（SD²/Hz）')
                else:
                    clock=t-t.mean()
                    slope=y@clock/(clock@clock)
                    a.plot(np.sort(slope),np.arange(1,len(slope)+1)/len(slope),style,label=lab)
                    a.set(xlabel='HbO 斜率（SD/s）',ylabel='累计比例')
            a.legend()
            fig.suptitle(f'{stage} · {mod.upper()}-{b} ID {word} · {len(pairs)} 对 / {len(people)} 人\n'
                         'v3 条件目标；实线为真实成员、点线为同人同位置近似幅度对照；Hb 每对依次为 HbO、HbR')
            fig.savefig(dest/f'word_{b}_{tag}.png',dpi=210);plt.close(fig)
    pd.DataFrame(led).to_csv(dest/'matched_members.csv',index=False)
    write_json(dest/'word_support.json',summary)


def shift_review(rows,examples,scales,model,device,dest):
    """Two genuine 30 s inputs on the original G0 grid; align their 14 common blocks."""
    allowed={}
    candidates=[]
    for i,t,p in examples:
        if i not in allowed:
            allowed[i]={(g['start_s'],q) for g in rows[i]['inputs']['train'] for q in g['positions']}
        if (t+2,p) in allowed[i]:
            candidates.append((i,t,p))
    selected=[candidates[j] for j in anchors(rows,candidates)]
    visits=defaultdict(list); cache=OrderedDict()
    for idx in range(0,len(selected),64):
        triples=selected[idx:idx+64]
        left=base_tokenizer_batch(rows,triples,scales,array_cache=cache)
        right=base_tokenizer_batch(rows,[(i,t+2,p) for i,t,p in triples],scales,array_cache=cache)
        np.testing.assert_array_equal(left['x'][:,1:],right['x'][:,:-1])
        np.testing.assert_array_equal(left['mask'][:,1:],right['mask'][:,:-1])
        common=left['mask'][:,1:].all(-1)
        _,a=infer(model,left,device)
        _,b=infer(model,right,device)
        for u,(i,t,p) in enumerate(triples):
            row=rows[i]
            for region,loc in [('edge',[0,13]),('internal',list(range(1,13)))]:
                for pos in (v for v in loc if common[u,v]):
                    for j,branch in enumerate('SO'):
                        visits[row['dataset'],row['subject'],region,branch].append((a[u,pos+1,j],b[u,pos,j]))
    records=[]
    for (d,sub,region,branch),pairs in visits.items():
        a,b=np.asarray(pairs,dtype=int).T
        size=max(a.max(),b.max())+1
        chance=float(np.dot(np.bincount(a,minlength=size)/len(a),np.bincount(b,minlength=size)/len(b)))
        exact=float(np.mean(a==b))
        records.append(dict(dataset=d,subject=sub,region=region,branch=branch,patch_pairs=len(a),
                            id_match=exact,chance=chance,adjusted=(exact-chance)/(1-chance) if chance<1 else None))
    write_json(dest/'shift_support.json',dict(all_score_positions=len(examples),eligible_pairs=len(candidates),sampled_pairs=len(selected),
                                              people=len({(v[0],v[1]) for v in visits})))
    pd.DataFrame(records).to_csv(dest/'natural_shift.csv',index=False)


def terminal_review(mod,root,summaries):
    baseline=summaries[0]
    base=np.asarray(baseline['metrics']['nmse']['macro'])
    corr=np.asarray(baseline['metrics']['correlation']['macro'])
    decisions=[]
    fig,axes=plt.subplots(1,2,figsize=(12,5.4),layout='constrained')
    for arm,color in [('full','#16728d'),('decoder','#be602a')]:
        matches=[baseline]+[s for s in summaries if s['stage'].startswith(arm)]
        name='普通续训' if arm=='full' else '仅解码器'
        axes[0].plot([10000,11000,12000],[s['selection_score'] for s in matches],marker='o',color=color,label=name)
        end=matches[-1];value=np.asarray(end['metrics']['nmse']['macro'])
        corr_delta=np.asarray(end['metrics']['correlation']['macro'])-corr
        delta=value/base-1
        total=end['selection_score']/baseline['selection_score']-1
        trend=all(a['selection_score']>b['selection_score'] for a,b in zip(matches,matches[1:]))
        guard=bool(np.isfinite(corr_delta).all() and (delta<=.05).all() and (corr_delta>=-.01).all())
        decisions.append(dict(arm=arm,macro_relative_change=float(total),component_relative_change=delta.tolist(),
                              component_correlation_change=corr_delta.tolist(),monotonic_macro_improvement=trend,
                              waveform_guardrails_pass=guard,extension_waveform_criteria=(arm=='full' and trend and total<=-.02 and guard)))
        labels=[b+(' '+col if mod=='hb' else '') for b in 'XSO' for col in (['HbO','HbR'] if mod=='hb' else [''])]
        loc=np.arange(len(labels))+(-.17 if arm=='full' else .17)
        axes[1].bar(loc,delta.ravel()*100,width=.33,color=color,label=name)
        axes[1].set_xticks(np.arange(len(labels)),labels)
    axes[0].set(xlabel='训练更新数',ylabel='内部留人宏 NMSE',xticks=[10000,11000,12000])
    axes[1].set(ylabel='相对父模型的 NMSE 变化（%）')
    axes[1].axhline(0,color='black',lw=.8)
    for a in axes:a.legend()
    fig.suptitle(f'{mod.upper()} · 12 位内部留出被试 · 负变化表示误差降低')
    fig.savefig(root/'waveform_comparison.png',dpi=210);plt.close(fig)
    write_json(root/'waveform_decision.json',dict(scope='waveform criteria only; joint promotion requires actual-word and visual review',
               comparisons=decisions,decoder_minus_full_nmse=summaries[-1]['selection_score']-summaries[2]['selection_score']))


@torch.inference_mode()
def score(mod,device):
    meta,scales,_=load(mod)
    rows=meta['rows']
    cfg,task=settings(mod)
    examples=base_tokenizer_examples(rows,mod,'validation')
    groups={}
    for split in ['train','validation']:
        all_ex=base_tokenizer_examples(rows,mod,split)
        groups[split]=[all_ex[i] for i in anchors(rows,all_ex)]
    root=OUT/mod/'review'
    root.mkdir(exist_ok=True)
    picked=root/'cases.json'
    for stage in STAGES:
        dest=root/stage
        dest.mkdir(exist_ok=True)
        if (dest/'summary.json').is_file():continue
        model=_base_training_model(cfg,task,single_id_spectrum=mod=='eeg').to(device).eval()
        saved=torch.load(checkpoint(mod,stage,task),map_location='cpu',weights_only=False)
        assert saved['step']==int(stage.split('_')[1])
        model.load_state_dict(saved['model_state']);del saved
        cache=OrderedDict();metrics=defaultdict(list);all_ids=[]
        channels=1 if mod=='eeg' else 2
        for i in range(0,len(examples),64):
            ex=examples[i:i+64]
            raw=base_tokenizer_batch(rows,ex,scales,array_cache=cache)
            pred,ids=infer(model,raw,device)
            all_ids.append(ids.astype(np.uint16))
            vals=base_training_metrics(pred,raw,channels)
            for k in ['nmse','correlation','rms_gain','boundary_jump_nmse','shape_nmse','level_nmse']:
                metrics[k].append(vals[k])
            diffs=[]
            for k,key in [('sum','x'),('S','s'),('O','o')]:
                error=(pred[k]-raw[key]).reshape(len(ex),-1,channels)
                mask=raw['mask'].reshape(error.shape);valid=mask[:,1:]&mask[:,:-1]
                mse=np.where(valid,np.diff(error,axis=1)**2,0).sum(1)/np.maximum(1,valid.sum(1))
                diffs.append(mse/raw['dv'+key][:,0,:channels])
            metrics['diff_nmse'].append(np.stack(diffs,axis=1))
        ids=np.concatenate(all_ids)
        ref=root/'reference_ids.npz'
        if stage.startswith('parent'):
            np.savez_compressed(ref,ids=ids,examples=np.array(examples))
        with np.load(ref) as z:
            np.testing.assert_array_equal(z['examples'],examples)
            match=(z['ids']==ids).mean(axis=(0,1))
        if stage.startswith('decoder'):
            np.testing.assert_equal(match,np.ones(2))
        metrics={k:np.concatenate(v) for k,v in metrics.items()}
        score=base_training_aggregate(metrics,examples,rows)
        assert score['subjects']==12 and np.isfinite(score['selection_score'])
        if stage.startswith('parent'):
            errors=metrics['nmse'].mean((1,2))
            cases={};used=set()
            for name,q in [('较好',.1),('居中',.5),('困难',.9)]:
                for ix in np.argsort(abs(errors-np.quantile(errors,q)),kind='stable'):
                    triple=examples[ix];r=rows[triple[0]];person=(r['dataset'],r['subject'])
                    if person not in used:
                        cases[name]=triple;used.add(person);break
            write_json(picked,cases)
        wavefig(mod,stage,read_json(picked),model,rows,scales,device,dest)
        patternfig(mod,stage,rows,scales,model,groups,device,dest)
        shift_review(rows,examples,scales,model,device,dest)
        write_json(dest/'summary.json',dict(status='completed',stage=stage,integer_replay='exact',
                   same_input_id_agreement_to_parent=match.tolist(),**score))
    summaries=[read_json(root/s/'summary.json') for s in STAGES]
    terminal_review(mod,root,summaries)
    write_json(root/'comparison.json',[dict(stage=s['stage'],macro_nmse=s['selection_score'],
               components=s['metrics']['nmse']['macro'],people=s['subjects'],reference_id_agreement=s['same_input_id_agreement_to_parent']) for s in summaries])



@torch.inference_mode()
def preview(mod, minimum_step):
    # Read one atomically saved checkpoint. No writes to the training tree.
    cfg,task=settings(mod)
    model_file=checkpoint(mod,'parent_10000',task)
    folder=OUT/mod/('first_hard_review' if minimum_step==1250 else f'visual_review_{minimum_step}')
    folder.mkdir(exist_ok=True)
    if (folder/'summary.json').exists():
        return
    while True:
        if model_file.is_file():
            state=torch.load(model_file,map_location='cpu',weights_only=False)
            if state['step']>=minimum_step:
                break
        import subprocess
        if subprocess.run(['systemctl','--user','is-active','--quiet',f'aligned-opt-1126-{mod}.service']).returncode:
            raise RuntimeError('training producer stopped before preview')
        time.sleep(5)
    model=_base_training_model(cfg,task,single_id_spectrum=mod=='eeg').eval()
    model.load_state_dict(state['model_state'])
    from run_dual_component_tokenizer import _torch_save
    _torch_save(folder/'model.pt',dict(model_state=state['model_state'],step=state['step'],signature=state['signature']))
    stage=f"硬量化图审 · 第 {state['step']} 步"
    meta,scales,_=load(mod)
    rows=meta['rows'];groups={}
    for split in ['train','validation']:
        examples=base_tokenizer_examples(rows,mod,split)
        groups[split]=[examples[i] for i in anchors(rows,examples)]
    # A reproducible case chosen by subject/crop independently of loss and model IDs.
    fixed=groups['validation'][len(groups['validation'])//2]
    wavefig(mod,stage,{'固定留人窗':fixed},model,rows,scales,'cpu',folder)
    patternfig(mod,stage,rows,scales,model,groups,'cpu',folder)
    write_json(folder/'summary.json',dict(stage=stage,step=state['step'],source=str(model_file),
               retained_state='model.pt',status='completed',scope='early visual inspection only',integer_replay='exact',
               fixed_case=fixed,fit_subjects=len(meta['split']['fit']),inner_subjects=len(meta['split']['inner']),
               sampled_fit_positions=len(groups['train']),sampled_inner_positions=len(groups['validation'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mod',choices=['eeg','hb'])
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--preview-step',type=int,choices=(1250,4000),default=1250)
    args=parser.parse_args()
    torch.set_num_threads(2 if args.preview else 1)
    if args.preview:
        with owner_lock(OUT/args.mod, f'preview_{args.preview_step}'): preview(args.mod,args.preview_step)
    else:
        with owner_lock(OUT/args.mod/'review'): score(args.mod,'cuda:0')
