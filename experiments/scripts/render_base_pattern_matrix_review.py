"""Render the dated A/B review from retained development scores and frozen galleries."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xlsxwriter

from render_base_diagnostic_review import DATASETS, LABELS, read

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT/'experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer'
A = RUNS/'20261120_pattern_a_v1'
B = RUNS/'20261120_input_quantization_b_v1'
MODELS = ['G1_eeg', 'G1_hb', 'G4_hb']
CELLS = ['X_continuous', 'X_K128', 'target_continuous', 'target_K128']
NAMES = ['X 连续', 'X 硬码', '目标连续', '目标硬码']
COLORS = ['#31668c','#b25825','#278773','#775389']
STYLES = ['-', '--', '-.', ':']
BANDS = ['1_4','4_8','8_13','13_30','30_45']
CONTRASTS = np.array([[1,0,-1,0],[-1,1,0,0],[0,0,-1,1],[-1,1,1,-1]])
CONTRAST_NAMES = ['连续输入差','X 量化差','目标量化差','交互差']
plt.rcParams.update({'font.family':['Noto Sans CJK JP','DejaVu Sans'],'font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})


def subjects(metric):
    return {(v['dataset'],v['subject']):np.array(v['value']) for v in metric['by_subject']}


def paired_interval(keys, values):
    """Stratified subject bootstrap, common draws across matched cells and endpoints."""
    rng = np.random.default_rng(17)
    boot = []
    for d in DATASETS:
        group = values[[i for i,key in enumerate(keys) if key[0]==d]]
        index = rng.integers(len(group), size=(10000,len(group)))
        boot.append(group[index].mean(1))
    return np.quantile(np.mean(boot,axis=0),[.025,.975],axis=0)


def collect():
    from src.analysis.dual_component_tokenizer import base_training_aggregate
    suite = read(B/'summary.json')
    plan = read(B/'plan.json')
    cfg = read(B/'resolved_config.json')['config']
    assert read(B/'controller.json')['status']==suite['status']=='completed'
    assert suite['planned_models']==suite['completed_models']==4
    result = read(A/'summary.json')
    assert read(A/'controller.json')['status']==result['status']=='completed'
    assert sorted(result['completed'])==sorted(MODELS) and not result['failures']
    rows = read(ROOT/cfg['data']['run']/'cache/manifest.json')['rows']
    jobs = {k:ROOT/v for k,v in plan['matrix_controls'].items()}
    for name,task in zip(cfg['matrix']['new_cells'],cfg['tasks']):
        jobs[name]=B/'models'/'__'.join(task[k] for k in ('modality','architecture','candidate'))
    cells,trajectory,health,panels,deltas = {},[],[],[],[]
    for modality in ['EEG','Hb']:
        matched={}
        for cell in CELLS:
            name=modality+'_'+cell
            job=jobs[name]
            record=read(job/'summary.json')
            assert record['status']=='completed' and record['completed_step']==10000
            assert record['signature']['input']==plan['manifest_identity'] and record['signature']['scales']==plan['scales_identity']
            for step in [4000,7000,10000]:
                folder=job/'evaluation'/f'step_{step:05d}'
                assert (job/f'checkpoint_step_{step:05d}.pt').stat().st_size
                assert read(folder/'summary.json')['status']=='completed'
                for role in ['train','validation','regression']:
                    value=read(folder/(role+'.json'))
                    trajectory.append([name,step,role,value['inputs'],value['subjects'],value['selection_score']])
                    np.testing.assert_allclose(value['selection_score'],record['milestones'][str(step)][role]['selection_score'])
                    if step==10000:
                        with np.load(folder/(role+'.npz')) as f:
                            ids=f['examples']
                            if role not in matched:
                                matched[role]=ids
                            np.testing.assert_array_equal(matched[role],ids)
                            if role=='validation':
                                recomputed=base_training_aggregate({'nmse':f['nmse']},ids,rows)
                                np.testing.assert_allclose(recomputed['selection_score'],value['selection_score'])
                                for k,v in subjects(recomputed['metrics']['nmse']).items():
                                    np.testing.assert_allclose(v,subjects(value['metrics']['nmse'])[k])
                        if role=='validation':
                            cells[name]=value
                            m=value['metrics']
                            np.testing.assert_allclose(suite['matrix'][name]['macro_nmse'],value['selection_score'])
                            np.testing.assert_allclose(np.array(m['level_nmse']['macro'])+m['shape_nmse']['macro'],m['nmse']['macro'])
                            panels.append([name,value['subjects'],value['records'],value['windows'],value['inputs']])
                            for h in value.get('code_health',[]):
                                if 'effective' in h:
                                    health.append([name,h['branch'],h['dataset'],h['active'],h['effective'],h['top1']])
        by_sub=[subjects(cells[modality+'_'+c]['metrics']['nmse']) for c in CELLS]
        keys=sorted(by_sub[0])
        assert all(sorted(v)==keys for v in by_sub) and len(keys)==20
        x=np.array([[np.mean(v[k]) for v in by_sub] for k in keys])
        diffs=x@CONTRASTS.T
        lo,hi=paired_interval(keys,diffs)
        macro=np.array([cells[modality+'_'+c]['selection_score'] for c in CELLS])
        for j,label in enumerate(CONTRAST_NAMES):
            deltas.append([modality,label,*(float(v) for v in [(CONTRASTS@macro)[j],lo[j],hi[j]]),
                           int((diffs[:,j]>0).sum()),int((diffs[:,j]<0).sum()),len(keys)])
    audits={n:read(A/n/'summary.json') for n in MODELS}
    atlases={}
    for n,s in audits.items():
        assert s['status']=='completed'
        for branch in 'SO':
            at=read(A/n/(branch+'_atlas.json'))
            atlases[n,branch]=at
            np.testing.assert_array_equal(np.array(at['by_dataset']['patches']).sum(0),at['support']['validation']['patches'])
            assert sum(at['support']['validation']['patches'])==s['coverage']['validation']['complete_patches']
            assert at['renumbering_audit']['total_shape_cost']==0
            with np.load(A/n/(branch+'_natural_context.npz')) as f:
                switch=f['original_ID']!=f['shifted_ID']
                ctx=s['branches'][branch]['natural_context']
                assert switch.sum()==ctx['changed_supported_pairs'] and len(switch)==s['natural_context']['complete_central_patches']
                np.testing.assert_allclose(f['attribute_squared_change'][switch].mean(0),ctx['mean_squared_attribute_change_given_switch_pooled'])
    return jobs,rows,cells,trajectory,health,panels,deltas,audits,atlases


def book(out,cells,trajectory,health,panels,deltas,audits,atlases):
    wb=xlsxwriter.Workbook(out/'AB_tables.xlsx',{'constant_memory':True})
    header=wb.add_format({'bold':True,'font_color':'white','bg_color':'#215674'})
    def sheet(name,columns,records):
        sh=wb.add_worksheet(name)
        sh.write_row(0,0,columns,header)
        last=0
        for last,row in enumerate(records,1):
            sh.write_row(last,0,[None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v for v in row])
        sh.freeze_panes(1,1)
        sh.autofilter(0,0,last,len(columns)-1)
        for j,c in enumerate(columns):sh.set_column(j,j,max(16,min(42,len(c)+9)))
    sheet('索引',['表','范围和单位'],[
        ['B_重建','固定 10k 验证；NMSE 为冻结训练方差归一化误差；band_* 是 log10 功率比，乘 10 得 dB。'],
        ['B_差值','95% 配对百分位区间，固定四库，库内重采样被试 10,000 次。'],
        ['A_读出','真实属性与预测属性之间的 MSE；同一行四个读出可比较。'],
        ['A_码字','每行一个分支与码字；低支持条件为训练被试 <3 或完整片段 <20。'],
        ['A_逐码逐库','完整验证片段的逐码条件方差、冻结训练 ID 均值的验证 MSE。空白表示无支持。'],
        ['A_换窗','宏平均及逐库/逐被试值按层级汇总；属性列为两个 ID 的训练属性均值之差的平方。'],
        ['A_换码条件','切换且两个 ID 有训练支持的片段池化平均；列为各属性的均方变化。']])
    def level(metric):
        yield 'macro','',np.asarray(metric['macro'])
        for d,v in metric['by_dataset'].items():yield d,'',np.asarray(v)
        for (d,s),v in subjects(metric).items():yield d,s,v
    def brecs():
        for name,v in cells.items():
            for metric,values in v['metrics'].items():
                for d,s,arr in level(values):
                    for (i,c),z in np.ndenumerate(arr):
                        yield [name,d,s,metric,'XSO'[i],('EEG' if name.startswith('EEG') else ['HbO','HbR'][c]),z]
    sheet('B_重建',['配置','库','被试','指标','分量','通道','值'],brecs())
    sheet('B_面板',['配置','被试','记录','窗口','位置输入'],panels)
    sheet('B_轨迹',['配置','步骤','面板','位置输入','被试','宏NMSE'],trajectory)
    sheet('B_差值',['模态','差值','估计','95%下限','95%上限','正差被试','负差被试','配对数'],deltas)
    sheet('B_码本',['配置','分支','库','活动码数','有效码数 exp(H)','最高码频率'],health)
    readouts=[];context=[];switches=[];codes=[];code_data=[];coverage=[]
    for name,v in audits.items():
        attrs=v['attributes']
        for role,r in v['coverage'].items():coverage.append([name,role,*[r[k] for k in ['inputs','planned_patches','complete_patches','excluded_partial_or_missing']]])
        for br,h in v['branches'].items():
            for d,s,arr in level(h['attribute_mse']):
                for j,attr in enumerate(attrs):
                    readouts.append([name,br,d,s,attr,*arr[:,j],1-arr[2,j]/arr[0,j],1-arr[3,j]/arr[1,j]])
            ctx=h['natural_context']
            for d,s,arr in level(ctx):
                context.extend([name,br,d,s,col,z] for col,z in zip(ctx['column_order'],arr))
            switches.extend([name,br,ctx['changed_supported_pairs'],attr,z] for attr,z in zip(attrs,ctx['mean_squared_attribute_change_given_switch_pooled']))
            at=atlases[name,br]; by=at['by_dataset']; sup=at['support']
            for i in range(128):
                codes.append([name,br,i,*[sup[r][c][i] for r in ['train','validation'] for c in ['patches','subjects','records']],at['low_training_support'][i]])
                for di,d in enumerate(by['dataset_order']):
                    for j,attr in enumerate(attrs):
                        code_data.append([name,br,i,d,attr,by['patches'][di][i],by['conditional_variance'][di][i][j],by['frozen_ID_mse'][di][i][j]])
    sheet('A_覆盖',['模型','面板','位置输入','计划片段','完整片段','不完整或缺失'],coverage)
    sheet('A_读出',['模型','分支','库','被试','属性','背景','单ID','背景加ID','完整序列','背景上ID增益','序列相对单ID增益'],readouts)
    sheet('A_码字',['模型','分支','码字','训练片段','训练被试','训练记录','验证片段','验证被试','验证记录','低训练支持'],codes)
    sheet('A_逐码逐库',['模型','分支','码字','库','属性','片段','条件方差','冻结ID验证MSE'],code_data)
    sheet('A_换窗',['模型','分支','库','被试','指标','值'],context)
    sheet('A_换码条件',['模型','分支','切换片段','属性','条件均方变化'],switches)
    wb.close()


def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=235,facecolor='white',bbox_inches='tight',pad_inches=.10)
    plt.close(fig)


def heat(ax,data,x,y,title,fmt='.2f',cmap='Blues',lo=0,hi=1):
    arr=np.asarray(data)
    im=ax.imshow(arr,cmap=cmap,vmin=lo,vmax=hi,aspect='auto')
    ax.set(xticks=range(len(x)),xticklabels=x,yticks=range(len(y)),yticklabels=y,title=title)
    ax.tick_params(length=0)
    for (i,j),v in np.ndenumerate(arr):
        rgba=im.cmap(im.norm(v))
        color='white' if np.dot(rgba[:3],[.299,.587,.114])<.49 else '#132d3c'
        ax.text(j,i,format(v,fmt),ha='center',va='center',color=color,fontsize=9)
    return im


def figures(out,jobs,rows,cells,trajectory,deltas,audits,atlases):
    for mod in ['EEG','Hb']:
        models=[cells[mod+'_'+c] for c in CELLS]
        fig,axes=plt.subplots(2,1,figsize=(8.5,5.8),layout='constrained')
        data=[[np.mean(m['metrics']['nmse']['by_dataset'][d]) for m in models] for d in DATASETS]
        im=heat(axes[0],data,NAMES,LABELS,'各库验证宏 NMSE，越低越好','.3f',hi=max(np.max(data),.001))
        fig.colorbar(im,ax=axes[0],label='NMSE')
        comp=['X','S','O'] if mod=='EEG' else [t+'·'+c for t in 'XSO' for c in ['HbO','HbR']]
        data=np.stack([np.reshape(m['metrics']['nmse']['macro'],-1) for m in models],axis=1)
        im=heat(axes[1],data,NAMES,comp,'各分量跨库宏 NMSE，越低越好','.3f',hi=max(np.max(data),.001))
        fig.colorbar(im,ax=axes[1],label='NMSE')
        save(fig,out,mod+'_matrix')
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    for ax,mod in zip(axes,['EEG','Hb']):
        for c,n,col,marker in zip(CELLS,NAMES,COLORS,'os^D'):
            vals=[(step,v) for name,step,r,_,_,v in trajectory if name==mod+'_'+c and r=='validation']
            ax.plot(*zip(*vals),color=col,marker=marker,label=n)
        ax.set(title=mod+' 验证宏 NMSE',xticks=[4000,7000,10000],xlabel='累计更新步数',ylabel='宏 NMSE')
        ax.legend(fontsize=8)
    save(fig,out,'trajectory')
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    for ax,mod in zip(axes,['EEG','Hb']):
        ax.axvline(0,c='.4',lw=1)
        for j,(_,label,v,lo,hi,*_) in enumerate(r for r in deltas if r[0]==mod):
            ax.errorbar(v,j,xerr=[[v-lo],[hi-v]],fmt='o',capsize=4,color=COLORS[j])
        ax.set(yticks=range(4),yticklabels=CONTRAST_NAMES,title=mod,xlabel='宏 NMSE 差，点及 95% 配对区间')
        ax.invert_yaxis()
    save(fig,out,'contrasts')
    fig,axes=plt.subplots(2,2,figsize=(8.7,5.5),layout='constrained',sharey=True)
    for ax,d,label in zip(axes.ravel(),DATASETS,LABELS):
        for c,n,col,style,marker in zip(CELLS,NAMES,COLORS,STYLES,'os^D'):
            m=cells['EEG_'+c]['metrics']
            y=[10*m['band_'+b+'_log10_power_ratio']['by_dataset'][d][0][0] for b in BANDS]
            ax.plot(range(5),y,ls=style,marker=marker,c=col,label=n)
        ax.axhline(0,c='.4',lw=.7)
        ax.set(xticks=range(5),xticklabels=[b.replace('_','–') for b in BANDS],ylabel='预测 / 目标功率（dB）',xlabel='频带（Hz）',title=label)
    axes[0,0].legend(fontsize=8,ncol=2)
    save(fig,out,'spectrum')
    example_notes=[]
    for mod in ['EEG','Hb']:
        fig,axes=plt.subplots(3,1,figsize=(9,5.9),layout='constrained',sharex=True)
        saved=None; rmse=[]
        for j,c in enumerate(CELLS):
            folder=jobs[mod+'_'+c]/'evaluation/step_10000'
            meta=read(folder/'validation_examples.json')
            ref=[i for i,v in enumerate(meta) if rows[v['row']]['dataset']=='refed']
            idx=ref[len(ref)//2]; entry=meta[idx]; row=rows[entry['row']]
            width,colors=(400,1) if mod=='EEG' else (20,2)
            with np.load(folder/'validation_examples.npz') as f:
                def wave(k):
                    v=f[k][idx].reshape(15,width,colors)
                    return (v[7] if mod=='EEG' else v.reshape(-1,colors))[:,0]
                mask=wave('mask').astype(bool)
                target=np.stack([np.where(mask,wave(k),np.nan) for k in 'xso'])
                pred=np.stack([np.where(mask,wave(k),np.nan) for k in ['prediction_sum','prediction_S','prediction_O']])
            if j==0:saved=(entry,target)
            assert saved[0]==entry
            np.testing.assert_allclose(saved[1],target,equal_nan=True)
            rmse.append(np.sqrt(np.nanmean((pred-target)**2,axis=1)))
            t=np.arange(len(mask))/(200 if mod=='EEG' else 10)+(14 if mod=='EEG' else 0)
            for i,ax in enumerate(axes):
                if j==0:ax.plot(t,target[i],color='#222222',lw=1.55,label='目标')
                ax.plot(t,pred[i],c=COLORS[j],ls=STYLES[j],lw=1.05,label=NAMES[j])
                ax.set(ylabel='XSO'[i]+' / 训练 X SD')
        axes[0].legend(fontsize=8,ncol=5)
        axes[-1].set_xlabel('相对 30 秒窗口起点（秒）')
        save(fig,out,mod+'_example')
        example_notes.append(dict(modality=mod,dataset=row['dataset'],subject=row['subject'],record=row['record'],
                                  parent=row['parent'],position=row['positions'][entry['position']],start_s=entry['start_s'],
                                  rmse=np.asarray(rmse).tolist()))
    eeg=['原电压','去均值','1–4','4–8','8–13','13–30','30–45','包络']
    hb=['HbO 波形','水平','斜率','曲率','HbR 波形','水平','斜率','曲率']
    for mod,names,xlabels in [('EEG',MODELS[:1],eeg),('Hb',MODELS[1:],hb)]:
        fig,axes=plt.subplots(2,1,figsize=(9,4.5),layout='constrained')
        labels=[n.split('_')[0]+' '+br for n in names for br in 'SO']
        metrics=[np.array(audits[n]['branches'][br]['attribute_mse']['macro']) for n in names for br in 'SO']
        for ax,(a,b),title in zip(axes,[(2,0),(3,1)],['背景加入当前 ID 后的误差降低（%）','完整解码相对单 ID 均值的误差降低（%）']):
            im=heat(ax,[100*(1-m[a]/m[b]) for m in metrics],xlabels,labels,title,'.1f','RdBu',-100,100)
            fig.colorbar(im,ax=ax,label='增益（%）')
        save(fig,out,mod+'_readout')
    fig,axes=plt.subplots(2,3,figsize=(9,5.2),layout='constrained')
    for j,name in enumerate(MODELS):
        for i,br in enumerate('SO'):
            ax=axes[i,j];at=atlases[name,br];sup=at['support']['validation']
            ax.scatter(sup['patches'],sup['subjects'],s=12,alpha=.5,color=COLORS[j])
            ax.set(xscale='log',xlim=(8,120000),ylim=(0,21),title=name.replace('_',' ')+' '+br,
                   xlabel='验证完整片段数（对数）',ylabel='验证被试数')
    save(fig,out,'support')
    fig,axes=plt.subplots(2,3,figsize=(9.2,5.1),layout='constrained')
    gallery=[]
    for j,name in enumerate(MODELS):
        with np.load(audits[name]['d0_gallery']) as f:
            for i,br in enumerate('SO'):
                at=atlases[name,br]
                code=int(np.argmax(at['support']['validation']['patches']))
                waves=f[f'{br}_{code}_waves'][:,:,0]
                waves=waves-waves.mean(1,keepdims=True)
                lo,median,hi=np.quantile(waves,[.1,.5,.9],axis=0)
                t=np.arange(waves.shape[1])/(200 if j==0 else 10)
                ax=axes[i,j]
                ax.fill_between(t,lo,hi,color=COLORS[j],alpha=.23)
                for w in waves[:3]:ax.plot(t,w,c='.65',lw=.6)
                ax.plot(t,median,c=COLORS[j],lw=1.4)
                ax.set(title=f'{name.replace("_"," ")} {br} · ID {code}',xlabel='片段时间（秒）',ylabel='去均值 / 训练 X SD')
                gallery.append([name,br,code,len(waves),at['support']['validation']['patches'][code]])
    save(fig,out,'gallery')
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    for i,br in enumerate('SO'):
        for j,n in enumerate(MODELS):
            ctx=audits[n]['branches'][br]['natural_context']
            x=j+(i-.5)*.26
            axes[0].bar(x,ctx['macro'][0]*100,.25,color=COLORS[i],hatch='//' if i else None,label=br if j==0 else None)
            rms=np.sqrt(ctx['mean_squared_attribute_change_given_switch_pooled'][0])
            axes[1].bar(x,rms,.25,color=COLORS[i],hatch='//' if i else None,label=br if j==0 else None)
    for ax,title,ylabel in zip(axes,['全部可配对中央片段','只取变码且有码字支持的片段'],['换码率（%，分层宏平均）','原型波形 RMS 变化 / 训练 X SD']):
        ax.set(xticks=range(3),xticklabels=['G1 EEG','G1 Hb','G4 Hb'],title=title,ylabel=ylabel);ax.legend()
    save(fig,out,'context')
    return example_notes,gallery


def transfer_figure(out,atlases):
    fig,axes=plt.subplots(2,3,figsize=(9,5.1),layout='constrained',sharex=True)
    for j,name in enumerate(MODELS):
        for i,br in enumerate('SO'):
            ax=axes[i,j];at=atlases[name,br]['by_dataset']
            groups=[]
            for n in range(4):
                y=np.array(at['frozen_ID_mse'][n],dtype=float)[:,0]
                groups.append(np.sqrt(y[np.isfinite(y)]))
            ax.boxplot(groups,showfliers=True,flierprops=dict(markersize=2,alpha=.45),widths=.55)
            ax.set(yscale='log',xticks=[1,2,3,4],xticklabels=['ST','REF','SIM','VIS'],
                   title=name.replace('_',' ')+' '+br,ylabel='逐码原型 RMSE')
    save(fig,out,'transfer')


def historical_eeg(jobs):
    old=RUNS/'20261117_base_g1_v2/models/eeg__waveform__vq128/evaluation/step_10000'
    new=jobs['EEG_X_K128']/'evaluation/step_10000'
    with np.load(old/'validation.npz') as x,np.load(new/'validation.npz') as y:
        np.testing.assert_array_equal(x['examples'],y['examples'])
    left,right=[read(p/'validation.json') for p in (old,new)]
    x,y=[subjects(v['metrics']['nmse']) for v in (left,right)]
    keys=sorted(x)
    assert keys==sorted(y)
    diffs=np.array([np.mean(y[k]-x[k]) for k in keys])
    return dict(old=left['selection_score'],new=right['selection_score'],lower_subjects=int((diffs<0).sum()),
                interval=paired_interval(keys,diffs).tolist(),by_dataset={d:float(np.mean(right['metrics']['nmse']['by_dataset'][d])-
                np.mean(left['metrics']['nmse']['by_dataset'][d])) for d in DATASETS})


def tex_table(cols, rows):
    line=lambda items:' & '.join(str(v) for v in items)+r' \\'
    return '\n'.join([r'\begin{center}\small\begin{tabular}{l'+'r'*(len(cols)-1)+'}',r'\toprule',line(cols),r'\midrule',
                      *[line(row) for row in rows],r'\bottomrule\end{tabular}\end{center}'])


def report(out,values,cells):
    lines=[r'''\documentclass[11pt,a4paper,fontset=none]{ctexart}
\usepackage[margin=18mm,top=18mm,bottom=18mm]{geometry}
\usepackage{fontspec,graphicx,booktabs,xcolor}
\setCJKmainfont{Noto Sans CJK SC}
\setmainfont{DejaVu Sans}
\definecolor{ink}{HTML}{183346}\definecolor{teal}{HTML}{087465}
\setlength{\parindent}{0pt}\setlength{\parskip}{5.5pt}\raggedbottom
\newcommand{\kicker}[1]{{\small\color{teal}\bfseries #1}\par}
\newcommand{\titleline}[1]{{\fontsize{20}{27}\selectfont\bfseries\color{ink}#1}\par\vspace{2mm}}
\newcommand{\topic}[1]{{\bfseries\color{teal}#1}\quad}
\newcommand{\chart}[2]{\begin{center}\includegraphics[width=\linewidth,height=#2\textheight,keepaspectratio]{#1.png}\end{center}}
\newcommand{\src}[1]{\vfill{\footnotesize\color{gray}证据入口\quad #1}\par}
\begin{document}\color{ink}''']
    def page(label,title,*parts):
        if len(lines)>1:lines.append(r'\newpage')
        lines.extend([r'\kicker{'+label+'}',r'\titleline{'+title+'}',*parts])
    def pic(name,height='.43'):return r'\chart{'+name+'}{'+height+'}'
    def src(s):return r'\src{'+s+'}'
    matrix=tex_table(['开发验证宏 NMSE','X 连续','X 硬码','目标连续','目标硬码'],
             [[m,*[f"{cells[m+'_'+c]['selection_score']:.5f}" for c in CELLS]] for m in ['EEG','Hb']])
    page('实验决策报告　／　2026 年 11 月 20 日','局部码字能解释什么，重建误差又来自哪里？',
         '阶段 A 冻结三个已有硬码模型，检验单个两秒码字能解释的局部属性。阶段 B 在相同训练预算下改变编码器输入与量化条件。两组结果分别回答模式含义与重建瓶颈，评价对象均为开发验证被试。',
         matrix,
         r'\topic{EEG}连续模型直接读取目标后，宏误差从 0.22957 降到 0.20180，但 REFED 总电压 30–45 Hz 功率仍比目标低 41.86 dB。两种输入下硬量化又分别增加 0.15737 和 0.17127 的宏误差。频带保真与量化适配均有缺口。',
         r'\topic{Hb}目标输入连续模型的宏误差为 0.00703，远低于 X 输入的 0.15498。G4 的局部 ID 对双色团斜率更有区分力，同时重建和曲率读出仍有取舍。',
         r'\topic{下一步}建议立即开展一项 EEG 连续模型的频带平衡对照。首轮保留 E-TC 的目标输入、结构、训练抽样和 10,000 步预算，只改变失真权重。Hb 的训练扩展和 K512 码本待这次判别后再决定。',
         r'\topic{阅读顺序}第 2–8 页给出设计、成对差值、频谱、波形和收敛轨迹。第 9–13 页给出单码属性、真实同码样本、跨库误差及换窗结果。第 14 页列明实验选择与判定条件。',
         src(r'阶段 A 三个模型终态、阶段 B 八个匹配单元的固定终点。附表 AB\_tables.xlsx 提供各库、被试、码字的数值。'))
    page('设计与核验','两组实验共享开发人口，各有独立验收指标',
         r'\topic{目标与接口}观测信号 X 分解为既有生理模型（teacher）估计的条件分量 S 与剩余量 O = X − S。EEG 使用 200 Hz 电压，Hb 使用 10 Hz 的 HbO、HbR。每个分支在完整 30 秒上下文中每两秒输出一个位置，共 15 个。K128 每分支每位置一个整数，名义码率 3.5 bit/s；解码器只接收本分支整数序列。完整序列利用了双向上下文。G1 为普通波形解码器；G4 的 Hb 解码器用当前码决定局部形状，限制序列上下文的形状修正。',
         r'\topic{输入与预热}实际应用路径从 X 预测 S、O，两分支相加得到 X 的重建。特权路径给各分支其自身目标，用来诊断有限预算的压缩误差。EEG 均使用八个片内位置的 E1 结构，Hb 使用 G1 waveform。E-Q 与 H-TQ 复用 E1、D1 的预热，E-TC 与 E-TQ 从同一个新预热分叉。每组 1,000 步预热加 9,000 步延续，seed 17，位置选择和训练尺度固定。',
         tex_table(['验证人口','Single-Trial','REFED','Simultaneous','Visual','合计'],[['被试',6,6,5,3,20]]),
         r'\topic{分母}每模态 144 条记录。EEG 有 1,920 个窗口、65,909 个窗口位置输入，Hb 有 1,923 个窗口、70,972 个输入。G0 已保留六个 EEG 上游失败父窗口。阶段 A 按完整两秒片段计分，EEG 为 955,640 / 988,635，Hb 为 1,034,497 / 1,064,580；阶段 B 在各窗口有效时间点上计分。',
         r'\topic{误差}各 X、S、O 分量和各色团分别以有效样本平方误差均值除以同库同位置冻结训练目标方差。由位置到窗口、记录、被试、数据集逐层等权，再对分量和色团等权得宏 NMSE。值越低越好。水平误差项采用每个两秒片段的平均残差平方，形状误差项是去掉该均值后的残差平方，两项相加等于 NMSE。',
         r'\topic{核验}阶段 A 三组审计均完成；阶段 B 四个新增模型均完成，各有 4k、7k、10k 检查点。与四个历史控制组逐模态核对训练评分、验证和训练人口内回归面板的最终输入身份，并从逐输入 NMSE 重新汇总验证被试值与宏值。固定 10k 为主终点。',
         src(r'G0 人口和尺度合同、A/B resolved\_config、plan、各模型 summary 及 evaluation/step\_10000。'))
    diff_rows=[[m,l,f'{d:+.5f}',f'[{lo:+.5f}, {hi:+.5f}]',f'{pos}/{neg}'] for m,l,d,lo,hi,pos,neg,_ in values['deltas']]
    page('阶段 B　／　配对差值','两种模态对输入与量化的敏感程度不同',
         pic('contrasts','.28'),
         tex_table(['模态','比较','差值',r'95\% 配对区间','正差 / 负差被试'],diff_rows),
         r'连续输入差取 X 连续减目标连续；另两项在相同输入下取硬码减连续。交互差取 X 量化差减目标量化差。正值表示相减的第一项误差更大。EEG 的 X 量化差约为连续输入差的 5.7 倍，Hb 的连续输入差约为 X 量化差的 4.9 倍。',
         r'固定训练种子、模型与四个数据库，库内按被试有放回配对重采样 10,000 次，每次四库等权，取 2.5\% 与 97.5\% 分位数。区间描述这组冻结模型在开发人口上的不确定性，未作多重比较校正。两模态交互均为负，数据没有显示量化对 X 输入施加额外的更大惩罚。',
         src(r'八个 10k 验证输出的逐被试 NMSE；附表 B\_差值。'))
    page('阶段 B　／　EEG','EEG 的输入改善主要落在 S，硬码影响四库',
         pic('EEG_matrix','.49'),
         r'每格均为验证 NMSE，颜色越深误差越大。上图每格汇总 X、S、O，下图每格对四库等权；两幅图色标分别标出。输入改善最大的是连续 S，从 0.3989 降至 0.3180。连续总电压 X 仅由 0.1416 变为 0.1414，说明宏误差改善不能直接读作总电压波形恢复。',
         r'在 X 输入下，连续模型的宏水平 / 形状 NMSE 为 0.0082 / 0.2214，硬码为 0.0219 / 0.3651。目标输入下两项分别为 0.0001 / 0.2017 和 0.0133 / 0.3598。硬码代价的主要贡献来自片内形状。',
         r'换回历史四位置 G1 硬码作补充比较，E-Q 的宏 NMSE 从 0.39752 降至 0.38694，只有 8/20 名被试下降。四库中 REFED 改善，另三库的均值略升。REFED 的配对均值差为 −0.04853，另三库为 +0.00072 至 +0.00334。这个结果支持保留 E-Q 为匹配基线，优先解决频带缺口。',
         src(r'阶段 B 固定验证终点；历史 G1 waveform K128 的同身份验证面板。'))
    page('阶段 B　／　EEG 频谱','直接看到目标后，REFED 高频仍明显衰减',
         pic('spectrum','.5'),
         r'各曲线对应同一库总电压 X。对 30 秒窗口在有效点去均值，无效点置零，乘 Hann 窗后计算频带功率比，先取对数再按评分层级汇总，乘 10 得 dB。各带左闭右开。零线表示预测与目标功率相同，负值表示衰减。图中各库共用纵轴。',
         r'REFED 目标连续组在 1–4 Hz 为 −0.21 dB，在 13–30 和 30–45 Hz 为 −33.86 和 −41.86 dB。X 连续在后两带为 −38.80、−45.22 dB。直接给定分支目标改善了部分高频恢复，剩余衰减仍很大。四库目标连续模型在 30–45 Hz 的偏差均落在 −42.46 至 −37.38 dB。',
         r'目标硬码在后两带达到 −53.17、−52.24 dB。量化增加了频带损失，但连续路径本身已显著压低高频功率。频带结果支持先在连续路径检验损失权重；功率与相位保真要分别检查。',
         src(r'evaluation/step\_10000/validation 的 X 分量频带对数功率比。'))
    ex=values['examples']
    def ex_table(v):return tex_table(['示例 RMSE','X','S','O'],[[n,*[f'{z:.3f}' for z in r]] for n,r in zip(NAMES,v['rmse'])])
    page('阶段 B　／　EEG 固定样例','同一两秒电压片段上，硬码的波形误差更大',
         pic('EEG_example','.48'),ex_table(ex[0]),
         r'REFED 被试 17，video\_1，M1，窗口 0–30 秒，展示 14–16 秒。对保存的 REFED 固定示例按原顺序取中间位置，选择独立于误差。黑线分别为观测 X、teacher 的 S、O；彩线为四组同窗预测。整窗可见，纵轴均以训练 X 标准差为单位，各行按分量取纵轴。',
         r'X 连续与目标连续在该片段的总电压误差接近，分别为 0.405 与 0.411。两种硬码总电压误差均在 0.69 以上，O 分量从连续的约 0.28 升到硬码的约 0.48。示例的 RMSE 使用上图所示时间点，验证频谱结论来自上一页的完整面板。',
         src(r'四组 validation\_examples 的共同 row/start/position，检查目标和掩码一致。'))
    page('阶段 B　／　Hb','Hb 的主要差距集中在从观测估计剩余分量',
         pic('Hb_matrix','.51'),
         r'各行 X、S、O 中的 HbO/HbR 分别为氧合与脱氧血红蛋白。两张图分别在库内汇总和四库等权汇总，色标各自给出。连续 X 输入的 HbO-O / HbR-O 误差为 0.4333 / 0.3056，直接给定目标后降至 0.0038 / 0.0042；X 自身的误差则一直较小。原路径能够贴近总曲线，同时允许两个分量出现较大的相反残差。',
         r'X 输入连续组宏水平 / 形状 NMSE 为 0.1407 / 0.0143；目标连续为 0.0059 / 0.0012。H-TQ 为 0.0344 / 0.0114，整体仍优于 X 输入连续。四个库均保留该排序。特权输入测出当前预算下的分解学习差距，无法据此把全部差距归为 X 缺少信息。',
         r'在总曲线 X 上，H-TQ 的 HbO/HbR 相关为 0.951/0.952，X 输入硬码为 0.960/0.970。H-TQ 的宏分量优势与总曲线的相关性改善是两个不同的结果。',
         src(r'四个 Hb 单元的最终 NMSE、水平/形状分解及相关。'))
    page('阶段 B　／　Hb 固定样例与训练轨迹','Hb 分量误差缩小，四个新增模型仍在改善',
         pic('Hb_example','.34'),ex_table(ex[1]),
         r'REFED 被试 17，video\_1，CH26，展示同一完整 30 秒的 HbO。样例按 EEG 页的相同规则选择。S 分量的目标硬码 RMSE 为 0.037，X 输入硬码为 0.119；两条分量在总量内相加，单看 X 会低估二者的差异。HbR 的全体结果见上一页与附表。',
         pic('trajectory','.23'),
         r'折线为 4k、7k、10k 的完整验证宏 NMSE。新增四组的最小已观测值均在 10k。H-TQ 从 7k 的 0.05129 降到 0.04574。这些差值反映等预算表现，训练收敛尚无更长轨迹验证。',
         src(r'固定 HbO 示例及所有八个配置的三个保留检查点。'))
    page('阶段 A　／　EEG 单码与序列','当前 ID 可读出部分电压形态，高频属性增益弱',
         pic('EEG_readout','.42'),
         r'同一完整两秒片段上，背景读出使用库、原生位置与局部观测 RMS 档（边界 0.5、1、2 个训练 X SD）。前三种读出使用训练片段条件均值；完整序列用同分支冻结解码器恢复波形，再提取相同属性。上图为 100 × (1 − 背景加 ID 误差 / 背景误差)；下图以单 ID 误差作分母。正值表示新增条件后误差降低。',
         r'未经对齐的 EEG 原电压在 G1-S/O 加入 ID 后分别改善 21.6\% / 45.7\%，去均值形状为 17.4\% / 17.8\%。两分支 30–45 Hz 的增益分别为 −1.8\% / −0.8\%。O 分支 1–4 Hz 宏增益为 −99.5\%；Simultaneous 为 +33.6\%，另三库为负。该属性的条件均值向验证人口迁移存在明显库间差异。',
         r'五个频带为两秒去均值 Hann 谱的积分功率，包络为八个 250 ms 箱的 RMS。频带属性 MSE 评价带内功率，波形 MSE 保留实际相位。下图比较两种已经拟合的读出，有限解码器的负增益只说明该读出在该属性上误差更大。这里评价的 G1 与阶段 B 的 E-Q 拥有不同前端。',
         src(r'A/G1\_eeg/summary.json。工作表 A\_读出含背景、单 ID、联合及序列的绝对 MSE。'))
    page('阶段 A　／　Hb 单码与序列','G4 将更多斜率信息集中到当前码字',
         pic('Hb_readout','.44'),
         r'列内依次表示同色团的原波形、两秒均值、线性斜率及二次拟合曲率。斜率为每秒变化率，曲率为每秒平方的二阶导；每个片段 20 个时间点。增益分母和符号与上一页相同，四种读出对同一批完整目标评分。',
         r'HbO 斜率在 S 分支的背景上 ID 增益从 G1 的 26.0\% 升到 G4 的 61.9\%，在 O 分支从 4.9\% 升到 47.9\%。HbR 斜率也从 23.2\% / 44.9\% 升到 56.2\% / 58.0\%。按同一被试的背景加 ID 斜率 MSE 配对，HbO-S 与 HbO-O 均有 20/20 名被试下降，HbR-S/O 分别为 17/20、18/20。与此同时，S 分支 HbO 曲率增益从 15.3\% 降到 9.6\%。',
         r'以 HbO-S 斜率为例，背景、单 ID、背景加 ID、序列 MSE 在 G1 为 0.0680、0.0505、0.0503、0.0197，在 G4 为 0.0680、0.0247、0.0259、0.0229。G4 的优势集中在局部读出；序列波形重建仍有代价。历史 H1 匹配解码器的 G1/G4 宏 NMSE 为 0.18265/0.20880。',
         src(r'A/G1\_hb 与 G4\_hb，历史 D0/D1/H1 的冻结赋码匹配解码器结果。'))
    page('阶段 A　／　逐码支持与跨库迁移','128 个码均有训练支持，跨库覆盖差异明显',
         pic('support','.32'),pic('transfer','.32'),
         r'上图每点一个码，下图每个盒子汇总一个库内有验证样本的码。盒内横线为中位数，盒边为四分位数，须长至 1.5 倍四分位距内的最远点，圆点为其外的码。下图在库内以训练码字均值预测验证波形，RMSE 以训练 X SD 表示；Hb 展示 HbO。纵轴取对数，各面板独立标度。ST/REF/SIM/VIS 依次对应四个数据库。',
         r'六个分支各 128 码均达到训练至少 3 名被试、20 个完整片段。四库均有验证支持的码数依次为 G1 EEG S/O 97/74、G1 Hb 122/127、G4 Hb 128/128。G1 EEG-O 最少的码只有 21 个验证片段、3 名被试。四个库均参与训练，这里的迁移评价针对各库留出的被试。',
         src(r'各分支 atlas，附表提供全部 768 个分支码及分库条件方差、验证原型误差。'))
    page('阶段 A　／　真实同码样本','同一码字仍覆盖具有不同幅度和形状的片段',
         pic('gallery','.44'),
         r'每个分支选择完整验证片段最多的码。各取 D0 已保存的 64 个同码真实片段，逐段减去自身均值；细灰线为其中 3 个真实片段，深色线为逐时刻中位数，阴影为 10\%–90\% 分位带。图中 EEG 为原相位电压，Hb 为 HbO，各图独立纵轴；该带描述样本分布。',
         r'例如，G1 EEG-O 的 ID 91 覆盖 91,795 个完整验证片段，G4 Hb-S 的 ID 55 覆盖 48,149 个。波形带只画保留样本。用全部验证片段逐码计算时，先逐时间点求条件方差，再对时间平均并开方。128 个码等权的中位离散度在 G1 EEG-S/O 为 0.185/0.706 个训练 X SD；G1 HbO-S/O 为 0.643/0.337，G4 为 0.569/0.345。',
         r'上述离散度使用未去均值的波形，因此包含水平变化。配合上一页的支持、迁移误差及单码属性读出，G4 Hb-S 支持可重复的局部方向属性；给单个码指定唯一精细波形仍超出这些结果。',
         src(r'D0 对应 validation\_same\_ID 的真实片段；阶段 A 全量 atlas 的条件方差。'))
    page('阶段 A　／　自然上下文','G4 换码更少，发生换码时原型差异反而更大',
         pic('context','.32'),
         r'在同一验证父窗口、同一位置内，从两个相邻且准入的 30 秒评分块构造真正向后平移 2 秒的输入。比较原输入第 8 个与平移后第 7 个两秒片段，两者物理时间一致。EEG 有 34,629 对，Hb 有 37,052 对；分别有 31,280、33,920 个位置输入缺少所需后继块。每个配对的中央片段均完整。',
         r'左图按位置、窗口、记录、被试与库逐层汇总。G1 EEG-S/O 换码率为 5.23\% / 6.24\%；G1 Hb 为 27.05\% / 35.91\%，G4 Hb 为 10.52\% / 19.64\%。原窗冻结模型 CPU 复算与历史 ID 的宏不一致率最高 0.17\%。',
         r'右图只对切换且有训练支持的配对求两个训练原型波形的均方差，再对片段池化平均并开方。HbO-S 在 G1/G4 为 0.302/0.533，HbO-O 为 0.173/0.282 个训练 X SD。因此换码频率与换码时的属性变化须分别评价。',
         r'训练形态重编号后六个分支的 128 个有支持码均能零总代价匹配。当前只有 seed 17；独立训练种子之间的形态复现，以及独立生理事件关联，仍缺少直接证据。',
         src(r'各分支 natural\_context 配对数组与训练原型；计数和条件均方变化已从配对记录复核。'))
    page('决策　／　一个可判别的下一步','立即做一个 EEG 连续频带平衡对照',
         r'\topic{优先级依据}目标连续模型在 REFED 仍损失近 42 dB 的 30–45 Hz 功率，且连续总电压的片内形状 NMSE 与 X 输入几乎相同。调整这一连续模型的频带失真，可以直接检验训练目标是否足以解释部分高频衰减。直接扩大词表只能检查量化环节。',
         r'\topic{匹配设计}建议以 E-TC 为控制，在同一 G0 训练人口、E1 八位置前端、seed 17、抽样及 10k 预算下，对 S、O、X 引入预先固定的非均匀频带权重，保留复数频谱相位与现有时域、差分评价。权重及各频带尺度仅由训练集确定，预先设置低能量频带的尺度下限；均匀加权的复数频谱平方误差与时域平方误差等价。若沿用原预热，则从同一 1k checkpoint 分叉、只在后续更新改变损失。这是本报告建议的新探索设计。',
         r'\topic{继续条件}运行前固定主要比较及可接受的时域代价。按四库、同被试检验 REFED 13–30 与 30–45 Hz 的功率偏差是否同时向零靠近；并列核对原相位时域 X/S/O NMSE、波形相关与其他库变化。仅当频带改善且既定时域容差满足时，才把这一改变带回可部署的 X 输入连续组，随后检查整数路径。',
         r'\topic{Hb 与局部模式}Hb 的连续输入差为 0.14795，约为 X 路径量化差的 4.9 倍。保留 G1 作为重建对照，保留 G4 作为斜率模式参照。若独立安排 Hb 的下一项诊断，应比较 X 输入连续模型有界延长训练后的 O 分量误差，并保持 teacher 和评分面板冻结。本轮优先只开展上述一项 EEG 对照。',
         r'\topic{停止与分流}若提升频带功率伴随原相位波形明显变差，应保留该负结果；若连续频带已改善而硬量化仍带来较大损失，再单独检验 K512。针对可部署候选，可在确定候选及匹配基线后安排两个额外种子。当前结论适用于四库开发验证与单种子等预算诊断。',
         r'\topic{可复核材料}附表给出四个库及每名被试的重建、背景与 ID 读出、每码支持和条件方差、换窗结果、训练轨迹与区间。对应运行目录为 20261120\_pattern\_a\_v1 与 20261120\_input\_quantization\_b\_v1。此次复核未启动后续训练。',
         src(r'阶段 A/B 运行合同、固定终点与阶段 A 审计；实验建议以本次开发结果为依据。'))
    tex=out/'AB_review.tex'
    tex.write_text('\n\n'.join([*lines,r'\end{document}'])+'\n')
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(['xelatex','-halt-on-error','-interaction=batchmode','-output-directory',tmp,tex.name],cwd=out,check=True)
        (out/'AB_review.pdf').write_bytes((Path(tmp)/'AB_review.pdf').read_bytes())


def phase_c(out, cells, audits, prior_deltas):
    """Resolve the four matched contrasts at each component and color, before pooling."""
    import pandas as pd

    contrasts, individual, local, patterns = [], [], [], []
    paired_shapes = []
    for mod, colors in [('EEG',['电压']),('Hb',['HbO','HbR'])]:
        panel = [cells[mod+'_'+c] for c in CELLS]
        metrics = [v['metrics']['nmse'] for v in panel]
        by_subject = [subjects(m) for m in metrics]
        keys = sorted(by_subject[0])
        assert len(keys)==20 and all(sorted(m)==keys for m in by_subject)
        for i,branch in enumerate('XSO'):
            for j,color in enumerate(colors):
                prefix = [mod,branch,color]
                values = np.array([[m[k][i,j] for m in by_subject] for k in keys])
                diffs = values @ CONTRASTS.T
                ci = paired_interval(keys,diffs)
                for key,v,d in zip(keys,values,diffs):
                    individual.append([*prefix,*key,*v,*d])
                macro = np.array([m['macro'][i][j] for m in metrics])
                for dataset in ['四库等权',*DATASETS]:
                    keep = [n for n,k in enumerate(keys) if dataset=='四库等权' or k[0]==dataset]
                    v = macro if dataset=='四库等权' else values[keep].mean(0)
                    d = v @ CONTRASTS.T
                    for col,label in enumerate(CONTRAST_NAMES):
                        bounds = ci[:,col] if dataset=='四库等权' else (None,None)
                        contrasts.append([*prefix,dataset,label,*v,d[col],*bounds,
                                          int((diffs[keep,col]>0).sum()),int((diffs[keep,col]<0).sum()),len(keep)])
                    if dataset!='四库等权':
                        np.testing.assert_allclose(v,[m['by_dataset'][dataset][i][j] for m in metrics])
                if branch=='X':
                    continue
                # Both endpoints remove each patch's level and use the same G0 target variance.
                for name,h in zip(CELLS[1::2],panel[1::2]):
                    lookup=h['single_ID_centered_readout']
                    idx=lookup['component_order'].index(branch+'_to_'+branch)
                    assert h['single_ID_unsupported_fraction'][branch+'_'+branch.lower()+'_unsupported_fraction']==0
                    for dataset in ['四库等权',*DATASETS]:
                        def coordinate(metric):
                            return metric['macro'] if dataset=='四库等权' else metric['by_dataset'][dataset]
                        one=coordinate(lookup)[idx][j]
                        seq=coordinate(h['metrics']['shape_nmse'])[i][j]
                        local.append([*prefix,name,dataset,one,seq,one-seq,1-seq/one])
                for label,metric_name,index in [('单 ID','single_ID_centered_readout',idx),('整序列','shape_nmse',i)]:
                    xq,yq = [subjects(h[metric_name] if metric_name in h else h['metrics'][metric_name]) for h in panel[1::2]]
                    assert sorted(xq)==sorted(yq)==keys
                    delta = np.array([yq[k][index,j]-xq[k][index,j] for k in keys])
                    lo,hi = paired_interval(keys,delta)
                    group_means = [delta[[k[0]==d for k in keys]].mean() for d in DATASETS]
                    paired_shapes.append([*prefix,label,np.mean(group_means),lo,hi,
                                          int((delta>0).sum()),int((delta<0).sum()),len(keys)])
        for col,label in enumerate(CONTRAST_NAMES):
            parts=[r[9] for r in contrasts if r[0]==mod and r[3]=='四库等权' and r[4]==label]
            reference=next(r[2] for r in prior_deltas if r[:2]==[mod,label])
            np.testing.assert_allclose(np.mean(parts),reference,rtol=1e-10)
    for name,h in audits.items():
        for branch,br in h['branches'].items():
            val=np.asarray(br['attribute_mse']['macro'])
            for j,attr in enumerate(h['attributes']):
                patterns.append([name,branch,attr,*val[:,j],br['additional_ID_gain_over_background'][j],
                                 br['natural_context']['macro'][0]])
    tables={
        '读表': (['项目','说明'],[
            ['范围','冻结 A/B 的 seed 17 开发验证输出。每个同模态四格组使用相同的被试、记录、窗口与位置。固定 10,000 步。'],
            ['四格顺序','X 连续、X 硬码、分支目标连续、分支目标硬码。X/S/O 分别为总量、teacher 条件估计、剩余分量。'],
            ['连续输入差','X 连续减目标连续。正值表示直接读取分支目标降低误差。'],
            ['量化差','在同一输入下取硬码减连续。正值表示该硬码条件误差更大。'],
            ['交互差','X 量化差减目标量化差。负值表示量化差在目标输入条件下更大。'],
            ['误差与区间','NMSE 采用各分量及色团的冻结训练方差。四库等权，库内被试配对重采样 10,000 次；95% 百分位区间未校正多重比较。逐库行提供点估计。'],
            ['同模型局部形状','同一 B 硬码模型、同一验证支持。比较训练拟合的当前 ID 去均值波形原型和整序列解码后去均值的形状 NMSE。正增益表示整序列降低误差。'],
            ['阶段 A 模式','三个原审计模型，按完整两秒片段评分。属性各有自己的 MSE 单位。背景为库、位置与局部 RMS 档。其模型身份与 B 的新目标输入组分别保留。'],
            ['解释','EEG 连续输入改善集中在 S，三个输出均有量化代价。Hb 的输入改善集中在 O。HbR-O 目标硬码的整序列形状改善，但单 ID 形状误差高于 X 硬码。']
        ]),
        '分支差分': (['模态','分量','信号','库','差值',*NAMES,'配对差','95%下限','95%上限','正差人数','负差人数','配对人数'],contrasts),
        '被试配对': (['模态','分量','信号','库','被试',*NAMES,*CONTRAST_NAMES],individual),
        '同模型局部形状': (['模态','分量','信号','硬码输入条件','库','单ID形状NMSE','整序列形状NMSE','单ID减序列','序列相对增益'],local),
        '硬码形状配对': (['模态','分量','信号','读出','目标硬码减X硬码','95%下限','95%上限','正差人数','负差人数','配对人数'],paired_shapes),
        '阶段A模式': (['模型','分支','属性','背景MSE','单ID MSE','背景加ID MSE','序列MSE','背景上ID增益','换码率'],patterns),
    }
    with pd.ExcelWriter(out/'phase_c.xlsx',engine='xlsxwriter') as writer:
        for name,(columns,data) in tables.items():
            pd.DataFrame(data,columns=columns).to_excel(writer,sheet_name=name,index=False)
            sheet=writer.sheets[name]
            sheet.freeze_panes(1,1)
            sheet.autofilter(0,0,len(data),len(columns)-1)
            sheet.set_column(0,len(columns)-1,16)
            if name=='读表':
                sheet.set_column(1,1,110)
            else:
                sheet.set_column(3,4,25)


def _e_pattern_tables(run,seeds):
    from src.analysis.tokenizer_pattern_audit import match_training_patterns
    support_rows=[]; attributes_rows=[]; examples=[]; matching=[]; pooled=[]
    for seed in seeds:
        for arm in ('baseline','balanced'):
            folder=run/'pattern_audit'/f'seed_{seed}'/arm
            result=read(folder/'summary.json')
            assert result['status']=='completed'
            for br in 'SO':
                at=read(folder/(br+'_atlas.json'))
                valid=at['support']['validation']
                assert np.sum(valid['patches'])==result['coverage']['validation']['complete_patches']
                source=at['by_dataset']
                for i in range(128):
                    support_rows.append([seed,arm,br,i,at['support']['train']['subjects'][i],at['support']['train']['patches'][i],
                                         valid['subjects'][i],valid['records'][i],valid['patches'][i],at['low_training_support'][i],
                                         *[v[i] for v in source['patches']],*at['pooled_validation_conditional_variance'][i]])
                v=result['branches'][br]
                metrics=v['attribute_mse']
                for d,values in [('四库等权',metrics['macro']),*metrics['by_dataset'].items()]:
                    vals=np.array(values)
                    for j,col in enumerate(result['attributes']):
                        attributes_rows.append([seed,arm,br,d,col,*vals[:,j],1-vals[2,j]/vals[0,j],1-vals[3,j]/vals[1,j]])
                examples.append([seed,arm,br,at['low_support_codes'],at['low_support_validation_patches'],
                                 result['coverage']['validation']['planned_patches'],result['coverage']['validation']['complete_patches'],
                                 v['natural_context']['macro'][0],result['validation_gallery']])
                pooled.append([seed,arm,br,*v['additional_ID_gain_over_background']])
                if seed!=17:
                    ref=run/'pattern_audit/seed_17'/arm
                    left_at=read(ref/(br+'_atlas.json'))
                    left=np.flatnonzero(~np.array(left_at['low_training_support']))
                    right=np.flatnonzero(~np.array(at['low_training_support']))
                    with np.load(ref/(br+'_training_prototypes.npz')) as l,np.load(folder/(br+'_training_prototypes.npz')) as r:
                        fields=dict(zip(result['attributes'],l['attribute_slices']))
                        np.testing.assert_array_equal(l['attribute_slices'],r['attribute_slices'])
                        i,j,cost=match_training_patterns(l['conditional_mean'][left],r['conditional_mean'][right],fields)
                        matching.extend([seed,arm,br,int(li),int(ri),float(v)] for li,ri,v in zip(left[i],right[j],cost))
    attr_names=['原电压','去均值','1–4 Hz','4–8 Hz','8–13 Hz','13–30 Hz','30–45 Hz','RMS 包络']
    tables={'E_码字覆盖':(['种子','损失','分支','码字','训练被试','训练片段','验证被试','验证记录','验证片段','低训练支持',
                          *LABELS,*['条件方差 '+s for s in attr_names]],support_rows),
            'E_属性读出':(['种子','损失','分支','数据集','属性','背景','单ID','背景加ID','序列','ID相对背景增益','序列相对单ID增益'],attributes_rows),
            'E_支持与换窗':(['种子','损失','分支','低支持码数','其验证片段','计划验证片段','完整验证片段','换码率','真实样本与分位带'],examples)}
    if matching:
        tables['E_跨种子匹配']=(['配对种子','损失','分支','seed17编号','配对编号','训练属性标准化匹配代价'],matching)
    return tables,pooled,attr_names


def phase_e(out):
    """Evaluate the stopped, conditional sequence and retain paired development estimates."""
    import pandas as pd

    run = RUNS/'20261120_eeg_band_de_v1'
    summary=read(run/'summary.json')
    cfg=read(run/'resolved_config.json')['config']
    if summary['status']!='completed' or read(run/'controller.json')['status']!='completed':
        raise ValueError('phase E requires a terminal D/E run')
    comparisons = [(r['phase'],17,read(run/(r['phase']+'.json'))) for r in summary['comparisons']]
    comparisons += [('E_X_K128',r['seed'],read(run/f"seed_{r['seed']}/paired.json")) for r in summary['confirmation']]
    paired_rows=[];gates=[];convergence=[];subject_rows=[];band_rows=[]
    for phase,seed,c in comparisons:
        for j,role in enumerate(['普通损失','频带平衡']):
            for step,val in zip(cfg['evaluation']['steps'],c['macro_nmse_4k_7k_10k'][j]):
                job=Path(c['candidate'] if j else c['baseline'])
                milestone=read(job/'evaluation'/f'step_{step:05d}'/'validation.json')
                np.testing.assert_allclose(milestone['selection_score'],val)
                m=milestone['metrics']
                bands=[10*m['band_'+b+'_log10_power_ratio']['by_dataset']['refed'][0][0] for b in ('13_30','30_45')]
                convergence.append([phase,seed,role,step,val,*np.ravel(m['nmse']['macro']),*bands])
        for metric,(original,candidate) in c['validation_metrics_baseline_candidate'].items():
            factor=10 if metric.startswith('band_') else 1.
            x,y=subjects(original),subjects(candidate)
            keys=sorted(x)
            assert sorted(y)==keys and c['subjects']==len(keys)
            delta=np.array([factor*(y[k]-x[k]) for k in keys])
            for (d,s),values in zip(keys,delta):
                for i,branch in enumerate('XSO'):
                    subject_rows.append([phase,seed,metric,branch,d,s,factor*x[d,s][i,0],factor*y[d,s][i,0],values[i,0]])
            for d in ['四库等权',*DATASETS]:
                def select(m):return np.array(m['macro'] if d=='四库等权' else m['by_dataset'][d]).ravel()*factor
                a,b=select(original),select(candidate)
                for i,branch in enumerate('XSO'):
                    bounds=np.array(c['paired'][metric]['interval95'])[:,i,0]*factor if d=='四库等权' else (None,None)
                    paired_rows.append([phase,seed,metric,branch,d,a[i],b[i],b[i]-a[i],*bounds])
            if metric.startswith('band_'):
                ref=[k for k in keys if k[0]=='refed']
                left=np.array([10*x[k][0,0] for k in ref]);right=np.array([10*y[k][0,0] for k in ref])
                idx=np.random.default_rng(17).integers(len(ref),size=(10000,len(ref)))
                gain=np.abs(left[idx].mean(1))-np.abs(right[idx].mean(1))
                band_rows.append([phase,seed,metric,left.mean(),right.mean(),abs(left.mean())-abs(right.mean()),
                                  *np.quantile(gain,[.025,.975]),len(ref)])
        nx,ny=[subjects(v) for v in c['validation_metrics_baseline_candidate']['nmse']]
        keys=sorted(nx)
        delta=np.array([np.mean(ny[k]-nx[k]) for k in keys])
        ci=paired_interval(keys,delta)
        endpoints=[c['baseline_endpoints'],c['candidate_endpoints']]
        gates.append([phase,seed,c['matched_inputs'],c['subjects'],*[v['macro_nmse'] for v in endpoints],
                      endpoints[1]['macro_nmse']-endpoints[0]['macro_nmse'],*ci,*c['gate']['checks'].values(),
                      c['gate']['passed']])
    combined=[]
    repeats=[c for phase,seed,c in comparisons if phase=='E_X_K128']
    if summary['confirmation']:
        assert len(repeats)==3
        for metric in repeats[0]['paired']:
            maps=[[subjects(v) for v in c['validation_metrics_baseline_candidate'][metric]] for c in repeats]
            keys=sorted(maps[0][0])
            assert all(sorted(m)==keys for pair in maps for m in pair)
            factor=10 if metric.startswith('band_') else 1.
            per_subject=np.mean([[factor*(y[k]-x[k]) for k in keys] for x,y in maps],axis=0)
            ci=paired_interval(keys,per_subject)
            macro=np.mean([np.mean(per_subject[[k[0]==d for k in keys]],axis=0) for d in DATASETS],axis=0)
            for i,br in enumerate('XSO'):
                combined.append([metric,br,macro[i,0],ci[0,i,0],ci[1,i,0],len(keys),len(maps)])
    tables={
        '筛选与宏分数':(['阶段','种子','验证位置输入','配对被试','普通宏NMSE','平衡宏NMSE','平衡减普通','95%下限','95%上限','两带改善','宏误差护栏','分量护栏','相关护栏','通过'],gates),
        '分量与库':(['阶段','种子','指标','分量','库','普通','平衡','平衡减普通','95%下限','95%上限'],paired_rows),
        '配对被试':(['阶段','种子','指标','分量','库','被试','普通','平衡','平衡减普通'],subject_rows),
        'REFED功率':(['阶段','种子','频带','普通dB','平衡dB','绝对偏差减少dB','95%下限','95%上限','被试'],band_rows),
        '检查点':(['阶段','种子','损失','步','宏NMSE','X NMSE','S NMSE','O NMSE','REFED X 13–30 Hz dB','REFED X 30–45 Hz dB'],convergence),
    }
    if combined:tables['三种子被试平均']=(['指标','分量','差值','95%下限','95%上限','配对被试','种子'],combined)
    atlas_inputs=[seed for phase,seed,_ in comparisons if phase=='E_X_K128']
    chart_rows=[]
    if atlas_inputs:
        atlas_tables,chart_rows,attr_names=_e_pattern_tables(run,atlas_inputs)
        tables.update(atlas_tables)
    evidence=[
        ['输入与量化、原波形与各色团','阶段 B / phase_ab_review_v1/AB_tables.xlsx','B_重建、B_差值，固定 10k；HbO、HbR 单列。'],
        ['训练支持、失败覆盖与逐码方差','阶段 A / 各模型 S_atlas.json 和 O_atlas.json；AB_tables.xlsx','A_覆盖、A_码字、A_逐码逐库，覆盖六个分支码本。'],
        ['ID、背景及同分支序列','AB_tables.xlsx 与 phase_c_v1/phase_c.xlsx','A_读出、同模型局部形状；不同读出的分母按表内说明。'],
        ['新目标、差值和绝对偏差','本工作簿','频带值用 dB；分量表差值为平衡减普通，REFED 页绝对偏差减少为正表示改善。'],
        ['统计单位','四个库固定、库内被试配对、记录与窗口先在被试内汇总','10,000 次百分位 bootstrap，95% 区间，无多重比较校正。'],
        ['停止状态',summary['stop_reason'],'额外两个种子用于通过筛选的 X 输入 K128 候选和对应基线。']]
    tables['证据与方法']=(['用途','位置或条件','读法'],evidence)
    with pd.ExcelWriter(out/'DE_tables.xlsx',engine='xlsxwriter') as writer:
        for name,(cols,records) in tables.items():
            pd.DataFrame(records,columns=cols).to_excel(writer,sheet_name=name,index=False)
            sheet=writer.sheets[name];sheet.freeze_panes(1,1)
            sheet.autofilter(0,0,len(records),len(cols)-1)
            sheet.set_column(0,len(cols)-1,19)
            if name=='证据与方法':sheet.set_column(0,2,80)
    # Show the matched spectrum and the time-domain cost for each tested arm.
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        pages=[]
        def fnum(v):return f'{v:+.3f}'
        labels={'D_target_continuous':'目标输入连续','E_X_continuous':'X 输入连续','E_X_K128':'X 输入 K128'}
        summary_rows=[]
        for g,(_,seed,c) in zip(gates,comparisons):
            lo,hi=g[7:9]
            a,b=c['gate']['refed_absolute_bias_gain_db']
            summary_rows.append([labels[g[0]],seed,f'{g[6]:+.5f}',f'[{lo:+.5f}, {hi:+.5f}]',
                                 fnum(a),fnum(b),'通过' if g[-1] else '停止'])
        overview=tex_table(['输入和表示','种子','宏 NMSE 差','95\\% 区间','13–30','30–45','筛选'],summary_rows)
        pages.append('\n\n'.join([r'\section*{频带平衡损失能否改善 EEG 整数重建？}',
            '阶段 D 与 E 在四库开发面板比较普通损失与新增频带损失，冻结 30 秒输入、两秒锚点与八位置波形结构。每个种子内，候选与基线共享普通损失的 1,000 步预热，随后各训练到第 10,000 步。目标输入将各分支的 teacher 目标分别交给编码器；X 输入只提供观测电压。硬码解码只读取同一分支的 K128 整数序列。',
            overview,
            '宏 NMSE 差为频带平衡减普通，负值表示时域误差降低。最后两个数值列为 REFED 总电压的绝对功率偏差减少量（dB），正值表示更接近零。各分量误差先除以训练冻结的目标方差，再按位置、窗口、记录、被试及数据集分层等权。区间为四库内配对重采样被试 10,000 次的 95\\% 百分位区间，未作多重比较校正。',
            '损失使用 1–4、4–8、8–13、13–30、30–45 Hz 五个左闭右开频带。对完整两秒片段去均值、加 Hann 窗，对预测与目标的复数谱之差取带内平方能量，保留相位误差。各带依训练记录的能量及原目标方差归一化，按 X、S、O 和频带等权，系数为 0.1，叠加到普通时域与差分损失。频带尺度来自 443 条训练记录中的 6,624 个完整片段，下限为 0.001。',
            '判定按预先声明的固定终点进行。下页逐模型给出误差、相关与五带功率；电子表同时保留分库、分被试结果以及 4k、7k、10k 的轨迹。既有 Hb 色团与输入量化矩阵见阶段 A/B/C 的保留表。']))
        if summary['stop_reason']=='E_X_K128_joint_criterion_failed':
            pages[0]+='\n\n'+('本轮两个连续模型通过筛选，K128 候选在两个高频带的偏差都扩大。'
                '频带平衡在目标输入及实际 X 输入的连续路径上产生收益，该收益未传入现有硬码接口。'
                '普通损失 K128 保留为整数基线。按预设停止规则，额外种子 29、43 未启动。')
        for phase,seed,c in comparisons:
            fig,(ax,bx)=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
            for j,(label,color,marker) in enumerate(zip(['普通损失','频带平衡'],COLORS,'os')):
                db=[c['validation_metrics_baseline_candidate']['band_'+key+'_log10_power_ratio'][j]['by_dataset']['refed'][0][0]*10 for key in BANDS]
                ax.plot(range(5),db,marker=marker,c=color,label=label,ls=STYLES[j])
                arr=np.array(c['validation_metrics_baseline_candidate']['nmse'][j]['macro']).ravel()
                bx.bar(np.arange(3)+(j-.5)*.35,arr,.35,color=color,label=label,hatch='//' if j else None)
            ax.axhline(0,color='.45',lw=.7);ax.set(xticks=range(5),xticklabels=[s.replace('_','–') for s in BANDS],
                ylabel='预测 / 目标总电压功率（dB）',xlabel='频带（Hz）',title='REFED 配对验证面板')
            bx.set(xticks=range(3),xticklabels=list('XSO'),ylabel='验证 NMSE',title='四库等权，越低越好')
            ax.legend();bx.legend()
            image=f'comparison{len(pages)}'
            save(fig,tmp,image)
            names={'D_target_continuous':'目标输入连续模型','E_X_continuous':'X 输入连续模型','E_X_K128':'X 输入 K128'}
            title=names[phase]
            x,y=c['baseline_endpoints'],c['candidate_endpoints']
            table=tex_table(['指标','普通损失','频带平衡','平衡减普通'],
                [[label,f'{l:.5f}',f'{r:.5f}',f'{r-l:+.5f}'] for label,l,r in
                [('宏 NMSE',x['macro_nmse'],y['macro_nmse']),*[(br+' NMSE',x['component_nmse'][i],y['component_nmse'][i]) for i,br in enumerate('XSO')],
                 *[(br+' 相关',x['correlation'][i],y['correlation'][i]) for i,br in enumerate('XSO')]]])
            failed=[n for n,v in zip(['两高频带改善','宏 NMSE','分量 NMSE','分量相关'],c['gate']['checks'].values()) if not v]
            checks=('未满足'+ '、'.join(failed)) if failed else '所有预设条件均满足'
            pages.append('\n\n'.join([r'\section*{'+title+f'：配对开发验证，seed {seed}'+r'}',
                f'同一输入、同一 1k 预热分叉，以 10k 为主终点。每个窗口 30 秒，评分按位置、窗口、记录、被试与库逐层等权。共 {c["matched_inputs"]:,} 个验证位置输入、{c["records"]} 条记录与 {c["subjects"]} 名被试。',
                r'\begin{center}\includegraphics[width=\linewidth]{'+image+r'.png}\end{center}',table,
                f'在 REFED 总电压上，13–30 Hz 由 {x["refed_db"][0]:.2f} 变为 {y["refed_db"][0]:.2f} dB，30–45 Hz 由 {x["refed_db"][1]:.2f} 变为 {y["refed_db"][1]:.2f} dB。零为预测与目标功率一致。两带须同时接近零，宏和各分量 NMSE 相对增加上限分别为 2\\% 和 5\\%，各分量相关的绝对下降上限为 0.01。判定为{checks}。',
                'S 是 teacher 条件估计，O 为剩余分量，二者相加得到总电压 X。训练在完整两秒窗对预测减目标的复数频谱计算带内残差能量，按训练记录冻结尺度加权；评分在 30 秒窗取预测 / 目标带内功率的对数，再分层汇总。详表另列水平、去均值形状、逐库及被试配对差。']))
        if chart_rows:
            examples=[r for r in chart_rows if r[0]==17]
            matrix=100*np.array([r[3:] for r in examples])
            labels=[('普通' if r[1]=='baseline' else '频带平衡')+' '+r[2] for r in examples]
            fig,ax=plt.subplots(figsize=(9,3.4),layout='constrained')
            edge=max(1,np.ceil(abs(matrix).max()))
            im=heat(ax,matrix,attr_names,labels,'背景加入当前 ID 后的属性误差降低（%）','.1f','RdBu',-edge,edge)
            fig.colorbar(im,ax=ax,label='增益（%）')
            save(fig,tmp,'local_readout')
            pages.append(r'\section*{K128 局部码字的属性读出}'+'\n'+
                         r'\begin{center}\includegraphics[width=\linewidth]{local_readout.png}\end{center}'+'\n'+
                         '在相同验证片段上，背景为数据集、原生位置和局部观测幅度档，当前码为同一分支的两秒 ID。图中增益为 100 倍的一减联合读出误差与背景误差之比，正值为误差降低。属性均值只在训练片段拟合。整序列读出由本模型的冻结整数解码器生成，未经时移或相位对齐。'+
                         '\n\n'+'图为种子 17。详表逐码保留按被试与记录的支持、观测波形与频带属性的类内方差、低支持覆盖及真实样本图集位置；另列各库单码、背景、联合及整序列的误差。')
            # Display the most frequent validation code per branch, selected without waveform error.
            fig,axes=plt.subplots(2,2,figsize=(9,4.8),layout='constrained')
            gallery_info=[]
            for j,arm in enumerate(('baseline','balanced')):
                folder=run/'pattern_audit/seed_17'/arm
                with np.load(folder/'validation_same_ID.npz') as gallery:
                    for i,br in enumerate('SO'):
                        code=int(np.argmax(gallery['frequency'][i]))
                        waves=gallery[f'{br}_{code}_waves'][:,:,0]
                        centered=waves-waves.mean(1,keepdims=True)
                        low,mid,high=np.quantile(centered,[.1,.5,.9],axis=0)
                        t=np.arange(400)/200
                        ax=axes[i,j]
                        ax.fill_between(t,low,high,color=COLORS[j],alpha=.25)
                        for w in centered[:3]:ax.plot(t,w,c='.60',lw=.65)
                        ax.plot(t,mid,c=COLORS[j],lw=1.3)
                        ax.set(title=('普通' if j==0 else '频带平衡')+f' · {br} · ID {code}',xlabel='时间（秒）',ylabel='去均值电压 / 训练 X SD')
                        gallery_info.append([('普通' if j==0 else '频带平衡'),br,code,int(gallery['frequency'][i,code]),len(waves)])
            save(fig,tmp,'gallery')
            pages.append('\n\n'.join([r'\section*{相同码字的真实验证片段}',r'\begin{center}\includegraphics[width=\linewidth]{gallery.png}\end{center}',
                tex_table(['损失','分支','ID','完整验证片段','显示分位带样本'],gallery_info),
                '每格选取本模型、本分支最常见的验证码，从固定随机抽取的最多 64 个真实片段计算中位数（实线）和 10%–90% 分位带（阴影），叠加前三个样本（灰线）。每段减自身均值，各面板独立纵轴。码编号为各自码本内部标签。'.replace('%',r'\%'),
                '全部码字的验证被试、记录、片段支持及属性条件方差见工作簿。A/B 的 Hb 图集继续使用原验证支持。']))
        notes='阶段 E 按已完成的固定终点评价。'
        if not summary['confirmation']:notes+='预设流程在首次未达联合条件的配对结束，本轮额外种子配对数为零。'
        else:notes+='种子 29 与 43 各自使用共享普通预热的候选与基线；三种子统计先在相同被试内平均。'
        pages[-1]+='\n\n'+notes+' 结论适用于该训练预算与开发人口。完整四格重建、Hb 色团、逐码支持与模式读出见附表的证据索引。'
        tex=r'''\documentclass[11pt,a4paper,fontset=none]{ctexart}
\usepackage[margin=20mm]{geometry}\usepackage{fontspec,graphicx,booktabs}
\setCJKmainfont{Noto Sans CJK SC}\setmainfont{DejaVu Sans}
\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\begin{document}
'''+'\n\\newpage\n'.join(pages)+r'\end{document}'
        (tmp/'DE_review.tex').write_text(tex)
        subprocess.run(['xelatex','-halt-on-error','-interaction=batchmode','DE_review.tex'],cwd=tmp,check=True)
        (out/'DE_review.pdf').write_bytes((tmp/'DE_review.pdf').read_bytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--phase-e',action='store_true',help='evaluate all eligible completed D/E arms')
    modes.add_argument('--phase-c',action='store_true',help='export component-level matched contrasts and local pattern evidence')
    args=parser.parse_args()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    if args.phase_e:
        phase_e(out)
        return
    jobs,rows,cells,trajectory,health,panels,deltas,audits,atlases=collect()
    if args.phase_c:
        phase_c(out,cells,audits,deltas)
        return
    book(out,cells,trajectory,health,panels,deltas,audits,atlases)
    examples,gallery=figures(out,jobs,rows,cells,trajectory,deltas,audits,atlases)
    transfer_figure(out,atlases)
    values=dict(deltas=deltas,examples=examples,gallery=gallery,historical_eeg=historical_eeg(jobs),
                matched_final_scores={k:v['selection_score'] for k,v in cells.items()},
                phase_a_terminal=read(A/'summary.json'),phase_b_models_completed=4)
    (out/'review_values.json').write_text(json.dumps(values,ensure_ascii=False,indent=2)+'\n')
    report(out,values,cells)


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    main()
