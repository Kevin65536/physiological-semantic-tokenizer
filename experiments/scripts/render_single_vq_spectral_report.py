"""Render the dated EEG-S loss-location report from retained ordinary development evidence."""
import argparse
import csv
import json
from pathlib import Path
import tempfile
import zipfile

import fitz
from fontTools import subset
from fontTools.ttLib import TTFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'experiments/runs/physiology_semantic_tokenizer/single_vq_spectral_location/20261125_v1'
OLD = RUN.parents[1]/'single_vq_capacity/20261124_v1/review_20261124_v1'
MODELS = [(17,'A'),(17,'B'),(17,'C'),(43,'A'),(43,'C')]
F = ['θ 对数功率','α 对数功率','θ 功率占比','α 功率占比']
DS = {'eeg_fnirs_single_trial':'Single-Trial','refed':'REFED','simultaneous_eeg_nirs':'Simultaneous','visual_cognitive_motivation':'Visual'}
COLOR = {'A':'#415c73','B':'#ac5a23','C':'#087971'}
MARK = {'A':'o','B':'s','C':'D'}
LINE = {'A':'--','B':':','C':'-.'}


def read(p):
    return json.loads(p.read_text())


def table(headers, rows):
    return '<table><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr>'+''.join(
        '<tr>'+''.join(f'<td>{v}</td>' for v in row)+'</tr>' for row in rows)+'</table>'


def save_csv(path, headers, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def macro(s,k):
    return np.ravel(s['metrics'][k]['macro'])


def evidence():
    result = read(RUN/'summary.json')
    assert result['status']=='completed' and read(RUN/'stage.json')['stage']=='completed'
    assert result['first_seed']==read(RUN/'selection.json')
    defs = read(RUN/'resolved_config.json')
    assert defs['features']==['log_theta_power','log_alpha_power','theta_fraction','alpha_fraction']
    groups = {}
    for seed,arm in MODELS:
        folder = RUN/f'seed_{seed}'/arm
        train = read(folder/'summary.json')
        assert (train['status'],train['completed_step'])==('completed',10000)
        s = read(folder/'evaluation/summary.json')
        assert (s['status'],s['inputs'],s['subjects'],s['train_people'])==('completed',53476,16,60)
        assert (s['records'],s['windows'],s['development_complete_patches'],s['train_complete_patches'])==(119,1557,780210,222885)
        for m in s['metrics'].values():
            assert m['nonfinite_entries']==0
            for d, values in m['by_dataset'].items():
                np.testing.assert_allclose(values,np.mean([p['value'] for p in m['by_subject'] if p['dataset']==d],axis=0))
            np.testing.assert_allclose(m['macro'],np.mean(list(m['by_dataset'].values()),axis=0))
        gain=1-macro(s,'conditional_id')/macro(s,'background')
        np.testing.assert_allclose(s['id_gain'],gain)
        np.testing.assert_allclose([s['mean_id_gain'],s['selection_score']],[gain.mean(),macro(s,'nmse').mean()])
        phase=result['first_seed' if seed==17 else 'repeat_seed']
        assert phase['selected']=='C' and phase['results'][arm]['mean_id_gain']==s['mean_id_gain']
        if arm!='A':
            base=groups[seed,'A']
            guard=defs['guardrails']
            actual=[gain.mean()>0,gain.mean()>base['mean_id_gain'],
                    s['selection_score']<=base['selection_score']*(1+guard['macro_nmse_relative_increase_max']),
                    (macro(s,'nmse')<=macro(base,'nmse')*(1+guard['component_nmse_relative_increase_max'])).all(),
                    (macro(s,'correlation')>=macro(base,'correlation')-guard['component_correlation_drop_max']).all()]
            assert dict(zip(['positive','improved','macro','components','correlations'],actual))==phase['checks'][arm]
        groups[seed,arm]=s
    # Compare the actual scored identities, complete-patch masks, background and targets across models.
    for ds in DS:
        for role in ('train','validation'):
            name=f'{ds}_{role}.npz'
            with np.load(RUN/'seed_17/A/evaluation'/name) as first:
                reference={k:first[k] for k in ('examples','valid','bg','target')}
            for seed,arm in MODELS[1:]:
                with np.load(RUN/f'seed_{seed}'/arm/'evaluation'/name) as z:
                    for k,v in reference.items():
                        np.testing.assert_array_equal(v,z[k])
    for (seed,arm),s in groups.items():
        folder=RUN/f'seed_{seed}'/arm/'evaluation'
        unseen=0
        for ds in DS:
            keys={}
            for role in ('train','validation'):
                with np.load(folder/f'{ds}_{role}.npz') as z:
                    keys[role]=z['bg'].astype(np.int64)*8192+z['id'].astype(np.int64)
            unseen+=int((~np.isin(keys['validation'],np.unique(keys['train']))).sum())
        s['unseen_joint_tokens']=unseen
    return result,groups


def dataset_gain(s,ds):
    m=s['metrics']
    return 100*(1-np.array(m['conditional_id']['by_dataset'][ds])/np.array(m['background']['by_dataset'][ds]))


def portraits(tables):
    folder=RUN/'seed_43/C/evaluation'
    calibration=read(RUN/'calibration.json')
    rows=[]
    for role in ('train','validation'):
        ids,values=[],[]
        people=[]
        for ds,c in calibration.items():
            with np.load(folder/f'{ds}_{role}.npz') as z:
                ids.append(z['id'])
                if role=='validation':
                    with np.load(OLD/'work'/f'features_eeg_{ds}.npz') as meta:
                        np.testing.assert_array_equal(meta['examples'],z['examples'])
                        names=np.char.add(ds+'/',meta['people'].astype(str))
                        people.append(np.repeat(names,z['valid'].sum(1)))
                values.append(100*(z['target'][:,2:]*np.array(c['sd'])[2:]+np.array(c['mean'])[2:]))
        ids=np.concatenate(ids)
        values=np.concatenate(values)
        if people:people=np.concatenate(people)
        if role=='train':
            counts=np.bincount(ids,minlength=8192)
            candidates=counts>=100
            picks=[np.argmax(np.where(candidates,np.bincount(ids,weights=v,minlength=8192)/np.maximum(counts,1),-1))
                   for v in values.T]
        for word in [*picks, None]:
            subset_values=values if word is None else values[ids==word]
            for j,feature in enumerate(F[2:]):
                rows.append([role,'全体' if word is None else word,feature,len(subset_values),
                             *np.quantile(subset_values[:,j],[.1,.5,.9]),
                             (len(np.unique(people if word is None else people[ids==word])) if role=='validation' else '')])
    save_csv(tables/'word_examples.csv',['split','S_ID','teacher_S_fraction','patches','P10_pct','P50_pct','P90_pct','development_people'],rows)
    return rows,picks,int(candidates.sum())


def export_tables(groups,result,dest):
    records,features,datasets,subjects=[],[],[],[]
    for (seed,a),s in groups.items():
        records.append([seed,a,s['mean_id_gain']*100,s['selection_score'],*macro(s,'nmse'),*macro(s,'correlation'),
                        s['emitted_development_ids'],s['supported_ids'],s['supported_development_tokens'],s['development_complete_patches'],s['unseen_joint_tokens']])
        for j,k in enumerate(F):
            features.append([seed,a,k,s['id_gain'][j]*100,*[macro(s,m)[j] for m in ('background','conditional_id','id_only','decoded')],
                             s['within_id_variance'][j],s['between_id_variance'][j]])
        for ds in DS:
            datasets.append([seed,a,ds,*dataset_gain(s,ds),dataset_gain(s,ds).mean()])
    for phase in [result['first_seed'],result['repeat_seed']]:
        for contrast,r in phase['contrasts'].items():
            for p in r['subjects']:
                subjects.append([phase['seed'],contrast,p['dataset'],p['subject'],*np.multiply(p['gain_delta'],100),
                                 100*np.mean(p['gain_delta']),*p['nmse_delta']])
    save_csv(dest/'models.csv',['seed','arm','mean_ID_gain_pct','macro_NMSE','X_NMSE','S_NMSE','O_NMSE','X_r','S_r','O_r',
                               'emitted_dev_IDs','supported_train_IDs','dev_tokens_supported','dev_tokens_total','dev_tokens_without_training_background_ID_cell'],records)
    save_csv(dest/'attributes.csv',['seed','arm','attribute','gain_pct','background_MSE','conditional_ID_MSE','ID_only_MSE',
                                    'decoded_MSE','within_ID_variance','between_ID_variance'],features)
    save_csv(dest/'dataset_gains.csv',['seed','arm','dataset',*F,'mean_gain_pct'],datasets)
    save_csv(dest/'paired_subjects.csv',['seed','contrast','dataset','subject',*[k+'_pp' for k in F],'mean_delta_pp',
                                        'X_NMSE_delta','S_NMSE_delta','O_NMSE_delta'],subjects)


def figures(g,result,word_rows,out,tab):
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=240,facecolor='white')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,1.85),layout='constrained')
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def node(x,y,label,color):
        ax.text(x,y,label,ha='center',va='center',fontsize=10,color=color,
                bbox=dict(boxstyle='round,pad=.45',fc='white',ec=color,lw=1))
    def arrow(start,end,color='#596e75'):
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',lw=1.2,color=color))
    node(.10,.72,'观测 X\n单电极 30 秒','#314e5b')
    node(.37,.72,'EEG-S 编码与量化\n15 个整数 ID','#314e5b')
    node(.67,.72,'同分支序列解码\n30 秒 S 波形','#314e5b')
    node(.37,.20,'C　中心词向量\n线性预测四项属性',COLOR['C'])
    node(.67,.20,'B　当前 2 秒波形\n计算四项频谱属性',COLOR['B'])
    ax.text(.95,.72,'波形监督\nA / B / C',ha='center',va='center',fontsize=10)
    arrow((.18,.72),(.264,.72));arrow((.478,.72),(.565,.72));arrow((.78,.72),(.89,.72))
    arrow((.37,.56),(.37,.37),COLOR['C']);arrow((.67,.56),(.67,.37),COLOR['B'])
    save(fig,'loss_locations')
    fig,axes=plt.subplots(1,2,figsize=(8,3.25),layout='constrained',gridspec_kw={'width_ratios':[1,1.15]})
    for seed,a in MODELS:
        if seed==17:
            axes[0].plot(100*np.array(g[seed,a]['id_gain']),np.arange(4)+{'A':-.15,'B':0,'C':.15}[a],ls='',marker=MARK[a],color=COLOR[a],label=a)
    for (seed,a),m,col in [((17,'B'),'s',COLOR['B']),((17,'C'),'o',COLOR['C']),((43,'C'),'D','#174c50')]:
        delta=100*(np.array(g[seed,a]['id_gain'])-g[seed,'A']['id_gain'])
        axes[1].plot(delta,np.arange(4)+({'s':-.15,'o':0,'D':.15}[m]),ls='',marker=m,color=col,label=f'{a}−A · {seed}')
    for ax,title,xlabel in zip(axes,['首轮各属性的单 ID 增益','相对同种子 A 的配对增量'],['相对背景的 MSE 降幅（%）','增益差（百分点）']):
        ax.set(yticks=np.arange(4),yticklabels=F,ylim=(3.6,-.6),title=title,xlabel=xlabel)
        ax.axvline(0,color='#64717b',lw=.8);ax.grid(axis='x',alpha=.18)
        ax.legend(loc='lower center',bbox_to_anchor=(.5,-.42),ncol=3,frameon=False,fontsize=8,handletextpad=.1,columnspacing=.6)
    save(fig,'attribute_gain')
    phases=[result['first_seed'],result['repeat_seed']]
    fig,ax=plt.subplots(figsize=(8,4.25),layout='constrained')
    for i,phase in enumerate(phases):
        people=sorted(phase['contrasts']['C-minus-A']['subjects'], key=lambda v:(list(DS).index(v['dataset']),v['subject']))
        y=np.arange(len(people))
        ax.scatter(y+[-.12,.12][i],[100*np.mean(v['gain_delta']) for v in people],s=35,marker=['o','D'][i],
                   edgecolor=COLOR['C'],facecolor=['white',COLOR['C']][i],label=f"种子 {phase['seed']}",zorder=3)
    labels=[p['subject'].replace('subject_','') for p in people]
    ax.set_xticks(y,labels,rotation=45)
    for d in DS:
        ii=[i for i,p in enumerate(people) if p['dataset']==d]
        ax.text(np.mean(ii),.97,DS[d],va='top',ha='center',fontsize=9,transform=ax.get_xaxis_transform())
        if ii[-1]!=15: ax.axvline(ii[-1]+.5,color='#c3cdd0',lw=.8)
    ax.set(ylabel='个人四属性平均增益差（百分点）',xlabel='同一开发被试',ylim=(-.12,.67))
    ax.axhline(0,color='#6b787d',lw=1);ax.grid(axis='y',alpha=.18);ax.legend(loc='upper left',bbox_to_anchor=(0,.92),ncol=2,frameon=False)
    save(fig,'paired_subjects')
    fig,axes=plt.subplots(1,2,figsize=(8,3.45),layout='constrained')
    for a in 'ABC':
        s=g[17,a]
        axes[0].plot(np.arange(4),macro(s,'decoded'),marker=MARK[a],ls=LINE[a],color=COLOR[a],label=a)
        axes[1].plot(np.arange(3),macro(s,'correlation'),marker=MARK[a],ls=LINE[a],color=COLOR[a],label=a)
    axes[0].set(yscale='log',xticks=range(4),xticklabels=['θ 对数','α 对数','θ 占比','α 占比'],ylabel='标准化属性 MSE（对数轴）',title='从解码波形计算的频谱误差')
    axes[1].set(xticks=range(3),xticklabels=['X','S','O'],ylabel='平均 Pearson 相关',title='同一窗口内的波形相关',ylim=(.48,.69))
    for ax in axes: ax.grid(axis='y',alpha=.18);ax.legend(ncol=3,frameon=False,loc='lower center',bbox_to_anchor=(.5,-.45),fontsize=9)
    save(fig,'decoded_waveform')
    fig,axes=plt.subplots(1,2,figsize=(8,2.8),layout='constrained',sharey=True)
    for ax,feature in zip(axes,F[2:]):
        block=[r for r in word_rows if r[0]=='validation' and r[2]==feature]
        for j,(_,word,_,count,lo,md,hi,_) in enumerate(block):
            ax.plot([lo,hi],[j,j],lw=3,color='#82a7a7',solid_capstyle='round')
            ax.plot(md,j,marker='D',ms=6,color='#174c50')
        ax.set(yticks=range(3),yticklabels=[str(r[1]) for r in block],ylim=(2.6,-.6),xlim=(0,45),xlabel='教师 S 片段功率占比（%）',title=feature)
        ax.grid(axis='x',alpha=.18)
    save(fig,'word_members')
    rows=[]
    for case in ('median','difficult'):
        fig,axes=plt.subplots(3,2,figsize=(8,4.6),layout='constrained',sharex='col',gridspec_kw={'width_ratios':[1.6,1]})
        source={}
        for seed,a in MODELS:
            folder=RUN/f'seed_{seed}'/a/'evaluation'
            with np.load(folder/f'{case}_waveforms.npz') as z:
                v={k:z[k] for k in z.files}
            if not source:source=v
            for k in ['example','mask','x','s','o']: np.testing.assert_array_equal(source[k],v[k])
            assert v['mask'].all()
            info=read(folder/'simultaneous_eeg_nirs_fixed_windows.json')[case]
            for i,(key,target) in enumerate(zip(['sum','S','O'],'xso')):
                ref=v[target].ravel(); pred=v['reconstructed_'+key].ravel(); t=np.arange(len(ref))/200
                rows.append([case,seed,a,info['subject'],info['record'],info['position'],target.upper(),info['nmse'][i],
                             np.std(ref),np.std(pred),np.corrcoef(ref,pred)[0,1],np.max(ref),np.max(pred)])
                if seed!=17: continue
                for j,ax in enumerate(axes[i]):
                    if a=='A': ax.plot(t,ref,color='#252d31',lw=.85,label='观测 / 教师目标')
                    ax.plot(t,pred,color=COLOR[a],ls=LINE[a],lw=.85,label=a)
                    ax.set_xlim((0,30) if j==0 else (14,16)); ax.grid(axis='y',alpha=.15)
                    if j==0:ax.set_ylabel(target.upper()+'（输入 SD）')
        for row in axes:
            row[1].set_ylim(row[0].get_ylim())
        for j,title in enumerate(['完整 30 秒','固定放大 14–16 秒']):
            axes[0,j].set_title(title);axes[2,j].set_xlabel('窗口内时间（秒）')
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside upper center',ncol=4,frameon=False)
        save(fig,'wave_'+case)
    save_csv(tab/'fixed_windows.csv',['case','seed','arm','subject','record','position','signal','NMSE','target_SD',
                                      'predicted_SD','correlation','target_max','predicted_max'],rows)


def make_report(out,g,result,word_rows,picks,n_candidates,work):
    fontdir=Path('/usr/share/fonts/truetype')
    for font in ('dejavu/DejaVuSans.ttf','droid/DroidSansFallbackFull.ttf'):
        face=TTFont(fontdir/font)
        options=subset.Options();options.layout_features=[];options.drop_tables+=['GSUB','GPOS','FFTM']
        small=subset.Subsetter(options=options)
        small.populate(text=Path(__file__).read_text()+''.join(chr(i) for i in range(32,127)))
        small.subset(face);face.save(work/Path(font).name)
    archive=fitz.Archive(str(work))
    css="""@font-face{font-family:latin;src:url(DejaVuSans.ttf)} @font-face{font-family:cn;src:url(DroidSansFallbackFull.ttf)}
    *{font-family:latin,cn} body{font-size:12pt;color:#213740;line-height:1.52;margin:0}
    h1{font-weight:normal;font-size:24pt;color:#153c45;margin:4pt 0 12pt}
    h2{font-weight:normal;font-size:14pt;color:#086861;margin:10pt 0 2pt}
    p{margin:3pt 0 7pt}table{width:100%;border-collapse:collapse;font-size:10.3pt;margin:5pt 0 8pt}
    th,td{padding:4pt 4pt;text-align:left;border-bottom:.5pt solid #c4d1d0}th{background:#e4efec;color:#1a504e}"""
    doc=fitz.open()
    def box(pg,rect,html):
        spare,scale=pg.insert_htmlbox(fitz.Rect(rect),html,css=css,archive=archive,scale_low=1)
        if spare<0:raise ValueError(f'page {pg.number+1} overflow near {html[:100]}')
        return rect[3]-spare
    def page(topic,title,body,source,figure=None,after=''):
        pg=doc.new_page(width=642,height=785)
        box(pg,(32,18,610,41),f'<span style="font-size:9pt;color:#237569">EEG-S 频谱损失位置实验　 /　 {topic}</span>')
        y=box(pg,(32,44,610,732),'<h1>'+title+'</h1>'+body)
        if figure:
            path=out/'figures'/(figure+'.png')
            pix=fitz.Pixmap(str(path)); h=578*pix.height/pix.width
            pg.insert_image(fitz.Rect(32,y+4,610,y+4+h),filename=str(path))
            y+=h+12
        if after:y=box(pg,(32,y,610,739),after)
        if y>739:raise ValueError((title,y))
        pg.draw_line((32,746),(610,746),color=(.75,.81,.81),width=.55)
        box(pg,(32,751,610,781),f'<span style="font-size:8pt;color:#536968">{source}　·　2026-11-25　·　{pg.number+1:02d}</span>')
    def p(v):return '<p>'+v+'</p>'
    def h(v):return '<h2>'+v+'</h2>'
    def pct(v):return f'{100*v:.3f}%'
    base,dec,local=[g[17,k] for k in 'ABC']
    ref2,loc2=g[43,'A'],g[43,'C']
    scores=table(['种子','条件','单 ID 平均增益','相对 A（百分点）','宏 NMSE','预设判定'],[
        [seed,arm,pct(s['mean_id_gain']),'参照' if arm=='A' else f"{100*(s['mean_id_gain']-g[seed,'A']['mean_id_gain']):+.3f}",
         f"{s['selection_score']:.5f}",{'A':'配对基线','B':'停止','C':'进入复核' if seed==17 else '通过复核'}[arm]]
        for (seed,arm),s in g.items()])
    page('结果总览','单个 EEG-S 词的频谱读出有小幅、可复现的收益',
        p('同一段 30 秒脑电每 2 秒发出一个 S 词。直接要求这个词预测局部 θ、α 属性（C），在两个训练种子上都提高了它对未见被试的属性读出，电压重建误差接近同种子基线（A）。')
        +scores
        +h('增量有多大')
        +p('在已知数据集、电极和观测幅度的条件下，C 的单 ID 对背景读出的四项相对误差降幅平均为 1.82% 和 1.98%。同种子 A 已有 1.69% 和 1.79% 的增益，C 带来的额外收益分别为 0.126 和 0.188 个百分点。')
        +h('损失的位置改变了学到的内容')
        +p('把同样的约束放在解码波形上（B），可将首轮解码 S 的 α 对数功率标准化均方误差从 102.78 降至 2.54；单 ID 平均增益却下降 0.039 个百分点，S 波形相关也跌破护栏。')
        +h('这轮结果支持的用途')
        +p('若下游要从整数词读取短时频谱，可保留 C 作为单码本候选。两次训练都通过重建护栏，但训练支持词的片段覆盖率没有一致提升。该结果限于离线教师频谱属性与同一开发人群，生理事件识别、跨模态机制和独立人群泛化仍待验证。')
        +p('证据目录位于 experiments/runs/physiology_semantic_tokenizer/ 下的 single_vq_spectral_location/20261125_v1/。各页标注对应文件，配套数据包含逐人配对和固定窗口数值。'),
        '运行总表 summary.json；筛查 selection.json；每臂 evaluation/summary.json')
    page('实验设计','三个条件共享训练暴露，只改变频谱监督的位置',
        p('模型输入为单电极 200 Hz 的观测电压 X，每窗 30 秒切成 15 个 400 点片段。EEG-S 表示离线教师估计的分量，EEG-O 表示 X 扣除 S 后的残差。训练时二者提供监督，应用时编码器只读取 X。')
        +table(['条件','训练中的计算路径','解码接口'],[
            ['A　基线','X → 上下文编码 → 逐片段量化 → 同分支解码','各分支的 15 个整数词恢复 S 或 O；两者相加恢复 X。'],
            ['B　波形约束','从完整解码序列中取当前 2 秒 S 波形，计算四项属性，与教师 S 比较。','沿用 A 的整数词与波形解码。'],
            ['C　单词约束','只将中心 S 词的 64 维量化码向量送入线性层，预测同一片段的四项属性。','线性层用于辅助训练。波形解码仍读取同分支的全部 15 个词。']])
        +p('每个分支一个 8,192 词的码本，网络宽度 256；Hb 两个分支沿用既有参照。本轮监督只加在 EEG-S。基础损失取 S/O 方差归一化误差的均值，加 0.5 倍 X 误差、量化承诺损失与 0.1 倍相邻差分项。B、C 另加 0.01 倍四属性平均平方误差。')
        +h('分母与训练边界')
        +p('60 名训练被试拟合模型、属性尺度和读出表。开发面板含 16 名其他被试，Single-Trial、REFED、Simultaneous、Visual 分别为 5、5、4、2 人。119 条记录提供 1,557 个时间窗、53,476 个窗口×位置和 780,210 个完整开发片段；冻结训练评分面板含 222,885 个完整片段。')
        +p('每个种子共享 1,000 步预热，批量 64，固定取第 10,000 步。首轮 A/B/C 和复核 A/C 全部完成。B 按规则在首轮停止。C 首次加载新增线性头的失败已修复，A/B 从第 1,500 步恢复，失败记录保留。历史 A 为 0.50602，所有配对都使用本轮重训的 A。'),
        'resolved_config.json；seed_*/resolved_training.json；launch.json；G0 数据合同','loss_locations')
    page('指标与读出','同一个 ID 能在已知背景之外解释多少频谱误差',
        h('四个评分目标')
        +p('每个完整 2 秒教师 S 片段先减均值、乘对称 Hann 窗，再在 0.5 Hz 频率网格上积分单边功率。θ 取 4 至 8 Hz、α 取 8 至 13 Hz，均含下界、不含上界。两项绝对属性为恢复原缓存电压尺度后的自然对数频带功率，两项相对属性分别除以 1 至 45 Hz 总功率。')
        +p('数值下限为输入尺度平方坐标下的 10⁻¹²。每项属性分别按本库训练完整片段的均值和总体标准差标准化。掩码不完整的片段不参加频谱评分；电压评分仍使用有效采样点。')
        +h('先冻结读出，再给未见被试打分')
        +p('背景表按数据集、电极和可见 X 的均方根幅度档分组，幅度按训练输入 SD 缩放，切点为 0.5、1、2。每个格内先求个人均值，再让训练被试等权。第二张表额外加入真实发出的单个 S 词。')
        +p('若“背景＋ID”在训练中有 n 人支持，则按 n/(n+4) 保留格均值相对背景的偏移；未见组合回到背景，未见背景回到训练人均值。三种模型都用这一独立读出评分。C 的辅助头不参与主终点评分。')
        +h('怎样得到百分比和波形护栏')
        +p('先汇总每个属性的均方误差（MSE）：完整片段在位置内平均，再在时间窗内平均位置，随后依次在记录、被试和库内平均，最后四库等权。单属性增益为 100% ×（1 − 加入 ID 后的宏 MSE / 仅背景的宏 MSE）；四项等权得到主终点。')
        +p('波形 NMSE 使用对应 X、S、O 在本库本位置的冻结训练方差作分母。各分量分层平均后，三项等权得到宏 NMSE。Pearson 相关在窗口×位置内计算，再作相同层级汇总。相对 A 的容差为宏 NMSE 最多增 2%、单分量最多增 5%、单分量相关最多降 0.01。'),
        'score_single_vq_spectral_location.py；frozen_token_readout.py；base_training_aggregate')
    page('单 ID 属性','θ 对数功率贡献了两次训练中最大的增量',
        p('每个点代表一个模型对一项教师属性的开发集宏增益。左图零线表示与背景读出持平，右图零线表示与同种子 A 持平；向右表示改善。'),
        '各臂 evaluation/summary.json；数据表 attributes.csv','attribute_gain',
        table(['C 相对 A（百分点）',*F],[[seed,*[f'{v:+.3f}' for v in
                    100*(np.array(g[seed,'C']['id_gain'])-g[seed,'A']['id_gain'])]] for seed in (17,43)])
        +p('θ 对数功率、α 对数功率和 θ 占比都在两个种子上获益。α 占比在种子 17 略降、在种子 43 上升。首轮 C 比 B 的四属性平均增益高 0.166 个百分点。'))
    def contrast_rows():
        for phase in [result['first_seed'],result['repeat_seed']]:
            for contrast,r in phase['contrasts'].items():
                x=np.array([np.mean(v['gain_delta']) for v in r['subjects']])*100
                yield [phase['seed'],contrast.replace('-minus-',' − '),f"{r['mean_gain_delta']*100:+.3f}",
                       f'{sum(x>0)} / {len(x)}',f'{x.min():+.3f} 至 {x.max():+.3f}']
    page('逐人配对','C 的增益在首轮 13 人、复核 16 人中为正',
        p('同一横坐标是同一名被试，两种符号为两个种子的 C−A。每个点先用该人的误差求四项增益，再取均值；零线以上表示该人的单 ID 读出改善。'),
        'summary.json 的 contrasts.subjects；数据表 paired_subjects.csv、dataset_gains.csv','paired_subjects',
        table(['种子','对比','宏增量（百分点）','个人正增量','个人范围（百分点）'],contrast_rows())
        +p('宏增量先按库等权汇总误差再求比值，因此与图中 16 点的算术均值可不同。两种子使用同一批人。预设筛查采用工程阈值，此处仅作配对描述。')
        +p('分库汇总时，C−A 在 Single-Trial 两轮分别为 +0.355、+0.320 个百分点。Visual 首轮为 −0.036，复核为 +0.153；另两库在两个种子上方向均为正。'))
    page('解码与波形','B 恢复了更多频谱细节，S 波形相关却下降',
        p('首轮四项解码属性误差均在 B 最低。左图每个点为从解码 S 波形重算属性后，相对教师的标准化 MSE，越低越好；右图越高越好。线段仅连接同一条件的结果。'),
        '各臂 evaluation/summary.json；selection.json；数据表 models.csv、attributes.csv','decoded_waveform',
        table(['种子 / 模型','X NMSE','S NMSE','O NMSE','X 相关','S 相关','O 相关'],
              [[f'{seed} / {a}',*[f'{v:.4f}' for v in [*macro(s,'nmse'),*macro(s,'correlation')]]] for (seed,a),s in g.items()])
        +p('B 的宏 NMSE 相对 A 上升 1.22%，仍在 2% 容差内；S 相关从 0.5278 降至 0.5116，降幅 0.0162 超过 0.01，且单 ID 主终点未改善。C 的宏 NMSE 两轮分别上升约 0.069% 和 0.020%，各分量误差及相关均在护栏内。'))
    page('词表覆盖与分散','C 让同词成员更集中，覆盖范围没有同步扩大',
        p('训练支持定义为某个 S 词至少出现于 10 个完整训练片段且覆盖 3 名训练被试。词表比例以 8,192 个可用词为分母。片段覆盖率按 780,210 个开发词的原始出现次数计算，高频词贡献更多。')
        +table(['种子 / 模型','开发出现词数','训练支持词数','占词表','受支持开发片段','开发片段占比'],
               [[f'{seed} / {a}',f"{s['emitted_development_ids']:,}",f"{s['supported_ids']:,}",f"{100*s['supported_fraction_vocabulary']:.2f}%",
                 f"{s['supported_development_tokens']:,}",f"{100*s['supported_development_tokens']/s['development_complete_patches']:.2f}%"] for (seed,a),s in g.items()])
        +p('C 相对 A 的支持词数减少 83 和 109 个；受支持开发片段占比变化分别为 +0.13 和 −0.15 个百分点。高覆盖说明多数片段落入有训练支持的词，每个词是否稳定代表某种模式还需检查成员分布。')
        +p('训练支持还取决于背景。C 的开发片段中，种子 17 有 '
           +f"{100*local['unseen_joint_tokens']/local['development_complete_patches']:.1f}%"
           +'、种子 43 有 '+f"{100*loc2['unseen_joint_tokens']/loc2['development_complete_patches']:.1f}%"
           +' 的“背景＋ID”组合未在训练出现，这些片段在主读出中直接使用背景值。')
        +h('四项属性的词内方差均下降')
        +p('按实际发出的 ID 为开发片段分组，先算每词内部标准化属性的总体方差，再按开发词频加权。下表在同一面板、同一尺度上比较 A 与 C，数值越小，组内越集中。')
        +table(['种子 / 模型',*F],[[f'{seed} / {a}',*[f'{v:.4f}' for v in s['within_id_variance']]] for (seed,a),s in g.items()])
        +p('以种子 43 的 θ 对数功率为例，C 的词内方差从 0.565 降至 0.547，词间均值方差从 0.322 升至 0.340。仅用 ID 的训练均值查表，其开发 MSE 从 0.630 降至 0.612；加入背景后，两者分别为 0.406 和 0.405。词内分散、单词读出和背景条件增益使用不同分母。'),
        '各臂 evaluation/summary.json 和 id_dispersion.npz；数据表 models.csv、attributes.csv')
    dev_rows=[r for r in word_rows if r[0]=='validation']
    page('词的真实成员','两个训练期选出的词在开发片段中仍有频谱偏好',
        p(f'用 C 的复核种子 43 说明实际词义。先在训练面板保留至少出现 100 次的 {n_candidates} 个词，再分别选平均 θ、α 功率占比最高者，得到 S{picks[0]} 和 S{picks[1]}。这项事后展示使用训练属性选词，开发结果只用于描述。'),
        'seed_43/C/evaluation/*_train.npz 与 *_validation.npz；calibration.json；word_examples.csv','word_members',
        table(['开发成员','完整片段数','覆盖被试','θ 占比中位数','α 占比中位数'],[[f'S{w}' if w!='全体' else w, f'{n:,}', str(n_people)+' 人',
                f'{md:.2f}%',f'{dev_rows[i+1][5]:.2f}%'] for i,(_,w,_,n,lo,md,hi,n_people) in enumerate(dev_rows) if i%2==0])
        +p('菱形为中位数，线段为成员片段的第 10–90 百分位。用各库冻结训练尺度还原为相对 1–45 Hz 的功率百分比。全体行汇集全部开发片段，未按电极、幅度或被试匹配。两词的成员范围与全体有重叠，选例不能估计全词表的事件纯度。'))
    evidence_cases=read(OLD/'tables/wave_eeg.json')
    for n,(case,title) in enumerate([('median','中位窗口中，C 与 A 的波形误差接近'),('difficult','困难窗口中，三种模型都压低了大幅瞬态')]):
        c=evidence_cases[n]
        f=RUN/'seed_17/A/evaluation/simultaneous_eeg_nirs_fixed_windows.json'
        assert read(f)[case]['example'][:2]==[c['row'],c['start_s']]
        detail=('C 重建的 X 正峰为 2.85 个输入 SD，观测峰为 4.36。S 的波形起伏也偏小，C 的标准差为 0.217，教师为 0.329；其 S 误差略低于 A，X 误差略高。'
                if n==0 else '观测 X 正峰为 9.85 个输入 SD，C 的重建正峰仅 1.20。教师 O 正峰为 7.88，C 为 0.86。三组 O 的 NMSE 均超过 2.27；C 守住总体重建护栏，在这个窗口仍有明显幅值损失。')
        page('冻结波形样例',title,
            p(f"Simultaneous · {c['subject']} · {c['record']} · {c['position']}，母段内第 {c['start_s']}–{c['start_s']+30} 秒。历史容量基线在该库 {c['support']:,} 个完整窗口×位置上按 X/S/O 平均 NMSE 选最接近第 {50 if n==0 else 95} 百分位者，本轮直接沿用。"),
            '容量报告 tables/wave_eeg.json；各臂 evaluation/'+case+'_waveforms.npz；fixed_windows.csv',
            'wave_'+case,
            p('黑实线在 X 行为观测、在 S/O 行为教师分量。彩色线均为种子 17 的整窗整数解码。左右共享本行纵轴；各行、各样例独立缩放。SD 指本电极训练 X 的标准差。')
            +table(['种子 17 · 窗口 NMSE','X','S','O'],[[a,*[f'{v:.3f}' for v in read(RUN/f'seed_17/{a}/evaluation'/f.name)[case]['nmse']]] for a in 'ABC'])
            +p(detail))
    doc.set_metadata(dict(title='EEG-S 单级 VQ 频谱损失位置实验结果',author='生理信号 tokenizer 项目',subject='四库配对开发评分与独立种子复核'))
    doc.set_toc([[1,f'{i:02d}　{t}',i] for i,t in enumerate(['结果总览','实验设计','指标与读出','单 ID 属性',
        '逐人配对','解码与波形','词表覆盖与分散','词的真实成员','中位窗口','困难窗口'],1)])
    name=out/'EEG-S_频谱损失位置_结果报告.pdf'
    doc.subset_fonts()
    doc.save(name,garbage=4,deflate=True);doc.close()
    return name


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='Fresh versioned export directory')
    out=parser.parse_args().output
    out.mkdir(parents=True,exist_ok=False)
    fig,tab=out/'figures',out/'tables'
    fig.mkdir();tab.mkdir()
    result,groups=evidence()
    export_tables(groups,result,tab)
    words,picks,n=portraits(tab)
    with tempfile.TemporaryDirectory() as temp:
        work=Path(temp)
        sc=work/'noto_sc.otf'
        TTFont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',fontNumber=2).save(sc)
        font_manager.fontManager.addfont(sc)
        plt.rcParams.update({'font.family':font_manager.FontProperties(fname=sc).get_name(),'font.size':9.7,
            'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#a5b3b7','axes.labelcolor':'#253d44',
            'text.color':'#253d44','axes.titleweight':'regular','axes.unicode_minus':False})
        figures(groups,result,words,fig,tab)
        pdf=make_report(out,groups,result,words,picks,n,work)
    with fitz.open(pdf) as d:
        images=[len(pg.get_image_info()) for pg in d]
        assert images==[0,1,0,1,1,1,0,1,1,1] and all(len(pg.get_text())>200 for pg in d)
    with zipfile.ZipFile(out/'图表与配对数据.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted([*fig.glob('*.png'),*tab.glob('*.csv')]):
            z.write(p,p.relative_to(out))
    print(pdf,images)


if __name__=='__main__':
    main()
