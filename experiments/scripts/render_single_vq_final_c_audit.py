"""Render a versioned Chinese visual audit from frozen-C scores and aligned examples."""
import argparse
from html import escape
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import torch

import run_single_vq_final_c_audit as audit
from run_dual_component_tokenizer import read_json, write_json

ROOT=audit.OUT
OLD=audit.relations.OUT
SEEDS=audit.SEEDS
BLUE,RED='#155684','#b84b29'
COLORS=['#dfedf4','#aad9d3','#efcb83','#ac6677']
DS={'simultaneous_eeg_nirs':'Simultaneous','eeg_fnirs_single_trial':'Single-Trial',
    'visual_cognitive_motivation':'Visual','refed':'REFED'}


def load(name):
    return pd.read_csv(ROOT/name)


def families(mod,seed,b):
    return np.array(read_json((ROOT if mod=='eeg' else OLD)/f'{mod}_{seed}'/f'{b}_family.json')['mapping'])


def strip(ax,seqs,labels,mod='eeg',seed=17,start=0):
    for j,(ids,b) in enumerate(seqs):
        family=families(mod,seed,b)[ids]
        for i,(word,f) in enumerate(zip(ids,family)):
            lo=start+2*i
            ax.add_patch(plt.Rectangle((lo,j-.44),2,.88,facecolor=COLORS[f],edgecolor='white'))
            ax.text(lo+1,j,str(word),ha='center',va='center',fontsize=8,color='#101923')
    ax.set(xlim=(start,start+30),ylim=(len(seqs)-.5,-.5),yticks=range(len(labels)),yticklabels=labels)
    ax.set_xlabel('窗口时间（秒）' if start==0 else '相对提示（秒）')
    ax.set_xticks(np.arange(start,start+31,2))
    ax.spines[['top','right','bottom','left']].set_visible(False)


def palette():
    return '格内为真实整数 ID。底色从浅蓝、青、黄到紫依次为训练属性四分位词族 Q1–Q4；每个分支独立分族。'


class Report:
    def __init__(self,out):
        self.out=out
        (out/'figures').mkdir()
        self.pdf=fitz.open()
        self.html=[]
        self.titles=[]

    def page(self,title,intro,fig,caption,source,slug):
        dest=self.out/'figures'/f'{slug}.png'
        fig.savefig(dest,dpi=230,facecolor='white')
        plt.close(fig)
        pg=self.pdf.new_page(width=936,height=690)
        def box(rect,html,size):
            spare,scale=pg.insert_htmlbox(fitz.Rect(rect),html,css=f'*{{font-family:sans-serif;color:#203a46;font-size:{size}pt;line-height:1.38}}',scale_low=1)
            if spare<0:
                raise ValueError(f'text overflow {slug}: {html}')
        box((27,13,908,35),f'最终 C · 冻结开发审计　　 /　　 2026-11-25　　 /　　 {len(self.titles)+1:02}',10)
        box((27,38,908,77),escape(title),21)
        box((27,82,908,142),escape(intro),13)
        pg.insert_image(fitz.Rect(25,144,911,530),filename=str(dest),keep_proportion=True)
        box((27,539,909,649),escape(caption),12.5)
        box((27,652,914,683),escape(source),9)
        self.titles.append(title)
        self.html.append(f'<section id="{slug}"><h2>{escape(title)}</h2><p>{escape(intro)}</p><a href="figures/{slug}.png"><img src="figures/{slug}.png"></a><p>{escape(caption)}</p><small>{escape(source)}</small></section>')

    def finish(self):
        self.pdf.set_toc([[1,t,i] for i,t in enumerate(self.titles,1)])
        self.pdf.subset_fonts()
        self.pdf.save(self.out/'REPORT.pdf',garbage=4,deflate=True)
        self.pdf.close()
        nav=''.join(f'<li><a href="#{s.split(chr(34))[1]}">{escape(t)}</a></li>' for t,s in zip(self.titles,self.html))
        (self.out/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>最终 C 开发审计</title><style>body{font:17px/1.65 sans-serif;color:#203a46;max-width:1160px;margin:auto;padding:30px}section{border-top:1px solid #ccd8da;padding:25px 0}img{width:100%}small{color:#56666b}a{color:#155684}</style><h1>最终 C 开发审计</h1><ol>'+nav+'</ol>'+''.join(self.html)+'</html>',encoding='utf8')


def subplots(n=1,w=1,**kw):
    return plt.subplots(n,w,figsize=(12.5,5.25),layout='constrained',**kw)


def windows(report,example):
    for c in read_json(example/'windows.json'):
        mod=c['mod']; is_eeg=mod=='eeg'
        tag='中位' if 'median' in c['name'] else '困难'
        channels=1 if is_eeg else 2
        with np.load(example/(c['name']+'.npz')) as z:
            for ch in range(channels):
                color='EEG' if is_eeg else ('HbO','HbR')[ch]
                fig=plt.figure(figsize=(13.2,5.8),layout='constrained')
                gs=fig.add_gridspec(4,2,height_ratios=[1,1,1,.95],width_ratios=[3,1.2])
                for j,(target,rec,label) in enumerate([('x','sum','X 实测'),('s','S','S 教师'),('o','O','O 教师')]):
                    y,p=[z[k].reshape(-1,channels)[:,ch] for k in (target,rec)]
                    t=np.arange(len(y))/(200 if is_eeg else 10)
                    peak=14
                    lim=(min(y.min(),p.min()),max(y.max(),p.max()))
                    margin=max((lim[1]-lim[0])*.07,.001)
                    for col in range(2):
                        ax=fig.add_subplot(gs[j,col])
                        ax.plot(t,y,color=BLUE,lw=.8,label=label)
                        ax.plot(t,p,color=RED,lw=.85,ls='--',label='整数解码')
                        ax.set_xlim((0,30) if col==0 else (peak,peak+2))
                        ax.set_ylim(lim[0]-margin,lim[1]+margin)
                        if col==0:
                            ax.set_ylabel(label+'\n训练 SD')
                            ax.axvspan(peak,peak+2,color='#e3c98d',alpha=.3,zorder=-5)
                        if j==0:
                            ax.set_title('完整 30 秒 · 蓝实线为目标，橙虚线为重建' if col==0 else '固定放大第 8 个词 · 14–16 秒')
                        if j<2: ax.tick_params(labelbottom=False)
                ax=fig.add_subplot(gs[3,0])
                strip(ax,[(z[b+'_id'],b) for b in 'SO'],['S 编码','O 编码'],mod)
                key=fig.add_subplot(gs[3,1]);key.axis('off')
                key.legend(handles=[plt.Rectangle((0,0),1,1,color=k,label=f'Q{i+1}') for i,k in enumerate(COLORS)],
                           title='训练属性词族 · 从低到高',ncol=4,loc='center',fontsize=8,columnspacing=.7,handlelength=1)
                v=np.array(c['nmse'])[:,ch]
                observed=z['x'].reshape(-1,channels)[:,ch]
                predicted=z['sum'].reshape(-1,channels)[:,ch]
                peak_index=int(np.argmax(np.abs(observed)))
                peak_time=peak_index/(200 if is_eeg else 10)
                caption=(f'观测 X 的最大绝对幅值出现在 {peak_time:.2f} 秒，该点观测为 {observed[peak_index]:+.2f}、重建为 {predicted[peak_index]:+.2f} 个训练 SD。'
                         f'X、S、O 的整窗归一化均方误差（NMSE）依次为 {v[0]:.3f}、{v[1]:.3f}、{v[2]:.3f}。'
                         '误差按本位置该目标的训练方差归一化，纵轴统一为本位置观测 X 的训练标准差单位。左右图每行共用纵轴，各行独立。'+palette())
                report.page(f'{color} {tag}窗口的观测、重建和两条 token 序列',
                    f'{DS[c["dataset"]]} · {c["subject"]} · {c["record"]} · {c["position"]} · 记录起点后 {c["native_start_s"]:.2f} 秒。沿用历史容量面板在该库 {c["support"]:,} 个完整窗口×位置中按基准误差选取的第 {50 if tag=="中位" else 95} 百分位样例。',fig,caption,'示例数组 examples/'+c['name']+'.npz；配对身份、原始时标与分母见 examples/windows.json。',c['name']+'_'+color)


def readouts(report):
    attrs=['power_4_8','power_8_13','rms_envelope','patch_rms']
    labels=['θ 功率','α 功率','250 ms 包络','2 s RMS']
    for b in 'SO':
        fig,axes=subplots(1,2)
        for ax,seed in zip(axes,SEEDS):
            tab=load(f'readout_{seed}_{b}/scores.csv').query("dataset=='macro'").pivot(index='attribute',columns='method',values='gain').loc[attrs]
            x=np.arange(4)
            for offset,(m,label,col) in enumerate([('shrunk','条件 ID','#92bec7'),('balanced_shrunk','被试平衡 ID','#deab63'),('embedding','码向量岭回归',BLUE)]):
                values=100*tab[m]
                bars=ax.bar(x+(offset-1)*.25,values,width=.24,label=label,color=col)
                ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
            ax.set(xticks=x,xticklabels=labels,ylabel='相对背景的 MSE 降幅（%）',title=f'{b} 分支 · 种子 {seed}')
            ax.axhline(0,color='gray',lw=.7);ax.margins(y=.15);ax.legend(fontsize=9)
        report.page(f'{b} 分支的条件读出与码向量读出', '同一 2 秒教师目标，在数据集、电极和可见 X 的 RMS 分箱相同的背景下评分。包络取八段 250 ms RMS，先平均八个坐标的平方误差。两个种子在同一批开发人上评价。',fig,
            '柱高为 100 ×（1 − 方法 MSE / 对应背景 MSE），正值表示误差下降。类别读出将训练中的背景×ID 均值向背景收缩，被试平衡版在每格先按人平均；码向量读出对背景残差拟合线性岭回归。各属性的强度由训练被试三折选取。先均有效片段，再等权汇总位置、窗、记录、被试和库。',
            f'readout_17_{b}、readout_43_{b} 下的 scores.csv、selection.json、support.json；decoded 行为直接解码属性误差。',f'readout_{b}')


def members(report):
    for seed in SEEDS:
        fig,axes=subplots(2,2)
        for j,b in enumerate('SO'):
            base=ROOT/f'eeg_{seed}'
            w=pd.read_csv(base/f'{b}_words.csv')
            m=pd.read_csv(base/f'{b}_members.csv').query("split=='validation'")
            ax=axes[0,j]
            metrics=[w.eligible.mean(),w.readable.mean(),w.loc[w.eligible,'dev_patches'].sum()/w.dev_patches.sum(),w.loc[w.readable,'dev_patches'].sum()/w.dev_patches.sum()]
            bars=ax.barh(['训练支持 · 词','可读 · 词','训练支持 · 片段','可读 · 片段'],100*np.array(metrics),color=[BLUE,RED,BLUE,RED])
            ax.bar_label(bars,fmt='%.1f%%',padding=3);ax.set(xlim=(0,115),xlabel='占全部词或开发片段的百分比',title=f'{b} 分支 · 全词表与实际发词')
            ax=axes[1,j]
            for i,tag in enumerate(('low','middle','high','heldout_failure')):
                g=m[m.group==tag]
                if not len(g):continue
                word=int(g.word.iloc[0])
                for h,(kind,offset,color,marker) in enumerate([('member',-.12,BLUE,'o'),('background',.12,RED,'s')]):
                    v=g[g.kind==kind].branch_attribute.to_numpy()
                    lo,mid,hi=np.quantile(v,[.1,.5,.9])
                    ax.plot([lo,hi],[i+offset]*2,color=color,lw=2)
                    ax.plot(mid,i+offset,marker=marker,color=color,label=('同 ID','匹配背景')[h] if i==0 else None)
                ax.text(.99,i,f'ID {word}',transform=ax.get_yaxis_transform(),ha='right',fontsize=9)
            ax.set(yticks=range(4),yticklabels=['低频','中频','高频','失败词'],xlabel='教师分量 α 功率占比',xlim=(0,1))
            ax.legend(fontsize=8)
        report.page(f'真实成员的属性分布和失败词 · 种子 {seed}','上排比较 8192 个词与开发片段的覆盖率。下排的点和线段分别是匹配样本的中位数和第 10–90 百分位。每个示例词最多抽 128 个开发成员，与同人、同位置、同 RMS 分箱的其他词配对。',fig,
            '训练支持要求至少 20 个片段和 3 人。可读词还要求至少 3 名开发人、逐人平均误差改善且过半数人改善，覆盖率包含开发筛选。前三个词按合格词的训练频数第 10、50、90 百分位选择；失败词是有开发支持且平均误差恶化最大的词。词内分布和匹配背景的重叠保留在图中。',f'eeg_{seed} 下的 S/O_words.csv、S/O_members.csv、coverage.csv；逐个真实成员波形见后页。',f'member_{seed}')
        for b in 'SO':
            folder=ROOT/f'eeg_{seed}'
            frame=pd.read_csv(folder/f'{b}_members.csv')
            fig,axes=subplots(2,2)
            for ax,(tag,g) in zip(axes.flat,frame.groupby('group',sort=False)):
                word=int(g.word.iloc[0])
                with np.load(folder/f'{b}_member_waves.npz') as z:
                    k=f'{tag}_{word}_'
                    train=z[k+'train_member'][:,1+'SO'.index(b)]
                    dev=z[k+'validation_member'][:,1+'SO'.index(b)]
                    bg=z[k+'validation_background'][:,1+'SO'.index(b)]
                t=np.arange(400)/200
                ax.plot(t,dev[:3].T,lw=.75,color='#96a9b6',alpha=.65)
                ax.plot(t,train.mean(0),'--',color=BLUE,label='训练成员均值')
                ax.plot(t,dev.mean(0),color=RED,label='开发成员均值')
                ax.plot(t,bg.mean(0),':',color='#437855',label='开发背景均值')
                ax.set(xlabel='片段时间（秒）',ylabel=f'教师 {b}（训练 SD）',title=f'{dict(low="低频",middle="中频",high="高频",heldout_failure="失败")[tag]}词 {word} · 配对开发成员 {len(dev)} 段')
            axes.flat[0].legend(ncol=3,fontsize=7)
            report.page(f'{b} 词中的真实 2 秒成员 · 种子 {seed}', '细灰线是固定随机样本中前三个真实开发成员；粗线是该组成员或配对背景的均值。每幅图沿用实际幅度，独立纵轴。',fig,'跨片段相位差会压低平均波形的振幅。灰线显示单个成员仍可有明显起伏；判定词的属性偏好还需结合上一页的成员分布及全体支持量。各例对应的实测 X 与时间中心化视图也保存在同一成员目录。',f'eeg_{seed}/{b}_member_waves.npz、{b}_members.csv、{b}_wave_template.csv。',f'real_{seed}_{b}')


def context_pages(report):
    frame=load('context.csv')
    support=load('context_support.csv')
    fig,axes=subplots(1,2,sharey=True)
    for ax,(b,g) in zip(axes,frame.groupby('branch',sort=False)):
        for i,(seed,v) in enumerate(g.groupby('seed')):
            z=v.groupby(['dataset','subject'])[['id_same','family_same','family_donor_same']].mean()
            for j,metric in enumerate(z):
                y=100*z[metric]; x=j+(i-.5)*.24
                ax.scatter(np.full(len(y),x),y,color=[BLUE,RED][i],alpha=.34,s=18)
                ax.plot(x,100*z[metric].groupby(level=0).mean().mean(),'D',color=[BLUE,RED][i],label=f'种子 {seed}' if j==0 else None)
        ax.set(xticks=range(3),xticklabels=['同中心 ID','同中心词族','供体自身词族'],ylim=(-3,105),ylabel='一致率（%）',title=b+' 分支')
        ax.legend()
    report.page('固定中心 2 秒，周围 28 秒能改变多少编码？',
        f'在 {len(support)} 名开发人共 {support.complete.sum():,} 个完整窗口×位置中，{support.eligible.sum():,} 个有同人、同记录、同电极且时间不重叠的供体。固定随机种子，每人最多抽 32 个，共 {support.used.sum()} 对。原始中心保留在第 8 格，两侧均换为该供体上下文。',fig,
        'S 的 ID 一致率为 43.3% / 56.0%，词族为 70.1% / 78.1%；O 的对应值为 72.7% / 69.6% 与 86.3% / 88.5%。淡点是一人，菱形是先按人、再按库等权的均值。前两列分别比较替换前后的整数与训练词族，第三列比较原窗口中心与供体自身中心的词族。原信号重复编码的一致率为 100%。替换可能在拼接处产生不连续，测到的是此扰动下的上下文敏感度。','context_support.csv、context.csv。逐窗保留原词、新词、词族一致性与训练属性均值漂移。','context_statistics')
    with np.load(ROOT/'context_example_17.npz') as z:
        fig=plt.figure(figsize=(13,5.4),layout='constrained')
        axes=fig.subplots(3,1,gridspec_kw={'height_ratios':[1,1,1.4]})
        for ax,k,label in zip(axes[:2],['x','swap'],['原窗口','替换两侧后']):
            ax.plot(np.arange(6000)/200,z[k].ravel(),color=BLUE,lw=.65)
            ax.axvspan(14,16,color='#e9b945',alpha=.4)
            ax.set(xlim=(0,30),ylabel=label+'\n训练 SD')
        lim=(min(a.get_ylim()[0] for a in axes[:2]),max(a.get_ylim()[1] for a in axes[:2]))
        for ax in axes[:2]:ax.set_ylim(lim)
        strip(axes[2],[(z[k+'_'+b],b) for b in 'SO' for k in ('original','changed')],['原 S','替换 S','原 O','替换 O'])
        r=frame.iloc[0]
        report.page('逐样点相同的中心在两套上下文中如何发词',
            f'固定随机顺序的第一对 · {DS[r.dataset]} · {r.subject} · {r.record} · {r.site}。两条实测输入波形用同一纵轴；黄色区为同一个 2 秒片段。',fig,
            '在第 8 格比较上下两行，即可观察同一中心是否换词。其他格同时改变了局部波形与上下文。'+palette(),'context_example_17.npz、context.csv；此页为机制示例，发生比例见前页。','context_trace')


def relationships(report,example):
    fig,axes=subplots(1,2)
    for ax,seed in zip(axes,SEEDS):
        df=load(f'eeg_{seed}/conditional.csv')
        main=load(f'eeg_{seed}/interaction_vs_family_main.csv')
        for j,(target,g) in enumerate(df.groupby('target',sort=False)):
            raw=g.iloc[[2,3]]; last=main[main.target==target].iloc[0]
            y=np.r_[raw.gain,last.gain]*100
            bars=ax.bar(np.arange(3)+(j-1)*.25,y,.23,label=target)
            ax.bar_label(bars,fmt='%.2f',fontsize=7,padding=2)
        ax.set(xticks=range(3),xticklabels=['S → S+O','O → S+O','主效应 → 加交互'],ylabel='相对参照的误差降幅（%）',title=f'种子 {seed}')
        ax.legend(title='被预测的 α 占比');ax.axhline(0,color='gray',lw=.7);ax.margins(y=.18)
    report.page('O 为实测 X 的 α 占比提供小幅额外信息','输出分别取实测 X、教师 S、教师 O 的 α 占比。背景固定为库、电极和可见 X 的 RMS 分箱。先加入单个分支的码向量，再加入另一分支；最后一组在两个分支和各自词族主效应上加入词族交互。',fig,
        '对实测 X，S 后加入 O 的误差下降 1.35% / 1.11%（种子 17 / 43），分别有 13 / 16、12 / 16 人改善。进一步控制两个词族的非线性主效应后，交互增益为 −0.005% / +0.020%，两组配对误差差值的区间均跨零。Hb 沿用原模型，同口径的实测 HbO 斜率条件增益为 40.80% / 37.29%。',
        'eeg_17、eeg_43 下的 conditional.csv、interaction_vs_family_main.csv；Hb 引自 single_vq_relations 的 hb_17、hb_43。','complement')
    trans=pd.read_csv(example/'transitions.csv')
    fig,axes=subplots(1,4)
    for ax,((seed,b),g) in zip(axes,trans.groupby(['seed','branch'],sort=False)):
        mat=g.pivot(index='source',columns='dest',values='prob').values
        im=ax.imshow(mat,cmap='Blues',vmin=0,vmax=1)
        for i,j in np.ndindex(4,4):
            ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',color='white' if mat[i,j]>.55 else '#16394d')
        ax.set(xticks=range(4),yticks=range(4),xticklabels=range(1,5),yticklabels=range(1,5),xlabel='下一词族',ylabel='当前词族',title=f'{b} · 种子 {seed}')
    fig.colorbar(im,ax=axes.tolist(),shrink=.57,label='条件概率')
    report.page('开发窗口中，四个词族怎样连续出现',f'在 {trans.windows.iloc[0]:,} 个 15 格均有效的开发窗口×位置内，逐一计数 14 对相邻词族。行表示当前族、列表示下一族；每行和为 1。四幅图共享色标，数字为条件概率。',fig,
        '词族 Q1–Q4 按训练时教师分量的 α 占比由低到高分组。对角线表示连续两个词落在同一族，也可以是不同整数。统计按实际转移次数加权；同一被试和同一时间的多个电极产生多个转移。图描述相邻结构，顺序的预测价值见下一页。','examples/transitions.csv；词族映射见各模型 S_family.json 和 O_family.json。','transitions')
    fig,axes=subplots(1,2,sharey=True)
    for ax,adj in zip(axes,[False,True]):
        labels=[]; series=[]
        for b in 'SO':
            for ref in ['bag','bigram_dwell','motif_bag_null','motif_transition_null','motif_dwell_null']:
                labels.append(b+' · '+dict(bag='转移驻留 − 词袋',bigram_dwell='三词 − 转移驻留',motif_bag_null='三词 vs 保词频',motif_transition_null='三词 vs 保转移',motif_dwell_null='三词 vs 保驻留')[ref])
                points=[]
                for seed in SEEDS:
                    tab=load(f'eeg_{seed}/sequence.csv')
                    row=tab[(tab.branch==b)&(tab.adjacent==adj)&(tab.reference==ref)].iloc[0]
                    points.append(row.gain*100)
                series.append(points)
        for i,seed in enumerate(SEEDS):
            ax.scatter(np.array(series)[:,i],np.arange(len(labels))+(i-.5)*.23,label=str(seed),color=[BLUE,RED][i],marker=['o','s'][i])
        ax.axvline(0,color='gray',lw=.8)
        ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='候选相对参照的误差降幅（%）',title='相邻不重叠窗' if adj else '同一 30 秒窗')
        ax.legend(title='种子',fontsize=8)
    axes[0].invert_yaxis()
    report.page('加入顺序和三词计数，EEG-S 的预测误差未下降',
        '用 14 个训练标定词族预测实测 X 的 α 占比。左侧取前 14 格预测本窗第 15 格，编码器已见全窗。右侧取本窗后 14 格预测同记录同电极下一不重叠窗的第一格。两侧均只用训练人拟合岭回归。',fig,
        '相邻窗中，S 的转移＋驻留相对词袋使误差升高 4.71% / 3.17%，再加三词计数又升高 8.24% / 9.08%。相邻训练配对仅 629 个窗口×位置，开发配对 28,106 个。后三行对照使用已拟合的三词模型，每窗 10 次置换，分别保留词频、有向转移或分族驻留结构。','eeg_17、eeg_43 下的 sequence.csv、sequence_subject.csv；gain 正值表示真实候选优于参照。','order')
    fig,axes=subplots(1,2)
    for ax,adj in zip(axes,[False,True]):
        for i,(mod,seed,col) in enumerate([('eeg',17,BLUE),('eeg',43,RED),('hb',17,'#467754'),('hb',43,'#925c89')]):
            base=ROOT if mod=='eeg' else OLD
            model=pd.read_csv(base/f'{mod}_{seed}/sequence_subject.csv')
            measured=pd.read_csv(base/f'{mod}_{seed}/observed_sequence_reference.csv')
            sources=[model[(model.branch==b)&(model.adjacent==adj)] for b in 'SO']+[measured[measured.adjacent==adj]]
            y=[]
            for g in sources:
                macro=g.groupby(['method','dataset']).mse.mean().groupby(level=0).mean()
                y.append(100*(1-macro.bigram_dwell/macro.bag))
            ax.plot(range(3),y,['o-','s--','^-','D--'][i],color=col,label=f'{mod.upper()} {seed}')
        ax.axhline(0,color='gray',lw=.7)
        ax.set(xticks=range(3),xticklabels=['S 词族','O 词族','实测 X 分箱'],ylabel='加入转移与驻留的误差降幅（%）',title=['同窗','相邻独立窗'][int(adj)])
        ax.legend(fontsize=8)
    report.page('实测属性历史与 token 历史的同口径顺序对照','连续参照先用训练 X 属性的四分位阈值，把每个实测 2 秒片段分为四档，然后采用相同历史长度、描述符和训练回归。EEG 预测 α 占比，Hb 预测 HbO 斜率。',fig,'每个点都是相对该输入自身词袋基线的误差降幅。实测参照与 tokenizer 种子无关，同模态两点重合。Hb-S 在相邻窗口中仍有较小的低阶顺序增益；EEG-S 的结果在两个 C 种子上方向一致。','各模型的 observed_sequence_reference.csv、sequence_subject.csv；Hb 为沿用证据。','raw_sequence')


def event_pages(report,example):
    logs=load('events_17/event_support.csv').query("split=='validation'")
    upstream=load('events_17/upstream_event_exclusions.csv').query("split=='validation'")
    pairs=load('events_17/coupling_pairs.csv').query("split=='validation' and donor_event>=0")
    used=logs[logs.B=='used']
    paired=used[used.groupby(['subject','record']).condition.transform('nunique')==2]
    plan=pd.concat([logs,upstream])
    ledger=pd.DataFrame({k:v.groupby('subject').size() for k,v in [('计划',plan),('上游 QC 后',logs),('事件双条件',paired),('Hb 错配齐全',pairs)]}).fillna(0).astype(int)
    ledger.to_csv(ROOT/'development_event_ledger.csv')
    fig,ax=subplots()
    ledger.plot.bar(ax=ax,color=['#9faeb4','#7694aa','#267f89','#bf713c'],rot=0)
    for bars in ax.containers:ax.bar_label(bars,fontsize=9,padding=1)
    ax.set(ylabel='试次数',xlabel='Single-Trial 开发被试',ylim=(0,73));ax.legend(ncol=4)
    report.page('5 名计划被试中，3 人能进入配对事件分析','Single-Trial 有 5 名开发被试、每人 60 个心算或显式基线提示。四库的 16 名开发人是 tokenizer 面板总人数；独立事件与实测 Hb 只使用含对应提示、区域和记录的 Single-Trial。',fig,
        f'计划 300 次。01、03 号各 60 次及 14 号 48 次被上游同步质控排除，剩 {len(logs)} 次；其中 {len(used)} 次具备完整事件窗，{len(paired)} 次属于同一会话有两种条件的配对支持，覆盖 {len(paired.groupby(["subject","record"]))} 个会话。另一路 Hb 预测有 {(logs.D=="candidate").sum()} 次通过对齐和移位初筛，最终 {len(pairs)} 次可匹配同条件供体。两条终点分别筛选。','development_event_ledger.csv、events_17/event_support.csv、upstream_event_exclusions.csv、coupling_pairs.csv；两个种子身份一致。','event_population')
    fig,axes=subplots(2,2)
    phases=['pre','entry','active','exit','recovery']; ticks=['−4–0','0–2','2–10','10–18','18–24']
    for ax,(seed,field,title,factor) in zip(axes.flat,[(17,'occupancy','S 高 α 词族占据率 · 种子 17',100),(43,'occupancy','S 高 α 词族占据率 · 种子 43',100),(17,'alpha','实测后顶区 EEG α 功率占比',100),(17,'HbO','实测额前 HbO',1)]):
        p=load(f'events_{seed}/event_profile.csv').query("split=='validation' and area=='posterior'")
        for cond,color,style in [('MA',BLUE,'o-'),('BL',RED,'s--')]:
            v=p[p.condition==cond].pivot(index='subject',columns='phase',values=field)[phases]*factor
            ax.plot(np.arange(5),v.T,color=color,alpha=.26,lw=.8)
            ax.plot(v.mean().values,style,color=color,label='心算' if cond=='MA' else '显式基线')
        ax.set(xticks=range(5),xticklabels=ticks,ylabel='相对 Hb 单位' if field=='HbO' else '百分比',title=title)
        ax.legend(fontsize=8,ncol=2)
    report.page('后顶区高 α 词族的事件效应随种子变号','按各自的 EEG/Hb 原生提示时钟取窗。横轴为提示后秒数的阶段区间，阶段等距绘制。淡线为被试，粗线为 3 人均值。主对照先算每试次 2–10 秒相对 −4–0 秒变化，再在同会话作心算减显式基线、按会话和人等权。',fig,
        '种子 17 的占据差为 +2.85 个百分点，描述性 95% 区间 [−7.08, +13.55]；种子 43 为 −7.54，[−29.17, +15.76]。两个双侧精确被试符号翻转检验均为 p=0.75。原始连续 EEG α 占比的同窗条件差为 −1.37 个百分点，区间 [−2.85, +0.69]，此面板也未清楚建立连续主效应。','events_17、events_43 中的 event_profile.csv、event_contrasts.csv；占据率是 P3/P4 两电极落入训练最高四分位词族的比例。','event_profile')
    fig,axes=subplots(1,2)
    for ax,seed in zip(axes,SEEDS):
        v=load(f'events_{seed}/event_associations.csv')
        for i,(area,marker,col) in enumerate([('posterior','o',BLUE),('frontal','s',RED)]):
            for j,m in enumerate(['alpha','HbO','HbR']):
                q=v[(v.area==area)&(v.measure==m)]
                ax.scatter(j+(i-.5)*.22+np.linspace(-.038,.038,len(q)),q.r,marker=marker,color=col,label=('顶区' if i==0 else '额区') if j==0 else None)
        ax.axhline(0,color='gray',lw=.7)
        ax.set(xticks=range(3),xticklabels=['实测 α 占比','额前 HbO','额前 HbR'],ylim=(-1,1),ylabel='试次内对照后的相关系数',title=f'种子 {seed}')
        ax.legend()
    report.page('同一人的词族波动与实测 EEG、Hb 的关系','每点是一人的试次相关。先求各试次活动期相对提示前的变化量，再分别减去该人同会话、同条件的平均变化，最后计算高 α 词族占据变化与连续测量变化的 Pearson 相关。',fig,
        '横向散开用于显示重叠点。α 占比来自相应区域的实测 EEG，HbO/HbR 来自固定额前区域。该探索性相关排除了会话和条件平均差，仍依赖同一批 3 人；跨模态的预测增量由独立的 Hb 历史对照检验。','events_17/event_associations.csv、events_43/event_associations.csv。','event_associations')
    trials=pd.read_csv(example/'trial_selection.csv')
    for row in trials.itertuples():
        with np.load(example/f'trial_{row.condition}.npz') as z:
            fig=plt.figure(figsize=(13,5.6),layout='constrained')
            axes=fig.subplots(4,1,gridspec_kw={'height_ratios':[1,1,1,1.15]})
            t=np.arange(6000)/200-6
            for ax,(k,offset,name,pos) in zip(axes[:2],[('event',-6,'P3 · 事件',2),('history',-26,'F3 · Hb 预测历史',0)]):
                ax.plot(np.arange(6000)/200+offset,z[k+'_x'][pos].ravel(),color=BLUE,lw=.65,label='实测 X')
                ax.plot(np.arange(6000)/200+offset,z[k+'_sum'][pos].ravel(),'--',color=RED,lw=.7,label='整数解码 X')
                ax.set(xlim=(-26,24),ylabel=name+'\n训练 SD');ax.axvline(0,color='#434343',ls=':',lw=.8);ax.legend(ncol=2,fontsize=8,loc='upper left')
            ax=axes[2];t=np.arange(len(z['hb']))/10-5
            for j,(color,style) in enumerate([(RED,'-'),(BLUE,'--')]):
                ax.plot(t,z['hb'][:,j],style,color=color,label=('HbO','HbR')[j])
            ax.axvspan(2,4,color='#9bc6ba',alpha=.5);ax.axvspan(10,12,color='#ebba79',alpha=.5)
            ax.axvline(0,color='gray',ls=':',lw=.8);ax.set(xlim=(-26,24),ylabel='额前实测 Hb\n相对 Hb 单位'); ax.legend(ncol=2,fontsize=8,loc='upper left')
            strip(axes[3],[(z['history_'+b][0],b) for b in 'SO'],['F3 历史 S','F3 历史 O'],start=-26)
            axes[3].set_xlim(-26,24);axes[3].set_xticks(range(-26,25,2))
            axes[3].axvline(0,ls=':',color='gray',lw=.8)
            report.page(f'真实{"心算" if row.condition=="MA" else "基线"}试次的 EEG 重建、token 与实测血流',f'Single-Trial · {row.subject} · {row.record} · 事件 {row.event} · 种子 17。取两项终点均可评分、且两条件齐全的首个被试会话，再取每种条件时间最早的一次。每行都以本模态的提示为 0 秒。',fig,
                'P3 事件窗与 F3 历史窗分别从完整 X 重新编码。底行与第二行共享横轴，每格 2 秒。Hb 目标为橙区 [10,12) 秒实测均值减绿区 [2,4) 秒均值；Hb 颜色为两侧额前位置均值。波形是离线预处理后的实测输入，稠密采样时间与 2 秒词格分别保留。'+palette(),
                'examples/trial_selection.csv、trial_'+row.condition+'.npz；原生 EEG 与 Hb 标记秒数单独保存在表中。','trial_'+row.condition)
    fig,axes=subplots(1,2,sharey=True)
    methods=['base','continuous','aligned','mismatch','shift','frozen_mismatch','frozen_shift']
    labels=['Hb 历史','+连续 EEG','+对齐 token','跨试次重拟合','移位重拟合','跨试次冻结','移位冻结']
    for ax,seed in zip(axes,SEEDS):
        f=load(f'events_{seed}/coupling_subject.csv').groupby(['subject','method']).nmse.mean().unstack()[methods]
        for sub,v in f.iterrows():
            ax.plot(range(len(v)),v.values,'o-',lw=.9,alpha=.6,label=sub[-2:])
        ax.plot(range(len(methods)),f.mean().values,'D--',color='#172d37',lw=2,label='3 人均值')
        ax.set(xticks=range(len(methods)),xticklabels=labels,ylabel='双色 NMSE（越低越好）',title=f'种子 {seed}')
        ax.tick_params(axis='x',rotation=28);ax.legend(ncol=4,fontsize=8)
    report.page('加入对齐 EEG token，3 人的实测 Hb 预测误差均上升','先用 14 名训练人的 614 个配对试次按人三折为历史基线在 2、4、6 秒中选延迟，选中 6 秒。基线含双颜色最近三个 2 秒均值、条件、会话及试次序号。EEG 增广使用 F3/F4 最近三格与整窗摘要。',fig,
        '在同一批 107 个开发试次上，历史基线为 0.18861，加入连续 EEG 为 0.18963，加入对齐 C 为 0.26641 / 0.23512。平方误差除以该颜色训练目标方差，先均衡会话内两条件，再按会话、被试和颜色平均。错配固定人、会话、条件；移位为同记录 −43 秒。“冻结”沿用对齐回归系数，另两项重拟合。离线全记录滤波限制了时序推断。','events_17、events_43 中的 events_summary.json、coupling_subject.csv、coupling_contrasts.csv、coupling_predictions.csv。','coupling')


def paired_results(report):
    fig,axes=subplots(1,2)
    for ax,seed in zip(axes,SEEDS):
        f=load(f'events_{seed}/event_contrasts.csv').query("area=='posterior' and phase=='active'").set_index('measure')
        keys=['occupancy','enter','leave','alpha']
        for j,m in enumerate(keys):
            r=f.loc[m]
            ax.hlines(j,100*r.ci_low,100*r.ci_high,color=BLUE,linewidth=2)
            ax.plot(100*r.ma_minus_bl_change,j,'D',color=RED)
        ax.axvline(0,color='gray',ls='--')
        ax.set(yticks=range(4),yticklabels=['占据率（主终点）','转入高 α 族','转出高 α 族','实测 α 占比'],xlim=(-35,40),xlabel='心算减基线的前后变化差（百分点）',title=f'种子 {seed}')
        ax.invert_yaxis()
    report.page('同一窗口内，实测 α、占据和转入转出给出不同读数','点是 3 人等权的后顶区变化差。线段是被试配对重采样 2,000 次的描述性百分位 95% 区间。转入和转出按当前格与上一格是否进入或离开训练高 α 族计数。',fig,
        '每个阶段都相对同一试次提示前 −4–0 秒，再作同会话心算与显式基线的差。占据率是预定主终点；其他终点用于解释，未作多重比较校正。两种子使用同一批人，不能合并扩大样本量。小样本区间的端点高度依赖这三人。','events_17/event_contrasts.csv、events_43/event_contrasts.csv。','event_changes')
    fig,axes=subplots(2,2)
    for j,color in enumerate(['HbO','HbR']):
        tables=[load(f'events_{s}/coupling_predictions.csv') for s in SEEDS]
        vals=np.concatenate([g[[color+k for k in ['_measured_change','_base','_aligned']]].values.ravel() for g in tables])
        lim=[vals.min(),vals.max()]
        for ax,g,s in zip(axes[j],tables,SEEDS):
            x=g[color+'_measured_change']
            ax.plot(lim,lim,':',color='gray',label='理想一致')
            for m,col,marker,l in [('base',BLUE,'o','Hb 历史'),('aligned',RED,'s','+对齐 C')]:
                ax.scatter(x,g[color+'_'+m],s=12,color=col,marker=marker,alpha=.62,label=l)
            ax.set(xlim=lim,ylim=lim,xlabel='实测变化（相对 Hb）',ylabel='预测变化（相对 Hb）',title=f'{color} · 种子 {s}')
            ax.legend(ncol=3,fontsize=7)
    report.page('同一试次的实测 Hb 变化与两种预测逐点对照','每个面板含 107 个试次，每个试次对应一个实测变化值和两个预测。对角线是预测值等于实测值。每种颜色在两种子上使用相同坐标，HbO 和 HbR 各用自身量程。',fig,
        '双色合并会掩盖单颜色差异。HbO 的 NMSE 从 0.26292 升至 0.37623 / 0.32142，HbR 从 0.11430 升至 0.15658 / 0.14882。图中每个点按试次展示；误差结论沿用会话内条件平衡、被试等权的统计口径。','events_17、events_43 的 coupling_predictions.csv、coupling_contrasts.csv。','hb_predictions')


def overview(report):
    fig,axes=subplots(1,3)
    for i,seed in enumerate(SEEDS):
        source=read_json(audit.spectral.folder(seed,'C')/'evaluation/summary.json')
        methods=[np.ravel(source['metrics']['nmse']['macro']),np.ravel(source['metrics']['correlation']['macro'])]
        for ax,v in zip(axes[:2],methods):
            bars=ax.bar(np.arange(3)+(i-.5)*.35,v,.34,color=[BLUE,RED][i],label=f'种子 {seed}')
            ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=2)
        d=load(f'events_{seed}/coupling_contrasts.csv').query("color=='two_color'").set_index('reference')
        v=[d.loc['base','reference_nmse'],d.loc['continuous','reference_nmse'],d.loc['base','aligned_nmse']]
        bars=axes[2].bar(np.arange(3)+(i-.5)*.35,v,.34,color=[BLUE,RED][i])
        axes[2].bar_label(bars,fmt='%.3f',fontsize=8,padding=2)
    for ax,title,labs in zip(axes,['EEG 整窗重建 NMSE ↓','EEG 整窗波形相关 ↑','实测 Hb 预测 NMSE ↓'],[['X','S','O'],['X','S','O'],['历史','+连续 EEG','+C 词族']]):
        ax.set(xticks=range(3),xticklabels=labs,title=title)
        ax.margins(y=.16)
    axes[0].legend()
    report.page('最终 C 保留了可读属性，独立事件和 Hb 增量证据仍然不足',
        '复核同一单级向量量化器的两个训练种子。EEG 每个分支为 8192 词、64 维，Hb 沿用 1024 词、128 维。每个位置输入 30 秒处理后实测 X，输出两串各 15 个整数；解码得到 S、O 两个条件估计，相加重建 X。',fig,
        '四库 60 名训练人用于标定和拟合读出，在 16 名普通开发人、53,476 个 EEG 窗口×位置上评价。X 是观测输入，S/O 是冻结教师给出的成分目标。两种子宏 X 相关均约 0.66，包络属性可以从 S 码向量读出。后顶区事件占据变化随种子变号；加入 EEG 词族后，3 人的双色 Hb 误差均升高。以下各页分别核对波形、词、序列和跨模态终点。',
        'single_vq_spectral_location 的两份 C/evaluation/summary.json；本次 events_17、events_43 的 coupling_contrasts.csv。','overview')


def decoder(report):
    fig,axes=subplots(1,2)
    for ax,seed in zip(axes,SEEDS):
        f=load(f'readout_{seed}_S/scores.csv').query("dataset=='macro'").copy()
        labels=['θ 功率','α 功率','250 ms 包络','2 s RMS']
        for m,marker,color,label in [('balanced_shrunk','s',RED,'平衡收缩 ID'),('embedding','o',BLUE,'码向量'),('decoded','D','#4b8156','整数解码波形')]:
            v=f[f.method==m].set_index('attribute').loc[['power_4_8','power_8_13','rms_envelope','patch_rms']]
            ax.plot(range(4),v.mse/v.background_mse,marker+'-',label=label,color=color)
        ax.set(yscale='log',xticks=range(4),xticklabels=labels,ylabel='方法 MSE / 背景 MSE（对数轴）',title=f'S 分支 · 种子 {seed}')
        ax.axhline(1,color='gray',ls='--');ax.legend(fontsize=8)
    report.page('从同一组 ID 线性读出包络，比从解码波形重新测量更准',
        '用同一开发教师 S 片段作目标。一种方法用训练数据学习读出，另一种直接把整数解码成 200 Hz 波形后重算相同属性。虚线 1 对应背景均值的误差；点低于 1 表示改善。',fig,
        '两个种子的码向量包络 MSE 比背景低 37.5% / 38.0%，直接解码后却达到背景的 2.72 / 2.79 倍。以整段 RMS 衡量，直接解码误差约为背景的 6.73 / 7.03 倍。原频谱实验另按库的训练均值和标准差标定对数功率，解码 log-α MSE 为 102.13 / 102.54；它使用另一种属性和尺度。','readout_17_S、readout_43_S 的 scores.csv；对数谱来自两种子 C/evaluation/summary.json。','decoder')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    for stage in ['extract','relations','readout','context','events']:
        if read_json(ROOT/f'{stage}_status.json')['status']!='completed':
            raise ValueError(f'incomplete stage {stage}')
    source=ROOT/'examples'
    if not (source/'transitions.csv').exists():
        from single_vq_final_c_examples import export
        export(source)
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    plt.rcParams.update({'font.family':['DejaVu Sans','Droid Sans Fallback'],'font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,
                         'axes.edgecolor':'#8899a1','axes.labelcolor':'#203a46','text.color':'#203a46'})
    args.output.mkdir(parents=True,exist_ok=False)
    report=Report(args.output)
    overview(report)
    windows(report,source)
    readouts(report)
    decoder(report)
    members(report)
    context_pages(report)
    relationships(report,source)
    event_pages(report,source)
    paired_results(report)
    report.finish()


if __name__=='__main__':
    main()
