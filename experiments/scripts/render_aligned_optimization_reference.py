"""Visual entry audit from the retained aligned-v1 waveform and real-member panels."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import periodogram

from run_aligned_tokenizer_optimization import OUT, ROOT

SOURCE=ROOT/'experiments/runs/physiology_semantic_tokenizer/aligned_token_audit/20261126_v1'
DEST=OUT/'reference_review'
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})
BASE='#176d8d'
OTHER='#bb592b'


def reconstruction():
    ledger=pd.read_csv(SOURCE/'paired_example.csv')
    ledger.to_csv(DEST/'example.csv',index=False)
    scores=[]
    with np.load(SOURCE/'paired_example.npz') as data:
        for mod in ('eeg','hb'):
            n=1 if mod=='eeg' else 2
            rate=200 if mod=='eeg' else 10
            idpair=[f"{data[mod+'S_id'][i]} / {data[mod+'O_id'][i]}" for i in range(15)]
            for c,channel in enumerate(['EEG'] if n==1 else ['HbO','HbR']):
                fig,axes=plt.subplots(3,2,figsize=(13,7.9),gridspec_kw={'width_ratios':[2.7,1]},layout='constrained')
                for i,b in enumerate('XSO'):
                    target=data[mod+b].reshape(-1,n)[:,c]
                    dec=data[mod+b+'_decoded'].reshape(-1,n)[:,c]
                    t=np.arange(len(target))/rate
                    for ax in axes[i]:
                        ax.plot(t,target,color='#6a7378',lw=.65,label='实测 X' if b=='X' else 'v3 条件目标 '+b)
                        ax.plot(t,dec,color=BASE,lw=1.15,label='整数解码')
                        ax.grid(axis='y',alpha=.15)
                    axes[i,0].legend(ncol=2,fontsize=9)
                    axes[i,0].set(ylabel=f'{b} / 训练输入 SD',xlim=(0,30))
                    axes[i,1].set_xlim(14,16)
                    rmse=float(np.sqrt(np.mean((target-dec)**2)))
                    corr=float(np.corrcoef(target,dec)[0,1])
                    scores.append(dict(modality=mod,channel=channel,component=b,rmse_sd=rmse,correlation=corr))
                    axes[i,0].set_title(f'{b}   30 秒 RMSE {rmse:.3f} SD，相关 {corr:.3f}',loc='left',fontsize=11)
                    axes[i,1].set_title('中心 2 秒放大',fontsize=10)
                axes[-1,0].set_xlabel('相对窗口起点（秒）')
                axes[-1,1].set_xlabel('秒')
                fig.suptitle(f'已对齐基线 · {channel}   |   Single-Trial / subject_03 / session_01 / 心算事件 17\n'
                             f'{ledger.query("mod==@mod").position.iloc[0]}   ·   中心片段 S / O 整数 {idpair[7]}   ·   19 个可用开发窗中按实测 α 占比取中位示例',fontsize=14)
                fig.savefig(DEST/f'reconstruction_{channel}.png',dpi=210)
                plt.close(fig)
    pd.DataFrame(scores).to_csv(DEST/'reconstruction.csv',index=False)


def members(mod,b):
    folder=SOURCE/(mod+'_17')
    meta=pd.read_csv(folder/(b+'_members.csv'))
    table=[]
    with np.load(folder/(b+'_member_waves.npz')) as arrays:
        for group in ['high','heldout_failure']:
            part=meta[(meta.group==group)&(meta.split=='validation')&(meta.kind=='member')].sort_values('pair')
            word=int(part.word.iloc[0])
            key=f'{group}_{word}_validation_'
            obs=arrays[key+'member'][:,1+'SO'.index(b)]
            ctl=arrays[key+'background'][:,1+'SO'.index(b)]
            assert list(part.pair)==list(range(len(obs))) and ctl.shape==obs.shape
            people=part.drop_duplicates(['dataset','subject']).iloc[:3]
            fig,ax=plt.subplots(2,2,figsize=(11.5,7.3),layout='constrained')
            rate=200 if mod=='eeg' else 10
            n=1 if mod=='eeg' else 2
            obs=obs.reshape(len(obs),-1,n)
            ctl=ctl.reshape(obs.shape)
            t=np.arange(obs.shape[1])/rate
            for ia,(_,r) in enumerate(people.iterrows()):
                a=ax.flat[ia]
                for c,col in enumerate([BASE] if n==1 else [OTHER,BASE]):
                    name='EEG' if n==1 else ['HbO','HbR'][c]
                    a.plot(t,obs[r.pair,:,c],color=col,lw=1.2,label=name+' 同词成员')
                    a.plot(t,ctl[r.pair,:,c],color=col,ls=':',lw=1.3,label=name+' 配对其他词')
                a.set(xlabel='片段时间（秒）',ylabel='条件目标 / 训练输入 SD',title=f'{r.dataset} · {r.subject} · {r.position}')
                a.legend(fontsize=8,ncol=n)
            a=ax.flat[3]
            for x,ls,col,name in [(obs,'-',BASE,'同词成员'),(ctl,'--',OTHER,'配对其他词')]:
                if n==1:
                    freq,powr=periodogram(x[:,:,0],fs=rate,window='hann',axis=1)
                    band=(freq>=1)&(freq<=45)
                    a.semilogy(freq[band],np.maximum(np.median(powr[:,band],axis=0),1e-12),ls,color=col,label=name)
                    a.set(xlim=(1,45),xlabel='频率（Hz）',ylabel='功率谱密度中位数（SD²/Hz）')
                    attr=powr[:,(freq>=8)&(freq<13)].sum(1)/np.maximum(powr[:,(freq>=1)&(freq<45)].sum(1),1e-14)
                else:
                    centered=t-t.mean()
                    attr=x[:,:,0]@centered/(centered@centered)
                    a.plot(np.sort(attr),np.arange(1,len(attr)+1)/len(attr),ls,color=col,label=name)
                    a.set(xlabel='HbO 两秒斜率（SD/s）',ylabel='累计片段比例')
                table.append(dict(modality=mod,branch=b,group=group,word=word,kind=name,pairs=len(x),
                                  people=len(part.drop_duplicates(['dataset','subject'])),median_attribute=float(np.median(attr))))
            a.legend(fontsize=9)
            a.set_title(f'全部 {len(obs)} 对片段 · 描述性分布')
            fig.suptitle(f'已对齐基线 · {mod.upper()}-{b} · ID {word} · '+('高训练词频' if group=='high' else '开发属性失败示例')+
                         '\n三个不同被试的真实片段；对照匹配同人、同位置、可见 X 幅度档；时域子图独立纵轴',fontsize=13)
            fig.savefig(DEST/f'{mod}_{b}_{group}.png',dpi=210)
            plt.close(fig)
    return table


if __name__=='__main__':
    DEST.mkdir(exist_ok=True,parents=True)
    reconstruction()
    tables=[row for m in ['eeg','hb'] for b in 'SO' for row in members(m,b)]
    pd.DataFrame(tables).to_csv(DEST/'member_attributes.csv',index=False)
