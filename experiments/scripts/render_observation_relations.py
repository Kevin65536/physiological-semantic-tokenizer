#!/usr/bin/env python3
"""Render the three held-out relation questions from the retained small tables only."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parents[2]
SUITE=ROOT/'experiments/runs/physiology_semantic_tokenizer/observation_factorization'
COL={'eeg':'#24628a','hb':'#a14836','neutral':'#596b72','null':'#92a3aa'}
STYLE={'font.family':['Noto Sans CJK JP','DejaVu Sans'],'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#acb9be','font.size':10.5,'axes.unicode_minus':False,'xtick.color':'#44535b','ytick.color':'#44535b','text.color':'#213440','axes.labelcolor':'#344a54','figure.facecolor':'white','savefig.facecolor':'white'}


def only(frame,**val):
    for k,v in val.items():frame=frame[frame[k]==v]
    if len(frame)!=1:raise ValueError(f'ambiguous result lookup: {val}: {len(frame)} rows')
    return frame.iloc[0]


def read(p):return json.loads(p.read_text())


def verify(run):
    """Recalculate macro contrasts from person-level exports, not from the figures."""
    panel=read(run/'panel_replay.json'); words=read(run/'phenotypes.json'); event=read(run/'events.json');coll=read(run/'collisions.json')
    if not panel['reproduced_all_64_split_modality_word_support_counts'] or not coll['synthetic_only'] or not event['same_exact_event_identities_as_frozen_probe']:
        raise ValueError('input/model evidence not replayed')
    nchecked=0
    d=pd.read_csv(run/'dynamics_scores.csv');ed=pd.read_csv(run/'dynamics_edges.csv')
    for f,keys in [(d,['modality','routing','metric','scope']),(ed,['modality','rank'])]:
        for _,block in f.groupby(keys):
            macro=only(block,level='macro');s=block[block.level=='subject']
            x=s.groupby('dataset')[['observed','null_mean','delta']].mean().mean(skipna=False)
            if not np.allclose(x, macro[x.index].to_numpy(dtype=float),atol=1e-10,rtol=1e-10,equal_nan=True):raise ValueError('dynamics subject average mismatch')
            nchecked+=1
    for name in ('coupling_primary','coupling_four'):
        summary=pd.read_csv(run/f'{name}_summary.csv');s=pd.read_csv(run/f'{name}_by_subject.csv')
        keys=['lag_s','endpoint','reference','candidate'];x=s.groupby(keys+['dataset'])[['reference_loss','candidate_loss','gain']].mean().groupby(keys).mean();y=summary.set_index(keys).reindex(x.index)
        if len(y)!=len(summary) or not np.allclose(x,y[x.columns],rtol=1e-9,atol=1e-10) or not np.allclose(y.gain,y.reference_loss-y.candidate_loss):raise ValueError('coupling paired losses mismatch')
        nchecked+=len(y)
    v=event['by_split']['validation'];assert v['retained_events']==264 and v['planned_events']==360 and v['n_registered_subjects']==6
    for key,row in v['estimates'].items():
        vals=[x['ma_minus_bl'][key] for x in v['subjects'] if x['ma_minus_bl'] is not None]
        if vals and not np.allclose([np.mean(vals),row['paired_mean_MA']-row['paired_mean_BL']],row['paired_mean_MA_minus_BL'],atol=1e-12):raise ValueError('session-paired contrast mismatch')
        nchecked+=1
    counts={m:sum(x['n'] for x in words['support'] if x['modality']==m and x['split']=='validation') for m in ('eeg','hb')}
    if counts!=panel['validation_cells']:raise ValueError('per-word validation support differs')
    (run/'verification.json').write_text(json.dumps(dict(status='passed',person_level_contrasts_recomputed=nchecked,verified_counts=counts,checks=['paired clocks and all 64 old ID masses','frozen synthetic nonidentifiability witnesses','macro sequence and selected edge rates from subject rows','all matched future-target losses from subject rows','event IDs vs original probe and session-paired summaries','complete 32-word validation support']),ensure_ascii=False,indent=2)+'\n')


def decision(run):
    """Compact reading card, derived only from the completed, audited scientific endpoints."""
    p=read(run/'phenotypes.json');v=read(run/'events.json')['by_split']['validation'];d=pd.read_csv(run/'dynamics_scores.csv');edges=pd.read_csv(run/'dynamics_edges.csv');cou=pd.read_csv(run/'coupling_primary_summary.csv');four=pd.read_csv(run/'coupling_four_summary.csv')
    seq={m:{scope:only(d,level='macro',routing='pattern',modality=m,metric='history_gain_bits',scope=scope)[['observed','null_mean','delta','ci_low','ci_high','n_subjects']].astype(float).to_dict() for scope in ('all','switch')} for m in ('eeg','hb')}
    main=only(edges,modality='hb',level='macro',rank=1);hbt={k:float(main[k]) for k in ('source_id','target_id','observed_count','expected_count','n_support_subjects','n_support_datasets','observed','null_mean','delta','ci_low','ci_high')}
    def cross(f):return {ep:only(f,lag_s=6,endpoint=ep,reference='base',candidate='pattern')[['reference_loss','candidate_loss','gain','ci_low','ci_high','n_val_subjects','n_val_origins']].astype(float).to_dict() for ep in ('hb_delta','hb_word')}
    data=dict(schema='observation_relations_decision_v1',status='completed',scope='four reused public development datasets; frozen K16 per modality, two second patches, integer ID only; no test results',
              verdicts={'single_word':'supported_as_observation_descriptions_not_sources; no shared support for additional adjusted alpha_peak or Hb_next_opposition effects',
                        'sequence':'history_order_information_strong_for_Hb_small_for_EEG_also_present_in_blind_words',
                        'coupling':'no_increment_for_this_offline_conditioned_lag_probe',
                        'task':'predeclared_frontal_Hb_occupancy_increase_not_demonstrated; secondary graded EEG band readout consistent with observed bands but exploratory'},
              word_comparisons={k:dict(status=x['status'],target_id=x['target_id'],target_patches=x['support']['target_n'],target_hits=x['support']['target_hits'],qualified_subjects=x['support']['qualified_subjects']) for k,x in p['comparisons'].items()},
              time_order_bits_per_transition=seq,train_selected_top_hb_edge=hbt,primary_two_marker_datasets=cross(cou),sensitivity_four_datasets=cross(four),
              mental_arithmetic={k:v['estimates'][k] for k in ('hb_joint_rise_fall_occupancy/all','posterior_alpha_centroid_fraction/all','frontal_theta_centroid_fraction/all')},
              primary_mental_arithmetic_sign_flip=v['primary_sign_flip'],audit='verification.json',interpretation='REPORT.md')
    (run/'decision.json').write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2)+'\n')


def lexicon(run):
    """One row per frozen physical word. Counts below are descriptive patch counts, not inferential n."""
    p=read(run/'phenotypes.json');t={(r['modality'],r['pattern_id']):r for r in p['support'] if r['split']=='train'};rows=[]
    for r in (x for x in p['support'] if x['split']=='validation'):
        m,i=r['modality'],r['pattern_id'];a=t[m,i];n=r['n'];q=lambda v: None if not r['direct_q'][v] else r['direct_q'][v][1]
        row=dict(modality=m,word=i,training_description=a['construction_label'],train_support_passed=a['train_supported'],train_patches=a['n'],validation_patches=n,validation_subjects=r['n_subjects'],validation_datasets=r['n_datasets'],validation_largest_subject_fraction=r['max_subject_fraction'])
        if m=='eeg':
            hi=r['high_classes'];row.update(val_median_alpha_fraction=q('alpha'),val_median_high_fraction=q('45-100'),val_alpha_narrow_peak_count=r['peak_hits'],val_alpha_evaluable=r['peak_evaluable'],val_line_like_count=sum(hi[k] for k in ('line50_like','line60_like','both_lines_like')),val_diffuse_like_count=hi['diffuse_like'])
        else:
            g=r['interior_classes'];row.update(val_median_HbO_slope=q('so'),val_median_HbR_slope=q('sr'),val_match_both_centroid_signs=r['ols_both_center_signs'],val_interior_opposed_count=g['opposed_o_up']+g['opposed_o_down'],val_interior_common_count=g['common_up']+g['common_down'],val_interior_unresolved=g['unresolved'])
        rows.append(row)
    if len(rows)!=32: raise ValueError('not the complete two-modality K16 atlas')
    pd.DataFrame(rows).sort_values(['modality','word']).to_csv(run/'word_lexicon.csv',index=False)


def atlas_figure(run):
    from matplotlib import colormaps
    colors=colormaps['YlGnBu'].copy();colors.set_bad('#d7dadc')
    ph=read(run/'phenotypes.json'); val=[x for x in ph['support'] if x['split']=='validation']
    with rc_context(STYLE):
        fig=Figure(figsize=(12,7));FigureCanvasAgg(fig);a,b=fig.subplots(2,1);fig.subplots_adjust(left=.155,right=.918,bottom=.15,top=.805,hspace=.79)
        fig.text(.021,.95,'每个观测词在留出被试中实际对应什么？',size=19,weight='bold');fig.text(.021,.9,'列为原词 ID（括号内为验证片数）；下列指标来自同一批实测片，不是独立的生理标签。',size=11,color='#445b68')
        for ax,m in [(a,'eeg'),(b,'hb')]:
            rr=sorted((r for r in val if r['modality']==m),key=lambda r:r['pattern_id']); ns=[r['n'] for r in rr]; vals=[]
            if m=='eeg':
                labels=['8–13 Hz 功率占比中位数','有局部窄 α 峰的比例','50/60 Hz 线样峰片比例'];title='EEG：同一频带富集词未必表示单一脑节律'
                for r in rr:
                    q=r['direct_q']['alpha'];d=r['high_classes'];vals.append([np.nan if q is None else q[1],r['peak_hits']/max(1,r['peak_evaluable']),sum(d[k] for k in ('line50_like','line60_like','both_lines_like'))/max(1,r['n'])])
            else:
                labels=['内段 HbO 升、HbR 降','内段 HbO 降、HbR 升','内段两者同向','两色方向未同时辨清'];title='Hb：检验内段是否持续同向或反向变化，而非只看拟合斜率'
                for r in rr:
                    g=r['interior_classes'];vals.append(np.array([g['opposed_o_up'],g['opposed_o_down'],g['common_up']+g['common_down'],g['unresolved']])/max(1,r['n']))
            z=np.array(vals).T*100;z[:,np.array(ns)==0]=np.nan;im=ax.imshow(z,aspect='auto',cmap=colors,vmin=0,vmax=100)
            ax.set_yticks(np.arange(len(labels)),labels,fontsize=9.2);ax.set_xticks(np.arange(16),[f'{i}\n({n})' for i,n in enumerate(ns)],fontsize=8.2);ax.tick_params(length=0);ax.set_title(title,loc='left',size=12.6,weight='bold',pad=10);ax.spines[:].set_visible(False)
            for (y,x),value in np.ndenumerate(z):ax.text(x,y,'—' if not np.isfinite(value) else f'{value:.0f}',ha='center',va='center',fontsize=8.7,color='white' if value>=56 else '#20323a')
        cb=fig.colorbar(im,cax=fig.add_axes([.939,.225,.012,.5]));cb.ax.set_ylabel('比例（%）',size=10);fig.text(.035,.035,'灰色 = 该 ID 在验证集没有样本；各格为描述性片级比例或中位数，不能以片数代替被试数。词源与跨模态关系未由这些指标识别。',size=9.3,color='#445b68')
        fig.savefig(run/'word_evidence.png',dpi=210)


def render(run):
    if run.parent!=SUITE or read(run/'summary.json')['status']!='completed':raise ValueError('completed fixed readout required')
    verify(run);decision(run);lexicon(run);atlas_figure(run)
    d=pd.read_csv(run/'dynamics_scores.csv');v=read(run/'events.json')['by_split']['validation'];cou=pd.read_csv(run/'coupling_primary_summary.csv')
    with rc_context(STYLE):
        fig=Figure(figsize=(11.6,7.7));FigureCanvasAgg(fig)
        a,b,c,e=fig.subplots(2,2).flat;fig.subplots_adjust(left=.135,right=.975,bottom=.13,top=.855,wspace=.45,hspace=.63)
        fig.text(.028,.962,'四位观测词：相邻关系、跨模态延迟与真实任务',size=19,weight='bold')
        fig.text(.028,.915,'冻结词表，只读开发验证；各图横轴单位不同。时间均为离线预处理后的观测时间。',size=11,color='#445b68')
        def axis(ax,title,xlabel):
            ax.set_title(title,loc='left',size=12.4,weight='bold',pad=10);ax.set_xlabel(xlabel,size=10);ax.xaxis.grid(True,alpha=.18);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0);ax.spines['left'].set_visible(False);ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        for j,m in enumerate(('hb','eeg')):
            s=only(d,level='macro',routing='pattern',modality=m,metric='self_rate',scope='all');x=[100*s.null_mean,100*s.observed];a.plot(x,[j,j],color=COL[m],lw=2);a.scatter(x[0],j,facecolor='white',edgecolor=COL[m],s=62,zorder=3);a.scatter(x[1],j,color=COL[m],s=56,zorder=4)
        a.set_yticks([0,1],['Hb','EEG']);a.set_ylim(-.5,1.55);a.set_xlim(0,100);axis(a,'A. 下一片还是同一个词吗？','相邻片同词率（%；实心=实测，空心=同窗无序期望）')
        keys=[('hb','switch'),('hb','all'),('eeg','switch'),('eeg','all')]; labels=['Hb · 仅换词','Hb · 全部','EEG · 仅换词','EEG · 全部']
        for j,(m,scope) in enumerate(keys):
            s=only(d,level='macro',routing='pattern',modality=m,metric='history_gain_bits',scope=scope);b.plot([s.ci_low,s.ci_high],[j,j],color=COL[m],lw=2);b.scatter(s.delta,j,color=COL[m],s=43,zorder=3)
        b.set_yticks(range(4),labels);b.set_ylim(-.55,3.55);b.axvline(0,color=COL['null'],lw=1,ls='--');axis(b,'B. 词序比同窗词袋多提供多少信息？','下一词交叉熵增益减去无序期望（bit/转移；正值有利）')
        for j,h in enumerate((8,6,4)):
            for method,dy,color in [('base',.12,COL['eeg']),('shifted',-.12,COL['hb'])]:
                s=only(cou,lag_s=h,endpoint='hb_delta',reference=method,candidate='pattern');c.plot([s.ci_low,s.ci_high],[j+dy,j+dy],color=color,lw=2);c.scatter(s.gain,j+dy,color=color,marker='o' if method=='base' else 's',s=34,zorder=3)
        c.set_yticks(range(3),['8 秒','6 秒（主）','4 秒']);c.set_ylim(-.55,2.55);c.axvline(0,color=COL['null'],lw=1,ls='--');axis(c,'C. EEG 词对延迟 Hb 变化有额外帮助吗？','训练方差标化误差的减少量（正值有利）');c.plot([],[],'o',color=COL['eeg'],label='对 Hb 自身历史');c.plot([],[],'s',color=COL['hb'],label='对错位 EEG 词');c.legend(frameon=False,fontsize=8,loc='best')
        names=[('hb_x_joint_rise_fall_fraction/all','直接测量斜率'),('hb_joint_rise_fall_occupancy/all','词表质心斜率')]
        for j,(key,label) in enumerate(names):
            vals=[100*s['ma_minus_bl'][key] for s in v['subjects'] if s['ma_minus_bl']];row=v['estimates'][key];lo,hi=np.array(row['paired_subject_ci95'])*100;mean=100*row['paired_mean_MA_minus_BL'];e.scatter(vals,j+np.linspace(-.12,.12,len(vals)),s=19,alpha=.55,color=COL['neutral']);e.plot([lo,hi],[j+.20]*2,lw=2,color=COL['hb']);e.scatter(mean,j+.20,marker='D',color=COL['hb'],s=38)
        e.set_yticks(range(2),[n[1] for n in names]);e.set_ylim(-.45,1.55);e.axvline(0,color=COL['null'],lw=1,ls='--');axis(e,'D. 心算时，额区 HbO 升且 HbR 降更常见吗？','心算 − 显式基线（百分点；小点=被试，菱形=均值）')
        fig.text(.028,.035,'A/B：四库 20 人，2 秒片；C：有独立同步标记的两个库（各原生位点，并非对应脑区）；D：Single-Trial 6 人。线段为被试级 95% 区间（A 为两种顺序的比较）。',size=9,color='#445b68')
        f=run/'relations.png';fig.savefig(f,dpi=210);return f


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path);print(render(p.parse_args().run.resolve()))
