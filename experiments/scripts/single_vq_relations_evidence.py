"""Build compact reading tables and figures directly from the frozen screen outputs."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def collect_evidence(root, models):
    def merged(name):
        return pd.concat([pd.read_csv(root/f'{m}_{s}'/name).assign(model=f'{m}_{s}') for m,s in models],ignore_index=True)
    coverage=merged('coverage.csv')
    conditional=merged('conditional.csv')
    sequences=merged('sequence.csv')
    interactions=merged('interaction_vs_family_main.csv')
    for name,table in [('word_coverage',coverage),('conditional',conditional),('order',sequences),('interaction',interactions)]:
        table.to_csv(root/f'{name}.csv',index=False)
    subj=pd.read_csv(root/'event_support.csv')
    upstream=pd.read_csv(root/'upstream_event_exclusions.csv')
    pairs=pd.read_csv(root/'coupling_pairs.csv')
    keys=['split','subject','record','event']
    plan=pd.concat([subj[keys],upstream[keys]],ignore_index=True)
    if len(plan.drop_duplicates())!=len(plan):
        raise ValueError('event ledger contains duplicate trial identities')
    audit=[]
    for split,g in plan.groupby('split'):
        log=subj[subj.split==split]
        admitted=pairs[(pairs.split==split)&(pairs.donor_event>=0)]
        use=log[log.B=='used']
        paired_session=use.groupby(['subject','record']).condition.transform('nunique')==2
        audit.append(dict(b_primary_paired_trials=int(paired_session.sum()),b_primary_paired_sessions=len(use[paired_session].groupby(['subject','record'])),split=split,planned_people=g.subject.nunique(),planned_trials=len(g),
            upstream_excluded=len(g)-len(log),b_valid_trials=len(use),b_valid_people=use.subject.nunique(),
            d_pre_pair=len(log[log.D=='candidate']),d_paired_trials=len(admitted),d_paired_people=admitted.subject.nunique()))
    pd.DataFrame(audit).to_csv(root/'event_denominators.csv',index=False)
    seed_profile=pd.read_csv(root/'hb_seed_correspondence.csv')
    seed_profile.groupby(['branch','dataset'])[['attribute_rank_r','family_agreement','family_ari']].mean().groupby('branch').mean().to_csv(root/'hb_seed_summary.csv')
    event_figures(root)
    # A portable entry to the six member sheets and the numeric evidence behind them.
    body=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>冻结生理词与序列 · 开发集证据</title>',
          '<style>body{max-width:1050px;margin:36px auto;padding:0 18px;font:17px/1.65 system-ui;color:#172738}img{max-width:100%;height:auto}table{border-collapse:collapse}td,th{padding:.3em .7em;border:1px solid #c8d0da}a{color:#174e81}</style>',
          '<h1>冻结单层 VQ 的真实成员、事件与跨模态证据</h1>',
          '<p>EEG 用 d64/K8192，Hb 用 d128/K1024。两秒一个整数，编码上下文为 30 秒。四库读出使用 60 名训练被试和 16 名普通开发被试。心算实验使用 Single-Trial 中可与两套原始事件时钟对齐的子集。</p>',
          '<h2>真实成员覆盖</h2><p>属性为 EEG 的 8–13 Hz 功率占 1–45 Hz 的比例，或 HbO 两秒线性斜率。逐词预测相应 S/O 监督分量的属性。背景为库、位置和可见 X 幅度档。训练支持至少 20 片、3 人，逐词的背景残差按 16 片收缩。在至少 3 名开发被试出现、平均误差有改善且严格多数人改善时，计入覆盖。覆盖按实际片段数计算，属于本轮开发集描述。</p>',
          coverage.query('dataset=="pooled"')[['model','branch','valid','readable_patches','readable_fraction']].round(4).to_html(index=False),
          '<p><a href="word_coverage.csv">分库覆盖与分母</a> · <a href="hb_seed_correspondence.csv">Hb 两个种子在同一片段上的属性排序与词族对应</a></p>',
          '<h2>实际成员与训练模板</h2><p>每张图从训练有支持的 ID 中按频数选择低、中、高三个代表。最后一行单列开发集上属性误差增大最多且支持足够的词。波形使用 G0 观测 X 与对应 S/O 目标，幅值采用各库各位置冻结的训练尺度。图中的时间从片段起点算起。细线为开发成员示例，实线为成员均值，虚线为冻结训练模板，点线为同人、同位置、同幅度档的开发对照。每个词、每个分区最多抽取 128 个成员。</p>']
    for m,s in models:
        for b in 'SO':
            prefix=f'{m}_{s}/{b}'
            body.append(f'<details><summary>{m.upper()} · 种子 {s} · {b} 分支</summary><a href="{prefix}_members.png"><img loading="lazy" src="{prefix}_members.png" alt="真实成员及匹配背景波形"></a><p><a href="{prefix}_members.csv">样例与背景身份</a> · <a href="{prefix}_wave_template.csv">波形模板误差</a> · <a href="{prefix}_words.csv">全部 ID 的属性与支持</a></p></details>')
    body.extend(['<h2>S/O 条件增量和短序列</h2><p>逐位置比较背景、S、O 和联合读出。联合的条件增量与受支持词族的交互效应分别检验。G0 中 X=S+O，线性 HbO 斜率也满足这一恒等式，双分支联合读出具有明确的代数基础。顺序检验用前 14 个词族估计实测 X 属性，逐步加入一阶转移、停留和三词计数。额外对照保留词频、精确保留一阶转移、或同时保留逐词停留长度分布。另在同电极相邻的不重叠窗口上评分，以排查 30 秒双向上下文造成的收益。</p>',
          '<p><a href="conditional.csv">条件增量</a> · <a href="interaction.csv">控制词族主效应后的交互</a> · <a href="order.csv">顺序、替代序列与不重叠窗口</a></p>',
          '<h2>独立心算提示与实测 Hb</h2>',pd.DataFrame(audit).to_html(index=False),
          '<p>事件锁定使用心算和显式基线的配对标记，各自在 EEG 和 Hb 的原生时钟上取窗。主要比较为顶区 P3/P4 上 EEG-S 高 α 词族在提示后 2–10 秒相对提示前 4 秒的占用率变化，再比较条件差。其他区域、时段、转入转出率和相关系数均为探索性。</p>',
          '<img src="event_profile.png" alt="事件锁定轨迹"><p><a href="event_contrasts.csv">事件效应与精确检验</a> · <a href="event_associations.csv">同人试次关联</a> · <a href="event_support.csv">逐试次准入</a> · <a href="upstream_event_exclusions.csv">上游排除</a></p>',
          '<p>跨模态读出只用截至提示后 4 秒的 30 秒 EEG 上下文，目标为独立预处理的额前实测 HbO/HbR 相对末两秒的变化。基线含血流历史、条件、会话及试次顺序；增广模型加入同一 EEG 历史的 S/O 词族。跨试次错配固定人、会话、条件和相位；记录内移位 43 秒，排除了与提示重新对齐的窗口。各比较共用相同留出试次。全记录离线预处理限制对真实时间先后的解释。</p>',
          '<img src="coupling_people.png" alt="分被试血流误差"><p><a href="coupling_contrasts.csv">配对结果与区间</a> · <a href="coupling_pairs.csv">供体和时标</a> · <a href="coupling_predictions.csv">实际及估计变化</a> · <a href="events_summary.json">训练侧延迟选择</a></p>',
          '<p>误差差值为参照 MSE 减候选 MSE。描述性 95% 区间在各库内以人为单位配对重采样，再等权平均各库。除预定事件主对照的双侧精确符号翻转检验外，其他对照均未作多重比较校正。</p></html>'])
    c=conditional.query('target=="X" and reference=="S" and candidate=="SO"').set_index('model')
    seq=sequences.query('branch=="S" and reference=="bigram_dwell" and candidate=="motif"')
    within=seq[~seq.adjacent].set_index('model')
    adjacent=seq[seq.adjacent].set_index('model')
    event=pd.read_csv(root/'event_contrasts.csv').query('primary').iloc[0]
    d=pd.read_csv(root/'coupling_contrasts.csv').query('color=="two_color"').set_index('reference')
    periods=next(row for row in audit if row['split']=='validation')
    body.insert(4,f'<p>联合 S/O 比只用 S 预测实测 EEG α 占比的误差低 {c.loc["eeg_17","gain"]:.2%}。'
                  f'HbO 斜率对应的误差降幅为 {c.loc["hb_17","gain"]:.2%} 和 {c.loc["hb_43","gain"]:.2%}。'
                  f'Hb-S 三词计数在同窗内相对一阶转移与停留模型的收益为 {within.loc["hb_17","gain"]:.2%} 和 {within.loc["hb_43","gain"]:.2%}，'
                  f'在下一不重叠窗口则为 {adjacent.loc["hb_17","gain"]:+.2%} 和 {adjacent.loc["hb_43","gain"]:+.2%}。</p>'
                  f'<p>事件主对照纳入 {periods["b_primary_paired_trials"]} 个同会话成对条件试次，'
                  f'心算相对显式基线的提示后占用增量为 {100*event.ma_minus_bl_change:+.2f} 个百分点，'
                  f'精确双侧 p={event.primary_exact_p:.2f}。'
                  f'在 {periods["d_paired_trials"]} 个可完成全部错配对照的开发试次上，'
                  f'加入 EEG 词后，实测 Hb 的双色宏 NMSE 从 {d.loc["base","reference_nmse"]:.5f} 升至 {d.loc["base","aligned_nmse"]:.5f}。'
                  f'两项事件检验均只有 {event.people} 名独立开发被试。</p>')
    (root/'index.html').write_text('\n'.join(body),encoding='utf-8')
    return audit


def event_figures(root):
    phase=['pre','entry','active','exit','recovery']
    f=pd.read_csv(root/'event_profile.csv').query('split=="validation"')
    fig,axes=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
    rows=[('posterior','occupancy','Posterior S high-alpha occupancy','fraction'),('frontal','occupancy','Frontal S high-alpha occupancy','fraction'),
          ('posterior','alpha','Observed posterior EEG alpha','power fraction'),('frontal','HbO','Measured prefrontal HbO','relative Hb')]
    for ax,(area,key,title,unit) in zip(axes.ravel(),rows):
        sub=f[f.area==area]
        for cond,color,style in [('MA','#15558a','o-'),('BL','#aa4d19','s--')]:
            v=sub[sub.condition==cond].pivot(index='subject',columns='phase',values=key)[phase]
            ax.plot(v.T.to_numpy(),color=color,lw=.75,alpha=.22)
            ax.plot(v.mean().values,style,color=color,label=cond)
        ax.set_xticks(range(5),['−4–0','0–2','2–10','10–18','18–24'])
        ax.set_title(title);ax.set_ylabel(unit);ax.set_xlabel('Seconds from cue (phase bins)');ax.legend()
    fig.suptitle('Offline event profiles | thin: person, bold: equal-person mean | MA = arithmetic, BL = explicit baseline')
    fig.savefig(root/'event_profile.png',dpi=200);plt.close(fig)
    s=pd.read_csv(root/'coupling_subject.csv')
    methods=['base','aligned','continuous','mismatch','shift']
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for ax,color in zip(axes,['HbO','HbR']):
        v=s[s.color==color].pivot(index='subject',columns='method',values='nmse')[methods]
        for name,values in v.iterrows():
            ax.plot(range(5),values,marker='o',label=name)
        ax.set_xticks(range(5),['Hb history','+ IDs','+ EEG\nfeatures','trial\nmismatch','time\nshift'])
        ax.set_title(color);ax.set_ylabel('MSE / training target variance');ax.legend(fontsize=8)
    fig.suptitle('Prefrontal measured Hb change | paired development subjects | lower error is better')
    fig.savefig(root/'coupling_people.png',dpi=200);plt.close(fig)
