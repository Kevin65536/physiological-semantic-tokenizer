#!/usr/bin/env python3
"""Read-only audit and bitmap figure report for a completed masked-state suite."""
import argparse
import json
from pathlib import Path
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz

ARMS=['temporal','anonymous','spatial','joint','spatial_random','spatial_span','spatial_space']
LABELS=['仅时间','匿名时空','带位置时空','跨模态联合','仅随机遮挡训练','仅时间块训练','仅空间遮挡训练']
TASKS=['random','span','space']; TL=['随机 token','连续时间块','空间位置']
DS=['eeg_fnirs_single_trial','refed','simultaneous_eeg_nirs','visual_cognitive_motivation']; DL=['Single-Trial','REFED','Simultaneous','Visual']
FEATURES={'eeg':['δ 功率','θ 功率','α 功率','β 功率','γ 功率','RMS','均值','斜率'],
          'hb':['HbO 均值','HbR 均值','HbO SD','HbR SD','HbO 斜率','HbR 斜率','HbO 端点差','HbR 端点差']}

def load(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def macro(df, keys, metric):
    return df.groupby(keys+['dataset'])[metric].mean().groupby(keys).mean()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    run=args.run.resolve();out=args.output.resolve()
    if out.exists():raise ValueError('fresh versioned export required')
    summary=load(run/'summary.json');assert summary['status']=='completed' and summary['completed_models']==21
    out.mkdir(parents=True)
    a=pd.read_csv(run/'masked_subjects.csv');b=pd.read_csv(run/'physiology_subjects.csv');contr=pd.read_csv(run/'paired_contrasts.csv')
    errors=[];diags=[];history=[];budgets=[]
    for arm in ARMS:
        for seed in [17,29,43]:
            job=run/f'models/{arm}/seed_{seed}';s=load(job/'summary.json');assert s['status']=='completed' and s['completed_steps']==10000
            assert (job/'checkpoint.pt').is_file();budgets.append({'arm':arm,'seed':seed,'steps':s['completed_steps']})
            for fname,table,groups,fields in [('masked_rows.json',a,['task','modality','branch'],['CE_bits','prior_bits','gain_bits','accuracy']),('physiology_probe_rows.json',b,['modality','reader','feature'],['NMSE'])]:
                raw=pd.DataFrame(load(job/fname));keys=['dataset','subject']+groups
                rec=raw.groupby(['dataset','subject','record']+groups)[fields].mean().reset_index()
                derived=rec.groupby(keys)[fields].mean().sort_index();saved=table[(table.arm==arm)&(table.seed==seed)].set_index(keys)[fields].sort_index()
                assert derived.index.equals(saved.index)
                err=float(np.abs(derived.to_numpy()-saved.to_numpy()).max());assert err<1e-9;errors.append(err)
            d=pd.DataFrame(load(job/'context_diagnostics.json'));d=d.groupby(['dataset','subject','record','modality','perturbation']).delta_bits.mean().reset_index().groupby(['dataset','subject','modality','perturbation']).delta_bits.mean().reset_index();d['arm']=arm;d['seed']=seed;diags.append(d)
            for h in s['history']:history.append(dict(arm=arm,seed=seed,**h))
    diag=pd.concat(diags);hist=pd.DataFrame(history)
    mapping={'spatial_interaction':('spatial','temporal'),'position':('spatial','anonymous'),'cross_modal_interaction':('joint','spatial'),'mixed_vs_random':('spatial','spatial_random'),'mixed_vs_span':('spatial','spatial_span'),'mixed_vs_space':('spatial','spatial_space')}
    ce=macro(a,['arm','task','modality'],'CE_bits');gain=macro(a,['arm','task','modality'],'gain_bits')
    for r in contr.itertuples():
        t,c=mapping[r.contrast];assert abs(ce[c,r.task,r.modality]-ce[t,r.task,r.modality]-r.gain_bits)<1e-9
    probe=macro(b,['arm','modality','reader'],'NMSE');features=macro(b,['arm','modality','reader','feature'],'NMSE')
    token=load(run/'tokens/manifest.json')
    audit=dict(status='passed',completed_models=21,updates=sum(v['steps'] for v in budgets),max_subject_aggregation_error=max(errors),contrast_point_estimates_verified=len(contr),validation_subjects=a[['dataset','subject']].drop_duplicates().shape[0],planned_inputs=token['planned_inputs'],available_inputs=token['available_inputs'],source_run=str(run),source_summary='summary.json',array_reads='retained numerical result tables only; no measured/test/protected arrays or new model fitting',completion=summary['completed_at'])
    write(out/'verification.json',audit)
    ce.rename('CE_bits').to_csv(out/'CE_macro.csv');probe.rename('NMSE').to_csv(out/'probe_macro.csv');diag.to_csv(out/'diagnostics_subjects.csv',index=False)
    from matplotlib import font_manager
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    figures=[]
    def save(fig,name,title,caption):
        fig.savefig(out/(name+'.png'),dpi=220,bbox_inches='tight',facecolor='white');plt.close(fig);figures.append(dict(name=name,title=title,caption=caption))
    fig,axs=plt.subplots(1,2,figsize=(12,5.4),layout='constrained')
    for ax,m in zip(axs,['eeg','hb']):
        v=np.array([[gain[arm,t,m] for t in TASKS] for arm in ARMS]);im=ax.imshow(v,vmin=-.8,vmax=3,cmap='RdYlGn');ax.set_xticks(range(3),TL,rotation=20,ha='right');ax.set_yticks(range(7),LABELS);ax.set_title(m.upper()+'：相对频率先验的补全增益')
        for i in range(7):
            for j in range(3):ax.text(j,i,f'{v[i,j]:.3f}',ha='center',va='center')
    fig.colorbar(im,ax=axs,label='先验 CE − 模型 CE（bit/ID）；越高越好',shrink=.85)
    save(fig,'01_masking','空间交互提供主要收益，单一遮挡训练存在迁移代价','每格为三个种子、S/O 分支平均后，先被试内汇总，再四库等权的增益。频率先验只用训练数据集×测点×时间槽计数并加 0.5 平滑。绿色为改善，负值表示劣于先验；两模态共用色标。列是评价遮挡，行是训练方案。所有方案隐藏约 50% 有效位置时间单元，S/O 同时隐藏；连续时间块约为 14–16 秒，空间遮挡为整段位置优先并按有效目标数补齐。')
    fig,axs=plt.subplots(1,3,figsize=(13,5.2),layout='constrained')
    for ax,c,title in zip(axs,list(mapping)[:3],['空间交互：带位置时空 − 仅时间','位置信息：带位置 − 匿名','跨模态：联合 − 模态内']):
        sub=contr[contr.contrast==c]
        for k,t in enumerate(TASKS):
            for j,m in enumerate(['eeg','hb']):
                r=sub[(sub.task==t)&(sub.modality==m)].iloc[0];y=k+(j-.5)*.22
                ax.errorbar(r.gain_bits,y,xerr=[[r.gain_bits-r.CI95_low],[r.CI95_high-r.gain_bits]],fmt='o' if j==0 else 's',color=['#2464ad','#ba4f15'][j],capsize=3,label=m.upper() if k==0 else None)
        ax.axvline(0,color='gray',ls='--');ax.set_yticks(range(3),TL);ax.invert_yaxis();ax.set_title(title,fontsize=11);ax.set_xlabel('对照 CE − 处理 CE（bit/ID）');ax.legend();ax.grid(axis='x',alpha=.2)
    save(fig,'02_contrasts','空间收益稳定，跨模态联合在六项比较中均略差','点为配对改善，线为 95% 探索性逐项 bootstrap 区间：先平均同一被试的三个种子与 S/O，再在各库内重采样被试，最后四库等权。共 20 名验证被试；窗口和种子不增加独立被试数，未作多重比较校正。三个子图使用不同横轴范围，以呈现量级差异。位置比较同时移除或加入测点身份 embedding 与坐标，不能单独归因为几何坐标。')
    fig,axs=plt.subplots(1,2,figsize=(12,5.2),layout='constrained')
    colors=['#2464ad','#c17827','#737b83','#8d4ba7']
    for ax,m in zip(axs,['eeg','hb']):
        x=np.arange(7)
        for j,(reader,label) in enumerate([('pretrained','预训练表示'),('ID','原双 ID'),('untrained','随机初始化'),('window_bag','窗口词频')]):
            ax.plot(x,[probe[arm,m,reader] for arm in ARMS],marker=['o','s','^','D'][j],label=label,color=colors[j])
        ax.set_xticks(x,LABELS,rotation=35,ha='right');ax.set_title(m.upper()+'：8 项观测特征平均误差');ax.set_ylim(.2,1.2);ax.set_ylabel('训练特征方差归一化 MSE（越低越好）');ax.grid(alpha=.2);ax.legend(fontsize=9)
    save(fig,'03_probes','Hb 读出收益更明确；EEG 平均收益接近原 ID','输入为完整未遮挡 token。冻结模型后，每库每模态单独用训练被试拟合 ridge（α=0.01），预测对应 2 秒观测 X 特征；验证不拟合任何系数。所有读出都附加同样的测点 one-hot。原双 ID 为 S/O one-hot 拼接，窗口词频忽略当前时间位置。按库、被试、特征和种子汇总；每记录前两个窗口、每模态至多四个按索引均匀选取的位置，属于解释性子面板。NMSE 分母来自训练特征方差，不等同于验证 R²；1 不是精确的验证均值预测基线。')
    fig,axs=plt.subplots(1,2,figsize=(12,5.4),layout='constrained')
    for ax,m in zip(axs,['eeg','hb']):
        x=np.arange(8)
        for arm,label,mark in [('spatial','带位置混合遮挡','o'),('spatial_random','带位置随机遮挡','s'),('anonymous','匿名混合遮挡','^')]:ax.plot(x,[features[arm,m,'pretrained',i] for i in range(8)],marker=mark,label=label)
        ax.plot(x,[features['spatial',m,'ID',i] for i in range(8)],'k--D',label='原双 ID')
        ax.set_xticks(x,FEATURES[m],rotation=35,ha='right');ax.set_title(m.upper());ax.set_ylim(0,1.65);ax.set_ylabel('特征 NMSE（越低越好）');ax.grid(alpha=.2);ax.legend(fontsize=9)
    save(fig,'04_features','功率、均值和动态特征需要分别判断','EEG 功率为 2 秒 Hann 周期图在 1–4、4–8、8–13、13–30、30–45 Hz 的积分后取 log10；RMS 亦取 log10。Hb 保留双色团各自均值、总体 SD、以秒为单位的 OLS 斜率及末点减首点。这里预测的是冻结尺度下的观测描述量。Hb 斜率与端点差高度相关，不能视为两项独立发现。较低的读出误差说明特征更容易被线性提取；本轮没有更新 tokenizer 或检验单个码字对应独立生理机制。')
    fig,axs=plt.subplots(1,2,figsize=(12,5.5),layout='constrained')
    for ax,m in zip(axs,['eeg','hb']):
        sub=b[(b.modality==m)&(b.reader=='pretrained')];base=b[(b.modality==m)&(b.arm=='spatial')&(b.reader=='ID')].groupby('dataset').NMSE.mean()
        arms=['temporal','anonymous','spatial','joint','spatial_random','spatial_span','spatial_space'];v=np.array([[100*(base[d]-sub[(sub.arm==arm)&(sub.dataset==d)].NMSE.mean())/base[d] for d in DS] for arm in arms])
        im=ax.imshow(v,vmin=-30,vmax=30,cmap='RdYlGn');ax.set_xticks(range(4),DL,rotation=30,ha='right');ax.set_yticks(range(7),LABELS);ax.set_title(m.upper())
        for i in range(7):
            for j in range(4):ax.text(j,i,f'{v[i,j]:+.1f}%',ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=axs,label='相对原双 ID 的特征平均 NMSE 降幅（%）',shrink=.85)
    save(fig,'05_datasets','收益具有数据集差异，REFED Hb 未获得读出改善','先在每库内平均 8 项特征、被试与三个种子，再计算（原 ID 误差−预训练表示误差）/原 ID 误差。正值表示改善。图中最小值也保留真实标注；同一色标上下限为 ±30%。REFED Hb 原 ID 的平均 NMSE 约 0.022，分母很小，相对退化须结合绝对误差理解；Simultaneous Hb 的绝对误差大，对四库等权 NMSE 的量级影响较大。')
    fig,axs=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    names=['time_permutation','space_permutation','other_modality_removed'];labs=['打乱可见时间内容','打乱可见位置内容','移除另一模态']
    for ax,m in zip(axs,['eeg','hb']):
        for j,arm in enumerate(['spatial','joint']):
            v=macro(diag[(diag.arm==arm)&(diag.modality==m)],['perturbation'],'delta_bits');xx=np.arange(3)+(j-.5)*.3
            bars=ax.bar(xx,[v[n] for n in names],.3,label=['模态内','联合'][j]);ax.bar_label(bars,fmt='%.4f',fontsize=9)
        ax.set_xticks(range(3),labs,rotation=15);ax.set_ylabel('扰动后 CE − 原 CE（bit/ID）');ax.set_title(m.upper());ax.set_ylim(0,3.4);ax.legend();ax.grid(axis='y',alpha=.2)
    save(fig,'06_perturbations','模型依赖时空排列，对另一模态的依赖很小','在随机遮挡及读出子面板上，保持目标、评分掩码不变，只置换原本可见的 S/O ID 对，绝不把隐藏标签移回输入。每条柱是按记录→被试→数据集及种子平均的误差增加。时间置换在同位置进行；位置置换在同模态、每个时间槽内进行，也会改变该位置的时间轨迹。该诊断表明模型使用了排列信息，但扰动为分布改变，不能据此定量分解独立生理贡献。')
    sources=['masked_subjects.csv','paired_contrasts.csv','physiology_subjects.csv','physiology_subjects.csv','physiology_subjects.csv','models/*/seed_*/context_diagnostics.json']
    for f,source in zip(figures,sources):f['caption']+=' 来源：'+source+'。'
    intro='21/21 个模型均完成 10,000 步，共 210,000 次更新；训练失败为零。2026-11-14 00:29（香港时间）后台服务正常完成。60 名训练被试、20 名验证被试；7,797/7,805 个输入可用，8 个缺失来自原缓存四个 EEG 不可用父窗。固定 seed17、K128 tokenizer：EEG dense_long，Hb waveform；编码 30 秒，每 2 秒每位置输出 S/O 两个 ID。预训练仅优化 masked ID 交叉熵，编码器和码本未更新，未读取 test/protected。'
    conclusion='结论：模态内空间建模明显改善 token 补全；位置身份与坐标对 EEG 有稳定收益，对 Hb 的收益取决于遮挡任务。联合跨模态模型未超越强模态内对照。Hb 的动态观测特征更容易从部分预训练表示读出，EEG 的频带功率与局部均值/斜率呈取舍。推荐以带位置混合遮挡作为通用结构基线，以随机遮挡和匿名时空模型作为 Hb 读出的竞争候选；保留现有原 ID 读出，后续用独立面板确认候选。'
    body='<h1>固定词表掩码预训练：核验与可视化分析</h1><p>'+intro+'</p><p>'+conclusion+'</p>'
    for f in figures:body+=f'<section><h2>{f["title"]}</h2><img src="{f["name"]}.png"><p>{f["caption"]}</p></section>'
    body+='<h2>完整数值与来源</h2>'+ce.unstack(['modality','task']).round(4).to_html()+probe.unstack('reader').round(4).to_html()+'<p>来源：父运行 summary.json、masked_subjects.csv、physiology_subjects.csv、各模型的逐窗结果；冻结实现 source/ 下的 runner、foundation 与 data owner。verification.json 独立复算全部 21 个模型的窗口→记录→被试汇总及 36 个配对点估计；不覆盖历史证据。</p>'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>掩码预训练结果</title><style>body{font:17px sans-serif;line-height:1.8;max-width:1200px;margin:40px auto;color:#182335}img{width:100%}section{margin:40px 0}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse;font-size:14px}</style>'+body)
    pdf=fitz.open();page=pdf.new_page(width=1000,height=760);page.insert_textbox(fitz.Rect(45,40,955,110),'固定词表掩码预训练：结果核验',fontname='china-s',fontsize=24)
    page.insert_textbox(fitz.Rect(45,130,955,680),intro+'\n\n'+conclusion+'\n\n指标：CE = −log₂ p(真实 ID)；增益 = 对照 CE − 模型 CE。特征 NMSE = 验证均方误差 / 训练特征方差。完整图表、配置与数值见配套 HTML。',fontname='china-s',fontsize=17)
    for f in figures:
        page=pdf.new_page(width=1000,height=760);page.insert_textbox(fitz.Rect(35,20,965,62),f['title'],fontname='china-s',fontsize=21)
        page.insert_image(fitz.Rect(35,72,965,610),filename=str(out/(f['name']+'.png')),keep_proportion=True)
        leftover=page.insert_textbox(fitz.Rect(40,620,960,744),f['caption'],fontname='china-s',fontsize=12)
        assert leftover>=0,'caption clipped'
    pdf.save(out/'RESULTS.pdf');pdf.close()
    doc=fitz.open(out/'RESULTS.pdf');assert len(doc)==7 and all(len(doc[i].get_images())==1 for i in range(1,7));assert len(''.join(p.get_text() for p in doc))>1000
    doc[4].get_pixmap(matrix=fitz.Matrix(1.2,1.2)).save(out/'pdf_preview.png')
    audit.update(pdf_pages=7,bitmap_figures=6,selectable_text=True);write(out/'verification.json',audit)
    print(json.dumps(audit,ensure_ascii=False))

if __name__=='__main__':main()
