#!/usr/bin/env python3
"""Export the dated observation-codec review from completed public development results only.

No loader, checkpoint, fitting or model inference. The only arrays opened are the four
named, already-exported illustration packages in ``example_figure``. HTML embeds its
own PNGs; Chromium preserves the surrounding Chinese text and tables in the PDF.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path
import re

import fitz
import numpy as np
import pandas as pd
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / 'experiments/runs/physiology_semantic_tokenizer/observation_factorization'
P, A, G = (SUITE / f'20261120_{s}' for s in ('development_v3','allocation_v1','gain_v1'))
DEST = SUITE / '20261120_baseline_review_v1'
HISTORY = ROOT / 'experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261120_input_quantization_b_v1/phase_ab_review_v1/review_values.json'
TITLE = '观测词与整数细节：EEG／Hb 开发基线决策'
STYLE = {'font.family':['Noto Sans CJK JP','DejaVu Sans'], 'font.size':9.5, 'axes.spines.right':False,
         'axes.spines.top':False, 'axes.edgecolor':'#a9b8c0', 'axes.labelcolor':'#254053',
         'xtick.color':'#435868', 'ytick.color':'#435868', 'text.color':'#203a4b', 'axes.unicode_minus':False}
ESC = lambda s: html.escape(str(s))


def js(path):
    return json.loads(path.read_text(encoding='utf-8'))


def fin(path):
    data = js(path/'summary.json')
    if data.get('status') != 'completed' or js(path/'controller.json').get('status') != 'completed':
        raise ValueError(f'incomplete source: {path.name}')
    return data


def tab(h, rows, cls=''):
    return (f'<table class="{cls}"><thead><tr>'+''.join('<th>'+ESC(x)+'</th>' for x in h)+'</tr></thead><tbody>'
            +''.join('<tr>'+''.join('<td>'+ESC(c)+'</td>' for c in r)+'</tr>' for r in rows)+'</tbody></table>')


def note(s, cls=''):
    return f'<p class="{cls}">{s}</p>'


def box(title, s):
    return f'<aside><h3>{title}</h3><p>{s}</p></aside>'


def row(df, mod, method, bits, level='macro'):
    v = df[(df.modality==mod)&(df.method==method)&(df.level==level)&(df.bits_per_patch==bits)]
    if len(v)!=1:
        raise ValueError(f'expected one retained {mod}/{method}/{bits}/{level}, found {len(v)}')
    return v.iloc[0]


def signed(x, p=3):
    return f'{x:+.{p}f}'.replace('-', '−')


def ci(delta, interval, p=3, multiplier=1):
    return f'{signed(delta*multiplier,p)} [{signed(interval[0]*multiplier,p)}, {signed(interval[1]*multiplier,p)}]'


def pickcontrast(summary, mod, reference, candidate):
    return next(c for c in summary['modalities'][mod]['contrasts'] if c['reference']==list(reference) and c['candidate']==list(candidate))


def png(fig):
    stream = io.BytesIO(); FigureCanvasAgg(fig); fig.savefig(stream, format='png',dpi=230,facecolor='white')
    return stream.getvalue()


def embedded(data, width, label):
    # Re-export the whole original figure at 230 dpi at its visible width. Never crop/recompose atlas content.
    with Image.open(io.BytesIO(data)) as source:
        target = round(width / 25.4 * 230)
        img = source.convert('RGB')
        if img.width > target:
            img = img.resize((target,round(img.height*target/img.width)),Image.Resampling.LANCZOS)
        out = io.BytesIO(); img.save(out,format='PNG',dpi=(230,230))
    return '<img alt="'+ESC(label)+'" style="width:'+str(width)+'mm" src="data:image/png;base64,'+base64.b64encode(out.getvalue()).decode()+'">'


def curve(allocation):
    with rc_context(STYLE):
        fig=Figure(figsize=(6.6,3.62)); axes=fig.subplots(1,2,sharey=True)
        fig.subplots_adjust(left=.11,right=.98,top=.76,bottom=.20,wspace=.15)
        for ax,mod,label in zip(axes,('eeg','hb'),('EEG / 电极','Hb / 双色团位点')):
            d=allocation[(allocation.modality==mod)&(allocation.level=='macro')]
            for k,color,mark,ls,name in [('adaptive_pattern','#0b736a','o','-','属性路由'),('flat','#35599f','s','--','无词 flat'),
                                         ('adaptive_blind','#835f90','^',':','盲路由'),('legacy_hard','#ad4f26','D','none','旧双 K128')]:
                s=d[d.method==k].sort_values('bits_per_patch'); ax.plot(s.bits_per_patch,s.x_nmse,label=name,marker=mark,color=color,ms=5,mfc='white',mew=1.3,ls=ls,lw=1.65)
            ax.set_xscale('log',base=2); bits=sorted(d.bits_per_patch.unique()); ax.set_xticks(bits,[str(int(x)) for x in bits],fontsize=8)
            ax.xaxis.set_minor_locator(NullLocator()); ax.set_yscale('log'); ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}')); ax.set_title(label,loc='left',fontsize=11,weight='bold')
            ax.grid(axis='y',color='#d5dfe4',linewidth=.55); ax.set_xlabel('bit / 位点 / 2 秒'); ax.margins(x=.1,y=.12)
        axes[0].set_ylabel('验证 X NMSE（越低越好）'); fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=4,frameon=False,fontsize=9,columnspacing=1.05)
        return png(fig)


def example_figure():
    ds='refed'; quantities=[]; identity={}; traces={}
    for mod in ['eeg','hb']:
        meta=pd.read_csv(P/f'{mod}_first_per_dataset.csv',dtype={'subject':str}); ix=meta.index[meta.dataset==ds]; assert len(ix)==1
        m=meta.loc[ix[0]]; identity[mod]=m.to_dict()
        a=js(A/f'{mod}_decision.json'); picked=next(v for v in a['examples'] if v['dataset']==ds)
        for k in ['dataset','record','subject','parent','start_s','position']:
            assert str(picked[k])==str(m[k]), (mod,k)
        with np.load(P/f'{mod}_first_per_dataset.npz',allow_pickle=False) as p, np.load(A/f'{mod}_refed_example.npz',allow_pickle=False) as q:
            patches=q['patch']; assert np.array_equal(np.flatnonzero(p['score_mask'][ix[0]]),patches)
            x=q['x']; assert np.array_equal(p['x'][ix[0],patches],x)
            arrays={'x':x,'old':p['legacy_hard'][ix[0],patches],'new':q['pred']}; traces[mod]=(patches,arrays,picked)
            for name in ['old','new']:
                nmse=(((arrays[name]-x)**2).mean(-1)/q['vx']).mean()
                quantities.append((mod,name,float(nmse)))
    assert all(identity['eeg'][k]==identity['hb'][k] for k in ['parent','record','subject','start_s'])
    valid=set(traces['eeg'][0]) & set(traces['hb'][0]); anchor=min(valid, key=lambda i:(abs(i-7),i)); start2=2*int(anchor)
    pp,ee,_=traces['eeg']; current=np.flatnonzero(pp==anchor).item(); seg=ee['x'][current,0]; j=int(abs(seg-seg.mean()).argmax())
    peak=[float(ee[k][current,0,j]) for k in ('x','old','new')]; _,hh,_=traces['hb']
    changes=[[float(hh[k][-1,c,-5:].mean()-hh[k][0,c,:5].mean()) for k in ('x','old','new')] for c in [0,1]]
    landmarks=dict(time=start2+j/200, peak=peak, changes=changes)
    with rc_context(STYLE):
        fig=Figure(figsize=(10.85,3.0)); axes=fig.subplots(1,3,gridspec_kw={'width_ratios':[1.42,1,1]}); fig.subplots_adjust(left=.065,right=.99,top=.74,bottom=.22,wspace=.28)
        for ax,mod,c,label in zip(axes,['eeg','hb','hb'],[0,0,1],['EEG  /  CB1','HbO  /  CH49','HbR  /  CH49']):
            pp,aa,entry=traces[mod]; sr=200 if mod=='eeg' else 10; ax.set_title(label,loc='left',fontsize=11,weight='bold')
            for k,color,linename,name,width in [('x','#243f52','-','观测 X',1.6),('old','#b45728','--','旧双 K128（14 bit）',1.3),('new','#1a8a82','-.','属性路由＋自适应整数细节',1.45)]:
                allwaves=np.full((15,2*sr),np.nan); allwaves[pp]=aa[k][:,c]; y=allwaves.ravel(); times=np.arange(len(y))/sr
                use=(times>=start2)&(times<start2+2) if mod=='eeg' else np.ones(len(times),bool)
                ax.plot(times[use],y[use],color=color,linestyle=linename,lw=width,label=name)
            ax.grid(axis='y',color='#d5dfe4',lw=.6); ax.set_xlabel('相对 30 秒输入起点（秒）'); ax.locator_params(axis='y',nbins=4)
        axes[0].set_ylabel('训练 X 尺度'); fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,fontsize=9.5)
        return png(fig),{(m,name):v for m,name,v in quantities},start2,landmarks


def page(n, tag, title, sub, body, source):
    return (f'<section><header><div><strong>研究决策</strong>　/　多模态生理观测编码</div><div>2026.11.20　·　公开开发集　·　{n:02}/05</div></header>'
            f'<main><div class="eyebrow">{tag}</div><h1>{title}</h1><p class="dek">{sub}</p>{body}</main>'
            f'<footer><b>证据</b>　{source}</footer></section>')


CSS = '''@page{size:297mm 210mm;margin:0}*{box-sizing:border-box}html,body{margin:0;padding:0;color:#203a4b;font-family:"Noto Sans CJK SC","Noto Sans CJK JP",sans-serif}body{background:#ecefed;font-size:9.9pt;line-height:1.48}section{position:relative;width:297mm;height:210mm;padding:7mm 10mm 10mm;background:white;break-after:page;page-break-after:always;margin:0 auto 5mm}section:last-child{break-after:auto}header,footer{display:flex;justify-content:space-between;align-items:center;color:#536875;font-size:8pt}header{padding-bottom:1.7mm;border-bottom:1px solid #cfdbdf}header strong,.eyebrow,footer b{color:#006f69}.eyebrow{font-size:8.2pt;letter-spacing:.07em;font-weight:bold;margin:3.2mm 0 .8mm}h1{font-size:20pt;line-height:1.24;margin:0;color:#183747;letter-spacing:.015em}.dek{margin:1.5mm 0 3mm;font-size:10.4pt;color:#415969;line-height:1.45}h2{font-size:11.6pt;margin:3.2mm 0 1mm;line-height:1.35;color:#095b60}h2.first{margin-top:0}h3{font-size:10.4pt;color:#14595e;margin:0 0 .9mm}p{margin:1.1mm 0 2mm}.small{font-size:8.4pt;line-height:1.43;color:#435967}.mild{color:#435967}.grid{display:grid;column-gap:6mm;align-items:start}.half{grid-template-columns:1fr 1fr}.col2{grid-template-columns:1.05fr 1fr}.atlas-layout{grid-template-columns:187mm 1fr;gap:5mm}.rate-layout{grid-template-columns:148mm 1fr;gap:5mm}.call{padding:2.4mm 3mm;background:#eff6f5;border-left:2.3px solid #008b79}.cards{display:grid;grid-template-columns:1fr 1fr 1fr;gap:3.1mm}aside{background:#f1f5f3;padding:2.5mm 3mm;border-top:1.8px solid #7bafaa}aside p{margin:0}.tally{font-size:14pt;color:#006e68;font-weight:700}table{width:100%;border-collapse:collapse;margin:1mm 0 2mm;font-size:9.3pt;font-variant-numeric:tabular-nums;line-height:1.33}th{text-align:left;background:#ecf2f4;color:#284d5b;font-weight:600}th,td{padding:1.4mm 1.7mm;border-bottom:1px solid #dce5e8;vertical-align:middle}tbody tr:nth-child(even){background:#f7f9f9}.compact{font-size:8.5pt}.compact th,.compact td{padding:1.1mm 1.35mm}.tight{font-size:8pt}.tight th,.tight td{padding:.85mm 1.1mm}img{height:auto;display:block;max-width:100%;margin:0}.rule{border-top:1px solid #cad6d8;padding-top:1.5mm}footer{position:absolute;bottom:3.5mm;left:10mm;right:10mm;border-top:1px solid #d3dfe2;padding-top:1.2mm;display:block;font-size:7.35pt;line-height:1.3;color:#536875}p b,td b{font-weight:700}strong{color:#153f49}@media print{html,body{background:white}section{margin:0;overflow:visible;-webkit-print-color-adjust:exact;print-color-adjust:exact}}'''


def gain_facts():
    """Show the prespecified operating point; a last-rate fallback is not a pass."""
    result=fin(G); d=result['modalities']['eeg']; bits=d['selected_bits']; table=pd.read_csv(G/'gain_shape.csv')
    if (result['experiment_id']!='TK-OBS-GAIN-SHAPE-v1' or d['method']!='gain_conditional' or d['n_validation_patches']!=4322
            or d['n_validation_subjects']!=20 or d['fixed_ID_checks']['mismatches'] or not table['split'].eq('validation').all()):
        raise ValueError('gain source or matched validation denominator changed')
    pt=row(table,'eeg','gain_conditional',bits)
    if not all(np.isclose(pt[k],d[k]) for k in ['x_nmse','centered_cosine','band_30_45_db','band_30_45_cosine']):
        raise ValueError('gain decision does not match retained table')
    passed=d['met_all_screens']; local=d['single_trial']; rows=[]
    for name,label in [('gain_conditional','EEG／量化水平＋增益'),('gain_independent','同率／词不参与解码')]:
        m=row(table,'eeg',name,bits); rows.append((label,bits,f'{m.x_nmse:.4f}',f'{m.centered_cosine:.3f}',
          f'{signed(m.band_30_45_db,2)} dB / {m.band_30_45_cosine:.3f}',('通过宏护栏' if passed else '末位仍未达标') if name=='gain_conditional' else '机制对照'))
    f=row(table,'eeg','flat',bits)
    con=next(r for r in d['contrasts'] if r['metric']=='x_nmse' and r['reference']=='flat' and r['candidate']=='gain_conditional')
    delta=ci(con['delta'],[con['ci95_low'],con['ci95_high']],4)
    bins=pd.read_csv(G/'rms_bins.csv'); low=bins[(bins.level=='macro')&(bins.rms_bin=='low')&(bins.bits_per_patch==bits)].set_index('method')
    lowr=low.loc['gain_conditional']; plain=low.loc['adaptive_pattern']
    return dict(mechanism=f'同 {bits} bit 时，低去均值 RMS 档（训练下四分位阈值；{int(lowr.n_patches):,} 片、{int(lowr.n_subjects)} 人、{int(lowr.n_datasets)} 库）的高频偏差由 {signed(plain.band_30_45_db,2)} 变为 {signed(lowr.band_30_45_db,2)} dB。幅值编码可能缓解小信号的过量重建；不是唯一机制的证明。', opening=(f'EEG 将水平、增益分别量化为 8 位、6 位，固定 4 位词；在预设格点中首次于 {bits} bit 通过四项宏护栏。' if passed
                         else f'EEG 另传量化水平与增益后，在最多 {bits} bit 仍未通过全部宏护栏，不晋级。'),
                rows=rows, compared=f'EEG 最后方案相对 {bits} bit flat（{f.x_nmse:.4f}）的配对差 {delta}；Single-Trial 高频功率偏差 {signed(local["no_gain_signed_db"],1)} → {signed(local["signed_db"],1)} dB。')


def compose(gain):
    ps,al=fin(P),fin(A); rd=pd.read_csv(P/'reconstruction.csv'); alloc=pd.read_csv(A/'allocation.csv')
    attr=pd.read_csv(P/'attributes.csv').query("level=='macro'"); atlas=pd.read_csv(P/'atlas.csv'); task=js(P/'task_probe.json'); pc=js(P/'paired_task_contrasts_v1.json'); hist=js(HISTORY)
    assert ps['experiment_id']=='TK-OBS-FACTOR-v1' and al['experiment_id']=='TK-OBS-ALLOCATION-v1' and task['status']==pc['source_status']=='completed'
    assert rd['split'].eq('validation').all() and alloc['split'].eq('validation').all() and set(atlas.split)=={'train','validation'}
    assert {i['subjects'] for i in ps['panel'] if i['split']=='validation'}=={20}
    aE = row(alloc,'eeg','adaptive_pattern',1024); aH = row(alloc,'hb','adaptive_pattern',32)
    D={tuple(i[:2]):i[2] for i in hist['deltas']}; panels={(x['modality'],x['split']):x for x in ps['panel']}; text=[]
    body='<div class="cards">'+box('① 保留可解释的局部观测词','每个位置每 2 秒先传一个 <b>4 bit</b> 整数。EEG 按六个相对频带功率分词；Hb 将 HbO/HbR 的水平与斜率一起分词。词描述测量模式，不命名生理源。')+box('② 要重建原波形，再传整数细节', '仅用模式词不承诺相位。按训练字典，在 DCT 变换域量化剩余波形；路由、幅值或增益等实际发送的位全计入总额。接收端不读取原始 X、浮点尺度或标签。')+box('③ 分开判断三种成效','波形压缩看同位率 X 重建；词的可读性看未直接用于聚类的观测属性；任务效用看冻结跨被试分类。三项均不能代替生理分离或神经血管耦合检验。')+'</div>'
    body+='<h2>旧模型为什么值得拆开，而不是继续扩大 K128 码本？</h2>'
    body+=tab(['已完成的诊断','验证结果','此结果约束哪种解释'],[
        ('A ｜同 ID 的真实片段','旧 EEG 单码加入库／位置／RMS 背景后，原相位 S/O 波形误差下降 21.6% / 45.7%；对 30–45 Hz 功率却为 −1.8% / −0.8% 的增益。','波形可读不保证高频保留；占用率不是语义证据。'),
        ('B ｜同架构，输入 × 量化',f'EEG：连续 X 输入相对分支目标输入差 +{D["EEG","连续输入差"]:.4f}、X 路径硬码差 +{D["EEG","X 量化差"]:.4f}；Hb 对应 +{D["Hb","连续输入差"]:.4f}、+{D["Hb","X 量化差"]:.4f}。','旧指标是 X/S/O 等权宏 NMSE；不是 X-only 误差下界，直接输入 S/O 仅是诊断。'),
        ('C ｜落到分支和色团','连续输入差集中于 EEG-S（0.08090）与 HbO-O / HbR-O（0.42953 / 0.30138）。硬码组间 HbR-O 序列形状改善、单 ID 形状却变差，20/20 人同向。','整段能够解码的形状，不一定由当前词独立携带。')], 'compact')
    body+='<div class="grid half"><div><h2>本轮只对观测 X 评分</h2>'+note(f'四个数据库、60 名训练被试、20 名不重复验证被试；每被试最多 3 条记录、每记录最多 3 个不重叠 30 秒窗，每窗最多 2 个位置。训练/验证有效片段：EEG {panels["eeg","train"]["available_patches"]:,}/{panels["eeg","validation"]["available_patches"]:,}，Hb {panels["hb","train"]["available_patches"]:,}/{panels["hb","validation"]["available_patches"]:,}。')+'</div><div><h2>工程出口，不开放新测试</h2>'+note(gain['opening']+' Hb 采用 32 bit 的双色团联合候选。旧 14 bit 编码仍是低码率总误差参照；少数已查看的开发被试不能作为独立确认。')+'</div></div>'
    body+='<h2>复现边界：字典如何得到、解码器看得到什么</h2>'+note('来源为 Single-Trial、REFED、Simultaneous、Visual 四库已完成的 G0 开发数据。每模态主词表只在训练集按库、被试、记录、窗、位置逐层均衡抽取 16,384 个片段（可重复），用种子 17 聚类 16 个模式；另两个种子只检查稳定性。验证 298 个窗口×位置中，由于不完整评分支持，EEG 排除 148/4,470 片，Hb 排除 112/4,470 片。无教师标签补全、无新测试。','small')+note('EEG 的每片 400 点分为四段，作离散余弦变换（DCT）；Hb 的每色 20 点先化为双色团和／差，再作 DCT。原等字长方案量化固定数量的系数；自适应方案由训练残差方差分配各系数的整数位数，允许按模式选字典。不计熵编码优惠，也不能把 30 秒上下文与 2 秒局部模型称作同感受野。','small')
    body+=note('三种失真读法：总 NMSE 保留偏置和幅度；片内去均值余弦只评价时间形状；30–45 Hz 的 dB 先在各 2 秒片去均值、乘 Hann 窗，取带内重建功率与观测功率之比的 10 倍常用对数，再分层汇总。复数余弦是带内复谱内积实部除以两谱范数乘积，兼顾相位；功率接近 0 dB 不代表相位已恢复。','small')
    body+=note(gain['mechanism']+' 带内复谱余弦不是跨样本或跨时间谱估计的相干度。','small')
    text.append(page(1,'01　/　决策摘要','把观测模式与细节拆开；不把观测词叫作生理来源','同一套码字不必既命名慢趋势，又精确还原快电压相位。三套完成的公开开发实验分别回答：能否读出属性、能否压缩波形、能否用于真实任务。',body,'旧 A/B：20261120_pattern_a_v1、20261120_input_quantization_b_v1/phase_ab_review_v1；C：phase_c_v1/phase_c.xlsx。新基线：20261120_development_v3/summary.json。'))
    body='<div class="grid rate-layout"><div>'+embedded(curve(alloc),148,'同位率波形比较')+note('曲线仅连接实际预算。左右横轴范围不同，纵轴共用对数尺度；旧模型看 30 秒上下文，新模型只看当前 2 秒。盲路由以带符号波形随机投影聚类；flat 不发属性词，将其 4 位也用作细节。','small')+'</div><div><h2 class="first">每 2 秒每位置 14 bit：旧模型总误差仍低</h2>'
    body+=tab(['同一 X 上的 NMSE ↓','EEG','Hb'],[(name,f'{row(alloc,"eeg",method,14).x_nmse:.4f}',f'{row(alloc,"hb",method,14).x_nmse:.4f}') for name,method in [('旧双 K128（2 × 7 位）','legacy_hard'),('4 位词＋等字长细节','full'),('4 位词＋自适应细节','adaptive_pattern'),('无词、全预算 flat','flat'),('盲路由＋自适应细节','adaptive_blind')]],'compact')
    t0=pickcontrast(al,'eeg',('legacy_hard',14),('adaptive_pattern',14)); t1=pickcontrast(al,'hb',('legacy_hard',14),('adaptive_pattern',14)); body+=note('自适应减旧模型的配对误差差：EEG '+ci(t0['candidate_minus_reference'],t0['ci95'])+'，Hb '+ci(t1['candidate_minus_reference'],t1['ci95'])+'；正值表示误差更高。','small')+'</div></div>'
    body+='<h2>有更多比特时，功率、相位和总波形必须同时核验</h2>'
    r=tab(['模态／细节实现','总 bit / 2 秒','X NMSE ↓','去均值余弦 ↑','30–45 Hz：功率偏差 / 复数余弦','开发判定'],[
        ('EEG／原固定字长',512,f'{row(rd,"eeg","full",512).x_nmse:.4f}',f'{row(rd,"eeg","full",512).centered_cosine:.3f}','−3.26 dB / 0.374','未过功率护栏'),
        ('EEG／自适应分配',1024,f'{aE.x_nmse:.4f}',f'{aE.centered_cosine:.3f}',f'{signed(aE.band_30_45_db,2)} dB / {aE.band_30_45_cosine:.3f}','未过功率护栏'), *gain['rows'],
        ('Hb／自适应分配',32,f'{aH.x_nmse:.4f}',f'{aH.centered_cosine:.3f}','不适用；HbO/HbR 等权','通过波形护栏')],'compact'); body+=r
    hc=pickcontrast(al,'hb',('flat',32),('adaptive_pattern',32)); body+=note(f'总误差为每片每色平方误差除冻结训练方差，先等权平均色团，再按位置、窗、记录、被试、库汇总。门槛：NMSE ≤ 0.10、去均值余弦 ≥ 0.80，EEG 另需 |高频偏差| ≤ 3 dB，后两轮并要求高频复数余弦 ≥ 0.70。Hb 32 bit 相对同率 flat 差为 {ci(hc["candidate_minus_reference"],hc["ci95"],4)}；不能称为全面胜出。','small')
    body+=note(gain['compared']+' 区间：库内成对重抽验证被试 4,000 次，四库等权，点态 95%，未校正多重比较。','small')
    text.append(page(2,'02　/　压缩与位率','14 bit 仍输给旧双词表；提高位率后分开核验失真','比较相同验证片段的观测 X，不比较模型条件 S/O。以下 bit 均为每 2 秒每 EEG 电极或每 HbO/HbR 位置对的固定有效载荷；固定字典共享，位率不是压缩文件大小。',body,'20261120_development_v3/reconstruction.csv；20261120_allocation_v1/allocation.csv、summary.json；20261120_gain_v1/gain_shape.csv、summary.json。'))
    summary_atlas=atlas.drop_duplicates(['modality','split','pattern_id']); aris=[]
    for mod in ('eeg','hb'):
        vals=np.array([v for ds in ps['modalities'][mod]['per_dataset_ARI'].values() for v in ds.values()]); aris.append(f'{vals.min():.2f}–{vals.max():.2f}')
    side='<h2 class="first">独立读出：新词增加了什么？</h2>'+note('只在训练被试拟合岭回归。对比相同库、位置、局部 RMS 及其变换的背景；Hb 背景还含双色团水平。评分目标不是分词输入本身。','small')
    rr=[]
    for z in attr.itertuples(): rr.append(('EEG' if z.modality=='eeg' else 'Hb','同片' if z.lag_patches==0 else '下一片',f'{100*z.pattern_gain:.1f}%',f'{100*z.rolled_pattern_gain:.1f}%'))
    side+=tab(['模态','目标','正确词','错位词'],rr,'tight')+note('表中增益 = 1 −（背景＋ID）的标准化误差 / 背景误差。错位词将同窗、同位置的验证词移半窗。EEG 预测六个时滞自相关，Hb 预测片段末与首各 0.5 秒均值之差。','small')
    side+='<h2>覆盖不是逐码复现</h2>'+note('验证 EEG 16/16 码出现，但最稀码只见于 2 人。Hb 14/16：ID 0、11 验证无人；它们不是可推广的稀有生理状态。','small')+note(f'3 个随机种子、4 库的成对调整兰德指数 ARI：EEG {aris[0]}，Hb {aris[1]}。ARI 比较同一批片段的分组，不依赖 ID 编号；种子不增加被试数。','small')
    side+=note('同片自相关与谱、Hb 首末差与斜率数学相关，不能因“换目标”就称独立生理语义。EEG 下一片正确词 13.1%、错位仍 10.1%，应警惕时间背景。','call small')
    side+=note('图例：横轴或纵轴的 0–15 是名义编号。EEG 的 ID 0 集中在 45–100 Hz，验证仅 3 人；不能命名为神经节律。Hb 同词的双色斜率可异向，横线宽度反映真实片段差异。','small')
    body='<div class="grid atlas-layout"><div>'+embedded((P/'pattern_atlas.png').read_bytes(),187,'全训练与验证局部观测词图谱')+'</div><div>'+side+'</div></div>'
    text.append(page(3,'03　/　观测词图谱','局部形态可以读出；同一词不等于固定生理状态','图按训练条件中位数排序。EEG 显示各词真实 2 秒片段的相对谱，不把相位未对齐的电压平均；Hb 点和线分别是经验中位数与中间 50%（不是置信区间）。',body,'20261120_development_v3：atlas.csv、pattern_atlas.png、attributes.csv、summary.json；标准化与探针定义：src/analysis/observation_tokenizer_evaluation.py。'))
    names=[('观测模式 ID','pattern_ids','112'),('同 K 盲波形 ID','waveform_ids_same_k','112'),('正确 ID＋训练标签置乱','pattern_ids_shuffled_train_labels','112'),('直接连续观测属性','continuous_attributes','不计码率'),('flat 解码→属性 ID','flat_to_pattern_ids','8,576'),('flat 解码→连续属性','flat_to_attributes','8,576'),('训练多数类','training_majority','0')]
    score=lambda k:task['scores'][k]['subject_balanced_accuracy']
    vals=[]
    for label,key,bits in names:
        vals.append((label,*[f'{score(key if key=="training_majority" else key+"/"+mod):.3f}' for mod in ['eeg','hb','fusion']],bits))
    ct={i['reference']:i for i in pc['contrasts'] if i['branch']=='fusion'}
    body='<div class="grid col2"><div><h2 class="first">冻结读出：各条件的被试等权 BA ↑</h2>'+tab(['输入','EEG','Hb','融合','融合 bit / 事件'],vals,'compact')+note('BA 是同一被试的心算与显式基线两类召回率均值，再对验证被试等权；不是按 264 次试验池化。编码器、字典与尺度均冻结。','small')+'</div><div><h2 class="first">融合模式 ID 减参照：百分点差 [95% 区间]</h2>'
    body+=tab(['参照','配对差（百分点）','高 / 平 / 低'],[(name,ci(ct[key]['mean_delta_ba'],ct[key]['ci95'],1,100),'/'.join(str(ct[key]['direction_counts'][k]) for k in ('higher','tie','lower'))) for name,key in [('同位率盲波形 ID','waveform_ids_same_k/fusion'),('置乱标签','pattern_ids_shuffled_train_labels/fusion'),('直接连续属性','continuous_attributes/fusion'),('flat 解码→属性 ID','flat_to_pattern_ids/fusion'),('flat 解码→连续属性','flat_to_attributes/fusion'),('训练多数类','training_majority')]],'compact')+note('六人配对自举，枚举 6⁶ = 46,656 个等可能重抽样取分位区间；不重拟合、不计试次作独立样本、未校正多重比较。一次训练标签置乱只是负对照，不是置换检验。','small')+'</div></div>'
    body+='<h2>单库、单任务：队列和信息边界</h2>'+note('Single-Trial 心算（MA）对正式基线试次（BL），使用两模态各自标记后 [2,10) 秒；不是静息段或无提示区。4 个 EEG、3 个 Hb 位置各取 4 个 2 秒槽；先在前后两个 4 秒段分别平均属性或 one-hot，再按位置拼接。只在训练集按被试等权拟合标准化与固定 C=1 逻辑回归；不以任务得分选码率。')
    cov=pc['coverage']; tr,va=cov['train'],cov['validation']; body+=note(f'17 训练被试：{tr["retained"]}/{tr["planned"]:,} 有效（288 无双时钟共同 G0 窗，5 条支持不足）；6 验证被试：{va["retained"]}/{va["planned"]}（95 无共同窗、1 条支持不足）。完整验证样本上，模式词融合相对多数类仅 +2.4 个百分点，区间跨零；融合点估计还低于 EEG-only。不能以此主张分类、配对耦合或临床获益。','call')
    pairs=ct['waveform_ids_same_k/fusion']['per_subject']; ids=[v['subject'].split('_')[-1] for v in pairs]
    body+='<h2>最严格的同率比较：六位验证被试分别怎样？</h2>'+tab(['融合，112 bit/事件',*[f'被试 {x}' for x in ids]],[('纳入 MA / BL',*[f'{z["true_ma"]} / {z["true_bl"]}' for z in pairs]),('模式词 − 盲波形词（百分点）',*[signed(100*z['delta_ba'],1) for z in pairs])],'compact')+note('输入为原已处理的离线信号。Hb 保留既有被试自己的无标签尺度，包含验证被试；因此不能称为无需个体校准的零样本部署或实时解码。','small')
    text.append(page(4,'04　/　真实任务','六位验证被试上，4 位模式词未显示可确认的心算优势','模式词并非任务标签。用同一训练和验证事件、公平的读出器，比较同预算离散词、无预算上限的属性，以及较高预算的 flat 解码；不使用任何新增测试集。',body,'20261120_development_v3/task_probe.json（冻结全部条件的分数及支持）、paired_task_contrasts_v1.json（6 位验证被试、原始混淆矩阵派生的成对差）。'))
    img,local,start,marks=example_figure(); body=embedded(img,273,'REFED 同一时间窗的 EEG、HbO 和 HbR 波形')
    body+=note(f'REFED / 被试 31 / video_4 / 记录起点 0 秒，父窗 e0000000000_n0079000。各库预先导出的第一个位置输入之一，本页固定 REFED，不按误差挑选。EEG 电极 CB1 展开距离窗口中央最近的完整 {start}–{start+2} 秒，Hb 位点 CH49 展示 30 秒；原 X 与所有解码在本模态共用训练尺度。旧模型均为 14 bit；绿色为中间代自适应分配（EEG 1024 / Hb 32 bit），不是最终 EEG 512 bit 增益模型，也不是同位率优势证据。','small')
    fmt=lambda values:' / '.join(f'{x:+.2f}' for x in values).replace('-', '−')
    body+=note(f'按观测／旧模型／自适应排列：EEG 放大段中距片均值最大的点在 {marks["time"]:.3f} 秒，电压为 {fmt(marks["peak"])} 个训练 X 尺度。Hb 全窗末、首各 0.5 秒均值之差为 HbO {fmt(marks["changes"][0])}、HbR {fmt(marks["changes"][1])}；方向相合不等于精细恢复。','small')
    body+='<div class="grid half"><div><h2>示例是读图辅助，不是新评分人群</h2>'+note(f'用各自所有有效片段和冻结训练方差复核示例，旧→自适应 X NMSE：EEG {local["eeg","old"]:.4f} → {local["eeg","new"]:.4f}（14/15 片），Hb {local["hb","old"]:.4f} → {local["hb","new"]:.4f}（15/15 片，双色团等权）。前图 EEG 放大仅为看清相位；快慢信号同窗不说明耦合。')+'</div><div><h2>给后续生理模块保留什么？</h2>'+note('传递整数载荷、字典版本、通道／色团、时间支持、缺失掩码及冻结尺度索引；不得把逐片连续增益当作免费旁路。接口需能回到相同单位的观测 X，保留频带绝对功率和 Hb 慢趋势。')+'</div></div>'
    body+=note('下一步：在相同可观察支持上，把旧模型、同率 flat、新词＋细节分别接入同一个冻结 Tak 观测算子；比较条件代理读出与独立遮挡端点，加入自身历史、时间移位及错配对照。若只过压缩护栏，保留编码工具；若词有独立增益而不助压缩，允许两条输出分开使用。当前没有恢复生理来源或取得因果识别的证据。','call')
    text.append(page(5,'05　/　从波形到可用接口','波形需要对照实测相位；生理模块仍须独立验证','同一记录、同一时间支持上的解码对照。黑实线为已导出的观测 X，橙虚线为旧双词表，绿点划线为自适应条件细节；每个子图独立纵轴，右侧两个子图保留各色团符号。',body,'20261120_development_v3/{eeg,hb}_first_per_dataset.{csv,npz}；20261120_allocation_v1/{eeg,hb}_refed_example.npz、{eeg,hb}_decision.json。以上为原已导出样例，无新拟合。'))
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+TITLE+'</title><style>'+CSS+'</style><body>'+''.join(text)+'</body></html>'


def render(output, gain):
    content=compose(gain); output=Path(output)
    if output.is_symlink() or output.parent.resolve()!=SUITE.resolve() or not re.fullmatch(r'20261120_baseline_review_v[1-9][0-9]*(?:_preview[1-9][0-9]*)?',output.name):
        raise ValueError('fresh, separate versioned review output required')
    output.mkdir(exist_ok=False)
    path=output/'observation_baseline_review.html'; path.write_text(content,encoding='utf-8'); pdf=path.with_suffix('.pdf')
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True); page=browser.new_page(); page.goto(path.resolve().as_uri(),wait_until='load'); page.emulate_media(media='print'); page.evaluate('document.fonts.ready')
        layout=page.locator('section').evaluate_all('''sections => sections.map(s=>{const foot=s.querySelector('footer').getBoundingClientRect(), main=s.querySelector('main').getBoundingClientRect(); return {gap:foot.top-main.bottom, overflow:s.scrollWidth-s.clientWidth, brokenImages:[...s.querySelectorAll('img')].filter(i=>!i.complete || !i.naturalWidth).length};})''')
        if any(v['gap']<=0 or v['overflow']>2 or v['brokenImages'] for v in layout):
            raise ValueError(f'HTML print layout needs attention: {layout}')
        page.pdf(path=str(pdf),prefer_css_page_size=True,print_background=True,display_header_footer=False,tagged=True);browser.close()
    checks=[]
    with fitz.open(pdf) as d:
        if len(d)!=5: raise ValueError(f'expected 5 pages, got {len(d)}')
        for i,page in enumerate(d):
            info=page.get_image_info(xrefs=True); txt=page.get_text(); spans=[s for b in page.get_text('dict')['blocks'] if b['type']==0 for l in b['lines'] for s in l['spans']]
            tol=page.rect+(-2,-2,2,2); images=[]
            for item in info:
                r=fitz.Rect(item['bbox']); px=float(item['width']*72/r.width); images.append(dict(xref=item['xref'],width=item['width'],height=item['height'],effective_dpi=round(px,1)))
                if not tol.contains(r) or px<195 or px>268 or r.get_area()/page.rect.get_area()>.78 or item['xref']<=0: raise ValueError(('invalid image extent or density',i+1,item['width'],r,px))
            if '证据' not in txt or len(re.findall('[\u3400-\u9fff]',txt))<120 or any(not tol.contains(fitz.Rect(s['bbox'])) for s in spans) or (bool(info)!=(i in [1,2,4])):
                raise ValueError(('searchable Chinese text or content bounds failed',i+1))
            checks.append(dict(page=i+1,chinese_characters=len(re.findall('[\u3400-\u9fff]',txt)),images=images,body_footer_gap_css_px=round(layout[i]['gap'],2)))
    (output/'verification.json').write_text(json.dumps(dict(status='passed',pages=checks,reading='all scientific figures are PNG image objects; searchable HTML/PDF body; same-page text and image bounds checked',standalone_html_embedded_images=True,source_runs=[P.name,A.name,G.name]),indent=2,ensure_ascii=False)+'\n')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEST)
    render(p.parse_args().output,gain_facts())
