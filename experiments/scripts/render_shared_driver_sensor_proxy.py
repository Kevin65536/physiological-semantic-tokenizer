#!/usr/bin/env python3
"""Read-only Chinese report and complete waveform atlas for sensor-proxy v1.

Only an explicit completed run and its retained JSON/NPZ are consumed. No raw
reader, optimizer, model selection or new confidence interval is invoked.
Figures and LaTeX equations are PNG; PDF text and tables remain selectable.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
from collections import defaultdict
from html import escape
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
import subprocess

import fitz
from fontTools.ttLib import TTFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image

FONT = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
DPI = 240
WIDTH = 523.
LABELS = dict(pc1='原 PC1', pc1_band_gain='PC1 频带修正', mean6='均值6',
    sixth_to_seventh='第6换第7', mean3='均值3', mean12='均值12', ring025='环带0.25',
    nearest6_far6='内6＋远6', nearest6_random6_0='随机扩展1',
    nearest6_random6_1='随机扩展2', nearest6_random6_2='随机扩展3',
    Hb_only='Hb-only 参照', zero='无 EEG')
COLORS = dict(pc1='#2874ad', mean6='#d16a22', mean12='#24846c', Hb_only='#8453a1',
              zero='#84909b', observed='#172d40', wrong='#bb4967', fir='#008b91')
STYLES = dict(pc1='--', mean6='-', mean12='-.', Hb_only=':', zero=':', observed='-')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def num(value, digits=4):
    return '未定义' if value is None or not np.isfinite(value) else f'{value:.{digits}f}'


def aggregate(rows, field='nmse', penalty=None):
    """Windows -> native record -> subject, retaining optional failures."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = row.get(field)
        if not row.get('success', True):
            value = penalty
        if value is None:
            continue
        groups[row['subject']][row['native_record']].append(float(value))
    by_subject = {s: float(np.mean([np.mean(x) for x in records.values()]))
                  for s, records in groups.items()}
    return (float(np.mean(list(by_subject.values()))) if by_subject else None), by_subject


def select_examples(rows, tasks, common):
    """Low/median/high baseline-prediction errors, preferring distinct subjects."""
    ranked = sorted((r for r in rows if r['id'] in common),
        key=lambda r: (tasks[(r['id'], 'pc1', 'prediction', 'paired')]['nmse'], r['id']))
    if len(ranked) < 3:
        raise ValueError('three common-success examples required')
    selected, subjects = [], set()
    for label, target in (('较低误差', 0), ('居中误差', (len(ranked)-1)//2), ('较高误差', len(ranked)-1)):
        choices = sorted(range(len(ranked)), key=lambda j: (abs(j-target), j))
        index = next((j for j in choices if ranked[j]['subject'] not in subjects), choices[0])
        row = ranked[index]
        subjects.add(row['subject'])
        selected.append(dict(label=label, row=row, rank=index+1, denominator=len(ranked)))
    return selected


class Evidence:
    def __init__(self, run):
        self.run = Path(run).resolve()
        self.cfg = read(self.run/'resolved_config.json')
        self.summary = read(self.run/'summary.json')
        self.plan = read(self.run/'plan.json')
        self.screen = read(self.run/'screen.json')
        self.audit = read(self.run/'array_audit.json')
        self.residual = read(self.run/'residual_analysis_v1/residual_summary.json')
        if (self.cfg['schema'] != 'shared_driver_sensor_proxy_v1'
                or self.summary['status'] != 'completed' or self.audit['status'] != 'passed'
                or self.summary['planned'] != 420 or self.summary['terminal'] != 420):
            raise ValueError('explicit completed and audited sensor-proxy v1 evidence required')
        self.sd = np.asarray(self.screen['coordinate']['sd'])[1:]
        self.rows = {r['id']: r for r in self.plan['rows']}
        self.tasks, self.arrays, self.all_tasks = {}, {}, []
        for spec in read(self.run/'nonlinear_tasks.json'):
            task = read(self.run/'tasks'/f"{spec['id']}.json")
            if any(task[k] != spec[k] for k in ('id','record_id','candidate','condition','pairing')):
                raise ValueError('task identity mismatch')
            key = tuple(task[k] for k in ('record_id','candidate','condition','pairing'))
            if key in self.tasks:
                raise ValueError('duplicate task')
            with np.load(self.run/'tasks'/f"{spec['id']}.npz", allow_pickle=False) as z:
                a = {k: z[k].copy() for k in z.files}
            if (a['prediction'].shape != (300,3) or a['target_hb'].shape != (300,2)
                    or not np.allclose(a['time_s'], np.arange(300)/10)):
                raise ValueError('wrong retained waveform shape/time')
            if not np.allclose(a['residual'], a['target_hb']-a['prediction'][:,1:], atol=1e-12):
                raise ValueError('residual sign or values changed')
            score = float(np.mean(((a['target_hb'][50:299]-a['prediction'][50:299,1:])/self.sd)**2))
            if not np.isclose(score, task['nmse'], rtol=1e-11, atol=1e-12):
                raise ValueError('retained waveform score mismatch')
            self.tasks[key], self.arrays[key] = task, a
            self.all_tasks.append(task)
        for name, g in self.summary['groups'].items():
            split, candidate, condition, pairing = name.split('|')
            rows = self.group(split, candidate, condition, pairing)
            value, _ = aggregate(rows, penalty=100.)
            if len(rows) != g['planned'] or not np.isclose(value, g['subject_equal_nmse_with_failure_penalty']):
                raise ValueError('summary denominator or aggregation mismatch')

    def group(self, split, candidate, condition='prediction', pairing='paired'):
        return [r for r in self.all_tasks if (r['split'],r['candidate'],r['condition'],r['pairing']) ==
                (split,candidate,condition,pairing)]

    def metric(self, split, candidate, condition='prediction', pairing='paired'):
        return self.summary['groups']['|'.join((split,candidate,condition,pairing))]


class Document:
    """Local fixed-page compositor with fail-on-overflow and HTML companion."""
    def __init__(self, out, name):
        self.out, self.name = Path(out), name
        buffer = BytesIO()
        font = TTFont(str(FONT), fontNumber=2)
        for cmap in font['cmap'].tables:
            if cmap.isUnicode():
                cmap.cmap = {c:g for c,g in cmap.cmap.items()
                             if not (0xF900<=c<=0xFAFF or 0x2F800<=c<=0x2FA1F)}
        font.save(buffer)
        self.font = fitz.Font(fontbuffer=buffer.getvalue())
        self.pdf = fitz.open()
        self.pages, self.images, self.boxes = [], [], []

    def page(self, title, source):
        self.p = self.pdf.new_page(width=595.28, height=841.89)
        self.y = 34.
        self.pages.append(dict(title=title, source=source, html=[]))
        self.text(title, size=17, gap=12, color=(.08,.18,.3))

    def wrap(self, value, width, size):
        lines = []
        for paragraph in str(value).split('\n'):
            current = ''
            for token in re.findall(r'[A-Za-z0-9_./%+\-−]+|.', paragraph):
                # Keep numerical values and ordinary Latin words intact. Long
                # provenance paths still wrap rather than overflow the page.
                pieces = list(token) if self.font.text_length(token,fontsize=size)>width else [token]
                for piece in pieces:
                    if current and self.font.text_length(current+piece, fontsize=size)>width:
                        lines.append(current); current=piece
                    else:
                        current += piece
            lines.append(current)
        return lines

    def text(self, value, size=10.1, gap=9, color=(.13,.18,.23)):
        lines = self.wrap(value, WIDTH, size)
        height = len(lines)*size*1.48
        if self.y+height>773:
            raise ValueError(f"text overflow page {len(self.pdf)} {self.pages[-1]['title']}: {str(value)[:60]}")
        writer = fitz.TextWriter(self.p.rect)
        for line in lines:
            if line: writer.append((36,self.y+size), line, font=self.font, fontsize=size)
            self.y += size*1.48
        writer.write_text(self.p, color=color)
        if value: self.boxes.append(dict(page=len(self.pdf),rect=list(writer.text_rect)))
        self.y += gap
        self.pages[-1]['html'].append('<p>'+escape(str(value)).replace('\n','<br>')+'</p>')

    def table(self, headers, rows, fractions, size=8.5):
        widths = np.asarray(fractions, float)/sum(fractions)*WIDTH
        html = ['<table>']
        for i, row in enumerate([headers,*rows]):
            cells = [self.wrap(v,w-10,size) for v,w in zip(row,widths)]
            height = max(len(x) for x in cells)*size*1.4+10
            if self.y+height>773: raise ValueError(f'table overflow page {len(self.pdf)}')
            self.p.draw_rect(fitz.Rect(36,self.y,559,self.y+height),color=None,
                fill=(.9,.94,.97) if i==0 else ((.97,.98,.99) if i%2 else (1,1,1)))
            x = 36.; writer=fitz.TextWriter(self.p.rect)
            for lines,w in zip(cells,widths):
                for j,line in enumerate(lines):
                    writer.append((x+5,self.y+5+size+j*size*1.4),line,font=self.font,fontsize=size)
                x += w
            writer.write_text(self.p,color=(.1,.16,.22))
            self.boxes.append(dict(page=len(self.pdf),rect=list(writer.text_rect)))
            self.y += height
            tag='th' if i==0 else 'td'
            html.append('<tr>'+''.join(f'<{tag}>{escape(str(v))}</{tag}>' for v in row)+'</tr>')
        self.y += 12
        self.pages[-1]['html'].append(''.join(html)+'</table>')

    def figure(self, path, caption='', width=WIDTH):
        path=Path(path)
        with Image.open(path) as im: px,py=im.size
        height=width*py/px
        if self.y+height>773: raise ValueError(f'figure overflow page {len(self.pdf)}: {path.name}')
        x=36+(WIDTH-width)/2
        self.p.insert_image(fitz.Rect(x,self.y,x+width,self.y+height),filename=str(path))
        self.images.append(dict(page=len(self.pdf),file=str(path.relative_to(self.out)),dpi=px/(width/72)))
        self.y += height+7
        encoded=base64.b64encode(path.read_bytes()).decode()
        self.pages[-1]['html'].append(f'<img src="data:image/png;base64,{encoded}" alt="{escape(caption)}">')
        if caption: self.text(caption,size=8.6,gap=9,color=(.28,.34,.4))

    def save(self):
        toc=[]
        for i,p in enumerate(self.pdf):
            writer=fitz.TextWriter(p.rect)
            source='证据：'+self.pages[i]['source']
            for j,line in enumerate(self.wrap(source,WIDTH,6.5)[:2]):
                writer.append((36,792+j*9),line,font=self.font,fontsize=6.5)
            writer.append((36,824),'固定 Tak 传感器代理实验 | 公开开发重分析 | 2026-10-28',font=self.font,fontsize=7)
            writer.append((516,824),f'{i+1}/{len(self.pdf)}',font=self.font,fontsize=7)
            writer.write_text(p,color=(.35,.4,.45))
            toc.append([1,self.pages[i]['title'],i+1])
        self.pdf.set_toc(toc)
        self.pdf.set_metadata(dict(title='固定 Tak 传感器代理：配置、结果与波形',author='PID-MCM project'))
        self.pdf.subset_fonts(fallback=True)
        target=self.out/(self.name+'.pdf')
        self.pdf.save(target,garbage=4,deflate=True); self.pdf.close()
        css='''body{font-family:"Noto Sans CJK SC",sans-serif;background:#e9eef3;color:#193047;line-height:1.6}section{background:white;width:880px;padding:40px;margin:25px auto;page-break-after:always}h1{font-size:25px}p{font-size:15px}img{display:block;width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #d8e2ec;text-align:left}th{background:#e6eef5}.source{font-size:11px;color:#657889}nav{width:880px;margin:auto}a{color:#226798}@media print{body{background:white}section{width:auto;margin:0;padding:0}}'''
        navigation='<nav><h1>目录</h1>'+''.join(f'<div><a href="#p{i+1}">{i+1}. {escape(p["title"])}</a></div>' for i,p in enumerate(self.pages))+'</nav>'
        body=''.join(f'<section id="p{i+1}"><h1>{escape(p["title"])}</h1>'+''.join(p['html'][1:])+'<p class="source">证据：'+escape(p['source'])+'</p></section>' for i,p in enumerate(self.pages))
        (self.out/(self.name+'.html')).write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>'+self.name+'</title><style>'+css+'</style>'+navigation+body+'</html>')
        return dict(pdf=target.name,pages=len(self.pages),figures=self.images,text_boxes=self.boxes,
                    contents=[dict(page=i+1,title=p['title'],source=p['source']) for i,p in enumerate(self.pages)])


def save_figure(fig, out, name):
    path=Path(out)/'figures'/(name+'.png')
    fig.savefig(path,dpi=DPI,facecolor='white')
    plt.close(fig)
    return path


def equation(out, name, lines):
    folder=Path(out)/'formulas';folder.mkdir(exist_ok=True)
    text=r'\documentclass[border=5pt]{standalone}\usepackage{amsmath,amssymb}\begin{document}$\begin{aligned}'+(r'\\[5pt]'+'\n').join(lines)+r'\end{aligned}$\end{document}'
    source=folder/(name+'.tex');source.write_text(text)
    result=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',source.name],cwd=folder,capture_output=True,text=True)
    if result.returncode: raise RuntimeError(result.stdout[-3000:])
    pdf=fitz.open(source.with_suffix('.pdf'))
    # Rasterize to exactly the final 523 pt display width at 240 dpi.
    pdf[0].get_pixmap(matrix=fitz.Matrix((WIDTH/72*DPI)/pdf[0].rect.width,(WIDTH/72*DPI)/pdf[0].rect.width),alpha=False).save(folder/(name+'.png'))
    pdf.close()
    return folder/(name+'.png')


def geometry_figure(e, out):
    row=e.plan['rows'][0];g=row['geometry_identity']['eeg'];sets=row['channel_sets']['sets']
    fig,ax=plt.subplots(figsize=(WIDTH/72,4.2))
    xy=np.asarray([[r['x'],r['y']] for r in g]);ax.scatter(*xy.T,c='#c5cdd5',s=28,label='其余通道')
    for key,color,label in [('nearest6',COLORS['pc1'],'内圈6'),('outer6',COLORS['mean6'],'近邻外圈6'),('far6',COLORS['Hb_only'],'远端6')]:
        ids=sets[key];ax.scatter(xy[ids,0],xy[ids,1],color=color,s=48,label=label)
    for pos,r in zip(xy,g):ax.annotate(r['channel_name'],pos,xytext=(3,3),textcoords='offset points',fontsize=7)
    h=np.asarray(row['patch']['hb_coordinates'])[0];ax.scatter(h[0],h[1],marker='*',color='#c43d46',s=160,label='目标 FpzFp1')
    ax.set(xlabel='模板 x（无量纲）',ylabel='模板 y（无量纲，向前为正）',title='模板传感器位置的顶视投影；选择依据为三维距离')
    ax.set_aspect('equal');ax.legend(loc='lower left',fontsize=8);fig.tight_layout()
    return save_figure(fig,out,'geometry')


def cv_figure(e,out):
    values=e.screen['cv_subject_mean_nmse'];keys=list(values)
    fig,axs=plt.subplots(1,2,figsize=(WIDTH/72,4.1),gridspec_kw={'width_ratios':[2.1,1]})
    y=np.arange(len(keys));colors=[COLORS.get(k,'#8b9bad') for k in keys]
    axs[0].barh(y,[values[k] for k in keys],color=colors)
    axs[0].set_yticks(y,[LABELS[k] for k in keys]);axs[0].invert_yaxis();axs[0].set(xlim=(0,1.6),xlabel='训练分组CV NMSE（零起点）')
    axs[1].plot([values[k] for k in keys[1:]],y[1:],'o',color=COLORS['mean6'])
    axs[1].set_ylim(axs[0].get_ylim());axs[1].set_yticks([]);axs[1].set(xlim=(1.253,1.264),xlabel='非PC1候选放大视图')
    axs[1].axvline(values['mean6'],ls='--',color=COLORS['pc1'],lw=1)
    fig.tight_layout();return save_figure(fig,out,'cv_comparison')


def subject_figure(e,out):
    fig,axs=plt.subplots(1,2,figsize=(WIDTH/72,3.7))
    keys=['pc1','mean6','zero'];subsets={k:aggregate(e.group('evaluation',k))[1] for k in keys};subjects=sorted(subsets['pc1'])
    for s in subjects:axs[0].plot(np.arange(3),[subsets[k][s] for k in keys],marker='o',label=s.replace('subject_','S'))
    axs[0].set_xticks(np.arange(3),[LABELS[k] for k in keys]);axs[0].set_ylabel('被试NMSE');axs[0].legend(ncol=2,fontsize=7)
    deltas=np.array([subsets['mean6'][s]-subsets['pc1'][s] for s in subjects]);axs[1].bar(np.arange(6),deltas,color=[COLORS['mean6'] if d<0 else '#b74956' for d in deltas])
    axs[1].axhline(0,color='#333',lw=.7);axs[1].set_xticks(np.arange(6),[s[-2:] for s in subjects]);axs[1].set(xlabel='被试编号',ylabel='均值6 − PC1 的NMSE')
    fig.tight_layout();return save_figure(fig,out,'subject_prediction')


def pairing_figure(e,out):
    fig,axs=plt.subplots(1,2,figsize=(WIDTH/72,3.3))
    for ax,c in zip(axs,['pc1','mean6']):
        v=e.residual['prediction_comparisons'][c+'_true_vs_wrong'];a=v['left_subject_nmse'];b=v['right_subject_nmse'];subjects=sorted(a)
        differences=[a[s]-b[s] for s in subjects]
        ax.bar(np.arange(6),differences,color=COLORS[c]);ax.axhline(0,color='#444',lw=.7)
        ax.set_xticks(np.arange(6),[s[-2:] for s in subjects]);ax.set(title=LABELS[c],xlabel='被试编号',ylabel='真实 − 错配 NMSE')
    limits=[min(ax.get_ylim()[0] for ax in axs),max(ax.get_ylim()[1] for ax in axs)]
    for ax in axs:ax.set_ylim(limits)
    fig.tight_layout();return save_figure(fig,out,'pairing')


def waveform(e,out,row,name,limits=None):
    fig,axs=plt.subplots(3,2,figsize=(WIDTH/72,6.25),sharex=True)
    candidates=['pc1','mean6']+(['mean12'] if row['split']=='selection' else [])
    base=(row['id'],'pc1','prediction','paired');obs=e.arrays[base]['target_hb']/e.sd;t=e.arrays[base]['time_s']
    handles={}
    for c in range(2):
        for ri in (0,1):
            line,=axs[ri,c].plot(t,obs[:,c],color=COLORS['observed'],lw=1.6,label='处理后实测');handles['处理后实测']=line
        axs[0,c].axhline(0,color=COLORS['zero'],lw=.7,ls=':')
        for method in candidates+['Hb_only']:
            for ri,condition in ((0,'prediction'),(1,'joint'),(2,'joint')):
                if method=='Hb_only' and ri==0:continue
                key=(row['id'],method,condition,'paired');a=e.arrays[key];task=e.tasks[key]
                val=(a['residual'] if ri==2 else a['prediction'][:,1:])/e.sd
                line,=axs[ri,c].plot(t,val[:,c],color=COLORS[method],lw=1.1,ls=STYLES[method],
                    marker=None if task['success'] else 'x',markevery=30,markersize=4,
                    label=LABELS[method]);handles[LABELS[method]]=line
        for ri in range(3):
            ax=axs[ri,c];ax.axvspan(0,5,color='#edf1f5');ax.axvline(5,color='#8a949e',lw=.7,ls='--');ax.axvspan(29.9,30,color='#e8e8e8');ax.set_xlim(0,30);ax.grid(alpha=.14)
            ax.set_ylabel(['隐藏预测','联合/物理参照','联合残差'][ri]+'\n训练SD')
            if limits is not None:ax.set_ylim(limits[ri][c])
        axs[0,c].set_title(['HbO','HbR'][c]);axs[2,c].set_xlabel('窗口时间（秒；5秒为事件锚点）')
    fig.legend(handles.values(),handles.keys(),loc='upper center',ncol=len(handles),fontsize=7.8,bbox_to_anchor=(.5,1.0))
    fig.subplots_adjust(top=.92,bottom=.09,left=.105,right=.985,wspace=.29,hspace=.28)
    return save_figure(fig,out,name)


def representative_limits(e,rows):
    values=[[[] for _ in range(2)] for _ in range(3)]
    for row in rows:
        for method in ('pc1','mean6','Hb_only'):
            for i,condition in ((0,'prediction'),(1,'joint'),(2,'joint')):
                if method=='Hb_only' and i==0:continue
                a=e.arrays[(row['id'],method,condition,'paired')]
                y=(a['residual'] if i==2 else a['prediction'][:,1:])/e.sd
                for c in range(2):
                    values[i][c].extend(y[:,c]);
                    if i<2:values[i][c].extend(a['target_hb'][:,c]/e.sd[c])
    return [[[float(np.min(v)-.08*np.ptp(v)-.05),float(np.max(v)+.08*np.ptp(v)+.05)] for v in line] for line in values]


def window_caption(e,row):
    parts=[]
    for method in ('pc1','mean6')+ (('mean12',) if row['split']=='selection' else ()):
        p=e.tasks[(row['id'],method,'prediction','paired')];j=e.tasks[(row['id'],method,'joint','paired')]
        parts.append(f"{LABELS[method]}：隐藏NMSE {p['nmse']:.3f}；联合 {j['nmse']:.3f}"+('（未收敛，仅诊断）' if not j['success'] else ''))
    h=e.tasks[(row['id'],'Hb_only','joint','paired')]
    parts.append(f"Hb-only NMSE {h['nmse']:.3f}"+('（未收敛，仅诊断）' if not h['success'] else ''))
    return '。'.join(parts)+'。'


def waveform_reading(e,row):
    p=e.arrays[(row['id'],'pc1','prediction','paired')];j=e.arrays[(row['id'],'mean6','prediction','paired')];texts=[]
    for c,hb in enumerate(('HbO','HbR')):
        y=p['target_hb'][50:299,c]/e.sd[c];t=p['time_s'][50:299]
        a=e.tasks[(row['id'],'pc1','prediction','paired')]['waveform_metrics'];b=e.tasks[(row['id'],'mean6','prediction','paired')]['waveform_metrics']
        texts.append(f"{hb}目标在{t[np.argmax(y)]:.1f}秒达到最高{y.max():.2f} SD，在{t[np.argmin(y)]:.1f}秒达到最低{y.min():.2f} SD；PC1/均值6的预测RMS比为{a['prediction_to_target_RMS_ratio'][c]:.2f}/{b['prediction_to_target_RMS_ratio'][c]:.2f}，相关为{num(a['target_prediction_correlation'][c],2)}/{num(b['target_prediction_correlation'][c],2)}。")
    return ''.join(texts)


def proxy_figure(e,out,examples):
    fig,axs=plt.subplots(3,2,figsize=(WIDTH/72,4.8),sharex=True)
    ratios=[]
    for i,example in enumerate(examples):
        row=example['row'];identity=hashlib.sha256(row['id'].encode()).hexdigest()[:24]
        with np.load(e.run/'prepared'/f'{identity}.npz',allow_pickle=False) as a:
            x=a['eeg_logpower'].copy();t=a['time_s'].copy()
        inner=row['channel_sets']['sets']['nearest6'];coord=e.screen['coordinate']
        pc=np.asarray(coord['pc']).reshape(6,5);factor=coord['eeg_factor']
        proxies={'pc1':np.sum(x[:,inner]*pc,axis=(1,2))*factor,
                 'mean6':np.mean(x[:,inner],axis=1)@np.asarray(e.screen['models']['mean6']['coefficients'])*factor}
        rms={}
        for method,z in proxies.items():
            key=(row['id'],method,'prediction','paired');a=e.arrays[key]
            data=.4*np.sum(((a['prediction'][:299,0]-z[:299])/.025)**2)
            if not np.isclose(data,e.tasks[key]['objective_components']['data'],rtol=1e-8,atol=1e-8):
                raise ValueError('proxy visualization does not reproduce retained EEG data cost')
            axs[i,0].plot(t,z/.025,color=COLORS[method],ls=STYLES[method],lw=1.1,label=LABELS[method])
            axs[i,1].plot(t,np.asarray(a['driver']).reshape(-1)/.025,color=COLORS[method],ls=STYLES[method],lw=1.1)
            rms[method]=float(np.sqrt(np.mean(z[50:299]**2))/.025)
        ratios.append(dict(id=row['id'],proxy_RMS_training_EEG_SD=rms))
        for ax in axs[i]:
            ax.axvspan(0,5,color='#edf1f5');ax.axvline(5,color='#999',lw=.7,ls='--');ax.axhline(0,color='#aaa',lw=.6);ax.grid(alpha=.12);ax.set_xlim(0,30)
        axs[i,0].set_ylabel('S'+row['subject'][-2:]+'\nEEG训练SD')
        axs[i,1].set_ylabel('模型驱动 / 0.025')
    axs[0,0].set_title('处理后标量代理 z');axs[0,1].set_title('仅EEG条件估计的驱动 r')
    axs[0,0].legend(fontsize=8,ncol=2)
    for ax in axs[2]:ax.set_xlabel('窗口时间（秒）')
    # Common scale within each type, while z and r remain distinct coordinates.
    for col in (0,1):
        lo=min(axs[i,col].get_ylim()[0] for i in range(3));hi=max(axs[i,col].get_ylim()[1] for i in range(3))
        for i in range(3):axs[i,col].set_ylim(lo,hi)
    fig.tight_layout();return save_figure(fig,out,'proxy_and_driver'),ratios


def fir_waveform_figure(e,out,examples):
    fig,axs=plt.subplots(2,3,figsize=(WIDTH/72,3.35),sharex=True)
    handles={}
    for j,example in enumerate(examples):
        row=example['row'];a=e.arrays[(row['id'],'pc1','prediction','paired')];t=a['time_s']
        curves={'处理后实测':a['target_hb'],'PC1物理':a['prediction'][:,1:]}
        for method in ('pc1','mean6'):
            name=hashlib.sha256((row['id']+method).encode()).hexdigest()[:24]
            with np.load(e.run/'fir'/f'{name}.npz',allow_pickle=False) as z:curves[LABELS[method]+' FIR']=z['prediction'].copy()
        for c in (0,1):
            ax=axs[c,j]
            for (name,y),color,style in zip(curves.items(),
                    [COLORS['observed'],COLORS['pc1'],COLORS['fir'],COLORS['mean6']],['-','--','-.',':']):
                line,=ax.plot(t,y[:,c]/e.sd[c],color=color,ls=style,lw=1.1,label=name);handles[name]=line
            ax.axvspan(0,5,color='#edf1f5');ax.axhline(0,color='#aaa',lw=.6);ax.set_xlim(0,30);ax.grid(alpha=.12)
            if j==0:ax.set_ylabel(['HbO / 训练SD','HbR / 训练SD'][c])
            if c==1:ax.set_xlabel('时间（秒）')
        axs[0,j].set_title('S'+row['subject'][-2:]+' / '+example['label'],fontsize=8)
    fig.legend(handles.values(),handles.keys(),ncol=4,loc='upper center',fontsize=7.7)
    fig.subplots_adjust(top=.82,bottom=.16,left=.08,right=.99,hspace=.28,wspace=.30)
    return save_figure(fig,out,'fir_waveforms')


def summary_figures(e,out):
    paths={}
    fig,ax=plt.subplots(figsize=(WIDTH/72,2.8));ax.axis('off')
    boxes=[(.01,.62,.27,'原生 EEG：90秒支持\n五频带 → 100ms功率 → log'),(.36,.62,.28,'共同30秒目标与5秒参考\n训练 PCA / 固定尺度'),(.72,.62,.27,'单一标量代理\nPC1、频带修正或空间均值'),(.02,.1,.27,'固定 Tak 模型\n驱动与四个初态条件估计'),(.36,.1,.28,'固定 HbO/HbR 观测关系\n相同参考算子'),(.72,.1,.27,'隐藏预测 / 联合重建\n同一训练SD评分')]
    for x,y,w,text in boxes:ax.text(x+w/2,y+.14,text,ha='center',va='center',fontsize=8.6,bbox=dict(boxstyle='round,pad=.7',facecolor='#edf3f8',edgecolor='#7c98ae'))
    for a,b in [((.29,.76),(.35,.76)),((.65,.76),(.71,.76)),((.85,.53),(.16,.43)),((.30,.24),(.35,.24)),((.65,.24),(.71,.24))]:ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#4c6274'))
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0);paths['pipeline']=save_figure(fig,out,'pipeline')
    pc=np.array(e.screen['coordinate']['pc']).reshape(6,5)
    fig,ax=plt.subplots(figsize=(WIDTH/72,3.3));im=ax.imshow(pc,cmap='Blues',aspect='auto',vmin=0,vmax=pc.max())
    ax.set_xticks(range(5),['1–4','4–8','8–13','13–30','30–45']);ax.set_yticks(range(6),e.plan['rows'][0]['channel_sets']['set_channel_names']['nearest6']);ax.set_xlabel('频带（Hz）')
    for i in range(6):
        for j in range(5):ax.text(j,i,f'{pc[i,j]:.3f}',ha='center',va='center',color='white' if pc[i,j]>.22 else '#172d40',fontsize=8)
    fig.colorbar(im,ax=ax,label='单位范数 PC loading（无量纲）');fig.tight_layout();paths['loadings']=save_figure(fig,out,'pc1_loadings')
    fig,axs=plt.subplots(1,2,figsize=(WIDTH/72,3.2))
    keys=['Hb_only','pc1','mean6'];ms=e.residual['methods']
    for c in range(2):axs[0].bar(np.arange(3)+(c-.5)*.3,[ms[k]['component_nmse'][c] for k in keys],width=.3,label=['HbO','HbR'][c],color=['#2874ad','#bd5d69'][c])
    axs[0].set_xticks(range(3),[LABELS[k] for k in keys],fontsize=8);axs[0].set_ylabel('共同34窗分色团 NMSE');axs[0].legend(fontsize=8)
    for c in range(2):axs[1].plot([0,1],[ms[k]['cosine_with_physical_residual'][c] for k in ['pc1','mean6']],marker='o',label=['HbO','HbR'][c],color=['#2874ad','#bd5d69'][c])
    axs[1].set_xticks([0,1],['PC1残差','均值6残差']);axs[1].set(ylim=(0,1),ylabel='与 Hb-only 残差的余弦相似度');axs[1].legend(fontsize=8)
    fig.tight_layout();paths['residual']=save_figure(fig,out,'common_residual')
    diag=e.residual['physical_residual_diagnostics'];fig,axs=plt.subplots(1,2,figsize=(WIDTH/72,3.1))
    acf=np.array(diag['autocorrelation_1s_5s']);power=np.array(diag['standardized_spectral_band_power'])
    for c in range(2):
        axs[0].bar(np.arange(2)+(c-.5)*.28,acf[:,c],width=.28,label=['HbO','HbR'][c],color=['#2874ad','#bd5d69'][c])
        axs[1].bar(np.arange(3)+(c-.5)*.28,power[c],width=.28,label=['HbO','HbR'][c],color=['#2874ad','#bd5d69'][c])
    axs[0].set_xticks([0,1],['1秒','5秒']);axs[0].set(ylim=(-1,1),ylabel='残差滞后相关');axs[0].axhline(0,lw=.7,color='#444');axs[0].legend(fontsize=8)
    axs[1].set_xticks([0,1,2],['0–.05','.05–.15','.15–.5']);axs[1].set(xlabel='频带（Hz）',ylabel='标准化残差带内功率（SD²）');axs[1].legend(fontsize=8)
    fig.tight_layout();paths['temporal']=save_figure(fig,out,'residual_temporal')
    return paths


def make_report(e,out,examples,figs,wavepaths):
    d=Document(out,'REPORT');cfg=e.cfg;sc=e.screen;res=e.residual
    g=lambda split,c,condition='prediction',pairing='paired':e.metric(split,c,condition,pairing)
    v=lambda split,c,condition='prediction',pairing='paired':g(split,c,condition,pairing)['subject_equal_nmse_with_failure_penalty']
    d.page('固定 Tak 下的 EEG 传感器代理：首轮报告','summary.json；array_audit.json；residual_analysis_v1/residual_summary.json')
    d.text('配置解释、逐项结果分析与波形核验',size=14,gap=16)
    d.text('本轮检验：小幅调整原六通道读出，或改变等容量的传感器聚合范围，能否提供更有用的 EEG→Hb 驱动信息。所有物理预测均通过同一个单驱动 Tak 模型；另保留不使用 EEG 的 Hb-only 条件参考。')
    d.text('术语：EEG为脑电；HbO/HbR为氧合／脱氧血红蛋白；NMSE为归一化均方误差，越低越好；SD为标准差。全文只在冻结的训练尺度下比较。',size=9.2)
    d.table(['评价条件','双Hb NMSE','解释'],[
        ['原 PC1',num(v('evaluation','pc1')),'原六通道代理'],['均值6',num(v('evaluation','mean6')),'冻结的诊断候选，未晋级'],
        ['均值6，错误EEG',num(v('evaluation','mean6',pairing='wrong_EEG')),'固定相容供体'],['无 EEG',num(v('evaluation','zero')),'先验中心的零预测']], [1.3,1,2.5],size=9.3)
    d.text('主要发现：没有候选满足预定晋级标准。均值6相对PC1的评价NMSE降低4.55%，但整被试配对bootstrap的描述性95%改善区间为 −0.46%～17.27%，包含零，且均值6不优于无EEG基线。')
    d.text('原PC1仍有探索性的真实／错配差异。因此，本轮结果不是“EEG没有信息”，而是当前代理尚未建立稳定的净预测价值。十二通道与环带在内部筛选中没有显示独特优势。')
    d.text('420个逻辑单元全部终态：416个数值成功、4个联合拟合达到预算仍未收敛。失败、首次准备错误与全部负结果均保留。正文报告主要比较；独立图册逐窗展示全部48个开发／评价目标。')
    d.text('本报告只读取保留证据，不新增测量数据访问、不重新拟合模型、不重新选择参数。报告日期：2026-10-28。',size=9.2)

    d.page('阅读顺序与证据层级','resolved_config.json；plan.json；screen.json；tasks/；summary.json')
    d.table(['部分','应回答的问题'],[
        ['输入与物理约束','比较对象是否相同？模型看到哪些信息？哪些自由度固定？'],
        ['11种代理与训练','每项修改增加了什么自由度？系数、正则与空间选择如何冻结？'],
        ['筛选与评价','内部CV、开发选择、评价结果分别支持什么判断？'],
        ['联合与残差','改变EEG代理后哪些残差改变，哪些仍接近Hb-only残差？'],
        ['波形正文和完整图册','误差改善是否对应幅度与形状恢复？失败有没有被隐藏？'],
        ['边界与后续','哪些结果可支持下一假设，哪些不能上升为生理来源结论？']],[1,3],size=10)
    d.text('配置、原始任务记录和结果数组是数值事实源。screen.json持有折分、冻结系数及线性筛选；summary.json持有420单元的汇总；共同成功残差表来自residual_analysis_v1/residual_summary.json。array_audit.json独立核验了数组、训练坐标及错配输入。')
    d.text('图中“实测”均指经过已登记预处理、窗口参考和训练尺度变换后的Hb目标，不是未处理原始浓度，也不是无噪声真值。“物理参照”是有先验与有限优化预算的Hb-only条件拟合，不是全局可达性下界。')
    d.text('单个时间点、重复窗口和同一被试的两个记录均不当作独立被试。评价只有6个独立被试，全部区间为探索性描述；报告不进行多重校正后的显著性声明。')

    d.page('数据、划分与完整分母','plan.json；donor_inventory；array_audit.json/counts')
    d.table(['用途','被试','目标窗','当前用途'],[
        ['训练','17','102','PCA、共同Hb尺度、读出与内部三折选择'],['开发选择','6','12','最多三个候选的非线性核查与冻结评价候选'],['评价','6','36','固定两候选的开发重分析，不再选择模型'],['错误EEG供体','沿用目标被试','34个额外窗口','仅准备EEG；不增加目标数或独立被试数']],[1,0.55,0.95,2.9])
    d.text('全部来自公开Single-Trial；目标Hb对为FpzFp1，模板native pair index为3。只读已冻结父空间计划的身份、分组与几何，旧4Hz预测、校准和prepared数组不作为本轮输入。目标与供体共184个准备窗口，来自58个已登记原记录组。')
    d.text('每个训练／评价被试保留两个记录，每记录三个不重叠30秒目标；开发选择只取每记录按身份顺序的第一个窗口。原生EEG滤波支持可在同一记录内重叠，因此窗口之间并非独立。')
    d.text('本面板session_00为运动想象（MI），LMI/RMI分别为左／右手想象；session_01为心算任务，MA/BL分别为心算／休息条件。标签只用于记录身份与相容错配，不作为代理预测输入。PCA指主成分分析，PC1是第一主成分。',size=9.3)
    for split,label in [('train','训练'),('selection','开发'),('evaluation','评价')]:
        d.text(label+'被试：'+', '.join(s.replace('subject_','') for s in e.plan['subjects'][split]),size=9.3)
    d.text('这些评价被试此前已经被用于项目开发。本版只保证本次选择后冻结评价，不能称为全新的未见最终确认集。公开数据界限不改变其他协议的保护分区。')

    d.page('测量到模型坐标：每一步保持共同支持','resolved_config.json/tensor；prepared/；src/data/shared_driver_spatial.py')
    d.figure(figs['pipeline'],'流程图中的每条物理路径最终只输出一个标量驱动；两个Hb分量共享同一条状态传播。')
    d.text('EEG：200Hz原生电位；在90秒连续支持上，对1–4、4–8、8–13、13–30、30–45Hz分别作四阶Butterworth双向带通；按100ms不重叠箱求均方功率并取自然对数。随后裁出原30秒目标，减前5秒参考。10Hz是功率统计输出率，不是载波采样率。')
    d.text('原生EEG单位由MAT的cnt.yUnit核验为µV，沿用双侧乳突参考（linked mastoids）。读取的是已登记with occular artifact分支；本轮未新增ICA眼电去除、局部平均参考或其他电压空间变换。',size=9.3)
    d.text('90秒支持尽量以目标居中；靠记录边界时整体向内平移，保留实际crop offset。58个目标窗发生边界平移，未拼接或补造记录。原30秒目标身份和各代理的滤波支持完全一致。')
    d.text('Hb：中央v5/none缓存已完成整记录0.01–0.2Hz带通；本轮只分箱、作相同5秒参考和训练正比例缩放，不再次带通或读取原始Hb。完整记录处理先于目标隐藏，因此这是离线处理后特征预测，不能称实时或原始传感器隔离。')
    d.text('缓存声明motion_method=none；Single-Trial的Hb输出单位为项目近似相对坐标（relative_Hb_repo_approximation），并非已核定绝对浓度。共同成分可能包含生理信号或伪迹，本轮没有独立观测将二者分离。',size=9.3)

    d.page('固定 Tak：一个驱动与四个血管状态','frozen source/src/inference/shared_driver_rk4.py；resolved_config.json/fixed')
    d.figure(equation(out,'tak',[
        r'\dot{s}=\beta r-\kappa s-\gamma(f-1),\qquad \dot{f}=s',
        r'\dot{v}=\frac{f-v^{1/\alpha}}{\tau+\tau_v},\qquad f_{\rm out}=v^{1/\alpha}+\tau_v\dot v',
        r'\dot q=\frac{f E(f)/E_0-f_{\rm out}q/v}{\tau},\qquad E(f)=1-(1-E_0)^{1/f}',
        r'p\equiv v,\quad H_O=P_0(v-1)-Q_0(q-1),\quad H_R=Q_0(q-1)']))
    d.text('r是模型输入驱动；s是流量诱导信号；f、v、q分别为相对静息的血流、容量和脱氧Hb。p是相对总Hb，恒等于v，没有独立p状态或过程噪声。时间以秒计，τ与τᵥ为秒。')
    d.text('自由初态为s₀、log f₀、log v₀、log q₀。每个30秒窗口独立估计，不跨未观察间隔传播；每窗300个驱动网格值加4个初态变量。保存数组虽含r、s、f、v、p、q六列，p列只是v的确定性副本。')
    d.text('固定Hb关系先产生两个相对模型观测，再通过各自相同的时间参考算子。两色团不能分别获得EEG权重、时间核或可调补偿曲线。正状态与P₀v≥Q₀q的数值域检查不能替代生理真值验证。')
    d.text('此处复现的是项目采用的Tak血流核心与既有EEG/Hb观测接口，不声称实现完整DCM；窗口ODE也不是整记录带通算子的严格前向重放。')

    d.page('固定参数、观测尺度与可估对象','resolved_config.json/fixed；screen.json/coordinate')
    d.table(['项目','数值','含义及限制'],[
        ['τ / τᵥ','2 / 2秒','血流传输及黏弹性项，整个代理比较固定'],['κ / γ','0.64 / 0.32','流量诱导衰减与反馈系数；不逐窗重估'],['α / E₀','0.32 / 0.32','流量-容量指数参数与基线提取率'],['P₀ / Q₀','1 / 0.35','Hb模型参考值，不是个体实测浓度标定'],['β / EEG loading','1 / 1','固定神经血管增益与EEG观测loading'],['全局代理系数','0或5','原PC1固定；其他代理训练后跨记录冻结'],['当前窗口未知量','300＋4','驱动网格和四初态；物理参数数为0'],['EEG尺度因子',num(sc['coordinate']['eeg_factor'],7),'仅训练PCA的0.025 SD锚点'],['Hb共同正尺度',num(sc['coordinate']['hb_factor'],5),'两个色团同乘，保留幅比与符号'],['评分SD：EEG / O / R','0.025 / 0.033057 / 0.012540','冻结训练SD；不按当前窗口重新归一化']],[1.25,1.2,2.55])
    d.text('PCA协方差中心只用于求loading；实际代理沿用窗口参考坐标，不再减训练均值。全训练坐标使用102窗的30600个特征时间点。评分和优化排除最后一个网格点，尺度拟合沿用既有坐标owner的完整训练特征支持。')
    d.text('Hb共同尺度使两色团训练方差的均值等于0.025²，并非让HbO/HbR各自具有0.025 SD。评分分母与图中所注明的展示坐标必须分开理解。Single-Trial的本轮结果不标注为绝对µM。')

    d.page('三种推断条件与四种基线用途','resolved_config.json/model；nonlinear_tasks.json；frozen runner/assemble_fit_input')
    d.table(['条件','当前窗口输入','允许估计','主要用途'],[
        ['EEG→Hb','EEG；整段双Hb隐藏','驱动＋四初态，仅由EEG和先验估计','预测晋级主证据'],['联合重建','EEG＋完整双Hb','同一驱动＋四初态','解释拟合妥协'],['Hb-only','完整双Hb，无EEG','同一驱动＋四初态','固定物理条件参照'],['无EEG预测','无当前观测','先验中心静息解','判断使用EEG是否有净价值'],['训练模板','训练Hb平均轨迹','本窗口不拟合','检验共同时间结构基线'],['FIR-ridge','仅冻结EEG代理','本窗口不拟合，训练线性系数冻结','经验信息诊断']],[1.15,1.6,1.7,1.6],size=8.7)
    d.text('预测输入在组装时就将两个Hb列置为缺失；不借用目标参考统计、Hb-only解或联合解作为初值。全窗Hb只在拟合外评分。没有其他Hb通道作为输入，因此也不需要“仅其他Hb”的平行基线。')
    d.text('错误EEG只用于预测条件：保持目标Hb、模板坐标、物理参数和读出系数，替换为固定相容供体的EEG。未对错误配对重新训练读出。联合中EEG权重为0恰好对应Hb-only；正权重比较统一为1。')

    d.page('同一损失：数据误差、先验和数值收敛分开保存','resolved_config.json/solver；tasks/*/objective_components；frozen shared_driver_reconstruction.py')
    d.figure(equation(out,'objective',[
        r'J=0.4\!\sum_{(t,m)\in\Omega}\!\left(\frac{\widehat y_{tm}-y_{tm}}{\sigma_m}\right)^2',
        r'\quad+0.01\sum_t\frac{(r_{t+2}-2r_{t+1}+r_t)^2}{\Delta t^3}',
        r'\quad+100\big[s_0^2+(f_0-1)^2+(v_0-1)^2+(q_0-1)^2\big]',
        r'\quad+\Delta t\sum_t\left(\frac{\log f_t}{\log 2}\right)^2,\qquad\Delta t=0.1\ {\rm s}']))
    d.text('Ω表示当前条件真正可见的观测网格；σ是冻结的训练SD。0.4等于0.1秒/0.25秒，用于保持每秒工程数据权重，不是独立样本数或校准噪声似然。曲率与log-flow先验按物理时间离散；初态先验每窗计一次，体积只罚一次。')
    d.table(['数值项','本轮设置','解释'],[['起点','统一单起点','预算下最佳可得局部解'],['优化预算','最多300迭代、600评价','达到预算而未达梯度标准仍失败'],['梯度容差','1×10⁻⁵','优化收敛标准，不是NMSE阈值'],['RK4积分','4子步；8子步复核','0.1秒输出网格与内部积分分开'],['积分通过条件','最大差≤0.005训练SD','全部模态检查，不能用缩短评分段掩盖差异']],[1,1.6,2.6])
    d.text('Hb-only拟合保留纯数据平方和（SSE）及各先验代价。残差可能来自观测结构、先验、噪声或局部优化，不能直接称为固定物理模型的全局不可达部分。')

    d.page('层0–1：原PC1与五个频带增益','screen.json/coordinate,models；src/data/ssm_prepared.py')
    d.figure(equation(out,'proxy',[
        r'u_b(t)=s_E\sum_{c=1}^{6}p_{cb}X_{cb}(t),\qquad z_0(t)=\sum_{b=1}^{5}u_b(t)',
        r'z_1(t)=\sum_{b=1}^{5}a_bu_b(t),\qquad a_b=1\ \Longrightarrow\ z_1=z_0']))
    d.text('X是已按前5秒参考的逐通道log-power，30维按通道优先、频带其次排列；p是原PC1单位范数loading；sE是原训练尺度。五个a在训练集学习，并向1正则化。通道集合、每频带内空间模式、状态空间模型（SSM）的状态数均不变。')
    d.figure(figs['loadings'],'颜色及单元数字为训练PC1 loading。较大的低频loading只说明该训练特征协方差方向，不是相应频带的神经贡献。')
    d.text('最终频带增益约为 −0.254、0.403、1.141、1.067、0.973。它允许原低频组合反向或减弱；这种改动不能恢复PC1投影已经丢弃的全部空间信息。',size=9.2)

    d.page('层2–3：等容量均值与一个共同环带参数','resolved_config.json/screen；plan.json/channel_sets')
    d.figure(equation(out,'spatial',[
        r'm_{S,b}(t)=\frac{1}{|S|}\sum_{c\in S}X_{cb}(t),\quad z_S(t)=s_E\sum_b w_bm_{S,b}(t)',
        r'm_{\alpha,b}=(1-\alpha)m_{{\rm inner},b}+\alpha m_{{\rm outer},b},\quad\alpha\in\{0,0.25,0.5\}']))
    d.table(['设置','相对最近6通道的修改','检验问题'],[
        ['均值6','同6通道，换成逐频带平均','原固定读出是否限制输入'],['第6换第7','只替换边界电极F3→F7','恰好六通道的边界稳定性'],['均值3 / 均值12','改变等权平均范围，仍5系数','哪种聚合范围有用'],['环带0.25','内6权重0.75、外6权重0.25','少量外圈信息是否补充'],['内6＋远6','相同12通道数和5系数','收益是否具有近邻特异性'],['三个随机扩展','内6固定，外部随机6路','增加通道平均或共同成分对照']],[1.3,2,2.1])
    d.text('均值族不是PC1的数学子模型。3、6、12通道是同形式容量的替代表示，不是严格嵌套；只有环带α=0与0.5分别精确复用均值6与均值12。0.25是本轮唯一额外环带格，五个频带共享同一α。')
    d.text('所有均值族使用原EEG锚点sE，不对输出重新标准化。ridge中心为零，所以均值读出可以收缩；不能将较小预测幅度自动解释为更准确的生理驱动。随机种子固定为20261027、20261028、20261029。')

    d.page('空间支持是传感器几何，不是来源定位','plan.json/geometry_identity,channel_sets；公开模板几何')
    d.figure(figs['geometry'],'每个点是一条EEG通道，星号为目标Hb位置。图上投影仅用于阅读；实际排序使用共同模板坐标中的三维欧氏距离，距离相等时按通道名排序。')
    d.text('最近六路为AFp1、AFp2、AFF5h、AFF1h、AFF2h、F3。近邻外圈为F7、AFF6h、F4、F8、FCC5h、FCC3h。全部候选保留同一个FpzFp1 HbO/HbR对。')
    d.text('模板坐标单位是normalized_head_unit，不是毫米，也不是个体皮层位置。远端控制与随机控制只能帮助解释传感器信息范围，不能将预测系数当作局部生理贡献。')

    d.page('每一组实际使用的通道','plan.json/channel_sets/set_channel_names；随机种子已冻结')
    names=e.plan['rows'][0]['channel_sets']['set_channel_names']
    d.table(['集合','通道列表'],[[label,', '.join(names[key])] for label,key in [
        ('最近3','nearest3'),('最近6','nearest6'),('第6换第7','sixth_to_seventh'),('外圈6','outer6'),('远端6','far6'),('随机外圈1','random6_0'),('随机外圈2','random6_1'),('随机外圈3','random6_2')]], [1,4.6],size=9.2)
    d.text('均值12＝最近6＋外圈6；远端和三个随机扩展均为最近6＋相应外部6路。三个随机外圈均排除内6且不重复采样，所有目标使用同一训练规则。模型不在每个频带、被试或窗口单独选择空间范围。')
    d.text('通道选择只依赖保留几何和固定种子，不按Hb误差、EEG幅度或信号相关性筛选。正文中原PC1始终保留，未把均值6悄悄替换成新的“原基线”。')

    d.page('线性筛选与三折训练：实际计算路径','frozen low_capacity_eeg_proxy.py；screen.json/folds,selected_ridges')
    d.figure(equation(out,'ridge',[
        r'\widehat y_H=d+K_{\theta_0,\Omega}X_G a',
        r'\widehat a=\arg\min_a\;\overline{\left\|\frac{d+KX_Ga-y_H}{\sigma_H}\right\|^2}'
        r'+\lambda\,\frac{\operatorname{tr}(G)}{5}\|a-a_{\rm ref}\|^2']))
    d.text('先在静息点线性化实际Tak离散前向模型，保留同一EEG/Hb参考算子、四初态和驱动/流量先验。K先由处理后的EEG条件反演驱动及初态，再映射到Hb；它不是把参考后的EEG直接当原始驱动卷积。')
    d.text('a是5维共享读出，d是固定条件映射offset；G是训练设计在评分坐标下的加权Gram矩阵。PC1频带增益以全一向量为中心；均值族以零向量为中心。λ候选为0.01、0.1、1，表示Gram平均对角线的无量纲倍数。')
    d.text('17名训练被试分三折：每次只用另外两折重拟PCA、尺度及读出，在整名留出被试上评分。按训练CV选择λ与入围表示后，再用全部17人重拟并冻结；开发和评价Hb不回流到训练坐标。')
    d.table(['训练CV折','留出被试'],[[str(i+1),', '.join(s[-2:] for s in fold)] for i,fold in enumerate(sc['folds'])],[1,4])
    d.text('原PC1不学习读出，序列化的ridge标签0.01不代表它拟合了ridge。其余10种代理各有9次折内拟合与1次全训练重拟，共100次五系数拟合。CV是内部选择证据，不能替代冻结后的评价。',size=9.2)

    d.page('11种代理的内部筛选：扩大范围的差异很小','screen.json/cv_subject_mean_nmse；selected_ridges')
    d.figure(cv_figure(e,out),'左图柱长从零开始；右图为非PC1候选的独立放大刻度，虚线是均值6。每个数值先按窗口、记录和被试汇总，最后跨17名训练被试等权。')
    d.table(['代理','CV NMSE','λ','相对PC1改善'],[[LABELS[k],num(value,6),('固定' if k=='pc1' else num(sc['models'][k]['ridge'],2)),f"{100*(1-value/sc['cv_subject_mean_nmse']['pc1']):.2f}%"] for k,value in sc['cv_subject_mean_nmse'].items()],[1.7,1,0.65,1.2],size=8.2)
    d.text('均值12相对均值6的内部NMSE改善只有约0.006%。远端及随机扩展略低于近邻扩展；这些很小的内部差异不支持“近邻外圈特异有效”。进入非线性开发核查的是原PC1、同六通道较优的均值6及空间候选均值12。',size=9.1)

    d.page('逐项筛选判断：收益不能都归给空间扩展','screen.json/cv_subject_mean_nmse；内部选择证据，不是独立确认')
    cv=sc['cv_subject_mean_nmse']
    d.table(['实验项','观察','可得出的有限判断'],[
        ['PC1五频带修正',f"1.444848 → {cv['pc1_band_gain']:.6f}，相对PC1降{100*(1-cv['pc1_band_gain']/cv['pc1']):.2f}%",'同样六通道已能改变误差；并非远端信息带来的收益'],
        ['六通道均值',f"{cv['mean6']:.6f}；较频带修正低{100*(1-cv['mean6']/cv['pc1_band_gain']):.3f}%",'同六通道候选差异很小；按预定训练排序入围'],
        ['第6换第7',f"{cv['sixth_to_seventh']:.6f}；较均值6高{100*(cv['sixth_to_seventh']/cv['mean6']-1):.3f}%",'此边界替换未引发明显内部误差变化，不能据此证明等价'],
        ['均值3 / 均值12',f"{cv['mean3']:.6f} / {cv['mean12']:.6f}",'扩大或缩小聚合范围未显示大的变化；函数空间并非嵌套'],
        ['环带0.25',f"{cv['ring025']:.6f}；较均值6高{100*(cv['ring025']/cv['mean6']-1):.3f}%",'唯一中间混合格没有额外可辨别收益'],
        ['内6＋远6',f"{cv['nearest6_far6']:.6f}，较均值6低{100*(1-cv['nearest6_far6']/cv['mean6']):.3f}%",'近邻外圈不是本面板内部最低误差；机制尚未区分'],
        ['三个随机扩展','1.256922 / 1.258053 / 1.257635','与近邻扩展同量级，不能建立局部空间关系的特异性']],[1.3,1.9,2.5],size=9)
    d.text('以上差异均是同一17名训练被试的内部CV点估计，没有为这些11项比较建立多重校正后的显著性结论。原PC1、均值6、均值12之外的代理没有做本轮非线性评价，因此不能把这一页写成全部模型的泛化排名。')
    d.text('空间均值、通道数及正则共同影响信息与幅度。近邻、远端或随机集合接近，可能与共同成分、平均降噪或信息不足相容；这些可能机制没有在本轮被独立识别。')

    d.page('训练冻结系数：同一锚点，允许符号变化','screen.json/models；screen.json/coordinate')
    d.table(['代理','1–4','4–8','8–13','13–30','30–45'],[[LABELS[k]]+[num(x,3) for x in m['coefficients']] for k,m in sc['models'].items()],[1.7,1,1,1,1,1],size=8.2)
    d.text('PC1行的五个1作用于已拆分的PC loading贡献；均值族的五个权重作用于通道均值。因此不同族的系数大小不能直接比较为同一位置的生理强度。PC1频带修正允许第一频带由正贡献转为负增益；均值族可能由正负频带组合相互抵消。')
    d.text('没有按新代理输出SD再定标。此举保留严格嵌套关系，也允许模型通过降低代理幅度来减少有害约束。是否利用了EEG信息，要结合真实／错配、无EEG和训练模板共同判断，而不能仅看相对原PC1的误差下降。')
    d.text('均值12的λ=1，而均值6的λ=0.01，说明两者同时包含训练选择的正则差异。实验比较的是完整冻结表示与训练规则的结果，不能只将差值归因于通道数量。')

    d.page('开发选择：改善均值不等于通过晋级','evaluation_selection.json；summary.json/selection；screen.json/shortlist')
    d.table(['开发12窗／6人','PC1','均值6','均值12'],[
        ['真实EEG NMSE',num(v('selection','pc1')),num(v('selection','mean6')),num(v('selection','mean12'))],
        ['错误EEG NMSE',num(v('selection','pc1',pairing='wrong_EEG')),num(v('selection','mean6',pairing='wrong_EEG')),num(v('selection','mean12',pairing='wrong_EEG'))],
        ['相对PC1改善','基准','14.23%','13.61%'],['改善被试数','基准','3/6','3/6'],['是否晋级','保留基线','否','否']],[1.6,1,1,1])
    d.text('开发无EEG NMSE为1.79692，训练模板为1.61153。均值6和均值12虽然平均优于PC1，却都只有3/6被试改善，没有达到至少三分之二；两者还不及训练模板。候选均未晋级。')
    d.table(['预定要求','含义'],[['≥3%相对NMSE下降','工程筛选阈值，不是显著性水平'],['≥2/3被试方向改善','避免均值被少数被试支配'],['优于真实相容错配与无EEG','且真实／错配双方成功的至少3名被试、至少一半目标可配对'],['优于训练模板','排除仅共同时间形状解释的收益'],['预测数值失败率不增加','失败不靠成功子集删除']],[1.8,3.2])
    d.text('按预定fallback保留误差最低的不同候选均值6，与PC1一起进入评价；它是诊断比较，不是已合格胜者。评价不反向调整候选、λ、尺度或阈值。')

    d.page('评分公式、归一化与区间读法','frozen runner/score_nmse,_subject_score,paired_comparisons；resolved_config.json/endpoints')
    d.figure(equation(out,'metrics',[
        r'L_w=\frac{1}{2|T|}\sum_{t\in T}\sum_{c\in\{O,R\}}\left(\frac{y_{tc}-\widehat y_{tc}}{\sigma_c}\right)^2',
        r'L_s=\frac{1}{|R_s|}\sum_{r\in R_s}\frac{1}{|W_r|}\sum_{w\in W_r}L_w,\quad L=\frac{1}{|S|}\sum_s L_s',
        r'\mathrm{gain}=1-\frac{L_{\rm candidate}}{L_{\rm baseline}},\qquad\mathrm{NRMSE}=\sqrt{L}']))
    d.text('T为索引50:299，即5.0至29.8秒的249个网格点；5秒参考段不计分，29.9秒最后一箱排除。σO=0.03305661，σR=0.01254036。先算每窗两色团、时间均方误差，再记录等权、被试等权；两个色团最终权重相等。')
    d.text('NMSE不是相关系数，也不是百分比。相对改善的分母是基线NMSE；报告归一化均方根误差（NRMSE）时在最终均方后开方。各色团均方根（RMS）幅比使用未再去均值的参考坐标；Pearson相关则在评分段分别去均值，因此二者不能互换。')
    d.text('1000次整被试配对bootstrap保留一名被试的全部窗口、记录和配对方法。物理参数、尺度、读出及选择均冻结；没有重新选择模型，也没有多重校正。6名评价被试的描述性区间不能支持总体确认或等价主张。')
    d.text('完整分母汇总对数值失败使用NMSE=100的固定惩罚，同时另列失败率及共同成功子集。报告不能把候选各自成功子集的均值直接当作配对收益。错配缺失不被赋巨大误差来制造信息收益；本轮所有评分目标均找到供体。')

    d.page('冻结评价：均值6未超越无EEG基线','summary.json/evaluation；paired_comparisons/evaluation|mean6_vs_pc1')
    d.figure(subject_figure(e,out),'左图每条线是一名被试的记录等权结果，不是单个窗口；右图负值表示均值6误差低于PC1。相同六被试的全部36窗均数值成功。')
    d.table(['预测条件','NMSE','相对PC1改善'],[[LABELS[c],num(v('evaluation',c)),f"{100*(1-v('evaluation',c)/v('evaluation','pc1')):.2f}%"] for c in ['pc1','mean6','zero']],[2,1,1.5])
    d.text('均值6与PC1的NMSE差为−0.10462，描述性95%区间为[−0.24419, 0.01789]。均值6比无EEG高0.00832；该差的描述性95%区间为[−0.00888, 0.02504]。现有证据不足以把这次相对PC1的改善解释为净跨模态收益。')
    d.text('均值6的HbO/HbR预测RMS分别约为目标的9.49%/10.86%，PC1为47.04%/54.75%。降低代理影响、趋近零预测与结果相容，但这仍是解释性推断，不是已经识别的因果机制。')

    d.page('代理与驱动波形：观察收缩，而不是重新归一化','prepared/*.npz；screen.json/models；tasks/*/driver；EEG数据项复算')
    d.figure(figs['proxy'],'与后文相同的三个代表例。每行一例；同列纵轴统一。左列是经过参考的观测代理，右列是仅由EEG与先验估计、尚未参考的模型驱动，两列不同坐标的偏移不能直接比较。')
    d.table(['被试／样例','PC1代理RMS','均值6代理RMS'],[[x['row']['subject'][-2:]+' / '+x['label'],num(r['proxy_RMS_training_EEG_SD']['pc1'],3),num(r['proxy_RMS_training_EEG_SD']['mean6'],3)] for x,r in zip(examples,figs['proxy_ratios'])],[1.8,1,1])
    d.text('RMS在5.0–29.8秒计算并除以固定EEG训练SD=0.025；没有按当前窗口或候选输出再归一化。代理由保留的逐通道功率和冻结系数重算，并验证能精确复现保存任务的EEG数据代价。',size=9.2)
    d.text('曲线的幅度变化是读出改变后实际出现的现象。驱动仍受曲率、流量和初态先验约束，因此不能把左侧代理直接当作右侧生理驱动。两者均不是真实神经活动测量。',size=9.2)

    d.page('错误配对：PC1有配对关联，均值6敏感性变弱','plan.json/wrong_eeg_partners；residual_analysis_v1/prediction_comparisons')
    d.figure(pairing_figure(e,out),'每根柱为一名被试的真实EEG NMSE减错误EEG NMSE；两个子图采用同一纵轴，负值有利于真实配对。每个供体保持相同被试、任务、条件和montage；同记录90秒滤波支持必须不重叠。')
    d.table(['代理','真实 / 错配 NMSE','真实−错配差的95%描述区间'],[
        ['PC1','2.29825 / 2.51105','[−0.37734, −0.09953]'],['均值6','2.19363 / 2.19686','[−0.02460, 0.01589]']],[.8,1.9,2.4])
    d.text('PC1真实配对较错配降低8.47%，6/6被试方向一致，但PC1仍未优于零预测或训练模板。配对关联与净预测价值是两个不同问题，不能从后者未建立推断EEG完全没有信息。')
    d.text('均值6的真实／错配差只约0.15%，区间跨零。全部相关预测任务均成功，微小差值不是通过对错配失败罚分获得。开发与评价共48个评分目标对应34个额外供体窗；供体身份对所有代理一致。')

    d.page('FIR-ridge：小型经验预测没有支持新代理','screen.json/fir；fir_evaluation_scores.json；fir/*.npz')
    d.figure(equation(out,'fir',[
        r'\widehat y_c(t)=A_H\sum_{\ell\in\{0,0.5,1,2,4,8\}{\rm s}} b_{c\ell}z(t-\ell),\quad c\in\{O,R\}']))
    d.text('FIR是有限冲激响应线性读出：使用冻结标量代理的6个时延项，每个Hb分量各6个系数，共12个训练参数，无额外截距。它不通过血流方程，只作为信息诊断，独立色团权重不会进入物理模型。')
    d.text('前窗历史在参考期相对坐标中采用零值假设，随后对设计乘同一Hb参考算子。λ仍在训练三折中从0.01、0.1、1选择，全训练重拟后冻结。评价均值12的FIR属于预定信息对照，不是增加一个非线性评价候选。')
    firrows=[]
    for c in sc['shortlist']:
        fm=sc['fir']['models'][c];val,_=aggregate(e.summary['FIR'][c]['rows']);dev,_=aggregate(fm['development'])
        firrows.append([LABELS[c],num(float(fm['ridge']),2),num(dev),num(val)])
    d.table(['代理','FIR λ','开发NMSE','评价NMSE'],firrows,[1.5,.8,1,1])
    d.text('评价FIR的PC1为2.15995，低于物理PC1的2.29825；但相对无EEG只低约1.16%。新代理的FIR没有改善原PC1：均值6为2.18895，均值12为2.18514。这里没有为FIR方法间差异另算确认性检验。')
    d.text('本轮不支持“新空间代理含有明显可利用信息，只是物理模型无法利用”的强结论。可以保留经验PC1作后续信息上限诊断，但不能据其微小平均收益扩大模型结构。')

    d.page('经验与物理预测的波形：仍需逐窗核验','fir/*.npz；tasks/PC1 prediction；与主体相同的解释样例')
    d.figure(figs['fir_waveforms'],'每列一个既定代表例，上下为HbO/HbR。黑实线为处理后实测；蓝虚线为PC1物理预测；青点划线为PC1 FIR；橙点线为均值6 FIR。每面板独立纵轴，单位为相应色团的训练SD。')
    d.text('样例由原PC1物理预测误差的低、中、高秩选择，不按FIR拟合好坏挑选。图中全部曲线都在当前窗口整段Hb隐藏条件下产生；FIR系数与代理均由训练集冻结，没有逐窗回归目标Hb。')
    d.text('经验读出可以改变符号、偏移与时间组合，但更接近某段峰谷不等于完整窗口误差改善。总体评价中，原PC1 FIR为2.15995，均值6 FIR为2.18895；它们都与零预测2.18531接近，不能仅据一幅波形宣布跨模态信息恢复。')
    d.text('物理预测与FIR的自由度和输出约束不同：前者的双Hb通过同一物理轨迹产生；后者每色团有六个独立时延系数。FIR用于分辨接口是否可能限制信息，不能作为更准确物理状态的替代证据。')

    d.page('联合重建：必须同时呈现失败和共同分母','summary.json/groups；residual_analysis_v1/methods')
    d.table(['评价联合条件','成功/计划','各自成功NMSE','全计划惩罚NMSE','共同34窗NMSE'],[
        [LABELS[c],f"{g('evaluation',c,'joint')['successful']}/36",num(g('evaluation',c,'joint')['success_only_subject_equal_nmse']),num(v('evaluation',c,'joint')),num(res['methods'][c]['joint_nmse'])] for c in ['Hb_only','pc1','mean6']],[1.2,.85,1.25,1.4,1.4],size=8.1)
    d.figure(figs['residual'],'左图只比较三个方法共同成功的34窗、6被试。右图为与Hb-only残差的有符号余弦相似度；不是解释率、能量占比或生理归属。')
    d.text('各自成功子集会给出误导：均值6为0.33279，看似优于Hb-only的0.40846；但统一为共同34窗后，Hb-only为0.26621，PC1联合为0.35158，均值6联合为0.33279。后者较PC1下降5.35%，仍比Hb-only高25.01%。')
    d.text('完整失败率分别为1/36、1/36、2/36。共同子集只服务残差形状比较，不能替代这份失败记录，也不说明加入EEG一定改善条件拟合。')

    d.page('物理参照残差保留的方向与时间结构','residual_analysis_v1/physical_residual_diagnostics,methods；tasks/*/residual_diagnostics')
    d.figure(figs['temporal'],'两色团分别使用各自冻结训练SD。柱值先按窗口、记录、被试等权汇总；频段功率来自评分段去均值的周期图，不是归一到总和为1的占比。')
    d.text('均值6联合残差与Hb-only残差的HbO/HbR余弦相似度为0.905/0.911，Pearson相关为0.906/0.909。它说明修改代理后仍保留相近的残差方向，不是“物理模型外成分占90%”。')
    d.text('相对PC1，均值6联合残差变化量的逐窗双Hb标准化RMS再分层平均为0.16874。Hb-only残差在1秒滞后的相关为0.880/0.887；5秒为−0.082/−0.142。参考后的短窗与已有带通都可能形成时间相关，不能仅凭自相关提出遗漏生理过程。')
    d.text('评分支持约24.9秒，基本频率分辨率约0.040Hz。0–0.05Hz频段几乎没有足够独立频点，不能据柱高精确确定慢过程。单目标Hb对不支持跨位置共同结构估计；本版也没有估计跨色团相位、运动或外周量的独立贡献。')

    d.page('四个预算失败：保留有限轨迹，不计为成功','array_audit.json/failed_cells；tasks/*/starts；revision.json')
    failed=[t for t in e.all_tasks if not t['success']]
    d.table(['被试/记录/事件','方法','状态','预算内NMSE'],[[f"{t['subject'][-2:]}/{t['native_record'][-2:]}/{e.rows[t['record_id']]['event_id']}",LABELS[t['candidate']],t['numerical_status'],num(t['nmse'])] for t in failed],[1.3,1.1,2,1],size=8.4)
    d.text('4个失败全部发生在评价联合／Hb-only条件，达到600次评价预算仍未满足收敛要求。subject_05/session_01/event4的三个方法均失败；subject_06/session_00/event4只有均值6联合失败。有限误差值与保存波形仅作诊断，不替代全计划惩罚和失败率。')
    d.text('积分加密差远低于0.005训练SD阈值，因而这些失败不能简化为积分分辨率不足；反过来，积分精确也不能证明优化收敛。没有为得到更好结果而延长预算、换样本或删除失败。')
    d.text('首次measured_v1在原生信号读取前因传入字符串路径而失败。修复仅在分组worker边界将路径转为Path，原证据与冻结源码保留；measured_v2重新准备全部184窗。该工程失败与本页四个优化失败分开记录。')

    d.page('波形读法与可复核的样例选择','plan.json；共同成功34窗；tasks/pc1 prediction；waveform_selection.json')
    d.text('正文从三个方法共同成功的34个评价窗口出发，按原PC1隐藏预测NMSE升序排序，分别选低端、靠中位、最高端样例；按距离目标秩最小原则优先选择不同被试。该选择只用于解释，不反馈训练或晋级。两个未收敛目标另页展示；图册覆盖全部48个目标。')
    d.table(['样例','被试/记录/事件','排序位置','PC1预测NMSE'],[[x['label'],f"{x['row']['subject'][-2:]}/{x['row']['record'][-2:]}/{x['row']['event_id']}",f"{x['rank']}/{x['denominator']}",num(e.tasks[(x['row']['id'],'pc1','prediction','paired')]['nmse'])] for x in examples],[1,1.4,1,1.3])
    d.text('每页两列为HbO/HbR，第一行整段目标隐藏预测，第二行全可见联合与Hb-only，第三行为“目标−预测”的联合残差。黑色实线是处理后实测；PC1蓝虚线、均值6橙实线、Hb-only紫点线；开发图册另有均值12绿点划线。灰色0–5秒为参考期，5秒虚线为原事件锚点，末箱不评分。')
    d.text('纵轴为各色团自己的训练SD，所以不能直接比较HbO/HbR的绝对浓度幅比；同一色团的目标与所有模型使用同一尺度。正文三个代表例同一面板固定纵轴，跨例可直接比较幅度；完整图册为提高局部可读性采用明确标注的独立纵轴。')
    d.text('零预测是第一行的水平零线。全部方法都是完整窗口的离线条件估计，时间横轴不表示在线预测延迟。没有把时间点当独立样本，也没有以波形相似证明潜状态或参数恢复。')

    for x in examples:
        row=x['row'];title=f"{x['label']}样例：被试{row['subject'][-2:]}，记录{row['record'][-2:]}，事件{row['event_id']}"
        d.page(title,'tasks/*.npz；'+row['id'])
        d.text(f"{row['task']} / {row['condition']}；EEG目标起点{row['eeg_start_s']:.3f}s，Hb目标起点{row['hb_start_s']:.3f}s。三代表例的对应面板纵轴一致。",size=9,gap=8)
        d.figure(wavepaths[row['id']],window_caption(e,row))
        d.text(waveform_reading(e,row),size=8.9)

    for i,identity in enumerate(res['excluded_window_ids']):
        row=e.rows[identity]
        d.page(f"未收敛目标 {i+1}：保留波形和失败身份",'tasks/*.json,npz；'+row['id'])
        d.text(f"被试{row['subject'][-2:]} / {row['record']} / event{row['event_id']}，{row['task']} / {row['condition']}。本页使用独立纵轴；×符号标记未收敛轨迹，方法线型保持不变。",size=9)
        d.figure(wavepaths[identity],window_caption(e,row))
        d.text('第一行隐藏预测仍为成功结果；失败发生于第二、三行所示的某些联合条件。不能因为图上已有有限曲线就把该任务纳入共同成功分析。',size=9)

    d.page('合成先导与独立审计：验证实现，不替代实测结论','../20261027_synthetic_v1/synthetic_summary.json；array_audit.json；verification.json')
    syn=read(e.run.parent/'20261027_synthetic_v1/synthetic_summary.json')
    d.table(['检查','结果','覆盖边界'],[
        ['正式网格合成先导','10/10成功','两条已知Tak信号，真实/错配、联合、Hb-only和零预测'],['积分加密最大差','4.12×10⁻⁹训练SD','合成4/8子步比较'],['独立数组审计','420 JSON＋420 NPZ','评分支持、分色团与联合NMSE、汇总分母一致'],['训练坐标重算','102窗 / 30600点','PCA、共同Hb尺度及SD一致'],['错误EEG输入核对','108个任务','供体身份及实际数据项一致'],['共同成功残差','34/36窗 / 6人','另列失败目标，未重拟合']],[1.4,1.3,2.7])
    sm=[]
    for candidate,condition,pairing in [('pc1','prediction','paired'),('pc1','prediction','wrong_EEG'),('zero','prediction','paired'),('Hb_only','joint','paired')]:
        rows=[r for r in syn['results'] if (r['candidate'],r['condition'],r['pairing'])==(candidate,condition,pairing)]
        sm.append([LABELS[candidate]+(' 错配' if pairing=='wrong_EEG' else ''),num(np.mean([r['nmse'] for r in rows]),6)])
    d.table(['合成条件','双Hb NMSE'],sm,[2,1])
    d.text('合成信号验证了这组已知生成条件下的程序路径与对照行为。它不校准实测噪声，不证明所有先验都准确，也不能把两个合成记录的成功当作实测生理资格。')

    d.page('结论：保留配对关联，暂停扩大空间自由度','summary.json；residual_analysis_v1/residual_summary.json；registry判读')
    d.table(['本轮问题','结果','允许的解释'],[
        ['同六通道读出能否降低PC1误差？','内部及平均评价误差下降','改善与读出收缩相容；未建立净预测价值'],['近邻外圈是否独特有用？','均值6/12几乎相同，远端/随机不差','未建立局部空间优势'],['是否真的使用了EEG信息？','PC1真实优于错配，但不优于零基线','有限配对关联，不是EEG完全无信息'],['更自由的经验读出是否支持新代理？','新代理FIR未优于PC1 FIR','没有明确的新增可利用信息证据'],['共同物理残差是否消失？','均值6残差仍接近Hb-only方向','固定参考有后续诊断价值，不能证明全局不可达']],[1.5,1.7,2.2],size=9)
    d.text('下一步应先围绕现有PC1的配对关联、预测幅度和共同物理残差提出可区分的具体假设。保留原PC1、均值6、零预测与经验PC1 FIR作为不同用途的对照；不要只选择误差最低的联合重建当作预测成功。')
    d.text('首轮没有执行电压层局部平均参考、全头空间优化、额外驱动、独立Hb补偿成分或物理结构交叉实验。短窗、单目标、公开开发反复使用和4个优化失败共同限定结论。扩大结构前需要新的明确问题与冻结验证设计。')
    d.text('表述复核：本报告把机制解释标为推断，将描述性区间与确认性检验分开，保留全部负结果、失败数及指标分母。公式和数值均依据保留合同与数组，没有为叙事补造新实验。',size=9.2)

    d.page('有限执行矩阵与运行配置','nonlinear_tasks.json；resources.json；launch.json；resolved_config.json/resources')
    d.table(['任务块','数量','当前窗口是否优化'],[
        ['开发：3候选×12窗×预测/联合','72','是'],
        ['评价：2候选×36窗×预测/联合','144','是'],
        ['48窗各一次Hb-only参考','48','是'],
        ['开发及评价的错误EEG预测','108','是，冻结正确配对训练所得系数'],
        ['48窗无观测／无EEG基线','48','否，先验中心的确定性前向'],
        ['总计','420逻辑单元','372次非线性优化＋48次零基线前向']],[2.4,1,2.6],size=9)
    d.text('因此416个“数值成功单元”包含48个无需优化的零预测；实际非线性优化为368成功、4未收敛。线性训练筛选和FIR训练另计，不混入420单元分母。')
    d.table(['资源或复现项','冻结执行设置'],[
        ['并行','36 worker，最多72个在途任务；数值库单线程'],
        ['资源依据','52物理核、用户CPU配额80核；兼顾当时其他作业，未使用GPU'],
        ['监督','user systemd，Linger=yes；CPUQuota 3600%，MemoryMax 96G'],
        ['种子','全局20261027；三个随机空间集合另固定三种子；外部分组沿用父metadata'],
        ['执行版本','measured_v2固定源快照；报告仅消费其完成结果'],
        ['断点与停止','原子保存逐任务状态；有限矩阵终态后停止，不按结果扩展实验']],[1.3,4.2],size=9)
    d.text('资源设置保证断开交互连接后继续运行，并避免线程超额。它们不是模型超参数，不能用机器并行数增加独立样本量。报告生成未再执行上述拟合。',size=9.2)

    d.page('复现入口、证据定位与完整图册索引','experiments/README.md；RESULTS_INDEX.md；launch.json；source_snapshot.tar.gz')
    d.text('主要运行：experiments/runs/physiology_semantic_tokenizer/shared_driver_sensor_proxy_v1/20261027_measured_v2。报告是该完成运行的版本化只读导出，不替代其配置、任务记录或项目状态。',size=9.2)
    d.table(['文件或目录','应核对的内容'],[['resolved_config.json','物理参数、代理矩阵、时间支持、先验与停止规则'],['plan.json','目标、供体、被试划分、几何与窗口支持'],['screen.json','三折、训练坐标、全部系数、正则和FIR训练'],['nonlinear_tasks.json / tasks/','420个计划任务、状态、目标和预测数组'],['summary.json / array_audit.json','完整分母汇总与独立复算'],['residual_analysis_v1/','共同34窗残差及真实/错配/zero/template配对'],['WAVEFORM_ATLAS.pdf','12开发窗＋36评价窗，逐页保留条件与失败'],['formulas/*.tex / figures/*.png','可修改公式源码与实际嵌入的位图'],['delivery_checks.json','PDF图像对象、文本、边界、页数和数值检查']],[1.9,3.4],size=9)
    d.text('报告重建入口：experiments/scripts/render_shared_driver_sensor_proxy.py --run <completed_run> --output <fresh_child_export>。必须使用新导出目录；不覆盖日期报告。图册中的A01–A12为开发，A13–A48为评价，均按subject/record/event身份排序。')
    return d.save()


def make_atlas(e,out,wavepaths):
    d=Document(out,'WAVEFORM_ATLAS')
    rows=sorted((r for r in e.plan['rows'] if r['split']!='train'),key=lambda r:(0 if r['split']=='selection' else 1,r['subject'],r['record'],r['event_id']))
    for i,row in enumerate(rows):
        split='开发' if row['split']=='selection' else '评价'
        d.page(f"A{i+1:02d} {split} | 被试{row['subject'][-2:]} / 记录{row['record'][-2:]} / 事件{row['event_id']}",row['id']+'；tasks/*.json,npz')
        d.text(f"{row['task']} / {row['condition']}；EEG/Hb起点={row['eeg_start_s']:.3f}/{row['hb_start_s']:.3f}秒。本页独立纵轴；单位为各色团训练SD。参考期0–5秒；计分5.0–29.8秒。",size=8.9,gap=8)
        d.figure(wavepaths['atlas:'+row['id']],window_caption(e,row))
        d.text('上行：仅EEG预测，整段双Hb隐藏，水平零线为无EEG预测。中行：EEG＋Hb联合与Hb-only参照。下行：相应目标减预测残差。×符号标记未收敛任务，仅供诊断；方法线型保持不变，没有因失败删除该目标。',size=8.7)
        d.text(waveform_reading(e,row),size=8.6)
    return d.save()


def verify_pdfs(out,documents):
    checks=[]
    preview=Path(out)/'previews';preview.mkdir()
    for doc in documents:
        pdf=fitz.open(Path(out)/doc['pdf']);page_checks=[]
        if len(pdf)!=doc['pages']:raise ValueError('PDF page count drift')
        for i,p in enumerate(pdf):
            text=p.get_text();images=p.get_images(full=True)
            if len(text)<50:raise ValueError('missing selectable report text')
            if any(not (0<=b['rect'][0]<b['rect'][2]<=596 and 0<=b['rect'][1]<b['rect'][3]<=780) for b in doc['text_boxes'] if b['page']==i+1):
                raise ValueError('body text clipped')
            expected=sum(x['page']==i+1 for x in doc['figures'])
            if len(images)!=expected:raise ValueError('figure is not an embedded bitmap')
            p.get_pixmap(dpi=80,alpha=False).save(preview/f'{Path(doc["pdf"]).stem}_{i+1:02d}.png')
            page_checks.append(dict(page=i+1,searchable_chars=len(text),image_objects=len(images)))
        if any(not 200<=f['dpi']<=300.5 for f in doc['figures']):raise ValueError('figure display DPI outside 200–300')
        checks.append(dict(pdf=doc['pdf'],pages=page_checks,figure_objects=len(doc['figures']),all_figures_bitmap=True))
        pdf.close()
    return checks


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);run=args.run.resolve();out=args.output.resolve()
    if run not in out.parents or out.exists():raise ValueError('fresh versioned child of the completed run required')
    e=Evidence(run);out.mkdir();(out/'figures').mkdir()
    font_manager.fontManager.addfont(str(FONT));plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(FONT)).get_name(),'font.size':9,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    examples=select_examples([r for r in e.plan['rows'] if r['split']=='evaluation'],e.tasks,set(e.residual['common_success_window_ids']))
    write(out/'waveform_selection.json',dict(rule='PC1 prediction NMSE low/median/high rank on common34; nearest rank with distinct subject preferred; explanatory only',examples=examples,failed_separately=e.residual['excluded_window_ids'],atlas='all_12_development_and_36_evaluation_targets'))
    figs=summary_figures(e,out);figs['geometry']=geometry_figure(e,out)
    figs['proxy'],figs['proxy_ratios']=proxy_figure(e,out,examples)
    figs['fir_waveforms']=fir_waveform_figure(e,out,examples)
    limits=representative_limits(e,[x['row'] for x in examples]);paths={}
    for x in examples:paths[x['row']['id']]=waveform(e,out,x['row'],'example_'+x['row']['subject']+'_'+str(x['row']['event_id']),limits)
    for i,key in enumerate(e.residual['excluded_window_ids']):paths[key]=waveform(e,out,e.rows[key],f'failure_{i+1}')
    for i,row in enumerate(r for r in e.plan['rows'] if r['split']!='train'):
        paths['atlas:'+row['id']]=waveform(e,out,row,f'atlas_{i+1:02d}')
    report=make_report(e,out,examples,figs,paths);atlas=make_atlas(e,out,paths)
    validation=verify_pdfs(out,[report,atlas])
    shutil.copy2(Path(__file__),out/'renderer_source.py')
    write(out/'figure_manifest.json',dict(report=report,atlas=atlas))
    write(out/'delivery_checks.json',dict(status='passed',source_run=str(run),task_arrays_verified=420,summary_groups_verified=len(e.summary['groups']),native_reads=0,new_fits=0,new_bootstrap_intervals=0,documents=validation,skill_review=dict(name='academic-humanizer',numbers_and_equations_preserved=True,claims_calibrated=True),visual_review='pending_contact_sheet_and_full_size_inspection'))
    print(json.dumps(dict(output=str(out),report_pages=report['pages'],atlas_pages=atlas['pages'],checks='passed_programmatic'),ensure_ascii=False))


if __name__=='__main__':main()
