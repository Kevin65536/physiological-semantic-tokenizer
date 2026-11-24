"""Render the retained census, with one matched example per detected class and dataset."""
from collections import defaultdict,Counter
from pathlib import Path
import sys,json,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path.cwd()
sys.path.insert(0,str(root/'experiments/scripts'))
import audit_tokenizer_observation_quality as audit
audit.init()
dst=audit.DST
plan=audit.WORK
defs={
 'missing_all':('整窗缺测','不可用','30 秒内有效且有限的观测点数为零。'),
 'constant':('整窗恒值','动态波形不可用','所有采样均有效，30 秒内最大值等于最小值。'),
 'line50':('50 Hz 工频占优','按已有合同排除','EEG 的 49–51 Hz 能量超过未加窗、未去均值的全窗能量的 50%。'),
 'near_flat':('近乎平直','复核','非恒值，窗内 SD 低于该库该位置冻结训练尺度的万分之一。Hb 两色共用尺度。'),
 'transient':('瞬态主导','复核','减去全窗中位数后，最强的连续 0.5 秒占全窗平方和至少 90%。'),
 'large_slow':('大幅慢波','复核','EEG 窗内 SD 至少为该位置训练尺度的 10 倍，且 1–4 Hz 占全窗能量至少 80%。'),
 'missing_part':('部分缺测','支持不完整','部分观测掩码为假或数值非有限，其余点有效。'),
 'frozen_segment':('局部冻结','片段复核','非恒值窗口中至少连续 2 秒的样本完全相等。'),
 'rail':('疑似削顶平台','复核','非恒值窗口中至少 20% 的采样精确落在同一侧极值。'),
 'few_levels':('少数离散电平','复核','非恒值窗口全程仅有 2–4 种电压或 Hb 数值。'),
 'duplicate':('跨位置波形复制','空间身份复核','同窗、同模态（Hb 同色）两个不同位置非恒值且全程逐点相同或反号相同。两端均计数。'),
 'line60':('60 Hz 窄带占优','复核','EEG 的 59–61 Hz 能量占全窗能量超过一半。'),
 'other_tone':('其他单频占优','复核','≥1 Hz 且在两条工频带外，单一 FFT 频点占 EEG 全窗能量至少 90%。'),
 'large_fast':('大幅高频','复核','EEG 窗内 SD 至少为该位置训练尺度的 10 倍，且 30–45 Hz 占全窗能量至少 80%。')}
mods=['EEG','HbO','HbR']
eegonly={'line50','line60','other_tone','large_fast','large_slow'}
short={'eeg_fnirs_single_trial':'Single-Trial','simultaneous_eeg_nirs':'Simultaneous','visual_cognitive_motivation':'Visual','refed':'REFED'}
rows=[]; den=Counter(); noscale=Counter(); posden=Counter();lookup={}
for i,w in enumerate(plan):
 m=w['metadata'];src=m['source'];ds,sub,rec,split=m['dataset'],m['subject'],src['native_record'],m['split']
 parts=json.loads((dst/'parts'/f'{i:05}.json').read_text())
 assert len(parts)==3*len(w['score_starts_s']) and {(s,l) for s,l,*_ in parts}=={(s,l) for s in w['score_starts_s'] for l in mods}
 for t,label,n,ns,hits in parts:
  names=m['EEG_channels'] if label=='EEG' else m['fNIRS_positions']
  assert len(names)==n
  den[ds,split,label]+=n;noscale[ds,split,label]+=ns
  key=(m['key'],t,label)
  lookup[key]=i
  for name in names:posden[ds,split,label,sub,name]+=1
  for j,k,v,peer in hits:
   rows.append(dict(pattern=k,label=label,dataset=ds,split=split,subject=sub,record=rec,position=names[j],
    parent=m['key'],parent_offset_s=t,window_start_EEG_clock_s=(src['start_sample']/200+t),
    value=v,peer=names[peer] if peer>=0 else '',work_index=i))
hits=pd.DataFrame(rows)
hits.to_csv(dst/'window_flags.csv',index=False)
by=['pattern','dataset','split','label','subject','position']
sites=hits.groupby(by).size().rename('hits').reset_index()
sites['planned']= [posden[r.dataset,r.split,r.label,r.subject,r.position] for r in sites.itertuples()]
sites.to_csv(dst/'subjects_positions.csv',index=False)
tab=[]
for (ds,sp,l),n in sorted(den.items()):
 for k in defs:
  if l!='EEG' and k in eegonly:continue
  sub=hits[(hits.pattern==k)&(hits.dataset==ds)&(hits.split==sp)&(hits.label==l)]
  tab.append(dict(dataset=ds,split=sp,label=l,pattern=k,planned=n,hits=len(sub),percent=100*len(sub)/n,
   subjects=sub.subject.nunique(),positions=sub.position.nunique(),no_training_scale=noscale[ds,sp,l]))
pd.DataFrame(tab).to_csv(dst/'counts.csv',index=False)
# Pair frozen train panel by physical record start rather than assuming both use the same parent.
def loc(r,t,name,l):
 return (r['dataset'],r['subject'],r['record'],round(r['native_start_sample']*200/r['native_rate_hz'])+t*200,name,l)
flags=defaultdict(set);available=set()
for w in plan:
 m=w['metadata'];s=m['source']
 if m['split']!='train':continue
 for start in w['score_starts_s']:
  available.add((m['dataset'],m['subject'],s['native_record'],s['start_sample']+start*200))
for h in rows:
 if h['split']=='train':flags[h['dataset'],h['subject'],h['record'],round(h['window_start_EEG_clock_s']*200),h['position'],h['label']].add(h['pattern'])
base=(root/audit.CFG['source_plan']).parent
manifest=json.loads((base/'cache/manifest.json').read_text())['rows']
base_run=base.parent/'20261123_joint_baseline_v1'
matched=[]
for ds in short:
 s=base_run/'component_source_transfer/dynamic_band_roles_v1'/ds/'held_scores.npz'
 with np.load(s) as z:
  ex=z['examples'];errs=z['frozen_word__nmse'][:,0,0]
  for (i,t,p),err in zip(ex,errs):
   r=manifest[i];assert r['split']=='train'
   key=loc(r,t,r['positions'][p],'EEG')
   if key[:4] in available:matched.append((ds,'EEG',flags.get(key,set()),float(err)))
 s=base_run/'multimodal_hb'/ds/'prepared.npz'
 with np.load(s) as z,np.load(base_run/'multimodal_hb/local_geometry_v1'/ds/'reference.npz') as b:
  ex=z['examples'];y=z['targets'][:,0].reshape(-1,300,2);mask=z['mask'].reshape(-1,300,2)
  delta=b['sum'].reshape(y.shape)-y; variance=z['variances'][:,0];denom=mask.sum(1)
  nmse=np.divide(np.where(mask,delta**2,0).sum(1),denom*variance,out=np.full(variance.shape,np.nan),where=denom>0)
  for ii,(i,t,p) in enumerate(ex):
   r=manifest[i];assert r['split']=='train'
   for c,l in enumerate(mods[1:]):
    key=loc(r,t,r['positions'][p],l)
    if key[:4] in available:matched.append((ds,l,flags.get(key,set()),nmse[ii,c]))
impact=[]
for ds in short:
 for l in mods:
  group=[x for x in matched if x[:2]==(ds,l)]
  for k in ['no_flag']+list(defs):
   vals=np.array([v for _,_,f,v in group if (not f if k=='no_flag' else k in f)])
   if len(vals):impact.append(dict(dataset=ds,label=l,pattern=k,n=len(vals),finite=int(np.isfinite(vals).sum()),median_X_NMSE=float(np.nanmedian(vals)),mean_X_NMSE=float(np.nanmean(vals))))
pd.DataFrame(impact).to_csv(dst/'frozen_tokenizer_overlap.csv',index=False)
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'path.simplify':False})
figs={}
examples=[]
for k,group in hits.groupby('pattern'):
 subsets=list(group.groupby('dataset',sort=True))
 fig,axes=plt.subplots(len(subsets),2,figsize=(12,2.9*len(subsets)+.8),squeeze=False,layout='constrained')
 for ri,(ds,subrows) in enumerate(subsets):
  # Pick a real observation nearest the median diagnostic value, then deterministic identity.
  r=subrows.assign(dist=(subrows.value-subrows.value.median()).abs()).sort_values(['dist','subject','record','window_start_EEG_clock_s','position','label']).iloc[0]
  w=plan[int(r.work_index)];m=w['metadata'];t0=int(r.parent_offset_s);is_e=r.label=='EEG'
  fs=200 if is_e else 10;names=m['EEG_channels'] if is_e else m['fNIRS_positions'];j=names.index(r.position)
  f='eeg' if is_e else 'fnirs';sl=slice(t0*fs,(t0+30)*fs)
  with np.load(m['array_file']) as z:
   x=z[f+'_x'][sl];g=z[f+'_observation_valid'][sl]
  if not is_e:
   c=mods.index(r.label)-1;x=x[:,:,c];g=g[:,:,c]
  y=np.where(g[:,j]&np.isfinite(x[:,j]),x[:,j],np.nan);t=np.arange(len(y))/fs
  ax,bx=axes[ri];unit=m['EEG_unit'].replace('uV','µV') if is_e else '被试归一化 Hb'
  if unit.startswith('unknown'):unit='发布相对电压'
  ax.plot(t,y,color='#205e75',lw=.75);ax.set(xlim=(0,30),ylabel=unit,xlabel='窗内时间（秒）')
  ax.set_title(f'{short[ds]}  ·  {r.subject} / {r.record} / {r.position} {r.label}\nEEG 时钟 {r.window_start_EEG_clock_s:.3f} s 起，{r.split}',fontsize=10)
  if k=='missing_all':
   ax.text(.5,.5,'整窗无有效观测',ha='center',transform=ax.transAxes)
   bx.step(t,g[:,j].astype(int),where='post');bx.set(ylim=(-.15,1.15),yticks=[0,1],yticklabels=['无效','有效'],title='原缓存观测掩码')
  elif k in ('constant','near_flat'):
   good=np.flatnonzero(g.all(0)&(np.std(x,axis=0)>0));good=good[good!=j]
   peer=int(good[0]);bx.plot(t,x[:,peer],color='#96682e',lw=1,label=names[peer]);bx.plot(t,y,color='#205e75',ls='--',label=r.position)
   bx.set(title='同窗同色团对照（共用电压或 Hb 刻度）',ylabel=unit);bx.legend(fontsize=8)
   if k=='near_flat':ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2),useOffset=False)
  else:
   if k=='line50':a=round(fs*14.9);n=41
   else:
    peak=np.argmax(abs(y-np.median(y)));n=2*fs;a=int(np.clip(peak-n//2,0,len(y)-n))
   bx.plot(t[a:a+n],y[a:a+n],lw=1,color='#205e75',marker='.' if n<50 else None);bx.set(title='局部波形（同一测量单位，纵轴独立）',ylabel=unit)
  bx.set_xlabel('窗内时间（秒）')
  for a in (ax,bx):a.grid(axis='y',alpha=.22)
  examples.append(dict(pattern=k,**{v:r[v] for v in ['dataset','split','subject','record','position','label','parent','parent_offset_s','window_start_EEG_clock_s','value']}))
 fig.suptitle(defs[k][0]+'  |  '+defs[k][1],fontsize=15)
 name=k+'.png';fig.savefig(dst/name,dpi=205);plt.close(fig);figs[k]=name
pd.DataFrame(examples).to_csv(dst/'figure_examples.csv',index=False)
esc=html.escape
body=['<meta charset=utf-8><style>body{font:16px/1.55 sans-serif;max-width:1050px;margin:3em auto}td,th{padding:5px 9px;border-bottom:1px solid #ddd}table{border-collapse:collapse}img{width:100%}a{color:#13677b}</style><title>V3 输入质量普查</title><h1>V3 tokenizer 输入质量普查</h1>',
'<p>四库 80 名公开训练／普通验证被试。按 G0 预先确定的 7,806 个非重叠 30 秒评分窗，读取 V3 观测及其观测掩码，保留原位置和原单位。每个 EEG 电极、Hb 位置的每个色团分别计数。分母包含缺测。短尾、专用回归窗、缓存在建时已排除的位置与未纳入此网格的记录不在分母内。多标签可重叠。</p>',
'<p>判定范围为缓存提供的处理后输入。尖峰、异常慢波和极低幅值的病因需要其它证据。滤波后的数据不能用于完整估计采集端削顶率。训练尺度沿用历史 G0 数据集／位置尺度，缺少尺度的轨迹只跳过依赖尺度的判据。频谱使用未加窗全窗单边功率，包含 DC 与 Nyquist。示例选每个检出库内最接近该类诊断量中位数的实际波形。</p>',
'<p><a href="counts.csv">分库分区频次与分母</a> · <a href="subjects_positions.csv">全部命中被试与位置（含分母）</a> · <a href="window_flags.csv">逐窗时间和原父窗</a> · <a href="frozen_tokenizer_overlap.csv">历史模型误差</a> · <a href="figure_examples.csv">图例溯源</a></p>',
'<table><tr><th>模式</th><th>EEG</th><th>HbO</th><th>HbR</th><th>处理与判据</th></tr>']
for k,(n,status,rule) in defs.items():
 counts=[str(len(hits[(hits.pattern==k)&(hits.label==l)])) if l=='EEG' or k not in eegonly else '—' for l in mods]
 body.append('<tr><td>'+n+'</td>'+''.join('<td>'+a+'</td>' for a in counts)+'<td>'+status+'。'+rule+'</td></tr>')
body+=['</table><p>EEG 分母 268,976；HbO、HbR 各 288,226。确定无动态信息与可疑形态分开解释。全窗缺测只能画掩码。未检出类别没有实测阳性图。</p>']
for k in defs:
 if k in figs:body.append('<h2>'+defs[k][0]+'</h2><img src="'+figs[k]+'">')
body+=['<h2>与历史 tokenizer 的交集</h2><p>按物理记录、电极或 Hb 位置及 200 Hz 起点匹配既有训练评分面板。EEG 为 K15351 训练人交叉验证，Hb 为冻结 Hb111 训练面板回放。表中逐位置窗口计算 X 的 MSE / 其冻结训练方差，再取窗口中位数，均值另见 CSV。无命中不表示编码成功。无标记组是同库同模态参照，组间暴露不匹配，不能推断因果。</p>',pd.DataFrame([a for a in impact if a['pattern']!='no_flag']).to_html(index=False)]
(dst/'index.html').write_text(''.join(body))
summary=dict(parents=len(plan),windows=sum(len(w['score_starts_s']) for w in plan),denominators={l:sum(v for (_,_,m),v in den.items() if m==l) for l in mods},unique_flagged_by_modality=hits.groupby('label').apply(lambda g:len(g.drop_duplicates(['dataset','subject','record','window_start_EEG_clock_s','position'])),include_groups=False).to_dict(),no_training_scale=sum(noscale.values()),matched_by_modality=dict(Counter(item[1] for item in matched)))
(dst/'verification.json').write_text(json.dumps(summary,indent=2))
print(summary)
