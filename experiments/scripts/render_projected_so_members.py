"""Chinese reading copy of the frozen S/O member and paired-partner audit."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.scripts.audit_projected_so_members import FREQ, OUT, SOURCE
from experiments.scripts.render_temporal_projection_so import DS, p, table

TIME = np.arange(400) / 200
BAND = (FREQ >= 1) & (FREQ <= 45)
NAMES = {'shape':'有符号波形', 'alpha':'α 功率占比', 'slow':'慢频功率占比'}
URL = 'https://proceedings.iclr.cc/paper_files/paper/2026/file/86d1ab582afb247ccaa84bec4a7e24f7-Paper-Conference.pdf'
CSS = """body{font-family:sans-serif;font-size:12pt;line-height:1.42;color:#203843}
h1{font-size:20pt;color:#134c63} h2{font-size:14pt} p{margin:4pt 0 9pt}
table{border-collapse:collapse;width:100%;font-size:11pt}th,td{padding:4pt;border-bottom:.5pt solid #c4d0d4}
th{background:#e4eff1;text-align:left}"""
RAW, TARGET, CONTROL = '#243f52', '#bf6428', '#298070'


def normalized_spectrum(x):
    centered = x - x.mean(axis=-1,keepdims=True)
    power = abs(np.fft.rfft(centered*np.hanning(400)))**2
    power = power[...,BAND]
    return 100*power/power.sum(axis=-1,keepdims=True)


def norm_wave(x):
    centered = x-x.mean(axis=-1,keepdims=True)
    return centered/np.sqrt(np.mean(centered**2,axis=-1,keepdims=True))


def member_fig(m, matches, x, s, templates, branch, name, token, folder, manifest):
    selected = matches[(matches.branch==branch)&(matches.endpoint==name)].copy()
    selected['person'] = m.loc[selected.member_row,'person'].values
    examples = selected.drop_duplicates('person').copy()
    examples['dataset'] = m.loc[examples.member_row,'dataset'].values
    examples = pd.concat([examples.drop_duplicates('dataset'),examples]).drop_duplicates('person').head(3)
    fig, axs = plt.subplots(len(examples)+1,2,figsize=(9,1.82*(len(examples)+1)),layout='constrained')
    wave_axes, freq_axes = [], []
    for (left,right), r in zip(axs,examples.itertuples()):
        j,k = m.loc[r.member_row],m.loc[r.control_row]
        y,v = x[j.example,j.block],x[k.example,k.block]
        target = s[j.example,j.block]
        if branch=='O':
            target=y-target
        left.plot(TIME,y,color=RAW,lw=.9,label='Assigned X')
        left.plot(TIME,target,color=TARGET,lw=1.15,ls='--',label=branch+' target')
        left.set_title(f'{DS[j.dataset]} {j.subject} · {j.site} · {j.start+2*j.block}–{j.start+2*j.block+2} s',fontsize=10)
        left.set(ylabel='Scaled voltage',xlabel='Time (s)')
        right.plot(FREQ[BAND],normalized_spectrum(y),color=RAW,lw=1,label='Assigned X')
        right.plot(FREQ[BAND],normalized_spectrum(v),color=CONTROL,lw=1,ls=':',label='Matched other ID')
        right.set(xlabel='Frequency (Hz)',ylabel='% power / bin')
        right.set_title(f'Member vs. control · {k.start+2*k.block}–{k.start+2*k.block+2} s control',fontsize=9)
        wave_axes.append(left);freq_axes.append(right)
        manifest.append(dict(branch=branch,endpoint=name,token=token,member_row=r.member_row,control_row=r.control_row))
    for group in [wave_axes,freq_axes]:
        bottom=min(ax.get_ylim()[0] for ax in group); top=max(ax.get_ylim()[1] for ax in group)
        for ax in group:
            ax.set_ylim(bottom,top)
    for ax in axs[0]: ax.legend(fontsize=7,loc='upper right')
    left,right=axs[-1]
    for key,co,ls,lab in [('member_row',RAW,'-','Assigned X'),('control_row',CONTROL,':','Matched X')]:
        waves,spec=[],[]
        for _,g in selected.groupby('person'):
            ids=m.loc[g[key]]; w=x[ids.example,ids.block]
            waves.append(norm_wave(w).mean(0));spec.append(normalized_spectrum(w).mean(0))
        left.plot(TIME,np.mean(waves,axis=0),color=co,ls=ls,lw=1.45,label=lab)
        right.plot(FREQ[BAND],np.mean(spec,axis=0),color=co,ls=ls,lw=1.45,label=lab)
    if name=='shape':
        left.plot(TIME,templates[branch+'_'+name]*20,color=TARGET,ls='--',lw=.9,label='Train template')
    left.set(xlabel='Time (s)',ylabel='Within-patch SD',title='All matched patches · person-balanced waveform')
    right.set(xlabel='Frequency (Hz)',ylabel='% power / bin',title='All matched patches · person-balanced spectrum')
    for ax in axs[-1]: ax.legend(fontsize=7)
    for ax in axs[:,1]:
        if name!='shape':
            ax.axvspan(*( (8,13) if name=='alpha' else (1,4)),color='#e6b84d',alpha=.14)
    for ax in axs.ravel(): ax.grid(axis='y',alpha=.15)
    path=folder/f'{branch}_{token}_{name}.png'
    fig.savefig(path,dpi=240);plt.close(fig)
    return path


def pair_fig(m, matches, pairs, x, folder, manifest):
    fig, axs=plt.subplots(2,2,figsize=(9,4.7),layout='constrained')
    for (left,right),row in zip(axs,pairs.itertuples()):
        r=matches[(matches.branch==row.branch)&(matches.endpoint=='pair_alpha')].iloc[0]
        other='O' if row.branch=='S' else 'S'
        for k,co,ls in [(r.member_row,RAW,'-'),(r.control_row,CONTROL,':')]:
            j=m.loc[k];w=x[j.example,j.block]
            label=f'{other}{j[other]} · {j.start+2*j.block} s'
            left.plot(TIME,w,color=co,ls=ls,lw=1,label=label)
            right.plot(FREQ[BAND],normalized_spectrum(w),color=co,ls=ls,lw=1.3,label=label)
        left.set_title(f'Fixed {row.branch}{row.token} · {DS[j.dataset]} {j.subject} · {j.site}',fontsize=10)
        left.set(xlabel='Time (s)',ylabel='Scaled voltage')
        right.set(xlabel='Frequency (Hz)',ylabel='% power / bin')
        right.axvspan(8,13,color='#e6b84d',alpha=.16)
        manifest.append(dict(branch=row.branch,endpoint='pair_alpha',token=row.token,member_row=r.member_row,control_row=r.control_row))
        for ax in [left,right]:ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15)
    path=folder/'combinations.png'
    fig.savefig(path,dpi=240);plt.close(fig)
    return path


def render(export):
    export.mkdir(parents=True,exist_ok=False)
    folder=export/'figures';folder.mkdir()
    m=pd.read_csv(OUT/'validation_members.csv.gz',index_col='row',dtype={'subject':str})
    matches=pd.read_csv(OUT/'matches.csv')
    sel=pd.read_csv(OUT/'selection.csv').set_index(['branch','endpoint'])
    val=pd.read_csv(OUT/'contrasts.csv').set_index(['branch','endpoint'])
    pairs=pd.read_csv(OUT/'pair_selection.csv')
    spec=pd.read_csv(OUT/'alpha_specificity.csv')
    x,s=(np.load(SOURCE/(k+'.npy'),mmap_mode='r').reshape(-1,15,400) for k in ['x','projected_v3'])
    doc=fitz.open(); manifest=[]; rates=[]
    def page(title,body,figure=None,tail=''):
        pg=doc.new_page(width=720,height=875)
        def box(top,bottom,html):
            rect=fitz.Rect(29,top,691,bottom)
            spare,_=pg.insert_htmlbox(rect,'<div lang="zh-Hans">'+html+'</div>',css=CSS,scale_low=1)
            if spare<0:raise ValueError((title,top,bottom))
            return bottom-spare
        y=box(12,838,'<small>实测 EEG 词元检索　 /　2026 年 11 月 25 日</small><h1>'+title+'</h1>'+body)
        if figure:
            pix=fitz.Pixmap(str(figure));h=646*pix.height/pix.width
            pg.insert_image(fitz.Rect(37,y+4,683,y+4+h),filename=str(figure));y+=h+9
        if tail:box(y,838,tail)
        box(844,872,f'<small>EEG-SO-PROJECTED-FROZEN-MEMBERS-v1　·　公开开发面板　·　第 {pg.number+1} 页</small>')
    page('从 CodeBrain Figure 4 到新 S/O 的真实成员',
         p('S 2109 更常命中实测 α 占比较高的片段。O 7822 命中的实测电压有重复峰谷。两者可以从编号追溯到具体波形；当前独立标记对事件归属的支持有限。')
         +'<h2>论文如何读图</h2>'+p('在 ISRUC-S3 睡眠分期数据上，作者按实际输出的编号找回 EEG。图 4a 叠画时域码 3323 的 N3 波形；4b 展开频域码 2685 的 N3 功率谱，共同的低频峰比单看时序更明显。4c 固定时域码 668，展示来自 N2、REM、N1 的不同谱；正式版将 REM 与 N1 两例的频域码都标为 2356。4d 固定频域码 1131，分别配时域码 1458、2296、3696，对应图示 N2、N3、N1。阶段标签随样本提供，同一组合可跨期。')
         +p('附录 B 按测试集的分期激活频率选例，再报告词内命中比例。作者报告时域码 1537 有约 40.4% 的成员含慢波，频域码 2298 有约 15.4% 的成员呈 σ 峰。图 4 展示形态，附录补充发生率；这些数值均是各自词内比例。')
         +p(f'<a href="{URL}#page=10">CodeBrain，ICLR 2026 正式版，第 10 页 Figure 4；附录 B 第 19–21 页。</a>')
         +'<h2>本次如何检索</h2>'+p('冻结第 500 步、种子 17 的模型。输入为一个电极的 30 秒实测 X，每 2 秒各发出一个 S、O 整数。新目标 S=P(S0)，把既有教师 S0 投影到训练选定的 32 对傅里叶基；O=X−S。两分支都输出带符号电压。以下编号只属于此检查点。')
         +p('在四个清醒任务数据集的 60 名训练被试、67,710 个有效电极片段中，每分支分别选择波形一致性最强、平均 α 占比最高和平均慢频占比最高的词。入围至少 80 段、8 人、2 库。波形先去均值并除以二范数，同词平均向量的长度用于排名，再固定其方向为模板。')
         +p('开发面板为此前使用过的另 8 名被试，每人一个 90 秒记录窗，共 12,414 个非重叠时间块×电极观测。同被试、同电极内找不同编号且实测 X 的 RMS 最近的片段，幅度比最大 1.4，允许重复用对照。差值先在人内平均，再等权平均有配对的人。仅作描述，不以电极片段充当独立样本。'))
    descriptions={
      ('S','shape'):'训练模板约在 0.33 秒出现最高点，约在 0.62 秒出现最低点。开发面板只找到 3 次，覆盖 2 人。',
      ('O','shape'):'训练模板约在 1.17 秒出现最低点，约在 1.53 秒达到最高点。3 人均比对照更接近模板，32 个成员中 24 个来自 REFED 的同一人。',
      ('S','alpha'):'8 人中 7 人的配对差为正。训练拟合的 P(X) 频谱与幅度背景可解释相当一部分差异，见末页。',
      ('O','alpha'):'训练中 α 占比最高的合格 O 词，开发面板中只有 3/7 人的差值为正。',
      ('S','slow'):'词内慢频占比很高，对照同样很高。额前独立眼电对照见末页。',
      ('O','slow'):'与波形检索选中同一编号。慢频占比的差异远小于这一编号的有符号波形相关差异。'}
    with np.load(OUT/'templates.npz') as templates:
        for branch,name in [('S','alpha'),('O','shape'),('S','shape'),('O','alpha'),('S','slow'),('O','slow')]:
            r=val.loc[branch,name]; t=sel.loc[branch,name]; token=int(r.token)
            fig=member_fig(m,matches,x,s,templates,branch,name,token,folder,manifest)
            if name=='shape':
                score=f'不平移、不翻转符号，实测 X 与固定训练模板的相关均值为 {r.member:.3f}，对照为 {r.control:.3f}。'
            else:
                score=f'实测 X 中该频带功率占比为 {r.member:.1%}，对照为 {r.control:.1%}，相差 {100*r.difference:+.2f} 个百分点。'
            group=matches[(matches.branch==branch)&(matches.endpoint==name)]
            hits=m.loc[group.member_row]
            epochs=len(hits.drop_duplicates(['parent','start','block']))
            detail=''
            if name in ('shape','alpha'):
                counts=[]
                for key in ('member_row','control_row'):
                    z=m.loc[group[key]]
                    w=x[z.example,z.block]
                    if name=='shape':
                        marked=(norm_wave(w) @ templates[branch+'_shape']/20) >= .5
                    else:
                        peak=FREQ[BAND][normalized_spectrum(w).argmax(1)]
                        marked=(peak>=8)&(peak<13)
                    counts.append(int(marked.sum()))
                    rates.append(dict(branch=branch,endpoint=name,token=token,group=key,
                                      positive=int(marked.sum()),n=len(z),unique_member_epochs=epochs,
                                      person_balanced=pd.Series(marked).groupby(z.person.values).mean().mean()))
                marker='波形相关至少 0.5' if name=='shape' else '1–45 Hz 内最高谱峰落在 α 频带'
                detail=f'逐段计数，{marker}的成员为 {counts[0]}/{len(group)}，配对对照为 {counts[1]}/{len(group)}。'
            page(f'{branch} {token}　·　{NAMES[name]}',
                 p(f'训练有 {int(t.train_n)} 段、{int(t.train_people)} 人；开发有 {int(r.hits)} 段命中，匹配 {int(r.matched)} 段，来自 {int(r.people)} 人的 {epochs} 个不重复时段。'+score+detail),fig,
                 p('上方各行取不同被试的首个可配对成员，优先覆盖不同数据集。左图实线为命中时的实测 X，虚线为同段分支目标；纵轴为训练尺度电压。右图实线与点线是成员和对照 X 的相对功率谱。横轴时间相对片段起点，标题秒数相对 90 秒窗起点。')
                 +p('底行使用全部匹配对，按被试等权。波形逐段除以本段标准差，频谱用 Hann 窗、0.5 Hz 频点，各点占 1–45 Hz 总功率的百分比。'+descriptions[branch,name]))
    pair_rows=[]
    for q in pairs.itertuples():
        v=val.loc[q.branch,'pair_alpha']
        pair_rows.append([q.branch+str(q.token),f'{q.first} / {q.second}',f'{q.validation_first} / {q.validation_second}',
                          f'{int(v.matched)} / {int(v.people)}',f'{100*v.difference:+.2f}'])
    fig=pair_fig(m,matches,pairs,x,folder,manifest)
    page('固定一个编号后，搭配另一编号的真实片段',
         p('每个方向在训练中选择出现最多且至少有两个合格搭档的编号。每个搭档至少覆盖 3 名训练被试，取其中出现最多的两个。验证时只比较固定编号、同人同电极且幅度相近的两种搭配。')
         +table(['固定编号','搭档 A / B','开发命中 A / B','匹配对 / 人','α 差 A−B（百分点）'],pair_rows),fig,
         p('每行画第一对合格匹配的实测 X 及频谱。图例标出另一分支编号和该片段相对 90 秒窗的起点。不同排的纵轴独立。表中使用全部匹配对。两种搭配确实复现；两方向的差异在被试间均有正有负。该检索没有评价任务状态的分类能力。'))
    eog=val.xs('eye',level='endpoint')
    sr=[]
    for b,g in spec.groupby('branch'):
        sr.append([b+str(g.token.iloc[0]),len(g),f'{100*g.raw.mean():+.2f}',f'{100*g.residual.mean():+.2f}',int((g.residual>0).sum())])
    page('哪些结果支持生理解释',
         '<h2>α 富集能否超出已知的投影频谱？</h2>'
         +p('附加分析保留两枚 α 候选及原有匹配，在各数据集和电极的训练片段拟合线性背景。输入为截距、直接投影 P(X) 的 α 占比、投影与 X 的均方比、前两项的乘积，以及 X 的对数 RMS。验证时比较实测 α 占比减去背景预测后的词内与对照差。')
         +table(['编号','人数','原始差（百分点）','残差差（百分点）','残差差为正的人数'],sr)
         +p('S 2109 的原始富集跨人较一致，扣除这项背景后方向分化。该结果支持把编号用作检索线索，对独立神经事件归属仍缺少证据。')
         +'<h2>慢波形能否由独立眼电标记确认？</h2>'
         +p('4 人有同步实测水平、垂直眼电。沿用原读取器的 1–45 Hz 双通道正交白化基，按各人可用且去重的 2 秒时段计算联合均方，最高四分位标记为眼电活动高。只在 FP/AF 开头的额区电极上，检查预先选出的慢频词。')
         +table(['候选','额区匹配对','配对人数 / 有眼电人数','成员高眼电率','对照高眼电率'],
                [[b+str(int(r.token)),int(r.matched),f'{int(r.people)} / 4',f'{r.member:.1%}',f'{r.control:.1%}'] for b,r in eog.iterrows()])
         +p('眼电率先在被试内计算，再在有配对的人中等权平均。O 词仅一对，S 词两人的配对差均为零。这一面板不足以用眼电给这些词定名。')
         +'<h2>证据边界</h2>'+p('四个数据集来自清醒任务，当前没有睡眠分期、慢波或纺锤波逐段注释。固定时间窗中的峰谷可重复，带内功率可重复；对睡眠事件、眼动、认知状态或具体脑源的归属，需要对应的独立标签。一次训练仅确定本检查点的编号。'))
    pd.DataFrame(manifest).to_csv(export/'displayed_examples.csv',index=False)
    pd.DataFrame(rates).to_csv(export/'member_marker_rates.csv',index=False)
    doc.subset_fonts()
    doc.save(export/'S_O_token_真实波形与频谱.pdf',garbage=4,deflate=True)
    doc.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    render(parser.parse_args().output)
