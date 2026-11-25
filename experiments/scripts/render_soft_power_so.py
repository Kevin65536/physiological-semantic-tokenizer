"""Chinese bitmap-figure report and per-subject voltage atlas from the paired pilot."""
import argparse
from html import escape
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import fitz
from fontTools import subset
from fontTools.ttLib import TTFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.scripts.evaluate_soft_power_so import OUT, CFG, read

ARMS = CFG['arms']
NAMES = ['旧 v3', '软 SSM', '固定参考']
COLORS = ['#30658b','#c35124','#16806a']
LINES = ['--','-',':']
DS = dict(eeg_fnirs_single_trial='Single-Trial', simultaneous_eeg_nirs='Simultaneous',
          refed='REFED',visual_cognitive_motivation='Visual')
plt.rcParams.update({'font.size':10,'axes.spines.right':False,'axes.spines.top':False,
                     'axes.edgecolor':'#9babad','axes.labelcolor':'#284149'})


def table(headers, rows):
    return '<table><tr>'+''.join(f'<th>{x}</th>' for x in headers)+'</tr>'+''.join(
        '<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in rows)+'</table>'


def p(text):
    return '<p>'+text+'</p>'


def save(fig, folder, name):
    path = folder/(name+'.png')
    fig.savefig(path,dpi=240,facecolor='white')
    plt.close(fig)
    return path


def overview(out, cases, change, base):
    fig,axes=plt.subplots(1,3,figsize=(9,3.6),layout='constrained')
    labels=[DS[c['dataset']]+' '+c['subject'].replace('subject_','') for c in cases]
    for ax,(df,metric,title) in zip(axes,[(change,'leak','Added EOG into S'),(change,'s_relative','Change / baseline S'),
                                           (base,'eog_relative','Natural EOG in S / X')]):
        for n,c in enumerate(cases):
            z=df[df.i==n].set_index('arm').reindex(ARMS)[metric]
            if z.isna().all(): continue
            ax.plot([z.min(),z.max()],[n,n],color='#a8b4b7',lw=1)
            for a,(val,co,mk) in enumerate(zip(z,COLORS,['o','s','^'])):
                ax.scatter(val,n,marker=mk,color=co,s=26,label=NAMES[a] if n==0 else None)
        ax.set_title(title,fontsize=11)
        ax.set_yticks(range(8),labels if ax==axes[0] else [])
        ax.set_ylim(7.6,-.6); ax.set_xlim(left=0);ax.set_xlabel('RMS ratio' if metric!='eog_relative' else 'Projection norm ratio')
        ax.grid(axis='x',alpha=.2)
    handles,_=axes[0].get_legend_handles_labels()
    fig.legend(handles,['v3','Soft SSM','Fixed reference'],loc='outside upper center',ncol=3,frameon=False)
    return save(fig,out,'ocular_subjects')


def band_figure(out, bands):
    keys=['1-4','4-8','8-13','13-30','30-45']
    fig,axes=plt.subplots(1,3,figsize=(9,3),layout='constrained',sharey=True)
    for ax,field,title in zip(axes,['s','o','cross'],['S power / X power','O power / X power','Cross term / X power']):
        for arm,name,color,ls,mk in zip(ARMS,['v3','Soft SSM','Fixed reference'],COLORS,LINES,['o','s','^']):
            z=bands[bands.arm==arm].groupby('band')[field].median().reindex(keys)
            ax.plot(keys,z,ls,marker=mk,label=name,color=color,ms=4)
        ax.set_title(title,fontsize=11); ax.set_xlabel('Band (Hz)');ax.grid(axis='y',alpha=.24)
    axes[0].set_ylabel('Subject median');axes[0].set_ylim(0,1)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=3,frameon=False)
    return save(fig,out,'band_allocation')


def example(out,c):
    folder=OUT/'targets'/str(c['i'])
    j=next(k for k,i in enumerate(c['valid']) if c['channels'][i].upper().startswith(('FP','AFP')))
    channel=c['channels'][c['valid'][j]]
    with np.load(folder/'baseline.npz') as f, np.load(folder/'ocular1.npz') as v:
        x=f['x'][:,j]; s=[f[a][:,j] for a in ARMS]; delta=v['delta'][:,j]
        ds=[v[a][:,j]-s0 for a,s0 in zip(ARMS,s)]
    t=np.arange(len(x))/200
    fig,axes=plt.subplots(3,2,figsize=(9,6.1),layout='constrained')
    unit='µV' if c['unit']=='uV' else 'export units'
    specs=[(axes[0,0],(30,60),x,s,'Baseline S'),(axes[0,1],(39,41),x,s,'Baseline S'),
           (axes[1,0],(30,60),x,[x-v for v in s],'Baseline O'),
           (axes[1,1],(39,41),x,[x-v for v in s],'Baseline O'),
           (axes[2,0],(30,50),delta,ds,'Added voltage and change in S'),
           (axes[2,1],(30,50),delta,[delta-v for v in ds],'Added voltage and change in O')]
    for ax,(lo,hi),obs,vals,title in specs:
        cut=(t>=lo)&(t<hi)
        ax.plot(t[cut],obs[cut],color='#797e84',lw=.8,alpha=.85,label='Observed X / injected voltage')
        for a,y,co,ls,lw in zip(['v3','Soft SSM','Fixed reference'],vals,COLORS,LINES,[1.1,1.05,1.35]):
            ax.plot(t[cut],y[cut],ls,color=co,lw=lw,label=a)
        ax.set_title(title+f'  |  {lo}–{hi} s',fontsize=9.5)
        ax.set_xlim(lo,hi);ax.set_ylabel(unit);ax.grid(axis='y',alpha=.18);ax.axhline(0,lw=.45,color='#999999')
    for group in (axes[:2,:].ravel(),axes[2,:]):
        low=min(ax.get_ylim()[0] for ax in group);high=max(ax.get_ylim()[1] for ax in group)
        for ax in group:ax.set_ylim(low,high)
    for ax in axes[-1]:ax.set_xlabel('Time from window start (s)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=4,frameon=False,fontsize=8.8)
    zoom=(t>=39)&(t<41);active=(t>=30)&(t<50)
    rms=lambda v:float(np.mean(v*v)**.5)
    values=[[NAMES[k],f'{rms(y[zoom]):.3g}',f'{rms((x-y)[zoom]):.3g}',
             f'{rms(ds[k][active])/rms(delta[active]):.3f}'] for k,y in enumerate(s)]
    return save(fig,out,f"wave_{c['i']}_{c['subject']}"),channel,values


def render(out):
    out.mkdir(parents=True,exist_ok=False)
    figures=out/'figures';figures.mkdir()
    cases=read(OUT/'plan.json')['cases']; summary=read(OUT/'summary.json')
    base=pd.read_csv(OUT/'baseline.csv'); changes=pd.read_csv(OUT/'interventions.csv')
    eye=changes[changes.variant=='ocular1']; bands=pd.read_csv(OUT/'bands.csv')
    hb=pd.read_csv(OUT/'hb_subjects.csv'); paired=pd.read_csv(OUT/'paired_contrasts.csv')
    sensitivity=pd.read_csv(OUT/'ocular_converged_channels.csv')
    doc=fitz.open()
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        for font in ['dejavu/DejaVuSans.ttf','droid/DroidSansFallbackFull.ttf']:
            face=TTFont(Path('/usr/share/fonts/truetype')/font)
            opts=subset.Options();opts.layout_features=[];opts.drop_tables+=['GSUB','GPOS','FFTM']
            cut=subset.Subsetter(options=opts)
            cut.populate(text=Path(__file__).read_text()+''.join(chr(i) for i in range(32,127)))
            cut.subset(face);face.save(tmp/Path(font).name)
        archive=fitz.Archive(str(tmp))
        css="""@font-face{font-family:L;src:url(DejaVuSans.ttf)} @font-face{font-family:C;src:url(DroidSansFallbackFull.ttf)}
        *{font-family:L,C}body{font-size:11pt;line-height:1.45;color:#253d49;margin:0}
        h1{font-size:21pt;font-weight:normal;color:#16485b;margin:0 0 8pt}p{margin:0 0 6pt}
        table{border-collapse:collapse;width:100%;font-size:10pt;margin:4pt 0 7pt}th,td{text-align:left;padding:3pt 4pt;border-bottom:.5pt solid #c5d3d6}
        th{background:#e5efef;color:#17485b}"""
        def box(pg,rect,html):
            spare,_=pg.insert_htmlbox(fitz.Rect(rect),html,css=css,archive=archive,scale_low=1)
            if spare<0:raise ValueError((pg.number,html[:80],rect))
            return rect[3]-spare
        def page(title,body,source,fig=None,after=''):
            pg=doc.new_page(width=684,height=820)
            box(pg,(30,15,653,35),'<span style="color:#527677;font-size:9pt">EEG 分量目标先导　 /　2026 年 11 月 25 日</span>')
            y=box(pg,(30,42,654,774),'<h1>'+title+'</h1>'+body)
            if fig:
                pix=fitz.Pixmap(str(fig));height=624*pix.height/pix.width
                pg.insert_image(fitz.Rect(30,y+3,654,y+3+height),filename=str(fig));y+=height+10
            if after:y=box(pg,(30,y,654,776),after)
            if y>776:raise ValueError(f'figure overflow {pg.number}')
            box(pg,(30,788,654,812),f'<span style="color:#5b7677;font-size:8pt">{source}　 /　{pg.number+1}</span>')
        def med(df,col):
            return [f'{df[df.arm==a][col].median():.3f}' for a in ARMS]
        def row(label,frame,col):return [label,*med(frame,col)]
        page('新目标更接近实测脑电，也让更多眼电进入 S',
            p('四库各取 2 名验证被试，每人一个固定 90 秒窗口。S 是离线条件估计，O 是从实测电压 X 中扣除 S 后的剩余电压。以下三种目标使用完全相同的时间、电极和输入。表中先算每位被试，再取中位数。')
            +table(['指标',*NAMES],[row('无注入时 S 与 X 的波形相关',base,'s_x_corr'),
                row('S 的 RMS / X 的 RMS',base,'s_rms'),row('O 的 RMS / X 的 RMS',base,'o_rms'),
                row('眼电注入后 S 变化 / 注入量',eye,'leak'),row('眼电注入后 S 变化 / 原 S',eye,'s_relative'),
                row('O 增量与眼电注入的相关',eye,'o_corr'),row('S 中实测眼电投影 / X 中投影（4 人）',base,'eog_relative')])
            +p('8 人的 S 与 X 相关都提高，8 人的眼电泄漏也都增加。软 SSM 相对 v3 的逐人泄漏差中位数为 +0.670。固定参考同样提高波形相关，在 8 人上都比软 SSM 少传入眼电。')
            +p('这次试验支持保留 v3 作为当前目标。若用途要求抗眼电，不能单凭目标更像 X 或频带保留更多来替换它。10 Hz 和 35 Hz 探针显示软目标也传入更多振荡，但它接收已知眼电的幅度更大。')
            +p('本轮只生成并评估电压目标。S 和 O 的生理来源没有独立逐点真值，离散化难度、码字含义及未见人群的泛化性能仍待单独测量。'),
            'summary.json · baseline.csv · interventions.csv · paired_contrasts.csv')
        page('怎样构造三组可以直接相减的 S/O 目标',
            p('输入为统一 200 Hz、整记录 1–45 Hz 预处理后的原电压坐标。五个 Butterworth 频带覆盖 1–4、4–8、8–13、13–30、30–45 Hz，第六路为原电压减去五路后的精确补项。')
            +table(['目标','时空处理','条件参考'],[
                ['旧 v3','逐帧对多电极载波施加空间二阶矩上界，重叠相加','三条 10 Hz SSM 代理，加冻结训练统计'],
                ['软 SSM','每个电极、每路载波独立乘 0–1 增益，加和为 S','与 v3 相同的条件功率参考'],
                ['固定参考','与软 SSM 相同的参数、目标函数和求解预算','只用各库训练均值与补项功率，随时间固定']])
            +p('SSM 指状态空间模型。它联合当前全窗的 EEG 功率代理和含氧、脱氧血红蛋白（HbO、HbR）拟合。沿用先导每个条件独立拟合的驱动、初态及参数；同一条件的 v3 和软 SSM 共享代理。未扰动时逐窗验证其与生产 v3 的代理、投影及 X 一致，并将 v3 的 S 与生产数组直接核对。')
            +p('两种软目标用每 5 秒一个节点的分段线性增益。优化第一项是 S 与 X 的均方差除以该电极训练 RMS 的平方，第二项是六路最终 S 的对数功率高于参考容许带部分的均方，两项权重均为 1。最终 S 用偶对称边界和双向 Butterworth 的频率响应重新分带，每 0.1 秒取一个完整 1 秒均方功率窗。')
            +p('容许带按各库训练协方差对角线开方，取电极及五频带的中位数再乘 0.5。Single-Trial、Simultaneous、REFED、Visual 分别为 0.648、0.627、0.603、0.628 自然对数单位。两项均采用冻结训练尺度，归一化功率加 10⁻¹² 后取对数。增益初值 0.7，每电极最多优化 100 次。')
            +p(f"计划 8 人、306 个电极位次。按 QC 排除 subject_01 的 PPO1h，保留 305 个。共 8 人 × 4 条件 × 3 目标，96 格均有输出。{summary['fits']} 次软目标电极拟合中 {summary['converged']} 次达到停止精度，39 次达到迭代上限，均纳入主分析。最大闭合误差除以 X 的 RMS 为 {summary['max_closure']:.2g}。"),
            'plan.json · targets/*/*.json · soft_power_so_pilot_v1.yaml · src/data/soft_power_so.py')
        page('已知眼电注入与自然眼电给出一致方向',
            p('每条横线连接同一人的三种方法。左图以实际进入 200 Hz 电压的注入量为分母，中图以未加扰动的 S 为分母。两图的分子均为扰动前后 S 之差在 30–50 秒的全有效电极 RMS。数值越小，S 受扰动越少。右图只含具有独立水平、垂直眼电的 4 人。'),
            'interventions.csv · baseline.csv · ocular_converged_channels.csv',
            overview(figures,cases,eye,base),
            p('使用历史眼电供体，在原采样率按额极最强的固定头皮权重混入，混入电压在 30–50 秒的全电极 RMS 等于原 X。再经与 EEG 匹配的滤波和重采样。Hb 实测输入维持原值。每位被试每种算法均在完整 90 秒上生成目标。')
            +p('未注入记录中，将额部电极的中心化电压投影到同期滤波后水平、垂直 EOG 的线性空间，右图给出 S 与 X 的投影范数比。4 人的软 SSM 均高于 v3。自然相关也可能包括共同活动；注入实验另外提供已知增量。')
            +p('剔除两软目标在眼电配对任一条件中达到迭代上限的电极后，三组眼电泄漏中位数依次为 '+ '、'.join(med(sensitivity,'leak'))+'。'))
        page('S 在五个频带中都保留了更多原电压功率',
            p('每点为 8 人中位数。对同一频带内的 S、O 和 X，在每人的有效电极与 0–89.9 秒取均方。左、中图分别除以 X 的同带均方，右图为 S 与 O 乘积均值的两倍除以同一分母。三项在每人每带精确加为 1，分别取中位数后可能偏离 1。'),
            'bands.csv · baseline.csv', band_figure(figures,bands),
            table(['1 秒功率包络与原 X 的相关',*NAMES],[
                row('θ，4–8 Hz',bands[bands.band=='4-8'],'envelope_corr'),
                row('α，8–13 Hz',bands[bands.band=='8-13'],'envelope_corr')])
            +p('包络按前 89 个完整、不重叠的 1 秒窗计算，每电极求 S、X 同带功率相关，再取电极和被试中位数。软目标保留更多 θ、α 包络起伏。交叉项较大，说明 S 和 O 会共享同频结构。')
            +p('以同一个软 SSM 容许带复核最终电压，v3、软 SSM、固定参考的正向超限对数功率 RMS 分别为 '+ '、'.join(med(base,'excess_rms'))+'。这里考察单电极功率，空间二阶矩上界另有定义。'))
        page('新增振荡更容易进入 S，跨模态增益尚无一致证据',
            p('固定实测 Hb，在一个预定近邻电极的 30–50 秒加入加窗正弦振荡，幅度由原始该电极局部 RMS 确定。定义有符号增益为 S 的变化与已知注入的内积除以注入平方和，跨有效电极与注入时段计算。增益为 1 表示沿注入方向完整传入。')
            +table(['注入方向上的 S 增益',*NAMES],[
                row('10 Hz 神经样探针',changes[changes.variant=='neural10'],'s_signed_gain'),
                row('35 Hz 神经样探针',changes[changes.variant=='neural35'],'s_signed_gain'),
                row('1 倍眼电供体',eye,'s_signed_gain')])
            +p('两种探针都是真值已知的工程增量，Hb 固定。软 SSM 比 v3 保留更多 10 Hz、35 Hz 振荡；同一方法对眼电的有符号增益达到 0.795。单凭对注入振荡的保留不能识别真实神经活动。')
            +p('另取每人三个既定代理位置附近有效电极，计算 S 的 1 秒 α 均方功率与晚 5 秒的实测 Hb 的相关。89 个完整秒组成 84 对；错配对照将同一段 Hb 循环移动 30 秒。先取有效位置中位数，再取 8 人中位数。')
            +table(['S 的 α 功率与 Hb',*NAMES],[row(h+'，匹配',hb[(hb.hb==h)&(hb.component=='S')],'r') for h in ['HbO','HbR']]+
                   [row(h+'，错配',hb[(hb.hb==h)&(hb.component=='S')],'shifted_r') for h in ['HbO','HbR']])
            +p('与 v3 相比，软 SSM 的匹配 HbO 正相关和 HbR 负相关在本面板上均更接近零。两者都使用同窗 Hb 构造教师，这项描述不能验证因果耦合或独立预测能力。固定参考未使用该窗的 Hb；各位置与 O、X 的对应数值见配对数据。'),
            'interventions.csv · hb_associations.csv · hb_subjects.csv')
        for c in cases:
            image,channel,values=example(figures,c)
            g=c['source'];sec=g['native_start_sample']/g['native_rate_hz']
            page(f"{DS[c['dataset']]} · {c['subject']} · {channel}",
                p(f"原记录 {escape(g['native_record'])}，EEG 窗起点 {sec:.3f} 秒。每人展示通道表中首个有效额极；所有人的片段固定。上两行灰线为无注入实测 X，彩线分别为 S 或 O。右列放大同一段 39–41 秒；下排灰线改为已知眼电增量，彩线为 S、O 的相应变化。"),
                f"targets/{c['i']}/baseline.npz · ocular1.npz · plan.json", image,
                table(['局部读数','S 的 RMS，39–41 秒','O 的 RMS，39–41 秒','该电极 S 眼电泄漏'],values)
                +p('上两行共用电压刻度，下排两图共用增量刻度。'+
                   ('REFED 采用发布数组原单位，未校定为 µV。' if c['unit']!='uV' else '纵轴及 RMS 单位均为 µV。')))
        pdf=out/'EEG_SO_目标配对比较.pdf'
        doc.subset_fonts();doc.save(pdf,deflate=True,garbage=4);doc.close()
    with fitz.open(pdf) as pages:
        assert len(pages)==13
        for n,pg in enumerate(pages):
            assert len(pg.get_text())>120 and len(pg.get_image_info())==int(n in (2,3) or n>=5)
            for obj in pg.get_image_info():
                assert pg.rect.contains(fitz.Rect(obj['bbox']))
    return pdf


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    print(render(parser.parse_args().output))
