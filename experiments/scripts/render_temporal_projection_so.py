"""Paired target / integer-replay report from the frozen temporal-projection pilot."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.scripts.evaluate_temporal_projection_so import CFG, OUT, read

ARMS=CFG['arms']
LABELS=['v3','P(S0)','P(X)']
COLORS=['#21597d','#b44520','#167c66']
LINES=['--','-',':']
DS={'eeg_fnirs_single_trial':'Single-Trial','simultaneous_eeg_nirs':'Simultaneous','refed':'REFED','visual_cognitive_motivation':'Visual'}
VQ=OUT/'vq_screen'


def table(cols,rows):
    return '<table><tr>'+''.join('<th>'+str(k)+'</th>' for k in cols)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+str(v)+'</td>' for v in r)+'</tr>' for r in rows)+'</table>'


def p(s):
    return '<p>'+s+'</p>'


def med(frame,col):
    return [f'{frame[frame.arm==a][col].median():.3f}' for a in ARMS]


def save(fig,path):
    fig.savefig(path,dpi=240,facecolor='white')
    plt.close(fig)
    return path


def rank_plot(folder):
    fig,axes=plt.subplots(1,3,figsize=(9,3),layout='constrained',sharey=True)
    for ax,(k,label) in zip(axes,[('s_energy','Energy / original S'),('theta_env_r','Theta envelope r'),('alpha_env_r','Alpha envelope r')]):
        for fn,style,name in [('training_rank_curve','--o','Training selection'),('validation_rank_curve','-s','Validation')]:
            df=pd.read_csv(OUT/(fn+'.csv'))
            vals=df.groupby('pairs')[k].median()
            ax.plot(vals.index*2,vals,style,label=name,ms=4)
        ax.set_title(label,fontsize=11);ax.axvline(64,color='#888',lw=.8);ax.set_xticks([16,32,64,128])
        ax.set_ylim(0,1.05);ax.set_xlabel('Real coefficients / 2 s');ax.grid(axis='y',alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=2,frameon=False)
    return save(fig,folder/'rank.png')


def scores_plot(folder,df,cases):
    fig,axes=plt.subplots(1,3,figsize=(9.2,3.5),layout='constrained')
    for ax,k in zip(axes,'xso'):
        for i,c in enumerate(cases):
            part=df[(df.dataset==c['dataset'])&(df.subject==c['subject'])].set_index('arm')
            vals=part.loc[ARMS,k+'_nmse'].to_numpy()
            ax.plot([vals.min(),vals.max()],[i,i],color='#aab1b3',lw=1)
            for a,v in enumerate(vals):
                ax.scatter(v,i,s=24,marker=['o','s','^'][a],color=COLORS[a],label=LABELS[a] if i==0 else None)
        ax.set_title(k.upper()+' NMSE');ax.set_xlabel('Lower is better');ax.set_ylim(7.6,-.6);ax.grid(axis='x',alpha=.2)
        ax.set_yticks(range(8),[DS[c['dataset']]+' '+c['subject'].replace('subject_','') for c in cases] if ax==axes[0] else [])
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=3,frameon=False)
    return save(fig,folder/'vq_subjects.png')


def waves(folder,c,examples):
    d=OUT/'targets'/str(c['i'])
    idx=next(j for j,k in enumerate(c['valid']) if c['channels'][k].upper() in ('FP1','AFP1'))
    site=c['channels'][c['valid'][idx]]
    vals=[]; decoded=[]
    with np.load(d/'baseline.npz') as f, np.load(d/'ocular1.npz') as noisy:
        offset=int(f['start_in_window']);x=f['x'][:,idx]
        s=[f[a][:,idx] for a in ARMS]
        with np.load(ROOT/CFG['parent']/'targets'/str(c['i'])/'ocular1.npz') as raw:
            inds=slice(offset,offset+len(x));delta=raw['delta'][inds,idx]
            ds=[raw['v3'][inds,idx]-s[0],noisy['projected_v3'][:,idx]-s[1],noisy['projected_x'][:,idx]-s[2]]
    for a in ARMS:
        with np.load(VQ/a/'waveforms.npz') as f:
            m=examples.loc[f['examples']]
            ix=np.flatnonzero((m.subject==c['subject'])&(m.dataset==c['dataset'])&(m.site==site))
            assert len(ix)==1
            decoded.append(f['predictions'][ix[0]].reshape(3,6000)*m.iloc[ix[0]].scale)
    t=(np.arange(len(x))+offset)/200
    fig,axes=plt.subplots(3,2,figsize=(9,6.1),layout='constrained')
    unit='µV' if c['unit']=='uV' else 'export units'
    specs=[(axes[0,0],t,39,41,x,s,'S targets'),(axes[0,1],t,39,41,x,[x-y for y in s],'O targets'),
           (axes[1,0],t,30,50,delta,ds,'Injected voltage and change in S'),
           (axes[1,1],t,30,50,delta,[delta-y for y in ds],'Injected voltage and change in O'),
           (axes[2,0],np.arange(6000)/200+30,30,60,x[6000-offset:12000-offset],[y[0] for y in decoded],'Decoded sum X'),
           (axes[2,1],np.arange(6000)/200+30,39,41,x[6000-offset:12000-offset],[y[0] for y in decoded],'Decoded sum X (zoom)')]
    for ax,clock,lo,hi,grey,ys,title in specs:
        cut=(clock>=lo)&(clock<hi)
        ax.plot(clock[cut],grey[cut],color='#899196',lw=1,label='X / injection')
        for co,ls,lab,y in zip(COLORS,LINES,LABELS,ys):
            ax.plot(clock[cut],y[cut],ls,color=co,lw=1.05,label=lab)
        ax.set_title(title,fontsize=10);ax.set_xlim(lo,hi);ax.set_ylabel(unit);ax.grid(axis='y',alpha=.18)
    for pair in axes:
        yl=min(ax.get_ylim()[0] for ax in pair),max(ax.get_ylim()[1] for ax in pair)
        for ax in pair:ax.set_ylim(*yl)
    for ax in axes[-1]:ax.set_xlabel('Time in parent window (s)')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside upper center',ncol=4,frameon=False)
    return save(fig,folder/f"case_{c['i']}.png"),site


def word_figure(folder,members):
    x=np.load(VQ/'x.npy',mmap_mode='r').reshape(-1,15,400)
    fig,axes=plt.subplots(2,3,figsize=(9,3.8),layout='constrained',sharey=True)
    for j,arm in enumerate(ARMS):
        hits=members[members.arm==arm].sort_values(['dataset','subject']).head(2)
        for k in range(2):
            ax=axes[k,j]
            if k<len(hits):
                row=hits.iloc[k];signal=x[int(row.example),int(row.patch)]
                ax.plot(np.arange(400)/200,signal,color=COLORS[j],lw=.8)
                ax.set_title(f"{LABELS[j]} · ID {row.token} · {row.subject} / {row.site}",fontsize=8)
            else:ax.set_title('No additional matched person',fontsize=8)
            ax.set_xlim(0,2);ax.grid(axis='y',alpha=.15)
            if k==1:ax.set_xlabel('Time in patch (s)')
            if j==0:ax.set_ylabel('X / training scale')
    return save(fig,folder/'real_id_members.png')


def render(dest):
    summary=read(OUT/'summary.json');vq=read(VQ/'summary.json');frozen=read(OUT/'frozen.json')
    assert summary['completed_cells']==96 and vq['status']=='completed'
    dest.mkdir(parents=True,exist_ok=False);figdir=dest/'figures';figdir.mkdir()
    cases=read(OUT/'plan.json')['cases'];b=pd.read_csv(OUT/'baseline.csv');noise=pd.read_csv(OUT/'interventions.csv')
    caps=pd.read_csv(OUT/'final_spatial_cap.csv');eog=pd.read_csv(OUT/'natural_eog.csv')
    vals=pd.read_csv(OUT/'paired_gates.csv');subj=pd.read_csv(VQ/'subject_scores.csv',dtype={'subject':str})
    examples=pd.read_csv(VQ/'examples.csv',dtype={'subject':str})
    macro=pd.read_csv(VQ/'macro.csv').set_index('arm').loc[ARMS]
    cv=pd.read_csv(OUT/'validation_rank_curve.csv')
    doc=fitz.open()
    css="""body{font-family:sans-serif;font-size:12pt;line-height:1.42;color:#223e4b;margin:0}
    h1{font-weight:normal;color:#174b5d;font-size:22pt;margin:0 0 8pt}
    p{margin:1pt 0 7pt}table{width:100%;border-collapse:collapse;font-size:10.5pt;margin:3pt 0 9pt}
    th,td{text-align:left;padding:3pt 4pt;border-bottom:.55pt solid #bdced1}th{background:#e5efee}"""
    def box(pg,rect,body):
        spare,_=pg.insert_htmlbox(fitz.Rect(rect),'<div lang="zh-Hans">'+body+'</div>',css=css,scale_low=1)
        if spare<0:raise ValueError((pg.number,rect,body[:90]))
        return rect[3]-spare
    def page(title,body,source,figure=None,after=''):
        pg=doc.new_page(width=700,height=832)
        box(pg,(30,13,670,36),'<span style="font-size:9pt;color:#40726d">EEG 目标构造与单级量化　 /　 2026 年 11 月 25 日</span>')
        y=box(pg,(30,42,670,785),'<h1>'+title+'</h1>'+body)
        if figure:
            pix=fitz.Pixmap(str(figure));height=640*pix.height/pix.width
            pg.insert_image(fitz.Rect(30,y+3,670,y+3+height),filename=str(figure));y+=height+11
        if after:y=box(pg,(30,y,670,785),after)
        if y>785:raise ValueError((title,y))
        box(pg,(30,799,670,827),f'<span style="font-size:8pt;color:#526a70">{source}　 /　{pg.number+1}</span>')
    cond=noise[noise.variant=='ocular1']
    gain=vq['joint_gain']
    page('限制 EEG-S 的时间自由度，联合重建'+('通过短训比较' if gain else '未通过短训比较'),
        p('比较同一电极在同一时刻的三套目标。S0 是生产版 v3 对实测 EEG 电压 X 的离线估计。P 在每 2 秒只保留固定频点的正余弦系数。候选 P(S0) 同时使用教师和时间投影；对照 P(X) 只使用时间投影。三组 O 均由 X 减去各自的 S。')
        +table(['被试指标',*LABELS],[[label,*med(frame,key)] for label,frame,key in [
            ('相对旧 S 的能量',b,'s_energy'),('α 包络与旧 S 的相关',b,'alpha_env_r'),
            ('眼电注入导致的 S 泄漏',cond,'leak'),('自然眼电在 S / X 的投影范数比，4 人',eog,'eog_ratio')]])
        +p('8 名验证被试来自四个数据集，每人一个既定 90 秒窗口。目标层面满足预设条件，随后在 60 名训练被试上用同一预算分别训练三个模型，验证只读取整数序列进行波形解码。')
        +table(['整数解码误差，越小越好',*LABELS],[[k.upper()+' NMSE',*[f'{v:.4f}' for v in macro[k+'_nmse']]] for k in 'xso'])
        +p('X 的归一化分母在三组间相同；S、O 的分母采用各自目标的冻结训练方差。相对 v3，P(S0) 的总 X 误差在 8 人上都下降；O 的原单位 RMSE 却在 8 人上都升高。P(X) 的总 X 误差最低，同时保留了更多眼电。当前一个种子、500 步的结果支持继续考察候选，尚不足以替换生产目标。'),
        'summary.json · baseline.csv · interventions.csv · vq_screen/macro.csv')
    rankn=sum(r['role']=='rank' for r in frozen['panel'])
    page('频点与维数均在验证被试之外确定',
        p(f'在 60 名 G0 训练被试中，每人按父窗键和起点取首个合格的 30 秒片段。库内按被试排序交替分组，{rankn} 人用于频点排序，{60-rankn} 人用于维数选择。频带内按旧 S 的能量排序：先对各电极作频谱能量归一化，再依次等权平均电极、被试、数据集。')
        +p('在五个频带中按 δ、θ、α、β、θ、α、β、γ 的顺序循环取频点。每个频点同时保留正弦和余弦。以原记录的 200 Hz 时钟每 400 点为绝对格点；只在完整 2 秒块投影，跨块不重叠，裁窗边缘未覆盖的块无标签。没有补功率或重置相位。'),
        'frozen.json · training_rank_curve.csv · validation_rank_curve.csv',rank_plot(figdir),
        p('图中每点是该组被试的中位数，竖线为冻结维数。选维按四个库分别验收，要求各库中位 S 能量至少为旧 S 的 50%，θ、α 带功率各保留 70%，两条功率包络相关各达 0.80。候选上限 32 对，64 对只用于观察维数曲线。')
        +p('8、16、32 对中仅 32 对通过。对应每 400 点 64 个实自由度，占原采样维数 16%，占 1 至低于 45 Hz 的 88 个候选频点对约 36%。整数出口仍是每支每 2 秒一个 ID，连续系数仅离线制作目标。'))
    page('θ 和 α 的起伏保留较好，高频成分明显减少',
        p('在完整的绝对时间块上重新分带，去除两端各 1 秒。在每个有效电极上按不重叠的 1 秒窗计算均方功率。功率比为与旧 S 同带均方之比，相关为两条包络的 Pearson 相关。相邻秒增减方向一致率描述局部变化时序。各项先取电极中位数，再取被试中位数。')
        +table(['候选 P(S0)','频带范围 Hz','功率保留','包络相关','增减方向一致率'],
            [[k,r,*[f"{b[b.arm=='projected_v3'][k+'_'+field].median():.3f}" for field in ['power','env_r','direction']]]
             for k,r in zip(['delta','theta','alpha','beta','gamma'],['1–4','4–8','8–13','13–30','30–45'])])
        +p('候选收窄了 S 的生理含义，β、γ 的局部包络尤其容易丢失。从 S 剔除的电压逐点进入 O，O 的全电极 RMS 相对旧 O 的逐人比值中位数为 '+med(b,'o_over_old_o')[1]+'。')
        +p('分块端点也有代价。块交界处相邻两点的电压差 RMS / 所有相邻点差分 RMS，在 v3、P(S0)、P(X) 上分别为 '+'、'.join(med(b,'boundary_jump_ratio'))+'。因此即使全段能量下降，局部跳变也可能变大。')
        +p('32 个保留频率（Hz）为 '+ '、'.join(f'{v/2:g}' for v in sorted(frozen['order'][:32]))+'。10 Hz 在集合内，35 Hz 在集合外。'),'baseline.csv · frozen.json')
    page('抗眼电主要来自 v3 教师，投影本身保留大量眼电',
        p('沿用同一已知眼电供体、空间加权和 1 倍幅度条件。原采样率下加入扰动后，经匹配的滤波和重采样；使用该条件已拟合的 v3。Hb 观测固定。30–50 秒的 S 变化分别除以实际注入量和基线 S 的 RMS。O 增量相关用相同有效电极和时间。')
        +table(['8 人中位数',*LABELS],[[label,*med(frame,key)] for label,frame,key in [
            ('S 变化 / 注入量',cond,'leak'),('S 变化 / 原 S',cond,'s_relative'),('扰动前后 S 相关',cond,'s_pair_corr'),
            ('O 增量与已知注入的相关',cond,'o_corr'),('10 Hz 注入方向上的 S 增益',noise[noise.variant=='neural10'],'s_gain'),
            ('35 Hz 注入方向上的 S 增益',noise[noise.variant=='neural35'],'s_gain')]])
        +p('正弦探针的有符号增益为 S 改变量与注入量的内积除以注入平方和。P(X) 几乎全收 10 Hz，同时也让大量眼电进入 S。独立水平、垂直眼电导联可用的 4 人中，额部 S 相对 X 的眼电投影范数比中位数分别为 '+'、'.join(med(eog,'eog_ratio'))+'。')
        +table(['最终电压空间上界审计',*LABELS],[[name,*med(caps[caps.variant==v],k)] for name,v,k in [
            ('基线最大广义特征值的 P90','baseline','p90'),('基线超 1 的帧×载波比例','baseline','above_one'),('注入后 P90','ocular1','p90')]])
        +p('按训练参考和当前 SSM 代理重建协方差，将最终波形重新分成五个频带和互补载波。1 秒帧、0.1 秒步长，比较各帧各带的最大广义特征值和上界 1；按人取 P90，再跨人取中位数。P(S0) 相对 v3 的基线和注入后 P90 配对比值中位数均低于 1.25 工程门槛。最终信号仍有超限帧。'),
        'interventions.csv · natural_eog.csv · final_spatial_cap.csv · paired_gates.csv')
    page('固定模型和相同预算下，逐人比较 S、O 与总电压',
        p('每人两条训练父窗，各取首个合格 30 秒，覆盖 60 人、120 条训练裁窗。每组从头训练最终 C 架构，输入单电极实测 X，隐宽 256、码字维数 64、每个 EEG 分支 8192 词，整数后保留同权重频谱监督。各组独立连续预热 100 步，再训练硬量化 400 步；种子 17、批量 64、相同抽样顺序。原 G0 输入尺度固定，目标方差和频谱标准化仅使用本轮训练样本。'),
        'vq_screen/examples.csv · scores.csv · subject_scores.csv · macro.csv',scores_plot(figdir,subj,cases),
        p('每个点是一名验证被试，同一行比较目标。每人按 0、30、60 秒起点取三段，按共同有效的完整 2 秒块评分。先在每个电极裁窗内求均方误差，除以对应库、电极与分支的冻结训练方差，再在被试、数据集之间等权汇总。X 是两条独立 ID 序列的解码和。')
        +p('判据要求 X 误差下降且 S、O 归一化误差均不恶化。'+('本轮达到该条件。' if gain else '本轮未达到该条件。')+'比较限于短预算、同一组开发被试。'))
    stats=[]
    for a in ARMS:
        d=pd.read_csv(VQ/a/'context_ids.csv');g=d.groupby(['dataset','subject'])[['matches','pairs']].sum()
        stats.append((g.matches/g.pairs).median())
    native=[]
    for ds,d in subj.groupby('dataset'):
        for k in 'xso':
            native.append([DS[ds],k.upper(),*[f'{d[d.arm==a][k+"_native_rmse"].mean():.3g}' for a in ARMS]])
    page('原电压误差与上下文稳定性',
        p('表为库内两名被试的原单位 RMSE 均值。每人先等权平均各电极裁窗的原单位均方误差，再开平方。Single-Trial、Simultaneous 和 Visual 使用 µV，REFED 保留发布数组单位。')
        +table(['数据集','输出',*LABELS],native)
        +p('另将第一个 30 秒上下文整体向后移动 2 秒，比较两个合法裁窗中完全相同的重叠 2 秒片段。每人先对 S-ID 精确相等的计数除以有效配对数，再取 8 人中位数。v3、P(S0)、P(X) 分别为 '+'、'.join(f'{100*v:.1f}%' for v in stats)+'。在 8 人中，仅 3 人的候选 O 归一化误差低于 v3，且 8 人的 O 原单位 RMSE 均上升，逐人相对增幅的中位数为 13.1%。新 O 承接了从 S 中移出的电压，自己的训练方差也随之改变。')
        +p('后续每人一页，展示有效通道表中首个 Fp1 或 AFp1。所有人的 39–41 秒局部和 30–60 秒全景按原窗位置固定，未按误差挑选。'),
        'vq_screen/subject_scores.csv · vq_screen/*/context_ids.csv')
    words=pd.read_csv(VQ/'word_attribute_gain.csv')
    members=pd.read_csv(VQ/'word_members.csv',dtype={'subject':str})
    page('从真实 EEG 片段检查单个 S-ID 的属性信息',
        p('每种方法仅用训练数据，按数据集、电极拟合背景回归，用投影后原信号的 θ、α 对数功率和实测 X 的对数均方预测实测 X 的 θ、α 对数功率。再给每个 S-ID 加上训练残差的均值，按 n/(n+20) 收缩。至少覆盖 3 人和 20 个训练片段的词才参与校正。')
        +table(['未见被试，8 人中位数',*LABELS],[[label,*[f'{100*words[words.arm==a][metric].median():.2f}%' for a in ARMS]] for label,metric in [('θ 对数功率额外误差降幅','theta_gain'),('α 对数功率额外误差降幅','alpha_gain'),('有支持的片段占比','support')]]),
        'vq_screen/word_attribute_gain.csv · word_members.csv · word_audit.json',word_figure(figdir,members),
        p('误差降幅为 1 减去加入 ID 后的均方误差与背景误差之比；负值表示在验证被试上误差增加。图中每列取该方法在训练集最常见的有支持 ID，按库、被试顺序列出最多两名验证被试各自第一次匹配到的实测 X。横轴为片段时间，各图共享冻结训练尺度下的纵轴。各方法独立训练词典，词编号只在本列内有意义。')
        +p('波形成员和预测结果仅检验这两项观测属性；稀有词、临床事件或跨人一致的脑源解释需独立证据。'))
    for c in cases:
        fig,site=waves(figdir,c,examples)
        val=subj[(subj.dataset==c['dataset'])&(subj.subject==c['subject'])].set_index('arm')
        page(f"{DS[c['dataset']]} · {c['subject']} · {site}",
            p(f"原记录 {c['source']['native_record']}，EEG 窗口起点 {c['source']['start_sample']/200:.3f} 秒。上排比较同一 2 秒的 S、O 目标，灰线为 X。中排比较 20 秒注入内 S、O 的变化，灰线为已知眼电增量。下排灰线为实测 X，彩线为两支整数解码之和。每行两图共享电压刻度。"),
            f"targets/{c['i']} · vq_screen/*/waveforms.npz · subject_scores.csv",fig,
            table(['该被试全部有效电极和评分裁窗',*LABELS],[[k.upper()+' NMSE',*[f'{v:.3f}' for v in val.loc[ARMS,k+'_nmse']]] for k in 'xso']))
    dest_pdf=dest/'低维时间投影_S目标与单级VQ配对实验.pdf'
    doc.subset_fonts();doc.save(dest_pdf,garbage=4,deflate=True);doc.close()
    with fitz.open(dest_pdf) as d:
        assert len(d)==15
        for i,page_ in enumerate(d):
            assert len(page_.get_text())>120 and len(page_.get_image_info())==int(i in (1,4) or i>=6)
            assert all(page_.rect.contains(fitz.Rect(img['bbox'])) for img in page_.get_image_info())
    return dest_pdf


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    print(render(parser.parse_args().output))
