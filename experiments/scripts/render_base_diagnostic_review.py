"""Render the dated D0/D1/H1 review from final development evaluations."""
import argparse
import json
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

DATASETS = ['eeg_fnirs_single_trial', 'refed', 'simultaneous_eeg_nirs', 'visual_cognitive_motivation']
LABELS = ['Single-Trial', 'REFED', 'Simultaneous', 'Visual']
TEAL, BLUE, ORANGE, INK = '#007960', '#315c96', '#b54a20', '#182d3c'
STYLES = [(TEAL, 'o', '-'), (BLUE, 's', '--'), (ORANGE, '^', ':')]
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,
                     'axes.titleweight':'bold','pdf.fonttype':42,'axes.unicode_minus':False})


def read(path):
    return json.loads(path.read_text())


def validation(job):
    return read(job/'evaluation/step_10000/validation.json')


def points(ax, data, labels, names, ylabel, styles=STYLES, decimals=3):
    x = np.arange(len(labels))
    for j,(values,name,(color,marker,ls)) in enumerate(zip(data,names,styles)):
        dx=(j-(len(data)-1)/2)*.19
        ax.plot(x+dx,values,marker=marker,ms=8,color=color,linestyle='none',label=name)
        if decimals is not None:
            for a,b in zip(x+dx,values):
                ax.annotate(f'{b:.{decimals}f}', (a,b), xytext=(0,8 if j!=1 else -14),
                            textcoords='offset points',ha='center',fontsize=9,color=color)
    ax.set(xticks=x,xticklabels=labels,ylabel=ylabel,xlim=(-.48,len(x)-.52))
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.legend(loc='upper left',fontsize=10)


def export(fig, out, name):
    path=out/(name+'.png')
    fig.savefig(path,dpi=240,facecolor='white')
    plt.close(fig)
    return path


def page(pdf,num,title,sub,image,paragraphs,source,total=4):
    fig=plt.figure(figsize=(11.7,8.3),facecolor='white')
    fig.text(.055,.94,title,fontsize=23,weight='bold',color=INK,va='top')
    fig.text(.055,.875,sub,fontsize=12,color='#405866',va='top')
    ax=fig.add_axes([.055,.345,.89,.48])
    ax.imshow(plt.imread(image),interpolation='none');ax.axis('off')
    for j,(label,body) in enumerate(paragraphs):
        y=.303-j*.082
        fig.text(.058,y,label,fontsize=11.7,color=TEAL,weight='bold',va='top')
        fig.text(.157,y,body,fontsize=11.25,color=INK,va='top',linespacing=1.5)
    fig.text(.055,.035,source,fontsize=8.2,color='#52636a')
    fig.text(.947,.035,f'{num} / {total}',ha='right',fontsize=9)
    for text in fig.texts:
        text.set_fontfamily(['DejaVu Sans','Droid Sans Fallback'])
    pdf.savefig(fig,dpi=240);plt.close(fig)


def render(run,out):
    if run.name != '20261118_d0_d1_h1_v1':
        raise ValueError('This dated review describes the November 18 D0/D1/H1 run.')
    out.mkdir(parents=True,exist_ok=True)
    cfg=read(run/'resolved_config.json')['config']
    result=read(run/'summary.json')
    d0=read(run/'D0/summary.json')
    assert result['status']==d0['status']=='completed' and d0['planned']==len(d0['models'])==6
    train=run/'models'
    g1=run.parent/Path(cfg['sources']['G1']['run']).name/'models'
    rounds=[4000,7000,10000]
    records={}
    for task in cfg['tasks']:
        name='__'.join(task[k] for k in ('modality','architecture','candidate'))
        rec=read(train/name/'summary.json')
        assert rec['completed_step']==10000 and rec['status']=='completed'
        for step in rounds:
            assert (train/name/f'checkpoint_step_{step:05d}.pt').stat().st_size>0
        records[name]=dict(selected=rec['selected_step'],validation={
            str(s):rec['milestones'][str(s)]['validation']['selection_score'] for s in rounds})
    modes=['target_input','X_input_continuous','X_input_K128']
    names=['D1 分支读自身目标','G1 从 X 连续重建','G1 从 X 经 K128 重建']
    arr={m:np.array([[result['D1_by_dataset'][m][d][key] for d in DATASETS] for key in modes]) for m in ['eeg','hb']}
    eeg=arr['eeg'][:,:,:,0]
    fig,axes=plt.subplots(1,3,figsize=(12.3,4.65),layout='constrained',sharey=True)
    for ax,matrix,title in zip(axes,eeg,names):
        im=ax.imshow(matrix,vmin=0,vmax=1.10,cmap='Blues',aspect='auto')
        ax.set(xticks=range(3),xticklabels=['总电压 X','S 分量','O 分量'],yticks=range(4),yticklabels=LABELS,title=title)
        for (i,j),v in np.ndenumerate(matrix):
            ax.text(j,i,f'{v:.3f}',ha='center',va='center',color='white' if v>.61 else INK,fontsize=13)
    fig.colorbar(im,ax=axes,shrink=.74,label='验证 NMSE（越低越好）')
    p1=export(fig,out,'01_eeg_dataset_component')
    panels=[(train/'eeg__target_input__continuous'),(g1/'eeg__waveform__continuous'),(g1/'eeg__waveform__vq128')]
    folders=[p/'evaluation/step_10000' for p in panels]
    meta=read(folders[0]/'validation_examples.json')
    manifest=read(run.parent/Path(cfg['data']['run']).name/'cache/manifest.json')
    samples=[i for i,s in enumerate(meta) if manifest['rows'][s['row']]['dataset']=='refed']
    n=samples[len(samples)//2];entry=meta[n];row=manifest['rows'][entry['row']]
    band_keys=['1_4','4_8','8_13','13_30','30_45']
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12.3,4.65),gridspec_kw={'width_ratios':[1,1.35]},layout='constrained')
    xx=np.arange(5)
    sample_rmse=[]
    for i,(folder,(color,marker,ls)) in enumerate(zip(folders,STYLES)):
        assert meta==read(folder/'validation_examples.json')
        m=read(folder/'validation.json')['metrics']
        v=[10*m['band_'+b+'_log10_power_ratio']['by_dataset']['refed'][0][0] for b in band_keys]
        ax.plot(xx,v,ls=ls,marker=marker,color=color,label=names[i])
        with np.load(folder/'validation_examples.npz') as data:
            mask=data['mask'][n,7].astype(bool)
            y=np.where(mask,data['x'][n,7],np.nan);yhat=np.where(mask,data['prediction_sum'][n,7],np.nan)
            if i==0:
                observed=y
                bx.plot(np.arange(len(y))/200+14,y,color=INK,lw=1.5,label='观测电压')
            else:np.testing.assert_allclose(observed,y,equal_nan=True)
            bx.plot(np.arange(len(y))/200+14,yhat,ls=ls,color=color,lw=1.35,label=names[i])
            sample_rmse.append(float(np.sqrt(np.nanmean((y-yhat)**2))))
    ax.axhline(0,color='#76898d',lw=1);ax.grid(axis='y',alpha=.18)
    ax.set(xticks=xx,xticklabels=[b.replace('_','–') for b in band_keys],xlabel='频段（Hz）',
           ylabel='预测 / 观测功率（dB）',title='REFED 全验证面板 · 总电压 X')
    ax.legend(loc='lower left',fontsize=9)
    bx.set(xlabel='30 秒窗口内时间（秒）',ylabel='电压 / 训练 X 标准差',title='同一验证窗口 · 中间 2 秒')
    bx.legend(fontsize=8,ncol=2,loc='upper right'); bx.margins(x=.005,y=.25)
    p2=export(fig,out,'02_refed_spectrum_waveform')
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12.3,4.65),layout='constrained',gridspec_kw={'width_ratios':[1.25,1]})
    hb=arr['hb'].mean(1)
    points(ax,hb.reshape(3,6),['X·HbO','X·HbR','S·HbO','S·HbR','O·HbO','O·HbR'],names,'验证 NMSE',decimals=None)
    ax.set(title='输入条件对六项 Hb 误差的影响',ylim=(-.02,.56))
    hm=read(run/'D0/G1_hb_vq128/reconstruction.json')['metrics']
    cross=np.array([hm['signed_two_residual_cross_term']['by_dataset'][d] for d in DATASETS])
    for c,(color,marker) in enumerate([(BLUE,'s'),(ORANGE,'^')]):
        bx.scatter(cross[:,c],np.arange(4)+(c-.5)*.18,marker=marker,c=color,label=['HbO','HbR'][c],s=66)
    bx.axvline(0,color=INK,lw=1);bx.set(yticks=np.arange(4),yticklabels=LABELS,xlabel='两倍 S/O 误差乘积均值（训练 X 标准差的平方）',
           title='G1 硬量化 · 加和时的误差相消',xlim=(-.21,.012),ylim=(3.45,-.5))
    bx.grid(axis='x',alpha=.18); bx.legend(loc='lower center',fontsize=9)
    p3=export(fig,out,'03_hb_targets_cancellation')
    h1=[validation(train/f'hb__frozen_{g}__vq128') for g in ('g1','g4')]
    h1vals=np.array([x['metrics']['nmse']['macro'] for x in h1]).reshape(2,6)
    subject_maps=[{(s['dataset'],s['subject']):np.mean(s['value']) for s in x['metrics']['nmse']['by_subject']} for x in h1]
    assert subject_maps[0].keys()==subject_maps[1].keys()
    delta={k:float(subject_maps[1][k]-subject_maps[0][k]) for k in subject_maps[0]}
    gains=np.array([read(run/f'D0/{g}_hb_vq128/shape_readout.json')['gain_macro'] for g in ('G1','G4')])[:,:,1,0,:].reshape(2,4)*100
    for g in ('g1','g4'):
        assert read(train/f'hb__frozen_{g}__vq128/assignment_verification.json')['status']=='passed'
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12.3,4.65),layout='constrained')
    pairnames=['冻结 G1 赋码','冻结 G4 赋码']
    points(ax,h1vals,['X·HbO','X·HbR','S·HbO','S·HbR','O·HbO','O·HbR'],pairnames,'验证 NMSE（越低越好）',STYLES[1:],None)
    ax.set(title='同一 waveform 解码器结构 · 全波形重建',ylim=(-.015,.55))
    idx=np.arange(4)
    for j,(v,col) in enumerate(zip(gains,[BLUE,ORANGE])):
        bars=bx.bar(idx+(j-.5)*.35,v,.32,label=pairnames[j],color=col,hatch='' if j==0 else '//',edgecolor='white')
        bx.bar_label(bars,fmt='%.1f',padding=3,fontsize=10)
    bx.axhline(0,color=INK,lw=.8)
    bx.set(xticks=idx,xticklabels=['S·HbO','S·HbR','O·HbO','O·HbR'],ylabel='相对无 ID 基线的误差降低（%）',
           ylim=(-13,78),title='单个 ID 对 2 秒局部形状的解释增量')
    bx.legend(loc='upper center',ncol=2,fontsize=9);bx.grid(axis='y',alpha=.16);bx.set_axisbelow(True)
    p4=export(fig,out,'04_hb_frozen_codes')
    metrics_path=out/'completion_review.json'
    valids=[read(train/f'hb__frozen_{g}__vq128/assignment_verification.json')['identical_ids'] for g in ('g1','g4')]
    recovery={c:read(run/f'D0/G1_hb_{c}/recovered_boundary_validation.json')['max_abs_nmse_difference'] for c in ('continuous','vq128')}
    metrics_path.write_text(json.dumps(dict(status='completed',seed=cfg['training']['seed'],completed_D0=len(d0['models']),
        formal_models=records,retained_source_failures=read(run/'plan.json')['source_failures'],H1_verified_ids=valids,
        H1_paired_subject_deltas=[dict(dataset=k[0],subject=k[1],G4_minus_G1=v) for k,v in delta.items()],
        recovery_max_abs_nmse_difference=recovery,figures=[p.name for p in (p1,p2,p3,p4)],
        d0_recovery='Original finalization failed on missing G1 boundary_jump_nmse. Recovered from the frozen validation panel and verified against retained per-input NMSE.',
        EEG_example=dict(dataset=row['dataset'],subject=row['subject'],record=row['record'],position=row['positions'][entry['position']],
                         window_start_s=entry['start_s'],display_patch=7,rmse_in_train_X_SD=sample_rmse)),ensure_ascii=False,indent=2)+'\n')
    with PdfPages(out/'D0_D1_H1_results_20261119_v1.pdf') as pdf:
        page(pdf,1,'EEG 的量化代价大于更换输入带来的平均收益','D0 / D1 / H1 实验复核　·　2026 年 11 月 19 日　·　seed 17　·　固定 10,000 步开发集评价',p1,[
            ('完成情况','D0 六个诊断完成；D1 两个连续模型和 H1 两个 Hb 解码器均完成 4k、7k、10k 评价，最优均在 10k。\n原 G0 的 6 个 EEG 父窗口失败保留。训练 60 人、443 条记录；验证 20 人、144 条记录，四库等权。'),
            ('比较方法','每 30 秒输出 15 个 2 秒表示。D1 的 S、O 分支各读自身监督目标，X 由两支相加；G1 两支均读本模态 X。\n先在有效点计算 MSE，除以各目标的训练方差（含合同下限），再按位置、窗口、记录、被试、数据集等权汇总。'),
            ('结果','EEG 宏 NMSE 为 0.2249、0.2515、0.3975（按图从左至右）。读自身目标使 S 在四库均改善；\n宏 S 从 0.4097 降至 0.3310。X 从 0.1630 变为 0.1652。两个对照差值仅描述本组结构与训练条件。')],
            '证据　summary.json · D1_by_dataset/eeg　｜　各模型 evaluation/step_10000/validation.json　｜　宏分数对 X/S/O 与色团等权。')
        page(pdf,2,'REFED 直接读取分量后，高频电压仍然偏弱','左图汇总 REFED 验证被试；右图展示同一模型在固定样例上的实际波形。两图均评价总电压 X。',p2,[
            ('频谱','每个 30 秒窗口去均值、加 Hann 窗，按频带汇总功率，计算 10 log₁₀（预测 / 观测），再逐层等权。\n0 dB 表示功率匹配。13–30 Hz 的 D1、G1 连续、G1 K128 分别约为 −43.2、−43.4、−58.1 dB。'),
            ('波形',f'取原固定 REFED 验证样例的中间位置（被试 {row["subject"]}，{row["record"]}，{row["positions"][entry["position"]]}），显示窗口第 14–16 秒。\n这 2 秒内三条预测的 RMSE 分别为 {sample_rmse[0]:.3f}、{sample_rmse[1]:.3f}、{sample_rmse[2]:.3f} 个训练 X 标准差；曲线共用坐标。'),
            ('判断','REFED 的 D1 宏 X/S/O 为 0.559 / 0.609 / 0.642。G1 连续 X 的片内形状 NMSE 约 0.551，总 NMSE 为 0.552。\n下一项优先检验连续前后端的片内时间分辨率；固定每两秒一个 token，再观察频谱和验证误差能否同步改善。')],
            '证据　D0/G1_eeg_{continuous,vq128}/reconstruction.json　｜　对应模型 validation_examples.npz 与 D1 validation.json。')
        page(pdf,3,'Hb-O 自身容易压缩，从观测 X 恢复则困难','HbO 与 HbR 在同一 Hb 分支内联合表示。验证集有 70,972 个 Hb 窗口与位置组合，独立验证被试为 20 人。',p3,[
            ('目标输入','Hb 宏 NMSE 从 G1 连续的 0.1550 降至 D1 的 0.00703。变化最大的是 O 分量：HbO 从 0.4333 降至 0.0038，\nHbR 从 0.3056 降至 0.0042。S/O 是缓存分解的监督目标；D1 用于诊断，推理时需要已有分量。'),
            ('误差相消','各分量误差取预测减目标。在同一有效支持、同一训练 X 尺度上，总信号 MSE 等于 S 与 O 的 MSE 之和加交叉项。\n四库、两色团的交叉项均为负；G1 K128 的 HbO/HbR 宏值为 −0.1474 / −0.0323。'),
            ('含义','当前架构可以保留 Hb-O 波形。只给 X 时，分解难度和 S/O 误差相消共同影响结果，较低的 X 误差不足以评价分量。\n各分量 NMSE 的训练分母不同。原生 RMSE、训练方差和实际分母在 D0 按库列出，可用于同库追溯。')],
            '证据　summary.json · D1_by_dataset/hb　｜　D0/G1_hb_vq128/reconstruction.json · signed_two_residual_cross_term。')
        level=read(run/'D0/G4_minus_G1.json')['metrics']['vq128']
        page(pdf,4,'G4 保留了形状分组优势，重建仍落后于 G1','H1 两组冻结各自编码器和码本，使用相同的新 waveform 解码器初始化、训练样本、损失与更新预算。',p4,[
            ('匹配结果',f'验证宏 NMSE 为 G1 0.18265、G4 0.20880，差 +0.02615（+14.3%）。{sum(v>0 for v in delta.values())}/{len(delta)} 名配对被试的 G4 误差更高。\n与原 G4 硬模型 0.21565 相比，新解码器改善约 3.2%。验证集两支各 1,064,580 个 ID 与各自来源逐项一致。'),
            ('形状读出','每个 2 秒片段先去均值，用训练集中同码片段拟合原型；在留出被试上与同库同位置的无 ID 均值原型比较。\n右图为 100 ×（1 − 单 ID 误差 / 无 ID 误差），使用完整验证汇总，原型按训练片段汇集。负值表示低于基线。'),
            ('解读',f'G4 的两项 S 形状增益均接近 60%，但 H1 的 X 误差从 0.0321 / 0.0349 升至 0.0628 / 0.0659。\n原 G4 对原 G1 的 X 水平项 NMSE 增量为 {level["level_nmse"][0][0]:.4f} / {level["level_nmse"][0][1]:.4f}。这些结果支持保留 G1 为重建基线。')],
            '证据　H1 两组 validation.json、assignment_verification.json　｜　D0/{G1,G4}_hb_vq128/shape_readout.json、G4_minus_G1.json。')


def render_phase(diagnostic_run, e1_run, out):
    """Verify matched development endpoints and assemble the six-page phase review."""
    import fitz
    from src.analysis.dual_component_tokenizer import base_training_aggregate

    out.mkdir(parents=True, exist_ok=False)
    prior = diagnostic_run / 'review_20261119_v1'
    review = read(prior / 'completion_review.json')
    run = read(e1_run / 'summary.json')
    plan = read(e1_run / 'plan.json')
    config = read(e1_run / 'resolved_config.json')['config']
    controller = read(e1_run / 'controller.json')
    assert controller['status'] == run['status'] == review['status'] == 'completed'
    assert run['completed_models'] == run['planned_models'] == 1
    assert diagnostic_run.name == Path(config['e1']['decision_run']).name
    cutoff = config['training']['total_steps']
    assert cutoff == 10000 and plan['seed'] == review['seed'] == 17
    rows = read(e1_run.parent / Path(plan['G0_run']).name / 'cache/manifest.json')['rows']
    baseline = Path(plan['baseline_job'])
    jobs = {'G1': e1_run.parent / baseline.parent.parent.name / 'models' / baseline.name,
            'E1': e1_run / 'models' / run['models'][0]['task_id']}
    bands = ['1_4', '4_8', '8_13', '13_30', '30_45']
    fields = ['nmse', *[f'band_{band}_log10_power_ratio' for band in bands]]
    models = {}
    reference_ids = None
    for label, job in jobs.items():
        record = read(job / 'summary.json')
        assert record['status'] == 'completed' and record['completed_step'] == cutoff
        milestones = {}
        for step in config['evaluation']['steps']:
            mark = record['milestones'][str(step)]
            assert mark['status'] == 'completed'
            if label == 'E1':
                assert (job / f'checkpoint_step_{step:05d}.pt').stat().st_size > 0
            milestones[step] = {role: mark[role]['selection_score'] for role in ('train', 'validation', 'regression')}
        folder = job / 'evaluation' / f'step_{cutoff:05d}'
        val = read(folder / 'validation.json')
        with np.load(folder / 'validation.npz') as arrays:
            ids = arrays['examples']
            if reference_ids is None:
                reference_ids = ids
            else:
                np.testing.assert_array_equal(ids, reference_ids)
            check = base_training_aggregate({key: arrays[key] for key in fields}, ids, rows)
        for key in fields:
            np.testing.assert_allclose(check['metrics'][key]['macro'], val['metrics'][key]['macro'])
            for ds in DATASETS:
                np.testing.assert_allclose(check['metrics'][key]['by_dataset'][ds], val['metrics'][key]['by_dataset'][ds])
        np.testing.assert_allclose(check['selection_score'], run['E1']['baseline_macro_nmse' if label == 'G1' else 'candidate_macro_nmse'])
        nmse = check['metrics']['nmse']
        models[label] = dict(source=job.relative_to(e1_run.parent).as_posix(), selected_step=record['selected_step'],
            parameters=record['parameters'], milestones=milestones, component_order=val['component_order'],
            macro_nmse=nmse['macro'], dataset_nmse=nmse['by_dataset'], subject_nmse=nmse['by_subject'],
            refed_X_dB={band: 10*check['metrics'][key]['by_dataset']['refed'][0][0] for band, key in zip(bands, fields[1:])})
    old, new = models.values()
    subjects = [{(s['dataset'], s['subject']): np.mean(s['value']) for s in m['subject_nmse']} for m in (old, new)]
    assert subjects[0].keys() == subjects[1].keys()
    n_improved = sum(subjects[1][s] < v for s, v in subjects[0].items())
    endpoint = run['E1']
    for band, v in endpoint['REFED_X'].items():
        np.testing.assert_allclose([old['refed_X_dB'][band], new['refed_X_dB'][band]], [v['baseline_dB'], v['candidate_dB']])
    diag = read(diagnostic_run / 'summary.json')
    assert diag['status'] == 'completed' and diag['completed_models'] == diag['planned_models'] == 4
    evidence = dict(seed=plan['seed'], controller=controller, source_failures=len(plan['source_failures']),
        D0=read(diagnostic_run / 'D0/summary.json'), D1_by_dataset=diag['D1_by_dataset'],
        diagnostic_review=f'{diagnostic_run.name}/{prior.name}/completion_review.json', E1=endpoint,
        population={k: check[k] for k in ('inputs', 'windows', 'records', 'subjects')},
        E1_subjects_improved=int(n_improved), models=models, formal_diagnostic_models=review['formal_models'],
        H1_subject_deltas=review['H1_paired_subject_deltas'], test_arrays_read=run['test_arrays_read']+diag['test_arrays_read'],
        teacher_refits=run['teacher_refits']+diag['teacher_refits'])
    (out/'phase_results.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.3,4.65), layout='constrained')
    phase_styles = [(BLUE, 's', '--'), ('#783e75', 'o', '-')]
    for (label, m), (color, marker, ls) in zip(models.items(), phase_styles):
        steps = list(m['milestones'])
        ax.plot(steps, [m['milestones'][s]['validation'] for s in steps], label=label, color=color, marker=marker, ls=ls)
        bx.plot(range(len(bands)), list(m['refed_X_dB'].values()), label=label, color=color, marker=marker, ls=ls)
    ax.set(xticks=steps, xlabel='累计训练步数', ylabel='开发验证宏 NMSE（越低越好）', title='EEG · 三个固定检查点')
    bx.set(xticks=range(len(bands)), xticklabels=[b.replace('_','–') for b in bands], xlabel='频带（Hz）',
           ylabel='预测 / 目标功率（dB）', title='REFED 总电压 X · 10,000 步')
    bx.axhline(0, color=INK, linewidth=.8, label='目标功率')
    for axis in (ax, bx):
        axis.legend(); axis.grid(axis='y', alpha=.18)
    spectrum = export(fig, out, 'e1_matched_validation')
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.3,4.65), layout='constrained', gridspec_kw={'width_ratios':[1.4,1]})
    points(ax, [[np.mean(m['dataset_nmse'][ds]) for ds in DATASETS] for m in models.values()], LABELS, list(models), '宏 NMSE（越低越好）', styles=phase_styles, decimals=None)
    ax.set_title('四个开发数据集 · 各库 X / S / O 等权')
    points(bx, [np.ravel(m['macro_nmse']) for m in models.values()], ['X 总电压','S 分量','O 分量'], list(models), '分量 NMSE（越低越好）', styles=phase_styles, decimals=None)
    bx.set_title('三个重建目标 · 四库等权')
    breakdown = export(fig, out, 'e1_datasets_components')
    front = BytesIO()
    with PdfPages(front) as pdf:
        page(pdf, 1, 'EEG 片内位置从 4 增至 8，重建误差降低 8.7%',
             'D0 / D1 / H1 / E1 阶段汇总　·　2026 年 11 月 19 日　·　四个开发数据集，单一训练种子 17', spectrum, [
            ('比较条件','G1 和 E1 均从 30 秒 EEG 总电压 X 预测 S、O，二者之和重建 X。两组均使用 15 个两秒锚点。\n每个锚点内从 4 个位置增至 8 个，参数从 129.1 万增至 135.7 万。两组训练预算均为 10,000 步。'),
            ('固定终点','验证宏 NMSE 从 0.25148 降至 0.22957。REFED 的 13–30 Hz 功率偏差从 −43.392 变为 −38.803 dB；\n30–45 Hz 从 −45.235 变为 −45.225 dB，仅靠近零 0.010 dB。三项达到预设方向，高频仍有明显衰减。'),
            ('读图','左图每点为完整开发验证的固定步评价。右图对每个 30 秒输入去均值、加 Hann 窗后计算频带功率，\n汇总 10 × log₁₀（预测功率 / 目标功率）。0 dB 表示功率匹配，负值表示衰减。')],
            '数据　G1 / E1 · 训练检查点与验证数组　｜　附录保留 D0、D1、H1 的四页原始图解。', total=2)
        page(pdf, 2, '四个数据集均有改善，Hb 的形状读出与重建表现分化',
             'E1 使用连续表示。离散码的波形与分量表现来自 D0、D1、H1 的独立对照。', breakdown, [
            ('EEG 分层',f'同一组 65,909 个窗口–位置输入，来自 1,920 个窗口、144 条记录、20 名被试。{n_improved}/20 名配对被试误差降低。\nNMSE 按相应目标的冻结训练方差归一，依次按位置、窗口、记录、被试、数据集等权；REFED 改善最大。'),
            ('输入诊断','D0 核对六组已有模型。D1 让两支直接读取自身的 S/O 教师目标，EEG 宏 NMSE 从 0.25148 降至 0.22491，\nHb 从 0.15498 降至 0.00703。原 G0 的六个 EEG 父窗口失败已计入记录。S/O 的参照来自缓存教师分解。'),
            ('离散读出','D0 中 G4 单码的 S 片段形状读出优于 G1。H1 冻结各自赋码，用同初始化的新解码器重建，\nG4 的 Hb 宏 NMSE 为 0.20880，高于 G1 的 0.18265，20/20 名配对被试方向一致。')],
            '证据　phase_results.json · 10,000 步按库、被试和分量汇总　｜　附录内页码单独编号。', total=2)
    with fitz.open(stream=front.getvalue(), filetype='pdf') as report, fitz.open(prior/'D0_D1_H1_results_20261119_v1.pdf') as appendix:
        assert len(appendix) == 4 and all(page.get_images() for page in appendix)
        report.insert_pdf(appendix)
        report.save(out/'D0_D1_H1_E1_results.pdf', deflate=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--e1-run',type=Path,help='Include E1 and assemble the dated phase report in a fresh output directory.')
    args=parser.parse_args()
    if args.e1_run:
        render_phase(args.run,args.e1_run,args.output)
    else:
        render(args.run,args.output)
