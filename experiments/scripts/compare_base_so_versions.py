"""Compare the two retained G1 waveform suites at their matched final exposure."""
import argparse
import base64
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

from run_dual_component_tokenizer import (
    SOURCE, _base_training_data, _base_training_examples, identity, read_json,
)
from src.analysis.dual_component_tokenizer import base_training_aggregate
from src.data.dual_component_tokenizer import base_tokenizer_batch

RUNS = SOURCE/'experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer'
RUN = dict(v2=RUNS/'20261117_base_g1_v2', v3=RUNS/'20261122_unified_base_g1_v1')
VERSIONS = list(RUN)
MODALITIES = ('eeg', 'hb')
COLORS = ['#225a87', '#b44c29']
DATASETS = {'eeg_fnirs_single_trial':'Single-Trial','refed':'REFED',
            'simultaneous_eeg_nirs':'Simultaneous','visual_cognitive_motivation':'Visual'}
plt.rcParams.update({'font.family':['Noto Sans CJK JP','DejaVu Sans'], 'font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})


def evaluation(version, modality, candidate):
    return RUN[version]/'models'/f'{modality}__waveform__{candidate}'/'evaluation/step_10000'


def sample_key(row, start, position):
    return (row['dataset'],row['subject'],row['record'],row['parent'],int(start),row['positions'][position])


def collect(version):
    root = RUN[version]
    cfg = read_json(root/'resolved_config.json')['config']
    manifest, scales, _, _ = _base_training_data(cfg, SOURCE)
    assert manifest['cfg_identity'] == identity(read_json(SOURCE/cfg['data']['run']/'resolved_config.json')['config'])
    assert read_json(SOURCE/cfg['data']['run']/'resolved_config.json')['config']['data']['source_schema'] == 'physiology_so_supervision_'+version
    rows = manifest['rows']
    sig = read_json(root/'plan.json')['manifest_identity']
    assert identity(manifest) == sig
    result = {}
    for modality in MODALITIES:
        hard = evaluation(version,modality,'vq128')
        expected = np.asarray(_base_training_examples(rows,modality,'validation'))
        values = {}
        for arm in ('continuous','vq128'):
            path = evaluation(version,modality,arm)
            summary = read_json(path.parent.parent/'summary.json')
            assert summary['completed_step']==10000 and summary['status']=='completed'
            assert summary['signature']['input']==sig and summary['signature']['config']==identity(cfg)
            with np.load(path/'validation.npz') as f:
                np.testing.assert_array_equal(f['examples'],expected)
                for field in ('nmse','correlation'):
                    values[arm+'_'+field]=f[field]
                if arm=='vq128':
                    ids = {b:f['ids_'+b] for b in 'SO'}
                    for b in 'SO':
                        values['pattern_'+b]=f['pattern_'+b+'_'+b.lower()]
                        assert not f['pattern_'+b+'_'+b.lower()+'_unsupported_fraction'].any()
        keys=[sample_key(rows[i],s,p) for i,s,p in expected]
        assert len(set(keys))==len(keys)
        train_health=read_json(hard/'train.json')['code_health']
        health={h['branch']:h for h in read_json(hard/'validation.json')['code_health'] if h['dataset']=='macro'}
        channels=1 if modality=='eeg' else 2
        with np.load(hard/'training_ID_prototypes.npz') as f:
            prototypes={b:f[b+'_'+b.lower()+'_prototype'].reshape(128,-1,channels) for b in 'SO'}
        gallery={}
        for b in 'SO':
            prob=next(h['probability'] for h in train_health if h['dataset']=='macro' and h['branch']==b)
            for code in np.argsort(-np.asarray(prob),kind='stable')[:2]:
                template=prototypes[b][code]
                gallery[b,int(code)]=dict(template=template, total=np.zeros_like(template), n=0,
                                          sample=np.empty((0,*template.shape)), priority=np.empty(0))
            values['zero_'+b]=np.empty((len(keys),channels))
        full=np.empty((len(keys),15),bool)
        rng=np.random.default_rng(17)
        cache=OrderedDict()
        for start in range(0,len(keys),48):
            sl=slice(start,start+48)
            batch=base_tokenizer_batch(rows,expected[sl],scales,array_cache=cache,cache_limit=16)
            mask=batch['mask'].reshape(len(expected[sl]),15,-1,channels)
            full[sl]=mask.all(axis=(2,3))
            count=mask.sum(2,keepdims=True)
            for b in 'SO':
                y=batch[b.lower()].reshape(mask.shape)
                y=np.where(mask,y-np.where(mask,y,0).sum(2,keepdims=True)/np.maximum(count,1),0)
                variance=batch['v'+b.lower()][:,0,:channels]
                denom=mask.sum((1,2))*variance
                values['zero_'+b][sl]=(y*y).sum((1,2))/denom
                prediction=prototypes[b][ids[b][sl]]
                error=np.where(mask,(prediction-y)**2,0).sum((1,2))/denom
                np.testing.assert_allclose(error,values['pattern_'+b][sl],atol=2e-6,rtol=1e-5)
                for (branch,code),g in gallery.items():
                    if branch!=b:continue
                    waves=y[(ids[b][sl]==code)&full[sl]]
                    if not len(waves):continue
                    g['n']+=len(waves);g['total']+=waves.sum(0)
                    priority=np.r_[g['priority'],rng.random(len(waves))]
                    keep=np.argsort(priority)[:10]
                    g['sample']=np.concatenate([g['sample'],waves])[keep];g['priority']=priority[keep]
        fixed=read_json(hard/'validation_examples.json')
        with np.load(hard/'validation_examples.npz') as f:
            examples={sample_key(rows[a['row']],a['start_s'],a['position']):
                      {k:f[k][i] for k in ('mask','s','o','prediction_S','prediction_O')} for i,a in enumerate(fixed)}
        result[modality]=dict(keys=keys,values=values,ids=ids,full=full,health=health,
                              gallery=gallery,examples=examples)
    return result


def write_csv(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize(inputs):
    records=[]; book=[]; counts=[]
    for m in MODALITIES:
        a,b=[d[m] for d in inputs]
        common=sorted(set(a['keys'])&set(b['keys']))
        rows=[dict(zip(('dataset','subject','record','parent'),k[:4])) for k in common]
        examples=[(i,k[4],k[5]) for i,k in enumerate(common)]
        ix=[]
        for d in (a,b):
            lut={k:i for i,k in enumerate(d['keys'])}
            ix.append(np.array([lut[k] for k in common]))
        valid=a['full'][ix[0]] & b['full'][ix[1]]
        counts.append(dict(modality=m,old=len(a['keys']),new=len(b['keys']),paired=len(common),
                           subjects=len({k[:2] for k in common}),full_patch_pairs=int(valid.sum())))
        for v,d,j in zip(VERSIONS,(a,b),ix):
            values=d['values']
            for arm in ('continuous','vq128'):
                metrics={f:values[arm+'_'+f][j] for f in ('nmse','correlation')}
                if arm=='vq128':
                    for name in ('pattern','zero'):
                        metrics[name]=np.stack([values[name+'_'+br][j] for br in 'SO'],axis=1)
                agg=base_training_aggregate(metrics,examples,rows)['metrics']
                for group in ('macro',*DATASETS):
                    get=lambda field:np.asarray(agg[field]['macro'] if group=='macro' else agg[field]['by_dataset'][group])
                    nmse=get('nmse');corr=get('correlation')
                    for component,errs,cc in zip('XSO',nmse,corr):
                        idx='SO'.find(component)
                        pat=arm=='vq128' and idx>=0
                        e=get('pattern')[idx].mean() if pat else None
                        zero=get('zero')[idx].mean() if pat else None
                        records.append(dict(version=v,modality=m,arm=arm,group=group,component=component,
                            nmse=errs.mean(),correlation=cc.mean(),single_ID_nmse=e,zero_shape_nmse=zero,
                            shape_gain=(1-e/zero) if pat else None))
                if arm=='vq128':
                    # Compare partitioning of identical, fully scored patches; code labels may permute.
                    for branch in 'SO':
                        h=d['health'][branch]
                        ari=adjusted_rand_score(a['ids'][branch][ix[0]][valid],b['ids'][branch][ix[1]][valid])
                        book.append(dict(version=v,modality=m,branch=branch,active=h['active'],
                                         effective=h['effective'],top1=h['top1'],paired_patch_ARI=ari))
    return records,book,counts


def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=210,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def plots(out,records,inputs,book):
    def get(v,m,c,g='macro',arm='vq128'):
        return next(r for r in records if (r['version'],r['modality'],r['component'],r['group'],r['arm'])==(v,m,c,g,arm))
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,m in zip(axs,MODALITIES):
        groups=list(DATASETS)
        diff=np.array([[get('v3',m,c,g)['nmse']-get('v2',m,c,g)['nmse'] for c in 'XSO'] for g in groups])
        for k,(c,color) in enumerate(zip('XSO',['#526578','#bd5939','#27847e'])):
            ax.barh(np.arange(4)+(k-1)*.23,diff[:,k],.21,label=c,color=color,hatch=['','//','..'][k])
        ax.set_yticks(range(4),[DATASETS[g] for g in groups]);ax.invert_yaxis();ax.axvline(0,c='#333',lw=.9)
        ax.set_title(m.upper());ax.set_xlabel('同一位置的验证 NMSE 差值（v3 − v2）')
        ax.legend(ncol=3,loc='lower center',bbox_to_anchor=(.5,-.33))
    fig.suptitle('K128 重建误差变化　｜　正值表示新版误差增大，各模态独立横轴')
    save(fig,out,'reconstruction')
    fig,axs=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    groups=[(m,c) for m in MODALITIES for c in 'SO']
    for j,v in enumerate(VERSIONS):
        for ax,ys in [(axs[0],[100*get(v,m,c)['shape_gain'] for m,c in groups]),
                      (axs[1],[next(r['effective'] for r in book if (r['version'],r['modality'],r['branch'])==(v,m,c)) for m,c in groups])]:
            bars=ax.bar(np.arange(4)+(j-.5)*.36,ys,.34,label=v,color=COLORS[j],hatch=['','//'][j])
            ax.bar_label(bars,fmt='%.1f',fontsize=9,padding=2)
    for ax,title in zip(axs,['单 ID 原型相对零形状的误差降幅（%）','有效码数（128 个可用码字）']):
        ax.set_xticks(range(4),[m.upper()+'-'+c for m,c in groups]);ax.set_title(title);ax.legend();ax.margins(y=.19)
    save(fig,out,'token_metrics')
    for m in MODALITIES:
        ch=1 if m=='eeg' else 2
        fig,axs=plt.subplots(4,2,figsize=(12,9.4),layout='constrained',sharex=True,sharey='row')
        for j,(v,d) in enumerate(zip(VERSIONS,inputs)):
            g=d[m]['gallery']
            for k,((branch,code),item) in enumerate(g.items()):
                ax=axs[k,j];x=np.arange(len(item['template']))/(200 if ch==1 else 10)
                for h in range(ch):
                    color=['#236282','#ae5035'][h]
                    ax.plot(x,item['sample'][:,:,h].T,color=color,alpha=.18,lw=.65)
                    label='EEG' if ch==1 else ['HbO','HbR'][h]
                    marker='.' if h==1 else None
                    ax.plot(x,item['template'][:,h],color=color,lw=1.65,marker=marker,label=label+' 训练原型')
                    ax.plot(x,item['total'][:,h]/item['n'],color=color,ls='--',lw=1.6,marker=marker,label=label+' 验证均值')
                ax.set_title(f"{v}  {m.upper()}-{branch}  ID {code}　验证完整片段 {item['n']:,} 个",fontsize=10)
                if j==0:ax.set_ylabel('去均值幅度 / 训练尺度')
                if k==0:ax.legend(ncol=2,fontsize=8)
                if k==3:ax.set_xlabel('片内时间（秒）')
        fig.suptitle('按各自训练使用率展示每分支前两码　｜　浅线为均匀抽取的 10 个验证片段，同排按频率排名')
        fig.savefig(out/('patterns_'+m+'.png'),dpi=210,bbox_inches='tight')
        for ax in axs.flat:
            for line in list(ax.lines):
                if line.get_label().startswith('_'):
                    line.remove()
            ax.relim()
        for ax in axs[:,0]:
            ax.autoscale_view()
        fig.suptitle('高频码的平均形状细节　｜　实线为训练原型，虚线为验证均值，同排共用纵轴')
        save(fig,out,'mean_patterns_'+m)
    # Pick one fixed, shared REFED window-position per modality, independently of errors.
    notes=[]
    for m in MODALITIES:
        aa,bb=[d[m]['examples'] for d in inputs]
        for dataset in ('simultaneous_eeg_nirs','refed'):
            common=sorted(k for k in aa.keys()&bb.keys() if k[0]==dataset)
            key=common[len(common)//2]
            fig,axs=plt.subplots(2,2,figsize=(12,5.6),sharex=True,sharey='row',layout='constrained')
            ch=1 if m=='eeg' else 2; hz=200 if ch==1 else 10
            span=slice(14*hz,16*hz) if ch==1 else slice(0,30*hz)
            for j,(v,ex) in enumerate(zip(VERSIONS,(aa[key],bb[key]))):
                for k,branch in enumerate('SO'):
                    ax=axs[k,j];mask=ex['mask'].reshape(-1,ch)[span]
                    y=ex[branch.lower()].reshape(-1,ch)[span];p=ex['prediction_'+branch].reshape(-1,ch)[span]
                    for h in range(ch):
                        color=['#205d89','#bd5134'][h];name='EEG' if ch==1 else ['HbO','HbR'][h]
                        t=np.arange(30*hz)[span]/hz
                        ax.plot(t,np.where(mask[:,h],y[:,h],np.nan),c=color,lw=1.15,label=name+' 目标')
                        ax.plot(t,np.where(mask[:,h],p[:,h],np.nan),c=color,lw=1,ls='--',label=name+' 重建')
                    rmse=np.sqrt(np.mean((p[mask]-y[mask])**2))
                    ax.set_title(f'{v}  {m.upper()}-{branch}　显示区间 RMSE {rmse:.3f}')
                    if j==0:ax.set_ylabel('幅度 / 本版训练尺度')
                    if k==0:ax.legend(ncol=2,fontsize=8)
                    if k==1:ax.set_xlabel('30 秒裁片内时间（秒）')
            fig.suptitle(f'{DATASETS[dataset]} 验证示例　被试 {key[1]} · {key[2]} · {key[-1]}　｜　每版预测对照本版监督目标')
            save(fig,out,'waveform_'+m+'_'+dataset)
            notes.append((m,dataset,str(key)))
    return notes


def report(out,records,counts,book,notes):
    def figure(name,caption):
        data=base64.b64encode((out/(name+'.png')).read_bytes()).decode()
        return f'<figure><img src="data:image/png;base64,{data}"><figcaption>{caption}</figcaption></figure>'
    def table(head,rows):
        return '<table><thead><tr>'+''.join(f'<th>{v}</th>' for v in head)+'</tr></thead><tbody>'+''.join(
            '<tr>'+''.join(f'<td>{v}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
    def fmt(xs):
        return ' → '.join(f'{x:.3f}' for x in xs)
    text=['''<!doctype html><html lang="zh"><meta charset="utf-8"><title>S/O 缓存版本对比</title>
<style>body{max-width:1040px;margin:38px auto;padding:0 22px;color:#1b2c38;font:16px/1.65 system-ui,sans-serif}h1,h2{line-height:1.3}h2{margin-top:40px}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:7px 10px;border-bottom:1px solid #c6d1d8}th{background:#edf3f5}figure{margin:22px 0}img{width:100%;height:auto}figcaption,small{color:#415866}</style>
<h1>更换 S/O 监督目标后，EEG 重建误差上升，Hb 变化较小</h1>
<p>四个 waveform 主模型均完成 10,000 步。模型以观测 X 为输入，从 S、O 两个分支重建各自的教师分量，两者之和重建 X。连续模型提供无硬量化对照，K128 模型每个分支用 128 个码字。两轮模型结构、训练日程、种子 17、验证被试及裁片方案相同。v2 使用原生 EEG 教师目标再转换到 200 Hz；v3 使用全记录 1–45 Hz、200 Hz EEG 重新拟合区域模型和生成目标。两版各自拟合训练尺度并独立训练。表中误差和相关均对照该版目标。</p>
<p>额外的 EEG dense-long K128 对照在 v3 第 7,000 步评价中有 3/384,000 个数值超出整数解码重放容差，首次超差记录中最大绝对差为 2.249×10⁻⁶。该臂缺少新版本 10,000 步分数。v2 原轮也在该阶段停止，后续单独恢复到 10,000 步。该辅助臂最后一个双方完整的检查点为 4,000 步，其各版完整验证集宏 NMSE 从 0.413 升至 0.656。以下比较使用两版均完成的四个 waveform 模型。</p>
<h2>验证人口与重建</h2><p>下表仅比较共有的位置与裁片。每项输入由数据集、被试、记录、父窗口、30 秒裁片起点和位置名称标识。每个版本使用保留的有效时间掩码与本版训练方差。NMSE 为有效点均方误差除以相应的训练目标方差。先在位置、裁片、记录、被试和数据集各层内等权平均，Hb 的 HbO/HbR 最后等权。相关为同一有效波形上的 Pearson 相关，排除未定义项。每个模态均有四库、20 名独立验证被试。</p>''']
    text.append(table(['模态','v2 位置×裁片','v3 位置×裁片','共同位置×裁片','双方完整的 2 秒片段'],[[c['modality'].upper(),*[f"{c[k]:,}" for k in ('old','new','paired','full_patch_pairs')]] for c in counts]))
    def pair(m,arm,c,field):
        return [next(r[field] for r in records if (r['version'],r['modality'],r['arm'],r['component'],r['group'])==(v,m,arm,c,'macro')) for v in VERSIONS]
    data=[]
    for m in MODALITIES:
        for arm in ('continuous','vq128'):
            macro=np.mean([pair(m,arm,c,'nmse') for c in 'XSO'],axis=0)
            data.append([m.upper(),'连续' if arm=='continuous' else 'K128',fmt(macro),*[fmt(pair(m,arm,c,'nmse')) for c in 'XSO']])
    text.append(table(['模态','模型','宏 NMSE · v2 → v3','X','S','O'],data))
    text.append(figure('reconstruction','每条柱是一个数据集的被试等权结果；正值表示 v3 对本版目标的重建误差更高。绝对变化的单位是 NMSE。'))
    text.append('<p>三个数据集的 EEG 宏误差升高；REFED 的 X、S、O 误差均略降。以下展示 Simultaneous 和 REFED 的固定样例，便于核对方向不同的两组观测。Hb-O 在四个数据集的误差均稍有升高。</p>')
    for m,dataset,note in notes:
        text.append(figure('waveform_'+m+'_'+dataset,'来自训练程序预先保存的固定验证示例，取该数据集两版共有位置按身份排序的中间项。实线是本版教师目标，虚线是 K128 重建。EEG 放大中央 2 秒，Hb 展示完整 30 秒，RMSE 按显示范围有效点和颜色汇总。样例身份 '+note))
    text.append('<h2>一个 token 能解释多少局部波形？</h2><p>每个 30 秒输入含 15 个 2 秒片段。训练阶段按分支、码字平均去除片内均值后的监督波形，得到单 ID 原型。验证时只给出本片 ID，用对应训练原型预测去均值后的真值。零形状基线在同样的有效位置预测 0。两者按上述层级聚合，误差降幅为 1 − 单 ID 原型 NMSE / 零形状 NMSE。它衡量单码能解释的形状，与完整 15 码解码器的重建能力分别计分。</p>')
    data=[]
    for m in MODALITIES:
        for br in 'SO':
            h=[next(r for r in book if (r['version'],r['modality'],r['branch'])==(v,m,br)) for v in VERSIONS]
            data.append([m.upper()+'-'+br,fmt(pair(m,'vq128',br,'correlation')),fmt(pair(m,'vq128',br,'single_ID_nmse')),fmt(pair(m,'vq128',br,'zero_shape_nmse')),
                         ' → '.join(f'{100*x:.1f}%' for x in pair(m,'vq128',br,'shape_gain')),f"{h[0]['paired_patch_ARI']:.3f}"])
    text.append(table(['分支','完整解码相关','单 ID NMSE','零形状 NMSE','单 ID 误差降幅','跨版本 ARI'],data))
    text.append(figure('token_metrics','有效码数为 exp(码字概率的熵)，按各版完整验证集内记录、被试及数据集等权的频率计算。两版四个码本各使用了 128/128 个 ID。ARI 在共同位置及双方均完整的 2 秒片段上比较码字分组，经机会校正并允许编号置换；1 表示分组一致，0 对应随机分组期望。ARI 按片段合并计算。'))
    text.append('<p>EEG 的连续和量化模型均出现重建退化，说明变化已出现在连续表示阶段。新版 EEG-O 的零形状误差增大，单 ID 原型虽然绝对误差更高，仍能解释更大比例的片内形状变化。Hb 两个分支的单码相对收益变化均小于 0.5 个百分点。</p>')
    for m in MODALITIES:
        text.append(figure('patterns_'+m,'每版每分支按训练集的层级等权使用率独立选前两码。实线是训练形状原型，虚线是所有分到该码的完整验证片段的均值，浅线为固定种子抽取的实际验证波形。Hb 两色各自去均值。纵轴单位为该版各位置训练 X 标准差；两列同排共用纵轴。两轮独立学习的 ID 只在本版内标识码字。'))
        text.append(figure('mean_patterns_'+m,'同一组高频码，纵轴缩放到两版平均形状的共同范围。两条线靠近表示该码在验证集中的平均波形与训练原型接近。单个片段的离散程度见上图。'))
    text.append('<h2>证据与核对</h2><p>各版 training_ID_prototypes.npz 与 validation.npz 提供训练原型、验证分配及逐输入指标。按冻结的 G0 验证数组独立重算单 ID 误差，与保存值逐输入核对。定量比较按相同位置×裁片配对；训练原型、示例和码本有效数均保留其各自注明的支持范围。本次为单种子描述性比较。</p>')
    for v in VERSIONS:
        text.append('<p>'+v+' · '+str(RUN[v].relative_to(SOURCE))+'/models/{eeg,hb}__waveform__{continuous,vq128}/evaluation/step_10000/</p>')
    (out/'index.html').write_text('\n'.join(text)+'</html>')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();out=args.output.resolve()
    if out.parent!=RUN['v3'] or (out/'index.html').exists():
        parser.error('use a fresh direct child of the completed v3 training run')
    out.mkdir(exist_ok=True)
    configs=[read_json(r/'resolved_config.json')['config'] for r in RUN.values()]
    for field in ('model','architectures','tasks','training','evaluation'):
        assert configs[0][field]==configs[1][field]
    with ProcessPoolExecutor(2) as pool:
        inputs=list(pool.map(collect,VERSIONS))
    records,book,counts=summarize(inputs)
    write_csv(out/'metrics.csv',records);write_csv(out/'codebooks.csv',book);write_csv(out/'support.csv',counts)
    notes=plots(out,records,inputs,book)
    report(out,records,counts,book,notes)


if __name__=='__main__':
    main()
