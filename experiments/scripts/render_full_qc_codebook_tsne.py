"""t-SNE of fitted assignment vectors, with held-subject observed attributes per emitted ID."""
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from apply_joint_tokenizer import checked_path
from render_full_qc_tokenizer_examples import DATASETS, EVAL, word_batches
from retrain_joint_qc_tokenizer import OUT, ROOT, configuration
from src.data.dual_component_tokenizer import base_tokenizer_training_data

BANDS = ['1–4','4–8','8–13','13–30','30–45']
PRIMARY = (30, 17)
CHECK = (50, 43)
K = 10


def read(path):
    return json.loads(path.read_text())


def load_books():
    bundle = OUT/'bundle.json'
    spec = read(bundle)
    books = []
    with np.load(checked_path(bundle, spec['eeg'])) as z:
        for branch in 'SO':
            for stage, offset, capacity, mode in [('pilot',1,1023,'balanced'),('residual',5116,2047,'innovation')]:
                prefix = 'pilot' if stage == 'pilot' else 'second'
                for role in range(5):
                    center = z[f'{prefix}_{branch}_{role}_{mode}_centers']
                    books.append((f'eeg_{branch}_{stage}_{role}', 'eeg', branch, stage, role,
                                  offset+role*capacity, center))
    hb = checked_path(bundle, spec['hb'])
    for _ in range(2):
        meta = torch.load(hb, map_location='cpu', weights_only=False)['metadata']
        path = hb.parent/meta['source_file']
        assert path.parent.resolve()==hb.parent.resolve()
        with path.open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest()==meta['source_sha256']
        hb = path
    state = torch.load(hb, map_location='cpu', weights_only=False)['model_state']
    for branch in 'SO':
        center = state[f'branches.{branch}.quantizer.codebook'].numpy()
        books.append((f'hb_{branch}', 'hb',branch,'ordinary',-1,0,center))
    assert len(books)==22 and sum(len(x[-1]) for x in books)==31084
    return books


def knn(values):
    return NearestNeighbors(n_neighbors=K,metric='euclidean',n_jobs=1).fit(values).kneighbors(return_distance=False)


def overlap(left,right):
    return float((left[:,:,None]==right[:,None,:]).any(axis=2).mean())


def project(job):
    (name,mod,branch,stage,role,start,v),out = job
    dest = out/'geometry'/f'{name}.npz'
    if dest.exists():
        return
    t = time.monotonic()
    with threadpool_limits(limits=1):
        high = knn(v)
        maps = []
        neighbors = []
        kl = []
        for pp,seed in [PRIMARY,CHECK]:
            alg = TSNE(n_components=2,perplexity=pp,random_state=seed,init='pca',learning_rate='auto',
                       max_iter=1000,metric='euclidean',method='barnes_hut',n_jobs=1)
            xy = alg.fit_transform(v)
            maps.append(xy); neighbors.append(knn(xy)); kl.append(alg.kl_divergence_)
    np.savez_compressed(dest,primary=maps[0],sensitivity=maps[1],knn_high=high,
                        id=np.arange(start,start+len(v)),kl=kl,
                        recall=[overlap(high,n) for n in neighbors],between_maps=overlap(*neighbors),
                        runtime=time.monotonic()-t)
    print(name,len(v),round(time.monotonic()-t,1),flush=True)


def geometry(out,workers):
    (out/'geometry').mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(project, [(b,out) for b in load_books()]))


def profile_features(waves,mod):
    if mod=='eeg':
        tape = np.hanning(400)
        power = np.abs(np.fft.rfft((waves-waves.mean(1,keepdims=True))*tape,axis=-1))**2
        # FFT bin width is 0.5 Hz. Ratios cancel the common Hann/one-sided PSD factor.
        total=power[:,2:91].sum(1)
        fractions=np.divide(np.stack([power[:,8:16].sum(1),power[:,16:26].sum(1)],axis=1),total[:,None],
                            out=np.full((len(total),2),np.nan),where=total[:,None]>0)
        return np.column_stack([100*fractions,np.sqrt(np.square(waves).mean(1))])
    wave=waves.reshape(-1,20,2)
    t=(np.arange(20)-9.5)/10
    slope=np.einsum('ntc,t->nc',wave,t/(t@t))
    return np.column_stack([slope,wave.mean(1)])


def profiles(out):
    cfg=configuration()
    admitted={tuple(v) for v in cfg['data']['validation_subjects']}
    manifest,scales,_,_=base_tokenizer_training_data(cfg,ROOT)
    assert len(admitted)==16
    rows=manifest['rows']
    for mod in ['eeg','hb']:
        names=['theta_pct','alpha_pct','rms'] if mod=='eeg' else ['hbo_slope','hbr_slope','hbo_level','hbr_level']
        for split in ['train','validation']:
            people=sorted({(r['dataset'],r['subject']) for r in rows if r['split']==split and
                           (split=='train' or (r['dataset'],r['subject']) in admitted)})
            npeople=len(people)
            person_index={p:i for i,p in enumerate(people)}
            assert npeople==(60 if split=='train' else 16)
            sizes={'S':15351,'O':15351} if mod=='eeg' else {'S':129,'O':257}
            counts={b:np.zeros((npeople,size),dtype=np.int32) for b,size in sizes.items()}
            attrs=['X_'+k for k in names]+['branch_'+k for k in names]
            sums={b:np.zeros((npeople,size,len(attrs))) for b,size in sizes.items()}
            finite={b:np.zeros((npeople,size,len(attrs)),dtype=np.int32) for b,size in sizes.items()}
            for ds,take,batch,ids in word_batches(rows,scales,admitted,mod,split):
                valid=batch['mask'].all(2)
                subjects=np.asarray([person_index[ds,rows[i]['subject']] for i,_,_ in take])
                person=np.broadcast_to(subjects[:,None],valid.shape)[valid]
                obs=profile_features(batch['x'][valid],mod)
                for b in 'SO':
                    code=ids[b][valid].astype(int)
                    assert np.all(code<sizes[b])
                    # Keep missing/silent sentinel occupancy separately from fitted codebook rows.
                    vals=np.column_stack([obs,profile_features(batch[b.lower()][valid],mod)])
                    np.add.at(counts[b],(person,code),1)
                    np.add.at(sums[b],(person,code),np.nan_to_num(vals))
                    np.add.at(finite[b],(person,code),np.isfinite(vals).astype(np.int32))
            for b in 'SO':
                pc=counts[b];count=finite[b]
                mean=np.divide(sums[b],count,out=np.full_like(sums[b],np.nan),where=count>0)
                nvalid=np.sum(count>0,axis=0)
                avg=np.divide(np.nansum(mean,axis=0),nvalid,out=np.full_like(mean[0],np.nan),where=nvalid>0)
                table=pd.DataFrame(avg,columns=attrs)
                table['id']=np.arange(sizes[b]);table['patches']=pc.sum(0);table['people']=(pc>0).sum(0)
                table['datasets']=sum((pc[[p[0]==ds for p in people]].sum(0)>0).astype(int) for ds in DATASETS)
                for i,k in enumerate(attrs):
                    table[k+'_people']=nvalid[:,i]
                    table[k+'_patches']=count[:,:,i].sum(0)
                assert table.patches.sum()==pc.sum()
                table.to_csv(out/f'{mod}_{b}_{split}.csv',index=False)
            print(split,mod,'profiles complete',flush=True)


def supported(frame,field,split):
    return ((frame[f'{split}_{field}_patches']>=8)&(frame[f'{split}_{field}_people']>=2)
            &np.isfinite(frame[f'{split}_{field}']))


def summarize(out):
    tables = []
    records = []
    features = []
    reserved = []
    books = load_books()
    for name,mod,b,stage,role,offset,vec in books:
        with np.load(out/'geometry'/f'{name}.npz') as z:
            ids=z['id']; xy=z['primary']; alt=z['sensitivity']
            record=dict(book=name,modality=mod,branch=b,stage=stage,role=role,n=len(ids),
                        recall10=z['recall'][0],check_recall10=z['recall'][1],map_overlap10=float(z['between_maps']),
                        chance10=K/(len(ids)-1),kl_main=z['kl'][0],kl_check=z['kl'][1])
        assert np.array_equal(ids,np.arange(offset,offset+len(vec)))
        frame=pd.DataFrame(dict(book=name,modality=mod,branch=b,stage=stage,role=role,id=ids,
                                x=xy[:,0],y=xy[:,1],check_x=alt[:,0],check_y=alt[:,1]))
        for split in ['train','validation']:
            all_words=pd.read_csv(out/f'{mod}_{b}_{split}.csv').set_index('id')
            frame=frame.join(all_words.add_prefix(split+'_'),on='id',validate='one_to_one')
            if role in (-1,0) and stage!='residual':
                flag=0 if mod=='eeg' else {'S':128,'O':256}[b]
                reserved.append(dict(modality=mod,branch=b,id=flag,split=split,patches=int(all_words.loc[flag,'patches']),
                                     people=int(all_words.loc[flag,'people'])))
        assert frame.validation_patches.notna().all()
        fields=(['theta_pct','alpha_pct','rms'] if mod=='eeg' else ['hbo_slope','hbr_slope','hbo_level','hbr_level'])
        for target in ['X','branch']:
            for f in fields:
                field=target+'_'+f
                good=(supported(frame,field,'train')&supported(frame,field,'validation')).to_numpy()
                a=frame.loc[good,'train_'+field].to_numpy()
                y=frame.loc[good,'validation_'+field].to_numpy()
                stat=dict(book=name,modality=mod,branch=b,stage=stage,role=role,attribute=field,n=len(y),
                          total=len(ids),train_dev_spearman=np.nan,neighbor_random_ratio=np.nan)
                if len(y)>K and np.ptp(y)>0 and np.ptp(a)>0:
                    pairs=knn(vec[good])
                    observed=np.abs(y[:,None]-y[pairs]).mean()
                    # Exact mean absolute difference of all distinct unordered pairs.
                    ordered=np.sort(y)
                    random_mean=(2*np.dot(2*np.arange(len(y))+1-len(y),ordered))/(len(y)*(len(y)-1))
                    stat.update(train_dev_spearman=spearmanr(a,y).statistic,
                                neighbor_random_ratio=observed/random_mean)
                features.append(stat)
        primary_attribute = 'X_alpha_pct' if mod == 'eeg' else 'X_hbo_slope'
        frame['supported'] = supported(frame, primary_attribute, 'train') & supported(frame, primary_attribute, 'validation')
        record.update(supported=int(frame.supported.sum()),dev_used=int((frame.validation_patches>0).sum()),
                      dev_patches=int(frame.validation_patches.sum()),
                      supported_dev_patches=int(frame.loc[frame.supported,'validation_patches'].sum()))
        tables.append(frame)
        records.append(record)
    words=pd.concat(tables,ignore_index=True)
    assert len(words)==31084 and not words.duplicated(['modality','branch','id']).any()
    words.to_csv(out/'word_map.csv.gz',index=False)
    checks=pd.DataFrame(records)
    stats=pd.DataFrame(features)
    checks.to_csv(out/'geometry_checks.csv',index=False)
    stats.to_csv(out/'attribute_checks.csv',index=False)
    pd.DataFrame(reserved).to_csv(out/'special_ids.csv',index=False)
    return words,checks,stats


def figures(out,words):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import colors as mc
    plt.rcParams.update({'font.family':['Noto Sans CJK JP','DejaVu Sans'],'font.size':11,
                         'axes.unicode_minus':False,'text.color':'#193545'})
    chosen=read(OUT/'reconstruction_and_words_v1/examples.json')['patterns']
    examples={(('eeg' if 'eeg' in p['file'] else 'hb'),w['branch'],w['ID']) for p in chosen for w in p['words']}

    detail={(v['book'],v['id']) for v in read(out/'selected_members.json')}
    def draw(ax,fr,field,norm,cmap,size,split='validation'):
        x,y=fr.x,fr.y
        if field:
            mask=supported(fr,field,'train')&supported(fr,field,'validation')
            vals=fr[split+'_'+field]
        else:
            mask=fr.validation_patches>0
            vals=np.log10(fr.validation_patches.clip(lower=1))
        ax.scatter(x[~mask],y[~mask],s=size*.60,c='#cdd1d4',linewidths=0)
        ax.scatter(x[mask],y[mask],s=size,c=vals[mask],cmap=cmap,norm=norm,linewidths=0)
        for r in fr.itertuples():
            linked=(r.book,r.id) in detail
            if linked or (r.modality,r.branch,r.id) in examples:
                ax.scatter(r.x,r.y,s=size*3+24,marker=('s' if linked else 'o'),facecolors='none',edgecolors='#141b23',linewidths=.9)
                ax.annotate(str(r.id),(r.x,r.y),xytext=(4,4 if r.id not in (81,50) else -11),textcoords='offset points',
                            fontsize=7,bbox=dict(boxstyle='round,pad=0.1',fc='white',ec='none',alpha=.9))
        ax.margins(.12)
        ax.set_aspect('equal',adjustable='datalim')
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_color('#dce1e2')
        return int(mask.sum())

    def bar(fig,norm,cmap,label,extend):
        cax=fig.add_axes([.34,.086,.32,.014])
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',extend=extend)
        cb.ax.set_xlabel(label,fontsize=10,labelpad=2)
        cb.outline.set_visible(False)

    e=words[words.modality=='eeg']
    for field,tag,title,unit in [
        ('X_alpha_pct','alpha','α 功率占比','8–13 Hz 功率 / 1–45 Hz 功率（%）'),
        ('X_theta_pct','theta','θ 功率占比','4–8 Hz 功率 / 1–45 Hz 功率（%）'),
        ('X_rms','rms','局部振幅','2 秒 RMS（位置训练 X 标准差）'),
        (None,'occupancy','实发词的使用频数','完整开发片段数的 log₁₀')]:
        if field:
            good=supported(e,field,'train')&supported(e,field,'validation')
            maximum=float(np.quantile(e.loc[good,'validation_'+field],.98))
        else:maximum=float(np.log10(e.validation_patches.max()))
        norm=mc.Normalize(0,maximum)
        fig,axes=plt.subplots(4,5,figsize=(14.5,10.4))
        fig.subplots_adjust(left=.091,right=.98,bottom=.137,top=.845,wspace=.08,hspace=.19)
        fig.text(.025,.949,'EEG 码本 t-SNE  ·  '+title,fontsize=22,weight='bold')
        subtitle=('每点一个赋码中心，颜色为该词开发成员的观测 X 属性，被试等权。灰点的训练或开发支持不足；色阶上端取受支持词的第 98 百分位。' if field else
                  '所有实发词按开发集完整 2 秒片段数着色；灰点在本开发审计中没有完整成员。各词点面积相同。')
        fig.text(.026,.898,subtitle,fontsize=11)
        for i,(b,stage) in enumerate([(b,s) for b in 'SO' for s in ['pilot','residual']]):
            for j,band in enumerate(BANDS):
                fr=e[(e.branch==b)&(e.stage==stage)&(e.role==j)]
                n=draw(axes[i,j],fr,field,norm,'viridis',5.8)
                if i==0:axes[i,j].set_title(band+' Hz',fontsize=13,color='#245963')
                axes[i,j].text(.025,.015,f'{n:,}/{len(fr):,}',transform=axes[i,j].transAxes,fontsize=8,
                               bbox=dict(fc='white',alpha=.88,ec='none',pad=.8))
            pos=axes[i,0].get_position()
            fig.text(.022,pos.y0+pos.height*.55,f'EEG-{b}\n'+('导引词' if stage=='pilot' else '残差词'),fontsize=13,weight='bold')
        bar(fig,norm,'viridis',unit,'max' if field else 'neither')
        fig.text(.025,.017,'各框独立投影；坐标无单位。角标为着色词数 / 该库全部中心数。方框词见观测 X 样例，圆圈词见分量成员图。横排为编码角色的五个频带，颜色来自完整观测片段。',fontsize=10)
        fig.savefig(out/f'eeg_{tag}.png',dpi=210,facecolor='white');plt.close(fig)

    h=words[words.modality=='hb']
    for metric,tag,unit in [('slope','slopes','斜率（位置训练 X 标准差 / 秒）'),('level','levels','片内水平（位置训练 X 标准差）')]:
        fields=['X_hbo_'+metric,'X_hbr_'+metric]
        both=np.concatenate([h.loc[supported(h,f,'train')&supported(h,f,'validation'),'validation_'+f] for f in fields])
        scale=float(np.quantile(abs(both),.98)); norm=mc.Normalize(-scale,scale)
        fig,axes=plt.subplots(2,2,figsize=(10.8,8.2))
        fig.subplots_adjust(left=.12,right=.96,bottom=.18,top=.81,wspace=.13,hspace=.12)
        fig.text(.04,.944,'fNIRS 码本 t-SNE  ·  双色'+('变化方向' if metric=='slope' else '片内水平'),fontsize=21,weight='bold')
        fig.text(.04,.888,'每行的两个面板复用相同词坐标。颜色来自实际赋码的开发集观测 X，每名被试等权。',fontsize=11)
        for i,b in enumerate('SO'):
            for j,f in enumerate(fields):
                fr=h[h.branch==b]
                n=draw(axes[i,j],fr,f,norm,'RdBu_r',22)
                if i==0:axes[i,j].set_title(['HbO 氧合','HbR 脱氧'][j],weight='bold',color='#245963')
                axes[i,j].text(.02,.025,f'{n}/{len(fr)}',transform=axes[i,j].transAxes,fontsize=9)
            pos=axes[i,0].get_position();fig.text(.032,pos.y0+pos.height*.5,'Hb-'+b,weight='bold',fontsize=15)
        bar(fig,norm,'RdBu_r',unit,'both')
        fig.text(.035,.018,'蓝色为负，红色为正。以零为中心，色阶限于双色绝对值 P98。灰点支持不足；方框词见 X 样例，圆圈词见分量成员图。S/O 两库独立投影。',fontsize=9)
        fig.savefig(out/f'hb_{tag}.png',dpi=225,facecolor='white');plt.close(fig)


def selected_member_figure(out,words):
    import matplotlib.pyplot as plt
    cfg=configuration()
    allowed={tuple(p) for p in cfg['data']['validation_subjects']}
    manifest,scales,_,_=base_tokenizer_training_data(cfg,ROOT)
    fig,axes=plt.subplots(2,3,figsize=(13,7.1))
    fig.subplots_adjust(left=.08,right=.97,top=.78,bottom=.17,hspace=.48,wspace=.28)
    fig.text(.035,.94,'用实际观测片段复核图中的颜色',fontsize=22,weight='bold')
    fig.text(.035,.875,'每行各取一个 EEG 词和一个 fNIRS 词；按训练人群的属性选词，曲线仅使用开发人群。',fontsize=12)
    summary=[]
    for mod,branch,book,attr in [('eeg','O','eeg_O_pilot_2','X_alpha_pct'),('hb','S','hb_S','X_hbo_slope')]:
        group=words[words.book==book]
        good=group[supported(group,attr,'train')&supported(group,attr,'validation')].copy()
        codes=[int(good.loc[(good['train_'+attr]-good['train_'+attr].quantile(q)).abs().idxmin(),'id']) for q in [.2,.8]]
        chosen={b:(codes if b==branch else []) for b in 'SO'}
        data={code:[] for code in codes}
        for ds,take,batch,ids in word_batches(manifest['rows'],scales,allowed,mod,'validation',chosen):
            valid=batch['mask'].all(2)
            for code in codes:
                i,slot=np.where(valid&(ids[branch]==code))
                for ix,t in zip(i,slot):
                    row=manifest['rows'][take[ix,0]]
                    data[code].append(((ds,row['subject']),batch['x'][ix,t]))
        for i,code in enumerate(codes):
            per={}
            for person,wave in data[code]:
                per.setdefault(person,[]).append(wave)
            primitive=[]
            for arr in per.values():
                y=np.array(arr,float)
                if mod=='eeg':
                    f=np.abs(np.fft.rfft((y-y.mean(1,keepdims=True))*np.hanning(400)))[:,2:91]**2
                    den=f.sum(1)
                    primitive.append(200*f[den>0]/den[den>0,None])
                else:
                    a=y.reshape(-1,20,2);primitive.append(a-a.mean(1,keepdims=True))
            primitive=[v for v in primitive if len(v)]
            mean=np.mean([a.mean(0) for a in primitive],axis=0)
            lo,hi=np.percentile(np.concatenate(primitive),[10,90],axis=0)
            rec=good[good.id==code].iloc[0]
            if mod=='eeg':
                np.testing.assert_allclose(.5*mean[14:24].sum(),rec['validation_'+attr],rtol=3e-5)
            else:
                t=(np.arange(20)-9.5)/10
                np.testing.assert_allclose(mean[:,0]@t/(t@t),rec['validation_'+attr],atol=1e-7)
            summary.append(dict(book=book,id=code,training_quantile=[.2,.8][i],attribute=attr,
                                train=rec['train_'+attr],development=rec['validation_'+attr],
                                subjects=len(primitive),patches=sum(map(len,primitive))))
            for j in ([0] if mod=='eeg' else [1,2]):
                ax=axes[i,j]
                if mod=='eeg':
                    t=np.arange(2,91)*.5;av=mean;lower,upper=lo,hi
                    ax.axvspan(8,13,color='#dec583',alpha=.23)
                    label='EEG-O  ·  α 导引词'
                else:
                    t=np.arange(20)/10;av=mean[:,j-1];lower,upper=lo[:,j-1],hi[:,j-1]
                    label='Hb-S  ·  '+['HbO','HbR'][j-1]
                ax.fill_between(t,lower,upper,color='#5795ab',alpha=.23)
                ax.plot(t,av,color='#125575',lw=2)
                ax.set_title(label+f'  ·  ID {code}',loc='left',fontsize=12)
                ax.text(.98,.96,f'{sum(map(len,primitive)):,} 片 / {len(primitive)} 人',ha='right',va='top',transform=ax.transAxes,fontsize=9)
                ax.set_ylabel('相对功率谱（%/Hz）' if mod=='eeg' else '去均值幅度（训练 SD）',fontsize=10)
                ax.set_xlim(t[0],t[-1]);ax.tick_params(labelsize=9)
                if mod=='eeg':ax.set_yscale('log')
                if i==1:ax.set_xlabel('频率（Hz）' if mod=='eeg' else '片内时间（s）')
                ax.grid(axis='y',alpha=.15)
    for j in range(3):
        lim=[ax.get_ylim() for ax in axes[:,j]]
        for ax in axes[:,j]:ax.set_ylim(min(v[0] for v in lim),max(v[1] for v in lim))
    fig.text(.04,.057,'上 / 下排分别最接近本子库训练属性的 P20 / P80。EEG 按 α 占比选词，fNIRS 按 HbO 斜率选词，HbR 沿用同一个 ID。',fontsize=10)
    fig.text(.04,.022,'实线先对每人取平均再跨人等权；浅色带为全部实际片段逐点 P10–P90。EEG 的阴影频带为 8–13 Hz。',fontsize=10)
    fig.savefig(out/'observed_examples.png',dpi=210,facecolor='white');plt.close(fig)
    return summary


def report(out,words,geometry_stats,attributes):
    import html
    from shutil import copyfile
    from zipfile import ZipFile, ZIP_DEFLATED
    from playwright.sync_api import sync_playwright
    import sklearn

    plots=['eeg_alpha','hb_slopes','observed_examples','eeg_theta','eeg_rms','eeg_occupancy','hb_levels']
    geom=geometry_stats.set_index('book')
    e=geometry_stats.query("modality=='eeg'")
    h=geometry_stats.query("modality=='hb'")
    members=read(out/'selected_members.json')
    def link(url,txt):
        return f'<a href="{html.escape(url,quote=True)}">{html.escape(txt)}</a>'
    def external(url,txt):
        return link(url,txt)
    def row(*items):
        return '<tr>'+''.join('<td>'+c+'</td>' for c in items)+'</tr>'
    def table(headers,data):
        return '<table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(data)+'</tbody></table>'
    studies=table(['研究','论文中实际分析的对象','本图采用的做法'],[
        row(external('https://arxiv.org/pdf/2506.01867','Ogg 等 · 图 1'),'将未用于预训练的 EEG 试次送入模型，对时间上各 Transformer 表示取均值，再用 t-SNE 按被试或任务着色。','明确每点单位。本图每点是一个词；同一词在多个人和位置的观测用于估计颜色。'),
        row(external('https://openreview.net/pdf/3c93628467752de1ab3f68d8ee45ae35fca06aed.pdf','CodeBrain · §3.5 图 4'),'在睡眠数据上把时间词和频率词映回实际 EEG 片段，比较慢波、频谱及两类码的不同组合。','让图上选定的词能追溯至实际开发成员，以谱或双色去均值波形检验颜色差异。'),
        row(external('https://arxiv.org/html/2510.09095v1#S4','BioCodec · §4.3、图 2–3'),'分残差量化层显示词频与利用率，并比较 EEG 通道相关矩阵和量化表示的相关矩阵。','分开编码阶段，另画实际词频；在原始向量空间计算近邻属性差。'),
        row(external('https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf','LaBraM · §2.2、图 2'),'用离散神经词重建 EEG 片段的傅里叶幅度与相位。','优先检查实际成员的频谱属性，并同时记录 RMS 以区别频谱构成与振幅。'),
        row(external('https://arxiv.org/abs/2610.17845','GeoNIRS · 方法与消融'),'预训练联合建模源探位置，并在 HbO、HbR 间双向重建。','固定同一套词坐标，分别看双色方向及水平；解剖位置的解释需回到源探对应关系。')])
    def agg(mod,b,f):
        return attributes.query('modality==@mod and branch==@b and attribute==@f')
    def panel_stats(mod,b,f):
        part=agg(mod,b,f)
        return part.train_dev_spearman.median(),part.neighbor_random_ratio.median()
    stat_table=[]
    for mod,f,attr in [('eeg','X_alpha_pct','EEG α 占比'),('eeg','X_theta_pct','EEG θ 占比'),('eeg','X_rms','EEG RMS'),
                       ('hb','X_hbo_slope','fNIRS HbO 斜率'),('hb','X_hbr_slope','fNIRS HbR 斜率')]:
        s,o=[panel_stats(mod,b,f) for b in 'SO']
        stat_table.append(row(attr,f'{s[0]:.3f} / {o[0]:.3f}',f'{s[1]:.3f} / {o[1]:.3f}'))
    stable=table(['观测属性','训练–开发词序相关 ρ（S / O）','原空间近邻 / 随机词属性差（S / O）'],stat_table)
    eg=e.supported.sum();hg=h.supported.sum()
    first=f"""<p>在 30,700 个 EEG 中心和 384 个 fNIRS 普通词上，分别拟合 22 张 t-SNE。EEG 的分支、编码阶段及频带拥有各自拟合的投影基和特征尺度；fNIRS 两支也独立学习 64 维向量。每框仅比较一个子码本内部的几何关系。</p>
    <p>EEG 有 <b>{eg:,} / {e.n.sum():,}</b> 个中心达到 α 占比着色门槛，占普通码全部完整开发赋码的 <b>{100*e.supported_dev_patches.sum()/e.dev_patches.sum():.1f}%</b>。fNIRS 有 <b>{hg} / 384</b> 个中心可按 HbO 斜率着色。模型有向量但支持不足的词仍以灰点显示。</p>
    <h2>相关工作的可借鉴部分</h2>{studies}
    <p>Ogg 等的图回答试次表示如何组织；CodeBrain 的方法更适合查验某个词对应哪些实际波形。这里将“词的几何位置、发生频数、观测属性和组内离散程度”分别展示。t-SNE 的簇间空白、边缘、面积和跨框方向不提供生理距离刻度。{external('https://distill.pub/2016/misread-tsne/','Wattenberg 等的 t-SNE 读图实验')}</p>"""
    methods=f"""<h2>点、颜色和分母</h2>
    <p>EEG 使用实际赋码的 balanced 导引中心、innovation 残差中心；Hb 使用 EMA 码本。直接采用模型使用的欧氏坐标，不再按维重标度。每词一票，不按出现频数重复投影。EEG 缺测 ID 0 和 Hb 两个静默 ID（S 128、O 256）按特殊接口单列；两支静默码对应同一批完整开发片段，各有 1,782 次使用。</p>
    <p>使用冻结 QC 面板的 60 名训练被试及 16 名普通开发被试。只计 2 秒内完整的评分位置。每词先在同一被试内对片段属性取均值，再在有成员的被试间等权。训练与开发两边均至少 8 个可计算片段、至少 2 人贡献才着色。这是显示门槛。不同词的被试、数据集和位置组成仍可不同。</p>
    <p>EEG 每片先去均值、乘 Hann 窗，分别取 [4,8) 或 [8,13) Hz 功率占 [1,45] Hz 功率的百分比；分母为零时该属性缺失。振幅读原片 2 秒 RMS。fNIRS 两色分别用 20 个 10 Hz 样点作带截距的时间线性拟合，斜率单位为训练 X 标准差/秒；水平为片内均值。主图颜色都取观测 X；逐词表还提供 S 或 O 监督分量的同类属性。</p>
    <h2>二维图能保留多少邻居？</h2>
    <p>scikit-learn {sklearn.__version__}，Barnes–Hut，欧氏距离，PCA 初始化，自适应学习率，最多 1,000 次迭代。主图 perplexity=30、seed=17；另用 50、43 联合复核。每词取 10 个近邻计算原空间与二维的交集比例。EEG 主图 20 库中位 {e.recall10.median():.1%}（{e.recall10.min():.1%}–{e.recall10.max():.1%}），另一组设置中位 {e.check_recall10.median():.1%}，两次二维近邻交集中位 {e.map_overlap10.median():.1%}。Hb-S / Hb-O 主图分别 {geom.loc['hb_S','recall10']:.1%} / {geom.loc['hb_O','recall10']:.1%}。随机基准为 10/(词数−1)。</p>
    {stable}
    <p>ρ 比较同一批受支持词在训练和开发中的属性排序。属性差为开发侧同属性词均值的绝对差，分子取每词在受支持原空间词中的 10 个近邻，分母取同库全部不同词对；小于 1 表示近邻属性更接近。EEG 按分支取十个子库的中位，Hb 每支一库。数字描述跨人群的词属性梯度；同一词仍容纳多种波形，也可能重复携带数据集、位置或振幅背景信息。</p>"""
    a,b,c,d=members
    captions={
       'eeg_alpha':'每个框对应同一分支、阶段和编码频带。即使都属于 α 导引词，观测 X 的 α 占比仍有高低。比较框内的局部渐变，跨框只比较同色标的数值。',
       'hb_slopes':'同一行左右严格复用坐标，可逐词对照 HbO、HbR。Hb-S 可见 HbO 斜率梯度；用同色标看 HbR，部分位置的变化较弱或方向不同。两个色团需分别读数。',
       'observed_examples':f"EEG-O 的 ID {a['id']} / {b['id']} 在训练中接近 α 占比的 P20 / P80；开发人群的实测占比分别为 {a['development']:.1f}% / {b['development']:.1f}%。Hb-S 的 ID {c['id']} / {d['id']} 在开发中的 HbO 斜率为 {c['development']:+.3f} / {d['development']:+.3f} SD/s。分位带显示每个词内部实际片段的差异。",
       'eeg_theta':'位置与 α 图相同。比较同一位置在两张图中的颜色，判断完整片段的 θ、α 构成如何变化。',
       'eeg_rms':'同坐标检查绝对振幅代理。词内 RMS 的训练–开发排序可重复，同时词号对背景已知时的额外振幅读出可能很弱；两个问题用不同统计量回答。',
       'eeg_occupancy':'颜色仅按开发集完整片段数。结合属性图识别未使用的中心、用量稀少的中心，以及已使用但跨人支持仍不足的中心。',
       'hb_levels':'同坐标检查片段平均水平。将这张图与斜率并读，可以区分一个词对应较高的基线，还是对应两秒内较强的上升趋势。'}
    labels=['EEG α 功率占比','fNIRS 双色斜率','观测 X 的真实成员','EEG θ 功率占比','EEG RMS','EEG 开发使用频数','fNIRS 双色水平']
    css="""@page{size:315mm 245mm;margin:9mm}*{box-sizing:border-box}body{font-family:'Noto Sans CJK SC',sans-serif;color:#193545;margin:24px auto;padding:0 18px;max-width:1250px;line-height:1.55}h1{font-size:27px}h2{font-size:20px;color:#164f56}p{font-size:15px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:7px 10px;text-align:left;vertical-align:top;border-bottom:1px solid #dbe3e5}th{background:#ecf2f2}a{color:#006c70}section{margin:28px 0}img{display:block;max-width:100%;height:auto}nav a{display:inline-block;margin:0 16px 5px 0}figcaption{font-size:15px}figure{margin:0}@media print{body{margin:0;max-width:none;padding:0}section{break-after:page;margin:0}section:last-child{break-after:auto}nav,.downloads{display:none}h1{margin:0 0 8px}img{max-height:198mm;max-width:100%;margin:auto}p{font-size:12pt}figcaption{margin-top:3mm}h2{margin:0 0 6px}}"""
    header="当前四分支 tokenizer 的词与生理特征"
    s='<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+header+'</title><style>'+css+'</style></head><body>'
    nav=' '.join(link('#'+k,v) for k,v in zip(plots,labels))
    downloads=' · '.join(link(f,t) for f,t in [('atlas.pdf','PDF 图册'),('word_map.csv.gz','逐词坐标与属性'),('geometry_checks.csv','二维近邻检查'),('attribute_checks.csv','训练–开发与原空间比较'),('special_ids.csv','特殊 ID 用量')])
    s+=f'<section><h1>{header}</h1><nav>{nav}</nav><p class="downloads">{downloads}</p>{first}</section>'
    for k,l in zip(plots,labels):
        s+=f'<section id="{k}"><figure><a href="{k}.png"><img src="{k}.png" alt="{l}"></a><figcaption>{captions[k]}</figcaption></figure></section>'
    s+='<section>'+methods+'</section></body></html>'
    (out/'index.html').write_text(s)
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        page=browser.new_page()
        page.goto((out/'index.html').resolve().as_uri(),wait_until='networkidle')
        page.evaluate('document.fonts.ready')
        assert page.locator('img').evaluate_all('(xs)=>xs.every(x=>x.complete&&x.naturalWidth>0)')
        page.pdf(path=str(out/'atlas.pdf'),prefer_css_page_size=True,print_background=True)
        browser.close()
    # Include the three original branch-target figures as companions to the circled codes.
    members_dir=OUT/'reconstruction_and_words_v1'
    for name in ('pattern_eeg_S.png','pattern_eeg_O.png','pattern_hb.png'):
        copyfile(members_dir/name,out/name)
    doc=out/'index.html'
    extra=' · '.join(link(name,title) for name,title in [('pattern_eeg_S.png','EEG-S 分量成员'),('pattern_eeg_O.png','EEG-O 分量成员'),('pattern_hb.png','Hb 分量成员')])
    doc.write_text(s.replace(downloads+'</p>',downloads+' · '+extra+'</p>'))
    specs=read(OUT/'bundle.json')
    record=dict(bundle=str((OUT/'bundle.json').relative_to(ROOT)),hashes={m:specs[m]['sha256'] for m in ('eeg','hb')},
                package=sklearn.__version__,primary=PRIMARY,check=CHECK,centers=len(words),books=len(geometry_stats),
                train_people=60,ordinary_development_people=16,confirmation_reads=0,test_reads=0,
                primary_mark='native encoding-space center',local_recall_k=K,support='>=8 finite complete patches and >=2 people in each split',
                code_units='Per-code attribute: within-person member mean then across-person equal mean. RNG/parameter check jointly changes perplexity and seed.',
                sources=['https://arxiv.org/pdf/2506.01867','https://openreview.net/pdf/3c93628467752de1ab3f68d8ee45ae35fca06aed.pdf','https://arxiv.org/html/2510.09095v1','https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf','https://arxiv.org/abs/2610.17845','https://distill.pub/2016/misread-tsne/'])
    (out/'methods.json').write_text(json.dumps(record,indent=2)+'\n')
    files=['index.html','atlas.pdf','word_map.csv.gz','geometry_checks.csv','attribute_checks.csv','special_ids.csv','methods.json','selected_members.json',
           'pattern_eeg_S.png','pattern_eeg_O.png','pattern_hb.png']+[k+'.png' for k in plots]
    with ZipFile(out/'codebook_tsne.zip','w',compression=ZIP_DEFLATED,compresslevel=2) as z:
        for f in files:z.write(out/f,f)


def render(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':['Noto Sans CJK JP','DejaVu Sans'],'axes.spines.right':False,
                         'axes.spines.top':False,'axes.unicode_minus':False})
    words,checks,attrs=summarize(out)
    selected=selected_member_figure(out,words)
    (out/'selected_members.json').write_text(json.dumps(selected,indent=2)+'\n')
    figures(out,words)
    report(out,words,checks,attrs)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['geometry','profiles','render'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.stage=='geometry':
        geometry(args.output,args.workers)
    elif args.stage=='profiles':
        profiles(args.output)
    else:
        render(args.output)
