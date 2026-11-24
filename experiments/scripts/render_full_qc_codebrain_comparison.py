"""Compare member waveforms and observed S/O pairs of the frozen EEG codec to CodeBrain Fig. 4."""
import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from render_full_qc_tokenizer_examples import DATASETS, EVAL
from retrain_joint_qc_tokenizer import ROOT, OUT, configuration, dump
from src.data.dual_component_tokenizer import base_tokenizer_batch, base_tokenizer_training_data

DATASET = 'simultaneous_eeg_nirs'
TASK = {'cnt_dsr': 'DSR', 'cnt_wg': 'WG', 'cnt_nback': 'N-back'}
BANDS = ['1–4', '4–8', '8–13', '13–30', '30–45']
FREQ = np.arange(2, 91) / 2
TIME = np.arange(400) / 200
PAPER = 'https://arxiv.org/html/2506.09110v1#S4.SS5'
SUBJECT = ['dataset', 'subject']


def role(code):
    return ('导引', (code-1)//1023) if code < 5116 else ('残差', (code-5116)//2047)


def spectrum(x):
    """Percent of total 1–45 Hz power per Hz. Hann, 200 Hz, 400 samples."""
    f = np.fft.rfft((x-x.mean(-1, keepdims=True))*np.hanning(400), axis=-1)[..., 2:91]
    power = np.square(abs(f))
    return 200*power/np.where(power.sum(-1, keepdims=True)>0, power.sum(-1, keepdims=True), np.nan)


def unit(x):
    y = x-x.mean(-1, keepdims=True)
    return y/np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), 1e-25)


def load_dataset(ds, split, rows, scales, admitted):
    with np.load(EVAL/f'eeg_{ds}_{split}.npz') as z:
        ex, s, o = (z[k] for k in ('examples', 'S', 'O'))
    for i in np.unique(ex[:, 0]):
        r = rows[i]
        assert (r['dataset'], r['modality'], r['split']) == (ds, 'eeg', split)
        assert split == 'train' or (ds, r['subject']) in admitted
    cache = OrderedDict()
    waves, frames = [], []
    dropped = 0
    for j in range(0, len(ex), 64):
        take = ex[j:j+64]
        batch = base_tokenizer_batch(rows, take, scales, array_cache=cache, cache_limit=40)
        i, slot = np.where(batch['mask'].all(2))
        x = batch['x'][i, slot].astype('f4')
        psd = spectrum(x)
        ordinary = (s[j+i, slot]>0) & (o[j+i, slot]>0)
        good = ordinary & np.isfinite(psd).all(1)
        dropped += int((~good).sum())
        x, i, slot = x[good], i[good], slot[good]
        if not len(x):
            continue
        row, start, pos = take[i].T
        meta = pd.DataFrame(dict(row=row, start=start, pos=pos, slot=slot,
                                 S=s[j+i, slot], O=o[j+i, slot], dataset=ds,
                                 alpha=psd[good, 14:24].sum(1)*.5,
                                 rmsbin=np.digitize(np.sqrt((x*x).mean(1)), [.5, 1, 2])))
        meta['subject'] = [rows[k]['subject'] for k in row]
        meta['site'] = [rows[k]['positions'][p].upper() for k, p in zip(row, pos)]
        meta['record'] = [rows[k]['record'] for k in row]
        waves.append(x)
        frames.append(meta)
    print(split, ds, sum(map(len, waves)), 'complete finite slots;', dropped, 'complete excluded', flush=True)
    return pd.concat(frames, ignore_index=True), np.concatenate(waves), dropped


def panel(split, rows, scales, admitted):
    before = monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(lambda ds: load_dataset(ds, split, rows, scales, admitted), DATASETS))
    frame = pd.concat([p[0] for p in parts], ignore_index=True)
    dropped = sum(p[2] for p in parts)
    print(split, 'seconds', round(monotonic()-before, 1), flush=True)
    return frame, np.concatenate([p[1] for p in parts]), dropped


def rank_words(frame, waves):
    """Cross-person mean of all pairwise patch correlations, equal weight per person pair."""
    table = []
    norm = unit(waves)
    for b in 'SO':
        for code, sub in frame.groupby(b, observed=True, sort=True):
            if len(sub) < 40:
                continue
            groups = list(sub.groupby(SUBJECT, observed=True).groups.values())
            if len(groups) < 8:
                continue
            means = np.stack([norm[g].mean(0, dtype='f8') for g in groups])
            total = means.sum(0)
            shape = (total@total-np.square(means).sum())/(len(groups)*(len(groups)-1))
            table.append(dict(branch=b, ID=int(code), people=len(groups), patches=len(sub),
                              shape_r=shape, alpha_pct=np.mean([sub.loc[g,'alpha'].mean() for g in groups])))
    return pd.DataFrame(table)


def select_pairs(frame, fixed):
    """Three recurrent partners on one electrode in the same recording family."""
    partner = 'O' if fixed=='S' else 'S'
    within = frame[frame.dataset==DATASET]
    pairs = within.groupby(['site', fixed, partner]).agg(n=('slot','size'), people=('subject','nunique')).reset_index()
    eligible = pairs[(pairs.n>=4) & (pairs.people>=2)]
    options = []
    for (site, code), group in eligible.groupby(['site', fixed]):
        if len(group) < 3:
            continue
        best = group.sort_values(['n','people',partner], ascending=[False,False,True]).head(3)
        options.append(((-int(best.n.min()), -int(best.n.sum()), site, int(code)), best))
    if not options:
        raise ValueError('No supported three-partner site')
    _, best = min(options, key=lambda p:p[0])
    return best, dict(eligible_pairs=len(eligible), eligible_site_anchors=len(options))


def mean_by_person(values, sub):
    # values are aligned with the order of sub, independent of source DataFrame index.
    groups = sub.reset_index(drop=True).groupby(SUBJECT, observed=True).indices.values()
    return np.mean([values[i].mean(0) for i in groups], axis=0)


def matched(frame, sub, b, code):
    """One reproducible draw per occurrence from its own person, electrode and X RMS bin."""
    stage, band = role(code)
    lo = 1+1023*band if stage=='导引' else 5116+2047*band
    width = 1023 if stage=='导引' else 2047
    others = frame[frame[b].between(lo, lo+width-1) & (frame[b]!=code)]
    keys = SUBJECT+['site','rmsbin']
    rng = np.random.default_rng(4321)
    pools = others.groupby(keys).groups
    hit, control = [], []
    for key, ids in sub.groupby(keys).groups.items():
        if key in pools:
            hit.extend(ids)
            control.extend(rng.choice(pools[key], len(ids), replace=True))
    return np.array(hit, dtype=int), np.array(control, dtype=int)


def example_record(frame, i, rows):
    row = frame.loc[i].to_dict()
    r = rows[row['row']]
    row['parent'] = r['parent']
    row['parent_relative_start_s'] = int(row['start']+2*row['slot'])
    row['native_start_s'] = r['native_start_sample']/r['native_rate_hz']+row['parent_relative_start_s']
    return row


def calculate(out):
    cfg = configuration()
    admitted = {tuple(v) for v in cfg['data']['validation_subjects']}
    assert len(admitted)==16
    manifest, scales, _, _ = base_tokenizer_training_data(cfg, ROOT)
    rows = manifest['rows']
    train, waves, tdrop = panel('train', rows, scales, admitted)
    ranks = rank_words(train, waves)
    ranks.to_csv(out/'single_candidates.csv', index=False)
    a = ranks.sort_values(['shape_r','people','patches'], ascending=False).iloc[0]
    b = ranks.sort_values(['alpha_pct','people','patches'], ascending=False).iloc[0]
    sub = train[train[a.branch]==a.ID]
    template = mean_by_person(unit(waves[sub.index]), sub)
    template /= np.linalg.norm(template)
    del waves
    pairs, counts = {}, {}
    for branch in 'SO':
        pairs[branch], counts[branch] = select_pairs(train, branch)
    dev, x, ddrop = panel('validation', rows, scales, admitted)
    assert len(dev[SUBJECT].drop_duplicates())==16 and len(train[SUBJECT].drop_duplicates())==60
    curves = dict(template=template)
    singles, examples, pair_rows = [], [], []
    for panel_id, choice in [('a', a), ('b', b)]:
        branch, code = choice.branch, int(choice.ID)
        sub = dev[dev[branch]==code]
        y = x[sub.index]
        spec = spectrum(y)
        vals = unit(y)@template if panel_id=='a' else sub.alpha.values
        people = sub.assign(value=vals).groupby(SUBJECT).value.mean().sort_values()
        # Show three actual people spread over the observed person-level distribution.
        shown = []
        for q in [.25,.5,.75]:
            key = min((k for k in people.index if k not in shown),
                      key=lambda k: abs(people[k]-people.quantile(q)))
            ix = np.where((sub.dataset==key[0]) & (sub.subject==key[1]))[0]
            k = ix[np.argmin(abs(vals[ix]-np.median(vals[ix])))]
            shown.append(key)
            examples.append(dict(panel=panel_id, series=len(shown), **example_record(dev, sub.index[k], rows)))
            curves[f'{panel_id}_ex{len(shown)}'] = y[k] if panel_id=='a' else spec[k]
        mat = y if panel_id=='a' else spec
        curves[panel_id+'_avg'] = mean_by_person(mat, sub)
        curves[panel_id+'_band'] = np.percentile(mat,[10,90],axis=0)
        ix, ctrl = matched(dev, sub, branch, code)
        if not len(ctrl):
            raise ValueError('no matched controls')
        def score(ids):
            return unit(x[ids])@template if panel_id=='a' else dev.loc[ids,'alpha'].values
        member = float(mean_by_person(score(ix), dev.loc[ix]))
        background = float(mean_by_person(score(ctrl), dev.loc[ix]))
        singles.append(dict(panel=panel_id, branch=branch, ID=code, train_patches=int(choice.patches),
            train_people=int(choice.people), train_selection=float(choice.shape_r if panel_id=='a' else choice.alpha_pct),
            patches=len(sub), people=len(people), dev=float(people.mean()), matched=len(ctrl),
            matched_member=member, matched_background=background, matched_difference=member-background,
            matched_unique_controls=len(np.unique(ctrl)), percentiles=np.percentile(vals,[10,50,90]).tolist()))
    for panel_id, fixed in [('c','S'),('d','O')]:
        pairset = pairs[fixed]
        for j, entry in enumerate(pairset.to_dict('records')):
            site, s, o = entry['site'], int(entry['S']), int(entry['O'])
            rec = dict(panel=panel_id, column=j, site=site, S=s, O=o, train_patches=int(entry['n']),
                       train_people=int(entry['people']))
            for frame, name in [(train,'train'),(dev,'dev')]:
                sub = frame[(frame.dataset==DATASET)&(frame.site==site)&(frame.S==s)&(frame.O==o)]
                rec[name+'_alpha'] = float(sub.groupby('subject').alpha.mean().mean()) if len(sub) else None
            if not len(sub):
                pair_rows.append(dict(rec, patches=0, people=0))
                continue
            spectrum_mean = mean_by_person(spectrum(x[sub.index]), sub)
            k = (sub.alpha-sub.alpha.median()).abs().idxmin()
            which = example_record(dev,k,rows)
            examples.append(dict(panel=panel_id,series=j+1,**which))
            curves[f'{panel_id}{j}_wave'] = x[k]
            curves[f'{panel_id}{j}_spec'] = spectrum_mean
            rec.update(patches=len(sub),people=sub.subject.nunique(),example_task=TASK[which['record']])
            for task, label in TASK.items():
                rec[label] = int((sub.record==task).sum())
            assert sum(rec[l] for l in TASK.values()) == len(sub)
            pair_rows.append(rec)
    np.savez_compressed(out/'curves.npz', **curves)
    pd.DataFrame(examples).to_csv(out/'displayed_patches.csv', index=False)
    pd.DataFrame(pair_rows).to_csv(out/'pairs.csv', index=False)
    contrast_span(out, train, dev)
    pair_contrasts(out, train, dev, pair_rows)
    summary = dict(bundle=str((OUT/'bundle.json').relative_to(ROOT)), paper=PAPER,
        selection='Top cross-person unshifted signed waveform correlation and top subject-balanced 8-13 Hz fraction on train, both branches; >=40 patches and >=8 people. Pairs in Simultaneous at one electrode: >=4 train occurrences and >=2 people per pair; choose 3 most frequent partners, maximize minimum then total count.',
        candidates=len(ranks), pair_search=counts, train_patches=len(train), dev_patches=len(dev),
        complete_excluded=[tdrop,ddrop], singles=singles, resources='4 concurrent dataset readers, 1 numeric thread each; G0 frozen pattern rows only; no fitting')
    dump(out/'summary.json', summary)
    print(singles, pair_rows, flush=True)




def contrast_span(out, train, dev):
    """Rank all supported same-site triples by training alpha span and check the top anchor per direction."""
    keys=['site','S','O']
    def summarize(frame):
        frame=frame[frame.dataset==DATASET]
        persons=frame.groupby(keys+['subject']).alpha.mean()
        return persons.groupby(keys).agg(['mean','size']).rename(columns={'mean':'alpha','size':'people'}).join(
            frame.groupby(keys).size().rename('patches'))
    t=summarize(train)
    v=summarize(dev)
    supported=t[(t.patches>=4)&(t.people>=2)].reset_index()
    choices=[]
    for fixed, partner in [('S','O'),('O','S')]:
        for (site,code), group in supported.groupby(['site', fixed]):
            if len(group)<3:
                continue
            ranked=group.sort_values(['alpha',partner]).reset_index(drop=True)
            lo,hi=ranked.iloc[0],ranked.iloc[-1]
            target=(lo.alpha+hi.alpha)/2
            mid=ranked.iloc[1:-1].assign(diff=lambda f:(f.alpha-target).abs()).sort_values(['diff','patches',partner],ascending=[True,False,True]).iloc[0]
            picks=[lo,mid,hi]
            span=float(hi.alpha-lo.alpha)
            for j,r in enumerate(picks):
                key=(site,int(r.S),int(r.O))
                vv=v.loc[key] if key in v.index else None
                choices.append(dict(fixed=fixed,anchor=int(code),site=site,S=int(r.S),O=int(r.O),order=j,
                                    train_span=span,train_alpha=r.alpha,train_n=int(r.patches),train_people=int(r.people),
                                    dev_alpha=vv.alpha if vv is not None else np.nan,
                                    dev_n=int(vv.patches) if vv is not None else 0,dev_people=int(vv.people) if vv is not None else 0))
    choices=pd.DataFrame(choices)
    choices['selected']=False
    for fixed, sub in choices.groupby('fixed'):
        best=sub.sort_values(['train_span','site','anchor'],ascending=[False,True,True]).iloc[0]
        chosen=(choices.fixed==fixed)&(choices.site==best.site)&(choices.anchor==best.anchor)
        choices.loc[chosen,'selected']=True
    choices.to_csv(out/'partner_span.csv',index=False)


def plot_span(out):
    span=pd.read_csv(out/'partner_span.csv')
    selected=span[span.selected]
    fig,axes=plt.subplots(1,2,figsize=(11.8,5.3),sharey=True)
    fig.subplots_adjust(top=.72,left=.08,right=.97,bottom=.22,wspace=.24)
    fig.text(.04,.925,'训练中差异最大的三个搭档，在开发人群中还能分开吗？',fontsize=18,weight='bold')
    fig.text(.04,.846,'同一数据集、同一电极固定一个词，按训练 α 占比选最低、中间、最高的搭档。虚线连接同一词对的两个人群均值。',fontsize=10)
    for ax, fixed, color in zip(axes,'SO',['#00675c','#aa5029']):
        group=selected[selected.fixed==fixed].sort_values('order')
        partner='O' if fixed=='S' else 'S'
        ax.set_title(f"{group.iloc[0].site}  ·  固定 {fixed} {group.iloc[0].anchor}",loc='left',weight='bold',fontsize=13)
        for j,(_,r) in enumerate(group.iterrows()):
            ax.plot([j-.11,j+.11],[r.train_alpha,r.dev_alpha],ls='--',color='#7e9098',lw=1.5)
            ax.scatter(j-.11,r.train_alpha,marker='o',color='#536e80',s=55)
            if r.dev_n:
                ax.scatter(j+.11,r.dev_alpha,marker='s',color=color,s=55)
            else:
                ax.text(j+.1,1,'缺',ha='center')
        ticks=[f"{partner} {int(r[partner])}\n{r.dev_n} 片 / {r.dev_people} 人" for _,r in group.iterrows()]
        ax.set_xticks(range(3),ticks,fontsize=10)
        ax.set_xlim(-.4,2.4)
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('实测 X 的 α 功率占比（%）')
    from matplotlib.lines import Line2D
    fig.legend([Line2D([],[],color='#536e80',marker='o',ls=''),Line2D([],[],color='#00675c',marker='s',ls='')],
               ['训练（选词）','开发（复核）'],ncols=2,loc='lower center',bbox_to_anchor=(.5,.059),frameon=False)
    fig.text(.04,.025,'角标为开发片段 / 被试。每对至少有 4 个训练片段、2 名训练被试；开发出现数不参与排序。每个词对先按人求均值，再跨人等权。',fontsize=9)
    fig.savefig(out/'strongest_pair_contrasts.png',dpi=225,facecolor='white')
    plt.close(fig)


def pair_contrasts(out, train, dev, pairs):
    """Evaluate training-fitted anchor and pair forecasts on exactly the illustrated combinations."""
    out_rows = []
    labels = list(TASK)
    for pair in pairs:
        fixed = 'S' if pair['panel']=='c' else 'O'
        def groups(frame):
            anchor = frame[(frame.dataset==DATASET)&(frame.site==pair['site'])&(frame[fixed]==pair[fixed])]
            return anchor, anchor[(anchor.S==pair['S'])&(anchor.O==pair['O'])]
        anchor, group = groups(train)
        dev_anchor, validation = groups(dev)
        base = anchor.record.value_counts().reindex(labels, fill_value=0).to_numpy()+1
        base = base/base.sum()
        freq = group.record.value_counts().reindex(labels, fill_value=0).to_numpy()
        probs = (freq+5*base)/(freq.sum()+5)
        center = anchor.groupby('subject').alpha.mean().mean()
        pair_center = group.groupby('subject').alpha.mean().mean()
        rec = dict(panel=pair['panel'], site=pair['site'], S=pair['S'], O=pair['O'],
                   train_anchor=len(anchor), dev_anchor=len(dev_anchor), train_pair=len(group),
                   dev_pair=len(validation), dev_people=validation.subject.nunique(),
                   train_anchor_alpha=center, train_pair_alpha=pair_center)
        assert len(validation)==pair['patches'] and len(group)==pair['train_patches']
        if len(validation):
            ids = pd.Categorical(validation.record, categories=labels).codes
            assert ids.min()>=0
            for k, val in dict(anchor_nll=-np.log(base[ids]),pair_nll=-np.log(probs[ids]),
                               anchor_mse=(validation.alpha-center)**2,pair_mse=(validation.alpha-pair_center)**2).items():
                rec[k] = float(mean_by_person(np.asarray(val), validation))
        for task in labels:
            rec['dev_anchor_'+TASK[task]] = int((dev_anchor.record==task).sum())
            rec['dev_pair_'+TASK[task]] = int((validation.record==task).sum())
        out_rows.append(rec)
    pd.DataFrame(out_rows).to_csv(out/'pair_controls.csv',index=False)


def render(out):
    import json
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    summary = json.loads((out/'summary.json').read_text())
    pairs = pd.read_csv(out/'pairs.csv')
    controls = pd.read_csv(out/'pair_controls.csv')
    ink, teal, orange, muted = '#16364a','#00675c','#a95129','#466172'
    colors = ['#46769c','#b96a3d','#719c8d']
    curves = np.load(out/'curves.npz')
    fig = plt.figure(figsize=(16,11.2), facecolor='white')
    fig.text(.045,.952,'同一个 EEG 词，对应怎样的实测波形与频谱？',fontsize=27,weight='bold',color=ink)
    fig.text(.046,.917,'冻结的 S / O 双分量 tokenizer   ·   训练人群选词，独立开发人群绘图   ·   每个成员是同一电极上完整的 2 秒观测 X',fontsize=13,color=muted)
    def axis(box):
        ax = fig.add_axes(box)
        ax.grid(axis='y',alpha=.14)
        ax.tick_params(labelsize=9,length=3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        return ax
    for idx, info in enumerate(summary['singles']):
        key = info['panel']
        left = .052+idx*.493
        xaxis = TIME if key=='a' else FREQ
        stage, band = role(info['ID'])
        color = teal if idx==0 else orange
        title = 'A   波形最一致的单词' if key=='a' else 'B   α 功率占比最高的单词'
        fig.text(left,.865,title,fontsize=17,color=color,weight='bold')
        fig.text(left,.833,f"{info['branch']} {info['ID']}   ·   {BANDS[band]} Hz {stage}词   ·   开发 {info['patches']} 片 / {info['people']} 人",fontsize=13)
        ax = axis([left+.019,.65,.385,.146])
        if key=='b':
            ax.axvspan(8,13,color='#e5bd67',alpha=.23,lw=0)
        low, high = curves[key+'_band']
        ax.fill_between(xaxis,low,high,color=color,alpha=.11,lw=0)
        for j in range(1,4):
            ax.plot(xaxis,curves[key+'_ex'+str(j)],color=colors[j-1],lw=.85,alpha=.80)
        ax.plot(xaxis,curves[key+'_avg'],color=ink,lw=2)
        ax.set(xlim=(xaxis[0],2 if key=='a' else 45),
               xlabel='片内时间（秒）' if key=='a' else '频率（Hz）',
               ylabel='观测 X（训练 SD）' if key=='a' else '相对功率（% / Hz）')
        if key=='b':
            ax.set_yscale('log')
            ax.set_xticks([1,8,13,20,30,45])
        scores=(f"与训练波形模板的相关：同词 {info['matched_member']:.3f}，匹配背景 {info['matched_background']:.3f}"
                if key=='a' else f"8–13 Hz 占比：同词 {info['matched_member']:.1f}%，匹配背景 {info['matched_background']:.1f}%")
        fig.text(left,.585,scores,fontsize=12,color=color,weight='bold')
    fig.text(.052,.550,'上排深线为开发被试等权均值，浅带为全部成员逐点 P10–P90。三条细线各取自不同被试，按被试内指标中位数选例。B 的金色带为 α 频段。',fontsize=11,color=muted)
    fig.add_artist(Line2D([.046,.953],[.525,.525],transform=fig.transFigure,color='#cbd6d8',lw=1))
    pair_wave = [curves[k] for k in curves.files if k.endswith('_wave')]
    pair_spec = [curves[k] for k in curves.files if k.endswith('_spec')]
    ylim_wave=(min(a.min() for a in pair_wave),max(a.max() for a in pair_wave))
    pad=.05*(ylim_wave[1]-ylim_wave[0])
    for bi, key in enumerate('cd'):
        rows = pairs[pairs.panel==key].sort_values('column')
        fixed = 'S' if key=='c' else 'O'
        partner = 'O' if key=='c' else 'S'
        left = .052+bi*.493
        first = rows.iloc[0]
        fig.text(left,.482,f"{key.upper()}   固定 {fixed} {int(first[fixed])}，比较三个 {partner} 词",fontsize=16,weight='bold',color=teal if not bi else orange)
        fig.text(left,.453,f"Simultaneous   ·   同一电极 {first.site}   ·   三列均来自实际同时赋码",fontsize=11,color=muted)
        for j, r in enumerate(rows.to_dict('records')):
            x0 = left+j*.145
            fig.text(x0,.411,f"{partner} {r[partner]:.0f}",fontsize=13,weight='bold')
            fig.text(x0,.386,f"{r['patches']} 片 / {r['people']} 人",fontsize=10,color=muted)
            if not r['patches']:
                fig.text(x0,.28,'开发中未出现',fontsize=11)
                continue
            ax = axis([x0,.278,.123,.078])
            ax.plot(TIME,curves[f'{key}{j}_wave'],color=ink,lw=.85)
            ax.set(xlim=(0,2),ylim=(ylim_wave[0]-pad,ylim_wave[1]+pad),xticks=[0,1,2])
            if j==0:
                ax.set_ylabel('X（SD）',fontsize=9)
            fig.text(x0,.246,f"样例记录：{r['example_task']}",fontsize=9,color=muted)
            sp = axis([x0,.150,.123,.071])
            sp.axvspan(8,13,color='#e5bd67',alpha=.23,lw=0)
            sp.plot(FREQ,curves[f'{key}{j}_spec'],color=teal if not bi else orange,lw=1.35)
            sp.set(xlim=(1,45),xticks=[1,20,45],ylim=(0,max(a.max() for a in pair_spec)*1.05))
            if j==0:
                sp.set_ylabel('% / Hz',fontsize=9)
            fig.text(x0,.110,f"α {r['dev_alpha']:.1f}%",fontsize=10,weight='bold')
            fig.text(x0,.088,f"{r['DSR']} / {r['N-back']} / {r['WG']}",fontsize=9,color=muted)
    fig.text(.052,.048,'下排每列上图为 α 占比最接近该词对中位数的一条实测波形（横轴为秒），下图为该词对的被试等权频谱（横轴为 Hz）；六列同类纵轴共用。',fontsize=10,color=muted)
    fig.text(.052,.026,'列底三数为 DSR（辨别反应）/ N-back / WG（词生成）记录中的开发片段数；记录任务覆盖整场实验，不能给每个两秒片段标记具体反应。',fontsize=10,color=muted)
    fig.savefig(out/'figure4.png',dpi=225,facecolor='white')
    plt.close(fig)

    curves.close()
    plot_span(out)
    report(out, summary, pairs, controls)


def report(out, summary, pairs, controls):
    from playwright.sync_api import sync_playwright
    a, b = summary['singles']
    data = []
    for key in 'cd':
        group = controls[controls.panel==key]
        pred = group[group.dev_pair>0]
        mse0, mse1, nll0, nll1 = pred[['anchor_mse','pair_mse','anchor_nll','pair_nll']].mean()
        data.append(dict(panel=key, anchor_MSE=mse0, pair_MSE=mse1, alpha_gain=1-mse1/mse0,
                         anchor_NLL=nll0, pair_NLL=nll1, NLL_decrease=nll0-nll1,
                         dev_pairs=len(pred), dev_slots=int(pred.dev_pair.sum())))
    comparison = pd.DataFrame(data)
    comparison.to_csv(out/'pair_comparison.csv', index=False)
    contrasts = pd.read_csv(out/'partner_span.csv')
    span_text = []
    for fixed in 'SO':
        sub = contrasts[contrasts.selected & (contrasts.fixed==fixed)].sort_values('order')
        lo, hi = sub.iloc[0], sub.iloc[-1]
        dev_diff = hi.dev_alpha-lo.dev_alpha
        delta = f'{dev_diff:+.1f} 个百分点' if np.isfinite(dev_diff) else '开发中有端点缺失'
        span_text.append(f"固定 {fixed} {int(lo.anchor)}（{lo.site}），训练高低端差 {hi.train_alpha-lo.train_alpha:.1f} 个百分点，开发同方向差为 {delta}。两端开发分别为 {lo.dev_n} 片/{lo.dev_people} 人和 {hi.dev_n} 片/{hi.dev_people} 人。")
    example_rows = ''
    for _, r in pairs.iterrows():
        example_rows += (f"<tr><td>{r.panel.upper()} · {r.site}</td><td>{r.S} + {r.O}</td>"
                         f"<td>{r.train_patches} / {r.train_people}</td><td>{r.patches} / {r.people}</td>"
                         f"<td>{r.train_alpha:.1f} → {r.dev_alpha:.1f}</td><td>{r.DSR} / {r['N-back']} / {r.WG}</td></tr>")
    body = ''
    for r in data:
        fixed='S' if r['panel']=='c' else 'O'
        body += (f"<tr><td>固定 {fixed}</td><td>{r['anchor_MSE']:.1f} → {r['pair_MSE']:.1f}</td><td>{r['alpha_gain']:+.1%}</td>"
                 f"<td>{r['anchor_NLL']:.3f} → {r['pair_NLL']:.3f}</td><td>{r['NLL_decrease']:+.3f}</td></tr>")
    css = """@page{size:360mm 270mm;margin:10mm}body{font:16px/1.55 'Noto Sans CJK SC',sans-serif;color:#16364a;margin:auto;max-width:1300px}h1{font-size:26px;margin:0 0 12px}h2{font-size:19px;color:#00675c;margin:.6em 0 .25em}p{margin:.35em 0 .7em}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:5px 9px;border-bottom:1px solid #ccd9db;text-align:left}th{background:#edf4f2}a{color:#00675c}section{padding:12px 16px}@media print{section{break-after:page;padding:0}section:last-child{break-after:auto}nav{display:none}img{max-height:235mm;width:auto;max-width:100%;display:block;margin:auto}}"""
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>冻结 EEG 词与 CodeBrain 图 4 的对照</title><style>{css}</style></head><body>
    <section><img src="figure4.png" alt="单词波形与频谱、固定 S 或 O 的实际词对"><p>图 1　角色频带来自赋码器，所有曲线来自实测观测 X。各面板内可直接比较同一物理量；A 与下排波形分别定标。选例分母、对照与解读见下一页。</p></section>
    <section><h1>哪些单词特征能够复现，词对又增加了什么？</h1>
    <h2>单词的例子能在独立开发被试上复现</h2><p>从 60 名训练被试的 {summary['train_patches']:,} 个完整电极片段中，在两支共 {summary['candidates']:,} 个支持充分的 ID 中选两个极值。A 对每片去均值并除以其向量范数，按不同被试对等权计算所有成员间的平均有符号 Pearson 相关，全程保留时间原点与极性。{a['branch']} {a['ID']} 的训练跨人两片相关为 {a['train_selection']:.3f}。以训练被试等权均形为冻结模板，开发成员与该模板相关为 {a['dev']:.3f}；它与训练两片之间的相关使用不同参照。</p>
    <p>B 在每个 2 秒片段上去均值、乘 Hann 窗，求 [8,13) Hz 功率占 [1,45] Hz 功率的比例。按被试等权后，{b['branch']} {b['ID']} 的训练值为 {b['train_selection']:.1f}%，开发值为 {b['dev']:.1f}%。A 和 B 分别在开发集全部 {summary['dev_patches']:,} 个完整片段中提取成员，合计开发面板为 16 人。</p>
    <p>每个成员从同一开发人、数据集、电极、局部 X RMS 档、分支、阶段和角色频带的其他词中等概率匹配一个背景片段，允许重复抽取。RMS 按 0.5、1、2 个训练标准差分档。按人等权，A 成员 / 背景相关为 {a['matched_member']:.3f} / {a['matched_background']:.3f}（匹配 {a['matched']}/{a['patches']} 片）；B 成员 / 背景 α 占比为 {b['matched_member']:.1f}% / {b['matched_background']:.1f}%（{b['matched']}/{b['patches']} 片）。图中三个样例来自被试均值接近第 25、50、75 百分位的不同被试，各取最接近该人中位数的一片。</p>
    <h2>固定词后，在同一电极观察到的配对</h2>
    <p>为使记录任务可比，C/D 限于 Simultaneous 数据集并固定电极。在训练中每对至少 4 片、2 人；对每个固定词取计数前三的搭档，以三对的最小片数、再以总片数最大选固定词。C/D 分别有 {summary['pair_search']['S']['eligible_site_anchors']} / {summary['pair_search']['O']['eligible_site_anchors']} 个“电极＋固定词”满足三个搭档要求。选择过程未用开发结果或任务标签。</p>
    <table><tr><th>面板 · 电极</th><th>S + O</th><th>训练 片/人</th><th>开发 片/人</th><th>α 占比，训 → 开（%）</th><th>开发记录 DSR / N-back / WG（片）</th></tr>{example_rows}</table>
    <h2>加入搭档后，在相同开发片段上的预测变化</h2>
    <table><tr><th>已知词</th><th>α 占比 MSE（百分点²），单词 → 词对</th><th>相对误差降低</th><th>任务负对数似然，单词 → 词对</th><th>降低量（nat）</th></tr>{body}</table>
    <p>α 预测取训练成员的被试等权均值，先按开发被试、再按三个词对等权汇总平方误差。任务概率取训练记录计数，单词先对每类加 1，词对用 5 个按单词分布分配的伪计数平滑。在同一批开发成员上，平均负对数似然越低越好。每个方向只评价图中三个复现词对，正降低量表示搭档提供额外信息。</p>
    </section><section><h1>当选例标准改为训练中最大的频谱差异</h1><img style="max-height:120mm" src="strongest_pair_contrasts.png" alt="固定一个分量词时训练与开发的 α 占比差异"><p>{' '.join(span_text)}</p>
    <p>主图优先选择在训练中反复出现的配对；补图从相同候选中，优先选择训练被试等权 α 均值极差最大的固定词和电极。取其最低、最接近两端中点、最高的搭档，保留开发中的缺失和次序逆转。两个视图回答词对可复现性和最好条件下可区分程度两个问题。</p>
    <h2>与 CodeBrain 的解释范围</h2>
    <p>CodeBrain 图 4 用同一 EEG patch 的时域与频域词展示成员及睡眠阶段共现。这里两支编码模型估计的 S/O 分量，按电极分配频带、传输导引词或条件残差。A、B 展示观测波形或谱形随词复现；C、D 展示同位置自然出现的分量词对。记录范式包含任务和间歇，这组比较衡量记录任务关联。A 的均形在约 0.11 秒有负谷、0.27 秒有正峰，可作为锁定片内位置的电位形态，定性归源仍需要事件或伪迹标注。B 在 10 Hz 附近有明显局部峰，低频仍有能量。主图加入搭档后，两个方向的 α 均方误差均降低，两个方向的任务负对数似然均升高。C、D 中有一对重复，六列共五种不同词对，单对的开发被试数为 1–4 人。数据和标签粒度的差异限制了两种模型间的数值比较。<a href="{PAPER}">CodeBrain 图 4 与正文</a></p>
    <nav><a href="single_candidates.csv">训练候选</a> · <a href="displayed_patches.csv">实测样例身份</a> · <a href="pair_controls.csv">词对及背景</a> · <a href="summary.json">选词与结果</a></nav></section></body></html>"""
    (out/'index.html').write_text(html)
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        page=browser.new_page()
        page.goto((out/'index.html').resolve().as_uri(),wait_until='networkidle')
        page.evaluate('document.fonts.ready')
        page.pdf(path=str(out/'comparison.pdf'),prefer_css_page_size=True,print_background=True)
        browser.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('calculate','render'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    if args.stage=='calculate':
        calculate(args.output)
    else:
        render(args.output)
