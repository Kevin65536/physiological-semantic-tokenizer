"""Pair native EEG and the retained whole-record 1–45 Hz cache on development validation."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import get_context
from pathlib import Path
import time

import run_dual_component_tokenizer as runner
import numpy as np
import pandas as pd
from scipy.fft import rfft
from scipy.signal import windows, butter, sosfreqz
from scipy.stats import rankdata

from src.analysis.observation_token_phenotypes import _eeg_shape
from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
from src.data.downstream_input_cache import DownstreamInputDataset, EEG_SCHEMA
from src.data.unified_physiology import _bandpass, _resample, load_native_eeg_record

ROOT = runner.SOURCE
CACHE = ROOT/'data/cache/physiology_downstream_v2'
G0 = ROOT/'experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261117_base_g0_v2'
BANDS = {'sub1': (0, 1), 'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13),
         'beta': (13, 30), 'gamma': (30, 45), 'upper_edge': (40, 45),
         'line50': (49, 51), 'line60': (59, 61), 'above45': (45, 101)}
INTERIOR = ('theta', 'alpha', 'beta', 'gamma')
COUNTS = ('patches', 'peak_valid', 'original_peaks', 'processed_peaks',
          'joint_peaks', 'peak_within_half_hz')


def check(value, message):
    if not value:
        raise ValueError(message)


def spectrum(x, fs):
    """Input time is the final axis; one-sided bin energies sum to tapered AC power."""
    n = x.shape[-1]
    taper = windows.hann(n, sym=False)
    z = rfft((x-x.mean(axis=-1, keepdims=True))*taper, axis=-1)
    z *= np.sqrt(2/(n*np.dot(taper, taper)))
    z[..., 0] = 0
    z[..., -1] /= np.sqrt(2)
    return np.fft.rfftfreq(n, 1/fs), z


def safe_ratio(num, den):
    return np.divide(num, den, out=np.full(np.broadcast_shapes(np.shape(num), np.shape(den)), np.nan),
                     where=np.asarray(den) > 0)


def ranks(a, b, axis):
    ra, rb = (rankdata(x, axis=axis) for x in (a, b))
    ra -= ra.mean(axis=axis, keepdims=True)
    rb -= rb.mean(axis=axis, keepdims=True)
    return safe_ratio((ra*rb).sum(axis), np.sqrt((ra**2).sum(axis)*(rb**2).sum(axis)))


def band_bins(f, band):
    lo, hi = BANDS[band]
    return (f > 0) & (f >= lo) & (f < hi)


def window_metrics(before, after, native, fs):
    # All channel sets and both 200 Hz time axes have already been matched.
    f, a = spectrum(before.T, 200)
    _, b = spectrum(after.T, 200)
    pa, pb = abs(a)**2, abs(b)**2
    xa = before.T.reshape(len(a), 15, 400)
    xb = after.T.reshape(len(b), 15, 400)
    fp, patch_a = spectrum(xa, 200)
    _, patch_b = spectrum(xb, 200)
    patch_a, patch_b = abs(patch_a)**2, abs(patch_b)**2
    stats = {}
    for key in BANDS:
        take = band_bins(f, key)
        ea, eb = pa[:, take].sum(1), pb[:, take].sum(1)
        stats['pre_frac_'+key] = safe_ratio(ea, pa.sum(1))
        stats['post_frac_'+key] = safe_ratio(eb, pb.sum(1))
        stats['db_'+key] = 10*np.log10(safe_ratio(eb, ea))
        stats['cos_'+key] = safe_ratio((a[:, take].conj()*b[:, take]).real.sum(1), np.sqrt(ea*eb))
        if key in INTERIOR:
            which = band_bins(fp, key)
            stats['time_'+key] = ranks(patch_a[:,:,which].sum(2), patch_b[:,:,which].sum(2), 1)
            stats['map_'+key] = ranks(ea, eb, 0)
    if fs > 200:
        fn, zn = spectrum(native.T, fs)
        pn = abs(zn)**2
        stats['native_fraction_above100'] = safe_ratio(pn[:, fn>100].sum(1), pn.sum(1))
    ha, hb = (_eeg_shape(v.reshape(-1,1,400)) for v in (xa, xb))
    valid = np.isfinite(ha['peak']) & np.isfinite(hb['peak'])
    a_hit, b_hit = (v['peak']==1 for v in (ha, hb))
    both = valid & a_hit & b_hit
    counts = dict(patches=valid.size, peak_valid=valid.sum(), original_peaks=(a_hit&valid).sum(),
                  processed_peaks=(b_hit&valid).sum(), joint_peaks=both.sum(),
                  peak_within_half_hz=(both & (abs(ha['peak_hz']-hb['peak_hz'])<=.5)).sum())
    # Average electrodes within each physical window; maps are one correlation per window.
    missing = {k:int(np.size(v)-np.isfinite(v).sum()) for k,v in stats.items()}
    metrics = {k:float(np.nanmean(v)) if np.isfinite(v).any() else None for k,v in stats.items()}
    return metrics, counts, missing


def prepare():
    verified = runner.read_json(CACHE/'verification.json')
    check(verified['status']=='passed' and not verified['errors'], 'Incomplete downstream cache')
    check(runner.read_json(G0/'verification.json')['status']=='passed', 'Incomplete G0 panel')
    ds = DownstreamInputDataset(CACHE, split='validation')
    plan = ds.plan
    groups = defaultdict(list)
    for e in plan['records']:
        if e['split']=='validation':
            check(e['subject'] in plan['subject_partitions'][e['dataset']]['validation'], 'Partition mismatch')
            groups[e['native_group']].append(ds.records[e['join_key']])
    jobs = {}
    manifest = runner.read_json(G0/'cache/manifest.json')
    check(manifest['allowed_splits']==['train','validation'] and manifest['status']=='completed','G0 split')
    for row in manifest['rows']:
        if row['modality']!='eeg' or row['split']!='validation' or not row['inputs']['score']:
            continue
        name = '|'.join([row['dataset'], row['subject'], row['record']])
        matched = groups[name]
        record = matched[0]
        m = record.manifest
        check(all((r.npz_path/'eeg.npy').samefile(record.npz_path/'eeg.npy') for r in matched),
              'Duplicate views must have identical EEG storage')
        state = m['measurement']['eeg_preprocessing_state']
        check(m['eeg_input_schema']==EEG_SCHEMA and state['filter_band_hz']==[1.,45.]
              and state['canonical_sample_rate_hz']==200 and state['signal_branch']=='raw_with_ocular_artifact'
              and state['artifact_cleaning']['action']=='no_artifact_removal', 'EEG branch mismatch')
        check(row['positions']==m['measurement']['eeg_channel_names']
              and row['native_rate_hz']==state['native_sample_rate_hz'], 'EEG channel or rate mismatch')
        start = row['native_start_sample']
        fs = row['native_rate_hz']
        job = jobs.setdefault(name, dict(record=record, windows=[]))
        for s in row['inputs']['score']:
            job['windows'].append((start+s['start_s']*fs,s['positions']))
    return jobs


def record_signals(record):
    m = record.manifest
    state = m['measurement']['eeg_preprocessing_state']
    check(state['source_path'] in [s['path'] for s in m['source_files']], 'Unregistered native EEG source')
    raw = load_native_eeg_record(ROOT, record)
    check(str(raw.source_path.resolve())==state['source_path'] and raw.reference==state['reference']
          and list(raw.channel_names)==m['measurement']['eeg_channel_names'], 'Native identity mismatch')
    arrays = CleanPhysiologyCacheIndex.load_record_arrays(record, ['eeg','eeg_supported_channels'])
    after = arrays['eeg']
    fs = int(raw.sample_rate_hz)
    factors = np.array([v['factor'] for v in state['unit_conversions']])
    native = raw.values*factors
    check(np.isfinite(native).all() and state['filter_input_repaired_nonfinite_samples']==0,
          'Nonfinite native source requires paired support review')
    # Identical whole-record resampling for the control, excluding the bandpass only.
    before = _resample(native, fs, 200.)
    check(before.shape==after.shape, 'Record grid mismatch')
    channel = len(factors)//2
    replay = _resample(_bandpass(native[:,channel:channel+1],fs,(1.,45.)),fs,200.)[:,0]
    maxdiff = np.max(abs(replay-after[:,channel]))
    check(maxdiff <= 1e-9*max(np.abs(replay).max(),np.finfo(float).tiny), 'Cached channel replay differs')
    return native, before, after, arrays['eeg_supported_channels'], fs, maxdiff, max(np.max(abs(replay)),1e-300)


def measure(name, job):
    record = job['record']
    native, before, after, supported, fs, maxdiff, refmax = record_signals(record)
    rows, invalid = [], defaultdict(int)
    for start, positions in job['windows']:
        idx = round(start*200/fs)
        use = np.asarray(positions)[supported[positions]]
        check(len(use)>0, 'Window without jointly supported electrodes')
        lo = round(idx*fs/200)
        n = native[lo:lo+30*fs, use]
        a, b = before[idx:idx+6000,use], after[idx:idx+6000,use]
        check(a.shape==b.shape==(6000,len(use)) and n.shape==(30*fs,len(use)), 'Truncated physical window')
        values, counts, missing = window_metrics(a,b,n,fs)
        for key,count in missing.items():
            invalid[key]+=count
        rows.append(dict(start_sample=start, rate=fs, round_ms=1000*(idx/200-start/fs),
                         planned_channels=len(positions), channels=len(use), **counts, **values))
    return dict(name=name, dataset=record.dataset_id, subject=record.canonical_subject_id, rows=rows,
                cache_channel_replay_max_abs=maxdiff, cache_channel_replay_relative=maxdiff/refmax,
                invalid=dict(invalid))


def synthetic():
    t = np.arange(6000)/200
    x = np.column_stack([np.sin(2*np.pi*10*t)+.2*np.cos(2*np.pi*22*t),
                         1.5*np.sin(2*np.pi*10*t)+np.cos(2*np.pi*8*t)])
    m,c,_ = window_metrics(x,x,x,200)
    np.testing.assert_allclose([m['db_alpha'],m['cos_alpha'],m['map_alpha']], [0.,1.,1.], atol=1e-10)
    check(c['original_peaks']>0 and c['peak_within_half_hz']==c['original_peaks'],'Synthetic peak identity')
    for fs in (200,500,1000):
        ts = np.arange(45*fs)/fs
        for freq in (.5,1,10,45,60):
            wave=np.cos(2*np.pi*freq*ts)[:,None]
            filtered=_bandpass(wave,fs,(1.,45.))
            sl=slice(10*fs,-10*fs)
            _,h=sosfreqz(butter(4,[1,45],fs=fs,btype='bandpass',output='sos'),worN=[freq],fs=fs)
            np.testing.assert_allclose(np.std(filtered[sl])/np.std(wave[sl]),abs(h[0])**2,atol=2e-5)


def resampling_response(out):
    """Amplitude gain of the actual whole-record resampler on passband test tones."""
    frequencies = [1, 4, 8, 10, 13, 20, 30, 40, 45]
    rows = []
    for fs in (200, 500, 1000):
        t = np.arange(20*fs)/fs
        waves = np.cos(2*np.pi*t[:,None]*frequencies)
        y = _resample(waves, fs, 200.)
        ratio = np.std(y[1000:-1000],axis=0)/np.std(waves[5*fs:-5*fs],axis=0)
        for f, db in zip(frequencies, 20*np.log10(ratio)):
            rows.append(dict(native_hz=fs,tone_hz=f,gain_db=db))
    pd.DataFrame(rows).to_csv(out/'resampling_response.csv',index=False)


def summarize(out, jobs):
    frames, reports = [], []
    for name in jobs:
        record=runner.read_json(out/'records'/(name+'.json'))
        reports.append(record)
        frames.append(pd.DataFrame(record['rows']).assign(dataset=record['dataset'],subject=record['subject'],record=name))
    frame=pd.concat(frames,ignore_index=True)
    ids=['dataset','subject','record']
    metrics=[c for c in frame if c.startswith(('pre_', 'post_', 'db_', 'cos_', 'map_', 'time_', 'native_fraction'))]
    by_record=frame.groupby(ids)[metrics].mean()
    by_subject=by_record.groupby(level=[0,1]).mean()
    by_dataset=by_subject.groupby(level=0).mean()
    cnt=frame.groupby(ids)[list(COUNTS)].sum()
    # Raw peak counts are supplied separately. Equal-weight rates start at record,
    # conditional on peak support; cases with zero denominators remain missing.
    rates=pd.DataFrame(dict(peak_recall=cnt.joint_peaks/cnt.original_peaks,
                            peak_precision=cnt.joint_peaks/cnt.processed_peaks,
                            peak_stable=cnt.peak_within_half_hz/cnt.original_peaks))
    for tab, group in [(by_record,rates),(by_subject,rates.groupby(level=[0,1]).mean()),
                       (by_dataset,rates.groupby(level=[0,1]).mean().groupby(level=0).mean())]:
        for k in group: tab[k]=group[k]
    by_record.join(cnt).to_csv(out/'record.csv')
    by_subject.to_csv(out/'subject.csv')
    by_dataset.join(cnt.groupby(level=0).sum()).to_csv(out/'dataset.csv')
    counts=frame.groupby('dataset').agg(subjects=('subject','nunique'), records=('record','nunique'),
        windows=('record','size'), planned_sites=('planned_channels','sum'), sites=('channels','sum'),
        max_abs_time_rounding_ms=('round_ms',lambda x:x.abs().max()))
    counts.to_csv(out/'coverage.csv')
    runner.write_json(out/'verification.json',dict(status='completed', split='validation',source=str(CACHE),
        window_source=str(G0/'cache/manifest.json'),records=len(jobs),windows=len(frame),
        subject_count=len(by_subject), maximum_relative_replay_error=max(r['cache_channel_replay_relative'] for r in reports),
        missing_feature_counts={k:sum(r['invalid'].get(k,0) for r in reports) for k in metrics},
        record_peak_denominators_zero=dict(before=int((cnt.original_peaks==0).sum()),after=int((cnt.processed_peaks==0).sum())),
        definition='Full-record native input converted to cache units. Reference = resample-only 200 Hz; output = actual native-rate 4th-order Butterworth 1–45 Hz forward/backward then 200 Hz cache. On identical aligned 30 s: remove mean, Hann taper, one-sided FFT; remove DC. Fractions average channel-wise power shares, dB=10 log10(post/pre) by channel, cosine=real normalized complex band inner product, map=Spearman channel power ranks; time=Spearman of fifteen 2-s band powers per electrode. Scores average channels/windows/records/subjects equally within level. Alpha peaks use the project 2-s above-background/shoulder rule; stable=detected post within 0.5 Hz / original positives. Native >100 fraction measured only where native Nyquist exceeds 100 Hz.'))
    return by_dataset


def plot(table,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':['Noto Sans CJK JP','DejaVu Sans'],'axes.unicode_minus':False})
    fig, axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    names={'refed':'REFED','eeg_fnirs_single_trial':'Single-Trial','simultaneous_eeg_nirs':'Simultaneous','visual_cognitive_motivation':'Visual'}
    bands=['sub1','delta','theta','alpha','beta','gamma','above45']
    labels=['<1','1–4','4–8','8–13','13–30','30–45','≥45']
    for d,c,m in zip(table.index,['#466f8a','#af472d','#287668','#82648b'],['s','o','^','D']):
        for ax,stage in zip(axes[:2],['pre','post']):
            ax.plot(labels,100*table.loc[d,[stage+'_frac_'+b for b in bands]].values,label=names[d],color=c,marker=m)
        axes[2].plot(labels[1:-1],table.loc[d,['db_'+b for b in bands[1:-1]]].values,label=names[d],color=c,marker=m)
    for ax,title in zip(axes,['仅统一采样率','实际统一带通后','带通后的功率变化']):
        ax.set_title(title); ax.set_xlabel('频带（Hz）'); ax.tick_params(axis='x',rotation=25)
    for ax in axes[:2]: ax.set_ylim(0,100); ax.set_ylabel('占 0–100 Hz 交流能量（%）')
    axes[0].legend(frameon=False); axes[2].axhline(0,c='gray',lw=.8);axes[2].set_ylabel('滤波后 / 滤波前（dB）')
    fig.savefig(out/'spectra.png',dpi=220);plt.close(fig)



def record_spectrum(job):
    _, before, after, support, fs, _, _ = record_signals(job['record'])
    curves = []
    for start, positions in job['windows']:
        use = np.asarray(positions)[support[positions]]
        idx = round(start*200/fs)
        pair = []
        for source in (before, after):
            _, z = spectrum(source[idx:idx+6000,use].T, 200)
            power = abs(z)**2
            bins = power[:,:-1].reshape(len(use),200,15).sum(2)
            bins[:,-1] += power[:,-1]
            pair.append(bins)
        ref = pair[0].sum(1,keepdims=True)
        curves.append(np.stack([safe_ratio(b, ref).mean(0)/.5 for b in pair]))
    return np.mean(curves,axis=0)


def spectrum_detail(jobs, out, workers):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('fork')) as pool:
        rows = []
        for name, pair in zip(jobs,pool.map(record_spectrum,jobs.values())):
            dataset, subject, record = name.split('|')
            rows.append(pd.DataFrame(dict(dataset=dataset,subject=subject,record=record,
                hz=np.arange(200)*.5+.25, pre=pair[0],post=pair[1])))
    data = pd.concat(rows).groupby(['dataset','subject','hz'])[['pre','post']].mean()
    data = data.groupby(['dataset','hz']).mean().reset_index()
    data.to_csv(out/'psd.csv',index=False)
    plt.rcParams.update({'font.family':['Noto Sans CJK JP','DejaVu Sans'],'axes.unicode_minus':False})
    names = dict(refed='REFED', eeg_fnirs_single_trial='Single-Trial',
                 simultaneous_eeg_nirs='Simultaneous', visual_cognitive_motivation='Visual')
    fig,axs=plt.subplots(2,2,sharex=True,sharey=True,figsize=(10,6.5),layout='constrained')
    for ax,(d,z) in zip(axs.flat,data.groupby('dataset')):
        ax.semilogx(z.hz,10*np.log10(z.pre),label='仅统一采样率',color='#496778')
        ax.semilogx(z.hz,10*np.log10(z.post),label='1–45 Hz 后',color='#ad452f',ls='--')
        for edge in (1,45): ax.axvline(edge,color='gray',lw=.7)
        ax.set(title=names[d],xticks=[.5,1,4,8,13,30,45,100],
               xticklabels=['0.5','1','4','8','13','30','45','100'])
    axs[0,0].legend(frameon=False)
    fig.supxlabel('频率（Hz，对数轴）；竖线为 1 和 45 Hz')
    fig.supylabel('每 Hz 功率 / 滤波前交流总功率（dB）')
    fig.savefig(out/'psd.png',dpi=220)
    plt.close(fig)


def event_response(out):
    """Actual EEG-clock Go/No-go contrasts, restricted to development validation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    view = DownstreamInputDataset(CACHE, split='validation', task='dsr')
    assert runner.read_json(CACHE/'verification.json')['status']=='passed'
    records = defaultdict(list)
    for sample in view.samples:
        check(sample['dataset']=='simultaneous_eeg_nirs' and sample['record']=='cnt_dsr'
              and sample['split']=='validation' and sample['subject'] in
              view.plan['subject_partitions']['simultaneous_eeg_nirs']['validation'], 'DSR event boundary')
        check(sample['condition']==sample['event']['label'] and sample['event']['event_type']=='stimulus'
              and np.isclose(sample['event']['eeg_time_ms'], sample['eeg_start_s']*1000),
              'DSR stimulus clock/label mismatch')
        records[sample['join_key']].append(sample)
    ts = np.arange(-40,200)/200
    traces, result = [], []
    for key, events in records.items():
        record = view.records[key]
        m = record.manifest
        state = m['measurement']['eeg_preprocessing_state']
        check(state['source_path'] in [v['path'] for v in m['source_files']] and
              state['signal_branch']=='raw_with_ocular_artifact', 'DSR native source')
        positions = [m['measurement']['eeg_channel_names'].index(ch) for ch in ('Cz','Pz')]
        native = load_native_eeg_record(ROOT,record)
        check(native.sample_rate_hz==200 and str(native.source_path.resolve())==state['source_path']
              and list(native.channel_names)==m['measurement']['eeg_channel_names'], 'DSR native sampling clock')
        saved=CleanPhysiologyCacheIndex.load_record_arrays(record,['eeg','eeg_supported_channels'])
        check(saved['eeg_supported_channels'][positions].all(), 'DSR electrode unsupported')
        check(all(v['output_unit']=='uV' for v in state['unit_conversions']), 'DSR unit')
        factors=np.array([state['unit_conversions'][i]['factor'] for i in positions])
        signals = [np.mean(native.values[:,positions]*factors,axis=1),saved['eeg'][:,positions].mean(1)]
        counts={}
        epochs={}
        for label in ('Go','No-go'):
            ends = [round(s['event']['eeg_time_ms']/5) for s in events if s['condition']==label]
            check(len(ends)==len(set(ends)) and len(ends)>0,'Repeated or empty DSR events')
            counts[label]=len(ends)
            indices = np.array(ends)[:,None] + np.arange(-40,200)
            check(indices.min()>=0 and indices.max()<len(signals[0]),'Epoch boundary')
            for i,signal in enumerate(signals):
                x=signal[indices]
                epochs[label,i]=(x-x[:,:40].mean(1,keepdims=True)).mean(0)
        check(sum(counts.values())==len(events),'Unexpected DSR condition')
        orig,filtered=(epochs['Go',i]-epochs['No-go',i] for i in (0,1))
        subject=record.canonical_subject_id
        traces.append(pd.DataFrame(dict(subject=subject, time_s=ts, original=orig, filtered=filtered)))
        post=ts>=0
        metrics=dict(subject=subject,go=counts['Go'],no_go=counts['No-go'],
                     corr_0_1=np.corrcoef(orig[post],filtered[post])[0,1],
                     rms_change_uV=np.sqrt(np.mean((orig[post]-filtered[post])**2)))
        for lo,hi in [(0.3,0.6),(0.6,1.)]:
            interval=(ts>=lo)&(ts<hi)
            metrics[f'pre_{lo}_{hi}']=orig[interval].mean()
            metrics[f'post_{lo}_{hi}']=filtered[interval].mean()
        result.append(metrics)
    pd.DataFrame(result).to_csv(out/'erp_subject.csv',index=False)
    frame=pd.concat(traces,ignore_index=True)
    frame.to_csv(out/'erp_waveform.csv',index=False)
    plt.rcParams.update({'font.family':['Noto Sans CJK JP','DejaVu Sans'], 'axes.unicode_minus':False})
    fig,axes=plt.subplots(1,len(result),sharey=True,figsize=(12,3.7),layout='constrained')
    for ax,(sub,z) in zip(np.atleast_1d(axes),frame.groupby('subject')):
        ax.plot(z.time_s,z.original,label='仅统一采样率',color='#496778')
        ax.plot(z.time_s,z.filtered,label='1–45 Hz 后',color='#ad452f',ls='--')
        ax.axvline(0,c='gray',lw=.8);ax.axhline(0,c='gray',lw=.8)
        ax.set(title=sub,xlabel='相对 EEG 刺激时间（秒）')
    axes[0].set_ylabel('Go 减 No-go 电位差（μV）')
    axes[0].legend(frameon=False)
    fig.suptitle('Cz、Pz 平均电位：逐试次减去刺激前 200 ms 基线')
    fig.savefig(out/'erp.png',dpi=220);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    synthetic()
    jobs=prepare()
    dest=args.output
    (dest/'records').mkdir(parents=True,exist_ok=True)
    pending=[name for name in jobs if not (dest/'records'/(name+'.json')).exists()]
    # Start with one validation record per dataset; retain these as part of the panel.
    pilot=list(dict((n.split('|')[0],n) for n in reversed(pending)).values())
    for batch in (pilot,[n for n in pending if n not in pilot]):
        if not batch: continue
        begin=time.monotonic()
        with ProcessPoolExecutor(max_workers=min(len(batch),args.workers),mp_context=get_context('fork')) as pool:
            futures={pool.submit(measure,n,jobs[n]):n for n in batch}
            for i,f in enumerate(as_completed(futures),1):
                n=futures[f]
                runner.write_json(dest/'records'/(n+'.json'),f.result())
                if len(batch)<=4 or i%10==0 or i==len(batch):
                    print(f'{i}/{len(batch)} records in {time.monotonic()-begin:.1f} s; last {n}',flush=True)
    resampling_response(dest)
    plot(summarize(dest,jobs),dest)
    event_response(dest)
    spectrum_detail(jobs,dest,args.workers)
