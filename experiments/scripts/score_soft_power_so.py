"""Score exact saved targets with paired denominators and independent natural EOG."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from experiments.scripts.evaluate_soft_power_so import CFG, OUT, PARENT, PRODUCTION, read, parent
from src.data.downstream_input_cache import DownstreamInputDataset
from src.data.dual_component_tokenizer import BANDS, voltage_power_cap_bank, voltage_power_cap_reference
from src.data.so_supervision_cache import bind_reference, write_json
from src.data.soft_power_so import final_power
from src.data.unified_physiology import load_native_eeg_record, _bandpass
from src.data.window_exclusions import local_spans

SCORE = slice(0, 17980)
ACTIVE = slice(6000, 10000)


def correlation(a, b):
    return parent.corr(a, b)


def eog_basis(view, c):
    if c['dataset'] not in ('eeg_fnirs_single_trial', 'simultaneous_eeg_nirs'):
        return None
    g = c['source']
    group = '|'.join(g[k] for k in ('dataset','subject','native_record'))
    entries = [r for r in view.plan['records'] if r['native_group'] == group and r['split'] == 'validation']
    assert len(entries) == 1 and c['subject'] in view.plan['subject_partitions'][c['dataset']]['validation']
    record = view.records[entries[0]['join_key']]
    raw = load_native_eeg_record(ROOT, record)
    assert set(raw.auxiliary_channel_names) == {'HEOG', 'VEOG'} and raw.sample_rate_hz == 200
    cut = slice(g['native_start_sample'], g['native_start_sample']+18000)
    np.testing.assert_array_equal(raw.values[cut], np.load(PARENT/'inputs'/str(c['i'])/'native.npy'))
    y = _bandpass(raw.auxiliary_values, 200, (1., 45.))[cut][SCORE]
    q, singular, _ = np.linalg.svd(y-y.mean(0), full_matrices=False)
    return q[:, singular > singular[0]*1e-10]


def score():
    assert read(OUT/'generation_status.json')['status'] == 'completed'
    cases = read(OUT/'plan.json')['cases']
    view = DownstreamInputDataset(ROOT/parent.CFG['downstream'], split='validation')
    rows, bands, changes, associations, sensitivity = [], [], [], [], []
    lag = 5
    for c in cases:
        i = c['i']
        folder = OUT/'targets'/str(i)
        idx = c['valid']
        ident = read(PARENT/'inputs'/str(i)/'identity.json')
        q = eog_basis(view, c)
        frontal = [k for k,j in enumerate(idx) if c['channels'][j].upper().startswith(('FP','AF'))]
        pid = dict(i=i, dataset=c['dataset'], subject=c['subject'])
        with np.load(folder/'baseline.npz') as f:
            x = f['x']
            targets = {a:f[a] for a in CFG['arms']}
            for a,s in targets.items():
                np.testing.assert_array_equal(x-s, f[a+'_o'])
        xb = voltage_power_cap_bank(x)
        with np.load(PARENT/'fits'/str(i)/'unified/baseline.npz') as f:
            proxy, hb = f['prediction'][:,:,0], f['target'][:,:,1:]
        with np.load(PARENT/'references'/(c['dataset']+'.npz')) as f:
            model = bind_reference(dict(f), np.load(PARENT/'inputs'/str(i)/'projection.npy'), 900)
        ref = voltage_power_cap_reference(model, proxy, np.arange(5,896)/10)[:-1, :, idx]
        tol = read(folder/'baseline.json')['delta']
        meta = read((PRODUCTION/'windows'/c['key']).with_suffix('.json'))
        blocked_hb = {j for j,lo,hi in local_spans(c['dataset'],'validation',c['subject'],
            c['source']['native_record'],'hb',meta['fNIRS_positions'],c['source']['start_sample'],18000)}
        for arm,s in targets.items():
            o = x-s
            sb = voltage_power_cap_bank(s)
            ob = xb-sb
            ratio = np.stack([final_power(s[:,k]).T[:-1] for k in range(len(idx))],axis=-1)
            eps = 1e-12*model['voltage_rms'][idx]**2
            excess = np.maximum(np.log((ratio.clip(min=0)+eps)/(ref+eps)) - tol, 0)
            r = dict(**pid, arm=arm, channels=len(idx),
                     s_x_corr=np.median([correlation(s[SCORE,j],x[SCORE,j]) for j in range(len(idx))]),
                     s_rms=parent.rms(s[SCORE])/parent.rms(x[SCORE]),
                     o_rms=parent.rms(o[SCORE])/parent.rms(x[SCORE]),
                     change_from_v3=parent.rms((s-targets['v3'])[SCORE])/parent.rms(x[SCORE]),
                     excess_rms=parent.rms(excess), excess_fraction=np.mean(excess>0))
            if q is not None:
                projected = {}
                for name, data in [('x',x), ('s',s), ('o',o)]:
                    front = data[SCORE,frontal]
                    centered = front-front.mean(0)
                    projected[name] = q.T@centered
                    r[name+'_eog_r2'] = np.sum(projected[name]**2)/np.sum(centered**2)
                r['eog_relative'] = np.linalg.norm(projected['s'])/np.linalg.norm(projected['x'])
                r['eog_frontal_sites'] = len(frontal)
            for j,(lo,hi) in enumerate(BANDS):
                xband, sband, oband = (b[SCORE,j,:] for b in (xb,sb,ob))
                den = np.mean(xband**2)
                br = dict(**pid, arm=arm, band=f'{lo}-{hi}', s=np.mean(sband**2)/den,
                          o=np.mean(oband**2)/den, cross=2*np.mean(sband*oband)/den)
                np.testing.assert_allclose(br['s']+br['o']+br['cross'],1.,atol=1e-10)
                xe,se = [(b[:17800,j].reshape(89,200,-1)**2).mean(1) for b in (xb,sb)]
                br['envelope_corr'] = np.median([correlation(xe[:,k],se[:,k]) for k in range(len(idx))])
                bands.append(br)
            # Non-overlapping full seconds, exclude final partial second. Same observed Hb for all arms.
            for j,name in enumerate(c['positions']):
                if meta['fNIRS_positions'].index(name) in blocked_hb:
                    continue
                sites = [idx.index(k) for k in ident['neighbors'][j] if k in idx]
                if not sites:
                    continue
                for component, bank in [('X',xb), ('S',sb), ('O',ob)]:
                    alpha = (bank[:17800,2,sites].reshape(89,200,-1)**2).mean((1,2))
                    for color, hname in enumerate(['HbO','HbR']):
                        h = hb[:890,j,color].reshape(89,10).mean(1)
                        hlag = h[lag:]
                        associations.append(dict(**pid,arm=arm,roi=name, component=component, hb=hname,
                             r=correlation(alpha[:-lag],hlag), shifted_r=correlation(alpha[:-lag],np.roll(hlag,30))))
            rows.append(r)
        for v in CFG['variants'][1:]:
            with np.load(folder/(v+'.npz')) as f:
                delta = f['delta'][ACTIVE]
                np.testing.assert_allclose(f['x'][ACTIVE]-x[ACTIVE],delta,atol=1e-12*np.max(abs(x)))
                for a in CFG['arms']:
                    s0 = targets[a][ACTIVE]
                    s1 = f[a][ACTIVE]
                    ds = s1-s0
                    if v == 'ocular1':
                        # Remove the union of capped channels in both soft arms and both paired conditions.
                        keep = np.ones(len(idx), dtype=bool)
                        for cond in ('baseline', v):
                            record = read(folder/(cond+'.json'))
                            for info in record['fit'].values():
                                keep &= [row['converged'] for row in info]
                        sensitivity.append(dict(**pid, arm=a, channels=keep.sum(),
                             leak=parent.rms(ds[:,keep])/parent.rms(delta[:,keep]),
                             s_relative=parent.rms(ds[:,keep])/parent.rms(s0[:,keep])))
                    vals = parent.change_score(ds, delta)
                    changes.append(dict(**pid, arm=a, variant=v, **vals, s_signed_gain=1-vals['gain'],
                        s_relative=parent.rms(ds)/parent.rms(s0), s_pair_corr=correlation(s1,s0)))
    tables = {}
    for name, items in [('baseline',rows), ('ocular_converged_channels',sensitivity), ('bands',bands), ('interventions',changes), ('hb_associations',associations)]:
        df = pd.DataFrame(items)
        df.to_csv(OUT/(name+'.csv'),index=False)
        tables[name] = df
    d = tables['interventions']
    base = tables['baseline']
    old_eog = pd.read_csv(PARENT/'natural_eog.csv').query("arm=='unified'").set_index('subject')
    actual = base.dropna(subset=['eog_relative']).query("arm=='v3'").set_index('subject')
    np.testing.assert_allclose(actual.eog_relative, old_eog.loc[actual.index,'s_eog_relative_norm'], rtol=1e-8)
    original = pd.read_csv(PARENT/'subject_metrics.csv').query("arm=='unified'").set_index(['i','variant'])
    complete = [c['i'] for c in cases if not c['excluded']]
    replay = d[d.i.isin(complete) & (d.arm=='v3')].set_index(['i','variant'])
    np.testing.assert_allclose(replay[['leak','o_corr']], original.loc[replay.index,['leak','o_corr']], rtol=1e-7)
    hb = tables['hb_associations']
    by_subject = hb.groupby(['arm','i','component','hb'])[['r','shifted_r']].median().reset_index()
    by_subject['difference'] = by_subject.r - by_subject.shifted_r
    by_subject.to_csv(OUT/'hb_subjects.csv',index=False)
    diffs=[]
    for name,frame,metrics,groups in [
        ('baseline',base,['s_x_corr','s_rms','o_rms','eog_relative','excess_rms'],[]),
        ('interventions',d,['leak','s_relative','o_corr','s_signed_gain'],['variant']),
        ('bands',tables['bands'],['s','o','cross','envelope_corr'],['band'])]:
        for group, part in (frame.groupby(groups[0]) if groups else [('all',frame)]):
            for metric in metrics:
                wide = part.pivot(index='i',columns='arm',values=metric)
                for comparator in ['v3','soft_fixed']:
                    diff = (wide.soft_ssm-wide[comparator]).dropna()
                    diffs.append(dict(table=name,group=group,metric=metric,comparator=comparator,n=len(diff),
                                      positive=int((diff>0).sum()),negative=int((diff<0).sum()),median=diff.median()))
    pd.DataFrame(diffs).to_csv(OUT/'paired_contrasts.csv',index=False)
    diagnostics = [read(OUT/'targets'/str(c['i'])/(v+'.json')) for c in cases for v in CFG['variants']]
    checks = [row for rec in diagnostics for fits in rec['fit'].values() for row in fits]
    write_json(OUT/'summary.json',dict(status='completed', subjects=len(cases), datasets=4, cells=len(diagnostics)*3,
        primary='paired 1x ocular leakage and relative S change', subjects_with_eog=int(base.eog_relative.count()//3),
        valid_electrodes=sum(len(c['valid']) for c in cases), planned_electrodes=sum(len(c['channels']) for c in cases),
        fits=len(checks), converged=sum(bool(r['converged']) for r in checks),
        max_closure=max(max(r['closure_over_x_rms'].values()) for r in diagnostics),
        median_ocular=d[d.variant=='ocular1'].groupby('arm')[['leak','s_relative','o_corr','s_pair_corr']].median().to_dict('index')))


if __name__ == '__main__':
    score()
