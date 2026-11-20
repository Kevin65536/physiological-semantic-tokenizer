#!/usr/bin/env python3
"""Final same-budget EEG ablation; no raw, teacher, test or task-label entry point."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

import joblib
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.scripts.run_observation_detail_allocation import replay_panel
from src.analysis.observation_tokenizer_evaluation import BANDS, KEYS, aggregate, hierarchy_weights, paired_contrast, reconstruction
from src.tokenizers.adaptive_observation_detail import AdaptiveObservationDetail
from src.tokenizers.factorized_observation import FactorizedObservationTokenizer
from src.tokenizers.gain_shape_observation import GainShapeObservation

CONFIG = ROOT/'experiments/configs/physiology_semantic_tokenizer/observation_gain_shape_v1.yaml'
SUITE = ROOT/'experiments/runs/physiology_semantic_tokenizer/observation_factorization'
PARENT = SUITE/'20261120_development_v3'
ALLOCATION = SUITE/'20261120_allocation_v1'
DATASETS = ('eeg_fnirs_single_trial', 'refed', 'simultaneous_eeg_nirs', 'visual_cognitive_motivation')
TABLE_KEYS = ['level', 'dataset', 'subject']
HF = 'band_30_45_db'
HC = 'band_30_45_cosine'
METRICS = ['x_nmse', 'level_nmse', 'shape_nmse', 'centered_cosine',
           *(f'band_{lo}_{hi}_{m}' for lo, hi in BANDS for m in ('db', 'cosine'))]
SOURCES = ('experiments/scripts/run_observation_gain_shape.py', 'src/tokenizers/gain_shape_observation.py',
           'experiments/scripts/run_observation_detail_allocation.py', 'src/tokenizers/adaptive_observation_detail.py',
           'src/tokenizers/factorized_observation.py', 'src/analysis/observation_tokenizer_evaluation.py',
           'src/data/observation_tokenizer_panel.py', 'tests/test_observation_gain_shape_screen.py')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path, obj):
    temp = path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temp, path)


def configuration(path):
    c = yaml.safe_load(path.read_text())
    if (json.loads(json.dumps(c)) != c or c != yaml.safe_load(CONFIG.read_text()) or c['experiment_id'] != 'TK-OBS-GAIN-SHAPE-v1'
            or ROOT/c['parent'] != PARENT or ROOT/c['allocation'] != ALLOCATION or c['rates'] != [256, 512, 1024]
            or c['rms_bins']['quantiles'] != [.25, .75] or c['resources']['fit_processes'] != 2
            or c['resources']['threads_per_process'] != 1):
        raise ValueError('exact EEG development gain-shape contract required')
    return c


def gate(c):
    paths = [CONFIG, *(ROOT/p for p in SOURCES)]
    return dict(status='passed', schema=c['schema'], files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def checked_sources():
    """Only direct known artifact paths from the two completed runs; no search or pickle fallbacks."""
    required = {PARENT: ['summary.json','source_manifest.json','resolved_config.json','panel.csv','reconstruction.csv','eeg_codec.joblib'],
                ALLOCATION: ['summary.json','resolved_config.json','synthetic.json','allocation.csv']}
    for folder, names in required.items():
        for name in names:
            p = folder/name
            if p.resolve(strict=True) != p or not p.is_file():
                raise ValueError('unexpected redirected/missing source artifact')
    p, a = read(PARENT/'summary.json'), read(ALLOCATION/'summary.json')
    af = read(ALLOCATION/'synthetic.json'); pf = read(PARENT/'source_manifest.json')
    if (p['status'] != 'completed' or p['experiment_id'] != 'TK-OBS-FACTOR-v1' or a['status'] != 'completed'
            or a['experiment_id'] != 'TK-OBS-ALLOCATION-v1' or ROOT/a['parent'] != PARENT or af['status'] != 'passed'):
        raise ValueError('exact completed parent and allocation required')
    old_cfg='experiments/configs/physiology_semantic_tokenizer/observation_detail_allocation_v1.yaml'
    v3_cfg='experiments/configs/physiology_semantic_tokenizer/factorized_observation_v1.yaml'
    expected_a={old_cfg,*SOURCES[2:-1]}
    expected_p={v3_cfg,'experiments/scripts/run_observation_factorization.py','src/analysis/observation_legacy_control.py',
                'src/analysis/observation_task_probe.py',*SOURCES[4:-1]}
    if (set(af['files'])!=expected_a or set(p['sources'])!=expected_p
            or read(ALLOCATION/'resolved_config.json')!=yaml.safe_load((ROOT/old_cfg).read_text())
            or read(PARENT/'resolved_config.json')!=yaml.safe_load((ROOT/v3_cfg).read_text())):
        raise ValueError('incomplete or mismatched source manifest/contract')
    for key in sorted(expected_a|expected_p):
        actual=hashlib.sha256((ROOT/key).read_bytes()).hexdigest()
        if ((key in expected_a and af['files'][key]!=actual)
                or (key in expected_p and (p['sources'][key]!=actual or pf.get(key)!=actual))):
            raise ValueError(f'reference source drift: {key}')
    return p


def scores(path):
    t = pd.read_csv(path, dtype={'dataset':str, 'subject':str}, keep_default_na=False)
    t = t[(t.modality=='eeg') & (t.split=='validation')].copy()
    t['bits_per_patch'] = pd.to_numeric(t.bits_per_patch, errors='coerce')
    # Shared historical CSVs use blank spectral columns on Hb; pandas may infer object/string.
    t[METRICS] = t[METRICS].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(t[METRICS].to_numpy()).all(): raise ValueError('nonfinite historical EEG endpoint')
    return t


def check_replay(actual, old, c):
    """Check all retained endpoint columns and all hierarchy identities, not just the macro."""
    def indexed(t):
        k = t[TABLE_KEYS].fillna('').astype(str)
        if k.duplicated().any() or t.empty:
            raise ValueError('empty or duplicate replay identity')
        return pd.concat([k, t[METRICS]], axis=1).set_index(TABLE_KEYS).sort_index()
    a, b = indexed(actual), indexed(old)
    if not a.index.equals(b.index) or not np.allclose(a, b, **c['replay_tolerance']):
        raise ValueError('parent per-subject or hierarchy score replay mismatch')
    return float(np.max(np.abs(a.to_numpy()-b.to_numpy())))


def ac_rms(x):
    a = np.asarray(x, dtype=np.float64)
    if not len(a) or a.shape[1:] != (1,400) or not np.isfinite(a).all():
        raise ValueError('expected complete two-second finite EEG X')
    return np.std(a[:,0], axis=-1, ddof=0)


def band_power_db(x):
    """Observed/reconstructed [30,45) integrated one-sided Hann power, dB re 1 (G0 X scale)^2."""
    a = np.asarray(x[:,0], dtype=np.float64); n = a.shape[-1]; w = np.hanning(n)
    spec = np.fft.rfft((a-a.mean(-1, keepdims=True))*w)
    f = np.fft.rfftfreq(n, 1/200); use = (f >= 30) & (f < 45)
    power = (2/(n*np.sum(w*w)))*np.sum(abs(spec[:,use])**2, axis=-1)
    return 10*np.log10(np.maximum(power, 1e-24))


def support(frame):
    def count(f, level, dataset='', subject=''):
        return dict(level=level, dataset=dataset, subject=subject, n_patches=len(f), n_datasets=f.dataset.nunique(),
                    n_subjects=len(f[KEYS[:2]].drop_duplicates()), n_records=len(f[KEYS[:3]].drop_duplicates()),
                    n_windows=len(f[KEYS[:5]].drop_duplicates()), n_window_sites=len(f[KEYS].drop_duplicates()))
    return pd.DataFrame([count(frame,'macro'), *(count(v,'dataset',d) for d,v in frame.groupby('dataset')),
                         *(count(v,'subject',d,s) for (d,s),v in frame.groupby(KEYS[:2]))])


def rms_diagnostic(f, z, pred, vx, bin_ids):
    # Missing library/subject strata remain explicit. Conditional macro averages represented datasets only.
    template = support(f).rename(columns=lambda k: k.replace('n_', 'all_') if k.startswith('n_') else k)
    obs, recon = band_power_db(z), band_power_db(pred); out = []
    for i, label in enumerate(('low','mid','high')):
        select = (bin_ids == i); sub = f.loc[select].reset_index(drop=True)
        den = support(sub)
        tab = template.merge(den, on=TABLE_KEYS, how='left', validate='one_to_one')
        nc = [k for k in den if k.startswith('n_')]; tab[nc] = tab[nc].fillna(0).astype(int)
        if select.any():
            score = reconstruction(sub,z[select],pred[select],vx[select],'eeg')[TABLE_KEYS+[HF,HC]]
            power = aggregate(sub, dict(observed_30_45_power_db_train_X2=obs[select], predicted_30_45_power_db_train_X2=recon[select]))
            tab = tab.merge(score, on=TABLE_KEYS, how='left').merge(power, on=TABLE_KEYS, how='left')
        else:
            for k in (HF,HC,'observed_30_45_power_db_train_X2','predicted_30_45_power_db_train_X2'):
                tab[k] = np.nan
        out.append(tab.assign(rms_bin=label))
    return pd.concat(out, ignore_index=True)


def packed_replay(model, z, mode, bits, expected_ids):
    rate = model.rate(mode)
    if rate['bits_per_item'] != bits or rate['pattern_bits'] != 4:
        raise ValueError('payload budget drift')
    gain = isinstance(model, GainShapeObservation)
    expected = {'pattern','level','gain','detail'} if gain else {'pattern','detail'}
    if gain and (rate['level_bits'],rate['gain_bits'],rate['detail_bits']) != (8,6,bits-18):
        raise ValueError('gain-shape allocation drift')
    packet = model.encode(z, mode=mode)
    if (set(packet) != expected or any(np.asarray(v).dtype.kind not in 'iu' for v in packet.values())
            or not np.array_equal(packet['pattern'],expected_ids)):
        raise ValueError('noninteger/extra fields or changed frozen IDs')
    packed = model.pack(packet,mode=mode)
    if not isinstance(packed, bytes) or len(packed) != (len(z)*bits+7)//8:
        raise ValueError('incorrect packed bit length')
    fields = model.unpack(packed, len(z), mode=mode)
    for k,v in packet.items():
        np.testing.assert_array_equal(fields[k],v)
    # Only integers reconstructed from the wire are used to produce scored outputs.
    y = model.decode(fields, mode=mode)
    if y.shape != z.shape or not np.isfinite(y).all():
        raise ValueError('invalid reconstructed EEG')
    return y


def cell(bits, router, train, all_train, val, ids_train, ids_val, bins, history, c, sample):
    f,z,vx=val; tables=[]; strat=[]; examples={}; replay={}
    with threadpool_limits(limits=1):
        gain=GainShapeObservation(router,bits).fit(train,partition='train')
        adaptive=AdaptiveObservationDetail(router,bits).fit(train,partition='train')
        for model, mode, name in [(gain,'conditional','gain_conditional'),(gain,'independent','gain_independent'),
                                  (adaptive,'conditional','adaptive_pattern'),(adaptive,'independent','independent_tail')]:
            # Verify the unselected training pool too, including observations not sampled by the balanced draw.
            for start in range(0,len(all_train),1024):
                sl=slice(start,start+1024)
                if not np.array_equal(model.encode(all_train[sl],mode=mode)['pattern'], ids_train[sl]):
                    raise ValueError('training ID assignment changed')
            y=packed_replay(model,z,mode,bits,ids_val)
            t=reconstruction(f,z,y,vx,'eeg').assign(method=name,bits_per_patch=bits,modality='eeg',split='validation')
            if model is adaptive:
                replay[name]=check_replay(t,history[(history.method==name)&(history.bits_per_patch==bits)],c)
            tables.append(t)
            b=rms_diagnostic(f,z,y,vx,bins); strat.append(b.assign(method=name,bits_per_patch=bits,modality='eeg',split='validation'))
            examples[name]=y[sample].copy()
    return dict(bits=bits, model=gain, scores=pd.concat(tables,ignore_index=True), bins=pd.concat(strat,ignore_index=True),
                examples=examples, replay=replay)


def band_audit(t):
    rows=[]
    for (method,bits),v in t.groupby(['method','bits_per_patch']):
        datasets=v[v.level=='dataset'].set_index('dataset'); subjects=v[v.level=='subject']
        if set(datasets.index)!=set(DATASETS) or not datasets.index.is_unique:
            raise ValueError('expected four complete unique dataset endpoints')
        for lo,hi in BANDS:
            band=f'band_{lo}_{hi}'; db=datasets[band+'_db']; cos=datasets[band+'_cosine']
            for d in DATASETS:
                sub=subjects.loc[subjects.dataset==d,band+'_db']
                rows.append(dict(level='dataset', dataset=d, band_hz=f'{lo}_{hi}', method=method,bits_per_patch=bits,
                                 signed_db=db[d], complex_cosine=cos[d], n_subjects=len(sub),
                                 sd_subject_signed_db=sub.std(ddof=1),max_abs_subject_db=sub.abs().max()))
            rows.append(dict(level='four_datasets',dataset='',band_hz=f'{lo}_{hi}',method=method,bits_per_patch=bits,
                             signed_db=db.mean(),complex_cosine=cos.mean(),n_subjects=len(subjects),
                             sd_four_dataset_signed_db=db.std(ddof=1),mean_abs_dataset_db=db.abs().mean(),
                             max_abs_dataset_db=db.abs().max(),max_abs_dataset=db.abs().idxmax()))
    return pd.DataFrame(rows)


def decisions(table,c):
    g=c['operating_point']; candidates=[]
    for bits in c['rates']:
        def row(name,level,d=''):
            x=table[(table.method==name)&(table.bits_per_patch==bits)&(table.level==level)&(table.dataset==d)]
            if len(x)!=1: raise ValueError('missing/duplicate decision endpoint')
            return x.iloc[0]
        a=row('gain_conditional','macro'); st=row('gain_conditional','dataset',DATASETS[0]); ref=row('adaptive_pattern','dataset',DATASETS[0]);
        checks={'nmse':a.x_nmse<=g['x_nmse_max'],'shape':a.centered_cosine>=g['centered_cosine_min'],
                'hf_gain':abs(a[HF])<=g['abs_db_30_45_max'],'hf_cosine':a[HC]>=g['complex_cosine_30_45_min']}
        checks={k:bool(v) for k,v in checks.items()}
        candidates.append(dict(bits=bits,checks=checks,macro_pass=all(checks.values()),x_nmse=float(a.x_nmse),
                               centered_cosine=float(a.centered_cosine),**{HF:float(a[HF]),HC:float(a[HC])},
                               single_trial=dict(signed_db=float(st[HF]),no_gain_signed_db=float(ref[HF]),
                                   complex_cosine=float(st[HC]),no_gain_complex_cosine=float(ref[HC]),
                                   delta_abs_signed_db=float(abs(st[HF])-abs(ref[HF])),
                                   meets_local_band_checks_and_improves_abs_bias=bool(abs(st[HF])<=g['abs_db_30_45_max']
                                     and abs(st[HF])<abs(ref[HF]) and st[HC]>=g['complex_cosine_30_45_min']))))
    return next((r for r in candidates if r['macro_pass']),candidates[-1]), candidates


def contrasts(table,c):
    rows=[]
    for b in c['rates']:
        for candidate,reference in [('gain_conditional','adaptive_pattern'),('gain_conditional','flat'),('gain_conditional','gain_independent'),
                                    ('gain_independent','independent_tail')]:
            for metric in ('x_nmse','centered_cosine',HF,HC):
                for dataset in ('',*DATASETS):
                    s=table[table.dataset==dataset] if dataset else table
                    z=paired_contrast(s,metric,(reference,b),(candidate,b),seed=c['paired']['seed'])
                    rows.append(dict(dataset=dataset,level='dataset' if dataset else 'macro',metric=metric,reference=reference,
                                     candidate=candidate,bits_per_patch=b,delta=z['candidate_minus_reference'],ci95_low=z['ci95'][0],
                                     ci95_high=z['ci95'][1],n_subjects=z['n_subjects'],n_lower=z['n_lower']))
    return pd.DataFrame(rows)


def screen(c,out):
    parent=checked_sources(); blocks,source=replay_panel(PARENT)
    train_frame,xx,_=blocks['eeg','train']; val=blocks['eeg','validation']; del blocks
    f,z,vx=val; m=source['model'];
    if (m['training_draw_seed']!=93 or m['training_draws']!=16384 or source['data']['protected_access']!='forbidden'
            or set(f.dataset)!=set(DATASETS) or len(f[KEYS[:2]].drop_duplicates())!=20):
        raise ValueError('changed development panel or training draw')
    for table,x,sp in [(train_frame,xx,'train'),(f,z,'validation')]:
        prior=next(row for row in parent['panel'] if row['split']==sp and row['modality']=='eeg')
        if len(table)!=len(x) or len(x)!=prior['available_patches'] or table[KEYS+['patch']].duplicated().any():
            raise ValueError('changed patch identity/count')
    tr_subjects=set(map(tuple,train_frame[KEYS[:2]].itertuples(index=False,name=None)))
    if set(map(tuple,f[KEYS[:2]].itertuples(index=False,name=None)))&tr_subjects: raise ValueError('subject split overlap')
    idx=np.random.default_rng(m['training_draw_seed']).choice(len(xx),m['training_draws'],replace=True,p=hierarchy_weights(train_frame))
    train=xx[idx]; edges=np.quantile(ac_rms(train),c['rms_bins']['quantiles']); bins=np.searchsorted(edges,ac_rms(z),side='right')
    router=joblib.load(PARENT/'eeg_codec.joblib')
    if not isinstance(router,FactorizedObservationTokenizer) or (router.modality,router.seed,router.routing,router.n_patterns)!=('eeg',17,'pattern',16):
        raise ValueError('wrong parent router')
    hist=scores(ALLOCATION/'allocation.csv'); original=scores(PARENT/'reconstruction.csv')
    op=parent['modalities']['eeg']['operating_point']['bits_per_patch']
    py=router.decode(router.encode(z,mode='full'),mode='full')
    replay=check_replay(reconstruction(f,z,py,vx,'eeg'),original[(original.method=='full')&(original.bits_per_patch==op)],c)
    ids_train=router.encode(xx,mode='prototype')['pattern']; ids_val=router.encode(z,mode='prototype')['pattern']
    # The very first included physical 2s patch/site per library; independent of model residuals.
    sample=f.reset_index(drop=True).drop_duplicates('dataset').index.to_numpy()
    with joblib.parallel_config(backend='loky',inner_max_num_threads=1):
        cells=joblib.Parallel(n_jobs=2,pre_dispatch=2)(joblib.delayed(cell)(bits,router,train,xx,val,ids_train,ids_val,bins,hist,c,sample) for bits in c['rates'])
    flat=hist[(hist.method=='flat')&hist.bits_per_patch.isin(c['rates'])].copy()
    # Flat is the retained equal-budget comparator. At common v3 rates check both independent exports.
    srcflat={}
    for b in c['rates']:
        cur=flat[flat.bits_per_patch==b]; match=original[(original.method=='flat')&(original.bits_per_patch==b)]
        if b!=1024: srcflat[str(b)]=check_replay(cur,match,c)
        panelids=cur[TABLE_KEYS].fillna('').astype(str).sort_values(TABLE_KEYS).reset_index(drop=True)
        wanted=support(f)[TABLE_KEYS].astype(str).sort_values(TABLE_KEYS).reset_index(drop=True)
        if not panelids.equals(wanted) or not np.isfinite(cur[METRICS]).all().all(): raise ValueError('flat comparator support changed')
    table=pd.concat([flat,*[r['scores'] for r in cells]],ignore_index=True)[TABLE_KEYS+METRICS+['method','bits_per_patch','modality','split']]
    table.to_csv(out/'gain_shape.csv',index=False)
    bt=pd.concat([r['bins'] for r in cells],ignore_index=True); bt.assign(training_q25=edges[0],training_q75=edges[1]).to_csv(out/'rms_bins.csv',index=False)
    band_audit(table).to_csv(out/'band_audit.csv',index=False); delta=contrasts(table,c);delta.to_csv(out/'paired_contrasts.csv',index=False)
    choice,candidates=decisions(table,c); bits=choice['bits']; result=next(r for r in cells if r['bits']==bits)
    model_name='eeg_gain_shape.joblib'; joblib.dump(result['model'],out/model_name,compress=3); loaded=joblib.load(out/model_name)
    examples=[]
    for i,j in enumerate(sample):
        ze=z[j:j+1]; pe=loaded.encode(ze,mode='conditional'); payload=loaded.pack(pe,mode='conditional')
        pred=loaded.decode(loaded.unpack(payload,1,mode='conditional'),mode='conditional'); np.testing.assert_array_equal(pred,result['examples']['gain_conditional'][i:i+1]);
        entry=json.loads(f.iloc[[j]][KEYS+['patch']].to_json(orient='records'))[0]; fn=f"eeg_{entry['dataset']}_example.npz"
        np.savez_compressed(out/fn,x=ze,pred=pred,adaptive_pattern=result['examples']['adaptive_pattern'][i:i+1],
                            gain_independent=result['examples']['gain_independent'][i:i+1],pattern=pe['pattern'],level=pe['level'],
                            gain=pe['gain'],patch=np.array([entry['patch']]),vx=vx[j:j+1],payload=np.frombuffer(payload,dtype=np.uint8))
        examples.append(dict(**entry,file=fn,bits_per_patch=bits))
    details=dict(selected_bits=bits, met_all_screens=choice['macro_pass'], method='gain_conditional',routing_mode='conditional',model=model_name,
                 **{k:choice[k] for k in ('x_nmse','centered_cosine',HF,HC,'single_trial')},selection_cells=candidates,examples=examples,
                 contrasts=json.loads(delta[(delta.level=='macro')&(delta.bits_per_patch==bits)].to_json(orient='records',double_precision=15)),
                 rms_training_edges=edges.tolist(),n_validation_patches=len(z),n_validation_subjects=len(f[KEYS[:2]].drop_duplicates()),
                 parent_full_replay_max_error=replay,parent_flat_table_agreement=srcflat,
                 allocation_score_replay_max_errors={str(r['bits']):r['replay'] for r in cells},
                 fixed_ID_checks=dict(n_train=len(xx),n_validation=len(z),mismatches=0,not_refitted=True))
    dump(out/'summary.json',dict(status='completed',experiment_id=c['experiment_id'],parent=c['parent'],allocation=c['allocation'],
         modalities={'eeg':details},scores='gain_shape.csv',bands='band_audit.csv',rms_bins='rms_bins.csv',contrasts='paired_contrasts.csv',
         scope='Frozen X-only offline development; conditional bootstrap does not account for selection; no Hb fit or task re-test.'))


def synthetic(c):
    rng=np.random.default_rng(4801); x=rng.normal(size=(96,1,400)); z=rng.normal(size=(8,1,400))*.23+2
    router=FactorizedObservationTokenizer('eeg',seed=17).fit(x,partition='train'); orig=router.encode(z,mode='prototype')['pattern']
    for b in c['rates']:
        a=GainShapeObservation(router,b).fit(x,partition='train')
        for mode in ('conditional','independent'): packed_replay(a,z,mode,b,orig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=CONFIG)
    p.add_argument('--stage',choices=['synthetic','screen'],required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv); c=configuration(args.config); out=args.output.absolute()
    if out!=out.resolve() or out.parent!=SUITE or not re.fullmatch(r'\d{8}_gain(?:_shape)?_v[1-9]\d*',out.name):
        raise ValueError('fresh versioned gain-shape output required')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'gain_shape.lock').open('a') as lock, threadpool_limits(limits=1):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'summary.json').exists() or (out/'controller.json').exists() and read(out/'controller.json')['status'] in ('running','failed','completed'):
            raise ValueError('previous attempt must be retained; use a fresh versioned run')
        if args.stage=='synthetic' and (out/'synthetic.json').exists():raise ValueError('previous synthetic gate is immutable')
        if args.stage=='screen' and (not (out/'synthetic.json').is_file() or read(out/'synthetic.json')!=gate(c) or read(out/'resolved_config.json')!=c):
            raise ValueError('exact passed synthetic gate required before any access')
        dump(out/'controller.json',dict(status='running',stage=args.stage,started_unix=time.time(),pid=os.getpid()))
        try:
            if args.stage=='synthetic':
                dump(out/'resolved_config.json',c);synthetic(c);dump(out/'synthetic.json',gate(c));status='synthetic_passed'
            else:
                screen(c,out);status='completed'
            dump(out/'controller.json',dict(status=status,stage=args.stage,ended_unix=time.time()))
        except Exception:
            fn=f'failure_{args.stage}_{time.time_ns()}.txt';(out/fn).write_text(traceback.format_exc())
            dump(out/'controller.json',dict(status='failed',stage=args.stage,error=fn));raise


if __name__=='__main__': main()
