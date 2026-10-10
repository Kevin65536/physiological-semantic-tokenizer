#!/usr/bin/env python3
"""Paired public EEG proxy-rate experiment using the existing nonlinear SSM.

Hb targets and training coordinates are retained verbatim. Packed observation
rows allow 4 Hz Hb and 4/10 Hz EEG on one existing 4/10 Hz dynamical solver;
unused rows are masked, never treated as observations. No physiological
parameters, PCA, or scaling are fitted to the evaluation panel.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from functools import lru_cache
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_name] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter1d

from src.inference.observation_baselines import eeg_band_power, native_feature_operators
from src.inference.shared_driver_reconstruction import (
    fit_nonlinear_shared_driver, nonlinear_driver_forward,
)
from src.inference.t3a_balloon_robust_ssm import (
    BalloonParameters, BalloonFixedParameters, BalloonFreeParameters,
)
from src.data.ssm_prepared import prepared_target, read_prepared_arrays

DEFAULT = ROOT/'experiments/configs/physiology_semantic_tokenizer/eeg_proxy_rate_v1.yaml'
SOURCE = 'experiments/runs/physiology_semantic_tokenizer/shared_driver_reconstruction/20260928_physiology_semantics_v1'
DATASETS = ('eeg_fnirs_single_trial', 'simultaneous_eeg_nirs', 'visual_cognitive_motivation')
ARMS = dict(A4=dict(eeg_rate_hz=4, model_rate_hz=4),
            B4_grid10=dict(eeg_rate_hz=4, model_rate_hz=10),
            C10=dict(eeg_rate_hz=10, model_rate_hz=10))
MODALITIES = ('EEG', 'HbO', 'HbR')


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [serial(x) for x in value]
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if (cfg['schema'] != 'eeg_proxy_rate_v1' or cfg['source_run'] != SOURCE
            or cfg['protected_data'] != 'forbidden' or tuple(cfg['datasets']) != DATASETS
            or cfg['cache_root'] != 'data/cache/physiology_semantic_clean_v5'):
        raise ValueError('unregistered source/data boundary')
    if cfg['arms'] != ARMS or cfg['modes'] != ['full', 'Hb_hidden']:
        raise ValueError('unregistered arm or mode')
    tensor = cfg['tensor']
    if (tensor['seconds'], tensor['baseline_seconds'], tensor['hb_rate_hz'], tensor['channels']) != (30, 5, 4, 6):
        raise ValueError('unexpected tensor contract')
    if tensor['eeg_bands_hz'] != [[1,4], [4,8], [8,13], [13,30], [30,45]]:
        raise ValueError('changed EEG frequency bands')
    if not 1 <= cfg['resources']['max_workers'] <= 40 or cfg['resources']['numerical_threads'] != 1:
        raise ValueError('invalid resource contract')
    return cfg


def parameters(cfg):
    fixed = {k: v for k, v in cfg['fixed'].items() if k not in ('tau', 'kappa')}
    return BalloonParameters(BalloonFixedParameters(**fixed),
                             BalloonFreeParameters(tau=cfg['fixed']['tau'], kappa=cfg['fixed']['kappa']))


def coordinate_key(ref):
    return f'{ref["dataset"]}__{ref["site"]}__outer{ref["subject_fold"]}'


def validate_panel(cfg, project_root=ROOT):
    """Validate metadata and all identities before any measured array read."""
    parent = (Path(project_root)/cfg['source_run']).resolve()
    if read_json(parent/'summary.json')['execution'] != 'completed':
        raise ValueError('parent is not terminal')
    panel = read_json(parent/'diagnostic_plan.json')['windows']
    refs = {r['id']: r for r in read_json(parent/'cohort.json')['refs']}
    inventory = {r['key']: r for r in read_json(parent/'inventory.json')['records']}
    if len(panel) != 72 or len({r['id'] for r in panel}) != 72:
        raise ValueError('expected exactly 72 unique parent windows')
    coordinates = {}
    for ds in DATASETS:
        group = [r for r in panel if r['dataset'] == ds]
        if len(group) != 24 or len({r['subject'] for r in group}) != 6:
            raise ValueError('expected 24 windows / six subjects per dataset')
    for ref in panel:
        if ref['dataset'] not in DATASETS or ref != refs.get(ref['id']):
            raise ValueError('identity not in exact public cohort')
        if ref['dataset'] == DATASETS[2] and ref['subject'] == 'S06' and 'part1' in ref['record'].lower():
            raise ValueError('excluded Visual support')
        if not Path(ref['array_path']).resolve().is_relative_to(parent/'prepared'):
            raise ValueError('prepared path outside parent')
        if not any(w['window'] == ref['window'] and w['eeg_start_s'] == ref['eeg_start_s']
                   and w['hb_start_s'] == ref['hb_start_s'] for w in inventory[ref['key']]['windows']):
            raise ValueError('native window does not match inventory')
        key = coordinate_key(ref)
        coord = read_json(parent/'coordinates'/(key+'.json'))
        if (ref['subject'] in coord['training_subjects'] or ref['id'] in coord['training_ids']
                or coord['eeg_channels'] != ref['eeg_channels']):
            raise ValueError('coordinate split or channel identity mismatch')
        if np.shape(coord['pc']) != (30,) or np.shape(coord['sd']) != (3,) or min(coord['sd']) <= 0:
            raise ValueError('invalid frozen coordinate')
        coordinates[key] = coord
    return panel, inventory, coordinates


def interpolation(source_time, target_time):
    return np.column_stack([np.interp(target_time, source_time, col) for col in np.eye(len(source_time))])


@lru_cache(maxsize=3)
def operators(arm):
    """Map a latent trajectory to independently packed EEG/Hb observation rows.

Hb always uses the exact parent 10-to-4 Hz filter/baseline operator. EEG
keeps the inherited left-bin timestamp convention. The final row times of
the 4 Hz -> native optical interpolation reproduce the parent endpoint hold.
"""
    spec = ARMS[arm]
    ne, n = 30*spec['eeg_rate_hz'], 30*spec['model_rate_hz']
    model_time = np.arange(n)/spec['model_rate_hz']
    eeg_time = np.arange(ne)/spec['eeg_rate_hz']
    op = native_feature_operators(120)
    if ne == 120:
        eeg_baseline = op['eeg']
    else:
        weights = (eeg_time < 5).astype(float)
        weights /= weights.sum()
        eeg_baseline = np.eye(ne)-np.ones((ne,1))*weights
    eeg = eeg_baseline@interpolation(model_time, eeg_time)
    hb = op['fnirs']@interpolation(model_time, np.arange(300)/10)
    mean = np.zeros((3*n,3*n))
    mean[np.ix_(np.arange(ne)*3, np.arange(n)*3)] = eeg
    for j in (1,2):
        mean[np.ix_(np.arange(120)*3+j, np.arange(n)*3+j)] = hb
    common = np.zeros((360,3*n))
    common[0::3,0::3] = op['eeg']@interpolation(model_time, np.arange(120)/4)
    common[1::3,1::3] = common[2::3,2::3] = hb
    return mean, common


def pack_observations(target4, eeg10, sd, arm, mode):
    if arm not in ARMS or mode not in ('full', 'Hb_hidden'):
        raise ValueError('unknown arm/mode')
    if np.shape(target4) != (120,3) or np.shape(eeg10) != (300,) or np.shape(sd) != (3,):
        raise ValueError('wrong observation shape')
    if not np.isfinite(target4).all() or not np.isfinite(eeg10).all() or np.any(np.asarray(sd) <= 0):
        raise ValueError('invalid observations or scale')
    spec = ARMS[arm]
    ne, n = 30*spec['eeg_rate_hz'], 30*spec['model_rate_hz']
    y = np.full((n,3), np.nan)
    y[:ne,0] = target4[:,0] if ne == 120 else eeg10
    if mode == 'full':
        y[:120,1:] = target4[:,1:]
    scale = np.broadcast_to(sd, y.shape).copy()
    scale[:,0] *= np.sqrt(spec['eeg_rate_hz']/4)
    return y, np.isfinite(y), scale


def project_power(signal, sample_rate, coordinate, target_rate, bands):
    power = eeg_band_power(signal, bands=bands, sample_rate=sample_rate, target_rate=target_rate)
    if power.shape != (30*target_rate,6,5) or not np.isfinite(power).all() or np.any(power <= 0):
        raise ValueError('invalid native power')
    features = np.log(power).reshape(30*target_rate,30)
    if target_rate == 4:
        features = native_feature_operators(120)['eeg']@features
    else:
        features -= features[:5*target_rate].mean(axis=0)
    return features@np.asarray(coordinate['pc'])*coordinate['eeg_factor'], features


def prepare_worker(payload):
    cfg, out_text, spec, refs, coordinates = payload
    out = Path(out_text)
    terminal = out/'preparation'/(spec['key']+'.json')
    if terminal.exists():
        return read_json(terminal)
    from src.data.clean_physiology_cache import CleanPhysiologyCacheIndex
    from src.data.unified_physiology import load_native_eeg_record
    started = time.monotonic()
    index = CleanPhysiologyCacheIndex(ROOT/cfg['cache_root'])
    records = [r for r in index.records if r.join_key == spec['join_key']]
    if len(records) != 1 or records[0].dataset_id != spec['dataset']:
        raise ValueError('native registered identity mismatch')
    native = load_native_eeg_record(ROOT, records[0])
    lookup = {name.upper(): i for i, name in enumerate(native.channel_names)}
    rows = []
    for ref in refs:
        coord = coordinates[coordinate_key(ref)]
        start, length = round(ref['eeg_start_s']*native.sample_rate_hz), round(30*native.sample_rate_hz)
        signal = native.values[start:start+length, [lookup[c.upper()] for c in ref['eeg_channels']]]
        if start < 0 or len(signal) != length:
            raise ValueError('native window outside available support')
        eeg4, f4 = project_power(signal, native.sample_rate_hz, coord, 4, cfg['tensor']['eeg_bands_hz'])
        eeg10, _ = project_power(signal, native.sample_rate_hz, coord, 10, cfg['tensor']['eeg_bands_hz'])
        old, _ = read_prepared_arrays(ref['array_path'])
        target = prepared_target(ref, coord)
        feature_error = float(abs(old[ref['array_index']]-f4).max())
        eeg_error = float(abs(target[:,0]-eeg4).max())
        if not np.allclose(old[ref['array_index']], f4, rtol=1e-10, atol=1e-10) or eeg_error > 1e-10:
            raise ValueError(f'4Hz native replay does not reproduce parent: {feature_error}, {eeg_error}')
        destination = out/'prepared'/(ref['id']+'.npz')
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, target4=target, eeg10=eeg10, sd=coord['sd'])
        rows.append(dict(id=ref['id'], feature_replay_max_abs=feature_error,
                         eeg_replay_max_abs=eeg_error, coordinate=coordinate_key(ref),
                         native_start_sample=start, native_length=length))
    result = dict(status='completed', key=spec['key'], rows=rows, seconds=time.monotonic()-started,
                  source=str(native.source_path), sample_rate_hz=native.sample_rate_hz,
                  native_unit=native.native_unit, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write_json(terminal, result)
    return result


def fit_case(cfg, target4, eeg10, sd, arm, mode):
    y, mask, scale = pack_observations(target4, eeg10, sd, arm, mode)
    mean, common = operators(arm)
    options = {k: v for k, v in cfg['solver'].items() if k != 'starts'}
    start = time.monotonic()
    fit = fit_nonlinear_shared_driver(y, parameters(cfg), 1/ARMS[arm]['model_rate_hz'],
                                     visible=mask, sd=scale, mean_operator=mean, **options)
    row = {k: fit.get(k) for k in ('status', 'converged', 'evaluations', 'objective',
            'scaled_gradient_inf_norm', 'rejected_steps', 'domain_rejections', 'derivative_rejections',
            'replay_max_abs_difference', 'state_ranges', 'roughness', 'flow_prior_cost')}
    row.update(arm=arm, mode=mode, seconds=time.monotonic()-start,
               fitted_driver_count=len(y), observed_points=int(mask.sum()))
    saved = {}
    if 'canonical_prediction' in fit:
        pred4 = (common@fit['canonical_prediction'].ravel()).reshape(120,3)
        row['common_nmse'] = np.mean(((pred4-target4)/sd)**2, axis=0)
        row['common_rmse'] = np.sqrt(np.mean((pred4-target4)**2, axis=0))
        row['common_correlation'] = [float(np.corrcoef(pred4[:,j],target4[:,j])[0,1])
                                     if min(pred4[:,j].std(),target4[:,j].std()) > 1e-14 else None for j in range(3)]
        ne = 30*ARMS[arm]['eeg_rate_hz']
        own_eeg = target4[:,0] if ne == 120 else eeg10
        row['own_eeg_nmse'] = float(np.mean(((fit['prediction'][:ne,0]-own_eeg)/sd[0])**2))
        row['all_common_below_half'] = bool(np.all(np.sqrt(row['common_nmse']) < .5) and fit['converged'])
        saved = {k: fit[k] for k in ('prediction', 'canonical_prediction', 'driver', 'states', 'initial_state')}
        saved['common_prediction'] = pred4
    return row, saved


def fit_worker(payload):
    cfg, out_text, ref, arm, mode, phase = payload
    out = Path(out_text)
    dest = out/phase/ref['id']/(arm+'__'+mode+'.json')
    if dest.exists():
        return read_json(dest)
    with np.load(out/'prepared'/(ref['id']+'.npz'), allow_pickle=False) as a:
        target, eeg10, sd = a['target4'], a['eeg10'], a['sd']
    try:
        row, saved = fit_case(cfg, target, eeg10, sd, arm, mode)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if saved:
            np.savez_compressed(dest.with_suffix('.npz'), **saved)
    except Exception:
        row = dict(arm=arm, mode=mode, status='failed_exception', converged=False,
                   traceback=traceback.format_exc())
    row.update(identity=ref['id'], dataset=ref['dataset'], subject=ref['subject'],
               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write_json(dest, row)
    return serial(row)


def parallel(function, payloads, workers, progress_path):
    started = time.monotonic()
    rows, iterator = [], iter(payloads)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {}
        def submit():
            try:
                payload = next(iterator)
            except StopIteration:
                return False
            pending[pool.submit(function,payload)] = payload
            return True
        for _ in range(min(len(payloads),2*workers)):
            submit()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                rows.append(future.result())
                submit()
            elapsed = time.monotonic()-started
            write_json(progress_path, dict(completed=len(rows), planned=len(payloads), workers=workers,
                       elapsed_s=elapsed, estimated_remaining_s=elapsed/len(rows)*(len(payloads)-len(rows))))
            print(f'{progress_path.stem}: {len(rows)}/{len(payloads)} elapsed={elapsed:.1f}s', flush=True)
    return rows


def synthetic(cfg, out):
    rng = np.random.default_rng(cfg['seed'])
    sd = np.array([.025,.025,.025])
    rows = []
    for case in range(cfg['synthetic']['cases']):
        driver = gaussian_filter1d(rng.normal(size=300), 12)
        if case % 2:
            driver += .2*gaussian_filter1d(rng.normal(size=300), 1.5)
        driver *= cfg['synthetic']['driver_sd']/driver.std()
        initial = np.r_[0.,np.ones(4)]
        f = nonlinear_driver_forward(driver, initial, parameters(cfg), .1, substeps=8,
                                     derivative=False, numerical_backend='numba')
        canonical = f['canonical_prediction']
        target = (operators('B4_grid10')[1]@canonical.ravel()).reshape(120,3)
        eeg10 = (operators('C10')[0]@canonical.ravel()).reshape(300,3)[:,0]
        target += rng.normal(size=target.shape)*sd*cfg['synthetic']['noise_fraction']
        eeg10 += rng.normal(size=300)*sd[0]*cfg['synthetic']['noise_fraction']
        for arm in ARMS:
            for mode in cfg['modes']:
                row, saved = fit_case(cfg, target, eeg10, sd, arm, mode)
                row['case'] = case
                if saved:
                    clock = np.arange(len(saved['driver']))/ARMS[arm]['model_rate_hz']
                    truth = np.interp(clock,np.arange(300)/10,driver)
                    row['driver_reference_centered_nrmse'] = float(np.sqrt(np.mean(
                        ((saved['driver']-saved['driver'].mean())-(truth-truth.mean()))**2))/truth.std())
                rows.append(row)
                write_json(out/'synthetic_rows.json', rows)
    completed = len(rows) == cfg['synthetic']['cases']*len(ARMS)*len(cfg['modes'])
    # Software gate, not a scientific qualification threshold on noisy recovery.
    valid = completed and all(r['status'] != 'failed_replay_validation' and 'common_nmse' in r
                              and np.isfinite(r['common_nmse']).all() for r in rows)
    write_json(out/'synthetic_summary.json', dict(completed=completed, software_pass=valid,
        fits=len(rows), converged=sum(r['converged'] for r in rows),
        interpretation=cfg['synthetic']['role']))
    if not valid:
        raise ValueError('synthetic forward / solver software check failed')


def paired_summary(frame, reference, candidate, rng, repeats):
    a = frame[frame.arm == reference].set_index('identity')
    b = frame[frame.arm == candidate].set_index('identity')
    ids = a.index.intersection(b.index)
    good = [i for i in ids if a.loc[i,'converged'] and b.loc[i,'converged']]
    result = dict(reference=reference,candidate=candidate,planned_pairs=len(ids),available_pairs=len(good),
                  reference_success=int(a.converged.sum()),candidate_success=int(b.converged.sum()))
    if not good:
        return result
    av = a.loc[good].groupby('subject')[['nmse_EEG','nmse_HbO','nmse_HbR']].mean()
    bv = b.loc[good].groupby('subject')[['nmse_EEG','nmse_HbO','nmse_HbR']].mean().reindex(av.index)
    draws = rng.integers(0,len(av),size=(repeats,len(av)))
    delta = np.sqrt(bv.to_numpy()[draws].mean(axis=1))-np.sqrt(av.to_numpy()[draws].mean(axis=1))
    result.update(subjects=len(av), reference_nrmse=np.sqrt(av.mean()).to_numpy(),
                  candidate_nrmse=np.sqrt(bv.mean()).to_numpy(),
                  delta_nrmse=(np.sqrt(bv.mean())-np.sqrt(av.mean())).to_numpy(),
                  delta_ci95=np.quantile(delta,[.025,.975],axis=0).T,
                  improved_windows=np.sum(b.loc[good][['nmse_EEG','nmse_HbO','nmse_HbR']].to_numpy()
                      < a.loc[good][['nmse_EEG','nmse_HbO','nmse_HbR']].to_numpy(),axis=0))
    return serial(result)


def summarize(cfg, out):
    plan = read_json(out/'measured_plan.json')
    rows = []
    for ref in plan['windows']:
        for arm in ARMS:
            for mode in cfg['modes']:
                path = out/'measured'/ref['id']/(arm+'__'+mode+'.json')
                if not path.exists():
                    raise ValueError(f'missing planned result: {path}')
                row = read_json(path)
                for j, name in enumerate(MODALITIES):
                    row['nmse_'+name] = row.get('common_nmse',[None]*3)[j]
                    row['rmse_'+name] = row.get('common_rmse',[None]*3)[j]
                rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'metrics.csv',index=False)
    aggregates, paired = [], []
    rng = np.random.default_rng(cfg['seed']+1)
    for (dataset, mode), group in frame.groupby(['dataset','mode']):
        for arm, part in group.groupby('arm'):
            ok = part[part.converged]
            values = ok.groupby('subject')[['nmse_EEG','nmse_HbO','nmse_HbR']].mean()
            aggregates.append(dict(dataset=dataset, mode=mode, arm=arm, planned=len(part),
                converged=int(part.converged.sum()), subjects=int(ok.subject.nunique()),
                common_nrmse=np.sqrt(values.mean()).to_numpy() if len(ok) else None,
                own_eeg_nrmse=float(np.sqrt(ok.groupby('subject').own_eeg_nmse.mean().mean())) if len(ok) else None,
                all_common_below_half=int(part.all_common_below_half.fillna(False).sum()),
                total_fit_seconds=float(part.seconds.sum()),median_fit_seconds=float(part.seconds.median())))
        for ref, cand in cfg['endpoints']['paired_contrasts']:
            paired.append(dict(dataset=dataset,mode=mode,**paired_summary(group,ref,cand,rng,2000)))
    summary = dict(experiment_id=cfg['experiment_id'],execution='completed',
                   windows=len(plan['windows']),subjects=len({(r['dataset'],r['subject']) for r in plan['windows']}),
                   fits=len(rows),converged=int(frame.converged.sum()),aggregates=aggregates,paired=paired,
                   endpoint=cfg['endpoints'],claim_limits=cfg['claim_limits'])
    write_json(out/'summary.json',summary)
    pd.DataFrame(serial(paired)).to_csv(out/'paired_comparisons.csv',index=False)
    render_figures(cfg,out,frame)
    return summary


def render_figures(cfg,out,frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(A4='#3366aa',B4_grid10='#cc8800',C10='#bb3344')
    fig,axes = plt.subplots(2,3,figsize=(12,6),constrained_layout=True)
    for i, mode in enumerate(cfg['modes']):
        for j,name in enumerate(MODALITIES):
            ax = axes[i,j]
            for k,arm in enumerate(ARMS):
                values=[]
                for ds in DATASETS:
                    part=frame[(frame['mode']==mode)&(frame.arm==arm)&(frame.dataset==ds)&frame.converged]
                    values.append(np.sqrt(part.groupby('subject')['nmse_'+name].mean().mean()))
                ax.bar(np.arange(3)+(k-1)*.24, values,width=.23,color=colors[arm],label=arm)
            ax.set_xticks(range(3),['Single','Simultaneous','Visual'])
            ax.set_title(mode+' / '+name)
            ax.set_ylabel('Common 4 Hz target NRMSE (lower is better)')
            if i==j==0:ax.legend(fontsize=8)
    fig.savefig(out/'reconstruction.png',dpi=220)
    plt.close(fig)
    plan=read_json(out/'measured_plan.json')['windows']
    selected=[]
    for ds in DATASETS:
        # Representative selection is frozen identity order, not post-fit quality.
        ref=next(r for r in plan if r['dataset']==ds)
        selected.append(ref['id'])
        with np.load(out/'prepared'/(ref['id']+'.npz')) as a:
            target,sd=a['target4'],a['sd']
        fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True,constrained_layout=True)
        for j,name in enumerate(MODALITIES):
            axes[j].plot(np.arange(120)/4,target[:,j]/sd[j],color='black',lw=1.4,label='Observed 4 Hz target')
            for arm in ARMS:
                path=out/'measured'/ref['id']/(arm+'__full.npz')
                if path.exists():
                    with np.load(path) as a:pred=a['common_prediction']
                    axes[j].plot(np.arange(120)/4,pred[:,j]/sd[j],color=colors[arm],label=arm,lw=1)
            axes[j].set_ylabel(name+' / training SD')
        axes[0].legend(ncol=4,fontsize=8)
        axes[0].set_title(ds+' / first fixed panel identity / full reconstruction')
        axes[-1].set_xlabel('Window time (s)')
        fig.savefig(out/('example_'+ds+'.png'),dpi=220)
        plt.close(fig)
    write_json(out/'figure_manifest.json',dict(format='PNG',dpi=220,examples=selected,
        example_selection='first_fixed_panel_identity_each_dataset_not_selected_by_fit'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--stage',choices=['check','synthetic','prepare','pilot','run','summarize'],required=True)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    cfg=read_config(args.config)
    out=args.run_dir.resolve()
    allowed=(ROOT/cfg['output_root']).resolve()
    if not out.is_relative_to(allowed) or out==allowed or out==(ROOT/SOURCE).resolve():
        raise ValueError('fresh versioned output must be below owning reconstruction root')
    if not 1<=args.workers<=cfg['resources']['max_workers']:
        raise ValueError('worker bound exceeded')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        resolved=out/'resolved_config.yaml'
        if resolved.exists() and yaml.safe_load(resolved.read_text())!=cfg:
            raise ValueError('cannot mutate retained resolved config')
        if not resolved.exists():shutil.copyfile(args.config,resolved)
        manifest=dict(stage=args.stage,execution='running',pid=os.getpid(),workers=args.workers,
            started_at=datetime.now(timezone.utc).isoformat(),command=sys.argv,cwd=str(ROOT))
        write_json(out/(args.stage+'_manifest.json'),manifest)
        try:
            if args.stage=='check':
                operators('A4');operators('B4_grid10');operators('C10')
                write_json(out/'check.json',dict(status='passed',measured_arrays_read=False,
                    configuration=cfg['experiment_id'],shapes={a:list(operators(a)[0].shape) for a in ARMS}))
            elif args.stage=='synthetic':
                synthetic(cfg,out)
            else:
                if not read_json(out/'synthetic_summary.json')['software_pass']:
                    raise ValueError('synthetic software check required before measured access')
                panel,inventory,coordinates=validate_panel(cfg)
                if args.stage=='prepare':
                    write_json(out/'measured_plan.json',dict(windows=panel,coordinate_keys=sorted(coordinates),
                        selection='exact_parent_panel_unchanged'))
                    for key,coord in coordinates.items():write_json(out/'coordinates'/(key+'.json'),coord)
                    payloads=[(cfg,str(out),inventory[key],[r for r in panel if r['key']==key],coordinates)
                              for key in sorted({r['key'] for r in panel})]
                    rows=parallel(prepare_worker,payloads,min(args.workers,cfg['resources']['preparation_workers']),out/'prepare_progress.json')
                    write_json(out/'preparation_summary.json',dict(execution='completed',records=len(rows),
                        windows=sum(len(r['rows']) for r in rows),
                        max_feature_replay_error=max(v['feature_replay_max_abs'] for r in rows for v in r['rows']),
                        max_eeg_replay_error=max(v['eeg_replay_max_abs'] for r in rows for v in r['rows']),
                        peak_worker_rss_bytes=max(r['peak_rss_bytes'] for r in rows)))
                elif args.stage in ('pilot','run'):
                    if read_json(out/'preparation_summary.json')['windows']!=72:
                        raise ValueError('full paired preparation required')
                    selected=[next(r for r in panel if r['dataset']==ds) for ds in DATASETS] if args.stage=='pilot' else panel
                    # Pilot identities are part of the declared panel; full stage reuses them verbatim.
                    payloads=[(cfg,str(out),ref,arm,mode,'measured') for ref in selected for arm in ARMS for mode in cfg['modes']]
                    started=time.monotonic()
                    rows=parallel(fit_worker,payloads,args.workers,out/(args.stage+'_progress.json'))
                    write_json(out/(args.stage+'_throughput.json'),dict(fits=len(rows),seconds=time.monotonic()-started,
                        workers=args.workers,worker_seconds=sum(r.get('seconds',0) for r in rows),
                        max_worker_rss_bytes=max(r['peak_rss_bytes'] for r in rows),
                        converged=sum(r['converged'] for r in rows)))
                    if args.stage=='run':summarize(cfg,out)
                elif args.stage=='summarize':summarize(cfg,out)
            manifest['execution']='completed'
        except Exception:
            manifest.update(execution='failed',traceback=traceback.format_exc())
            raise
        finally:
            manifest.update(finished_at=datetime.now(timezone.utc).isoformat(),
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            write_json(out/(args.stage+'_manifest.json'),manifest)


if __name__=='__main__':
    main()
